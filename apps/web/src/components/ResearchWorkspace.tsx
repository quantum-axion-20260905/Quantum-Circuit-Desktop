"use client";

import React from "react";
import { cancelAsyncJob, getCapabilities, preflight, runJob, runSweep as runParameterSweep, type AgentCapabilities, type AgentResult, type AsyncJob, type JsonObject, type PreflightReport } from "../lib/agent";
import type { ReplayRequest } from "../lib/runHistory";
import { editorToIr } from "../ir/converters";
import { irToAgentTNPayload } from "../ir/agentMapping";
import { useCircuit } from "../state/circuitStore";
import { useUIContext } from "../state/uiContext";
import { Button, Metric } from "../ui";
import { ComputeProgress } from "./ComputeProgress";
import { ExperimentHistory } from "./ExperimentHistory";
import { ConvergenceDiagnostics } from "./ConvergenceDiagnostics";

type WorkspaceMode = "run" | "results";
type Backend = "auto" | "reference" | "statevector" | "tensor-network";
type Optimizer = "auto" | "cotengra";
type TNResultType = "samples" | "selected_amplitudes";
type NoiseConfig = { one_qubit_depolarizing: number; two_qubit_depolarizing: number; readout_flip: number };

const card: React.CSSProperties = { border: "1px solid var(--qc-border)", borderRadius: "var(--qc-radius-lg)", background: "var(--qc-surface)", padding: 16, boxShadow: "var(--qc-shadow-sm)" };

function errorMessage(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

export function DesignSummary() {
  const { nQubits, ops } = useCircuit();
  const depth = Math.max(0, ...ops.map((op) => op.col), -1) + 1;
  const twoQ = ops.filter((op) => op.name === "cx" || op.name === "cz").length;
  const activeQubits = new Set(ops.flatMap((op) => op.control == null ? [op.target] : [op.target, op.control]));
  const disconnected = Array.from({ length: nQubits }, (_, q) => q).filter((q) => !activeQubits.has(q));
  const parameters = ops.filter((op) => op.theta != null || op.parameter != null).length;
  const invalid = ops.some((op) => op.target >= nQubits || (op.control != null && (op.control >= nQubits || op.control === op.target)));
  return <aside style={{ width: 280, minWidth: 280, padding: 18, borderLeft: "1px solid #e2e8f0", background: "#fff" }}>
    <div style={{ fontWeight: 650, marginBottom: 18 }}>Circuit analysis</div>
    <div style={{ display: "grid", gap: 16 }}>
      <Metric label="Qubits" value={nQubits} />
      <Metric label="Gates" value={ops.length} />
      <Metric label="Depth" value={depth} />
      <Metric label="Two-qubit gates" value={twoQ} />
      <Metric label="Parameterized gates" value={parameters} />
      <Metric label="Entangling structure" value={twoQ > 0 ? `${twoQ} interaction${twoQ === 1 ? "" : "s"}` : "None"} tone={twoQ > 0 ? "info" : undefined} />
      {disconnected.length > 0 ? <Metric label="Inactive qubits" value={disconnected.map((q) => `q${q}`).join(", ")} tone="warning" /> : null}
      {invalid ? <Metric label="Circuit check" value="Fix invalid gate" tone="danger" /> : <Metric label="Circuit check" value="Ready" tone="success" />}
    </div>
  </aside>;
}

export function ResearchWorkspace({ mode }: { mode: WorkspaceMode }) {
  const { nQubits, ops } = useCircuit();
  const { latestOutput, setLatestOutput, addExperiment } = useUIContext();
  const [backend, setBackend] = React.useState<Backend>("auto");
  const [optimizer, setOptimizer] = React.useState<Optimizer>("auto");
  const [tnResultType, setTnResultType] = React.useState<TNResultType>("samples");
  const [bondDim, setBondDim] = React.useState(16);
  const [truncationCutoff, setTruncationCutoff] = React.useState(0);
  const [simulationQubits, setSimulationQubits] = React.useState(4);
  const [shots, setShots] = React.useState(1024);
  const [bitstrings, setBitstrings] = React.useState("auto");
  const [noiseEnabled, setNoiseEnabled] = React.useState(false);
  const [noise, setNoise] = React.useState<NoiseConfig>({ one_qubit_depolarizing: 0, two_qubit_depolarizing: 0, readout_flip: 0 });
  const [sweepValues, setSweepValues] = React.useState<Record<string, string>>({});
  const [advanced, setAdvanced] = React.useState(false);
  const [raw, setRaw] = React.useState(false);
  const [capabilities, setCapabilities] = React.useState<AgentCapabilities | null>(null);
  const [reportState, setReportState] = React.useState<{ key: string; value: PreflightReport } | null>(null);
  const [busy, setBusy] = React.useState<"analyze" | "run" | null>(null);
  const [jobProgress, setJobProgress] = React.useState<AsyncJob | null>(null);
  const [canceling, setCanceling] = React.useState(false);
  const cancelRequestedRef = React.useRef(false);
  const retryRef = React.useRef<(() => void) | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => { getCapabilities().then(setCapabilities).catch(() => setCapabilities(null)); }, []);
  const parameterNames = React.useMemo(
    () => [...new Set(ops.map((op) => op.parameter).filter((value): value is string => Boolean(value)))],
    [ops],
  );
  const activeNoise = noiseEnabled && Object.values(noise).some((value) => value > 0);
  const simulationNQubits = backend === "tensor-network" ? Math.max(nQubits, simulationQubits) : nQubits;

  const circuitKey = React.useMemo(
    () => JSON.stringify({ nQubits, ops, backend, optimizer, tnResultType, bondDim, truncationCutoff, simulationQubits: simulationNQubits, shots, noise: activeNoise ? noise : null }),
    [nQubits, ops, backend, optimizer, tnResultType, bondDim, truncationCutoff, simulationNQubits, shots, activeNoise, noise],
  );
  const report = reportState?.key === circuitKey ? reportState.value : null;
  const output: AgentResult | null = latestOutput;
  const updateJob = React.useCallback((job: AsyncJob) => setJobProgress(job), []);
  const cancelJob = React.useCallback(async () => {
    const job = jobProgress;
    if (!job || (job.status !== "queued" && job.status !== "running")) return;
    cancelRequestedRef.current = true;
    setCanceling(true);
    try {
      await cancelAsyncJob(job.job_id);
    } catch (e: unknown) {
      cancelRequestedRef.current = false;
      setCanceling(false);
      setError(errorMessage(e, "Jobni bekor qilishda xato yuz berdi."));
    }
  }, [jobProgress]);

  const payload = React.useCallback(() => {
    const ir = editorToIr(nQubits, ops);
    return irToAgentTNPayload(ir);
  }, [nQubits, ops]);

  async function analyze() {
    setBusy("analyze"); setError(null);
    try {
      const resultType = backend === "tensor-network" ? tnResultType : "samples";
      setReportState({ key: circuitKey, value: await preflight({
        ...payload(), n_qubits: simulationNQubits,
        backend, optimize: optimizer, tn_method: "mps", bond_dim: bondDim, truncation_cutoff: truncationCutoff,
        result_type: resultType, shots, noise: activeNoise ? noise : undefined, max_time_ms: 120000, max_mem_mb: 4096,
      }) });
    }
    catch (e: unknown) { setError(errorMessage(e, "Analysis failed.")); }
    finally { setBusy(null); }
  }

  async function run() {
    setBusy("run"); setError(null); setJobProgress(null); setCanceling(false); cancelRequestedRef.current = false;
    retryRef.current = () => { void run(); };
    let request: ReplayRequest | null = null;
    try {
      const selected = bitstrings.trim() === "auto" || !bitstrings.trim() ? [] : bitstrings.split(",").map((x) => x.trim()).filter(Boolean);
      if (parameterNames.length > 0) throw new Error("This circuit has named parameters. Use Run parameter sweep.");
      const resultType = backend === "tensor-network" ? tnResultType : "samples";
      const ir = editorToIr(nQubits, ops);
      const circuit = irToAgentTNPayload(ir);
      const config: JsonObject = {
        n_qubits: simulationNQubits,
        backend, optimize: optimizer, tn_method: "mps", bond_dim: bondDim, truncation_cutoff: truncationCutoff,
        shots, bitstrings: selected, result_type: resultType, noise: activeNoise ? noise : undefined,
      };
      request = { source: "circuit", kind: "simulation", circuit, config, circuitQasm: ir.qasm };
      const result = await runJob("simulation", circuit, config, updateJob);
      setLatestOutput(result);
      addExperiment({ label: `Circuit simulation · ${backend}`, source: request.source, kind: request.kind, status: "done", request, result, provenance: result.provenance });
    } catch (e: unknown) {
      const canceled = cancelRequestedRef.current;
      const message = canceled ? "Job user tomonidan bekor qilindi." : errorMessage(e, "Simulation failed.");
      if (request) addExperiment({ label: `Circuit simulation · ${backend}`, source: request.source, kind: request.kind, status: canceled ? "canceled" : "failed", request, error: message });
      setError(message);
    }
    finally { setBusy(null); setCanceling(false); cancelRequestedRef.current = false; }
  }

  async function runSweep() {
    setBusy("run"); setError(null); setJobProgress(null); setCanceling(false); cancelRequestedRef.current = false;
    retryRef.current = () => { void runSweep(); };
    let request: ReplayRequest | null = null;
    try {
      if (parameterNames.length === 0) throw new Error("No named rotation parameters found in this circuit.");
      const parameter_values = Object.fromEntries(parameterNames.map((name) => {
        const values = (sweepValues[name] ?? "").split(",").map((value) => Number(value.trim())).filter((value) => Number.isFinite(value));
        if (values.length === 0) throw new Error(`Enter at least one finite value for ${name}.`);
        return [name, values];
      }));
      const resultType = backend === "tensor-network" ? tnResultType : "samples";
      const ir = editorToIr(nQubits, ops);
      const replayPayload: JsonObject = {
        ...payload(), n_qubits: simulationNQubits, backend, optimize: optimizer, tn_method: "mps",
        bond_dim: bondDim, truncation_cutoff: truncationCutoff, shots, result_type: resultType,
        parameter_values, noise: activeNoise ? noise : undefined,
      };
      request = { source: "circuit", kind: "sweep", payload: replayPayload, circuitQasm: ir.qasm };
      const result = await runParameterSweep(replayPayload, updateJob);
      setLatestOutput(result);
      addExperiment({ label: `Parameter sweep · ${backend}`, source: request.source, kind: request.kind, status: "done", request, result, provenance: result.provenance });
    } catch (e: unknown) {
      const canceled = cancelRequestedRef.current;
      const message = canceled ? "Job user tomonidan bekor qilindi." : errorMessage(e, "Parameter sweep failed.");
      if (request) addExperiment({ label: `Parameter sweep · ${backend}`, source: request.source, kind: request.kind, status: canceled ? "canceled" : "failed", request, error: message });
      setError(message);
    }
    finally { setBusy(null); setCanceling(false); cancelRequestedRef.current = false; }
  }

  const allCounts = output?.counts && typeof output.counts === "object" ? Object.entries(output.counts).map(([key, value]) => ({ key, value: Number(value) })).sort((a, b) => b.value - a.value) : [];
  const counts = allCounts.slice(0, 8);
  const amplitudes = Array.isArray(output?.amplitudes) ? output.amplitudes.map((a) => ({ key: a.bitstring, value: Math.hypot(Number(a.re ?? 0), Number(a.im ?? 0)) })).sort((a, b) => b.value - a.value).slice(0, 8) : [];
  const totalCounts = allCounts.reduce((sum, item) => sum + item.value, 0);
  const maxCount = Math.max(1, ...counts.map((x) => x.value));
  const maxAmplitude = Math.max(1e-12, ...amplitudes.map((x) => x.value));
  const dominant = counts[0];
  const entropy = totalCounts > 0 ? -allCounts.reduce((sum, item) => { const p = item.value / totalCounts; return p > 0 ? sum + p * Math.log2(p) : sum; }, 0) : null;
  // complex64 MPS chains accumulate a small floating-point norm error as
  // qubit count grows; keep the UI threshold aligned with the agent check.
  const isSweep = output?.backend === "parameter-sweep";
  const sweepRows = isSweep && Array.isArray(output.results) ? output.results as Array<Record<string, unknown>> : [];
  const sweepCompleted = isSweep ? Number(output.completed ?? 0) : 0;
  const sweepPoints = isSweep ? Number(output.points ?? sweepRows.length) : 0;
  const sweepFailed = isSweep ? Number(output.failed ?? 0) : 0;
  const validation = output ? (isSweep ? Number(output.failed ?? 1) === 0 : output.norm2 != null ? Math.abs(Number(output.norm2) - 1) < 1e-5 : totalCounts > 0 ? totalCounts === Number(output.shots ?? totalCounts) : false) : false;

  if (mode === "run") return <main style={{ maxWidth: 960, width: "100%", margin: "0 auto", padding: "42px 32px" }}>
    <div style={{ marginBottom: 30 }}><h1 style={{ fontSize: 24, margin: 0 }}>Run simulation</h1><p style={{ color: "#64748b", margin: "8px 0 0" }}>Check the circuit first, then run it locally.</p></div>
    <section style={card}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 24 }}>
        <Metric label="Circuit" value={`${nQubits} qubits · ${ops.length} gates`} />
        {backend === "tensor-network" && simulationQubits > nQubits ? <Metric label="Simulation" value={`${Math.max(nQubits, simulationQubits)} qubits`} /> : null}
        <Metric label="Compute" value={backend === "reference" ? "Reference CPU" : "Local backend"} />
        <Metric label="GPU" value={capabilities?.gpu?.available ? "Available" : "Not required"} tone={capabilities?.gpu?.available ? "success" : undefined} />
      </div>
      <div style={{ display: "flex", gap: 10, marginTop: 28 }}>
        <Button variant="secondary" onClick={analyze} disabled={busy !== null}>{busy === "analyze" ? "Analyzing…" : "Analyze feasibility"}</Button>
        {parameterNames.length === 0 ? <Button variant="primary" onClick={run} disabled={busy !== null || !report?.feasible}>{busy === "run" ? "Running…" : "Run simulation"}</Button> : <Button variant="primary" onClick={runSweep} disabled={busy !== null}>{busy === "run" ? "Running…" : "Run parameter sweep"}</Button>}
      </div>
    </section>
    {report ? <section style={{ ...card, marginTop: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 18 }}><strong>{report.feasible ? "Ready to run" : "Run not recommended"}</strong><span style={{ color: report.feasible ? "#047857" : "#b45309", fontSize: 13 }}>{report.feasible ? "Within current budget" : "Adjust circuit or limits"}</span></div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 24 }}><Metric label="Estimated memory" value={report.estimated_peak_memory_mb != null ? `${report.estimated_peak_memory_mb} MB` : "—"} /><Metric label="Path cost" value={report.path_cost ?? "—"} /><Metric label="Estimated time" value={report.estimated_time_ms != null ? `${report.estimated_time_ms} ms` : "—"} />{report.tn_method === "mps" ? <Metric label="Bond dimension" value={report.bond_dim != null ? Number(report.bond_dim) : bondDim} /> : null}</div>
      {Array.isArray(report.warnings) && report.warnings.length > 0 ? <div style={{ marginTop: 18, color: "#92400e", fontSize: 13 }}>{report.warnings.join(" · ")}</div> : null}
    </section> : null}
    {jobProgress ? <ComputeProgress job={jobProgress} onCancel={() => void cancelJob()} canceling={canceling} onRetry={() => retryRef.current?.()} retrying={busy === "run"} /> : null}
    {error ? <div style={{ color: "#b91c1c", marginTop: 16, fontSize: 13 }}>{error}</div> : null}
    <details open={advanced} onToggle={(e) => setAdvanced((e.target as HTMLDetailsElement).open)} style={{ marginTop: 24 }}><summary style={{ cursor: "pointer", color: "#475569", fontSize: 13 }}>Advanced settings</summary><div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginTop: 14, alignItems: "center" }}><label>Backend <select value={backend} onChange={(e) => setBackend(e.target.value as Backend)}><option value="auto">Auto</option><option value="statevector">GPU statevector</option><option value="tensor-network">GPU tensor network</option><option value="reference">Reference CPU</option></select></label><label>Optimizer <select value={optimizer} onChange={(e) => setOptimizer(e.target.value as Optimizer)}><option value="auto">Auto</option><option value="cotengra">Cotengra</option></select></label>{backend === "tensor-network" ? <><label>Simulation qubits <input type="number" min={nQubits} max={4096} value={simulationNQubits} onChange={(e) => setSimulationQubits(Math.max(nQubits, Math.min(4096, Number(e.target.value) || nQubits)))} /></label><label>TN output <select value={tnResultType} onChange={(e) => setTnResultType(e.target.value as TNResultType)}><option value="samples">Samples</option><option value="selected_amplitudes">Selected amplitudes</option></select></label><label>Bond dimension <input type="number" min={1} max={4096} value={bondDim} onChange={(e) => setBondDim(Math.max(1, Math.min(4096, Number(e.target.value) || 1)))} /></label><label>Truncation cutoff <input type="number" min={0} max={1} step="any" value={truncationCutoff} onChange={(e) => setTruncationCutoff(Math.max(0, Math.min(1, Number(e.target.value) || 0)))} /></label></> : null}<label>Shots <input type="number" min={1} max={200000} value={shots} onChange={(e) => setShots(Number(e.target.value) || 1)} /></label><label>Bitstrings <input value={bitstrings} onChange={(e) => setBitstrings(e.target.value)} placeholder="auto" /></label><label><input type="checkbox" checked={noiseEnabled} onChange={(e) => setNoiseEnabled(e.target.checked)} /> Noise trajectories</label>{noiseEnabled ? <><label>1q depolarizing <input type="number" min={0} max={1} step="any" value={noise.one_qubit_depolarizing} onChange={(e) => setNoise((current) => ({ ...current, one_qubit_depolarizing: Math.max(0, Math.min(1, Number(e.target.value) || 0)) }))} /></label><label>2q depolarizing <input type="number" min={0} max={1} step="any" value={noise.two_qubit_depolarizing} onChange={(e) => setNoise((current) => ({ ...current, two_qubit_depolarizing: Math.max(0, Math.min(1, Number(e.target.value) || 0)) }))} /></label><label>Readout flip <input type="number" min={0} max={1} step="any" value={noise.readout_flip} onChange={(e) => setNoise((current) => ({ ...current, readout_flip: Math.max(0, Math.min(1, Number(e.target.value) || 0)) }))} /></label></> : null}{parameterNames.length > 0 ? <div style={{ flexBasis: "100%", display: "flex", gap: 14, flexWrap: "wrap", marginTop: 4, alignItems: "center", color: "#475569", fontSize: 13 }}><span>Sweep values (comma-separated radians)</span>{parameterNames.map((name) => <label key={name}>{name} <input value={sweepValues[name] ?? "0, 1.57079632679, 3.14159265359"} onChange={(e) => setSweepValues((current) => ({ ...current, [name]: e.target.value }))} placeholder="0, 1.57, 3.14" /></label>)}</div> : null}</div></details>
  </main>;

  return <main style={{ maxWidth: 960, width: "100%", margin: "0 auto", padding: "42px 32px" }}>
    <div style={{ marginBottom: 30 }}><h1 style={{ fontSize: 24, margin: 0 }}>Results</h1><p style={{ color: "#64748b", margin: "8px 0 0" }}>{output ? "Latest simulation result." : "Run a simulation to see its result here."}</p></div>
    <ExperimentHistory />
    {output ? <ConvergenceDiagnostics result={output} /> : null}
    {output ? <><section style={card}><div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 24 }}><Metric label="Validation" value={validation ? "Passed" : "Needs review"} tone={validation ? "success" : "warning"} /><Metric label="Backend" value={output.backend ?? "—"} /><Metric label="Method" value={output.method ?? "—"} /><Metric label="Simulation qubits" value={output.n_qubits ?? "—"} /><Metric label="Dominant outcome" value={dominant ? `|${dominant.key}⟩` : "—"} /><Metric label="Probability" value={dominant && totalCounts ? `${(dominant.value / totalCounts * 100).toFixed(1)}%` : "—"} /></div></section>
    <section style={{ ...card, marginTop: 16 }}><div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 24 }}><Metric label="Samples" value={(output.shots ?? totalCounts) || "—"} /><Metric label="Distribution entropy" value={entropy == null ? "—" : `${entropy.toFixed(3)} bits`} /><Metric label="Normalization" value={output.norm2 != null ? Number(output.norm2).toFixed(6) : "Not evaluated"} />{output.bond_dim_requested != null ? <Metric label="Bond dimension" value={`${output.bond_dim_used ?? "?"}/${output.bond_dim_requested}`} /> : null}{output.discarded_weight != null ? <Metric label="Discarded weight" value={Number(output.discarded_weight).toExponential(3)} tone={Number(output.discarded_weight) > 1e-6 ? "warning" : "success"} /> : null}</div></section>
    {output.approximate || (output.warnings && output.warnings.length > 0) ? <div style={{ marginTop: 16, color: "#92400e", fontSize: 13 }}>{output.warnings?.join(" · ") ?? "Approximate MPS result; inspect truncation diagnostics."}</div> : null}
    {output.provenance ? <section style={{ ...card, marginTop: 16 }}><div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: 24 }}><Metric label="Problem fingerprint" value={output.provenance.problem_sha256 ? `${output.provenance.problem_sha256.slice(0, 12)}…` : "—"} /><Metric label="Circuit fingerprint" value={output.provenance.circuit_sha256 ? `${output.provenance.circuit_sha256.slice(0, 12)}…` : "—"} /><Metric label="Resolved backend" value={output.provenance.resolved_backend ?? "—"} /><Metric label="Seed" value={output.provenance.seed ?? "Not set"} /><Metric label="Elapsed" value={output.provenance.elapsed_ms != null ? `${Number(output.provenance.elapsed_ms).toFixed(2)} ms` : "—"} /><Metric label="Hardware snapshot" value={output.provenance.device ? "Captured" : "—"} /></div></section> : null}
    {isSweep ? <section style={{ ...card, marginTop: 16 }}><strong>Parameter sweep</strong><div style={{ marginTop: 12, color: "#475569", fontSize: 13 }}>{sweepCompleted}/{sweepPoints} points completed · {sweepFailed} failed</div><div style={{ marginTop: 16, display: "grid", gap: 8 }}>{sweepRows.map((row, index) => { const point = row.result as Record<string, unknown> | undefined; const params = row.parameters as Record<string, unknown> | undefined; const pointCounts = point && typeof point.counts === "object" && point.counts !== null ? Object.entries(point.counts as Record<string, number>).sort((a, b) => Number(b[1]) - Number(a[1]))[0] : null; return <div key={index} style={{ display: "grid", gridTemplateColumns: "minmax(180px, 1fr) 1fr", gap: 12, padding: "8px 0", borderTop: "1px solid #e2e8f0", fontSize: 13 }}><code>{JSON.stringify(params ?? {})}</code><span>{point ? `${String(point.backend ?? "done")}${pointCounts ? ` · dominant ${String(pointCounts[0])} (${String(pointCounts[1])})` : ""}` : `Error: ${String(row.error ?? "unknown")}`}</span></div>; })}</div></section> : null}
    {counts.length ? <section style={{ ...card, marginTop: 16 }}><strong>Probability distribution</strong><div style={{ marginTop: 16, display: "grid", gap: 10 }}>{counts.map((item) => <div key={item.key} style={{ display: "grid", gridTemplateColumns: "72px 1fr 60px", gap: 12, alignItems: "center", fontSize: 13 }}><code>{item.key}</code><div style={{ height: 8, background: "#e2e8f0", borderRadius: 8, overflow: "hidden" }}><div style={{ width: `${item.value / maxCount * 100}%`, height: "100%", background: "#0f766e" }} /></div><span>{item.value}</span></div>)}</div></section> : null}
    {amplitudes.length ? <section style={{ ...card, marginTop: 16 }}><strong>Selected amplitudes</strong><div style={{ marginTop: 16, display: "grid", gap: 10 }}>{amplitudes.map((item) => <div key={item.key} style={{ display: "grid", gridTemplateColumns: "72px 1fr 60px", gap: 12, alignItems: "center", fontSize: 13 }}><code>{item.key}</code><div style={{ height: 8, background: "#e2e8f0", borderRadius: 8, overflow: "hidden" }}><div style={{ width: `${item.value / maxAmplitude * 100}%`, height: "100%", background: "#2563eb" }} /></div><span>{item.value.toFixed(4)}</span></div>)}</div></section> : null}
    <details open={raw} onToggle={(e) => setRaw((e.target as HTMLDetailsElement).open)} style={{ marginTop: 24 }}><summary style={{ cursor: "pointer", color: "#475569", fontSize: 13 }}>Raw data</summary><pre style={{ marginTop: 12, padding: 12, overflow: "auto", background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: 8, fontSize: 12 }}>{JSON.stringify(output, null, 2)}</pre></details></> : null}
  </main>;
}
