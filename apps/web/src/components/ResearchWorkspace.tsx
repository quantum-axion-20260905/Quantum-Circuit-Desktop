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

  if (mode === "run") {
    const depth = Math.max(0, ...ops.map((op) => op.col), -1) + 1;
    const twoQ = ops.filter((op) => op.name === "cx" || op.name === "cz").length;

    return (
      <main style={{ maxWidth: "100%", width: "100%", margin: 0, padding: "16px 20px 48px", boxSizing: "border-box" }}>
        {/* Top Control Header */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 16,
            padding: "10px 16px",
            background: "#ffffff",
            borderRadius: 10,
            border: "1px solid var(--qc-border)",
            boxShadow: "var(--qc-shadow-sm)",
            marginBottom: 16,
            flexWrap: "wrap",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <div>
              <div style={{ fontSize: 11, fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: ".05em" }}>
                Quantum Execution Engine
              </div>
              <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>
                {nQubits} Qubits · {ops.length} Gates · Depth {depth}
              </div>
            </div>
            <div style={{ width: 1, height: 28, background: "#e2e8f0" }} />
            <div style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 12, color: "#475569" }}>
              <span>2-Qubit gates: <strong>{twoQ}</strong></span>
              <span>•</span>
              <span>GPU: <strong style={{ color: capabilities?.gpu?.available ? "#059669" : "#64748b" }}>{capabilities?.gpu?.available ? "Available" : "Standard"}</strong></span>
            </div>
          </div>

          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <Button variant="secondary" onClick={analyze} disabled={busy !== null}>
              {busy === "analyze" ? "Analyzing Feasibility…" : "🔍 Preflight Feasibility"}
            </Button>
            {parameterNames.length === 0 ? (
              <Button
                variant="primary"
                onClick={run}
                disabled={busy !== null || (report != null && !report.feasible)}
                style={{ minWidth: 150, padding: "8px 18px", fontSize: 13, fontWeight: 700 }}
              >
                {busy === "run" ? "Simulating…" : "▶ Run Simulation"}
              </Button>
            ) : (
              <Button
                variant="primary"
                onClick={runSweep}
                disabled={busy !== null || (report != null && !report.feasible)}
                style={{ minWidth: 160, padding: "8px 18px", fontSize: 13, fontWeight: 700 }}
              >
                {busy === "run" ? "Sweeping…" : "▶ Run Parameter Sweep"}
              </Button>
            )}
          </div>
        </div>

        {/* Two-Column Studio Layout */}
        <div style={{ display: "grid", gridTemplateColumns: "minmax(340px, 0.9fr) minmax(440px, 1.1fr)", gap: 18 }}>
          {/* LEFT: Execution & Simulation Settings */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Backend & Parameters Card */}
            <section style={card}>
              <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 4, color: "#0f172a" }}>
                Backend & Solver Configuration
              </div>
              <div style={{ fontSize: 12, color: "#64748b", marginBottom: 14 }}>
                Select execution backend and measurement sampling parameters.
              </div>

              <div style={{ display: "grid", gap: 12 }}>
                <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 13 }}>
                  <span style={{ fontWeight: 500, color: "#334155" }}>Compute Backend</span>
                  <select
                    value={backend}
                    onChange={(e) => setBackend(e.target.value as Backend)}
                    style={{ padding: "5px 10px", borderRadius: 6, border: "1px solid #cbd5e1", fontSize: 13, background: "#f8fafc" }}
                  >
                    <option value="auto">Auto (Adaptive)</option>
                    <option value="statevector">GPU Statevector (Exact)</option>
                    <option value="tensor-network">GPU Tensor Network (MPS)</option>
                    <option value="reference">Reference CPU</option>
                  </select>
                </label>

                <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 13 }}>
                  <span style={{ fontWeight: 500, color: "#334155" }}>Shots (Samples)</span>
                  <div style={{ display: "flex", gap: 6 }}>
                    {[1024, 4096, 8192].map((s) => (
                      <button
                        key={s}
                        type="button"
                        onClick={() => setShots(s)}
                        style={{
                          padding: "3px 8px",
                          borderRadius: 4,
                          border: shots === s ? "1px solid #2563eb" : "1px solid #e2e8f0",
                          background: shots === s ? "#eff6ff" : "#fff",
                          color: shots === s ? "#1d4ed8" : "#475569",
                          fontSize: 12,
                          fontWeight: shots === s ? 700 : 500,
                          cursor: "pointer",
                        }}
                      >
                        {s}
                      </button>
                    ))}
                    <input
                      type="number"
                      min={1}
                      max={200000}
                      value={shots}
                      onChange={(e) => setShots(Number(e.target.value) || 1)}
                      style={{ width: 70, padding: "3px 6px", borderRadius: 4, border: "1px solid #cbd5e1", fontSize: 12, textAlign: "center" }}
                    />
                  </div>
                </label>

                <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 13 }}>
                  <span style={{ fontWeight: 500, color: "#334155" }}>Target Bitstrings</span>
                  <input
                    value={bitstrings}
                    onChange={(e) => setBitstrings(e.target.value)}
                    placeholder="auto (or 00, 11)"
                    style={{ width: 140, padding: "4px 8px", borderRadius: 6, border: "1px solid #cbd5e1", fontSize: 12, background: "#f8fafc" }}
                  />
                </label>

                <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 13 }}>
                  <span style={{ fontWeight: 500, color: "#334155" }}>Tensor Optimizer</span>
                  <select
                    value={optimizer}
                    onChange={(e) => setOptimizer(e.target.value as Optimizer)}
                    style={{ padding: "5px 10px", borderRadius: 6, border: "1px solid #cbd5e1", fontSize: 13, background: "#f8fafc" }}
                  >
                    <option value="auto">Auto (Greedy / TN)</option>
                    <option value="cotengra">Cotengra (Hyper-optimized)</option>
                  </select>
                </label>

                {backend === "tensor-network" && (
                  <div style={{ padding: 12, borderRadius: 8, background: "#f8fafc", border: "1px solid #e2e8f0", display: "grid", gap: 10, marginTop: 4 }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: "#1e293b" }}>Tensor Network Parameters</div>
                    <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 12 }}>
                      <span>Simulation Qubits</span>
                      <input
                        type="number"
                        min={nQubits}
                        max={4096}
                        value={simulationNQubits}
                        onChange={(e) => setSimulationQubits(Math.max(nQubits, Math.min(4096, Number(e.target.value) || nQubits)))}
                        style={{ width: 70, padding: "3px 6px", borderRadius: 4, border: "1px solid #cbd5e1", textAlign: "center" }}
                      />
                    </label>
                    <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 12 }}>
                      <span>Bond Dimension χ</span>
                      <input
                        type="number"
                        min={1}
                        max={4096}
                        value={bondDim}
                        onChange={(e) => setBondDim(Math.max(1, Math.min(4096, Number(e.target.value) || 1)))}
                        style={{ width: 70, padding: "3px 6px", borderRadius: 4, border: "1px solid #cbd5e1", textAlign: "center" }}
                      />
                    </label>
                    <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 12 }}>
                      <span>Truncation Cutoff</span>
                      <input
                        type="number"
                        min={0}
                        max={1}
                        step="any"
                        value={truncationCutoff}
                        onChange={(e) => setTruncationCutoff(Math.max(0, Math.min(1, Number(e.target.value) || 0)))}
                        style={{ width: 70, padding: "3px 6px", borderRadius: 4, border: "1px solid #cbd5e1", textAlign: "center" }}
                      />
                    </label>
                    <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 12 }}>
                      <span>Output Type</span>
                      <select
                        value={tnResultType}
                        onChange={(e) => setTnResultType(e.target.value as TNResultType)}
                        style={{ padding: "3px 6px", borderRadius: 4, border: "1px solid #cbd5e1", fontSize: 12 }}
                      >
                        <option value="samples">Measurement Samples</option>
                        <option value="selected_amplitudes">Selected Amplitudes</option>
                      </select>
                    </label>
                  </div>
                )}

                {/* Advanced Noise & Sweep Details */}
                <details open={advanced} onToggle={(e) => setAdvanced((e.target as HTMLDetailsElement).open)} style={{ borderTop: "1px solid #f1f5f9", paddingTop: 8 }}>
                  <summary style={{ cursor: "pointer", fontSize: 13, fontWeight: 600, color: "#475569" }}>Advanced Noise Model & Environment</summary>
                  <div style={{ marginTop: 10 }}>
                  <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, cursor: "pointer", fontWeight: 500, color: "#334155" }}>
                    <input
                      type="checkbox"
                      checked={noiseEnabled}
                      onChange={(e) => setNoiseEnabled(e.target.checked)}
                      style={{ cursor: "pointer" }}
                    />
                    Enable Realistic Depolarizing & Readout Noise
                  </label>
                  {noiseEnabled && (
                    <div style={{ display: "grid", gap: 8, marginTop: 10, padding: 10, background: "#fffbeb", borderRadius: 6, border: "1px solid #fde68a", fontSize: 12 }}>
                      <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span>1-qubit depolarizing:</span>
                        <input
                          type="number"
                          step="0.001"
                          min={0}
                          max={1}
                          value={noise.one_qubit_depolarizing}
                          onChange={(e) => setNoise((c) => ({ ...c, one_qubit_depolarizing: Number(e.target.value) || 0 }))}
                          style={{ width: 70, padding: "2px 6px", borderRadius: 4, border: "1px solid #cbd5e1" }}
                        />
                      </label>
                      <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span>2-qubit depolarizing:</span>
                        <input
                          type="number"
                          step="0.001"
                          min={0}
                          max={1}
                          value={noise.two_qubit_depolarizing}
                          onChange={(e) => setNoise((c) => ({ ...c, two_qubit_depolarizing: Number(e.target.value) || 0 }))}
                          style={{ width: 70, padding: "2px 6px", borderRadius: 4, border: "1px solid #cbd5e1" }}
                        />
                      </label>
                      <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span>Readout flip error:</span>
                        <input
                          type="number"
                          step="0.001"
                          min={0}
                          max={1}
                          value={noise.readout_flip}
                          onChange={(e) => setNoise((c) => ({ ...c, readout_flip: Number(e.target.value) || 0 }))}
                          style={{ width: 70, padding: "2px 6px", borderRadius: 4, border: "1px solid #cbd5e1" }}
                        />
                      </label>
                    </div>
                  )}
                  </div>
                </details>

                {/* Parameter sweep values */}
                {parameterNames.length > 0 && (
                  <div style={{ padding: 10, background: "#f0fdf4", borderRadius: 6, border: "1px solid #bbf7d0", fontSize: 12 }}>
                    <strong style={{ color: "#166534" }}>Parameterized Circuit:</strong>
                    <div style={{ display: "grid", gap: 6, marginTop: 6 }}>
                      {parameterNames.map((name) => (
                        <label key={name} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                          <span>{name} (radians):</span>
                          <input
                            value={sweepValues[name] ?? "0, 1.57, 3.14"}
                            onChange={(e) => setSweepValues((c) => ({ ...c, [name]: e.target.value }))}
                            style={{ width: 140, padding: "3px 6px", borderRadius: 4, border: "1px solid #cbd5e1" }}
                          />
                        </label>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </section>

            {/* Feasibility Preflight Report */}
            {report && (
              <section style={card}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                  <strong style={{ fontSize: 14, color: report.feasible ? "#059669" : "#b45309" }}>
                    {report.feasible ? "✓ Feasibility Preflight Passed" : "⚠️ High Computational Budget"}
                  </strong>
                  <span style={{ fontSize: 11, padding: "2px 8px", borderRadius: 999, background: report.feasible ? "#dcfce7" : "#fef3c7", color: report.feasible ? "#166534" : "#92400e", fontWeight: 600 }}>
                    {report.feasible ? "Feasible" : "Check Limit"}
                  </span>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, fontSize: 12 }}>
                  <div style={{ padding: 8, background: "#f8fafc", borderRadius: 6, border: "1px solid #e2e8f0" }}>
                    <div style={{ color: "#64748b", fontSize: 10, textTransform: "uppercase" }}>Estimated RAM</div>
                    <div style={{ fontWeight: 700, marginTop: 2, color: "#0f172a" }}>{report.estimated_peak_memory_mb != null ? `${report.estimated_peak_memory_mb} MB` : "—"}</div>
                  </div>
                  <div style={{ padding: 8, background: "#f8fafc", borderRadius: 6, border: "1px solid #e2e8f0" }}>
                    <div style={{ color: "#64748b", fontSize: 10, textTransform: "uppercase" }}>Estimated Time</div>
                    <div style={{ fontWeight: 700, marginTop: 2, color: "#0f172a" }}>{report.estimated_time_ms != null ? `${report.estimated_time_ms} ms` : "—"}</div>
                  </div>
                  <div style={{ padding: 8, background: "#f8fafc", borderRadius: 6, border: "1px solid #e2e8f0" }}>
                    <div style={{ color: "#64748b", fontSize: 10, textTransform: "uppercase" }}>Path Cost</div>
                    <div style={{ fontWeight: 700, marginTop: 2, color: "#0f172a" }}>{report.path_cost ?? "—"}</div>
                  </div>
                </div>
              </section>
            )}

            {jobProgress && (
              <ComputeProgress job={jobProgress} onCancel={() => void cancelJob()} canceling={canceling} onRetry={() => retryRef.current?.()} retrying={busy === "run"} />
            )}

            {error && (
              <div role="alert" style={{ padding: 12, borderRadius: 8, color: "#991b1b", background: "#fef2f2", border: "1px solid #fecaca", fontSize: 13 }}>
                ⚠️ {error}
              </div>
            )}
          </div>

          {/* RIGHT: Live Results & Measurement Histogram */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {output ? (
              <>
                {/* Result KPI Summary Card */}
                <section style={card}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                    <div style={{ fontWeight: 700, fontSize: 15, color: "#0f172a" }}>Simulation Output & Metrics</div>
                    <span style={{ fontSize: 12, padding: "3px 8px", borderRadius: 4, background: validation ? "#dcfce7" : "#fef3c7", color: validation ? "#166534" : "#92400e", fontWeight: 600 }}>
                      {validation ? "✓ State Validated" : "Review Needed"}
                    </span>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
                    <div style={{ padding: "8px 10px", background: "#f8fafc", borderRadius: 6, border: "1px solid #e2e8f0" }}>
                      <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase" }}>Dominant State</div>
                      <div style={{ fontSize: 15, fontWeight: 700, color: "#0f766e", marginTop: 2 }}>{dominant ? `|${dominant.key}⟩` : "—"}</div>
                    </div>
                    <div style={{ padding: "8px 10px", background: "#f8fafc", borderRadius: 6, border: "1px solid #e2e8f0" }}>
                      <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase" }}>Dominant Prob</div>
                      <div style={{ fontSize: 15, fontWeight: 700, color: "#0f766e", marginTop: 2 }}>
                        {dominant && totalCounts ? `${(dominant.value / totalCounts * 100).toFixed(1)}%` : "—"}
                      </div>
                    </div>
                    <div style={{ padding: "8px 10px", background: "#f8fafc", borderRadius: 6, border: "1px solid #e2e8f0" }}>
                      <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase" }}>Total Shots</div>
                      <div style={{ fontSize: 15, fontWeight: 700, color: "#1e293b", marginTop: 2 }}>{output.shots ?? totalCounts}</div>
                    </div>
                    <div style={{ padding: "8px 10px", background: "#f8fafc", borderRadius: 6, border: "1px solid #e2e8f0" }}>
                      <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase" }}>Execution Time</div>
                      <div style={{ fontSize: 15, fontWeight: 700, color: "#1e293b", marginTop: 2 }}>
                        {output.provenance?.elapsed_ms != null ? `${Number(output.provenance.elapsed_ms).toFixed(1)} ms` : "—"}
                      </div>
                    </div>
                  </div>
                </section>

                {/* Probability Histogram Card */}
                {counts.length > 0 && (
                  <section style={card}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
                      <div style={{ fontWeight: 700, fontSize: 15 }}>Measurement Probability Distribution</div>
                      <span style={{ fontSize: 12, color: "#64748b" }}>{allCounts.length} measured states</span>
                    </div>
                    <div style={{ display: "grid", gap: 10 }}>
                      {counts.map((item) => {
                        const pct = totalCounts > 0 ? (item.value / totalCounts) * 100 : 0;
                        return (
                          <div key={item.key} style={{ display: "grid", gridTemplateColumns: "64px 1fr 70px 50px", gap: 10, alignItems: "center", fontSize: 12 }}>
                            <code style={{ fontWeight: 700, color: "#0f172a", background: "#f1f5f9", padding: "2px 6px", borderRadius: 4, textAlign: "center" }}>
                              |{item.key}⟩
                            </code>
                            <div style={{ height: 18, background: "#f1f5f9", borderRadius: 4, overflow: "hidden", display: "flex", alignItems: "center" }}>
                              <div
                                style={{
                                  width: `${pct}%`,
                                  height: "100%",
                                  background: "linear-gradient(90deg, #0f766e, #2dd4bf)",
                                  borderRadius: 4,
                                  transition: "width 300ms ease",
                                }}
                              />
                            </div>
                            <span style={{ textAlign: "right", fontWeight: 700, color: "#0f766e" }}>{pct.toFixed(2)}%</span>
                            <span style={{ textAlign: "right", color: "#64748b" }}>{item.value}x</span>
                          </div>
                        );
                      })}
                    </div>
                  </section>
                )}

                {/* Selected Amplitudes */}
                {amplitudes.length > 0 && (
                  <section style={card}>
                    <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 10 }}>Statevector Complex Amplitudes</div>
                    <div style={{ display: "grid", gap: 8 }}>
                      {amplitudes.map((item) => (
                        <div key={item.key} style={{ display: "grid", gridTemplateColumns: "64px 1fr 60px", gap: 10, alignItems: "center", fontSize: 12 }}>
                          <code style={{ background: "#eff6ff", color: "#1d4ed8", padding: "2px 6px", borderRadius: 4, textAlign: "center" }}>|{item.key}⟩</code>
                          <div style={{ height: 8, background: "#e2e8f0", borderRadius: 4, overflow: "hidden" }}>
                            <div style={{ width: `${(item.value / maxAmplitude) * 100}%`, height: "100%", background: "#2563eb" }} />
                          </div>
                          <span style={{ textAlign: "right", fontWeight: 600 }}>{item.value.toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  </section>
                )}

                {/* Sweep results */}
                {isSweep && sweepRows.length > 0 && (
                  <section style={card}>
                    <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 8 }}>Parameter Sweep Results</div>
                    <div style={{ fontSize: 12, color: "#64748b", marginBottom: 12 }}>{sweepCompleted}/{sweepPoints} points completed</div>
                    <div style={{ maxHeight: 200, overflowY: "auto", display: "grid", gap: 6, fontSize: 12 }}>
                      {sweepRows.map((row, index) => {
                        const point = row.result as Record<string, unknown> | undefined;
                        const params = row.parameters as Record<string, unknown> | undefined;
                        return (
                          <div key={index} style={{ padding: "6px 8px", background: "#f8fafc", borderRadius: 4, border: "1px solid #e2e8f0", display: "flex", justifyContent: "space-between" }}>
                            <code>{JSON.stringify(params ?? {})}</code>
                            <span style={{ color: "#059669", fontWeight: 600 }}>{point ? "✓ Done" : "Failed"}</span>
                          </div>
                        );
                      })}
                    </div>
                  </section>
                )}
              </>
            ) : (
              <div
                style={{
                  ...card,
                  minHeight: 380,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  textAlign: "center",
                  color: "#64748b",
                  background: "#f8fafc",
                  border: "2px dashed #cbd5e1",
                }}
              >
                <div style={{ fontSize: 36, marginBottom: 12 }}>⚡</div>
                <div style={{ fontSize: 17, fontWeight: 700, color: "#1e293b", marginBottom: 6 }}>
                  Ready to Run Simulation
                </div>
                <p style={{ maxWidth: 380, fontSize: 13, color: "#64748b", margin: "0 0 20px" }}>
                  Your circuit contains <strong>{nQubits} qubits</strong> and <strong>{ops.length} gates</strong>. Click the run button below to evaluate the quantum state on the active compute engine.
                </p>
                <Button
                  variant="primary"
                  onClick={run}
                  disabled={busy !== null}
                  style={{ padding: "10px 24px", fontSize: 14, fontWeight: 700 }}
                >
                  ▶ Run Simulation Now
                </Button>
              </div>
            )}
          </div>
        </div>
      </main>
    );
  }

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
