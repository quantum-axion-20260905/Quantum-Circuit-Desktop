"use client";

import React from "react";
import { getCapabilities, preflight, runJob } from "../lib/agent";
import { editorToIr } from "../ir/converters";
import { irToAgentTNPayload } from "../ir/agentMapping";
import { useCircuit } from "../state/circuitStore";
import { useUIContext } from "../state/uiContext";

type WorkspaceMode = "run" | "results";

const card: React.CSSProperties = { border: "1px solid #e2e8f0", borderRadius: 10, background: "#fff", padding: 16 };
const label: React.CSSProperties = { color: "#64748b", fontSize: 12 };

function Metric({ label: text, value, tone }: { label: string; value: React.ReactNode; tone?: string }) {
  return <div><div style={label}>{text}</div><div style={{ marginTop: 3, fontWeight: 650, color: tone ?? "#0f172a" }}>{value}</div></div>;
}

export function DesignSummary() {
  const { nQubits, ops } = useCircuit();
  const depth = Math.max(0, ...ops.map((op) => op.col), -1) + 1;
  const twoQ = ops.filter((op) => op.name === "cx" || op.name === "cz").length;
  const activeQubits = new Set(ops.flatMap((op) => op.control == null ? [op.target] : [op.target, op.control]));
  const disconnected = Array.from({ length: nQubits }, (_, q) => q).filter((q) => !activeQubits.has(q));
  const parameters = ops.filter((op) => op.theta != null).length;
  const invalid = ops.some((op) => op.target >= nQubits || (op.control != null && (op.control >= nQubits || op.control === op.target)));
  return <aside style={{ width: 280, minWidth: 280, padding: 18, borderLeft: "1px solid #e2e8f0", background: "#fff" }}>
    <div style={{ fontWeight: 650, marginBottom: 18 }}>Circuit analysis</div>
    <div style={{ display: "grid", gap: 16 }}>
      <Metric label="Qubits" value={nQubits} />
      <Metric label="Gates" value={ops.length} />
      <Metric label="Depth" value={depth} />
      <Metric label="Two-qubit gates" value={twoQ} />
      <Metric label="Parameterized gates" value={parameters} />
      <Metric label="Entangling structure" value={twoQ > 0 ? `${twoQ} interaction${twoQ === 1 ? "" : "s"}` : "None"} tone={twoQ > 0 ? "#1d4ed8" : "#475569"} />
      {disconnected.length > 0 ? <Metric label="Inactive qubits" value={disconnected.map((q) => `q${q}`).join(", ")} tone="#b45309" /> : null}
      {invalid ? <Metric label="Circuit check" value="Fix invalid gate" tone="#b91c1c" /> : <Metric label="Circuit check" value="Ready" tone="#047857" />}
    </div>
  </aside>;
}

export function ResearchWorkspace({ mode }: { mode: WorkspaceMode }) {
  const { nQubits, ops } = useCircuit();
  const { latestOutput, setLatestOutput } = useUIContext();
  const [backend, setBackend] = React.useState<"auto" | "reference" | "tensor-network">("auto");
  const [optimizer, setOptimizer] = React.useState<"auto" | "cotengra">("auto");
  const [shots, setShots] = React.useState(1024);
  const [bitstrings, setBitstrings] = React.useState("auto");
  const [advanced, setAdvanced] = React.useState(false);
  const [raw, setRaw] = React.useState(false);
  const [capabilities, setCapabilities] = React.useState<any>(null);
  const [report, setReport] = React.useState<any>(null);
  const [output, setOutput] = React.useState<any>(latestOutput ?? null);
  const [busy, setBusy] = React.useState<"analyze" | "run" | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => { getCapabilities().then(setCapabilities).catch(() => setCapabilities(null)); }, []);
  React.useEffect(() => { if (latestOutput) setOutput(latestOutput); }, [latestOutput]);
  React.useEffect(() => { setReport(null); }, [nQubits, ops]);

  const payload = React.useCallback(() => {
    const ir = editorToIr(nQubits, ops);
    return irToAgentTNPayload(ir);
  }, [nQubits, ops]);

  async function analyze() {
    setBusy("analyze"); setError(null);
    try { setReport(await preflight({ ...payload(), backend, optimize: optimizer, max_time_ms: 120000, max_mem_mb: 4096 })); }
    catch (e: any) { setError(e?.message ?? "Analysis failed."); }
    finally { setBusy(null); }
  }

  async function run() {
    setBusy("run"); setError(null);
    try {
      const selected = bitstrings.trim() === "auto" || !bitstrings.trim() ? [] : bitstrings.split(",").map((x) => x.trim()).filter(Boolean);
      const result = await runJob("simulation", payload(), { backend, optimize: optimizer, shots, bitstrings: selected, result_type: "samples" });
      setOutput(result); setLatestOutput(result);
    } catch (e: any) { setError(e?.message ?? "Simulation failed."); }
    finally { setBusy(null); }
  }

  const counts = output?.counts && typeof output.counts === "object" ? Object.entries(output.counts).map(([key, value]) => ({ key, value: Number(value) })).sort((a, b) => b.value - a.value).slice(0, 8) : [];
  const amplitudes = Array.isArray(output?.amplitudes) ? output.amplitudes.map((a: any) => ({ key: String(a.bitstring), value: Math.hypot(Number(a.re ?? 0), Number(a.im ?? 0)) })).sort((a: any, b: any) => b.value - a.value).slice(0, 8) : [];
  const totalCounts = counts.reduce((sum, item) => sum + item.value, 0);
  const maxCount = Math.max(1, ...counts.map((x) => x.value));
  const maxAmplitude = Math.max(1e-12, ...amplitudes.map((x: any) => x.value));
  const dominant = counts[0];
  const entropy = totalCounts > 0 ? -counts.reduce((sum, item) => { const p = item.value / totalCounts; return p > 0 ? sum + p * Math.log2(p) : sum; }, 0) : null;
  const validation = output ? (output.norm2 != null ? Math.abs(Number(output.norm2) - 1) < 1e-6 : totalCounts > 0 ? totalCounts === Number(output.shots ?? totalCounts) : false) : false;

  if (mode === "run") return <main style={{ maxWidth: 960, width: "100%", margin: "0 auto", padding: "42px 32px" }}>
    <div style={{ marginBottom: 30 }}><h1 style={{ fontSize: 24, margin: 0 }}>Run simulation</h1><p style={{ color: "#64748b", margin: "8px 0 0" }}>Check the circuit first, then run it locally.</p></div>
    <section style={card}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 24 }}>
        <Metric label="Circuit" value={`${nQubits} qubits · ${ops.length} gates`} />
        <Metric label="Compute" value={backend === "reference" ? "Reference CPU" : "Local backend"} />
        <Metric label="GPU" value={capabilities?.gpu?.available ? "Available" : "Not required"} tone={capabilities?.gpu?.available ? "#047857" : "#475569"} />
      </div>
      <div style={{ display: "flex", gap: 10, marginTop: 28 }}>
        <button onClick={analyze} disabled={busy !== null}>{busy === "analyze" ? "Analyzing…" : "Analyze feasibility"}</button>
        <button onClick={run} disabled={busy !== null || !report?.feasible} style={{ background: "#0f172a", color: "#fff", border: 0, borderRadius: 6, padding: "7px 12px", opacity: report?.feasible ? 1 : 0.45 }}>{busy === "run" ? "Running…" : "Run simulation"}</button>
      </div>
    </section>
    {report ? <section style={{ ...card, marginTop: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 18 }}><strong>{report.feasible ? "Ready to run" : "Run not recommended"}</strong><span style={{ color: report.feasible ? "#047857" : "#b45309", fontSize: 13 }}>{report.feasible ? "Within current budget" : "Adjust circuit or limits"}</span></div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 24 }}><Metric label="Estimated memory" value={report.estimated_peak_memory_mb != null ? `${report.estimated_peak_memory_mb} MB` : "—"} /><Metric label="Path cost" value={report.path_cost ?? "—"} /><Metric label="Estimated time" value={report.estimated_time_ms != null ? `${report.estimated_time_ms} ms` : "—"} /></div>
      {Array.isArray(report.warnings) && report.warnings.length > 0 ? <div style={{ marginTop: 18, color: "#92400e", fontSize: 13 }}>{report.warnings.join(" · ")}</div> : null}
    </section> : null}
    {error ? <div style={{ color: "#b91c1c", marginTop: 16, fontSize: 13 }}>{error}</div> : null}
    <details open={advanced} onToggle={(e) => setAdvanced((e.target as HTMLDetailsElement).open)} style={{ marginTop: 24 }}><summary style={{ cursor: "pointer", color: "#475569", fontSize: 13 }}>Advanced settings</summary><div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginTop: 14, alignItems: "center" }}><label>Backend <select value={backend} onChange={(e) => setBackend(e.target.value as any)}><option value="auto">Auto</option><option value="reference">Reference CPU</option><option value="tensor-network">Tensor network</option></select></label><label>Optimizer <select value={optimizer} onChange={(e) => setOptimizer(e.target.value as any)}><option value="auto">Auto</option><option value="cotengra">Cotengra</option></select></label><label>Shots <input type="number" min={1} max={200000} value={shots} onChange={(e) => setShots(Number(e.target.value) || 1)} /></label><label>Bitstrings <input value={bitstrings} onChange={(e) => setBitstrings(e.target.value)} placeholder="auto" /></label></div></details>
  </main>;

  return <main style={{ maxWidth: 960, width: "100%", margin: "0 auto", padding: "42px 32px" }}>
    <div style={{ marginBottom: 30 }}><h1 style={{ fontSize: 24, margin: 0 }}>Results</h1><p style={{ color: "#64748b", margin: "8px 0 0" }}>{output ? "Latest simulation result." : "Run a simulation to see its result here."}</p></div>
    {output ? <><section style={card}><div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 24 }}><Metric label="Validation" value={validation ? "Passed" : "Needs review"} tone={validation ? "#047857" : "#b45309"} /><Metric label="Backend" value={output.backend ?? "—"} /><Metric label="Dominant outcome" value={dominant ? `|${dominant.key}⟩` : "—"} /><Metric label="Probability" value={dominant && totalCounts ? `${(dominant.value / totalCounts * 100).toFixed(1)}%` : "—"} /></div></section>
    <section style={{ ...card, marginTop: 16 }}><div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 24 }}><Metric label="Samples" value={(output.shots ?? totalCounts) || "—"} /><Metric label="Distribution entropy" value={entropy == null ? "—" : `${entropy.toFixed(3)} bits`} /><Metric label="Normalization" value={output.norm2 != null ? Number(output.norm2).toFixed(6) : "Not evaluated"} /></div></section>
    {counts.length ? <section style={{ ...card, marginTop: 16 }}><strong>Probability distribution</strong><div style={{ marginTop: 16, display: "grid", gap: 10 }}>{counts.map((item) => <div key={item.key} style={{ display: "grid", gridTemplateColumns: "72px 1fr 60px", gap: 12, alignItems: "center", fontSize: 13 }}><code>{item.key}</code><div style={{ height: 8, background: "#e2e8f0", borderRadius: 8, overflow: "hidden" }}><div style={{ width: `${item.value / maxCount * 100}%`, height: "100%", background: "#0f766e" }} /></div><span>{item.value}</span></div>)}</div></section> : null}
    {amplitudes.length ? <section style={{ ...card, marginTop: 16 }}><strong>Selected amplitudes</strong><div style={{ marginTop: 16, display: "grid", gap: 10 }}>{amplitudes.map((item: any) => <div key={item.key} style={{ display: "grid", gridTemplateColumns: "72px 1fr 60px", gap: 12, alignItems: "center", fontSize: 13 }}><code>{item.key}</code><div style={{ height: 8, background: "#e2e8f0", borderRadius: 8, overflow: "hidden" }}><div style={{ width: `${item.value / maxAmplitude * 100}%`, height: "100%", background: "#2563eb" }} /></div><span>{item.value.toFixed(4)}</span></div>)}</div></section> : null}
    <details open={raw} onToggle={(e) => setRaw((e.target as HTMLDetailsElement).open)} style={{ marginTop: 24 }}><summary style={{ cursor: "pointer", color: "#475569", fontSize: 13 }}>Raw data</summary><pre style={{ marginTop: 12, padding: 12, overflow: "auto", background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: 8, fontSize: 12 }}>{JSON.stringify(output, null, 2)}</pre></details></> : null}
  </main>;
}
