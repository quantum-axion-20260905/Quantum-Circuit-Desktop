"use client";

import React from "react";
import { useCircuit } from "../../state/circuitStore";
import { useUIContext } from "../../state/uiContext";

function card(title: string, body: React.ReactNode) {
  return (
    <section style={{ border: "1px solid #d1d5db", borderRadius: 6, padding: 10, background: "#fff", marginBottom: 10 }}>
      <div style={{ fontWeight: 600, marginBottom: 8 }}>{title}</div>
      {body}
    </section>
  );
}

export function MetricsPanel() {
  const { nQubits, ops } = useCircuit();
  const { latestOutput } = useUIContext();
  const depth = Math.max(0, ...ops.map((o) => o.col), -1) + 1;
  const twoQ = ops.filter((o) => o.name === "cx" || o.name === "cz").length;
  const oneQ = ops.length - twoQ;
  const density = depth > 0 ? (ops.length / (depth * Math.max(1, nQubits))).toFixed(3) : "0.000";

  const runMetrics = React.useMemo(() => {
    if (!latestOutput || typeof latestOutput !== "object") return null;
    return {
      backend: latestOutput.backend ?? "-",
      timeMs: latestOutput.time_ms ?? "-",
      pathSteps: latestOutput.path_steps ?? "-",
      optCost: latestOutput.opt_cost ?? "-",
      largestIntermediate: latestOutput.largest_intermediate ?? "-"
    };
  }, [latestOutput]);

  return (
    <div style={{ padding: 12 }}>
      {card(
        "Circuit Metrics",
        <div style={{ fontSize: 12, display: "grid", gap: 4 }}>
          <div>qubits: {nQubits}</div>
          <div>gates: {ops.length}</div>
          <div>depth: {depth}</div>
          <div>1q/2q: {oneQ}/{twoQ}</div>
          <div>density: {density}</div>
        </div>
      )}
      {runMetrics
        ? card(
            "Latest Run Metrics",
            <div style={{ fontSize: 12, display: "grid", gap: 4 }}>
              <div>backend: {String(runMetrics.backend)}</div>
              <div>time_ms: {String(runMetrics.timeMs)}</div>
              <div>path_steps: {String(runMetrics.pathSteps)}</div>
              <div>opt_cost: {String(runMetrics.optCost)}</div>
              <div>largest_intermediate: {String(runMetrics.largestIntermediate)}</div>
            </div>
          )
        : null}
    </div>
  );
}

export function WarningsPanel() {
  const { nQubits, ops } = useCircuit();
  const { latestOutput } = useUIContext();
  const warnings: string[] = [];

  if (ops.length === 0) warnings.push("Circuit is empty.");
  if (ops.some((op) => op.target >= nQubits || (op.control != null && op.control >= nQubits))) warnings.push("Some gate indices exceed qubit range.");
  if (latestOutput?.counts && typeof latestOutput.counts === "object") {
    const total = Object.values(latestOutput.counts).reduce((s: number, v: any) => s + Number(v || 0), 0);
    if (total <= 0) warnings.push("Counts sum is non-positive.");
  }
  if (latestOutput?.amplitudes && Array.isArray(latestOutput.amplitudes)) {
    let norm2 = 0;
    for (const a of latestOutput.amplitudes) {
      const re = Number(a?.re ?? 0);
      const im = Number(a?.im ?? 0);
      norm2 += re * re + im * im;
    }
    if (!Number.isFinite(norm2)) warnings.push("Amplitude norm is not finite.");
  }

  return (
    <div style={{ padding: 12 }}>
      {card(
        "Warnings",
        warnings.length ? (
          <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12 }}>
            {warnings.map((w) => <li key={w}>{w}</li>)}
          </ul>
        ) : (
          <div style={{ fontSize: 12 }}>No warnings.</div>
        )
      )}
    </div>
  );
}

export function ComparePanel() {
  const { selectedRunSummary } = useUIContext();
  return (
    <div style={{ padding: 12 }}>
      {card(
        "Compare Workspace",
        <div style={{ fontSize: 12 }}>
          {selectedRunSummary ? (
            <pre style={{ margin: 0, whiteSpace: "pre-wrap", background: "#f9fafb", border: "1px solid #d1d5db", padding: 8 }}>
              {JSON.stringify(selectedRunSummary, null, 2)}
            </pre>
          ) : (
            "Select runs from Run/Results panel to compare."
          )}
        </div>
      )}
    </div>
  );
}

export function NotesPanel() {
  const { notesText, setNotesText } = useUIContext();
  return (
    <div style={{ padding: 12 }}>
      {card(
        "Research Notes",
        <div>
          <textarea
            value={notesText}
            onChange={(e) => setNotesText(e.target.value)}
            style={{ width: "100%", minHeight: 180, fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace", fontSize: 12, border: "1px solid #d1d5db", borderRadius: 4, padding: 8 }}
            placeholder="Write experiment notes, assumptions, and observations..."
          />
        </div>
      )}
    </div>
  );
}

export function RunInspectorPanel() {
  const { latestOutput } = useUIContext();
  return (
    <div style={{ padding: 12, fontSize: 12 }}>
      <strong>Run Inspector</strong>
      <div style={{ marginTop: 8 }}>Shows current run/result snapshot.</div>
      {latestOutput ? (
        <pre style={{ marginTop: 8, whiteSpace: "pre-wrap", background: "#f9fafb", border: "1px solid #d1d5db", padding: 8 }}>
          {JSON.stringify({ backend: latestOutput.backend, status: latestOutput.status, time_ms: latestOutput.time_ms }, null, 2)}
        </pre>
      ) : (
        <div style={{ marginTop: 8 }}>No run output yet.</div>
      )}
    </div>
  );
}

export function ResultsInspectorPanel() {
  const { latestOutput } = useUIContext();
  return (
    <div style={{ padding: 12, fontSize: 12 }}>
      <strong>Results Inspector</strong>
      <div style={{ marginTop: 8 }}>Validation-oriented summary of latest output.</div>
      {latestOutput ? (
        <pre style={{ marginTop: 8, whiteSpace: "pre-wrap", background: "#f9fafb", border: "1px solid #d1d5db", padding: 8 }}>
          {JSON.stringify({ counts: !!latestOutput.counts, amplitudes: !!latestOutput.amplitudes, path_steps: latestOutput.path_steps ?? null }, null, 2)}
        </pre>
      ) : (
        <div style={{ marginTop: 8 }}>No results loaded.</div>
      )}
    </div>
  );
}

