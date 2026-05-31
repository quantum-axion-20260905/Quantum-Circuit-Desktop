"use client";

import React from "react";
import { useCircuit } from "../../state/circuitStore";

function section(title: string, body: React.ReactNode) {
  return (
    <section style={{ border: "1px solid #d1d5db", borderRadius: 6, padding: 10, background: "#fff", marginBottom: 10 }}>
      <div style={{ fontWeight: 600, marginBottom: 8 }}>{title}</div>
      {body}
    </section>
  );
}

export function DesignInspectorPanel() {
  const { nQubits, ops } = useCircuit();
  const maxCol = ops.reduce((m, op) => Math.max(m, op.col), -1);
  const twoQ = ops.filter((op) => op.name === "cx" || op.name === "cz").length;
  const oneQ = ops.length - twoQ;
  const depth = Math.max(0, maxCol + 1);
  const density = depth > 0 ? (ops.length / (depth * Math.max(1, nQubits))).toFixed(3) : "0.000";
  const hasOutOfRange = ops.some((op) => op.target >= nQubits || (op.control != null && op.control >= nQubits));

  return (
    <div style={{ padding: 12 }}>
      {section(
        "Design Summary",
        <div style={{ fontSize: 12, display: "grid", gap: 4 }}>
          <div>qubits: {nQubits}</div>
          <div>gates: {ops.length}</div>
          <div>1q gates: {oneQ}</div>
          <div>2q gates: {twoQ}</div>
          <div>depth (columns): {depth}</div>
          <div>layout density: {density}</div>
        </div>
      )}
      {section(
        "Sanity",
        <div style={{ fontSize: 12, display: "grid", gap: 4 }}>
          <div>index range: {hasOutOfRange ? "warning" : "ok"}</div>
          <div>empty circuit: {ops.length === 0 ? "yes" : "no"}</div>
        </div>
      )}
    </div>
  );
}

export function DesignWorkbenchPanel() {
  const { nQubits, ops } = useCircuit();
  const cols = Array.from(new Set(ops.map((o) => o.col))).sort((a, b) => a - b);
  const byCol = cols.map((col) => ({ col, gates: ops.filter((op) => op.col === col) }));

  return (
    <div style={{ padding: 12 }}>
      {section(
        "Gate List",
        <div style={{ maxHeight: 180, overflow: "auto", border: "1px solid #e5e7eb" }}>
          {ops.map((op) => (
            <div key={op.id} style={{ padding: "6px 8px", borderBottom: "1px solid #f1f5f9", fontSize: 12 }}>
              {op.id} | {op.name} | col={op.col} | t={op.target}{op.control != null ? ` | c=${op.control}` : ""}{op.theta != null ? ` | theta=${op.theta}` : ""}
            </div>
          ))}
          {ops.length === 0 ? <div style={{ padding: 8, fontSize: 12, color: "#6b7280" }}>No gates.</div> : null}
        </div>
      )}
      {section(
        "Layer View",
        <div style={{ maxHeight: 180, overflow: "auto", border: "1px solid #e5e7eb" }}>
          {byCol.map((x) => (
            <div key={`col_${x.col}`} style={{ padding: "6px 8px", borderBottom: "1px solid #f1f5f9", fontSize: 12 }}>
              col {x.col}: {x.gates.map((g) => g.name).join(", ")}
            </div>
          ))}
          {byCol.length === 0 ? <div style={{ padding: 8, fontSize: 12, color: "#6b7280" }}>No layers.</div> : null}
        </div>
      )}
      {section(
        "Qubit Occupancy",
        <div style={{ fontSize: 12, display: "grid", gap: 4 }}>
          {Array.from({ length: nQubits }).map((_, q) => {
            const touched = ops.filter((op) => op.target === q || op.control === q).length;
            return <div key={`q_${q}`}>q[{q}] gates: {touched}</div>;
          })}
        </div>
      )}
    </div>
  );
}

