"use client";

import React from "react";
import { editorToIr, irToEditor } from "../ir/converters";
import { validateCircuitIrV1 } from "../ir/ir";
import { useCircuit } from "../state/circuitStore";

const STORAGE_KEY = "qc:circuit_ir_v1";

export function IrPanel() {
  const { nQubits, ops, setOps, setNQubits } = useCircuit();
  const [qasm, setQasm] = React.useState<string>("");
  const [msg, setMsg] = React.useState<string | null>(null);

  function refreshFromEditor() {
    const ir = editorToIr(nQubits, ops);
    setQasm(ir.qasm);
  }

  function exportIrJson() {
    const ir = editorToIr(nQubits, ops);
    return JSON.stringify(ir, null, 2);
  }

  function saveLocal() {
    try {
      const ir = editorToIr(nQubits, ops);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(ir));
      setMsg("Saved to localStorage.");
    } catch (e: any) {
      setMsg(e?.message ?? String(e));
    }
  }

  function loadLocal() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        setMsg("No saved IR found in localStorage.");
        return;
      }
      const ir = validateCircuitIrV1(JSON.parse(raw));
      const { nQubits: n, ops: nextOps } = irToEditor(ir);
      setNQubits(n);
      setOps(nextOps);
      setMsg("Loaded from localStorage.");
    } catch (e: any) {
      setMsg(e?.message ?? String(e));
    }
  }

  async function copyIrJson() {
    try {
      await navigator.clipboard.writeText(exportIrJson());
      setMsg("Copied IR JSON to clipboard.");
    } catch (e: any) {
      setMsg(e?.message ?? String(e));
    }
  }

  return (
    <div style={{ borderTop: "1px solid #e5e7eb", paddingTop: 12 }}>
      <h3 style={{ marginBottom: 8 }}>IR (OpenQASM 3)</h3>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
        <button onClick={refreshFromEditor}>Refresh QASM</button>
        <button onClick={saveLocal}>Save</button>
        <button onClick={loadLocal}>Load</button>
        <button onClick={copyIrJson}>Copy IR JSON</button>
      </div>
      {msg && <div style={{ fontSize: 12, color: "#374151", marginBottom: 8 }}>{msg}</div>}
      <textarea
        value={qasm}
        readOnly
        rows={10}
        style={{ width: "100%", fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" }}
      />
    </div>
  );
}

