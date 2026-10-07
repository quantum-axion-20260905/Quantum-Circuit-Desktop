"use client";

import React, { useCallback, useMemo } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  applyNodeChanges,
  Handle,
  Position,
  type ReactFlowInstance,
  type Edge,
  type Node,
  type NodeChange
} from "reactflow";
import "reactflow/dist/style.css";

import { useCircuit, type GateName, type GateOp } from "../state/circuitStore";
import { editorToIr, irToEditor, qasmToIr } from "../ir/converters";
import { validateCircuitIrV1 } from "../ir/ir";
import { invokeDesktop, isDesktop } from "../lib/desktop";
import { loadLatestCircuitVersion, saveCircuitVersion } from "../lib/projectStore";
import { CIRCUIT_PRESETS } from "../lib/circuitPresets";
import { circuitToQuantikz, circuitToCsv } from "../ir/latexExport";
import { Button } from "../ui";

const LANE_START_X = 80;
const LANE_END_PADDING = 180;
const COL_W = 140;
const X0 = 120;
const nodeTypes = { gateNode: GateNode, laneNode: LaneNode };
const edgeTypes = {};

type RenderRole = "single" | "control" | "target" | "lane";
type GateNodeData = {
  opId: string;
  role: RenderRole;
  label: string;
  gate: GateOp;
  score?: number;
};

function clampInt(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, Math.round(n)));
}

function xToCol(x: number) {
  return Math.max(0, Math.round((x - X0) / COL_W));
}

function colToX(col: number) {
  return X0 + col * COL_W;
}

function laneToQubit(y: number, nQubits: number) {
  return clampInt((y - 60) / 60, 0, Math.max(0, nQubits - 1));
}

function qubitToLaneY(q: number) {
  return 60 + q * 60;
}

function opQubits(op: GateOp): number[] {
  if (op.name === "cx" || op.name === "cz") {
    const c = op.control ?? 0;
    const t = op.target;
    return c === t ? [t] : [c, t];
  }
  return [op.target];
}

function buildOccupancy(ops: GateOp[], excludeOpId?: string): Map<string, string> {
  const occ = new Map<string, string>(); // key = `${col}:${qubit}` -> opId
  for (const op of ops) {
    if (excludeOpId && op.id === excludeOpId) continue;
    for (const q of opQubits(op)) {
      occ.set(`${op.col}:${q}`, op.id);
    }
  }
  return occ;
}

function resolveCol(occ: Map<string, string>, desired: number, qubits: number[], opId: string): number {
  let col = Math.max(0, desired);
  for (let guard = 0; guard < 5000; guard += 1) {
    let conflict = false;
    for (const q of qubits) {
      const owner = occ.get(`${col}:${q}`);
      if (owner && owner !== opId) {
        conflict = true;
        break;
      }
    }
    if (!conflict) return col;
    col += 1;
  }
  return col;
}

function GateNode({ data }: { data: GateNodeData }) {
  const isControl = data.role === "control";
  const isCnotTarget = data.role === "target" && data.gate.name === "cx";
  const rotation = data.gate.theta === undefined ? null : Number(data.gate.theta.toFixed(3));
  const score = Math.max(0, Math.min(1, Number(data.score ?? 0)));
  const glow = `0 0 0 ${1 + score * 2}px rgba(16, 185, 129, ${0.08 + score * 0.25})`;
  const boxStyle: React.CSSProperties = isControl
    ? {
        width: 16,
        height: 16,
        borderRadius: 999,
        border: "2px solid #111827",
        background: "#111827"
      }
    : isCnotTarget
      ? {
        width: 28,
        height: 28,
        borderRadius: 999,
        border: "2px solid #0f172a",
        background: "#fff",
        position: "relative"
      }
      : {
        minWidth: 42,
        height: 32,
        borderRadius: 5,
        border: "2px solid #0f172a",
        background: data.gate.name === "h" ? "#dbeafe" : "#fff",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "0 8px",
        fontSize: 12,
        fontWeight: 700,
        color: "#0f172a",
        boxShadow: "0 1px 2px rgba(15, 23, 42, 0.12)"
      };

  return (
    <div style={{ ...boxStyle, boxSizing: "border-box", boxShadow: `${boxStyle.boxShadow ?? ""}, ${glow}` }} title={`${data.label} (${data.role})`}>
      {isCnotTarget ? (
        <>
          <span style={{ position: "absolute", left: 5, right: 5, top: 11, borderTop: "2px solid #0f172a" }} />
          <span style={{ position: "absolute", top: 5, bottom: 5, left: 11, borderLeft: "2px solid #0f172a" }} />
        </>
      ) : !isControl ? (
        <span>{data.label}{rotation === null ? null : <small style={{ display: "block", fontSize: 8 }}>({rotation})</small>}</span>
      ) : null}
      <Handle type="source" position={Position.Right} style={{ opacity: 0 }} />
      <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />
    </div>
  );
}

function LaneNode({ data }: { data: GateNodeData }) {
  return (
    <div style={{ position: "relative", width: "100%", height: 20, pointerEvents: "none" }}>
      <span style={{ position: "absolute", right: "100%", top: 1, width: 66, paddingRight: 10, color: "#475569", fontFamily: "monospace", fontSize: 13, textAlign: "right" }}>
        {data.label}
      </span>
      <span style={{ position: "absolute", left: 0, right: 0, top: 9, borderTop: "2px solid #94a3b8" }} />
    </div>
  );
}

export function CircuitEditor() {
  const { ops, setOps, nQubits, setNQubits, replaceCircuit, clear, undo, redo, canUndo, canRedo, analysis } = useCircuit();
  const rfRef = React.useRef<ReactFlowInstance | null>(null);
  const importInputRef = React.useRef<HTMLInputElement | null>(null);
  const [rfReady, setRfReady] = React.useState(false);
  const [editorNotice, setEditorNotice] = React.useState<string | null>(null);
  const edges = useMemo<Edge[]>(() => {
    const out: Edge[] = [];
    for (const op of ops) {
      if (op.name !== "cx" && op.name !== "cz") continue;
      out.push({
        id: `${op.id}::wire`,
        source: `${op.id}::c`,
        target: `${op.id}::t`,
        type: "straight",
        style: { stroke: "#111827", strokeWidth: 2 }
      });
    }
    return out;
  }, [ops]);

  const nodes = useMemo<Node<GateNodeData>[]>(() => {
    const out: Node<GateNodeData>[] = [];
    const laneWidth = Math.max(
      760,
      colToX(Math.max(0, ops.reduce((m, op) => Math.max(m, op.col), 0))) + LANE_END_PADDING - LANE_START_X
    );
    for (let qubit = 0; qubit < nQubits; qubit += 1) {
      out.push({
        id: `lane::${qubit}`,
        position: { x: LANE_START_X, y: qubitToLaneY(qubit) + 6 },
        data: { opId: `lane::${qubit}`, role: "lane", label: `q[${qubit}]`, gate: { id: `lane::${qubit}`, name: "h", target: qubit, col: 0, x: 0, y: 0 } },
        type: "laneNode",
        style: { width: laneWidth, height: 20 },
        draggable: false,
        selectable: false
      });
    }
    for (const op of ops) {
      const isTwoQ = op.name === "cx" || op.name === "cz";
      if (!isTwoQ) {
        out.push({
          id: op.id,
          position: { x: colToX(op.col), y: op.y },
          data: { opId: op.id, role: "single", label: op.name.toUpperCase(), gate: op },
          type: "gateNode",
          style: { width: 64, height: 32 }
        });
        continue;
      }
      const control = op.control ?? 0;
      out.push({
        id: `${op.id}::c`,
        position: { x: colToX(op.col), y: qubitToLaneY(control) },
        data: { opId: op.id, role: "control", label: "•", gate: op },
        type: "gateNode",
        style: { width: 20, height: 20 },
        draggable: true
      });
      out.push({
        id: `${op.id}::t`,
        position: { x: colToX(op.col), y: op.y },
        data: { opId: op.id, role: "target", label: op.name.toUpperCase(), gate: op },
        type: "gateNode",
        style: { width: 64, height: 32 },
        draggable: true
      });
    }
    return out;
  }, [nQubits, ops]);

  const setNodesFromChanges = useCallback(
    (changes: NodeChange[]) => {
      const positionChanges = changes.filter((change) => change.type === "position");
      if (positionChanges.length === 0) return;

      setOps((prev) => {
        // Rebuild render nodes from current ops.
        const prevNodes: Node<GateNodeData>[] = [];
        for (const op of prev) {
          const isTwoQ = op.name === "cx" || op.name === "cz";
          if (!isTwoQ) {
            prevNodes.push({
              id: op.id,
              position: { x: op.x, y: op.y },
              data: { opId: op.id, role: "single", label: op.name.toUpperCase(), gate: op },
              type: "gateNode"
            });
            continue;
          }
          const control = op.control ?? 0;
          prevNodes.push({
            id: `${op.id}::c`,
            position: { x: op.x, y: qubitToLaneY(control) },
            data: { opId: op.id, role: "control", label: "•", gate: op },
            type: "gateNode"
          });
          prevNodes.push({
            id: `${op.id}::t`,
            position: { x: op.x, y: op.y },
            data: { opId: op.id, role: "target", label: op.name.toUpperCase(), gate: op },
            type: "gateNode"
          });
        }
        const nextNodes = applyNodeChanges(positionChanges, prevNodes);

        // Collect per-op updates from moved render nodes.
        const byOp = new Map(prev.map((o) => [o.id, { ...o }]));
        for (const n of nextNodes) {
          const opId = n.data?.opId;
          if (!opId) continue;
          const op = byOp.get(opId);
          if (!op) continue;

          // remove current op from occupancy before re-placing it
          const occWithout = buildOccupancy(Array.from(byOp.values()), opId);
          const desiredCol = xToCol(n.position.x);
          const newQ = laneToQubit(n.position.y, nQubits);
          const previousTarget = op.target;
          const previousControl = op.control;

          // Update qubit(s) first so collision resolver uses intended qubits.
          if (n.data.role === "single") {
            op.target = newQ;
            op.y = qubitToLaneY(newQ);
          } else if (n.data.role === "control") {
            op.control = newQ;
          } else if (n.data.role === "target") {
            op.target = newQ;
            op.y = qubitToLaneY(newQ);
          }

          // A two-qubit operation cannot collapse onto one lane while dragging.
          // Keep the moved node's previous lane instead of emitting an invalid IR.
          if ((op.name === "cx" || op.name === "cz") && op.control === op.target) {
            if (n.data.role === "control") op.control = previousControl;
            if (n.data.role === "target") {
              op.target = previousTarget;
              op.y = qubitToLaneY(previousTarget);
            }
          }

          const qubits = opQubits(op);
          const placedCol = resolveCol(occWithout, desiredCol, qubits, op.id);
          op.col = placedCol;
          op.x = colToX(placedCol);
        }

        const nextOps = Array.from(byOp.values());
        const changed = nextOps.some((op, index) => {
          const previous = prev[index];
          return op.x !== previous.x || op.y !== previous.y || op.target !== previous.target || op.control !== previous.control;
        });
        return changed ? nextOps : prev;
      });
    },
    [nQubits, setOps]
  );

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => setNodesFromChanges(changes),
    [setNodesFromChanges]
  );

  React.useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const key = e.key.toLowerCase();
      const isMod = e.ctrlKey || e.metaKey;
      if (!isMod) return;
      if (key === "z" && !e.shiftKey) {
        e.preventDefault();
        undo();
      } else if (key === "y" || (key === "z" && e.shiftKey)) {
        e.preventDefault();
        redo();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [undo, redo]);

  const [newGate, setNewGate] = React.useState<{
    name: GateName;
    target: number;
    control: number;
    theta: number;
  }>({ name: "h", target: 0, control: 0, theta: 1.234 });

  const addGate = useCallback(() => {
    const id = `g${typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : Date.now().toString(36)}`;
    const maxCol = ops.reduce((m, o) => Math.max(m, o.col), -1);
    const base: GateOp = {
      id,
      name: newGate.name,
      target: clampInt(newGate.target, 0, nQubits - 1),
      col: maxCol + 1,
      x: colToX(maxCol + 1),
      y: 60 + clampInt(newGate.target, 0, nQubits - 1) * 60
    };
    if (newGate.name === "rx" || newGate.name === "ry" || newGate.name === "rz") base.theta = newGate.theta;
    if (newGate.name === "cx" || newGate.name === "cz") {
      base.control = clampInt(newGate.control, 0, nQubits - 1);
      base.target = clampInt(newGate.target, 0, nQubits - 1);
      if (base.control === base.target) {
        setEditorNotice("Control va target bir xil bo‘lishi mumkin emas.");
        return;
      }
      base.y = 60 + base.target * 60;
    }
    // Place into first non-colliding column (same column reserves both qubits for 2q gates)
    const occ = buildOccupancy(ops);
    const placedCol = resolveCol(occ, base.col, opQubits(base), base.id);
    base.col = placedCol;
    base.x = colToX(placedCol);

    setEditorNotice(null);
    setOps((p) => [...p, base]);
    // Bring the new gate into view so it doesn't look like "nothing happened".
    setTimeout(() => {
      try {
        rfRef.current?.setCenter(base.x, base.y, { zoom: 1.2, duration: 250 });
      } catch {
        // ignore view-focus errors; adding the node is the important part
      }
    }, 0);
  }, [nQubits, newGate, ops, setOps]);

  const download = useCallback((name: string, content: string, type: string) => {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = name;
    anchor.click();
    URL.revokeObjectURL(url);
  }, []);

  const exportIr = useCallback(() => {
    try {
      const ir = editorToIr(nQubits, ops);
      download("quantum-circuit.qcir.json", JSON.stringify(ir, null, 2), "application/json");
      setEditorNotice("IR v1 export qilindi.");
    } catch (error) {
      setEditorNotice(error instanceof Error ? error.message : "IR export qilishda xato yuz berdi.");
    }
  }, [download, nQubits, ops]);

  const exportQasm = useCallback(() => {
    try {
      download("quantum-circuit.qasm", editorToIr(nQubits, ops).qasm, "text/plain");
      setEditorNotice("OpenQASM 3 export qilindi.");
    } catch (error) {
      setEditorNotice(error instanceof Error ? error.message : "QASM export qilishda xato yuz berdi.");
    }
  }, [download, nQubits, ops]);

  const exportLatex = useCallback(() => {
    try {
      const tex = circuitToQuantikz(nQubits, ops);
      download("circuit-quantikz.tex", tex, "text/plain");
      setEditorNotice("LaTeX Quantikz kodi export qilindi.");
    } catch (error) {
      setEditorNotice(error instanceof Error ? error.message : "LaTeX export xatosi.");
    }
  }, [download, nQubits, ops]);

  const exportCsv = useCallback(() => {
    try {
      const csv = circuitToCsv(nQubits, ops);
      download("circuit-gates.csv", csv, "text/csv");
      setEditorNotice("Circuit CSV ro'yxati export qilindi.");
    } catch (error) {
      setEditorNotice(error instanceof Error ? error.message : "CSV export xatosi.");
    }
  }, [download, nQubits, ops]);

  const loadPreset = useCallback((presetId: string) => {
    const preset = CIRCUIT_PRESETS.find((p) => p.id === presetId);
    if (!preset) return;
    replaceCircuit(preset.nQubits, preset.ops);
    setEditorNotice(`'${preset.name}' shabloni yuklandi: ${preset.description}`);
  }, [replaceCircuit]);

  const saveLocal = useCallback(async () => {
    try {
      const ir = editorToIr(nQubits, ops);
      if (isDesktop()) {
        await invokeDesktop("save_circuit", { circuit: ir });
        setEditorNotice("Circuit desktop SQLite storage’ga saqlandi.");
      } else {
        try {
          const saved = await saveCircuitVersion(ir);
          setEditorNotice(`Circuit backend storage’ga saqlandi (version #${saved.id}).`);
        } catch {
          download("quantum-circuit.qcir.json", JSON.stringify(ir, null, 2), "application/json");
          setEditorNotice("Backend ishlamadi; IR fayl sifatida saqlandi.");
        }
      }
    } catch (error) {
      setEditorNotice(error instanceof Error ? `Save xatosi: ${error.message}` : "Save qilishda xato yuz berdi.");
    }
  }, [download, nQubits, ops]);

  const loadLatest = useCallback(async () => {
    try {
      const ir = isDesktop()
        ? validateCircuitIrV1((await invokeDesktop<{ metadata?: unknown }>("load_circuit", { versionId: null })).metadata)
        : await loadLatestCircuitVersion();
      const next = irToEditor(ir);
      replaceCircuit(next.nQubits, next.ops);
      setEditorNotice(isDesktop() ? "Oxirgi desktop circuit yuklandi." : "Oxirgi backend circuit yuklandi.");
    } catch (error) {
      setEditorNotice(error instanceof Error ? `Load xatosi: ${error.message}` : "Load qilishda xato yuz berdi.");
    }
  }, [replaceCircuit]);

  const importCircuit = useCallback(async (file: File) => {
    try {
      const text = await file.text();
      let ir;
      try {
        ir = validateCircuitIrV1(JSON.parse(text));
      } catch {
        ir = qasmToIr(text);
      }
      const next = irToEditor(ir);
      replaceCircuit(next.nQubits, next.ops);
      setEditorNotice(`${file.name} yuklandi.`);
    } catch (error) {
      setEditorNotice(error instanceof Error ? `Import xatosi: ${error.message}` : "Import qilishda xato yuz berdi.");
    } finally {
      if (importInputRef.current) importInputRef.current.value = "";
    }
  }, [replaceCircuit]);

  const [openMenu, setOpenMenu] = React.useState<"file" | "edit" | "presets" | "export" | "view" | null>(null);
  const [showMiniMap, setShowMiniMap] = React.useState(true);
  const menuBarRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuBarRef.current && !menuBarRef.current.contains(event.target as unknown as HTMLElement)) {
        setOpenMenu(null);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const gateList: { id: GateName; label: string; desc: string }[] = [
    { id: "h", label: "H", desc: "Hadamard (Superposition)" },
    { id: "x", label: "X", desc: "Pauli-X (NOT)" },
    { id: "rx", label: "Rx", desc: "X-rotation gate" },
    { id: "ry", label: "Ry", desc: "Y-rotation gate" },
    { id: "rz", label: "Rz", desc: "Z-rotation gate" },
    { id: "cx", label: "CX", desc: "CNOT (Controlled-X)" },
    { id: "cz", label: "CZ", desc: "Controlled-Z" },
  ];

  return (
    <div style={{ height: "100%", width: "100%", display: "flex", flexDirection: "column" }}>
      {/* 1. Professional Application Menubar */}
      <div
        ref={menuBarRef}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 4,
          padding: "5px 12px",
          background: "#f8fafc",
          borderBottom: "1px solid #e2e8f0",
          fontSize: 13,
          position: "relative",
          zIndex: 50,
          userSelect: "none"
        }}
      >
        {/* FILE MENU */}
        <div style={{ position: "relative" }}>
          <button
            onClick={() => setOpenMenu(openMenu === "file" ? null : "file")}
            style={{
              padding: "4px 9px",
              borderRadius: 4,
              border: "none",
              background: openMenu === "file" ? "#e2e8f0" : "transparent",
              fontWeight: 500,
              color: "#1e293b",
              display: "flex",
              alignItems: "center",
              gap: 4,
              cursor: "pointer"
            }}
          >
            File ▾
          </button>
          {openMenu === "file" && (
            <div
              style={{
                position: "absolute",
                top: "100%",
                left: 0,
                marginTop: 4,
                width: 220,
                background: "#ffffff",
                border: "1px solid #cbd5e1",
                borderRadius: 8,
                boxShadow: "0 10px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.08)",
                padding: "6px 0",
                zIndex: 100
              }}
            >
              <button
                onClick={() => { clear(); setOpenMenu(null); }}
                style={{ width: "100%", textAlign: "left", padding: "8px 14px", border: "none", background: "transparent", fontSize: 13, color: "#0f172a", display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#f1f5f9")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <span>📄 New Circuit</span>
                <span style={{ fontSize: 11, color: "#94a3b8" }}>Clear</span>
              </button>
              <div style={{ height: 1, background: "#e2e8f0", margin: "4px 0" }} />
              <button
                onClick={() => { void saveLocal(); setOpenMenu(null); }}
                style={{ width: "100%", textAlign: "left", padding: "8px 14px", border: "none", background: "transparent", fontSize: 13, color: "#0f172a", display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#f1f5f9")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <span>💾 Save Circuit</span>
                <span style={{ fontSize: 11, color: "#94a3b8" }}>Storage</span>
              </button>
              <button
                onClick={() => { void loadLatest(); setOpenMenu(null); }}
                style={{ width: "100%", textAlign: "left", padding: "8px 14px", border: "none", background: "transparent", fontSize: 13, color: "#0f172a", display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#f1f5f9")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <span>📂 Load Latest</span>
                <span style={{ fontSize: 11, color: "#94a3b8" }}>Restore</span>
              </button>
              <div style={{ height: 1, background: "#e2e8f0", margin: "4px 0" }} />
              <button
                onClick={() => { importInputRef.current?.click(); setOpenMenu(null); }}
                style={{ width: "100%", textAlign: "left", padding: "8px 14px", border: "none", background: "transparent", fontSize: 13, color: "#0f172a", display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#f1f5f9")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <span>📥 Import Circuit...</span>
                <span style={{ fontSize: 11, color: "#94a3b8" }}>QASM/JSON</span>
              </button>
            </div>
          )}
        </div>

        {/* EDIT MENU */}
        <div style={{ position: "relative" }}>
          <button
            onClick={() => setOpenMenu(openMenu === "edit" ? null : "edit")}
            style={{
              padding: "4px 9px",
              borderRadius: 4,
              border: "none",
              background: openMenu === "edit" ? "#e2e8f0" : "transparent",
              fontWeight: 500,
              color: "#1e293b",
              display: "flex",
              alignItems: "center",
              gap: 4,
              cursor: "pointer"
            }}
          >
            Edit ▾
          </button>
          {openMenu === "edit" && (
            <div
              style={{
                position: "absolute",
                top: "100%",
                left: 0,
                marginTop: 4,
                width: 200,
                background: "#ffffff",
                border: "1px solid #cbd5e1",
                borderRadius: 8,
                boxShadow: "0 10px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.08)",
                padding: "6px 0",
                zIndex: 100
              }}
            >
              <button
                disabled={!canUndo}
                onClick={() => { undo(); setOpenMenu(null); }}
                style={{ width: "100%", textAlign: "left", padding: "8px 14px", border: "none", background: "transparent", fontSize: 13, color: canUndo ? "#0f172a" : "#cbd5e1", display: "flex", justifyContent: "space-between", alignItems: "center", cursor: canUndo ? "pointer" : "default" }}
                onMouseEnter={(e) => { if (canUndo) e.currentTarget.style.background = "#f1f5f9"; }}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <span>↶ Undo</span>
                <span style={{ fontSize: 11, color: "#94a3b8" }}>Ctrl+Z</span>
              </button>
              <button
                disabled={!canRedo}
                onClick={() => { redo(); setOpenMenu(null); }}
                style={{ width: "100%", textAlign: "left", padding: "8px 14px", border: "none", background: "transparent", fontSize: 13, color: canRedo ? "#0f172a" : "#cbd5e1", display: "flex", justifyContent: "space-between", alignItems: "center", cursor: canRedo ? "pointer" : "default" }}
                onMouseEnter={(e) => { if (canRedo) e.currentTarget.style.background = "#f1f5f9"; }}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <span>↷ Redo</span>
                <span style={{ fontSize: 11, color: "#94a3b8" }}>Ctrl+Y</span>
              </button>
              <div style={{ height: 1, background: "#e2e8f0", margin: "4px 0" }} />
              <button
                onClick={() => { clear(); setOpenMenu(null); }}
                style={{ width: "100%", textAlign: "left", padding: "8px 14px", border: "none", background: "transparent", fontSize: 13, color: "#ef4444", display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#fef2f2")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <span>🗑 Clear All Gates</span>
                <span style={{ fontSize: 11, color: "#f87171" }}>Reset</span>
              </button>
            </div>
          )}
        </div>

        {/* PRESETS MENU */}
        <div style={{ position: "relative" }}>
          <button
            onClick={() => setOpenMenu(openMenu === "presets" ? null : "presets")}
            style={{
              padding: "4px 9px",
              borderRadius: 4,
              border: "none",
              background: openMenu === "presets" ? "#e2e8f0" : "transparent",
              fontWeight: 500,
              color: "#1e293b",
              display: "flex",
              alignItems: "center",
              gap: 4,
              cursor: "pointer"
            }}
          >
            ⚡ Presets ▾
          </button>
          {openMenu === "presets" && (
            <div
              style={{
                position: "absolute",
                top: "100%",
                left: 0,
                marginTop: 4,
                width: 290,
                background: "#ffffff",
                border: "1px solid #cbd5e1",
                borderRadius: 8,
                boxShadow: "0 10px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.08)",
                padding: "6px 0",
                zIndex: 100,
                maxHeight: 380,
                overflowY: "auto"
              }}
            >
              <div style={{ padding: "6px 14px 4px", fontSize: 11, fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: ".05em" }}>
                Quantum Algorithm Library
              </div>
              {CIRCUIT_PRESETS.map((p) => (
                <button
                  key={p.id}
                  onClick={() => { loadPreset(p.id); setOpenMenu(null); }}
                  style={{ width: "100%", textAlign: "left", padding: "8px 14px", border: "none", background: "transparent", fontSize: 13, color: "#0f172a", display: "block", cursor: "pointer" }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "#f1f5f9")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                >
                  <div style={{ fontWeight: 600, color: "#1e293b" }}>{p.name}</div>
                  <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>{p.description}</div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* EXPORT MENU */}
        <div style={{ position: "relative" }}>
          <button
            onClick={() => setOpenMenu(openMenu === "export" ? null : "export")}
            style={{
              padding: "4px 9px",
              borderRadius: 4,
              border: "none",
              background: openMenu === "export" ? "#e2e8f0" : "transparent",
              fontWeight: 500,
              color: "#1e293b",
              display: "flex",
              alignItems: "center",
              gap: 4,
              cursor: "pointer"
            }}
          >
            📤 Export ▾
          </button>
          {openMenu === "export" && (
            <div
              style={{
                position: "absolute",
                top: "100%",
                left: 0,
                marginTop: 4,
                width: 250,
                background: "#ffffff",
                border: "1px solid #cbd5e1",
                borderRadius: 8,
                boxShadow: "0 10px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.08)",
                padding: "6px 0",
                zIndex: 100
              }}
            >
              <button
                onClick={() => { exportQasm(); setOpenMenu(null); }}
                style={{ width: "100%", textAlign: "left", padding: "8px 14px", border: "none", background: "transparent", fontSize: 13, color: "#0f172a", display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#f1f5f9")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <span>📄 OpenQASM 3.0</span>
                <span style={{ fontSize: 11, color: "#059669", fontWeight: 600 }}>.qasm</span>
              </button>
              <button
                onClick={() => { exportLatex(); setOpenMenu(null); }}
                style={{ width: "100%", textAlign: "left", padding: "8px 14px", border: "none", background: "transparent", fontSize: 13, color: "#0f172a", display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#f1f5f9")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <span>📐 LaTeX Quantikz2</span>
                <span style={{ fontSize: 11, color: "#0284c7", fontWeight: 600 }}>.tex</span>
              </button>
              <button
                onClick={() => { exportCsv(); setOpenMenu(null); }}
                style={{ width: "100%", textAlign: "left", padding: "8px 14px", border: "none", background: "transparent", fontSize: 13, color: "#0f172a", display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#f1f5f9")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <span>📊 Circuit Operations CSV</span>
                <span style={{ fontSize: 11, color: "#d97706", fontWeight: 600 }}>.csv</span>
              </button>
              <div style={{ height: 1, background: "#e2e8f0", margin: "4px 0" }} />
              <button
                onClick={() => { exportIr(); setOpenMenu(null); }}
                style={{ width: "100%", textAlign: "left", padding: "8px 14px", border: "none", background: "transparent", fontSize: 13, color: "#0f172a", display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#f1f5f9")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <span>⚙️ Native Quantum IR</span>
                <span style={{ fontSize: 11, color: "#64748b", fontWeight: 600 }}>.json</span>
              </button>
            </div>
          )}
        </div>

        {/* VIEW MENU */}
        <div style={{ position: "relative" }}>
          <button
            onClick={() => setOpenMenu(openMenu === "view" ? null : "view")}
            style={{
              padding: "4px 9px",
              borderRadius: 4,
              border: "none",
              background: openMenu === "view" ? "#e2e8f0" : "transparent",
              fontWeight: 500,
              color: "#1e293b",
              display: "flex",
              alignItems: "center",
              gap: 4,
              cursor: "pointer"
            }}
          >
            View ▾
          </button>
          {openMenu === "view" && (
            <div
              style={{
                position: "absolute",
                top: "100%",
                left: 0,
                marginTop: 4,
                width: 200,
                background: "#ffffff",
                border: "1px solid #cbd5e1",
                borderRadius: 8,
                boxShadow: "0 10px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.08)",
                padding: "6px 0",
                zIndex: 100
              }}
            >
              <button
                onClick={() => {
                  try { rfRef.current?.fitView({ padding: 0.25, duration: 250 }); } catch {}
                  setOpenMenu(null);
                }}
                style={{ width: "100%", textAlign: "left", padding: "8px 14px", border: "none", background: "transparent", fontSize: 13, color: "#0f172a", display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#f1f5f9")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <span>⛶ Fit to View</span>
                <span style={{ fontSize: 11, color: "#94a3b8" }}>Auto</span>
              </button>
              <button
                onClick={() => {
                  try { rfRef.current?.setViewport({ x: 0, y: 12, zoom: 1 }, { duration: 0 }); } catch {}
                  setOpenMenu(null);
                }}
                style={{ width: "100%", textAlign: "left", padding: "8px 14px", border: "none", background: "transparent", fontSize: 13, color: "#0f172a", display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#f1f5f9")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <span>↺ Reset Scale</span>
                <span style={{ fontSize: 11, color: "#94a3b8" }}>100%</span>
              </button>
              <div style={{ height: 1, background: "#e2e8f0", margin: "4px 0" }} />
              <button
                onClick={() => { setShowMiniMap((s) => !s); setOpenMenu(null); }}
                style={{ width: "100%", textAlign: "left", padding: "8px 14px", border: "none", background: "transparent", fontSize: 13, color: "#0f172a", display: "flex", justifyContent: "space-between", alignItems: "center", cursor: "pointer" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#f1f5f9")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
              >
                <span>🗺 MiniMap</span>
                <span style={{ fontSize: 11, color: showMiniMap ? "#10b981" : "#94a3b8", fontWeight: 600 }}>{showMiniMap ? "ON" : "OFF"}</span>
              </button>
            </div>
          )}
        </div>

        {/* Hidden Import file input */}
        <input
          ref={importInputRef}
          type="file"
          accept=".json,.qcir.json,.qasm,.txt,application/json,text/plain"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void importCircuit(file);
          }}
          style={{ display: "none" }}
        />

        {/* Right side stats & quick actions */}
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 14 }}>
          <button
            onClick={() => {
              try { rfRef.current?.fitView({ padding: 0.25, duration: 250 }); } catch {}
            }}
            title="Ekranga moslash"
            style={{ padding: "3px 8px", fontSize: 12, borderRadius: 4, border: "1px solid #cbd5e1", background: "#fff", color: "#475569", cursor: "pointer" }}
          >
            Fit View
          </button>
          <span style={{ fontSize: 12, color: "#64748b" }}>
            <strong style={{ color: "#0f172a" }}>{nQubits}</strong> qubits · <strong style={{ color: "#0f172a" }}>{ops.length}</strong> gates · <span style={{ color: rfReady ? "#10b981" : "#f59e0b" }}>{rfReady ? "Ready" : "Loading"}</span>
          </span>
        </div>
      </div>

      {/* 2. Professional Gate Composer & Placement Bar */}
      <div
        style={{
          padding: "8px 14px",
          background: "#ffffff",
          borderBottom: "1px solid #e5e7eb",
          display: "flex",
          alignItems: "center",
          gap: 12,
          flexWrap: "wrap",
          fontSize: 13
        }}
      >
        {/* Qubit count spinner */}
        <label style={{ display: "flex", gap: 6, alignItems: "center", fontWeight: 500, color: "#334155" }}>
          Qubits
          <input
            type="number"
            min={1}
            max={64}
            value={nQubits}
            onChange={(e) => setNQubits(Math.max(1, Math.min(64, Number(e.target.value) || 1)))}
            style={{ width: 56, padding: "5px 8px", borderRadius: 5, border: "1px solid #cbd5e1", fontSize: 13, textAlign: "center" }}
          />
        </label>

        <div style={{ width: 1, height: 22, background: "#e2e8f0" }} />

        {/* Gate Selection Pills */}
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ fontSize: 12, color: "#64748b", marginRight: 2, fontWeight: 500 }}>Gate:</span>
          {gateList.map((g) => {
            const isSelected = newGate.name === g.id;
            return (
              <button
                key={g.id}
                onClick={() => setNewGate((prev) => ({ ...prev, name: g.id }))}
                title={g.desc}
                style={{
                  padding: "4px 9px",
                  borderRadius: 5,
                  border: isSelected ? "1px solid #2563eb" : "1px solid #e2e8f0",
                  background: isSelected ? "#eff6ff" : "#f8fafc",
                  color: isSelected ? "#1d4ed8" : "#334155",
                  fontWeight: isSelected ? 700 : 500,
                  fontSize: 12,
                  cursor: "pointer",
                  transition: "all 100ms ease"
                }}
              >
                {g.label}
              </button>
            );
          })}
        </div>

        <div style={{ width: 1, height: 22, background: "#e2e8f0" }} />

        {/* Target and Control Inputs */}
        <label style={{ display: "flex", gap: 6, alignItems: "center", color: "#334155", fontSize: 13 }}>
          Target q
          <input
            type="number"
            min={0}
            max={nQubits - 1}
            value={newGate.target}
            onChange={(e) => setNewGate((g) => ({ ...g, target: Number(e.target.value) || 0 }))}
            style={{ width: 50, padding: "5px 8px", borderRadius: 5, border: "1px solid #cbd5e1", fontSize: 13, textAlign: "center" }}
          />
        </label>

        {(newGate.name === "cx" || newGate.name === "cz") && (
          <label style={{ display: "flex", gap: 6, alignItems: "center", color: "#334155", fontSize: 13 }}>
            Control q
            <input
              type="number"
              min={0}
              max={nQubits - 1}
              value={newGate.control}
              onChange={(e) => setNewGate((g) => ({ ...g, control: Number(e.target.value) || 0 }))}
              style={{ width: 50, padding: "5px 8px", borderRadius: 5, border: "1px solid #cbd5e1", fontSize: 13, textAlign: "center" }}
            />
          </label>
        )}

        {(newGate.name === "rx" || newGate.name === "ry" || newGate.name === "rz") && (
          <label style={{ display: "flex", gap: 6, alignItems: "center", color: "#334155", fontSize: 13 }}>
            Angle θ
            <input
              type="number"
              step="0.05"
              value={newGate.theta}
              onChange={(e) => setNewGate((g) => ({ ...g, theta: Number(e.target.value) || 0 }))}
              style={{ width: 75, padding: "5px 8px", borderRadius: 5, border: "1px solid #cbd5e1", fontSize: 13, textAlign: "center" }}
            />
          </label>
        )}

        {/* Add Gate Primary Action */}
        <Button variant="primary" onClick={addGate}>
          + Add Gate
        </Button>

        {/* Undo/Redo quick buttons */}
        <div style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
          <button
            disabled={!canUndo}
            onClick={undo}
            title="Undo (Ctrl+Z)"
            style={{
              padding: "5px 9px",
              borderRadius: 4,
              border: "1px solid #e2e8f0",
              background: "#fff",
              color: canUndo ? "#334155" : "#cbd5e1",
              fontSize: 12,
              cursor: canUndo ? "pointer" : "not-allowed"
            }}
          >
            ↶
          </button>
          <button
            disabled={!canRedo}
            onClick={redo}
            title="Redo (Ctrl+Y)"
            style={{
              padding: "5px 9px",
              borderRadius: 4,
              border: "1px solid #e2e8f0",
              background: "#fff",
              color: canRedo ? "#334155" : "#cbd5e1",
              fontSize: 12,
              cursor: canRedo ? "pointer" : "not-allowed"
            }}
          >
            ↷
          </button>
        </div>
      </div>

      {editorNotice ? <div role="status" style={{ padding: "6px 10px", color: "#92400e", background: "#fffbeb", borderBottom: "1px solid #fde68a", fontSize: 12 }}>{editorNotice}</div> : null}
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onError={(code, message) => {
          // React Strict Mode invokes React Flow's type-key memo twice in dev;
          // error 002 is a known false positive when these maps are stable.
          if (code !== "002") console.warn(`[React Flow ${code}] ${message}`);
        }}
        onInit={(inst) => {
          rfRef.current = inst;
          setRfReady(true);
          try {
            inst.setViewport({ x: 0, y: 12, zoom: 1 }, { duration: 0 });
          } catch {}
        }}
      >
        <Background gap={20} size={1} color="#e2e8f0" />
        {showMiniMap ? <MiniMap /> : null}
        <Controls />
        {ops.length === 0 ? (
          <div style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            background: "rgba(255, 255, 255, 0.95)",
            backdropFilter: "blur(4px)",
            border: "1px solid #cbd5e1",
            borderRadius: 12,
            padding: "24px 32px",
            boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)",
            textAlign: "center",
            maxWidth: 500,
            zIndex: 10,
          }}>
            <h3 style={{ margin: "0 0 8px 0", fontSize: 17, color: "#0f172a" }}>Kvant Sxemasi Laboratoriyasiga Xush Kelibsiz!</h3>
            <p style={{ margin: "0 0 16px 0", fontSize: 13, color: "#64748b", lineHeight: 1.5 }}>
              Sxema tuzish uchun yuqoridagi paneldan geyt qo‘shing yoki quyidagi tayyor shablonlardan birini tanlang:
            </p>
            <div style={{ display: "flex", gap: 8, justifyContent: "center", flexWrap: "wrap", marginBottom: 14 }}>
              <Button variant="primary" onClick={() => loadPreset("bell-state")}>
                ✨ Bell State (|Φ⁺⟩)
              </Button>
              <Button variant="secondary" onClick={() => loadPreset("ghz-state")}>
                GHZ State (3q)
              </Button>
              <Button variant="secondary" onClick={() => loadPreset("superposition")}>
                Superposition (4q)
              </Button>
            </div>
            <div style={{ fontSize: 12, color: "#94a3b8", borderTop: "1px solid #e2e8f0", paddingTop: 10 }}>
              💡 Maslahat: Sxemani hisoblash va natijalarni ko‘rish uchun yuqoridagi <b>run</b> tabiga o‘ting.
            </div>
          </div>
        ) : null}

      </ReactFlow>
      {(analysis?.counts || analysis?.amplitudes || analysis?.complexity) && (
        <div style={{ borderTop: "1px solid #d1d5db", background: "#ffffff", padding: 10, fontSize: 12 }}>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>Latest Analysis</div>
          {analysis.complexity ? (
            <div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginBottom: 6 }}>
              <span>path_steps: {String(analysis.complexity.pathSteps ?? "-")}</span>
              <span>opt_cost: {String(analysis.complexity.optCost ?? "-")}</span>
              <span>largest_intermediate: {String(analysis.complexity.largestIntermediate ?? "-")}</span>
              <span>speedup: {String(analysis.complexity.speedup ?? "-")}</span>
            </div>
          ) : null}
          {analysis.counts ? (
            <details>
              <summary>Counts</summary>
              <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{JSON.stringify(analysis.counts, null, 2)}</pre>
            </details>
          ) : null}
          {analysis.amplitudes ? (
            <details>
              <summary>Amplitudes</summary>
              <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{JSON.stringify(analysis.amplitudes, null, 2)}</pre>
            </details>
          ) : null}
        </div>
      )}
    </div>
  );
}

