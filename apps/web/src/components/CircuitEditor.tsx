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

const LANE_START_X = 80;
const LANE_END_PADDING = 180;
const COL_W = 140;
const X0 = 120;

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

const nodeTypes = { gateNode: GateNode, laneNode: LaneNode };

export function CircuitEditor() {
  const { ops, setOps, nQubits, setNQubits, clear, undo, redo, canUndo, canRedo, analysis } = useCircuit();
  const rfRef = React.useRef<ReactFlowInstance | null>(null);
  const [rfReady, setRfReady] = React.useState(false);
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
        const occ = buildOccupancy(prev);
        for (const n of nextNodes) {
          const opId = n.data?.opId;
          if (!opId) continue;
          const op = byOp.get(opId);
          if (!op) continue;

          // remove current op from occupancy before re-placing it
          const occWithout = buildOccupancy(Array.from(byOp.values()), opId);
          const desiredCol = xToCol(n.position.x);
          const newQ = laneToQubit(n.position.y, nQubits);

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
    const id = `g${Date.now().toString(36)}`;
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
      base.y = 60 + base.target * 60;
    }
    // Place into first non-colliding column (same column reserves both qubits for 2q gates)
    const occ = buildOccupancy(ops);
    const placedCol = resolveCol(occ, base.col, opQubits(base), base.id);
    base.col = placedCol;
    base.x = colToX(placedCol);

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

  const showMiniMap = nodes.length <= 200;

  return (
    <div style={{ height: "100%", width: "100%" }}>
      <div style={{ padding: 8, borderBottom: "1px solid #e5e7eb", display: "flex", gap: 8, flexWrap: "wrap" }}>
        <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
          Qubits
          <input
            type="number"
            min={1}
            max={64}
            value={nQubits}
            onChange={(e) => setNQubits(Math.max(1, Math.min(64, Number(e.target.value) || 1)))}
            style={{ width: 70 }}
          />
        </label>
        <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
          Gate
          <select value={newGate.name} onChange={(e) => setNewGate((g) => ({ ...g, name: e.target.value as GateName }))}>
            <option value="h">h</option>
            <option value="x">x</option>
            <option value="rx">rx</option>
            <option value="ry">ry</option>
            <option value="rz">rz</option>
            <option value="cx">cx</option>
            <option value="cz">cz</option>
          </select>
        </label>
        <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
          Target
          <input
            type="number"
            min={0}
            max={nQubits - 1}
            value={newGate.target}
            onChange={(e) => setNewGate((g) => ({ ...g, target: Number(e.target.value) || 0 }))}
            style={{ width: 70 }}
          />
        </label>
        {(newGate.name === "cx" || newGate.name === "cz") && (
          <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
            Control
            <input
              type="number"
              min={0}
              max={nQubits - 1}
              value={newGate.control}
              onChange={(e) => setNewGate((g) => ({ ...g, control: Number(e.target.value) || 0 }))}
              style={{ width: 70 }}
            />
          </label>
        )}
        {(newGate.name === "rx" || newGate.name === "ry" || newGate.name === "rz") && (
          <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
            θ
            <input
              type="number"
              value={newGate.theta}
              onChange={(e) => setNewGate((g) => ({ ...g, theta: Number(e.target.value) || 0 }))}
              style={{ width: 90 }}
            />
          </label>
        )}
        <button onClick={addGate}>Add</button>
        <button
          onClick={() => {
            try {
              rfRef.current?.fitView({ padding: 0.25, duration: 250 });
            } catch {}
          }}
        >
          Fit
        </button>
        <button
          onClick={() => {
            try {
              rfRef.current?.setViewport({ x: 0, y: 0, zoom: 1 }, { duration: 0 });
            } catch {}
          }}
        >
          Reset View
        </button>
        <button onClick={undo} disabled={!canUndo}>
          Undo
        </button>
        <button onClick={redo} disabled={!canRedo}>
          Redo
        </button>
        <button onClick={clear}>Clear</button>
        <span style={{ marginLeft: "auto", fontSize: 12, color: "#6b7280" }}>
          {nQubits} qubits | {ops.length} gates | {rfReady ? "ready" : "loading"} | analysis: {analysis?.source ?? "none"}
        </span>
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        nodeTypes={nodeTypes}
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

