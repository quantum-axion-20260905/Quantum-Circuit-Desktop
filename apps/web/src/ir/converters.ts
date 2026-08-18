import type { GateOp } from "../state/circuitStore";
import type { CircuitIrV1, IrNode } from "./ir";
import { emitQasm3, parseMinimalQasm3 } from "./qasm3";

export function editorToIr(nQubits: number, ops: GateOp[]): CircuitIrV1 {
  const nodes: IrNode[] = ops.map((g) => ({
    id: g.id,
    name: g.name,
    target: g.target,
    control: g.control,
    theta: g.theta,
    col: g.col,
    moment: g.col,
    sequence: 0,
    x: g.x,
    y: g.y
  }));

  const qasm = emitQasm3(nQubits, nodes);
  return {
    qasm,
    ui: {
      version: 1,
      source_format: "qasm3",
      n_qubits: nQubits,
      nodes
    }
  };
}

export function irToEditor(ir: CircuitIrV1): { nQubits: number; ops: GateOp[] } {
  // Prefer UI nodes for exact layout.
  if (ir.ui?.nodes?.length) {
    return {
      nQubits: ir.ui.n_qubits,
      ops: ir.ui.nodes.map((n) => ({
        id: n.id,
        name: n.name,
        target: n.target,
        control: n.control,
        theta: n.theta,
        col: (n as any).col ?? 0,
        x: n.x,
        y: n.y
      }))
    };
  }

  // Fallback: parse minimal QASM and generate a basic layout.
  const parsed = parseMinimalQasm3(ir.qasm);
  const ops: GateOp[] = parsed.ops.map((o, idx) => ({
    id: `p${idx}`,
    name: o.name,
    target: o.target,
    control: o.control,
    theta: o.theta,
    col: idx,
    x: 120 + idx * 140,
    y: 60 + o.target * 60
  }));
  return { nQubits: parsed.nQubits, ops };
}
