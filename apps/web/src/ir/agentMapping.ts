import type { CircuitIrV1, IrNode } from "./ir";

type AgentGate = Pick<IrNode, "name" | "target"> & Partial<Pick<IrNode, "control" | "theta" | "parameter">>;

export function irToAgentTNPayload(ir: CircuitIrV1) {
  // Current agent expects { n_qubits, gates, bitstrings?, dtype?, optimize? }
  const gates = [...ir.ui.nodes]
    .sort((a, b) => ((a.moment ?? a.col) - (b.moment ?? b.col)) || ((a.sequence ?? 0) - (b.sequence ?? 0)))
    .map((g) => {
      const out: AgentGate = { name: g.name, target: g.target };
      if (g.control !== undefined) out.control = g.control;
      if (g.theta !== undefined) out.theta = g.theta;
      if (g.parameter !== undefined) out.parameter = g.parameter;
      return out;
    });

  return { n_qubits: ir.ui.n_qubits, gates };
}
