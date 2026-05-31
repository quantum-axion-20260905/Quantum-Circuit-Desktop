import type { GateOp } from "../state/circuitStore";

export function compileToAgentPayload(nQubits: number, ops: GateOp[]) {
  const gates = [...ops]
    .sort((a, b) => (a.x - b.x) || (a.y - b.y))
    .map((g) => {
      const base: any = { name: g.name, target: g.target };
      if (g.control !== undefined) base.control = g.control;
      if (g.theta !== undefined) base.theta = g.theta;
      return base;
    });

  return { n_qubits: nQubits, gates };
}

