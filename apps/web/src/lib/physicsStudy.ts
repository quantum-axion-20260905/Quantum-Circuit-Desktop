import type { JsonObject, PauliTerm } from "./agent";

export type PhysicsStudyMode = "dmrg" | "tebd" | "peps";

export type PhysicsStudyVariant = {
  label: string;
  parameters: string;
  payload: JsonObject;
};

export type PhysicsStudyConfig = {
  mode: PhysicsStudyMode;
  nQubits: number;
  terms: PauliTerm[];
  lattice?: JsonObject;
  bondDim: number;
  sweeps: number;
  steps: number;
  dt: number;
  truncationCutoff?: number;
  maxPoints?: number;
};

export const MAX_PHYSICS_STUDY_POINTS = 3;

function boundedInteger(value: number, minimum: number, maximum: number, fallback: number): number {
  return Number.isFinite(value) ? Math.min(maximum, Math.max(minimum, Math.trunc(value))) : fallback;
}

function boundedNumber(value: number, minimum: number, fallback: number): number {
  return Number.isFinite(value) && Math.abs(value) >= minimum ? value : fallback;
}

/**
 * Build a small, deterministic set of convergence points for the Physics Lab.
 * The helper owns all UI-side bounds so a new caller cannot accidentally turn
 * one click into an unbounded Cartesian study.
 */
export function buildPhysicsStudyVariants(config: PhysicsStudyConfig): PhysicsStudyVariant[] {
  const safeBond = boundedInteger(config.bondDim, 1, 32, 16);
  const safeSteps = boundedInteger(config.steps, 1, 64, 1);
  const safeSweeps = boundedInteger(config.sweeps, 1, 64, 4);
  const safeDt = boundedNumber(config.dt, 1e-8, 0.01);
  const safeCutoff = Number.isFinite(config.truncationCutoff ?? 0) ? Math.max(0, config.truncationCutoff ?? 0) : 0;
  const pepsLowBond = Math.max(1, Math.min(2, Math.floor(Math.min(4, safeBond) / 2) || 1));
  const pepsHighBond = Math.min(4, Math.max(pepsLowBond + 1, Math.min(4, safeBond)));
  const common: JsonObject = {
    n_qubits: config.nQubits,
    terms: config.terms,
    dtype: "complex64",
    truncation_cutoff: safeCutoff,
    max_time_ms: 120000,
    max_mem_mb: 1024,
  };
  const lattice = config.lattice ? { lattice: config.lattice } : {};
  const variants: PhysicsStudyVariant[] = config.mode === "dmrg"
    ? [
      { label: "Lower bond", parameters: `χ=${Math.max(1, Math.floor(safeBond / 2))}, sweeps=${safeSweeps}`, payload: { ...common, backend: "tensor-network", bond_dim: Math.max(1, Math.floor(safeBond / 2)), sweeps: safeSweeps, tolerance: 1e-7 } },
      { label: "Baseline bond", parameters: `χ=${safeBond}, sweeps=${safeSweeps}`, payload: { ...common, backend: "tensor-network", bond_dim: safeBond, sweeps: safeSweeps, tolerance: 1e-7 } },
      { label: "More sweeps", parameters: `χ=${safeBond}, sweeps=${Math.min(64, Math.max(safeSweeps + 1, safeSweeps * 2))}`, payload: { ...common, backend: "tensor-network", bond_dim: safeBond, sweeps: Math.min(64, Math.max(safeSweeps + 1, safeSweeps * 2)), tolerance: 1e-7 } },
    ]
    : config.mode === "tebd"
      ? [
        { label: "Baseline time step", parameters: `dt=${safeDt}, steps=${safeSteps}, χ=${safeBond}`, payload: { ...common, ...lattice, backend: "tensor-network", dt: safeDt, steps: safeSteps, order: 2, bond_dim: safeBond } },
        { label: "Half time step", parameters: `dt=${safeDt / 2}, steps=${Math.min(64, safeSteps * 2)}, χ=${safeBond}`, payload: { ...common, ...lattice, backend: "tensor-network", dt: safeDt / 2, steps: Math.min(64, safeSteps * 2), order: 2, bond_dim: safeBond } },
        { label: "Half dt + higher bond", parameters: `dt=${safeDt / 2}, steps=${Math.min(64, safeSteps * 2)}, χ=${Math.min(64, safeBond * 2)}`, payload: { ...common, ...lattice, backend: "tensor-network", dt: safeDt / 2, steps: Math.min(64, safeSteps * 2), order: 2, bond_dim: Math.min(64, safeBond * 2) } },
      ]
      : [
        { label: "Lower PEPS bond", parameters: `dt=${safeDt}, steps=${safeSteps}, χ=${pepsLowBond}`, payload: { ...common, ...lattice, backend: "tensor-network", bond_dim: pepsLowBond, dt: safeDt, steps: safeSteps, order: 2, max_contraction_states: 1_000_000 } },
        { label: "Higher PEPS bond", parameters: `dt=${safeDt}, steps=${safeSteps}, χ=${pepsHighBond}`, payload: { ...common, ...lattice, backend: "tensor-network", bond_dim: pepsHighBond, dt: safeDt, steps: safeSteps, order: 2, max_contraction_states: 1_000_000 } },
        { label: "Half dt", parameters: `dt=${safeDt / 2}, steps=${Math.min(64, safeSteps * 2)}, χ=${pepsHighBond}`, payload: { ...common, ...lattice, backend: "tensor-network", bond_dim: pepsHighBond, dt: safeDt / 2, steps: Math.min(64, safeSteps * 2), order: 2, max_contraction_states: 1_000_000 } },
      ];
  const requestedPoints = boundedInteger(config.maxPoints ?? MAX_PHYSICS_STUDY_POINTS, 1, MAX_PHYSICS_STUDY_POINTS, MAX_PHYSICS_STUDY_POINTS);
  return variants.slice(0, requestedPoints);
}
