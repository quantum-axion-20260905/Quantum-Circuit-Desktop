import type { JsonObject, PauliTerm } from "./agent";

export type PhysicsStudyMode = "dmrg" | "tebd" | "peps";

export type PhysicsStudyStatus = "queued" | "running" | "done" | "failed" | "canceled";

export type PhysicsStudyRow = {
  id: string;
  label: string;
  parameters: string;
  status: PhysicsStudyStatus;
  request?: JsonObject;
  result?: JsonObject;
  error?: string;
};

export type PhysicsStudySummary = {
  completed: number;
  failed: number;
  canceled: number;
  pending: number;
  energy_range: number | null;
  energy_half_range: number | null;
  max_discarded_weight: number | null;
  max_norm_drift: number | null;
  verdict: "stable" | "needs_review" | "incomplete" | "not_run";
};

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
  boundaryBondDim?: number;
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

function isOpen2DLattice(lattice: JsonObject | undefined): boolean {
  const dimensions = lattice?.dimensions;
  return Array.isArray(dimensions)
    && dimensions.length === 2
    && lattice?.boundary !== "periodic";
}

function numeric(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function studyEnergy(result: JsonObject | undefined): number | null {
  if (!result) return null;
  const direct = numeric(result.ground_energy ?? result.energy);
  if (direct != null) return direct;
  const energies = Array.isArray(result.energies) ? result.energies.map(numeric).filter((value): value is number => value != null) : [];
  return energies.length ? energies[energies.length - 1] : null;
}

export function studyDiscardedWeight(result: JsonObject | undefined): number | null {
  if (!result) return null;
  const researchResult = result.research_result;
  if (researchResult && typeof researchResult === "object") {
    const truncation = (researchResult as JsonObject).truncation;
    if (truncation && typeof truncation === "object") {
      const value = numeric((truncation as JsonObject).discarded_weight);
      if (value != null) return value;
    }
  }
  return numeric(result.discarded_weight);
}

export function studyNormDrift(result: JsonObject | undefined): number | null {
  const norm = numeric(result?.norm2);
  return norm == null ? null : Math.abs(norm - 1);
}

export function summarizePhysicsStudy(rows: PhysicsStudyRow[]): PhysicsStudySummary {
  const completed = rows.filter((row) => row.status === "done");
  const energies = completed.map((row) => studyEnergy(row.result)).filter((value): value is number => value != null);
  const discarded = completed.map((row) => studyDiscardedWeight(row.result)).filter((value): value is number => value != null);
  const normDrifts = completed.map((row) => studyNormDrift(row.result)).filter((value): value is number => value != null);
  const energyRange = energies.length > 1 ? Math.max(...energies) - Math.min(...energies) : null;
  const maxDiscarded = discarded.length ? Math.max(...discarded) : null;
  const maxNormDrift = normDrifts.length ? Math.max(...normDrifts) : null;
  const pending = rows.filter((row) => row.status === "queued" || row.status === "running").length;
  const failed = rows.filter((row) => row.status === "failed").length;
  const canceled = rows.filter((row) => row.status === "canceled").length;
  const stable = completed.length >= 2
    && failed === 0
    && canceled === 0
    && pending === 0
    && energyRange != null
    && maxDiscarded != null
    && maxNormDrift != null
    && energyRange <= 1e-4
    && maxDiscarded <= 1e-4
    && maxNormDrift <= 1e-4;
  return {
    completed: completed.length,
    failed,
    canceled,
    pending,
    energy_range: energyRange,
    energy_half_range: energyRange == null ? null : energyRange / 2,
    max_discarded_weight: maxDiscarded,
    max_norm_drift: maxNormDrift,
    verdict: rows.length === 0 ? "not_run" : pending > 0 || failed > 0 || canceled > 0 ? "incomplete" : stable ? "stable" : "needs_review",
  };
}

function pointUncertainty(row: PhysicsStudyRow): JsonObject {
  const result = row.result;
  const energy = studyEnergy(result);
  const energyStd = numeric(result?.energy_std);
  const energyVariance = numeric(result?.energy_variance);
  const discarded = studyDiscardedWeight(result);
  const normDrift = studyNormDrift(result);
  const preflight = result?.preflight;
  return {
    energy: energy ?? null,
    energy_std: energyStd,
    energy_variance: energyVariance,
    discarded_weight: discarded,
    norm_drift: normDrift,
    preflight_feasible: preflight && typeof preflight === "object" ? (preflight as JsonObject).feasible ?? null : null,
  };
}

export function buildPhysicsStudyManifest(input: {
  mode: PhysicsStudyMode;
  nQubits: number;
  startedAt: string;
  finishedAt: string;
  configuration: JsonObject;
  rows: PhysicsStudyRow[];
}): JsonObject {
  const summary = summarizePhysicsStudy(input.rows);
  return {
    schema: "quantum-circuit/physics-study-v1",
    domain: "spin-lattice",
    source: "physics",
    kind: "convergence",
    mode: input.mode,
    n_qubits: input.nQubits,
    started_at: input.startedAt,
    finished_at: input.finishedAt,
    configuration: input.configuration,
    summary,
    points: input.rows.map((row) => ({
      point_id: row.id,
      label: row.label,
      parameters: row.parameters,
      status: row.status,
      request: row.request ?? {},
      result: row.result ?? null,
      uncertainty: pointUncertainty(row),
      error: row.error ?? null,
    })),
  };
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
  const safeBoundaryBond = boundedInteger(config.boundaryBondDim ?? 16, 1, 64, 16);
  const boundaryLowBond = Math.max(1, Math.floor(safeBoundaryBond / 2));
  const boundaryHighBond = Math.max(boundaryLowBond, safeBoundaryBond);
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
      : (() => {
        if (isOpen2DLattice(config.lattice)) {
          return [
            { label: "Lower environment", parameters: `D=${pepsHighBond}, χ_env=${boundaryLowBond}`, payload: { ...common, ...lattice, backend: "tensor-network", bond_dim: pepsHighBond, boundary_bond_dim: boundaryLowBond, contraction_method: "boundary-mps", dt: safeDt, steps: safeSteps, order: 2, max_contraction_states: 1_000_000 } },
            { label: "Higher environment", parameters: `D=${pepsHighBond}, χ_env=${boundaryHighBond}`, payload: { ...common, ...lattice, backend: "tensor-network", bond_dim: pepsHighBond, boundary_bond_dim: boundaryHighBond, contraction_method: "boundary-mps", dt: safeDt, steps: safeSteps, order: 2, max_contraction_states: 1_000_000 } },
            { label: "Higher χ + half dt", parameters: `D=${pepsHighBond}, χ_env=${boundaryHighBond}, dt=${safeDt / 2}`, payload: { ...common, ...lattice, backend: "tensor-network", bond_dim: pepsHighBond, boundary_bond_dim: boundaryHighBond, contraction_method: "boundary-mps", dt: safeDt / 2, steps: Math.min(64, safeSteps * 2), order: 2, max_contraction_states: 1_000_000 } },
          ];
        }
        return [
          { label: "Lower PEPS bond", parameters: `dt=${safeDt}, steps=${safeSteps}, D=${pepsLowBond}`, payload: { ...common, ...lattice, backend: "tensor-network", bond_dim: pepsLowBond, dt: safeDt, steps: safeSteps, order: 2, max_contraction_states: 1_000_000 } },
          { label: "Higher PEPS bond", parameters: `dt=${safeDt}, steps=${safeSteps}, D=${pepsHighBond}`, payload: { ...common, ...lattice, backend: "tensor-network", bond_dim: pepsHighBond, dt: safeDt, steps: safeSteps, order: 2, max_contraction_states: 1_000_000 } },
          { label: "Half dt", parameters: `dt=${safeDt / 2}, steps=${Math.min(64, safeSteps * 2)}, D=${pepsHighBond}`, payload: { ...common, ...lattice, backend: "tensor-network", bond_dim: pepsHighBond, dt: safeDt / 2, steps: Math.min(64, safeSteps * 2), order: 2, max_contraction_states: 1_000_000 } },
        ];
      })();
  const requestedPoints = boundedInteger(config.maxPoints ?? MAX_PHYSICS_STUDY_POINTS, 1, MAX_PHYSICS_STUDY_POINTS, MAX_PHYSICS_STUDY_POINTS);
  return variants.slice(0, requestedPoints);
}
