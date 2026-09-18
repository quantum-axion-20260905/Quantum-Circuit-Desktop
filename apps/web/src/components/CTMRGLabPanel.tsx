"use client";

import React from "react";
import { buildCTMRG, cancelAsyncJob, runCTMRG, runCTMRGConvergence, type AgentResult, type AsyncJob, type JsonObject } from "../lib/agent";
import type { ReplayRequest } from "../lib/runHistory";
import { Button, Card, Field, Metric, MetricGrid } from "../ui";
import { ComputeProgress } from "./ComputeProgress";

type SpinModel = "ising" | "heisenberg" | "xxz";
type InitialState = "up" | "down" | "plus" | "neel";

type Props = {
  dimensions: number[];
  model: SpinModel;
  coupling: number;
  field: number;
  anisotropy: number;
  onResult?: (result: AgentResult, request: ReplayRequest, label: string) => void;
};

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function CTMRGLabPanel({ dimensions, model, coupling, field, anisotropy, onResult }: Props) {
  const eligible = dimensions.length === 2 && dimensions.every((value) => value >= 1 && value <= 2);
  const [initialState, setInitialState] = React.useState<InitialState>("up");
  const [environmentChi, setEnvironmentChi] = React.useState(2);
  const [iterations, setIterations] = React.useState(8);
  const [busy, setBusy] = React.useState<"run" | "study" | null>(null);
  const [job, setJob] = React.useState<AsyncJob | null>(null);
  const [canceling, setCanceling] = React.useState(false);
  const [result, setResult] = React.useState<AgentResult | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  const modelRequest = React.useCallback((chi: number): JsonObject => ({
    dimensions,
    boundary: "periodic",
    model,
    coupling,
    field,
    anisotropy,
    initial_state: initialState,
    environment_bond_dim: chi,
    iterations,
    tolerance: 1e-8,
  }), [anisotropy, coupling, dimensions, field, initialState, iterations, model]);

  async function execute(kind: "run" | "study") {
    if (!eligible) return;
    setBusy(kind); setError(null); setResult(null); setJob(null); setCanceling(false);
    try {
      const problem = await buildCTMRG("spin-lattice", modelRequest(kind === "study" ? 4 : environmentChi));
      const request: ReplayRequest = kind === "study"
        ? { source: "physics", kind: "ctmrg_convergence", payload: { problem, environment_bond_dims: [1, 2, 4], backend: "auto" } }
        : { source: "physics", kind: "ctmrg", payload: { ...problem, backend: "auto" } };
      const output = kind === "study"
        ? await runCTMRGConvergence(request.payload, setJob)
        : await runCTMRG(request.payload, setJob);
      setResult(output);
      onResult?.(output, request, kind === "study" ? "CTMRG χ convergence" : "CTMRG contraction");
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : "CTMRG run failed.");
    } finally {
      setBusy(null); setCanceling(false);
    }
  }

  async function cancel() {
    if (!job || (job.status !== "queued" && job.status !== "running")) return;
    setCanceling(true);
    try { await cancelAsyncJob(job.job_id); } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : "Unable to cancel CTMRG job.");
      setCanceling(false);
    }
  }

  const isStudy = result?.method === "ipeps-ctmrg-environment-convergence-study";
  const latestPoint = isStudy && Array.isArray(result?.points) ? result.points[result.points.length - 1] as Record<string, unknown> : null;
  const singleEnergy = finite(result?.energy) ? result?.energy : latestPoint?.energy;
  const variance = finite(result?.energy_variance) ? result?.energy_variance : latestPoint?.energy_variance;
  const correlationLength = finite(result?.correlation_length) ? result?.correlation_length : latestPoint?.correlation_length;
  const environmentResidual = finite(result?.residual) ? result?.residual : latestPoint?.residual;
  const rawBasisResidual = finite(result?.raw_boundary_basis_residual) ? result?.raw_boundary_basis_residual : latestPoint?.raw_boundary_basis_residual;
  const reference = result?.reference_validation && typeof result.reference_validation === "object" ? result.reference_validation as Record<string, unknown> : null;
  const referenceName = typeof reference?.reference === "string" ? reference.reference : null;
  const referenceError = finite(reference?.max_abs_error) ? Number(reference.max_abs_error) : null;
  const gaugeConditioning = result?.gauge_conditioning && typeof result.gauge_conditioning === "object" ? result.gauge_conditioning as Record<string, unknown> : null;
  const virtualLegStatus = gaugeConditioning?.well_conditioned === true ? "conditioned" : isStudy ? "per point" : "review";
  const researchGate = result?.research_gate && typeof result.research_gate === "object" ? result.research_gate as Record<string, unknown> : null;
  const studyGateSummary = isStudy && result?.research_gate_summary && typeof result.research_gate_summary === "object" ? result.research_gate_summary as Record<string, unknown> : null;
  const admissionStatus = researchGate?.status === "passed" || studyGateSummary?.status === "passed" ? "bounded pass" : "needs review";
  const points = isStudy && Array.isArray(result?.points) ? result.points as Array<Record<string, unknown>> : [];
  const studySummary = isStudy && result?.reference_summary && typeof result.reference_summary === "object" ? result.reference_summary as Record<string, unknown> : null;
  const studyReference = typeof studySummary?.consistent_reference === "string" ? studySummary.consistent_reference : "mixed / unavailable";
  const studyPassed = finite(studySummary?.passed_points) && finite(studySummary?.performed_points)
    ? `${String(studySummary.passed_points)}/${String(studySummary.performed_points)} passed`
    : null;

  return (
    <Card tone="dark" className="qc-ctmrg-panel">
      <div className="qc-diagnostics-header">
        <div><strong>Infinite 2D · CTMRG / iPEPS</strong><p>Bounded periodic 1×1–2×2 research path with explicit χ and reference evidence.</p></div>
        <span className={`qc-diagnostics-badge ${eligible ? "qc-diagnostics-good" : "qc-diagnostics-warn"}`}>{eligible ? "Admitted" : "2×2 limit"}</span>
      </div>
      {!eligible ? <p className="qc-diagnostics-limitations">CTMRG currently accepts only a periodic 2D unit cell with each dimension between 1 and 2. Larger lattices remain disabled until a wider environment contract is implemented.</p> : null}
      <div className="qc-ctmrg-controls">
        <Field label="Initial state"><select value={initialState} onChange={(event) => setInitialState(event.target.value as InitialState)} disabled={!eligible || busy !== null}><option value="up">|↑⟩ product</option><option value="down">|↓⟩ product</option><option value="plus">|+⟩ product</option><option value="neel">Néel product</option></select></Field>
        <Field label="Environment χ"><input type="number" min={1} max={16} value={environmentChi} onChange={(event) => setEnvironmentChi(Math.max(1, Math.min(16, Number(event.target.value) || 1)))} disabled={!eligible || busy !== null} /></Field>
        <Field label="Iterations"><input type="number" min={1} max={64} value={iterations} onChange={(event) => setIterations(Math.max(1, Math.min(64, Number(event.target.value) || 1)))} disabled={!eligible || busy !== null} /></Field>
        <div className="qc-ctmrg-actions"><Button variant="accent" onClick={() => void execute("run")} disabled={!eligible || busy !== null}>{busy === "run" ? "Running…" : "Run CTMRG"}</Button><Button variant="secondary" onClick={() => void execute("study")} disabled={!eligible || busy !== null}>{busy === "study" ? "Studying…" : "χ study · 1/2/4"}</Button></div>
      </div>
      {job ? <ComputeProgress job={job} dark onCancel={() => void cancel()} canceling={canceling} /> : null}
      {result ? <>
        <MetricGrid className="qc-ctmrg-metrics"><Metric label="Energy" value={finite(singleEnergy) ? Number(singleEnergy).toFixed(8) : "—"} tone="info" /><Metric label="Variance" value={finite(variance) ? Number(variance).toExponential(2) : "not available"} tone={finite(variance) && Number(variance) <= 1e-6 ? "success" : "warning"} /><Metric label="CTMRG residual" value={finite(environmentResidual) ? Number(environmentResidual).toExponential(2) : "—"} tone={finite(environmentResidual) && Number(environmentResidual) <= 1e-6 ? "success" : "warning"} /><Metric label="Basis drift" value={finite(rawBasisResidual) ? Number(rawBasisResidual).toExponential(2) : "—"} tone={finite(rawBasisResidual) && finite(environmentResidual) && Number(rawBasisResidual) > Number(environmentResidual) * 10 ? "warning" : "info"} /><Metric label="Virtual legs" value={virtualLegStatus} tone={virtualLegStatus === "conditioned" ? "success" : "warning"} /><Metric label="Research gate" value={admissionStatus} tone={admissionStatus === "bounded pass" ? "success" : "warning"} /><Metric label="Correlation length" value={finite(correlationLength) ? Number(correlationLength).toFixed(5) : "—"} /><Metric label="Reference" value={reference?.passed === true ? "passed" : isStudy ? "per point" : "needs review"} tone={reference?.passed === true ? "success" : "warning"} /></MetricGrid>
        {isStudy ? <div className="qc-ctmrg-study-table"><div className="qc-label">Environment-χ convergence</div>{points.map((point, index) => { const pointReference = point.reference_validation && typeof point.reference_validation === "object" ? point.reference_validation as Record<string, unknown> : null; const pointError = finite(pointReference?.max_abs_error) ? Number(pointReference.max_abs_error).toExponential(2) : "—"; const gatePassed = point.research_gate_status === "passed"; return <div className="qc-ctmrg-study-row" key={`${String(point.environment_bond_dim)}-${index}`}><span>χ={String(point.environment_bond_dim)}</span><code>{finite(point.energy) ? Number(point.energy).toFixed(8) : "—"}</code><code>ΔE {finite(point.energy_delta) ? Number(point.energy_delta).toExponential(2) : "—"}</code><code>Δobs {finite(point.observable_max_abs_delta) ? Number(point.observable_max_abs_delta).toExponential(2) : "—"}</code><code>ξ {finite(point.correlation_length) ? Number(point.correlation_length).toFixed(4) : "—"}</code><code className={pointReference?.passed === true ? "qc-ctmrg-ref-pass" : "qc-ctmrg-ref-review"}>{pointReference?.passed === true ? "ref ✓" : `ref ${pointError}`}</code><code className={gatePassed ? "qc-ctmrg-ref-pass" : "qc-ctmrg-ref-review"}>{gatePassed ? "gate ✓" : "gate review"}</code></div>; })}</div> : null}
        {isStudy && studySummary ? <p className="qc-ctmrg-reference"><strong>Study reference:</strong> {studyReference}{studyPassed ? ` · ${studyPassed}` : ""}</p> : null}
        {isStudy && studyGateSummary ? <p className="qc-ctmrg-reference"><strong>Study gate:</strong> {studyGateSummary.status === "passed" ? "bounded pass" : "needs review"}{finite(studyGateSummary.points) ? ` · ${String(studyGateSummary.passed_points ?? 0)}/${String(studyGateSummary.points)} points passed` : ""}</p> : null}
        {referenceName ? <p className="qc-ctmrg-reference"><strong>Reference:</strong> {referenceName}{referenceError != null ? ` · max error ${referenceError.toExponential(2)}` : ""}</p> : null}
        {Array.isArray(result.warnings) && result.warnings.length ? <p className="qc-diagnostics-limitations"><strong>Declared limits:</strong> {result.warnings.slice(0, 2).join(" · ")}</p> : null}
      </> : null}
      {error ? <p className="qc-ctmrg-error" role="alert">{error}</p> : null}
    </Card>
  );
}
