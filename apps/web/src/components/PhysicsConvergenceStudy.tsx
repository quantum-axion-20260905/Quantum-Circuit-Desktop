"use client";

import React from "react";
import { Button, Card, Metric, MetricGrid } from "../ui";
import type { JsonObject } from "../lib/agent";
import { studyDiscardedWeight, studyEnergy, summarizePhysicsStudy, type PhysicsStudyMode, type PhysicsStudyRow } from "../lib/physicsStudy";

export type { PhysicsStudyRow } from "../lib/physicsStudy";

type Props = {
  mode: PhysicsStudyMode;
  rows: PhysicsStudyRow[];
  running: boolean;
  disabled?: boolean;
  availableModes?: PhysicsStudyMode[];
  onModeChange: (mode: PhysicsStudyMode) => void;
  onRun: () => void;
  manifest?: JsonObject | null;
};

function modeLabel(mode: PhysicsStudyMode) {
  return mode === "dmrg" ? "DMRG" : mode === "tebd" ? "TEBD" : "PEPS";
}

export function PhysicsConvergenceStudy({ mode, rows, running, disabled = false, availableModes = ["dmrg", "tebd", "peps"], onModeChange, onRun, manifest }: Props) {
  const summary = summarizePhysicsStudy(rows);
  const downloadManifest = () => {
    if (!manifest || typeof window === "undefined") return;
    const blob = new Blob([JSON.stringify(manifest, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `physics-${mode}-study.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return <Card className="qc-study-card">
    <div className="qc-diagnostics-header">
      <div>
        <strong>{modeLabel(mode)} convergence study</strong>
        <p>Bounded serial comparison; every point keeps its own provenance and can be replayed from history.</p>
      </div>
      <div className="qc-study-actions">
        {(["dmrg", "tebd", "peps"] as PhysicsStudyMode[]).map((candidate) => <Button key={candidate} variant={candidate === mode ? "accent" : "secondary"} onClick={() => onModeChange(candidate)} disabled={running || disabled || !availableModes.includes(candidate)}>{modeLabel(candidate)}</Button>)}
        <Button variant="accent" onClick={onRun} disabled={disabled || running || !availableModes.includes(mode)}>{running ? "Studying…" : "Run bounded study"}</Button>
        {manifest ? <Button variant="secondary" onClick={downloadManifest} disabled={running}>Export artifact</Button> : null}
      </div>
    </div>
    {rows.length > 0 ? <>
      <MetricGrid>
        <Metric label="Completed" value={`${summary.completed}/${rows.length}`} />
        <Metric label="Energy range" value={summary.energy_range == null ? "—" : summary.energy_range.toExponential(2)} tone={summary.energy_range != null && summary.energy_range <= 1e-4 ? "success" : "warning"} />
        <Metric label="Numerical uncertainty" value={summary.energy_half_range == null ? "—" : `±${summary.energy_half_range.toExponential(2)}`} tone={summary.energy_half_range != null && summary.energy_half_range <= 5e-5 ? "success" : "warning"} />
        <Metric label="Max discarded" value={summary.max_discarded_weight == null ? "—" : summary.max_discarded_weight.toExponential(2)} tone={summary.max_discarded_weight != null && summary.max_discarded_weight <= 1e-4 ? "success" : "warning"} />
        <Metric label="Max ‖norm²−1‖" value={summary.max_norm_drift == null ? "—" : summary.max_norm_drift.toExponential(2)} tone={summary.max_norm_drift != null && summary.max_norm_drift <= 1e-4 ? "success" : "warning"} />
        <Metric label="Verdict" value={summary.verdict === "stable" ? "Stable at tested points" : summary.verdict === "incomplete" ? "Incomplete / review" : summary.verdict === "needs_review" ? "Needs review" : "Not run"} tone={summary.verdict === "stable" ? "success" : "warning"} />
      </MetricGrid>
      <div className="qc-study-table" role="table" aria-label={`${modeLabel(mode)} convergence points`}>
        <div className="qc-study-row qc-study-head" role="row"><span>Point</span><span>Energy</span><span>Norm²</span><span>Discarded</span><span>Status</span></div>
        {rows.map((row) => {
          const energy = studyEnergy(row.result);
          const norm = row.result?.norm2 == null ? null : Number(row.result.norm2);
          const lost = studyDiscardedWeight(row.result);
          return <div className="qc-study-row" role="row" key={row.id}>
            <span><strong>{row.label}</strong><small>{row.parameters}</small></span>
            <span>{energy == null ? "—" : energy.toFixed(8)}</span>
            <span>{norm == null ? "—" : norm.toFixed(7)}</span>
            <span>{lost == null ? "—" : lost.toExponential(2)}</span>
            <span className={row.status === "done" ? "qc-study-ok" : row.status === "failed" ? "qc-study-fail" : "qc-study-pending"}>{row.status === "failed" ? row.error ?? "Failed" : row.status}</span>
          </div>;
        })}
      </div>
    </> : <p className="qc-study-empty">Run this study after generating a Hamiltonian. It uses at most three serial points to protect the local GPU/RAM budget.</p>}
  </Card>;
}
