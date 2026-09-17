"use client";

import React from "react";
import { Button, Card, Metric, MetricGrid } from "../ui";
import type { PhysicsStudyMode } from "../lib/physicsStudy";

export type PhysicsStudyRow = {
  id: string;
  label: string;
  parameters: string;
  status: "queued" | "running" | "done" | "failed" | "canceled";
  result?: Record<string, unknown>;
  error?: string;
};

type Props = {
  mode: PhysicsStudyMode;
  rows: PhysicsStudyRow[];
  running: boolean;
  disabled?: boolean;
  availableModes?: PhysicsStudyMode[];
  onModeChange: (mode: PhysicsStudyMode) => void;
  onRun: () => void;
};

function numberValue(value: unknown): number | null {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function energyValue(result: Record<string, unknown> | undefined): number | null {
  if (!result) return null;
  const direct = numberValue(result.ground_energy ?? result.energy);
  if (direct != null) return direct;
  const energies = Array.isArray(result.energies) ? result.energies.map(numberValue).filter((value): value is number => value != null) : [];
  return energies.length ? energies[energies.length - 1] : null;
}

function discardedValue(result: Record<string, unknown> | undefined): number | null {
  return numberValue(result?.discarded_weight);
}

function normDrift(result: Record<string, unknown> | undefined): number | null {
  const norm = numberValue(result?.norm2);
  return norm == null ? null : Math.abs(norm - 1);
}

function modeLabel(mode: PhysicsStudyMode) {
  return mode === "dmrg" ? "DMRG" : mode === "tebd" ? "TEBD" : "PEPS";
}

export function PhysicsConvergenceStudy({ mode, rows, running, disabled = false, availableModes = ["dmrg", "tebd", "peps"], onModeChange, onRun }: Props) {
  const completed = rows.filter((row) => row.status === "done");
  const energies = completed.map((row) => energyValue(row.result)).filter((value): value is number => value != null);
  const discarded = completed.map((row) => discardedValue(row.result)).filter((value): value is number => value != null);
  const normDrifts = completed.map((row) => normDrift(row.result)).filter((value): value is number => value != null);
  const spread = energies.length > 1 ? Math.max(...energies) - Math.min(...energies) : null;
  const maxDiscarded = discarded.length ? Math.max(...discarded) : null;
  const maxNormDrift = normDrifts.length ? Math.max(...normDrifts) : null;
  const stable = spread != null && maxDiscarded != null && maxNormDrift != null && spread <= 1e-4 && maxDiscarded <= 1e-4 && maxNormDrift <= 1e-4;

  return <Card className="qc-study-card">
    <div className="qc-diagnostics-header">
      <div>
        <strong>{modeLabel(mode)} convergence study</strong>
        <p>Bounded serial comparison; every point keeps its own provenance and can be replayed from history.</p>
      </div>
      <div className="qc-study-actions">
        {(["dmrg", "tebd", "peps"] as PhysicsStudyMode[]).map((candidate) => <Button key={candidate} variant={candidate === mode ? "accent" : "secondary"} onClick={() => onModeChange(candidate)} disabled={running || disabled || !availableModes.includes(candidate)}>{modeLabel(candidate)}</Button>)}
        <Button variant="accent" onClick={onRun} disabled={disabled || running || !availableModes.includes(mode)}>{running ? "Studying…" : "Run bounded study"}</Button>
      </div>
    </div>
    {rows.length > 0 ? <>
      <MetricGrid>
        <Metric label="Completed" value={`${completed.length}/${rows.length}`} />
        <Metric label="Energy spread" value={spread == null ? "—" : spread.toExponential(2)} tone={spread != null && spread <= 1e-4 ? "success" : "warning"} />
        <Metric label="Max discarded" value={maxDiscarded == null ? "—" : maxDiscarded.toExponential(2)} tone={maxDiscarded != null && maxDiscarded <= 1e-4 ? "success" : "warning"} />
        <Metric label="Max ‖norm²−1‖" value={maxNormDrift == null ? "—" : maxNormDrift.toExponential(2)} tone={maxNormDrift != null && maxNormDrift <= 1e-4 ? "success" : "warning"} />
        <Metric label="Verdict" value={stable ? "Stable at tested points" : completed.length ? "Needs review" : "Not run"} tone={stable ? "success" : "warning"} />
      </MetricGrid>
      <div className="qc-study-table" role="table" aria-label={`${modeLabel(mode)} convergence points`}>
        <div className="qc-study-row qc-study-head" role="row"><span>Point</span><span>Energy</span><span>Norm²</span><span>Discarded</span><span>Status</span></div>
        {rows.map((row) => {
          const energy = energyValue(row.result);
          const norm = numberValue(row.result?.norm2);
          const lost = discardedValue(row.result);
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
