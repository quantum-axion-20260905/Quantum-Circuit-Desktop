"use client";

import React from "react";
import type { AgentResult } from "../lib/agent";
import { Card, Metric, MetricGrid } from "../ui";

function numericSeries(value: unknown) {
  return Array.isArray(value) ? value.map(Number).filter(Number.isFinite) : [];
}
function LineChart({ values, label, color = "#0f766e" }: { values: number[]; label: string; color?: string }) {
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(1e-12, max - min);
  const points = values.map((value, index) => `${18 + index * (504 / Math.max(1, values.length - 1))},${176 - ((value - min) / span) * 140}`).join(" ");
  return <div className="qc-diagnostics-chart"><svg viewBox="0 0 540 210" role="img" aria-label={label}><line x1="18" y1="176" x2="522" y2="176" stroke="#cbd5e1" /><polyline points={points} fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />{values.map((value, index) => <circle key={`${index}-${value}`} cx={18 + index * (504 / Math.max(1, values.length - 1))} cy={176 - ((value - min) / span) * 140} r="3.5" fill={color} />)}<text x="18" y="198" fill="#64748b" fontSize="11">0</text><text x="494" y="198" fill="#64748b" fontSize="11">{values.length - 1}</text><text x="18" y="20" fill="#475569" fontSize="12">{label}</text></svg></div>;
}

export function ConvergenceDiagnostics({ result }: { result: AgentResult }) {
  const dmrgHistory = Array.isArray(result.history) ? result.history.filter((entry): entry is Record<string, unknown> => typeof entry === "object" && entry !== null) : [];
  const dmrgEnergies = dmrgHistory.map((entry) => Number(entry.energy)).filter(Number.isFinite);
  const trajectory = numericSeries(result.energies);
  if (dmrgHistory.length === 0 && trajectory.length === 0) return null;

  if (dmrgHistory.length > 0) {
    const last = dmrgHistory[dmrgHistory.length - 1];
    const delta = Number(last.delta_energy);
    const discarded = Number(last.discarded_weight ?? result.discarded_weight);
    const norm = Number(last.norm2 ?? result.norm2);
    const variance = Number(last.energy_variance ?? result.energy_variance);
    const varianceTolerance = Number(result.variance_tolerance);
    return <Card className="qc-diagnostics-card"><div className="qc-diagnostics-header"><div><strong>DMRG convergence</strong><p>Per-sweep diagnostics for stopping and bond-dimension sensitivity.</p></div><span className={`qc-diagnostics-badge ${result.converged ? "qc-diagnostics-good" : "qc-diagnostics-warn"}`}>{result.converged ? "Converged" : "Needs more sweeps"}</span></div><MetricGrid><Metric label="Sweeps" value={`${dmrgHistory.length}/${result.sweeps_requested ?? dmrgHistory.length}`} /><Metric label="Final energy" value={Number(last.energy).toFixed(8)} /><Metric label="Δ energy" value={Number.isFinite(delta) ? delta.toExponential(2) : "—"} /><Metric label="Variance" value={Number.isFinite(variance) ? variance.toExponential(2) : "—"} tone={Number.isFinite(varianceTolerance) && variance > varianceTolerance ? "warning" : "success"} /><Metric label="Discarded weight" value={Number.isFinite(discarded) ? discarded.toExponential(2) : "—"} tone={discarded > 1e-8 ? "warning" : "success"} /><Metric label="Norm²" value={Number.isFinite(norm) ? norm.toFixed(8) : "—"} /><Metric label="Local residual" value={result.local_solver_residual != null ? Number(result.local_solver_residual).toExponential(2) : "—"} /></MetricGrid><LineChart values={dmrgEnergies} label="DMRG energy per sweep" /></Card>;
  }

  const initial = trajectory[0];
  const final = trajectory[trajectory.length - 1];
  const energyDrift = Math.abs(final - initial);
  const norm = Number(result.norm2);
  const discarded = Number(result.discarded_weight);
  return <Card className="qc-diagnostics-card"><div className="qc-diagnostics-header"><div><strong>{result.backend?.toString().includes("peps") ? "PEPS trajectory" : "TEBD convergence"}</strong><p>Energy trajectory and approximation diagnostics for the selected time step.</p></div><span className={`qc-diagnostics-badge ${energyDrift < 1e-4 && Math.abs(norm - 1) < 1e-4 ? "qc-diagnostics-good" : "qc-diagnostics-warn"}`}>{energyDrift < 1e-4 && Math.abs(norm - 1) < 1e-4 ? "Stable" : "Inspect convergence"}</span></div><MetricGrid><Metric label="Points" value={trajectory.length} /><Metric label="Energy drift" value={energyDrift.toExponential(2)} tone={energyDrift > 1e-3 ? "warning" : "success"} /><Metric label="Norm²" value={Number.isFinite(norm) ? norm.toFixed(8) : "—"} tone={Math.abs(norm - 1) > 1e-4 ? "warning" : "success"} /><Metric label="Bond dimension" value={`${result.bond_dim_used ?? "—"}/${result.bond_dim_requested ?? "—"}`} /><Metric label="Discarded weight" value={Number.isFinite(discarded) ? discarded.toExponential(2) : "—"} tone={discarded > 1e-8 ? "warning" : "success"} /><Metric label="dt" value={result.dt != null ? Number(result.dt).toString() : "—"} /></MetricGrid><LineChart values={trajectory} label="Energy over evolution points" color="#2563eb" /></Card>;
}
