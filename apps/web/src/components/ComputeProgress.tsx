"use client";

import React from "react";
import type { AsyncJob } from "../lib/agent";
import { Button, ProgressBar } from "../ui";

export function ComputeProgress({ job, dark = false, onCancel, canceling = false, onRetry, retrying = false }: { job: AsyncJob; dark?: boolean; onCancel?: () => void; canceling?: boolean; onRetry?: () => void; retrying?: boolean }) {
  const status = job.status === "queued" ? "Queued" : job.status === "running" ? "Running" : job.status;
  const percent = Math.round(Math.max(0, Math.min(1, job.progress)) * 100);
  const terminal = job.status === "failed" || job.status === "canceled";
  return (
    <section className={`qc-compute-progress${dark ? " qc-compute-progress-dark" : ""}`} aria-live="polite">
      <div className="qc-compute-progress-header"><div className="qc-compute-progress-state"><strong>{status}</strong><span>{percent}%</span></div><div style={{ display: "flex", gap: 8, alignItems: "center" }}>{onRetry && terminal ? <Button variant="secondary" size="sm" onClick={onRetry} disabled={retrying}>{retrying ? "Retrying…" : "Retry job"}</Button> : null}{onCancel && (job.status === "queued" || job.status === "running") ? <Button variant="danger" size="sm" onClick={onCancel} disabled={canceling}>{canceling ? "Canceling…" : "Cancel job"}</Button> : null}</div></div>
      <ProgressBar value={job.progress} label="Compute job progress" />
      <div className="qc-compute-progress-description">{job.error ? job.error : dark ? "GPU job progress from the local compute agent." : "The compute agent is reserving the selected backend and reporting live progress."}</div>
    </section>
  );
}
