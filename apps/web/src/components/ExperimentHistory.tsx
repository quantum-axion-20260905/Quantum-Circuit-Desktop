"use client";

import React from "react";
import { cancelAsyncJob, type AsyncJob } from "../lib/agent";
import { replayExperiment, type ExperimentRecord } from "../lib/runHistory";
import { useUIContext } from "../state/uiContext";
import { Button, Card } from "../ui";
import { ComputeProgress } from "./ComputeProgress";

type HistoryFilter = "all" | "done" | "failed" | "canceled";
type CompareKey = "backend" | "energy" | "norm2" | "bond_dim_used" | "discarded_weight" | "elapsed_ms";

const statusLabel: Record<ExperimentRecord["status"], string> = {
  done: "Done",
  failed: "Failed",
  canceled: "Canceled",
};

function metricValue(record: ExperimentRecord, key: CompareKey): string | number {
  if (!record.result) return "—";
  if (key === "backend") return String(record.result.provenance?.resolved_backend ?? record.result.backend ?? "—");
  if (key === "elapsed_ms") return Number(record.result.provenance?.elapsed_ms ?? record.result.time_ms ?? NaN);
  if (key === "energy") return Number(record.result.ground_energy ?? record.result.energy ?? NaN);
  return Number(record.result[key] ?? NaN);
}

function formatMetric(value: string | number) {
  if (typeof value === "string") return value;
  return Number.isFinite(value) ? value.toExponential(4) : "—";
}

export function ExperimentHistory() {
  const { experimentHistory, addExperiment, clearExperimentHistory, setLatestOutput } = useUIContext();
  const [filter, setFilter] = React.useState<HistoryFilter>("all");
  const [activeId, setActiveId] = React.useState<string | null>(null);
  const [job, setJob] = React.useState<AsyncJob | null>(null);
  const jobRef = React.useRef<AsyncJob | null>(null);
  const [canceling, setCanceling] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [exportNotice, setExportNotice] = React.useState<string | null>(null);
  const [compareA, setCompareA] = React.useState<string | null>(null);
  const [compareB, setCompareB] = React.useState<string | null>(null);

  const filtered = filter === "all" ? experimentHistory : experimentHistory.filter((record) => record.status === filter);
  const selectedA = experimentHistory.find((record) => record.id === compareA && record.status === "done" && record.result);
  const selectedB = experimentHistory.find((record) => record.id === compareB && record.status === "done" && record.result);

  function selectCompare(id: string, side: "a" | "b") {
    if (side === "a") setCompareA((current) => current === id ? null : id);
    else setCompareB((current) => current === id ? null : id);
  }

  async function replay(record: ExperimentRecord) {
    setActiveId(record.id);
    setJob(null);
    jobRef.current = null;
    setError(null);
    setCanceling(false);
    try {
      const result = await replayExperiment(record.request, (nextJob) => {
        jobRef.current = nextJob;
        setJob(nextJob);
      });
      setLatestOutput(result);
      addExperiment({ label: `Replay · ${record.label}`, source: record.source, kind: record.kind, status: "done", request: record.request, result, provenance: result.provenance });
    } catch (reason: unknown) {
      const replayJob = jobRef.current as AsyncJob | null;
      const canceled = replayJob?.status === "canceled";
      const message = canceled ? "Replay user tomonidan bekor qilindi." : reason instanceof Error ? reason.message : "Replay failed.";
      setError(message);
      addExperiment({ label: `Replay · ${record.label}`, source: record.source, kind: record.kind, status: canceled ? "canceled" : "failed", request: record.request, error: message });
    } finally {
      setActiveId(null);
      setCanceling(false);
    }
  }

  async function cancelReplay() {
    const currentJob = jobRef.current;
    if (!currentJob || (currentJob.status !== "queued" && currentJob.status !== "running")) return;
    setCanceling(true);
    try {
      await cancelAsyncJob(currentJob.job_id);
    } catch (reason: unknown) {
      setCanceling(false);
      setError(reason instanceof Error ? reason.message : "Replayni bekor qilishda xato yuz berdi.");
    }
  }

  function exportHistory() {
    const blob = new Blob([JSON.stringify(experimentHistory, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "quantum-circuit-experiment-history.json";
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
    setExportNotice(`JSON export ready · ${experimentHistory.length} record(s)`);
  }

  return <Card className="qc-history-card">
    <div className="qc-history-header">
      <div><strong>Experiment history</strong><p>Local reproducibility log for completed and failed compute jobs.</p></div>
      <div className="qc-history-actions"><select aria-label="History filter" value={filter} onChange={(event) => setFilter(event.target.value as HistoryFilter)}><option value="all">All runs</option><option value="done">Completed</option><option value="failed">Failed</option><option value="canceled">Canceled</option></select>{exportNotice ? <span role="status">{exportNotice}</span> : null}<Button variant="secondary" size="sm" onClick={exportHistory} disabled={experimentHistory.length === 0}>Export JSON</Button><Button variant="danger" size="sm" onClick={clearExperimentHistory} disabled={experimentHistory.length === 0}>Clear</Button></div>
    </div>
    {job ? <ComputeProgress job={job} onCancel={() => void cancelReplay()} canceling={canceling} /> : null}
    {error ? <div className="qc-history-error" role="alert">{error}</div> : null}
    {filtered.length === 0 ? <div className="qc-history-empty">No runs recorded yet. A successful or failed compute job will appear here.</div> : <div className="qc-history-list">{filtered.map((record) => <div className="qc-history-row" key={record.id}>
      <div className="qc-history-main"><strong>{record.label}</strong><span>{new Date(record.createdAt).toLocaleString()} · {record.source} · {record.kind}</span>{record.provenance?.resolved_backend ? <span>Backend: {record.provenance.resolved_backend}</span> : null}{record.error ? <span className="qc-history-row-error">{record.error}</span> : null}</div>
      <div className="qc-history-row-actions"><span className={`qc-history-status qc-history-status-${record.status}`}>{statusLabel[record.status]}</span>{record.status === "done" && record.result ? <><Button variant={compareA === record.id ? "primary" : "ghost"} size="sm" onClick={() => selectCompare(record.id, "a")}>A</Button><Button variant={compareB === record.id ? "primary" : "ghost"} size="sm" onClick={() => selectCompare(record.id, "b")}>B</Button></> : null}<Button variant="secondary" size="sm" onClick={() => void replay(record)} disabled={activeId !== null}>{activeId === record.id ? "Replaying…" : "Replay"}</Button></div>
    </div>)}</div>}
    {selectedA && selectedB ? <section className="qc-compare-panel"><div className="qc-compare-header"><div><strong>Run comparison</strong><p>A/B comparison uses the stored result artifacts; no recomputation is triggered.</p></div><Button variant="ghost" size="sm" onClick={() => { setCompareA(null); setCompareB(null); }}>Clear comparison</Button></div><div className="qc-compare-labels"><span>A · {selectedA.label}</span><span>B · {selectedB.label}</span></div><div className="qc-compare-table">{(["backend", "energy", "norm2", "bond_dim_used", "discarded_weight", "elapsed_ms"] as CompareKey[]).map((key) => { const a = metricValue(selectedA, key); const b = metricValue(selectedB, key); const delta = typeof a === "number" && typeof b === "number" && Number.isFinite(a) && Number.isFinite(b) ? b - a : null; return <div className="qc-compare-row" key={key}><span>{key}</span><code>{formatMetric(a)}</code><code>{formatMetric(b)}</code><span>{delta == null ? "—" : `Δ ${delta.toExponential(3)}`}</span></div>; })}</div></section> : null}
  </Card>;
}
