"use client";

import "katex/dist/katex.min.css";
import React, { useMemo } from "react";
import katex from "katex";
import { benchMatmul, cancelAsyncJob, getAsyncJob, getHardware, submitAsync, tnAmplitudes, tnEstimate } from "../lib/agent";
import { editorToIr, irToEditor } from "../ir/converters";
import { irToAgentTNPayload } from "../ir/agentMapping";
import { useCircuit, type CircuitAnalysis } from "../state/circuitStore";
import { IrPanel } from "./IrPanel";
import { AGENT_BASE_URL } from "../lib/agent";
import { API_BASE_URL } from "../lib/backend";
import { createArtifact, createRun, createVersion, ensureDefaultProject, getVersion, listArtifacts, listRuns, listVersions, patchRun, type Run, type RunArtifact } from "../lib/backend";
import { validateCircuitIrV1 } from "../ir/ir";
import { useUIContext } from "../state/uiContext";

function Latex({ tex }: { tex: string }) {
  const html = useMemo(
    () =>
      katex.renderToString(tex, {
        throwOnError: false,
        displayMode: true
      }),
    [tex]
  );
  return <div dangerouslySetInnerHTML={{ __html: html }} />;
}

type ExplainMode = "run" | "results" | "experiments" | "full";

export function ExplainPanel({ mode = "full" }: { mode?: ExplainMode }) {
  const { nQubits, ops, setNQubits, setOps, setAnalysis } = useCircuit();
  const { setLatestOutput, setSelectedRunSummary } = useUIContext();
  const [agentStatus, setAgentStatus] = React.useState<
    | { kind: "idle" }
    | { kind: "loading" }
    | { kind: "ready"; data: any }
    | { kind: "error"; message: string }
  >({ kind: "idle" });
  const [jobOut, setJobOut] = React.useState<any>(null);
  const [jobErr, setJobErr] = React.useState<string | null>(null);
  const [jobRunning, setJobRunning] = React.useState<boolean>(false);
  const [jobProgress, setJobProgress] = React.useState<number>(0);
  const [jobId, setJobId] = React.useState<string | null>(null);
  const [bitstringsText, setBitstringsText] = React.useState<string>("auto");
  const [tnOptimize, setTnOptimize] = React.useState<"auto" | "cotengra">("auto");
  const [shots, setShots] = React.useState<number>(1024);
  const [versionId, setVersionId] = React.useState<number | null>(null);
  const [runs, setRuns] = React.useState<Run[]>([]);
  const [backendMsg, setBackendMsg] = React.useState<string | null>(null);
  const [selectedRun, setSelectedRun] = React.useState<Run | null>(null);
  const [selectedArtifacts, setSelectedArtifacts] = React.useState<RunArtifact[]>([]);
  const [selectedAgentJob, setSelectedAgentJob] = React.useState<any>(null);
  const [liveSync, setLiveSync] = React.useState<boolean>(true);
  const [loadVersionId, setLoadVersionId] = React.useState<string>("");
  const [autoAttachResult, setAutoAttachResult] = React.useState<boolean>(true);
  const [compareRunA, setCompareRunA] = React.useState<number | null>(null);
  const [compareRunB, setCompareRunB] = React.useState<number | null>(null);
  const [connLastCheckedAt, setConnLastCheckedAt] = React.useState<string>("");
  const [autoMonitor, setAutoMonitor] = React.useState<boolean>(true);
  const [activeExample, setActiveExample] = React.useState<string>("");
  const [exampleCode, setExampleCode] = React.useState<{ qasm: string; qiskit: string; cirq: string } | null>(null);
  const [queueItems, setQueueItems] = React.useState<Array<{
    id: string;
    kind: "tn_estimate" | "tn_amplitudes" | "sample";
    label: string;
    req: any;
    status: "queued" | "running" | "done" | "failed" | "canceled";
    error?: string;
    validation?: "valid" | "warning" | "unknown";
    checks?: string[];
    sweepGroupId?: string;
    parentRunId?: number | null;
  }>>([]);
  const [queueRunning, setQueueRunning] = React.useState<boolean>(false);
  const [queueCancelRequested, setQueueCancelRequested] = React.useState<boolean>(false);
  const [sweepShots, setSweepShots] = React.useState<string>("256,1024,4096");
  const [sweepThetas, setSweepThetas] = React.useState<string>("0.2,0.6,1.0");
  const [connChecking, setConnChecking] = React.useState<boolean>(false);
  const [conn, setConn] = React.useState<{
    backend: "unknown" | "ok" | "down";
    agent: "unknown" | "ok" | "down";
    gpu: "unknown" | "ok" | "down";
    backendMsg: string;
    agentMsg: string;
    gpuMsg: string;
  }>({
    backend: "unknown",
    agent: "unknown",
    gpu: "unknown",
    backendMsg: "not checked",
    agentMsg: "not checked",
    gpuMsg: "not checked"
  });

  const sectionStyle: React.CSSProperties = {
    border: "1px solid #d1d5db",
    borderRadius: 6,
    padding: 10,
    background: "#ffffff"
  };

  function buildAnalysis(kind: string, out: any): CircuitAnalysis {
    const gateScores: Record<string, number> = {};
    const normalize = (x: number, max: number) => (max > 0 ? Math.max(0.1, Math.min(1, x / max)) : 0.1);
    const opIds = ops.map((o) => o.id);

    if (kind === "sample" && out?.counts && typeof out.counts === "object") {
      const counts = out.counts as Record<string, number>;
      const total = Object.values(counts).reduce((s, v) => s + Number(v || 0), 0);
      const probs = Object.values(counts).map((v) => Number(v || 0) / Math.max(1, total));
      const spread = probs.length ? probs.reduce((s, p) => s + p * p, 0) : 1;
      const score = normalize(1 - spread, 1);
      for (const id of opIds) gateScores[id] = score;
      return { source: "sample", attachedAt: new Date().toISOString(), gateScores, counts };
    }

    if (kind === "tn_amplitudes" && Array.isArray(out?.amplitudes)) {
      const raw = out.amplitudes.map((a: any) => {
        const re = Number(a?.re ?? 0);
        const im = Number(a?.im ?? 0);
        return { bitstring: String(a?.bitstring ?? ""), re, im, mag: Math.sqrt(re * re + im * im) };
      });
      const maxMag = raw.reduce((m: number, a: any) => Math.max(m, a.mag), 0);
      const score = normalize(maxMag, 1);
      for (const id of opIds) gateScores[id] = score;
      return { source: "tn_amplitudes", attachedAt: new Date().toISOString(), gateScores, amplitudes: raw };
    }

    if (kind === "tn_estimate") {
      const pathSteps = Number(out?.path_steps ?? 0);
      const score = normalize(1 / Math.max(1, pathSteps), 1);
      for (const id of opIds) gateScores[id] = score;
      return {
        source: "tn_estimate",
        attachedAt: new Date().toISOString(),
        gateScores,
        complexity: {
          pathSteps: out?.path_steps,
          optCost: out?.opt_cost,
          largestIntermediate: out?.largest_intermediate,
          speedup: out?.speedup
        }
      };
    }

    for (const id of opIds) gateScores[id] = 0.25;
    return { source: "unknown", attachedAt: new Date().toISOString(), gateScores };
  }

  function attachAnalysis(kind: string, out: any) {
    if (!autoAttachResult) return;
    setAnalysis(buildAnalysis(kind, out));
  }

  async function checkConnections() {
    setConnChecking(true);
    const next = {
      backend: "down" as "unknown" | "ok" | "down",
      agent: "down" as "unknown" | "ok" | "down",
      gpu: "down" as "unknown" | "ok" | "down",
      backendMsg: "unreachable",
      agentMsg: "unreachable",
      gpuMsg: "unreachable"
    };
    try {
      const backendRes = await fetch(`${API_BASE_URL}/api/projects/`);
      if (backendRes.ok) {
        next.backend = "ok";
        next.backendMsg = "reachable";
      } else {
        next.backendMsg = `http ${backendRes.status}`;
      }
    } catch (e: any) {
      next.backendMsg = e?.message ?? "network error";
    }

    try {
      const hw = await getHardware();
      next.agent = "ok";
      next.agentMsg = "reachable";
      if (hw?.gpu?.available) {
        next.gpu = "ok";
        next.gpuMsg = `${hw.gpu.device0?.name ?? "GPU"} | cc ${hw.gpu.device0?.compute_capability ?? "?"}`;
      } else {
        next.gpu = "down";
        next.gpuMsg = hw?.gpu?.reason ?? "gpu unavailable";
      }
    } catch (e: any) {
      next.agent = "down";
      next.agentMsg = e?.message ?? "agent error";
      next.gpu = "down";
      next.gpuMsg = "agent unavailable";
    }

    setConn(next);
    setConnLastCheckedAt(new Date().toISOString());
    setConnChecking(false);
  }

  React.useEffect(() => {
    checkConnections();
  }, []);

  React.useEffect(() => {
    if (!autoMonitor) return;
    const timer = setInterval(() => {
      checkConnections();
    }, 10000);
    return () => clearInterval(timer);
  }, [autoMonitor]);

  function StatusChip({ label, state, detail }: { label: string; state: "unknown" | "ok" | "down"; detail: string }) {
    const bg = state === "ok" ? "#ecfdf5" : state === "down" ? "#fef2f2" : "#f9fafb";
    const bd = state === "ok" ? "#10b981" : state === "down" ? "#ef4444" : "#d1d5db";
    const fg = state === "ok" ? "#065f46" : state === "down" ? "#991b1b" : "#374151";
    return (
      <div style={{ border: `1px solid ${bd}`, background: bg, color: fg, borderRadius: 6, padding: "6px 8px", minWidth: 150 }}>
        <div style={{ fontWeight: 600, fontSize: 12 }}>{label}</div>
        <div style={{ fontSize: 11 }}>{detail}</div>
      </div>
    );
  }

  function applyExample(name: "bell" | "ghz3" | "qft2") {
    if (name === "bell") {
      const exOps = [
        { id: "ex_h0", name: "h", target: 0, col: 0, x: 120, y: 60 },
        { id: "ex_cx0", name: "cx", target: 1, control: 0, col: 1, x: 260, y: 120 }
      ] as any;
      setNQubits(2);
      setOps(exOps);
      setActiveExample("Bell state");
      setExampleCode({
        qasm: `OPENQASM 3;\ninclude "stdgates.inc";\nqubit[2] q;\nh q[0];\ncx q[0], q[1];`,
        qiskit: `from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0,1)\nprint(qc)`,
        cirq: `import cirq\nq0,q1 = cirq.LineQubit.range(2)\nc = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0,q1))\nprint(c)`
      });
      return;
    }
    if (name === "ghz3") {
      const exOps = [
        { id: "ex_h0", name: "h", target: 0, col: 0, x: 120, y: 60 },
        { id: "ex_cx01", name: "cx", target: 1, control: 0, col: 1, x: 260, y: 120 },
        { id: "ex_cx12", name: "cx", target: 2, control: 1, col: 2, x: 400, y: 180 }
      ] as any;
      setNQubits(3);
      setOps(exOps);
      setActiveExample("GHZ(3)");
      setExampleCode({
        qasm: `OPENQASM 3;\ninclude "stdgates.inc";\nqubit[3] q;\nh q[0];\ncx q[0], q[1];\ncx q[1], q[2];`,
        qiskit: `from qiskit import QuantumCircuit\nqc = QuantumCircuit(3)\nqc.h(0)\nqc.cx(0,1)\nqc.cx(1,2)\nprint(qc)`,
        cirq: `import cirq\nq = cirq.LineQubit.range(3)\nc = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0],q[1]), cirq.CNOT(q[1],q[2]))\nprint(c)`
      });
      return;
    }
    const exOps = [
      { id: "ex_h0", name: "h", target: 0, col: 0, x: 120, y: 60 },
      { id: "ex_h1", name: "h", target: 1, col: 0, x: 120, y: 120 },
      { id: "ex_cz", name: "cz", target: 1, control: 0, col: 1, x: 260, y: 120 },
      { id: "ex_h0b", name: "h", target: 0, col: 2, x: 400, y: 60 },
      { id: "ex_h1b", name: "h", target: 1, col: 2, x: 400, y: 120 }
    ] as any;
    setNQubits(2);
    setOps(exOps);
    setActiveExample("QFT-like (2q)");
    setExampleCode({
      qasm: `OPENQASM 3;\ninclude "stdgates.inc";\nqubit[2] q;\nh q[0];\nh q[1];\ncz q[0], q[1];\nh q[0];\nh q[1];`,
      qiskit: `from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.h(0); qc.h(1)\nqc.cz(0,1)\nqc.h(0); qc.h(1)\nprint(qc)`,
      cirq: `import cirq\nq0,q1 = cirq.LineQubit.range(2)\nc = cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.CZ(q0,q1), cirq.H(q0), cirq.H(q1))\nprint(c)`
    });
  }

  function forceAttachLatestResult() {
    const out = jobOut ?? selectedRun?.result;
    if (!out) return;
    const kind = selectedRun?.kind ?? (Array.isArray(out?.amplitudes) ? "tn_amplitudes" : out?.counts ? "sample" : "unknown");
    setAnalysis(buildAnalysis(kind, out));
  }

  async function refreshHardware() {
    setAgentStatus({ kind: "loading" });
    try {
      const data = await getHardware();
      setAgentStatus({ kind: "ready", data });
    } catch (e: any) {
      setAgentStatus({ kind: "error", message: e?.message ?? String(e) });
    }
  }

  async function runBench() {
    setJobErr(null);
    setJobOut(null);
    try {
      const out = await benchMatmul({ size: 1024, iters: 10, dtype: "fp16" });
      setJobOut(out);
      attachAnalysis("unknown", out);
    } catch (e: any) {
      setJobErr(e?.message ?? String(e));
    }
  }

  async function runBellTN() {
    setJobErr(null);
    setJobOut(null);
    try {
      const out = await tnAmplitudes({
        n_qubits: 2,
        gates: [
          { name: "h", target: 0 },
          { name: "cx", control: 0, target: 1 }
        ],
        bitstrings: ["00", "01", "10", "11"],
        optimize: "auto"
      });
      setJobOut(out);
      attachAnalysis("tn_amplitudes", out);
    } catch (e: any) {
      setJobErr(e?.message ?? String(e));
    }
  }

  async function runEstimate() {
    setJobErr(null);
    setJobOut(null);
    try {
      const gates: Array<{ name: string; target: number; control?: number }> = [{ name: "h", target: 0 }];
      for (let i = 0; i < 23; i++) gates.push({ name: "cx", control: i, target: i + 1 });
      const out = await tnEstimate({
        n_qubits: 24,
        gates,
        optimize: "auto"
      });
      setJobOut(out);
      attachAnalysis("tn_estimate", out);
    } catch (e: any) {
      setJobErr(e?.message ?? String(e));
    }
  }

  function parseBitstrings(): string[] {
    const v = bitstringsText.trim();
    if (!v || v === "auto") return ["0".repeat(nQubits), "1".repeat(nQubits)];
    return v
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  }

  async function saveVersionToBackend() {
    setBackendMsg(null);
    try {
      const project = await ensureDefaultProject();
      const ir = editorToIr(nQubits, ops);
      const v = await createVersion(project.id, ir.qasm, ir.ui);
      setVersionId(v.id);
      setBackendMsg(`Saved version #${v.id} to backend.`);
      const rs = await listRuns(v.id);
      setRuns(rs);
    } catch (e: any) {
      setBackendMsg(e?.message ?? String(e));
    }
  }

  async function loadVersionFromBackend(versionIdToLoad: number) {
    setBackendMsg(null);
    try {
      const v = await getVersion(versionIdToLoad);
      const ir = validateCircuitIrV1({ qasm: v.qasm, ui: v.metadata });
      const ed = irToEditor(ir);
      setNQubits(ed.nQubits);
      setOps(ed.ops);
      setVersionId(v.id);
      setBackendMsg(`Loaded version #${v.id} from backend.`);
      const rs = await listRuns(v.id);
      setRuns(rs);
      setSelectedRun(null);
    } catch (e: any) {
      setBackendMsg(e?.message ?? String(e));
    }
  }

  async function loadLatestVersion() {
    setBackendMsg(null);
    try {
      const project = await ensureDefaultProject();
      const versions = await listVersions(project.id);
      if (!versions.length) {
        setBackendMsg("No versions found for Default Project.");
        return;
      }
      // DRF returns newest first due to queryset ordering; but be safe:
      const latest = [...versions].sort((a, b) => (a.created_at < b.created_at ? 1 : -1))[0];
      await loadVersionFromBackend(latest.id);
    } catch (e: any) {
      setBackendMsg(e?.message ?? String(e));
    }
  }

  async function refreshRuns() {
    if (!versionId) return;
    const rs = await listRuns(versionId);
    setRuns(rs);
  }

  React.useEffect(() => {
    const agentJobId = selectedRun?.agent_job_id;
    if (!agentJobId) return;
    if (!liveSync) return;
    let stopped = false;

    async function tick() {
      try {
        const j = await getAsyncJob(agentJobId as string);
        if (stopped) return;
        setSelectedAgentJob(j);
      } catch (e: any) {
        if (stopped) return;
        setSelectedAgentJob({ error: e?.message ?? String(e) });
      }
    }

    tick();
    const iv = setInterval(tick, 500);
    return () => {
      stopped = true;
      clearInterval(iv);
    };
  }, [selectedRun?.agent_job_id, liveSync]);

  React.useEffect(() => {
    // If we have an agent job loaded and the backend run is not finalized, finalize it.
    if (!selectedRun) return;
    const run = selectedRun;
    const j = selectedAgentJob;
    if (!j || typeof j !== "object") return;
    const status = j.status as string | undefined;
    if (!status) return;
    if (run.status === "done" || run.status === "failed" || run.status === "canceled") return;

    async function sync() {
      try {
        if (status === "done") {
          const out = j.artifacts?.result ?? j;
          const finishedAt = j.finished_at ?? new Date().toISOString();
          const updated = await patchRun(run.id, {
            status: "done",
            finished_at: finishedAt,
            result: out
          } as any);
          setSelectedRun(updated);
          await createArtifact({
            run: updated.id,
            kind:
              updated.kind === "sample"
                ? "counts"
                : updated.kind === "tn_amplitudes"
                  ? "amplitudes"
                  : updated.kind === "tn_estimate"
                    ? "estimate"
                    : "raw",
            content: out
          });
          await refreshRuns();
        } else if (status === "failed" || status === "canceled") {
          const finishedAt = j.finished_at ?? new Date().toISOString();
          const updated = await patchRun(run.id, {
            status: status === "canceled" ? "canceled" : "failed",
            finished_at: finishedAt,
            error: j.error ?? ""
          } as any);
          setSelectedRun(updated);
          await refreshRuns();
        }
      } catch {
        // ignore sync errors; UI can retry on next poll
      }
    }

    sync();
  }, [selectedAgentJob, selectedRun]);

  function extractPhaseFromJob(job: any): string | null {
    const logs: any[] = Array.isArray(job?.logs) ? job.logs : [];
    for (let i = logs.length - 1; i >= 0; i--) {
      const e = logs[i];
      if (e?.event === "progress" && typeof e?.phase === "string") return e.phase;
      if (e?.event === "running" && typeof e?.phase === "string") return e.phase;
    }
    return null;
  }

  async function loadAgentJobOnce(run: Run) {
    if (!run.agent_job_id) return;
    const j = await getAsyncJob(run.agent_job_id);
    setSelectedAgentJob(j);
    return j;
  }

  function makeSeed() {
    // deterministic enough for logging; not crypto.
    return Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
  }

  async function runAndPersist(kind: "tn_estimate" | "tn_amplitudes" | "sample", req: any) {
    setJobRunning(true);
    setJobProgress(0);
    setJobId(null);
    const seed = makeSeed();
    const startedAt = new Date().toISOString();
    const device = await getHardware().catch(() => null);
    const reproducibility = {
      seed,
      app: "qc-web",
      captured_at: startedAt,
      gpu: device?.gpu ?? null
    };
    const budget = { max_qubits: 64, max_shots: 200000, max_mem_mb: 12000 };

    let runId: number | null = null;
    if (versionId) {
      const run = await createRun({
        kind,
        version: versionId,
        status: "running",
        seed,
        started_at: startedAt,
        finished_at: null,
        error: "",
        agent_base_url: AGENT_BASE_URL,
        agent_job_id: "",
        backend: kind,
        device,
        request: { ...req, seed, reproducibility },
        result: {}
      });
      runId = run.id;
      await refreshRuns();
    }

    try {
      // Run via async agent so we can show progress + allow cancel.
      const job = await submitAsync(kind, { ...req, seed, budget, reproducibility });
      setJobId(job.job_id);
      if (runId) {
        await patchRun(runId, { agent_job_id: job.job_id } as any);
      }
      let out: any = null;
      for (let i = 0; i < 600; i++) {
        const cur = await getAsyncJob(job.job_id);
        setJobProgress(Number(cur.progress ?? 0));
        if (cur.status === "done") {
          out = cur.artifacts?.result ?? cur;
          break;
        }
        if (cur.status === "failed") {
          throw new Error(cur.error ?? "Agent job failed");
        }
        if (cur.status === "canceled") {
          throw new Error("Canceled");
        }
        await new Promise((r) => setTimeout(r, 250));
      }
      if (!out) throw new Error("Timed out waiting for agent job");
      const finishedAt = new Date().toISOString();
      setJobOut(out);
      attachAnalysis(kind, out);
      if (runId) {
        const updated = await patchRun(runId, {
          status: "done",
          finished_at: finishedAt,
          result: out
        } as any);
        setSelectedRun((cur) => (cur && cur.id === runId ? updated : cur));
        // persist normalized artifact for indexing
        const artifactKind =
          updated.kind === "sample"
            ? "counts"
            : updated.kind === "tn_amplitudes"
              ? "amplitudes"
              : updated.kind === "tn_estimate"
                ? "estimate"
                : "raw";
        await createArtifact({ run: updated.id, kind: artifactKind, content: out });
        await refreshRuns();
      }
    } catch (e: any) {
      const finishedAt = new Date().toISOString();
      const msg = e?.message ?? String(e);
      setJobErr(msg);
      if (runId) {
        const updated = await patchRun(runId, {
          status: "failed",
          finished_at: finishedAt,
          error: msg
        } as any);
        setSelectedRun((cur) => (cur && cur.id === runId ? updated : cur));
        await refreshRuns();
      }
    } finally {
      setJobRunning(false);
    }
  }

  async function runAndPersistWithResult(kind: "tn_estimate" | "tn_amplitudes" | "sample", req: any): Promise<{ ok: boolean; out?: any; error?: string }> {
    try {
      await runAndPersist(kind, req);
      return { ok: true, out: jobOut ?? null };
    } catch (e: any) {
      return { ok: false, error: e?.message ?? String(e) };
    }
  }

  function addQueueItem(kind: "tn_estimate" | "tn_amplitudes" | "sample", label: string, req: any, sweepGroupId?: string) {
    setQueueItems((prev) => [
      ...prev,
      { id: `q_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`, kind, label, req, status: "queued", validation: "unknown", checks: [], sweepGroupId, parentRunId: selectedRun?.id ?? null }
    ]);
  }

  function enqueueCurrent(kind: "tn_estimate" | "tn_amplitudes" | "sample") {
    const ir = editorToIr(nQubits, ops);
    const base = irToAgentTNPayload(ir);
    if (kind === "sample") addQueueItem(kind, `sample shots=${shots}`, { ...base, shots });
    else if (kind === "tn_amplitudes") addQueueItem(kind, `tn amplitudes`, { ...base, bitstrings: parseBitstrings(), optimize: tnOptimize });
    else addQueueItem(kind, `tn estimate`, { ...base, optimize: tnOptimize });
  }

  function enqueueSweepShots() {
    const parsed = sweepShots.split(",").map((s) => Number(s.trim())).filter((n) => Number.isFinite(n) && n > 0);
    const ir = editorToIr(nQubits, ops);
    const base = irToAgentTNPayload(ir);
    const sweepId = `shots_${Date.now()}`;
    for (const s of parsed) addQueueItem("sample", `sample shots=${s}`, { ...base, shots: s }, sweepId);
  }

  function enqueueSweepTheta() {
    const parsed = sweepThetas.split(",").map((s) => Number(s.trim())).filter((n) => Number.isFinite(n));
    const sweepId = `theta_${Date.now()}`;
    for (const theta of parsed) {
      const nextOps = ops.map((op) => (op.name === "rx" || op.name === "ry" || op.name === "rz" ? { ...op, theta } : op));
      const ir = editorToIr(nQubits, nextOps as any);
      const base = irToAgentTNPayload(ir);
      addQueueItem("tn_estimate", `estimate theta=${theta.toFixed(3)}`, { ...base, optimize: tnOptimize }, sweepId);
    }
  }

  function validateOut(out: any): { status: "valid" | "warning" | "unknown"; checks: string[] } {
    if (!out || typeof out !== "object") return { status: "unknown", checks: [] };
    const checks: string[] = [];
    let ok = true;
    if (out.counts && typeof out.counts === "object") {
      const total = Object.values(out.counts).reduce((s: number, v: any) => s + Number(v || 0), 0);
      checks.push(`counts_sum=${total}`);
      if (total <= 0) ok = false;
    }
    if (Array.isArray(out.amplitudes)) {
      let norm2 = 0;
      for (const a of out.amplitudes) {
        const re = Number(a?.re ?? 0);
        const im = Number(a?.im ?? 0);
        norm2 += re * re + im * im;
      }
      checks.push(`amp_norm2=${norm2.toFixed(6)}`);
      if (!Number.isFinite(norm2)) ok = false;
    }
    return { status: ok ? "valid" : "warning", checks };
  }

  async function runQueue() {
    if (queueRunning) return;
    setQueueRunning(true);
    setQueueCancelRequested(false);
    for (const item of queueItems) {
      if (queueCancelRequested) {
        setQueueItems((prev) => prev.map((x) => (x.id === item.id && x.status === "queued" ? { ...x, status: "canceled" } : x)));
        continue;
      }
      if (item.status !== "queued") continue;
      setQueueItems((prev) => prev.map((x) => (x.id === item.id ? { ...x, status: "running" } : x)));
      const res = await runAndPersistWithResult(item.kind, item.req);
      if (res.ok) {
        const v = validateOut(jobOut ?? selectedRun?.result ?? null);
        setQueueItems((prev) => prev.map((x) => (x.id === item.id ? { ...x, status: "done", validation: v.status, checks: v.checks } : x)));
      } else {
        setQueueItems((prev) => prev.map((x) => (x.id === item.id ? { ...x, status: "failed", error: res.error } : x)));
      }
    }
    setQueueRunning(false);
  }

  async function runSample() {
    setJobErr(null);
    setJobOut(null);
    try {
      const ir = editorToIr(nQubits, ops);
      const base = irToAgentTNPayload(ir);
      const req = { ...base, shots };
      // Persist as a run too if we have a saved version
      await runAndPersist("sample", req);
    } catch (e: any) {
      setJobErr(e?.message ?? String(e));
    }
  }

  async function runEditorEstimate() {
    setJobErr(null);
    setJobOut(null);
    try {
      const ir = editorToIr(nQubits, ops);
      const base = irToAgentTNPayload(ir);
      const req = { ...base, optimize: tnOptimize };
      await runAndPersist("tn_estimate", req);
    } catch (e: any) {
      setJobErr(e?.message ?? String(e));
    }
  }

  async function runEditorAmps() {
    setJobErr(null);
    setJobOut(null);
    try {
      const ir = editorToIr(nQubits, ops);
      const base = irToAgentTNPayload(ir);
      const req = { ...base, bitstrings: parseBitstrings(), optimize: tnOptimize };
      await runAndPersist("tn_amplitudes", req);
    } catch (e: any) {
      setJobErr(e?.message ?? String(e));
    }
  }

  function exportBundle() {
    const payload = {
      exported_at: new Date().toISOString(),
      circuit: { nQubits, ops },
      latest_output: jobOut ?? null,
      selected_run: selectedRun ?? null,
      validation,
      compare: compareSummary
      ,
      lineage: {
        parent_run_id: selectedRun?.id ?? null,
        queue_items: queueItems.map((x) => ({
          id: x.id,
          kind: x.kind,
          label: x.label,
          status: x.status,
          validation: x.validation ?? "unknown",
          sweep_group_id: x.sweepGroupId ?? null,
          parent_run_id: x.parentRunId ?? null
        }))
      }
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `qc_bundle_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  const compareSummary = React.useMemo(() => {
    if (compareRunA == null || compareRunB == null) return null;
    const a = runs.find((r) => r.id === compareRunA);
    const b = runs.find((r) => r.id === compareRunB);
    if (!a || !b) return null;
    const aDur = a.started_at && a.finished_at ? (new Date(a.finished_at).getTime() - new Date(a.started_at).getTime()) : null;
    const bDur = b.started_at && b.finished_at ? (new Date(b.finished_at).getTime() - new Date(b.started_at).getTime()) : null;
    const aCounts = a.result?.counts && typeof a.result.counts === "object" ? a.result.counts as Record<string, number> : null;
    const bCounts = b.result?.counts && typeof b.result.counts === "object" ? b.result.counts as Record<string, number> : null;
    const countsDistance = (() => {
      if (!aCounts || !bCounts) return null;
      const keys = new Set([...Object.keys(aCounts), ...Object.keys(bCounts)]);
      const sumA = Object.values(aCounts).reduce((s, v) => s + Number(v || 0), 0);
      const sumB = Object.values(bCounts).reduce((s, v) => s + Number(v || 0), 0);
      let tv = 0;
      for (const k of keys) {
        const pa = Number(aCounts[k] || 0) / Math.max(1, sumA);
        const pb = Number(bCounts[k] || 0) / Math.max(1, sumB);
        tv += Math.abs(pa - pb);
      }
      return tv * 0.5;
    })();

    const aAmps = Array.isArray(a.result?.amplitudes) ? a.result.amplitudes : null;
    const bAmps = Array.isArray(b.result?.amplitudes) ? b.result.amplitudes : null;
    const amplitudeDelta = (() => {
      if (!aAmps || !bAmps) return null;
      const mapA = new Map<string, { re: number; im: number }>();
      for (const x of aAmps) mapA.set(String(x.bitstring ?? ""), { re: Number(x.re ?? 0), im: Number(x.im ?? 0) });
      let l2 = 0;
      let maxAbs = 0;
      for (const y of bAmps) {
        const key = String(y.bitstring ?? "");
        const ay = mapA.get(key) ?? { re: 0, im: 0 };
        const dr = ay.re - Number(y.re ?? 0);
        const di = ay.im - Number(y.im ?? 0);
        const abs = Math.sqrt(dr * dr + di * di);
        l2 += abs * abs;
        if (abs > maxAbs) maxAbs = abs;
      }
      return { l2: Math.sqrt(l2), maxAbs };
    })();

    const estimateDelta = (() => {
      const toNum = (v: any) => (v == null ? null : Number(v));
      const aPath = toNum(a.result?.path_steps);
      const bPath = toNum(b.result?.path_steps);
      const aCost = toNum(a.result?.opt_cost);
      const bCost = toNum(b.result?.opt_cost);
      return {
        pathSteps: aPath != null && bPath != null ? bPath - aPath : null,
        optCost: aCost != null && bCost != null ? bCost - aCost : null
      };
    })();

    return {
      a: { id: a.id, kind: a.kind, status: a.status, durationMs: aDur },
      b: { id: b.id, kind: b.kind, status: b.status, durationMs: bDur },
      deltaDurationMs: aDur != null && bDur != null ? bDur - aDur : null,
      countsDistanceTV: countsDistance,
      amplitudeDelta,
      estimateDelta
    };
  }, [compareRunA, compareRunB, runs]);

  React.useEffect(() => {
    setLatestOutput(jobOut ?? selectedRun?.result ?? null);
  }, [jobOut, selectedRun?.result, setLatestOutput]);

  React.useEffect(() => {
    setSelectedRunSummary(compareSummary ?? null);
  }, [compareSummary, setSelectedRunSummary]);

  const validation = React.useMemo(() => {
    const out = jobOut ?? selectedRun?.result;
    if (!out || typeof out !== "object") return { status: "unknown", checks: [] as string[] };
    const checks: string[] = [];
    let ok = true;
    if (out.counts && typeof out.counts === "object") {
      const total = Object.values(out.counts).reduce((s: number, v: any) => s + Number(v || 0), 0);
      if (total <= 0) {
        ok = false;
        checks.push("counts sum <= 0");
      } else {
        checks.push(`counts sum = ${total}`);
      }
    }
    if (Array.isArray(out.amplitudes)) {
      let norm2 = 0;
      for (const a of out.amplitudes) {
        const re = Number(a?.re ?? 0);
        const im = Number(a?.im ?? 0);
        norm2 += re * re + im * im;
      }
      checks.push(`amplitude partial norm^2 = ${norm2.toFixed(6)}`);
      if (!Number.isFinite(norm2)) {
        ok = false;
        checks.push("amplitude norm is not finite");
      }
    }
    if (out.path_steps != null && Number(out.path_steps) < 0) {
      ok = false;
      checks.push("path_steps < 0");
    }
    return { status: ok ? "valid" : "warning", checks };
  }, [jobOut, selectedRun?.result]);

  const resultCharts = React.useMemo(() => {
    const out = jobOut ?? selectedRun?.result;
    if (!out || typeof out !== "object") return null;

    let counts: Array<{ key: string; value: number }> = [];
    if (out.counts && typeof out.counts === "object") {
      counts = Object.entries(out.counts)
        .map(([k, v]) => ({ key: k, value: Number(v || 0) }))
        .sort((a, b) => b.value - a.value)
        .slice(0, 16);
    }

    let amps: Array<{ key: string; mag: number }> = [];
    if (Array.isArray(out.amplitudes)) {
      amps = out.amplitudes
        .map((a: any) => {
          const re = Number(a?.re ?? 0);
          const im = Number(a?.im ?? 0);
          return { key: String(a?.bitstring ?? ""), mag: Math.sqrt(re * re + im * im) };
        })
        .sort((a: { key: string; mag: number }, b: { key: string; mag: number }) => b.mag - a.mag)
        .slice(0, 16);
    }

    return { counts, amps };
  }, [jobOut, selectedRun?.result]);

  const showRun = mode === "run" || mode === "full";
  const showResults = mode === "results" || mode === "full";
  const showExperiments = mode === "experiments" || mode === "full";

  return (
    <div style={{ padding: 16, overflow: "auto" }}>
      <h2 style={{ marginTop: 0, marginBottom: 10 }}>Compute Console</h2>
      <div style={{ display: "grid", gap: 10 }}>
        <section style={sectionStyle}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Connection Status</div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
            <StatusChip label="Web UI" state="ok" detail="running in browser" />
            <StatusChip label="Backend DRF" state={conn.backend} detail={`${API_BASE_URL} | ${conn.backendMsg}`} />
            <StatusChip label="Compute Agent" state={conn.agent} detail={`${AGENT_BASE_URL} | ${conn.agentMsg}`} />
            <StatusChip label="GPU / CUDA" state={conn.gpu} detail={conn.gpuMsg} />
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 12 }}>
            <button onClick={checkConnections} disabled={connChecking}>{connChecking ? "Checking..." : "Check All"}</button>
            <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
              auto monitor (10s)
              <input type="checkbox" checked={autoMonitor} onChange={(e) => setAutoMonitor(e.target.checked)} />
            </label>
            <span>last check: {connLastCheckedAt ? new Date(connLastCheckedAt).toLocaleTimeString() : "-"}</span>
            <span>pipeline: UI {"->"} DRF {"->"} Agent {"->"} GPU</span>
          </div>
        </section>

        {showRun ? (
        <section style={sectionStyle}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Quick Simulations (immediate)</div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
            <button onClick={refreshHardware}>Refresh hardware</button>
            <button onClick={runBench}>Start GPU bench</button>
            <button onClick={runBellTN}>Start Bell amplitudes</button>
            <button onClick={runEstimate}>Start 24q estimate</button>
          </div>
          <div style={{ fontSize: 12 }}>{agentStatus.kind === "idle" && "status: unknown"}{agentStatus.kind === "loading" && "status: checking"}{agentStatus.kind === "error" && `status: error | ${agentStatus.message}`}{agentStatus.kind === "ready" && `status: ready | ${agentStatus.data?.gpu?.device0?.name ?? "GPU"}`}</div>
        </section>
        ) : null}

        {showRun ? (
        <section style={sectionStyle}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Run Controls</div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            <label style={{ display: "flex", gap: 6, alignItems: "center" }}>Optimize<select value={tnOptimize} onChange={(e) => setTnOptimize(e.target.value as any)}><option value="auto">auto</option><option value="cotengra">cotengra</option></select></label>
            <label style={{ display: "flex", gap: 6, alignItems: "center" }}>shots<input type="number" min={1} max={200000} value={shots} onChange={(e) => setShots(Number(e.target.value) || 1)} style={{ width: 90 }} /></label>
            <button onClick={runEditorEstimate}>Start TN estimate</button>
            <button onClick={runEditorAmps}>Start TN amplitudes</button>
            <button onClick={runSample}>Start sampling</button>
            <label style={{ display: "flex", gap: 6, alignItems: "center" }}>auto-attach<input type="checkbox" checked={autoAttachResult} onChange={(e) => setAutoAttachResult(e.target.checked)} /></label>
            <button onClick={forceAttachLatestResult} disabled={!jobOut && !selectedRun?.result}>Attach latest result to canvas</button>
          </div>
          <div style={{ fontSize: 12, marginTop: 8 }}>run: {jobRunning ? `running ${Math.round(jobProgress * 100)}%` : "idle"}{jobId ? ` | job ${jobId.slice(0,8)}` : ""}</div>
          <div style={{ fontSize: 12, marginTop: 4 }}>
            phase: {extractPhaseFromJob(selectedAgentJob) ?? "n/a"}
          </div>
          {jobRunning && jobId ? <button onClick={async () => { try { await cancelAsyncJob(jobId); } catch {} }} style={{ marginTop: 8 }}>Cancel job</button> : null}
          {jobErr ? <pre style={{ marginTop: 8, background: "#f9fafb", border: "1px solid #d1d5db", padding: 8, whiteSpace: "pre-wrap" }}>{jobErr}</pre> : null}
        </section>
        ) : null}

        {(showRun || showResults) ? (
        <section style={sectionStyle}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Version and Runs</div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            <button onClick={saveVersionToBackend}>Save version</button>
            <button onClick={loadLatestVersion}>Load latest</button>
            <label style={{ display: "flex", gap: 6, alignItems: "center" }}>version id<input value={loadVersionId} onChange={(e) => setLoadVersionId(e.target.value)} style={{ width: 90 }} /></label>
            <button onClick={() => { const id = Number(loadVersionId); if (Number.isFinite(id) && id > 0) loadVersionFromBackend(id); }}>Load</button>
            <button onClick={refreshRuns}>Refresh runs</button>
          </div>
          <div style={{ fontSize: 12, marginTop: 8 }}>backend version: {versionId ? `#${versionId}` : "not saved"}{backendMsg ? ` | ${backendMsg}` : ""}</div>
          <ul style={{ marginBottom: 8 }}>{runs.slice(0, 12).map((r) => (<li key={r.id}><button onClick={() => setSelectedRun(r)} style={{ marginRight: 8 }}>View</button><button onClick={() => setCompareRunA(r.id)} style={{ marginRight: 4 }}>A</button><button onClick={() => setCompareRunB(r.id)} style={{ marginRight: 8 }}>B</button>#{r.id} {r.kind} [{r.status}]</li>))}</ul>
          <div style={{ fontSize: 12 }}>compare: A={compareRunA ?? "-"}, B={compareRunB ?? "-"}</div>
          {compareSummary ? (
            <div style={{ marginTop: 8, border: "1px solid #d1d5db", borderRadius: 6, overflow: "hidden" }}>
              <div style={{ padding: "6px 8px", borderBottom: "1px solid #e5e7eb", fontWeight: 600, fontSize: 12 }}>Compare Summary</div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <tbody>
                  <tr><td style={{ borderBottom: "1px solid #f1f5f9", padding: 6 }}>Run A</td><td style={{ borderBottom: "1px solid #f1f5f9", padding: 6 }}>#{compareSummary.a.id} {compareSummary.a.kind} [{compareSummary.a.status}]</td></tr>
                  <tr><td style={{ borderBottom: "1px solid #f1f5f9", padding: 6 }}>Run B</td><td style={{ borderBottom: "1px solid #f1f5f9", padding: 6 }}>#{compareSummary.b.id} {compareSummary.b.kind} [{compareSummary.b.status}]</td></tr>
                  <tr><td style={{ borderBottom: "1px solid #f1f5f9", padding: 6 }}>Delta duration (ms)</td><td style={{ borderBottom: "1px solid #f1f5f9", padding: 6 }}>{String(compareSummary.deltaDurationMs ?? "-")}</td></tr>
                  <tr><td style={{ borderBottom: "1px solid #f1f5f9", padding: 6 }}>Counts TV distance</td><td style={{ borderBottom: "1px solid #f1f5f9", padding: 6 }}>{compareSummary.countsDistanceTV == null ? "-" : compareSummary.countsDistanceTV.toFixed(6)}</td></tr>
                  <tr><td style={{ borderBottom: "1px solid #f1f5f9", padding: 6 }}>Amplitude L2</td><td style={{ borderBottom: "1px solid #f1f5f9", padding: 6 }}>{compareSummary.amplitudeDelta ? compareSummary.amplitudeDelta.l2.toFixed(6) : "-"}</td></tr>
                  <tr><td style={{ padding: 6 }}>Amplitude max abs</td><td style={{ padding: 6 }}>{compareSummary.amplitudeDelta ? compareSummary.amplitudeDelta.maxAbs.toFixed(6) : "-"}</td></tr>
                </tbody>
              </table>
            </div>
          ) : null}
        </section>
        ) : null}

        {showResults ? (
        <section style={sectionStyle}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Validation and Export</div>
          <div style={{ fontSize: 12, marginBottom: 8 }}>validation: <strong>{validation.status}</strong></div>
          <ul style={{ marginTop: 0 }}>
            {validation.checks.map((c, idx) => (
              <li key={`${idx}_${c}`} style={{ fontSize: 12 }}>{c}</li>
            ))}
            {validation.checks.length === 0 ? <li style={{ fontSize: 12 }}>no checks yet</li> : null}
          </ul>
          <button onClick={exportBundle}>Export bundle (JSON)</button>
        </section>
        ) : null}

        {showResults && resultCharts && (resultCharts.counts.length > 0 || resultCharts.amps.length > 0) ? (
          <section style={sectionStyle}>
            <div style={{ fontWeight: 600, marginBottom: 8 }}>Result Charts</div>
            {resultCharts.counts.length > 0 ? (
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>Counts (top 16)</div>
                {(() => {
                  const max = Math.max(1, ...resultCharts.counts.map((x) => x.value));
                  return resultCharts.counts.map((item) => (
                    <div key={`c_${item.key}`} style={{ display: "grid", gridTemplateColumns: "80px 1fr 60px", gap: 8, alignItems: "center", marginBottom: 4, fontSize: 12 }}>
                      <span>{item.key}</span>
                      <div style={{ height: 10, background: "#e5e7eb", borderRadius: 4, overflow: "hidden" }}>
                        <div style={{ width: `${(item.value / max) * 100}%`, height: "100%", background: "#334155" }} />
                      </div>
                      <span>{item.value}</span>
                    </div>
                  ));
                })()}
              </div>
            ) : null}
            {resultCharts.amps.length > 0 ? (
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>Amplitude |a| (top 16)</div>
                {(() => {
                  const max = Math.max(1e-12, ...resultCharts.amps.map((x) => x.mag));
                  return resultCharts.amps.map((item) => (
                    <div key={`a_${item.key}`} style={{ display: "grid", gridTemplateColumns: "80px 1fr 70px", gap: 8, alignItems: "center", marginBottom: 4, fontSize: 12 }}>
                      <span>{item.key}</span>
                      <div style={{ height: 10, background: "#e5e7eb", borderRadius: 4, overflow: "hidden" }}>
                        <div style={{ width: `${(item.mag / max) * 100}%`, height: "100%", background: "#0f766e" }} />
                      </div>
                      <span>{item.mag.toFixed(4)}</span>
                    </div>
                  ));
                })()}
              </div>
            ) : null}
          </section>
        ) : null}

        {showExperiments ? (
        <section style={sectionStyle}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Experiment Queue (batch)</div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginBottom: 8 }}>
            <button onClick={() => enqueueCurrent("tn_estimate")}>Enqueue estimate</button>
            <button onClick={() => enqueueCurrent("tn_amplitudes")}>Enqueue amplitudes</button>
            <button onClick={() => enqueueCurrent("sample")}>Enqueue sample</button>
            <button onClick={runQueue} disabled={queueRunning || queueItems.every((x) => x.status !== "queued")}>
              {queueRunning ? "Queue running..." : "Run queue"}
            </button>
            <button onClick={() => setQueueCancelRequested(true)} disabled={!queueRunning}>Cancel queue</button>
            <button onClick={() => setQueueItems([])} disabled={queueRunning}>Clear queue</button>
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginBottom: 8 }}>
            <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
              sweep shots
              <input value={sweepShots} onChange={(e) => setSweepShots(e.target.value)} style={{ width: 160 }} />
            </label>
            <button onClick={enqueueSweepShots}>Add shot sweep</button>
            <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
              sweep theta
              <input value={sweepThetas} onChange={(e) => setSweepThetas(e.target.value)} style={{ width: 160 }} />
            </label>
            <button onClick={enqueueSweepTheta}>Add theta sweep</button>
          </div>
          <div style={{ fontSize: 12, marginBottom: 6 }}>
            queued: {queueItems.filter((x) => x.status === "queued").length} | running: {queueItems.filter((x) => x.status === "running").length} | done: {queueItems.filter((x) => x.status === "done").length} | failed: {queueItems.filter((x) => x.status === "failed").length}
          </div>
          <div style={{ maxHeight: 180, overflow: "auto", border: "1px solid #e5e7eb" }}>
            {queueItems.map((item) => (
              <div key={item.id} style={{ padding: "6px 8px", borderBottom: "1px solid #f1f5f9", fontSize: 12 }}>
                <strong>{item.kind}</strong> | {item.label} | {item.status} | validation: {item.validation ?? "unknown"}{item.error ? ` | ${item.error}` : ""}{item.sweepGroupId ? ` | group=${item.sweepGroupId}` : ""}
                {item.checks && item.checks.length > 0 ? (
                  <div style={{ marginTop: 4, color: "#475569" }}>{item.checks.join(" ; ")}</div>
                ) : null}
              </div>
            ))}
            {queueItems.length === 0 ? <div style={{ padding: 8, fontSize: 12, color: "#6b7280" }}>Queue empty.</div> : null}
          </div>
        </section>
        ) : null}

        {selectedRun ? <section style={sectionStyle}><div style={{ fontWeight: 600, marginBottom: 8 }}>Run Detail #{selectedRun.id}</div><div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}><button onClick={async () => { setJobErr(null); setJobOut(null); const req = selectedRun.request ?? {}; if (selectedRun.kind === "tn_estimate") await runAndPersist("tn_estimate", req); else if (selectedRun.kind === "tn_amplitudes") await runAndPersist("tn_amplitudes", req); else if (selectedRun.kind === "sample") await runAndPersist("sample", req); }}>Re-run</button><button onClick={async () => { const arts = await listArtifacts(selectedRun.id); setSelectedArtifacts(arts); }}>Load artifacts</button></div><pre style={{ margin: 0, background: "#f9fafb", border: "1px solid #d1d5db", padding: 8, whiteSpace: "pre-wrap" }}>{JSON.stringify({ kind: selectedRun.kind, status: selectedRun.status, request: selectedRun.request, result: selectedRun.result }, null, 2)}</pre></section> : null}

        {jobOut ? <section style={sectionStyle}><div style={{ fontWeight: 600, marginBottom: 8 }}>Latest Output</div><pre style={{ margin: 0, background: "#f9fafb", border: "1px solid #d1d5db", padding: 8, whiteSpace: "pre-wrap" }}>{JSON.stringify(jobOut, null, 2)}</pre></section> : null}

        {showRun ? (
        <section style={sectionStyle}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Examples (click to load)</div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
            <button onClick={() => applyExample("bell")}>Bell (2q)</button>
            <button onClick={() => applyExample("ghz3")}>GHZ (3q)</button>
            <button onClick={() => applyExample("qft2")}>QFT-like (2q)</button>
          </div>
          <div style={{ fontSize: 12, marginBottom: 8 }}>active: {activeExample || "-"}</div>
          {exampleCode ? (
            <div style={{ display: "grid", gap: 8 }}>
              <div>
                <div style={{ fontWeight: 600, fontSize: 12 }}>OpenQASM 3</div>
                <pre style={{ margin: 0, background: "#f9fafb", border: "1px solid #d1d5db", padding: 8, whiteSpace: "pre-wrap" }}>{exampleCode.qasm}</pre>
              </div>
              <div>
                <div style={{ fontWeight: 600, fontSize: 12 }}>Qiskit</div>
                <pre style={{ margin: 0, background: "#f9fafb", border: "1px solid #d1d5db", padding: 8, whiteSpace: "pre-wrap" }}>{exampleCode.qiskit}</pre>
              </div>
              <div>
                <div style={{ fontWeight: 600, fontSize: 12 }}>Cirq</div>
                <pre style={{ margin: 0, background: "#f9fafb", border: "1px solid #d1d5db", padding: 8, whiteSpace: "pre-wrap" }}>{exampleCode.cirq}</pre>
              </div>
            </div>
          ) : null}
        </section>
        ) : null}

        {(showRun || showExperiments) ? <section style={sectionStyle}><IrPanel /></section> : null}
      </div>
    </div>
  );
}

