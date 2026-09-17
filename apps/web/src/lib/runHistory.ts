import {
  runDMRG,
  runExpectation,
  runGroundState,
  runJob,
  runPEPS,
  runSweep,
  runTEBD,
  type AgentProvenance,
  type AgentResult,
  type AsyncJobUpdate,
  type JsonObject,
} from "./agent";

export type ReplayRequest =
  | { source: "circuit"; kind: "simulation"; circuit: JsonObject; config: JsonObject; circuitQasm?: string }
  | { source: "circuit"; kind: "sweep"; payload: JsonObject; circuitQasm?: string }
  | { source: "physics"; kind: "expectation" | "ground_state" | "dmrg" | "tebd" | "peps"; payload: JsonObject };

export type ExperimentRecord = {
  id: string;
  createdAt: string;
  label: string;
  source: ReplayRequest["source"];
  kind: ReplayRequest["kind"];
  status: "done" | "failed" | "canceled";
  request: ReplayRequest;
  result?: AgentResult;
  error?: string;
  provenance?: AgentProvenance;
};

const STORAGE_KEY = "qc-experiment-history-v1";
const MAX_RECORDS = 50;
export const HISTORY_EVENT = "qc-experiment-history-change";
let cachedRaw: string | null | undefined;
let cachedRecords: ExperimentRecord[] = [];

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isReplayRequest(value: unknown): value is ReplayRequest {
  if (!isObject(value) || (value.source !== "circuit" && value.source !== "physics")) return false;
  if (typeof value.kind !== "string") return false;
  if (value.source === "circuit") {
    return value.kind === "simulation"
      ? isObject(value.circuit) && isObject(value.config)
      : value.kind === "sweep" && isObject(value.payload);
  }
  return ["expectation", "ground_state", "dmrg", "tebd", "peps"].includes(value.kind) && isObject(value.payload);
}

function isExperimentRecord(value: unknown): value is ExperimentRecord {
  return isObject(value)
    && typeof value.id === "string"
    && typeof value.createdAt === "string"
    && typeof value.label === "string"
    && (value.status === "done" || value.status === "failed" || value.status === "canceled")
    && isReplayRequest(value.request);
}

export function loadExperimentHistory(): ExperimentRecord[] {
  if (typeof window === "undefined") return [];
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (raw === cachedRaw) return cachedRecords;
  cachedRaw = raw;
  try {
    const parsed: unknown = JSON.parse(raw ?? "[]");
    cachedRecords = Array.isArray(parsed) ? parsed.filter(isExperimentRecord).slice(0, MAX_RECORDS) : [];
  } catch {
    cachedRecords = [];
  }
  return cachedRecords;
}

export function saveExperimentHistory(records: ExperimentRecord[]) {
  if (typeof window === "undefined") return;
  try {
    const next = records.slice(0, MAX_RECORDS);
    const raw = JSON.stringify(next);
    cachedRaw = raw;
    cachedRecords = next;
    window.localStorage.setItem(STORAGE_KEY, raw);
    window.dispatchEvent(new Event(HISTORY_EVENT));
  } catch {
    // Local history is best-effort; compute results remain available in memory.
  }
}

export function createExperimentId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export async function replayExperiment(request: ReplayRequest, onUpdate?: AsyncJobUpdate): Promise<AgentResult> {
  if (request.source === "circuit") {
    if (request.kind === "simulation") return runJob("simulation", request.circuit, request.config, onUpdate);
    return runSweep(request.payload, onUpdate);
  }
  switch (request.kind) {
    case "expectation": return runExpectation(request.payload, onUpdate);
    case "ground_state": return runGroundState(request.payload, onUpdate);
    case "dmrg": return runDMRG(request.payload, onUpdate);
    case "tebd": return runTEBD(request.payload, onUpdate);
    case "peps": return runPEPS(request.payload, onUpdate);
  }
}
