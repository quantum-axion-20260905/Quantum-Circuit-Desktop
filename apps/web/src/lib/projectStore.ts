import type { CircuitIrV1 } from "../ir/ir";
import type { ExperimentRecord } from "./runHistory";
import { invokeDesktop, isDesktop } from "./desktop";
import { buildPhysicsStudyManifest, type PhysicsStudyMode, type PhysicsStudyRow } from "./physicsStudy";

const BACKEND_BASE_URL = "http://127.0.0.1:8000/api";
const DEFAULT_PROJECT_NAME = "Default Project";

type JsonObject = Record<string, unknown>;

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null;
}

function collection<T>(value: unknown): T[] {
  if (Array.isArray(value)) return value as T[];
  if (isObject(value) && Array.isArray(value.results)) return value.results as T[];
  return [];
}

async function backendFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BACKEND_BASE_URL}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
  });
  const contentType = response.headers.get("content-type") ?? "";
  const body: unknown = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = isObject(body) && "detail" in body ? String(body.detail) : typeof body === "string" ? body : JSON.stringify(body);
    throw new Error(`Backend error ${response.status}: ${detail}`);
  }
  return body as T;
}

async function defaultProjectId(create = true): Promise<number> {
  const projects = collection<{ id: number; name: string }>(await backendFetch("/projects/?page_size=100"));
  const existing = projects.find((project) => project.name === DEFAULT_PROJECT_NAME);
  if (existing) return existing.id;
  if (!create) throw new Error("No saved project was found.");
  const created = await backendFetch<{ id: number }>("/projects/", {
    method: "POST",
    body: JSON.stringify({ name: DEFAULT_PROJECT_NAME }),
  });
  return created.id;
}

export async function saveCircuitVersion(ir: CircuitIrV1): Promise<{ id: number; fingerprint: string }> {
  const project = await defaultProjectId();
  return backendFetch<{ id: number; fingerprint: string }>("/versions/", {
    method: "POST",
    body: JSON.stringify({ project, qasm: ir.qasm, metadata: ir }),
  });
}

export async function loadLatestCircuitVersion(): Promise<CircuitIrV1> {
  const project = await defaultProjectId(false);
  const versions = collection<{ metadata: unknown }>(await backendFetch(`/versions/?project=${project}&page_size=1`));
  if (!versions[0]) throw new Error("No saved circuit version was found.");
  return versions[0].metadata as CircuitIrV1;
}

function backendQasm(record: ExperimentRecord) {
  if (record.request.source === "circuit" && record.request.circuitQasm) return record.request.circuitQasm;
  return `OPENQASM 3;\ninclude "stdgates.inc";\n// ${record.source} plugin problem: ${record.kind}\n`;
}

/**
 * Best-effort server sync. Local history remains the source of truth when the
 * Django API is not running, while an available API receives a queryable Run
 * row plus a raw result artifact for replay/audit.
 */
export async function syncExperimentRecord(record: ExperimentRecord): Promise<void> {
  if (isDesktop()) {
    await invokeDesktop("record_run", {
      request: { ...record.request, kind: record.kind, source: record.source },
      result: record.result ?? {},
      status: record.status,
      error: record.error ?? null,
    });
    return;
  }
  const nQubits = Math.max(1, Math.min(64, Number(record.result?.n_qubits ?? 1)));
  const version = await saveCircuitVersion({
    qasm: backendQasm(record),
    ui: { version: 1, source_format: record.source === "physics" ? "physics-plugin" : "qasm3", n_qubits: nQubits, nodes: [] },
  });
  const run = await backendFetch<{ id: number }>("/runs/", {
    method: "POST",
    body: JSON.stringify({
      kind: record.kind,
      version: version.id,
      status: record.status,
      finished_at: record.createdAt,
      error: record.error ?? "",
      agent_base_url: "http://127.0.0.1:8788",
      backend: record.provenance?.resolved_backend ?? record.result?.backend ?? "",
      device: record.provenance?.device ?? {},
      request: record.request,
      result: record.result ?? {},
    }),
  });
  if (record.result) {
    await backendFetch("/artifacts/", {
      method: "POST",
      body: JSON.stringify({ run: run.id, kind: "raw", content: record.result }),
    });
  }
}

export type PhysicsStudySyncInput = {
  mode: PhysicsStudyMode;
  startedAt: string;
  finishedAt?: string;
  rows: PhysicsStudyRow[];
  nQubits: number;
  configuration: JsonObject;
};

/** Persist one aggregate convergence campaign alongside its point-level Runs. */
export async function syncPhysicsStudy(input: PhysicsStudySyncInput): Promise<void> {
  const finishedAt = input.finishedAt ?? new Date().toISOString();
  const manifest = buildPhysicsStudyManifest({
    mode: input.mode,
    nQubits: input.nQubits,
    startedAt: input.startedAt,
    finishedAt,
    configuration: input.configuration,
    rows: input.rows,
  });
  const summary = manifest.summary as JsonObject;
  const status = Number(summary.failed ?? 0) > 0
    ? "failed"
    : Number(summary.canceled ?? 0) > 0
      ? "canceled"
      : "done";
  const study = {
    kind: `${input.mode}-convergence`,
    label: `Physics · ${input.mode.toUpperCase()} convergence study`,
    status,
    started_at: input.startedAt,
    finished_at: finishedAt,
    request: manifest,
    result: manifest,
  };
  if (isDesktop()) {
    await invokeDesktop("record_study", { study });
    return;
  }
  const project = await defaultProjectId();
  await backendFetch("/studies/", {
    method: "POST",
    body: JSON.stringify({ project, ...study }),
  });
}
