export const AGENT_BASE_URL = "http://127.0.0.1:8788";
import { desktopAgentGet, desktopAgentPost, isDesktop } from "./desktop";

export type JsonObject = Record<string, unknown>;

export type AgentMethodCapability = {
  id: string;
  method: "dmrg" | "tebd" | "tdvp" | "vumps" | "ctmrg";
  backend: string;
  representation: string;
  operation: "ground_state" | "evolve";
  available: boolean;
  status: "available" | "unavailable" | "planned";
  description: string;
  limitations?: string[];
};

export type AgentCapabilities = {
  backends?: Array<{ name: string; available: boolean; device?: string }>;
  methods?: AgentMethodCapability[];
  gpu?: { available?: boolean; [key: string]: unknown };
  agent_version?: string;
  queue?: { max_workers?: number; jobs?: Record<string, number>; resources?: JsonObject };
  features?: Record<string, unknown>;
};

export type PreflightReport = {
  feasible?: boolean;
  status?: string;
  backend?: string;
  estimated_peak_memory_mb?: number;
  path_cost?: number;
  estimated_time_ms?: number;
  warnings?: string[];
  [key: string]: unknown;
};

export type AgentAmplitude = { bitstring: string; re?: number; im?: number };

export type AgentProvenance = {
  schema?: string;
  request_sha256?: string;
  circuit_sha256?: string;
  problem_sha256?: string;
  result_sha256?: string;
  requested_backend?: string;
  resolved_backend?: string;
  backend?: string;
  agent_version?: string;
  seed?: number | null;
  elapsed_ms?: number;
  device?: JsonObject;
};

export type AgentResult = {
  status?: string;
  backend?: string;
  method?: string;
  n_qubits?: number;
  counts?: Record<string, number>;
  amplitudes?: AgentAmplitude[];
  norm2?: number;
  shots?: number;
  bond_dim_requested?: number;
  bond_dim_used?: number;
  bond_dim_history?: number[];
  boundary_bond_dim_requested?: number;
  boundary_bond_dim_used?: number;
  boundary_diagnostics?: JsonObject | null;
  bond_growth?: number;
  truncation_cutoff?: number;
  discarded_weight?: number;
  discarded_weight_history?: number[];
  norm_drift?: number;
  norm_history?: number[];
  approximate?: boolean;
  warnings?: string[];
  provenance?: AgentProvenance;
  [key: string]: unknown;
};

export type AsyncKind =
  | "run"
  | "sample"
  | "simulate"
  | "bench_matmul"
  | "expectation"
  | "tebd"
  | "ground_state"
  | "dmrg"
  | "peps"
  | "tn_estimate"
  | "tn_amplitudes"
  | "sweep"
  | "cross_validate";

export type AsyncJob = AgentResult & {
  job_id: string;
  kind: AsyncKind | string;
  status: "queued" | "running" | "done" | "failed" | "canceled";
  progress: number;
  error?: string | null;
  resource?: JsonObject;
  logs?: Array<Record<string, unknown>>;
  artifacts?: { result?: AgentResult; [key: string]: unknown };
};

export type AsyncJobUpdate = (job: AsyncJob) => void;

export type LatticeGraph = {
  dimensions: number[];
  dimension: number;
  boundary: "open" | "periodic";
  ordering: string;
  sites: Array<{ index: number; coordinate: number[] }>;
  edges: Array<{ source: number; target: number }>;
};

export type PauliTerm = {
  paulis: Record<string, "I" | "X" | "Y" | "Z">;
  coefficient: number;
  label?: string | null;
};

export type FermionOperator = { mode: number; action: "create" | "annihilate" };
export type FermionTerm = { operators: FermionOperator[]; coefficient: number; label?: string | null };
export type FermionMappingResponse = {
  mapping: string;
  n_qubits: number;
  terms: PauliTerm[];
  complex_terms?: Array<{ paulis: Record<string, string>; real: number; imag: number; label?: string }>;
  expectation_ready: boolean;
  max_imaginary_coefficient: number;
  warnings?: string[];
};

export type SparseHamiltonianResponse = LatticeGraph & {
  n_qubits: number;
  terms: PauliTerm[];
  tebd_ready?: boolean;
  warnings?: string[];
};

export type HamiltonianResponse = SparseHamiltonianResponse & {
  model: string;
  coupling: number;
  field: number;
  anisotropy: number;
  terms: PauliTerm[];
};

export type DomainPlugin = {
  id: string;
  name: string;
  version: string;
  description: string;
  capabilities: string[];
  api_version?: string;
  actions?: string[];
};

function isJsonObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null;
}

async function agentFetch<T = unknown>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  const publicToken = process.env.NEXT_PUBLIC_QC_AGENT_TOKEN;
  if (publicToken) headers.set("x-qc-agent-token", publicToken);
  const res = await fetch(`${AGENT_BASE_URL}${path}`, { ...init, headers });
  const contentType = res.headers.get("content-type") ?? "";
  const body = contentType.includes("application/json") ? await res.json() : await res.text();
  if (!res.ok) {
    const msg =
      typeof body === "string"
        ? body
        : isJsonObject(body) && "detail" in body
          ? String(body.detail)
          : JSON.stringify(body);
    throw new Error(`Agent error ${res.status}: ${msg}`);
  }
  return body as T;
}

export async function getHardware(): Promise<AgentCapabilities> {
  if (isDesktop()) return desktopAgentGet<AgentCapabilities>("/hardware");
  return agentFetch<AgentCapabilities>("/hardware");
}

export async function getCapabilities(): Promise<AgentCapabilities> {
  if (isDesktop()) return desktopAgentGet<AgentCapabilities>("/capabilities");
  return agentFetch<AgentCapabilities>("/capabilities");
}

export async function preflight(payload: JsonObject): Promise<PreflightReport> {
  if (isDesktop()) return desktopAgentPost<PreflightReport>("/jobs/preflight", payload);
  return agentFetch<PreflightReport>("/jobs/preflight", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload) });
}

export async function runJob(kind: string, circuit: JsonObject, config: JsonObject, onUpdate?: AsyncJobUpdate): Promise<AgentResult> {
  // Keep browser and desktop on the same durable async contract. The Tauri
  // shell only supplies a local transport; it must not have a second compute
  // lifecycle that can drift from the agent API.
  return runAsyncAndWait("run", { ...circuit, ...config, backend: config.backend ?? "auto" }, onUpdate);
}

export async function runSweep(payload: JsonObject, onUpdate?: AsyncJobUpdate): Promise<AgentResult> {
  return runAsyncAndWait("sweep", payload, onUpdate);
}

export async function listPlugins(): Promise<{ plugins: DomainPlugin[] }> {
  if (isDesktop()) return desktopAgentGet<{ plugins: DomainPlugin[] }>("/plugins");
  return agentFetch<{ plugins: DomainPlugin[] }>("/plugins");
}

export async function previewLattice(pluginId: string, payload: JsonObject): Promise<LatticeGraph> {
  if (isDesktop()) return desktopAgentPost<LatticeGraph>(`/plugins/${pluginId}/lattice`, payload);
  return agentFetch<LatticeGraph>(`/plugins/${pluginId}/lattice`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload) });
}

export async function buildHamiltonian(pluginId: string, payload: JsonObject): Promise<HamiltonianResponse> {
  if (isDesktop()) return desktopAgentPost<HamiltonianResponse>(`/plugins/${pluginId}/hamiltonian`, payload);
  return agentFetch<HamiltonianResponse>(`/plugins/${pluginId}/hamiltonian`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload) });
}

export async function mapFermions(pluginId: string, payload: JsonObject): Promise<FermionMappingResponse> {
  if (isDesktop()) return desktopAgentPost<FermionMappingResponse>(`/plugins/${pluginId}/fermion_mapping`, payload);
  return agentFetch<FermionMappingResponse>(`/plugins/${pluginId}/fermion_mapping`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload) });
}

export async function buildHubbard(pluginId: string, payload: JsonObject): Promise<SparseHamiltonianResponse & { material: string; expectation_ready: boolean }> {
  if (isDesktop()) return desktopAgentPost<SparseHamiltonianResponse & { material: string; expectation_ready: boolean }>(`/plugins/${pluginId}/hubbard`, payload);
  return agentFetch<SparseHamiltonianResponse & { material: string; expectation_ready: boolean }>(`/plugins/${pluginId}/hubbard`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload) });
}

export async function runExpectation(payload: JsonObject, onUpdate?: AsyncJobUpdate): Promise<AgentResult> {
  return runAsyncAndWait("expectation", payload, onUpdate);
}

export async function runTEBD(payload: JsonObject, onUpdate?: AsyncJobUpdate): Promise<AgentResult> {
  return runAsyncAndWait("tebd", payload, onUpdate);
}

export async function runGroundState(payload: JsonObject, onUpdate?: AsyncJobUpdate): Promise<AgentResult> {
  return runAsyncAndWait("ground_state", payload, onUpdate);
}

export async function runDMRG(payload: JsonObject, onUpdate?: AsyncJobUpdate): Promise<AgentResult> {
  return runAsyncAndWait("dmrg", payload, onUpdate);
}

export async function runPEPS(payload: JsonObject, onUpdate?: AsyncJobUpdate): Promise<AgentResult> {
  return runAsyncAndWait("peps", payload, onUpdate);
}

export async function benchMatmul(payload: { size?: number; iters?: number; dtype?: "fp16" | "fp32" }): Promise<AgentResult> {
  return runAsyncAndWait("bench_matmul", payload);
}

export async function tnAmplitudes(payload: JsonObject): Promise<AgentResult> {
  return runAsyncAndWait("tn_amplitudes", payload);
}

export async function tnEstimate(payload: JsonObject): Promise<AgentResult> {
  return runAsyncAndWait("tn_estimate", payload);
}

export async function sample(payload: JsonObject): Promise<AgentResult> {
  return runAsyncAndWait("sample", payload);
}

export async function submitAsync(kind: AsyncKind, payload: JsonObject): Promise<AsyncJob> {
  const request = { kind, payload };
  if (isDesktop()) return desktopAgentPost<AsyncJob>("/async/jobs", request);
  return agentFetch<AsyncJob>("/async/jobs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(request)
  });
}

export async function getAsyncJob(jobId: string): Promise<AsyncJob> {
  if (isDesktop()) return desktopAgentGet<AsyncJob>(`/async/jobs/${jobId}`);
  return agentFetch<AsyncJob>(`/async/jobs/${jobId}`, { method: "GET" });
}

export async function cancelAsyncJob(jobId: string): Promise<{ ok?: boolean }> {
  if (isDesktop()) return desktopAgentPost<{ ok?: boolean }>(`/async/jobs/${jobId}/cancel`, {});
  return agentFetch<{ ok?: boolean }>(`/async/jobs/${jobId}/cancel`, { method: "POST" });
}

export async function waitForAsyncJob(jobId: string, options: { intervalMs?: number; timeoutMs?: number; onUpdate?: AsyncJobUpdate } = {}): Promise<AgentResult> {
  const intervalMs = options.intervalMs ?? 250;
  const timeoutMs = options.timeoutMs ?? 3_600_000;
  const deadline = Date.now() + timeoutMs;
  while (Date.now() <= deadline) {
    const job = await getAsyncJob(jobId);
    options.onUpdate?.(job);
    if (job.status === "done") return job.artifacts?.result ?? job;
    if (job.status === "failed") throw new Error(job.error || "Async agent job failed.");
    if (job.status === "canceled") throw new Error("Async agent job was canceled.");
    await new Promise((resolve) => window.setTimeout(resolve, intervalMs));
  }
  await cancelAsyncJob(jobId);
  throw new Error("Async agent job timed out and was canceled.");
}

async function runAsyncAndWait(kind: AsyncKind, payload: JsonObject, onUpdate?: AsyncJobUpdate): Promise<AgentResult> {
  const submitted = await submitAsync(kind, payload);
  onUpdate?.(submitted);
  return waitForAsyncJob(submitted.job_id, { onUpdate });
}
