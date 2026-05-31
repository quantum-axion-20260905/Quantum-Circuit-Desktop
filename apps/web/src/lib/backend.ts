const DEFAULT_API_BASE = "http://127.0.0.1:8000";

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? DEFAULT_API_BASE;

async function apiFetch(path: string, init?: RequestInit) {
  const res = await fetch(`${API_BASE_URL}${path}`, init);
  const contentType = res.headers.get("content-type") ?? "";
  const body = contentType.includes("application/json") ? await res.json() : await res.text();
  if (!res.ok) {
    const msg =
      typeof body === "string"
        ? body
        : body && typeof body === "object" && "detail" in body
          ? String((body as any).detail)
          : JSON.stringify(body);
    throw new Error(`API error ${res.status}: ${msg}`);
  }
  return body;
}

export type Project = { id: number; name: string; created_at: string };
export type CircuitVersion = { id: number; project: number; created_at: string; qasm: string; metadata: any };
export type Run = {
  id: number;
  kind: string;
  version: number;
  created_at: string;
  status: string;
  seed: number | null;
  started_at: string | null;
  finished_at: string | null;
  error: string;
  agent_base_url: string;
  agent_job_id: string;
  backend: string;
  device: any;
  request: any;
  result: any;
};

export type RunArtifact = { id: number; run: number; kind: string; created_at: string; content: any };

export async function ensureDefaultProject(name = "Default Project"): Promise<Project> {
  const resp = (await apiFetch("/api/projects/")) as any;
  const projects: Project[] = Array.isArray(resp) ? resp : (resp.results ?? []);
  const existing = projects.find((p) => p.name === name);
  if (existing) return existing;
  return apiFetch("/api/projects/", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name })
  });
}

export async function createVersion(projectId: number, qasm: string, metadata: any): Promise<CircuitVersion> {
  return apiFetch("/api/versions/", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ project: projectId, qasm, metadata })
  });
}

export async function listVersions(projectId?: number): Promise<CircuitVersion[]> {
  const url = projectId ? `/api/versions/?project=${projectId}` : "/api/versions/";
  const resp = (await apiFetch(url)) as any;
  return Array.isArray(resp) ? resp : (resp.results ?? []);
}

export async function getVersion(versionId: number): Promise<CircuitVersion> {
  return apiFetch(`/api/versions/${versionId}/`);
}

export async function listRuns(versionId?: number): Promise<Run[]> {
  const url = versionId ? `/api/runs/?version=${versionId}` : "/api/runs/";
  const resp = (await apiFetch(url)) as any;
  const runs: Run[] = Array.isArray(resp) ? resp : (resp.results ?? []);
  return runs;
}

export async function createRun(payload: Omit<Run, "id" | "created_at">): Promise<Run> {
  return apiFetch("/api/runs/", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function patchRun(runId: number, patch: Partial<Run>): Promise<Run> {
  return apiFetch(`/api/runs/${runId}/`, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(patch)
  });
}

export async function createArtifact(payload: Omit<RunArtifact, "id" | "created_at">): Promise<RunArtifact> {
  return apiFetch("/api/artifacts/", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function listArtifacts(runId: number): Promise<RunArtifact[]> {
  const resp = (await apiFetch(`/api/artifacts/?run=${runId}`)) as any;
  return Array.isArray(resp) ? resp : (resp.results ?? []);
}
