export const AGENT_BASE_URL = "http://127.0.0.1:8788";
import { invokeDesktop, isDesktop } from "./desktop";

async function agentFetch(path: string, init?: RequestInit) {
  const res = await fetch(`${AGENT_BASE_URL}${path}`, init);
  const contentType = res.headers.get("content-type") ?? "";
  const body = contentType.includes("application/json") ? await res.json() : await res.text();
  if (!res.ok) {
    const msg =
      typeof body === "string"
        ? body
        : body && typeof body === "object" && "detail" in body
          ? String((body as any).detail)
          : JSON.stringify(body);
    throw new Error(`Agent error ${res.status}: ${msg}`);
  }
  return body;
}

export async function getHardware() {
  if (isDesktop()) return invokeDesktop("get_hardware_capabilities");
  return agentFetch("/hardware");
}

export async function getCapabilities() {
  if (isDesktop()) return invokeDesktop("get_hardware_capabilities");
  return agentFetch("/capabilities");
}

export async function preflight(payload: any) {
  if (isDesktop()) return invokeDesktop("run_preflight", { payload });
  return agentFetch("/jobs/preflight", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload) });
}

export async function runJob(kind: string, circuit: any, config: any) {
  if (isDesktop()) return invokeDesktop("submit_run", { request: { kind, circuit, config } });
  return agentFetch("/jobs/run", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ ...circuit, ...config, backend: config.backend ?? "auto" }) });
}

export async function benchMatmul(payload: { size?: number; iters?: number; dtype?: "fp16" | "fp32" }) {
  return agentFetch("/jobs/bench_matmul", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function tnAmplitudes(payload: any) {
  return agentFetch("/jobs/tn_amplitudes", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function tnEstimate(payload: any) {
  return agentFetch("/jobs/tn_estimate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function sample(payload: any) {
  return agentFetch("/jobs/sample", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function submitAsync(kind: "tn_estimate" | "tn_amplitudes" | "sample", payload: any) {
  return agentFetch(`/async/${kind}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function getAsyncJob(jobId: string) {
  return agentFetch(`/async/jobs/${jobId}`, { method: "GET" });
}

export async function cancelAsyncJob(jobId: string) {
  return agentFetch(`/async/jobs/${jobId}/cancel`, { method: "POST" });
}
