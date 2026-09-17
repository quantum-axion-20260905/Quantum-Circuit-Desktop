"use client";

declare global {
  interface Window {
    __TAURI_INTERNALS__?: { invoke?: (command: string, args?: Record<string, unknown>) => Promise<unknown> };
  }
}

export function isDesktop(): boolean {
  return typeof window !== "undefined" && typeof window.__TAURI_INTERNALS__?.invoke === "function";
}

export async function invokeDesktop<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  const invoke = typeof window !== "undefined" ? window.__TAURI_INTERNALS__?.invoke : undefined;
  if (!invoke) throw new Error("This operation is available in Quantum Circuit Desktop only.");
  return invoke(command, args) as Promise<T>;
}

async function desktopAgentRequest<T>(path: string, body?: object): Promise<T> {
  // The desktop shell chooses a free port at runtime. Start is idempotent, so
  // this also removes the initial-render race between the UI and the agent.
  await invokeDesktop("start_compute_agent");
  let lastError: unknown;
  for (let attempt = 0; attempt < 15; attempt += 1) {
    try {
      return await invokeDesktop<T>("agent_request", { path, body: body ?? null });
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  throw lastError instanceof Error ? lastError : new Error(`Desktop agent request failed: ${path}`);
}

export async function desktopAgentGet<T>(path: string): Promise<T> {
  return desktopAgentRequest<T>(path);
}

export async function desktopAgentPost<T>(path: string, body: object): Promise<T> {
  return desktopAgentRequest<T>(path, body);
}
