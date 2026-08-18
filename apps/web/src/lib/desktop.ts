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
