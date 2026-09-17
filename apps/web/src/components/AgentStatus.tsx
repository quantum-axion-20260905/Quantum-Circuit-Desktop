"use client";

import React from "react";
import { getCapabilities, type AgentCapabilities } from "../lib/agent";

type ConnectionState = "loading" | "online" | "offline";

function numberValue(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

export function AgentStatus() {
  const [state, setState] = React.useState<ConnectionState>("loading");
  const [capabilities, setCapabilities] = React.useState<AgentCapabilities | null>(null);

  const refresh = React.useCallback(async () => {
    try {
      const next = await getCapabilities();
      setCapabilities(next);
      setState("online");
    } catch {
      setCapabilities(null);
      setState("offline");
    }
  }, []);

  React.useEffect(() => {
    const initial = window.setTimeout(() => void refresh(), 0);
    const timer = window.setInterval(() => void refresh(), 5000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(timer);
    };
  }, [refresh]);

  const jobs = capabilities?.queue?.jobs ?? {};
  const active = numberValue(jobs.queued) + numberValue(jobs.running);
  const gpuAvailable = capabilities?.gpu?.available === true;
  const stateLabel = state === "loading" ? "Checking agent" : state === "online" ? "Agent online" : "Agent offline";
  const stateClass = state === "online" ? "online" : state === "offline" ? "offline" : "loading";

  return (
    <div className="qc-agent-status" title={state === "offline" ? "Start the local compute agent on port 8788" : "Local compute status"}>
      <span className={`qc-agent-state qc-agent-state-${stateClass}`}>
        <span aria-hidden="true" className={`qc-agent-dot qc-agent-dot-${stateClass}`} />
        {stateLabel}
      </span>
      {state === "online" ? <>
        <span aria-hidden="true" className="qc-status-separator">·</span>
        <span>{gpuAvailable ? "GPU ready" : "GPU unavailable"}</span>
        {active > 0 ? <><span aria-hidden="true" className="qc-status-separator">·</span><span>{active} active</span></> : null}
      </> : null}
    </div>
  );
}
