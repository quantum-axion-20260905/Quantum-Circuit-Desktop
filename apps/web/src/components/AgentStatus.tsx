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
  const [showHelp, setShowHelp] = React.useState(false);
  const [copied, setCopied] = React.useState(false);

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

  const copyCommand = () => {
    navigator.clipboard.writeText("python agent/app.py").then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const jobs = capabilities?.queue?.jobs ?? {};
  const active = numberValue(jobs.queued) + numberValue(jobs.running);
  const gpuAvailable = capabilities?.gpu?.available === true;
  const stateLabel = state === "loading" ? "Checking agent" : state === "online" ? "Agent online" : "Agent offline";
  const stateClass = state === "online" ? "online" : state === "offline" ? "offline" : "loading";

  return (
    <div style={{ position: "relative" }}>
      <div
        className="qc-agent-status"
        onClick={() => state === "offline" && setShowHelp((v) => !v)}
        style={{ cursor: state === "offline" ? "pointer" : "default" }}
        title={state === "offline" ? "Agentni yoqish bo'yicha yo'riqnoma uchun bosing" : "Local compute status"}
      >
        <span className={`qc-agent-state qc-agent-state-${stateClass}`}>
          <span aria-hidden="true" className={`qc-agent-dot qc-agent-dot-${stateClass}`} />
          {stateLabel}
          {state === "offline" ? <span style={{ marginLeft: 4, opacity: 0.8, fontSize: 11 }}>❓</span> : null}
        </span>
        {state === "online" ? (
          <>
            <span aria-hidden="true" className="qc-status-separator">·</span>
            <span>{gpuAvailable ? "GPU ready" : "GPU unavailable (CPU mode)"}</span>
            {active > 0 ? (
              <>
                <span aria-hidden="true" className="qc-status-separator">·</span>
                <span>{active} active</span>
              </>
            ) : null}
          </>
        ) : null}
      </div>

      {showHelp && state === "offline" ? (
        <div style={{
          position: "absolute",
          top: "100%",
          right: 0,
          marginTop: 8,
          width: 320,
          background: "#ffffff",
          border: "1px solid #cbd5e1",
          borderRadius: 8,
          boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1)",
          padding: 14,
          zIndex: 100,
          fontSize: 13,
          color: "#1e293b",
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
            <span style={{ fontWeight: 600, color: "#b91c1c" }}>⚠️ Agent ulanmagan</span>
            <button
              onClick={() => setShowHelp(false)}
              style={{ background: "none", border: "none", cursor: "pointer", fontSize: 16, color: "#64748b" }}
            >
              ✕
            </button>
          </div>
          <p style={{ margin: "0 0 8px 0", color: "#475569", lineHeight: 1.4 }}>
            Hisoblash agenti (FastAPI) orqa fonda ishga tushirilishi kerak.
          </p>
          <div style={{
            background: "#0f172a",
            color: "#38bdf8",
            padding: "6px 10px",
            borderRadius: 6,
            fontFamily: "monospace",
            fontSize: 12,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 10,
          }}>
            <span>python agent/app.py</span>
            <button
              onClick={copyCommand}
              style={{
                background: "#334155",
                color: "#f8fafc",
                border: "none",
                borderRadius: 4,
                padding: "2px 8px",
                fontSize: 11,
                cursor: "pointer",
              }}
            >
              {copied ? "Nusxalandi!" : "Copy"}
            </button>
          </div>
          <button
            onClick={() => void refresh()}
            style={{
              width: "100%",
              padding: "6px 0",
              background: "#2563eb",
              color: "#ffffff",
              border: "none",
              borderRadius: 6,
              cursor: "pointer",
              fontWeight: 500,
            }}
          >
            Qayta tekshirish
          </button>
        </div>
      ) : null}
    </div>
  );
}
