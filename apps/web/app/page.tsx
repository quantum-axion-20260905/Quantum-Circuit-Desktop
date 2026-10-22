"use client";

import React from "react";
import { CircuitEditor } from "../src/components/CircuitEditor";
import { PhysicsLab } from "../src/components/PhysicsLab";
import { MaterialsLab } from "../src/components/MaterialsLab";
import { DesignSummary, ResearchWorkspace } from "../src/components/ResearchWorkspace";
import { CircuitProvider } from "../src/state/circuitStore";
import { UIContextProvider } from "../src/state/uiContext";
import { invokeDesktop, isDesktop } from "../src/lib/desktop";
import { AgentStatus } from "../src/components/AgentStatus";

type Tab = "design" | "physics" | "materials" | "run" | "results";

function App() {
  const [tab, setTab] = React.useState<Tab>("physics");
  React.useEffect(() => {
    if (isDesktop()) invokeDesktop("start_compute_agent").catch(() => undefined);
  }, []);

  const tabLabels: Record<Tab, string> = {
    design: "Circuit Studio",
    physics: "Many-Body Physics",
    materials: "Materials Science",
    run: "Simulator & Run",
    results: "History & Results",
  };

  return (
    <div className="qc-app-shell">
      <header className="qc-topbar">
        <div className="qc-brand" style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 18 }}>⚛</span>
          <span style={{ fontWeight: 800, letterSpacing: "-.02em" }}>Quantum Circuit Workbench</span>
        </div>
        <nav className="qc-nav" aria-label="Primary navigation">
          {(["design", "physics", "materials", "run", "results"] as Tab[]).map((item) => (
            <button
              key={item}
              onClick={() => setTab(item)}
              className="qc-nav-button"
              aria-current={tab === item ? "page" : undefined}
              style={{ fontWeight: tab === item ? 700 : 500 }}
            >
              {tabLabels[item]}
            </button>
          ))}
        </nav>
        <AgentStatus />
      </header>

      {tab === "design" ? (
        <div className="qc-design-layout">
          <div style={{ minWidth: 0, flex: 1, height: "100%" }}>
            <CircuitEditor />
          </div>
          <DesignSummary />
        </div>
      ) : tab === "physics" ? (
        <div style={{ minHeight: 0, height: "100%", width: "100%", overflow: "auto", background: "#0b0f19" }}>
          <PhysicsLab />
        </div>
      ) : tab === "materials" ? (
        <div style={{ minHeight: 0, height: "100%", width: "100%", overflow: "auto", background: "#0b0f19" }}>
          <MaterialsLab />
        </div>
      ) : (
        <div style={{ minHeight: 0, height: "100%", width: "100%", overflow: "auto" }}>
          <ResearchWorkspace mode={tab} />
        </div>
      )}
    </div>
  );
}

export default function Page() {
  return (
    <CircuitProvider>
      <UIContextProvider>
        <App />
      </UIContextProvider>
    </CircuitProvider>
  );
}
