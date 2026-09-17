"use client";

import React from "react";
import { CircuitEditor } from "../src/components/CircuitEditor";
import { LatticeLab } from "../src/components/LatticeLab";
import { DesignSummary, ResearchWorkspace } from "../src/components/ResearchWorkspace";
import { CircuitProvider } from "../src/state/circuitStore";
import { UIContextProvider } from "../src/state/uiContext";
import { invokeDesktop, isDesktop } from "../src/lib/desktop";
import { AgentStatus } from "../src/components/AgentStatus";

type Tab = "design" | "lattice" | "run" | "results";

function App() {
  const [tab, setTab] = React.useState<Tab>("design");
  React.useEffect(() => { if (isDesktop()) invokeDesktop("start_compute_agent").catch(() => undefined); }, []);
  return <div className="qc-app-shell">
    <header className="qc-topbar">
      <div className="qc-brand">Quantum Circuit</div>
      <nav className="qc-nav" aria-label="Primary navigation">{(["design", "lattice", "run", "results"] as Tab[]).map((item) => <button key={item} onClick={() => setTab(item)} className="qc-nav-button" aria-current={tab === item ? "page" : undefined}>{item === "lattice" ? "Physics Lab" : item}</button>)}</nav>
      <AgentStatus />
    </header>
    {tab === "design" ? <div style={{ minHeight: 0, display: "flex", background: "#fff" }}><div style={{ minWidth: 0, flex: 1 }}><CircuitEditor /></div><DesignSummary /></div> : tab === "lattice" ? <div style={{ minHeight: 0, overflow: "auto", background: "#0f172a" }}><LatticeLab /></div> : <div style={{ minHeight: 0, overflow: "auto" }}><ResearchWorkspace mode={tab} /></div>}
  </div>;
}

export default function Page() { return <CircuitProvider><UIContextProvider><App /></UIContextProvider></CircuitProvider>; }
