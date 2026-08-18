"use client";

import React from "react";
import { CircuitEditor } from "../src/components/CircuitEditor";
import { DesignSummary, ResearchWorkspace } from "../src/components/ResearchWorkspace";
import { CircuitProvider } from "../src/state/circuitStore";
import { UIContextProvider } from "../src/state/uiContext";
import { invokeDesktop, isDesktop } from "../src/lib/desktop";

type Tab = "design" | "run" | "results";

function App() {
  const [tab, setTab] = React.useState<Tab>("design");
  React.useEffect(() => { if (isDesktop()) invokeDesktop("start_compute_agent").catch(() => undefined); }, []);
  return <div style={{ height: "100vh", display: "grid", gridTemplateRows: "52px 1fr", background: "#f8fafc", color: "#0f172a", overflow: "hidden" }}>
    <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 20px", background: "#fff", borderBottom: "1px solid #e2e8f0" }}>
      <div style={{ fontWeight: 700, letterSpacing: "-0.02em" }}>Quantum Circuit</div>
      <nav style={{ display: "flex", gap: 4 }}>{(["design", "run", "results"] as Tab[]).map((item) => <button key={item} onClick={() => setTab(item)} style={{ border: 0, borderRadius: 6, padding: "7px 11px", textTransform: "capitalize", background: tab === item ? "#e2e8f0" : "transparent", color: "#0f172a" }}>{item}</button>)}</nav>
      <div style={{ fontSize: 12, color: "#64748b" }}>Local workspace</div>
    </header>
    {tab === "design" ? <div style={{ minHeight: 0, display: "flex", background: "#fff" }}><div style={{ minWidth: 0, flex: 1 }}><CircuitEditor /></div><DesignSummary /></div> : <div style={{ minHeight: 0, overflow: "auto" }}><ResearchWorkspace mode={tab} /></div>}
  </div>;
}

export default function Page() { return <CircuitProvider><UIContextProvider><App /></UIContextProvider></CircuitProvider>; }
