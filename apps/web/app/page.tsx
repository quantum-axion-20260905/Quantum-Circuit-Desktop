"use client";

import React from "react";
import { CircuitEditor } from "../src/components/CircuitEditor";
import { ExplainPanel } from "../src/components/ExplainPanel";
import { CircuitProvider } from "../src/state/circuitStore";
import { DesignInspectorPanel, DesignWorkbenchPanel } from "../src/components/panels/DesignPanels";
import { UIContextProvider } from "../src/state/uiContext";
import { ComparePanel, MetricsPanel, NotesPanel, ResultsInspectorPanel, RunInspectorPanel, WarningsPanel } from "../src/components/panels/WorkbenchPanels";

type MainTab = "design" | "run" | "results" | "theory" | "code" | "experiments";
type BottomTab = "workbench" | "metrics" | "warnings" | "compare" | "notes";
type RightTab = "inspector";

type PageLayout = {
  showRight: boolean;
  showBottom: boolean;
  bottomTabs: BottomTab[];
  defaultBottom: BottomTab;
};

const topTabs: Array<{ id: MainTab; label: string }> = [
  { id: "design", label: "Design" },
  { id: "run", label: "Run" },
  { id: "results", label: "Results" },
  { id: "theory", label: "Theory" },
  { id: "code", label: "Code" },
  { id: "experiments", label: "Experiments" }
];

export default function Page() {
  const [mainTab, setMainTab] = React.useState<MainTab>("design");
  const [rightTabByPage, setRightTabByPage] = React.useState<Record<MainTab, RightTab>>({
    design: "inspector",
    run: "inspector",
    results: "inspector",
    theory: "inspector",
    code: "inspector",
    experiments: "inspector"
  });
  const [bottomTabByPage, setBottomTabByPage] = React.useState<Record<MainTab, BottomTab>>({
    design: "workbench",
    run: "workbench",
    results: "compare",
    theory: "notes",
    code: "notes",
    experiments: "workbench"
  });
  const [rightWidth, setRightWidth] = React.useState<number>(420);
  const [bottomHeight, setBottomHeight] = React.useState<number>(280);

  const dragState = React.useRef<{ type: "right" | "bottom"; startX: number; startY: number; startRightWidth: number; startBottomHeight: number } | null>(null);

  React.useEffect(() => {
    function onMove(e: MouseEvent) {
      const d = dragState.current;
      if (!d) return;
      if (d.type === "right") {
        const delta = d.startX - e.clientX;
        setRightWidth(Math.max(320, Math.min(620, d.startRightWidth + delta)));
      } else {
        const delta = d.startY - e.clientY;
        setBottomHeight(Math.max(180, Math.min(520, d.startBottomHeight + delta)));
      }
    }
    function onUp() {
      dragState.current = null;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  function startDrag(type: "right" | "bottom", ev: React.MouseEvent) {
    dragState.current = {
      type,
      startX: ev.clientX,
      startY: ev.clientY,
      startRightWidth: rightWidth,
      startBottomHeight: bottomHeight
    };
    document.body.style.cursor = type === "right" ? "col-resize" : "row-resize";
    document.body.style.userSelect = "none";
  }

  function TabButton(props: { active: boolean; label: string; onClick: () => void }) {
    return (
      <button
        onClick={props.onClick}
        style={{
          border: "1px solid #d1d5db",
          borderBottom: props.active ? "2px solid #111827" : "1px solid #d1d5db",
          background: props.active ? "#ffffff" : "#f9fafb",
          padding: "6px 10px",
          fontSize: 12
        }}
      >
        {props.label}
      </button>
    );
  }

  const pageLayout: Record<MainTab, PageLayout> = {
    design: { showRight: true, showBottom: true, bottomTabs: ["workbench", "metrics", "warnings", "notes"], defaultBottom: "workbench" },
    run: { showRight: true, showBottom: true, bottomTabs: ["workbench", "metrics", "warnings", "compare", "notes"], defaultBottom: "workbench" },
    results: { showRight: true, showBottom: true, bottomTabs: ["workbench", "metrics", "warnings", "compare", "notes"], defaultBottom: "compare" },
    experiments: { showRight: true, showBottom: true, bottomTabs: ["workbench", "metrics", "warnings", "compare", "notes"], defaultBottom: "workbench" },
    theory: { showRight: true, showBottom: false, bottomTabs: ["notes"], defaultBottom: "notes" },
    code: { showRight: false, showBottom: true, bottomTabs: ["notes"], defaultBottom: "notes" }
  };

  const activeLayout = pageLayout[mainTab];
  const rightTab = rightTabByPage[mainTab] ?? "inspector";
  const rawBottomTab = bottomTabByPage[mainTab] ?? activeLayout.defaultBottom;
  const bottomTab = activeLayout.bottomTabs.includes(rawBottomTab) ? rawBottomTab : activeLayout.defaultBottom;

  const setRightTab = (tab: RightTab) => setRightTabByPage((prev) => ({ ...prev, [mainTab]: tab }));
  const setBottomTab = (tab: BottomTab) => setBottomTabByPage((prev) => ({ ...prev, [mainTab]: tab }));

  return (
    <CircuitProvider>
      <UIContextProvider>
      <div style={{ height: "100vh", display: "grid", gridTemplateRows: "42px 1fr", background: "#f3f4f6", overflow: "hidden" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 10px", borderBottom: "1px solid #d1d5db", background: "#ffffff", overflowX: "auto", whiteSpace: "nowrap" }}>
          {topTabs.map((t) => (
            <TabButton key={t.id} active={mainTab === t.id} label={t.label} onClick={() => setMainTab(t.id)} />
          ))}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: activeLayout.showRight ? `minmax(520px, 1fr) 6px ${rightWidth}px` : "1fr", overflow: "hidden" }}>
          <div style={{ display: "grid", gridTemplateRows: activeLayout.showBottom ? `minmax(240px, 1fr) 6px ${bottomHeight}px` : "1fr", overflow: "hidden", borderRight: activeLayout.showRight ? "1px solid #d1d5db" : "none", background: "#ffffff" }}>
            <div style={{ minHeight: 0, overflow: "hidden" }}>
              {mainTab === "design" ? (
                <CircuitEditor />
              ) : mainTab === "run" ? (
                <div style={{ padding: 14, fontSize: 13 }}>
                  <strong>Run</strong>
                  <p style={{ marginTop: 8 }}>Use the bottom Workbench for run controls, queue, and execution.</p>
                </div>
              ) : mainTab === "results" ? (
                <div style={{ padding: 14, fontSize: 13 }}>
                  <strong>Results</strong>
                  <p style={{ marginTop: 8 }}>Use the bottom Workbench for validation, compare metrics, and exports.</p>
                </div>
              ) : mainTab === "experiments" ? (
                <div style={{ padding: 14, fontSize: 13 }}>
                  <strong>Experiments</strong>
                  <p style={{ marginTop: 8 }}>Use the bottom Workbench for queue, sweeps, and reproducible batches.</p>
                </div>
              ) : (
                <div style={{ padding: 14, fontSize: 13 }}>
                  <strong>{topTabs.find((x) => x.id === mainTab)?.label}</strong>
                  <p style={{ marginTop: 8 }}>This tab is scaffolded. Existing controls remain in Inspector/Workbench until migration completes.</p>
                </div>
              )}
            </div>

            {activeLayout.showBottom ? <div onMouseDown={(e) => startDrag("bottom", e)} style={{ cursor: "row-resize", background: "#e5e7eb" }} /> : null}

            {activeLayout.showBottom ? <div style={{ minHeight: 0, overflow: "hidden", borderTop: "1px solid #d1d5db", background: "#ffffff", display: "grid", gridTemplateRows: "34px 1fr" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "4px 8px", borderBottom: "1px solid #e5e7eb", overflowX: "auto", whiteSpace: "nowrap" }}>
                {activeLayout.bottomTabs.includes("workbench") ? <TabButton active={bottomTab === "workbench"} label="Workbench" onClick={() => setBottomTab("workbench")} /> : null}
                {activeLayout.bottomTabs.includes("metrics") ? <TabButton active={bottomTab === "metrics"} label="Metrics" onClick={() => setBottomTab("metrics")} /> : null}
                {activeLayout.bottomTabs.includes("warnings") ? <TabButton active={bottomTab === "warnings"} label="Warnings" onClick={() => setBottomTab("warnings")} /> : null}
                {activeLayout.bottomTabs.includes("compare") ? <TabButton active={bottomTab === "compare"} label="Compare" onClick={() => setBottomTab("compare")} /> : null}
                {activeLayout.bottomTabs.includes("notes") ? <TabButton active={bottomTab === "notes"} label="Notes" onClick={() => setBottomTab("notes")} /> : null}
              </div>
              <div style={{ minHeight: 0, overflow: "auto" }}>
                {bottomTab === "workbench" && mainTab === "design" ? (
                  <DesignWorkbenchPanel />
                ) : bottomTab === "workbench" && (mainTab === "run" || mainTab === "results" || mainTab === "experiments") ? (
                  <ExplainPanel mode={mainTab} />
                ) : bottomTab === "metrics" ? (
                  <MetricsPanel />
                ) : bottomTab === "warnings" ? (
                  <WarningsPanel />
                ) : bottomTab === "compare" ? (
                  <ComparePanel />
                ) : bottomTab === "notes" ? (
                  <NotesPanel />
                ) : (
                  <div style={{ padding: 12, fontSize: 13 }}>
                    This panel is reserved for {bottomTab}.
                    <div style={{ marginTop: 8 }}>
                      Active page: <strong>{topTabs.find((x) => x.id === mainTab)?.label}</strong>
                    </div>
                  </div>
                )}
              </div>
            </div> : null}
          </div>

          {activeLayout.showRight ? <div onMouseDown={(e) => startDrag("right", e)} style={{ cursor: "col-resize", background: "#e5e7eb" }} /> : null}

          {activeLayout.showRight ? <div style={{ minWidth: 0, overflow: "hidden", background: "#ffffff", display: "grid", gridTemplateRows: "34px 1fr" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "4px 8px", borderBottom: "1px solid #e5e7eb", overflowX: "auto", whiteSpace: "nowrap" }}>
              <TabButton active={rightTab === "inspector"} label="Inspector" onClick={() => setRightTab("inspector")} />
            </div>
            <div style={{ minHeight: 0, overflow: "auto" }}>
              {mainTab === "design" ? <DesignInspectorPanel /> :
              mainTab === "run" ? <RunInspectorPanel /> :
              mainTab === "results" ? <ResultsInspectorPanel /> :
              <div style={{ padding: 12, fontSize: 13 }}>
                <strong>Inspector</strong>
                <p style={{ marginTop: 8 }}>
                  Active page: <strong>{topTabs.find((x) => x.id === mainTab)?.label}</strong>
                </p>
              </div>}
            </div>
          </div> : null}
        </div>
      </div>
      </UIContextProvider>
    </CircuitProvider>
  );
}
