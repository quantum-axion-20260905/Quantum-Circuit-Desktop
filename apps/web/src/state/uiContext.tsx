"use client";

import React, { createContext, useContext } from "react";
import type { AgentResult } from "../lib/agent";
import { createExperimentId, HISTORY_EVENT, loadExperimentHistory, saveExperimentHistory, type ExperimentRecord } from "../lib/runHistory";
import { syncExperimentRecord } from "../lib/projectStore";

type UIContextState = {
  selectedRunSummary: Record<string, unknown> | null;
  setSelectedRunSummary: React.Dispatch<React.SetStateAction<Record<string, unknown> | null>>;
  latestOutput: AgentResult | null;
  setLatestOutput: React.Dispatch<React.SetStateAction<AgentResult | null>>;
  notesText: string;
  setNotesText: React.Dispatch<React.SetStateAction<string>>;
  experimentHistory: ExperimentRecord[];
  addExperiment: (record: Omit<ExperimentRecord, "id" | "createdAt">) => void;
  clearExperimentHistory: () => void;
};

const UIContext = createContext<UIContextState | null>(null);
const EMPTY_EXPERIMENT_HISTORY: ExperimentRecord[] = [];

export function UIContextProvider({ children }: { children: React.ReactNode }) {
  const [selectedRunSummary, setSelectedRunSummary] = React.useState<Record<string, unknown> | null>(null);
  const [latestOutput, setLatestOutput] = React.useState<AgentResult | null>(null);
  const [notesText, setNotesText] = React.useState<string>("");
  const experimentHistory = React.useSyncExternalStore(
    (onStoreChange) => {
      window.addEventListener(HISTORY_EVENT, onStoreChange);
      window.addEventListener("storage", onStoreChange);
      return () => {
        window.removeEventListener(HISTORY_EVENT, onStoreChange);
        window.removeEventListener("storage", onStoreChange);
      };
    },
    loadExperimentHistory,
    () => EMPTY_EXPERIMENT_HISTORY,
  );

  const addExperiment = React.useCallback((record: Omit<ExperimentRecord, "id" | "createdAt">) => {
    const nextRecord: ExperimentRecord = { ...record, id: createExperimentId(), createdAt: new Date().toISOString() };
    saveExperimentHistory([nextRecord, ...loadExperimentHistory()]);
    void syncExperimentRecord(nextRecord).catch(() => undefined);
  }, []);
  const clearExperimentHistory = React.useCallback(() => saveExperimentHistory([]), []);

  return (
    <UIContext.Provider
      value={{
        selectedRunSummary,
        setSelectedRunSummary,
        latestOutput,
        setLatestOutput,
        notesText,
        setNotesText,
        experimentHistory,
        addExperiment,
        clearExperimentHistory
      }}
    >
      {children}
    </UIContext.Provider>
  );
}

export function useUIContext() {
  const v = useContext(UIContext);
  if (!v) throw new Error("useUIContext must be used within UIContextProvider");
  return v;
}

