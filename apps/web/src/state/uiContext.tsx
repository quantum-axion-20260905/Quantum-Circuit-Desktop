"use client";

import React, { createContext, useContext } from "react";

type UIContextState = {
  selectedRunSummary: any | null;
  setSelectedRunSummary: React.Dispatch<React.SetStateAction<any | null>>;
  latestOutput: any | null;
  setLatestOutput: React.Dispatch<React.SetStateAction<any | null>>;
  notesText: string;
  setNotesText: React.Dispatch<React.SetStateAction<string>>;
};

const UIContext = createContext<UIContextState | null>(null);

export function UIContextProvider({ children }: { children: React.ReactNode }) {
  const [selectedRunSummary, setSelectedRunSummary] = React.useState<any | null>(null);
  const [latestOutput, setLatestOutput] = React.useState<any | null>(null);
  const [notesText, setNotesText] = React.useState<string>("");

  return (
    <UIContext.Provider
      value={{
        selectedRunSummary,
        setSelectedRunSummary,
        latestOutput,
        setLatestOutput,
        notesText,
        setNotesText
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

