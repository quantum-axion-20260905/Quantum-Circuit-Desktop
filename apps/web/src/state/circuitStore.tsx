"use client";

import React, { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

export type GateName = "h" | "x" | "rx" | "ry" | "rz" | "cx" | "cz";

export type GateOp = {
  id: string;
  name: GateName;
  target: number;
  control?: number;
  theta?: number;
  parameter?: string;
  col: number;
  x: number;
  y: number;
};

export type CircuitAnalysis = {
  source: "sample" | "tn_amplitudes" | "tn_estimate" | "unknown";
  attachedAt: string;
  gateScores: Record<string, number>;
  counts?: Record<string, number>;
  amplitudes?: Array<{ bitstring: string; re: number; im: number; mag: number }>;
  complexity?: {
    pathSteps?: number;
    optCost?: string | number;
    largestIntermediate?: string | number;
    speedup?: string | number;
  };
};

type CircuitState = {
  nQubits: number;
  setNQubits: (n: number) => void;
  ops: GateOp[];
  setOps: React.Dispatch<React.SetStateAction<GateOp[]>>;
  replaceCircuit: (nQubits: number, ops: GateOp[]) => void;
  clear: () => void;
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  analysis: CircuitAnalysis | null;
  setAnalysis: React.Dispatch<React.SetStateAction<CircuitAnalysis | null>>;
};

const CircuitCtx = createContext<CircuitState | null>(null);

export function CircuitProvider({ children }: { children: ReactNode }) {
  const [present, setPresent] = useState<{ nQubits: number; ops: GateOp[] }>({
    nQubits: 4,
    ops: [
      { id: "g0", name: "h", target: 0, col: 0, x: 120, y: 60 },
      { id: "g1", name: "cx", target: 1, control: 0, col: 1, x: 260, y: 120 }
    ]
  });

  const [past, setPast] = useState<Array<{ nQubits: number; ops: GateOp[] }>>([]);
  const [future, setFuture] = useState<Array<{ nQubits: number; ops: GateOp[] }>>([]);
  const [analysis, setAnalysis] = useState<CircuitAnalysis | null>(null);

  const push = useCallback((next: { nQubits: number; ops: GateOp[] }) => {
    setPast((p) => [...p, present]);
    setFuture([]);
    setPresent(next);
  }, [present]);

  const setOps: React.Dispatch<React.SetStateAction<GateOp[]>> = useCallback((updater) => {
    const nextOps = typeof updater === "function" ? updater(present.ops) : updater;
    push({ nQubits: present.nQubits, ops: nextOps });
  }, [present, push]);

  const setNQubits = useCallback((n: number) => {
    const next = Math.max(1, Math.min(64, Math.floor(n)));
    if (next === present.nQubits) return;
    // Removing qubits must not silently turn a two-qubit gate into an invalid
    // same-control/same-target operation. Gates that reference removed lanes
    // are dropped; surviving gates keep their exact qubit indices.
    const nextOps = present.ops
      .filter((g) => {
        if (g.target >= next || (g.control !== undefined && g.control >= next)) return false;
        return g.control === undefined || g.control !== g.target;
      })
      .map((g) => ({ ...g, y: 60 + g.target * 60 }));
    push({ nQubits: next, ops: nextOps });
  }, [present, push]);

  const replaceCircuit = useCallback((nextNQubits: number, nextOps: GateOp[]) => {
    const bounded = Math.max(1, Math.min(64, Math.floor(nextNQubits)));
    setPresent({ nQubits: bounded, ops: nextOps });
    setPast([]);
    setFuture([]);
    setAnalysis(null);
  }, []);

  const clear = useCallback(() => {
    push({ nQubits: present.nQubits, ops: [] });
  }, [present.nQubits, push]);

  const undo = useCallback(() => {
    if (past.length === 0) return;
    const prev = past[past.length - 1];
    setFuture((f) => [present, ...f]);
    setPast((p) => p.slice(0, -1));
    setPresent(prev);
  }, [past, present]);

  const redo = useCallback(() => {
    if (future.length === 0) return;
    const next = future[0];
    setPast((p) => [...p, present]);
    setFuture((f) => f.slice(1));
    setPresent(next);
  }, [future, present]);

  const { nQubits, ops } = present;

  const value = useMemo<CircuitState>(
    () => ({
      nQubits,
      setNQubits,
      ops,
      setOps,
      replaceCircuit,
      clear,
      undo,
      redo,
      canUndo: past.length > 0,
      canRedo: future.length > 0,
      analysis,
      setAnalysis
    }),
    [nQubits, ops, setNQubits, setOps, replaceCircuit, clear, undo, redo, past.length, future.length, analysis]
  );

  return <CircuitCtx.Provider value={value}>{children}</CircuitCtx.Provider>;
}

export function useCircuit() {
  const v = useContext(CircuitCtx);
  if (!v) throw new Error("useCircuit must be used within CircuitProvider");
  return v;
}
