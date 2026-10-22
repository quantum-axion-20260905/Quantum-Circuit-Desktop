"use client";

import React, { useState, useMemo, useCallback } from "react";
import { useCircuit, type GateOp } from "../state/circuitStore";
import { useUIContext } from "../state/uiContext";
import { Button } from "../ui";
import {
  PotentialEnergySurfaceChart,
  MolecularOrbitalsDiagram,
  VQEOptimizationTrace,
  type PESPoint,
  type OrbitalLevel,
} from "./ChemistryVisuals";
import { Lattice3DView, type Site3D, type Bond3D } from "./Lattice3DView";

export type MoleculePreset = "H2" | "LiH" | "BeH2" | "H2O" | "N2";
export type QubitMapping = "jordan_wigner" | "bravyi_kitaev" | "parity";
export type VQEAnsatz = "hardware_efficient" | "uccsd" | "ry_rz";
export type OptimizerType = "COBYLA" | "SLSQP" | "Adam";
export type ChemistryTab = "pes" | "orbitals" | "vqe" | "structure3d" | "export";

// Molecule definition specifications
interface MoleculeConfig {
  name: string;
  formula: string;
  defaultR: number; // in Angstrom
  equilibriumR: number;
  dissociationEnergy: number; // in Hartree
  activeQubits: number;
  homoLumoGap: number; // in eV
  sites: Site3D[];
  bonds: Bond3D[];
  orbitals: OrbitalLevel[];
  pesCurve: PESPoint[];
}

const MOLECULES: Record<MoleculePreset, MoleculeConfig> = {
  H2: {
    name: "Hydrogen Molecule",
    formula: "H₂",
    defaultR: 0.74,
    equilibriumR: 0.7414,
    dissociationEnergy: 0.174,
    activeQubits: 2, // With parity reduction (4 without)
    homoLumoGap: 11.2,
    sites: [
      { id: 0, x: -0.37, y: 0, z: 0, element: "H", label: "H1", radius: 12 },
      { id: 1, x: 0.37, y: 0, z: 0, element: "H", label: "H2", radius: 12 },
    ],
    bonds: [{ source: 0, target: 1, type: "single" }],
    orbitals: [
      { id: "1", name: "1σ_g (B)", energy: -15.4, occupancy: 2 },
      { id: "2", name: "1σ_u* (AB)", energy: -4.2, occupancy: 0 },
    ],
    pesCurve: [
      { r: 0.3, hfEnergy: -0.601, fciEnergy: -0.603, vqeEnergy: -0.602 },
      { r: 0.5, hfEnergy: -1.055, fciEnergy: -1.065, vqeEnergy: -1.064 },
      { r: 0.74, hfEnergy: -1.117, fciEnergy: -1.137, vqeEnergy: -1.136 },
      { r: 0.9, hfEnergy: -1.096, fciEnergy: -1.121, vqeEnergy: -1.120 },
      { r: 1.2, hfEnergy: -1.021, fciEnergy: -1.069, vqeEnergy: -1.068 },
      { r: 1.5, hfEnergy: -0.945, fciEnergy: -1.025, vqeEnergy: -1.024 },
      { r: 2.0, hfEnergy: -0.852, fciEnergy: -1.002, vqeEnergy: -1.001 },
      { r: 2.5, hfEnergy: -0.795, fciEnergy: -0.999, vqeEnergy: -0.998 },
    ],
  },
  LiH: {
    name: "Lithium Hydride",
    formula: "LiH",
    defaultR: 1.595,
    equilibriumR: 1.595,
    dissociationEnergy: 0.092,
    activeQubits: 4,
    homoLumoGap: 4.8,
    sites: [
      { id: 0, x: -0.8, y: 0, z: 0, element: "Li", label: "Li", radius: 18 },
      { id: 1, x: 0.8, y: 0, z: 0, element: "H", label: "H", radius: 12 },
    ],
    bonds: [{ source: 0, target: 1, type: "single" }],
    orbitals: [
      { id: "1", name: "1σ (1s Li)", energy: -64.2, occupancy: 2 },
      { id: "2", name: "2σ (Bonding)", energy: -7.9, occupancy: 2 },
      { id: "3", name: "1π_x (Non-bond)", energy: -3.1, occupancy: 0 },
      { id: "4", name: "1π_y (Non-bond)", energy: -3.1, occupancy: 0 },
      { id: "5", name: "3σ* (Anti-bond)", energy: 1.2, occupancy: 0 },
    ],
    pesCurve: [
      { r: 1.0, hfEnergy: -7.82, fciEnergy: -7.85, vqeEnergy: -7.848 },
      { r: 1.3, hfEnergy: -7.94, fciEnergy: -7.97, vqeEnergy: -7.969 },
      { r: 1.595, hfEnergy: -7.987, fciEnergy: -8.070, vqeEnergy: -8.068 },
      { r: 2.0, hfEnergy: -7.95, fciEnergy: -8.04, vqeEnergy: -8.038 },
      { r: 2.5, hfEnergy: -7.88, fciEnergy: -8.01, vqeEnergy: -8.008 },
      { r: 3.0, hfEnergy: -7.81, fciEnergy: -7.98, vqeEnergy: -7.978 },
    ],
  },
  BeH2: {
    name: "Beryllium Hydride",
    formula: "BeH₂",
    defaultR: 1.326,
    equilibriumR: 1.326,
    dissociationEnergy: 0.22,
    activeQubits: 6,
    homoLumoGap: 6.4,
    sites: [
      { id: 0, x: -1.326, y: 0, z: 0, element: "H", label: "H1", radius: 12 },
      { id: 1, x: 0, y: 0, z: 0, element: "Be", label: "Be", radius: 17 },
      { id: 2, x: 1.326, y: 0, z: 0, element: "H", label: "H2", radius: 12 },
    ],
    bonds: [
      { source: 0, target: 1, type: "single" },
      { source: 1, target: 2, type: "single" },
    ],
    orbitals: [
      { id: "1", name: "1σ_g (1s Be)", energy: -126.5, occupancy: 2 },
      { id: "2", name: "2σ_g (Bonding)", energy: -13.2, occupancy: 2 },
      { id: "3", name: "1σ_u (Bonding)", energy: -10.8, occupancy: 2 },
      { id: "4", name: "1π_u (Non-bond)", energy: -4.4, occupancy: 0 },
      { id: "5", name: "3σ_g* (Anti-bond)", energy: 2.1, occupancy: 0 },
    ],
    pesCurve: [
      { r: 0.9, hfEnergy: -15.2, fciEnergy: -15.35 },
      { r: 1.1, hfEnergy: -15.65, fciEnergy: -15.82 },
      { r: 1.326, hfEnergy: -15.773, fciEnergy: -15.980, vqeEnergy: -15.978 },
      { r: 1.6, hfEnergy: -15.71, fciEnergy: -15.94 },
      { r: 2.0, hfEnergy: -15.55, fciEnergy: -15.86 },
      { r: 2.6, hfEnergy: -15.32, fciEnergy: -15.75 },
    ],
  },
  H2O: {
    name: "Water Molecule",
    formula: "H₂O",
    defaultR: 0.957,
    equilibriumR: 0.957,
    dissociationEnergy: 0.36,
    activeQubits: 8,
    homoLumoGap: 7.3,
    sites: [
      { id: 0, x: 0, y: 0, z: 0, element: "O", label: "O", radius: 17 },
      { id: 1, x: 0.957 * Math.sin(0.91), y: 0.957 * Math.cos(0.91), z: 0, element: "H", label: "H1", radius: 12 },
      { id: 2, x: -0.957 * Math.sin(0.91), y: 0.957 * Math.cos(0.91), z: 0, element: "H", label: "H2", radius: 12 },
    ],
    bonds: [
      { source: 0, target: 1, type: "single" },
      { source: 0, target: 2, type: "single" },
    ],
    orbitals: [
      { id: "1", name: "1a_1 (1s O)", energy: -538.0, occupancy: 2 },
      { id: "2", name: "2a_1 (O-H σ)", energy: -32.2, occupancy: 2 },
      { id: "3", name: "1b_2 (O-H σ)", energy: -18.5, occupancy: 2 },
      { id: "4", name: "3a_1 (Lone pair)", energy: -14.7, occupancy: 2 },
      { id: "5", name: "1b_1 (HOMO lone pair)", energy: -12.6, occupancy: 2 },
      { id: "6", name: "4a_1* (LUMO σ*)", energy: -5.3, occupancy: 0 },
      { id: "7", name: "2b_2* (σ*)", energy: 1.8, occupancy: 0 },
    ],
    pesCurve: [
      { r: 0.7, hfEnergy: -75.1, fciEnergy: -75.32 },
      { r: 0.85, hfEnergy: -75.95, fciEnergy: -76.22 },
      { r: 0.957, hfEnergy: -76.067, fciEnergy: -76.438, vqeEnergy: -76.435 },
      { r: 1.2, hfEnergy: -75.89, fciEnergy: -76.31 },
      { r: 1.5, hfEnergy: -75.52, fciEnergy: -76.08 },
      { r: 2.0, hfEnergy: -75.1, fciEnergy: -75.85 },
    ],
  },
  N2: {
    name: "Nitrogen Molecule",
    formula: "N₂",
    defaultR: 1.098,
    equilibriumR: 1.098,
    dissociationEnergy: 0.364,
    activeQubits: 8,
    homoLumoGap: 8.7,
    sites: [
      { id: 0, x: -0.55, y: 0, z: 0, element: "N", label: "N1", radius: 16 },
      { id: 1, x: 0.55, y: 0, z: 0, element: "N", label: "N2", radius: 16 },
    ],
    bonds: [{ source: 0, target: 1, type: "triple" }],
    orbitals: [
      { id: "1", name: "1σ_g (1s)", energy: -410.0, occupancy: 2 },
      { id: "2", name: "1σ_u* (1s*)", energy: -410.0, occupancy: 2 },
      { id: "3", name: "2σ_g (2s)", energy: -37.5, occupancy: 2 },
      { id: "4", name: "2σ_u* (2s*)", energy: -19.2, occupancy: 2 },
      { id: "5", name: "1π_u (2px, 2py)", energy: -16.8, occupancy: 2 },
      { id: "6", name: "3σ_g (HOMO)", energy: -15.6, occupancy: 2 },
      { id: "7", name: "1π_g* (LUMO)", energy: -6.9, occupancy: 0 },
    ],
    pesCurve: [
      { r: 0.8, hfEnergy: -108.1, fciEnergy: -108.35 },
      { r: 0.95, hfEnergy: -108.82, fciEnergy: -109.25 },
      { r: 1.098, hfEnergy: -108.993, fciEnergy: -109.542, vqeEnergy: -109.539 },
      { r: 1.3, hfEnergy: -108.81, fciEnergy: -109.41 },
      { r: 1.6, hfEnergy: -108.45, fciEnergy: -109.22 },
      { r: 2.2, hfEnergy: -108.12, fciEnergy: -109.05 },
    ],
  },
};

const shell: React.CSSProperties = {
  maxWidth: "100%",
  width: "100%",
  margin: 0,
  padding: "16px 20px 48px",
  color: "#f1f5f9",
  background: "#0b0f19",
  boxSizing: "border-box",
  fontFamily: "inherit",
};

const card: React.CSSProperties = {
  border: "1px solid #334155",
  borderRadius: 8,
  background: "#111827",
  padding: 18,
  boxShadow: "0 1px 3px rgba(0,0,0,0.3)",
};

const fieldStyle: React.CSSProperties = {
  display: "grid",
  gap: 4,
  color: "#94a3b8",
  fontSize: 12,
  fontWeight: 500,
};

const inputStyle: React.CSSProperties = {
  width: "100%",
  border: "1px solid #475569",
  borderRadius: 4,
  background: "#1e293b",
  color: "#f8fafc",
  padding: "6px 8px",
  fontSize: 13,
  outline: "none",
  boxSizing: "border-box",
};

export function ChemistryLab() {
  const { replaceCircuit } = useCircuit();
  const { addExperiment, setActiveTab } = useUIContext();

  // Molecule & Geometry state
  const [moleculeKey, setMoleculeKey] = useState<MoleculePreset>("H2");
  const molecule = MOLECULES[moleculeKey];
  const [bondR, setBondR] = useState<number>(molecule.defaultR);

  // Basis & Mapping
  const [basisSet, setBasisSet] = useState<"STO-3G" | "6-31G">("STO-3G");
  const [mapping, setMapping] = useState<QubitMapping>("parity");
  const [ansatz, setAnsatz] = useState<VQEAnsatz>("hardware_efficient");
  const [optimizer, setOptimizer] = useState<OptimizerType>("COBYLA");
  const [maxIterations, setMaxIterations] = useState<number>(50);
  const [reps, setReps] = useState<number>(2);

  // Active sub-tab
  const [tab, setTab] = useState<ChemistryTab>("pes");

  // VQE Simulation State
  const [isSolving, setIsSolving] = useState(false);
  const [vqeHistory, setVqeHistory] = useState<number[]>([]);
  const [finalEnergy, setFinalEnergy] = useState<number | null>(null);
  const [fciEnergy, setFciEnergy] = useState<number>(molecule.pesCurve.find((p) => Math.abs(p.r - bondR) < 0.1)?.fciEnergy ?? molecule.pesCurve[2].fciEnergy);

  // Dynamic 3D sites based on current bond distance R
  const dynamicSites = useMemo(() => {
    if (moleculeKey === "H2") {
      return [
        { id: 0, x: -bondR / 2, y: 0, z: 0, element: "H", label: "H1", radius: 12 },
        { id: 1, x: bondR / 2, y: 0, z: 0, element: "H", label: "H2", radius: 12 },
      ];
    }
    if (moleculeKey === "LiH") {
      return [
        { id: 0, x: -bondR * 0.7, y: 0, z: 0, element: "Li", label: "Li", radius: 18 },
        { id: 1, x: bondR * 0.3, y: 0, z: 0, element: "H", label: "H", radius: 12 },
      ];
    }
    if (moleculeKey === "BeH2") {
      return [
        { id: 0, x: -bondR, y: 0, z: 0, element: "H", label: "H1", radius: 12 },
        { id: 1, x: 0, y: 0, z: 0, element: "Be", label: "Be", radius: 17 },
        { id: 2, x: bondR, y: 0, z: 0, element: "H", label: "H2", radius: 12 },
      ];
    }
    if (moleculeKey === "H2O") {
      const angle = 104.5 * (Math.PI / 180);
      return [
        { id: 0, x: 0, y: 0, z: 0, element: "O", label: "O", radius: 17 },
        { id: 1, x: bondR * Math.sin(angle / 2), y: bondR * Math.cos(angle / 2), z: 0, element: "H", label: "H1", radius: 12 },
        { id: 2, x: -bondR * Math.sin(angle / 2), y: bondR * Math.cos(angle / 2), z: 0, element: "H", label: "H2", radius: 12 },
      ];
    }
    return molecule.sites;
  }, [moleculeKey, bondR, molecule.sites]);

  // Handle molecule preset switch
  const handleSelectMolecule = (key: MoleculePreset) => {
    setMoleculeKey(key);
    const m = MOLECULES[key];
    setBondR(m.defaultR);
    const targetFci = m.pesCurve.find((p) => Math.abs(p.r - m.defaultR) < 0.15)?.fciEnergy ?? m.pesCurve[2].fciEnergy;
    setFciEnergy(targetFci);
    setVqeHistory([]);
    setFinalEnergy(null);
  };

  // Run VQE Optimization
  const handleRunVQE = useCallback(async () => {
    setIsSolving(true);
    setVqeHistory([]);

    // Determine target energy for current bond distance R
    const nearest = molecule.pesCurve.reduce((prev, curr) =>
      Math.abs(curr.r - bondR) < Math.abs(prev.r - bondR) ? curr : prev
    );
    const target = nearest.fciEnergy;
    setFciEnergy(target);

    // Simulated optimizer convergence trajectory
    const steps = maxIterations;
    const history: number[] = [];
    let current = nearest.hfEnergy;

    for (let i = 0; i < steps; i++) {
      const progress = (i + 1) / steps;
      // Exponential convergence towards FCI ground state
      const noise = (Math.random() - 0.5) * 0.005 * Math.exp(-progress * 3);
      current = target + (nearest.hfEnergy - target) * Math.exp(-progress * 4.2) + noise;
      history.push(current);
    }
    const finalVal = history[history.length - 1];

    await new Promise((r) => setTimeout(r, 450));
    setVqeHistory(history);
    setFinalEnergy(finalVal);
    setIsSolving(false);

    // Save to Experiment History
    addExperiment({
      label: `VQE · ${molecule.formula} (R=${bondR.toFixed(2)}Å, ${ansatz})`,
      source: "materials",
      kind: "ground_state",
      status: "done",
      request: {
        source: "materials",
        kind: "ground_state",
        payload: {
          molecule: moleculeKey,
          r: bondR,
          ansatz,
          mapping,
          optimizer,
          reps,
        },
      },
      result: {
        ground_energy: finalVal,
        fci_energy: target,
        energy_error_mha: Math.abs(finalVal - target) * 1000,
        qubits: molecule.activeQubits,
        iterations: steps,
        converged: Math.abs(finalVal - target) < 0.0016, // chemical accuracy 1 kcal/mol
      },
    });
  }, [molecule, bondR, moleculeKey, ansatz, mapping, optimizer, maxIterations, reps, addExperiment]);

  // Deploy Ansatz to Circuit Studio
  const handleDeployToCircuitStudio = useCallback(() => {
    const nQ = molecule.activeQubits;
    const ops: GateOp[] = [];
    let col = 0;

    // Initial Hartree-Fock state preparation (X gates on occupied orbitals)
    ops.push({ id: `g_init_0`, name: "x", target: 0, col: col++, x: 120, y: 60 });
    if (nQ > 2) {
      ops.push({ id: `g_init_1`, name: "x", target: 1, col: col++, x: 120 + col * 120, y: 120 });
    }

    // Variational layers
    for (let r = 0; r < reps; r++) {
      // Rotation layer
      for (let q = 0; q < nQ; q++) {
        ops.push({
          id: `g_ry_${r}_${q}`,
          name: "ry",
          target: q,
          parameter: `θ_${r}_${q}`,
          theta: 0.785,
          col,
          x: 120 + col * 120,
          y: 60 + q * 60,
        });
      }
      col++;

      // Entangling layer (Linear CNOT chain)
      for (let q = 0; q < nQ - 1; q++) {
        ops.push({
          id: `g_cx_${r}_${q}`,
          name: "cx",
          control: q,
          target: q + 1,
          col,
          x: 120 + col * 120,
          y: 60 + (q + 1) * 60,
        });
        col++;
      }
    }

    replaceCircuit(nQ, ops);
    if (setActiveTab) {
      setActiveTab("design");
    }
  }, [molecule.activeQubits, reps, replaceCircuit, setActiveTab]);

  return (
    <div style={shell}>
      {/* CAD Header Banner */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          marginBottom: 16,
          flexWrap: "wrap",
          gap: 12,
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 20 }}>🧪</span>
            <h1 style={{ margin: 0, fontSize: 20, fontWeight: 800, letterSpacing: "-.02em", color: "#f8fafc" }}>
              Quantum Chemistry & Molecular Workstation
            </h1>
            <span
              style={{
                fontSize: 11,
                padding: "2px 8px",
                borderRadius: 4,
                background: "#0284c7",
                color: "#ffffff",
                fontWeight: 600,
              }}
            >
              VQE & Full CI
            </span>
          </div>
          <p style={{ margin: "4px 0 0", color: "#94a3b8", fontSize: 13 }}>
            Ab-initio electronic structure · Born-Oppenheimer PES · Molecular orbitals · Variational quantum eigensolver
          </p>
        </div>

        {/* Global Action Bar */}
        <div style={{ display: "flex", gap: 8 }}>
          <Button
            variant="secondary"
            onClick={handleDeployToCircuitStudio}
            style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 6 }}
          >
            <span>⚡ Deploy Ansatz to Circuit Studio</span>
          </Button>
          <Button
            variant="primary"
            onClick={handleRunVQE}
            disabled={isSolving}
            style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 6 }}
          >
            {isSolving ? "Optimizing VQE..." : "▶ Run VQE Solver"}
          </Button>
        </div>
      </div>

      {/* Main 2-Column Responsive Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))",
          gap: 18,
          alignItems: "start",
        }}
      >
        {/* Left Column: Chemistry & VQE Parameters */}
        <div style={{ display: "grid", gap: 16 }}>
          {/* Molecule Selection Card */}
          <div style={card}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#f8fafc", marginBottom: 12 }}>
              1. Molecule & Geometry
            </div>

            <div style={{ display: "grid", gap: 12 }}>
              <div style={fieldStyle}>
                <span>Target Molecule Preset:</span>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 6 }}>
                  {(["H2", "LiH", "BeH2", "H2O", "N2"] as MoleculePreset[]).map((key) => (
                    <button
                      key={key}
                      type="button"
                      onClick={() => handleSelectMolecule(key)}
                      style={{
                        padding: "6px 4px",
                        fontSize: 12,
                        fontWeight: moleculeKey === key ? 700 : 500,
                        borderRadius: 4,
                        border: moleculeKey === key ? "1px solid #38bdf8" : "1px solid #334155",
                        background: moleculeKey === key ? "#0284c7" : "#1e293b",
                        color: "#f8fafc",
                        cursor: "pointer",
                      }}
                    >
                      {MOLECULES[key].formula}
                    </button>
                  ))}
                </div>
              </div>

              {/* Nuclear Separation Slider R */}
              <div style={fieldStyle}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>Bond Distance R (Å):</span>
                  <span style={{ color: "#38bdf8", fontWeight: 700 }}>{bondR.toFixed(3)} Å</span>
                </div>
                <input
                  type="range"
                  min="0.3"
                  max="3.0"
                  step="0.02"
                  value={bondR}
                  onChange={(e) => setBondR(parseFloat(e.target.value))}
                  style={{ width: "100%", accentColor: "#0284c7", cursor: "pointer" }}
                />
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#64748b" }}>
                  <span>0.30 Å (Repulsive)</span>
                  <span>Equilibrium: {molecule.equilibriumR} Å</span>
                  <span>3.00 Å (Dissociated)</span>
                </div>
              </div>

              {/* Basis Set & Qubit Mapping */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                <div style={fieldStyle}>
                  <span>Atomic Basis Set:</span>
                  <select
                    value={basisSet}
                    onChange={(e) => setBasisSet(e.target.value as "STO-3G" | "6-31G")}
                    style={inputStyle}
                  >
                    <option value="STO-3G">STO-3G (Minimal)</option>
                    <option value="6-31G">6-31G (Split-valence)</option>
                  </select>
                </div>
                <div style={fieldStyle}>
                  <span>Fermion-Qubit Mapping:</span>
                  <select
                    value={mapping}
                    onChange={(e) => setMapping(e.target.value as QubitMapping)}
                    style={inputStyle}
                  >
                    <option value="parity">Parity (Z₂ 2-Qubit Red.)</option>
                    <option value="jordan_wigner">Jordan-Wigner</option>
                    <option value="bravyi_kitaev">Bravyi-Kitaev</option>
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* VQE Algorithm & Ansatz Card */}
          <div style={card}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#f8fafc", marginBottom: 12 }}>
              2. VQE Ansatz & Optimizer
            </div>

            <div style={{ display: "grid", gap: 12 }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                <div style={fieldStyle}>
                  <span>Ansatz Architecture:</span>
                  <select
                    value={ansatz}
                    onChange={(e) => setAnsatz(e.target.value as VQEAnsatz)}
                    style={inputStyle}
                  >
                    <option value="hardware_efficient">Hardware-Efficient (Ry+CZ)</option>
                    <option value="uccsd">Unitary Coupled-Cluster (UCCSD)</option>
                    <option value="ry_rz">Ry-Rz Alternating</option>
                  </select>
                </div>
                <div style={fieldStyle}>
                  <span>Classical Optimizer:</span>
                  <select
                    value={optimizer}
                    onChange={(e) => setOptimizer(e.target.value as OptimizerType)}
                    style={inputStyle}
                  >
                    <option value="COBYLA">COBYLA (Derivative-free)</option>
                    <option value="SLSQP">SLSQP (Gradient-based)</option>
                    <option value="Adam">Adam (Stochastic)</option>
                  </select>
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                <div style={fieldStyle}>
                  <span>Ansatz Repetitions (Layers):</span>
                  <input
                    type="number"
                    min="1"
                    max="6"
                    value={reps}
                    onChange={(e) => setReps(parseInt(e.target.value, 10) || 1)}
                    style={inputStyle}
                  />
                </div>
                <div style={fieldStyle}>
                  <span>Max Iterations:</span>
                  <input
                    type="number"
                    min="10"
                    max="200"
                    step="10"
                    value={maxIterations}
                    onChange={(e) => setMaxIterations(parseInt(e.target.value, 10) || 50)}
                    style={inputStyle}
                  />
                </div>
              </div>

              {/* Status & Diagnostic Badges */}
              <div
                style={{
                  background: "#090d16",
                  padding: "10px 12px",
                  borderRadius: 6,
                  border: "1px solid #1e293b",
                  display: "grid",
                  gap: 6,
                  fontSize: 11,
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#94a3b8" }}>Active Qubits required:</span>
                  <span style={{ color: "#38bdf8", fontWeight: 700 }}>{molecule.activeQubits} Qubits</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#94a3b8" }}>Variational Parameters:</span>
                  <span style={{ color: "#f8fafc", fontWeight: 600 }}>{molecule.activeQubits * reps} angles</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "#94a3b8" }}>Chemical Accuracy Target:</span>
                  <span style={{ color: "#34d399", fontWeight: 600 }}>1.6 mHa (1 kcal/mol)</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Visualizer Workstation */}
        <div style={{ display: "grid", gap: 16 }}>
          {/* Sub-Tabs Navigation */}
          <div
            style={{
              display: "flex",
              borderBottom: "1px solid #334155",
              gap: 4,
              paddingBottom: 2,
              overflowX: "auto",
            }}
          >
            {[
              { id: "pes", label: "PES Dissociation E(R)" },
              { id: "orbitals", label: "Molecular Orbitals" },
              { id: "vqe", label: "VQE Convergence" },
              { id: "structure3d", label: "3D Molecular Geometry" },
              { id: "export", label: "Export & Code" },
            ].map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setTab(item.id as ChemistryTab)}
                style={{
                  padding: "8px 12px",
                  fontSize: 12,
                  fontWeight: tab === item.id ? 700 : 500,
                  border: "none",
                  borderBottom: tab === item.id ? "2px solid #38bdf8" : "2px solid transparent",
                  background: "transparent",
                  color: tab === item.id ? "#38bdf8" : "#94a3b8",
                  cursor: "pointer",
                  whiteSpace: "nowrap",
                }}
              >
                {item.label}
              </button>
            ))}
          </div>

          {/* Tab 1: Potential Energy Surface */}
          {tab === "pes" && (
            <PotentialEnergySurfaceChart
              moleculeName={molecule.name}
              data={molecule.pesCurve}
              currentR={bondR}
              equilibriumR={molecule.equilibriumR}
              dissociationEnergy={molecule.dissociationEnergy}
              onSelectR={(r) => setBondR(r)}
            />
          )}

          {/* Tab 2: Molecular Orbitals Spectrum */}
          {tab === "orbitals" && (
            <MolecularOrbitalsDiagram
              orbitals={molecule.orbitals}
              homoLumoGap={molecule.homoLumoGap}
              title={`${molecule.formula} Molecular Orbital Spectrum`}
            />
          )}

          {/* Tab 3: VQE Convergence Trace */}
          {tab === "vqe" && (
            <div style={{ display: "grid", gap: 12 }}>
              <VQEOptimizationTrace
                energies={vqeHistory}
                targetEnergy={fciEnergy}
                optimizerName={optimizer}
              />
              {finalEnergy != null && (
                <div
                  style={{
                    background: "#111827",
                    border: "1px solid #334155",
                    borderRadius: 8,
                    padding: 14,
                    display: "grid",
                    gridTemplateColumns: "repeat(3, 1fr)",
                    gap: 10,
                    textAlign: "center",
                  }}
                >
                  <div>
                    <div style={{ fontSize: 11, color: "#94a3b8" }}>VQE Ground State</div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: "#34d399" }}>
                      {finalEnergy.toFixed(6)} Ha
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: "#94a3b8" }}>Full CI (Exact)</div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: "#38bdf8" }}>
                      {fciEnergy.toFixed(6)} Ha
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: "#94a3b8" }}>Energy Error ΔE</div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: "#f59e0b" }}>
                      {(Math.abs(finalEnergy - fciEnergy) * 1000).toFixed(3)} mHa
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Tab 4: 3D Molecular Geometry */}
          {tab === "structure3d" && (
            <Lattice3DView
              sites={dynamicSites}
              bonds={molecule.bonds}
              title={`${molecule.formula} (${molecule.name}) 3D Molecular Geometry`}
              subtitle={`R = ${bondR.toFixed(3)} Å · CPK Color coded`}
              colorMode="element"
              height={340}
            />
          )}

          {/* Tab 5: Export & Code */}
          {tab === "export" && (
            <div style={{ display: "grid", gap: 14 }}>
              <div style={card}>
                <div style={{ fontSize: 13, fontWeight: 700, color: "#f8fafc", marginBottom: 8 }}>
                  Python Qiskit Chemistry / PySCF Script
                </div>
                <pre
                  style={{
                    background: "#090d16",
                    border: "1px solid #1e293b",
                    padding: 12,
                    borderRadius: 6,
                    fontSize: 11,
                    color: "#cbd5e1",
                    overflowX: "auto",
                  }}
                >
{`from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ${mapping === "parity" ? "ParityMapper" : "JordanWignerMapper"}
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import ${optimizer}
from qiskit.circuit.library import EfficientSU2

# 1. Driver setup for ${molecule.formula} at R = ${bondR.toFixed(3)} Å
driver = PySCFDriver(
    atom="${molecule.sites.map((s, i) => `${s.element} ${dynamicSites[i].x.toFixed(3)} ${dynamicSites[i].y.toFixed(3)} 0.0`).join("; ")}",
    basis="${basisSet.toLowerCase()}",
    unit=DistanceUnit.ANGSTROM,
)
problem = driver.run()
hamiltonian = ${mapping === "parity" ? "ParityMapper().map(problem.hamiltonian)" : "JordanWignerMapper().map(problem.hamiltonian)"}

# 2. VQE setup
ansatz = EfficientSU2(${molecule.activeQubits}, reps=${reps})
optimizer = ${optimizer}(maxiter=${maxIterations})
vqe = VQE(ansatz=ansatz, optimizer=optimizer)
print(f"Target Qubits: {ansatz.num_qubits}")`}
                </pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
