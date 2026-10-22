"use client";

import React from "react";
import {
  buildHamiltonian,
  previewLattice,
  runDMRG,
  runGroundState,
  runTEBD,
  type AsyncJob,
  type LatticeGraph,
  type SparseHamiltonianResponse,
  type PauliTerm,
} from "../lib/agent";
import { useUIContext } from "../state/uiContext";
import { Button } from "../ui";
import { EnergyChart, LatticeCanvas } from "./LatticeVisuals";
import { CTMRGLabPanel } from "./CTMRGLabPanel";

export type ModelName =
  | "ising"
  | "heisenberg"
  | "xxz"
  | "xy"
  | "j1_j2"
  | "dimerized"
  | "kitaev_chain";

export type Boundary = "open" | "periodic";

export type PhysicsPreset =
  | "ising_crit"
  | "ising_ferro"
  | "ising_para"
  | "heisenberg_afm"
  | "xxz_luttinger"
  | "j1_j2_mg"
  | "dimerized_ssh"
  | "kitaev_topo";

export type PhysicsTab = "model" | "solvers" | "dynamics" | "phase" | "ctmrg";


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
  width: 90,
  border: "1px solid #475569",
  borderRadius: 4,
  background: "#1e293b",
  color: "#f8fafc",
  padding: "6px 8px",
  fontSize: 13,
  outline: "none",
};

function errorText(error: unknown) {
  return error instanceof Error ? error.message : "Operation failed.";
}

export function PhysicsLab() {
  const { setLatestOutput, addExperiment } = useUIContext();

  const [dimension, setDimension] = React.useState<1 | 2>(1);
  const [sizes, setSizes] = React.useState([8, 4, 1]);
  const [boundary, setBoundary] = React.useState<Boundary>("open");
  const [model, setModel] = React.useState<ModelName>("ising");

  // Coupling parameters
  const [coupling, setCoupling] = React.useState(1.0);
  const [field, setField] = React.useState(1.0);
  const [anisotropy, setAnisotropy] = React.useState(1.0);
  const [j2Coupling, setJ2Coupling] = React.useState(0.5);
  const [dimerizationDelta, setDimerizationDelta] = React.useState(0.3);
  const [pairingDelta, setPairingDelta] = React.useState(1.0);
  const [chemicalPotential, setChemicalPotential] = React.useState(0.0);

  // Solver parameters
  const [bondDim, setBondDim] = React.useState(16);
  const [sweeps, setSweeps] = React.useState(4);
  const [dt, setDt] = React.useState(0.05);
  const [steps, setSteps] = React.useState(20);

  // State
  const [graph, setGraph] = React.useState<LatticeGraph | null>(null);
  const [hamiltonian, setHamiltonian] = React.useState<SparseHamiltonianResponse | null>(null);
  const [dmrgResult, setDmrgResult] = React.useState<Record<string, unknown> | null>(null);
  const [tebdResult, setTebdResult] = React.useState<Record<string, unknown> | null>(null);
  const [groundResult, setGroundResult] = React.useState<Record<string, unknown> | null>(null);

  const [busy, setBusy] = React.useState<"preview" | "hamiltonian" | "ground" | "dmrg" | "tebd" | null>(null);
  const [jobProgress, setJobProgress] = React.useState<AsyncJob | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  const [physicsTab, setPhysicsTab] = React.useState<PhysicsTab>("model");

  const activeDimensions = dimension === 1 ? [sizes[0]] : [sizes[0], sizes[1]];
  const siteCount = activeDimensions.reduce((acc, dim) => acc * dim, 1);
  const qubitCount = siteCount;

  function applyPreset(preset: PhysicsPreset) {
    setError(null);
    if (preset === "ising_crit") {
      setDimension(1);
      setSizes([8, 1, 1]);
      setBoundary("open");
      setModel("ising");
      setCoupling(1.0);
      setField(1.0);
    } else if (preset === "ising_ferro") {
      setDimension(1);
      setSizes([8, 1, 1]);
      setBoundary("open");
      setModel("ising");
      setCoupling(1.0);
      setField(0.25);
    } else if (preset === "ising_para") {
      setDimension(1);
      setSizes([8, 1, 1]);
      setBoundary("open");
      setModel("ising");
      setCoupling(1.0);
      setField(2.5);
    } else if (preset === "heisenberg_afm") {
      setDimension(1);
      setSizes([8, 1, 1]);
      setBoundary("open");
      setModel("heisenberg");
      setCoupling(1.0);
      setField(0.0);
    } else if (preset === "xxz_luttinger") {
      setDimension(1);
      setSizes([10, 1, 1]);
      setBoundary("open");
      setModel("xxz");
      setCoupling(1.0);
      setField(0.0);
      setAnisotropy(0.5);
    } else if (preset === "j1_j2_mg") {
      setDimension(1);
      setSizes([8, 1, 1]);
      setBoundary("open");
      setModel("j1_j2");
      setCoupling(1.0);
      setJ2Coupling(0.5);
    } else if (preset === "dimerized_ssh") {
      setDimension(1);
      setSizes([8, 1, 1]);
      setBoundary("open");
      setModel("dimerized");
      setCoupling(1.0);
      setDimerizationDelta(0.4);
    } else if (preset === "kitaev_topo") {
      setDimension(1);
      setSizes([8, 1, 1]);
      setBoundary("open");
      setModel("kitaev_chain");
      setCoupling(1.0);
      setPairingDelta(1.0);
      setChemicalPotential(0.0);
    }
  }

  function getPhaseInsight() {
    if (model === "ising") {
      const ratio = Math.abs(coupling) > 1e-9 ? Math.abs(field / coupling) : 999;
      if (ratio < 0.95) {
        return {
          title: "Ferromagnetic Ordered Phase",
          badge: "Z2 Symmetry Broken",
          description: "Kvant fluktuatsiyalari kuchsiz; qo'shni spinlar parallel tartiblangan. Spontan qutblanish mavjud.",
          orderParameter: "⟨Z_i⟩ = (1 - (h/J)²)^(1/8) ≠ 0",
          centralCharge: "c = 0 (Gapped)",
          gap: `Δ ≈ ${Math.abs(2 * (coupling - field)).toFixed(3)} J`,
        };
      } else if (ratio > 1.05) {
        return {
          title: "Quantum Paramagnetic Phase",
          badge: "Disordered / Polarized",
          description: "Ko'ndalang magnit maydoni ustun; spinlar X-o'qi bo'ylab qutblangan, energetik spektr mass gapga ega.",
          orderParameter: "⟨X_i⟩ ≈ 1, ⟨Z_i⟩ = 0",
          centralCharge: "c = 0 (Gapped)",
          gap: `Δ ≈ ${Math.abs(2 * (field - coupling)).toFixed(3)} J`,
        };
      } else {
        return {
          title: "Quantum Critical Point (h = J)",
          badge: "Conformal Invariant (c = 1/2 CFT)",
          description: "Massasiz Majorana fermionlari va 2D Ising CFT bilan tavsiflanuvchi uzluksiz fazaviy o'tish nuqtasi.",
          orderParameter: "Power-law ⟨Z_0 Z_r⟩ ~ r^(-1/4)",
          centralCharge: "c = 1/2 (Free Majorana)",
          gap: "Δ = 0 (Gapless Critical)",
        };
      }
    } else if (model === "heisenberg") {
      if (coupling > 0) {
        return {
          title: "Antiferromagnetic Phase (Spin-1/2 Chain)",
          badge: "SU(2) Invariant Gapless",
          description: "Bethe Ansatz asosida yechiluvchi gapless spinon to'lqinlari. Asosiy holat energiyasi e₀ = 1/4 - ln 2 ≈ -0.443147 J.",
          orderParameter: "Staggered ⟨(-1)^r S_0^z S_r^z⟩ ~ (ln r)^(1/2) / r",
          centralCharge: "c = 1 (WZW SU(2)_1)",
          gap: "Δ = 0 (Gapless)",
        };
      } else {
        return {
          title: "Ferromagnetic State",
          badge: "Fully Polarized",
          description: "Barcha spinlar bir yo'nalishga qutblangan; asosiy holat to'liq ferromagnit multiplet.",
          orderParameter: "Total Spin S = N/2",
          centralCharge: "c = 0",
          gap: "Δ = 0 (Goldstone mode)",
        };
      }
    } else if (model === "xxz") {
      if (anisotropy > 1.05) {
        return {
          title: "Ising-like Gapped Antiferromagnet",
          badge: "Discrete Z2 Broken",
          description: "Z-anizotropiya z-o'qi bo'ylab Néel antiferromagnit tartib hosil qiladi, qo'zg'alishlar energetik tirqishga ega.",
          orderParameter: "Exponential Decay ⟨S_0^z S_r^z⟩ ~ e^(-r/ξ)",
          centralCharge: "c = 0",
          gap: "Δ > 0 (Gapped Néel)",
        };
      } else if (anisotropy < -1.05) {
        return {
          title: "Ferromagnetic Phase",
          badge: "Anisotropic FM",
          description: "Z-yo'nalishidagi kuchli ferromagnit o'zaro ta'sir spinlarni to'liq qutblaydi.",
          orderParameter: "Ferromagnetic Polarization",
          centralCharge: "c = 0",
          gap: "Δ = 0",
        };
      } else {
        return {
          title: "Tomonaga–Luttinger Liquid (XY Critical)",
          badge: "Gapless Critical (c = 1)",
          description: "Kvant anizotropik suyuqligi; korrelyatsiyalar algebraik pasayadi, erkin bosonik maydon bilan tavsiflanadi.",
          orderParameter: "Algebraic Decay, K = π / [2(π - arccos Δ)]",
          centralCharge: "c = 1 (Luttinger Liquid)",
          gap: "Δ = 0 (Gapless)",
        };
      }
    } else if (model === "j1_j2") {
      const ratio = Math.abs(coupling) > 1e-9 ? j2Coupling / coupling : 0;
      if (Math.abs(ratio - 0.5) < 0.05) {
        return {
          title: "Majumdar–Ghosh Point (J₂/J₁ = 0.5)",
          badge: "Exact Valence Bond Solid (VBS)",
          description: "Asosiy holat qo'shni singlet juftliklarining aniq mahsulotidir. E₀ = -0.375 J₁ N aynan to'g'ri keladi.",
          orderParameter: "Dimer Singlet Order, Spontaneous Parity Breaking",
          centralCharge: "c = 0 (Gapped VBS)",
          gap: "Δ ≈ 0.233 J₁ (Majumdar-Ghosh gap)",
        };
      } else if (ratio < 0.2411) {
        return {
          title: "Gapless Luttinger Liquid Phase",
          badge: "c = 1 Critical",
          description: "Kuchsiz frustratsiya; Heisenberg tipidagi gapless spinon to'lqinlari saqlanadi.",
          orderParameter: "Power-law Correlations",
          centralCharge: "c = 1",
          gap: "Δ = 0 (Gapless)",
        };
      } else {
        return {
          title: "Frustrated Gapped Dimerized Phase",
          badge: "Frustrated Spin Liquid Precursor",
          description: "Kuchli keyingi qo'shni J₂ o'zaro ta'siri antiferromagnit tartibni buzadi va dimerizatsiyalashgan energetik tirqish hosil qiladi.",
          orderParameter: "Dimerization Order Parameter",
          centralCharge: "c = 0",
          gap: "Δ > 0 (Dimer Gap)",
        };
      }
    } else if (model === "dimerized") {
      return {
        title: "Topological Dimerized Spin-Peierls Chain",
        badge: "SSH Spin Analog",
        description: "Alternativ bog'lanish J(1 ± δ); ochiq chegaralarda topologik chekka spinon holatlari (S = 1/2 fractionalized edge modes) paydo bo'ladi.",
        orderParameter: "Dimerization Gap Δ ~ J δ^(2/3)",
        centralCharge: "c = 0 (Topological Bulk Gap)",
        gap: `Δ ≈ ${(2 * Math.abs(dimerizationDelta) * coupling).toFixed(3)} J`,
      };
    } else if (model === "kitaev_chain") {
      const isTopological = Math.abs(chemicalPotential) < 2 * Math.abs(coupling);
      return {
        title: isTopological ? "Topological Superconductor Phase (Majorana Zero Modes)" : "Trivial Band Insulator",
        badge: isTopological ? "Z2 Topological (M = -1)" : "Trivial Phase (M = +1)",
        description: isTopological
          ? "Zanjir uchlarida lokalizatsiyalangan juftlanmagan Majorana nol rejimlari (γ_left, γ_right). Topologik kvant hisoblash uchun asos."
          : "Kimyoviy potentsial ustun (|μ| > 2t); topologik chekka holatlar yo'q.",
        orderParameter: isTopological ? "Majorana Edge Parity Operator i γ_1 γ_2N" : "Trivial Vacuum",
        centralCharge: "c = 0 (Topological Bulk Gap)",
        gap: `Δ_bulk ≈ ${Math.abs(2 * coupling - Math.abs(chemicalPotential)).toFixed(3)}`,
      };
    } else {
      // XY Model
      return {
        title: "XY Spin Chain in Transverse Field",
        badge: "Jordan-Wigner Solvable",
        description: "Anizotrop XY spin modeli; erkin fermionlar to'plamiga to'liq keltiriladi.",
        orderParameter: "Transverse Magnetization ⟨Z⟩",
        centralCharge: Math.abs(field) === Math.abs(coupling) ? "c = 1/2" : "c = 0",
        gap: `Δ = 2|J - h|`,
      };
    }
  }

  // Construct Pauli terms for all 7 models
  function constructPauliTerms(currentGraph: LatticeGraph): SparseHamiltonianResponse {
    const terms: PauliTerm[] = [];
    const n = qubitCount;

    // Helper for adding edge interactions
    const addEdge = (src: number, dst: number, jVal: number, xxFactor = 1.0, yyFactor = 1.0, zzFactor = 1.0) => {
      if (Math.abs(xxFactor) > 1e-9) {
        terms.push({ paulis: { [src]: "X", [dst]: "X" }, coefficient: jVal * xxFactor, label: "XX" });
      }
      if (Math.abs(yyFactor) > 1e-9) {
        terms.push({ paulis: { [src]: "Y", [dst]: "Y" }, coefficient: jVal * yyFactor, label: "YY" });
      }
      if (Math.abs(zzFactor) > 1e-9) {
        terms.push({ paulis: { [src]: "Z", [dst]: "Z" }, coefficient: jVal * zzFactor, label: "ZZ" });
      }
    };

    if (model === "ising") {
      // H = -J \sum Z_i Z_j - h \sum X_i
      for (let i = 0; i < n - 1; i++) {
        terms.push({ paulis: { [i]: "Z", [i + 1]: "Z" }, coefficient: -coupling, label: "ZZ" });
      }
      if (boundary === "periodic" && n > 2) {
        terms.push({ paulis: { [n - 1]: "Z", [0]: "Z" }, coefficient: -coupling, label: "ZZ" });
      }
      if (field !== 0) {
        for (let i = 0; i < n; i++) {
          terms.push({ paulis: { [i]: "X" }, coefficient: -field, label: "X field" });
        }
      }
    } else if (model === "heisenberg") {
      // H = J \sum (X_i X_j + Y_i Y_j + Z_i Z_j) - h \sum Z_i
      for (let i = 0; i < n - 1; i++) {
        addEdge(i, i + 1, coupling, 1.0, 1.0, 1.0);
      }
      if (boundary === "periodic" && n > 2) {
        addEdge(n - 1, 0, coupling, 1.0, 1.0, 1.0);
      }
      if (field !== 0) {
        for (let i = 0; i < n; i++) {
          terms.push({ paulis: { [i]: "Z" }, coefficient: -field, label: "Z field" });
        }
      }
    } else if (model === "xxz") {
      // H = J \sum (X_i X_j + Y_i Y_j + \Delta Z_i Z_j) - h \sum Z_i
      for (let i = 0; i < n - 1; i++) {
        addEdge(i, i + 1, coupling, 1.0, 1.0, anisotropy);
      }
      if (boundary === "periodic" && n > 2) {
        addEdge(n - 1, 0, coupling, 1.0, 1.0, anisotropy);
      }
      if (field !== 0) {
        for (let i = 0; i < n; i++) {
          terms.push({ paulis: { [i]: "Z" }, coefficient: -field, label: "Z field" });
        }
      }
    } else if (model === "xy") {
      // H = -J \sum ((1+γ)/2 X_i X_j + (1-γ)/2 Y_i Y_j) - h \sum Z_i
      const xxF = (1 + anisotropy) / 2;
      const yyF = (1 - anisotropy) / 2;
      for (let i = 0; i < n - 1; i++) {
        addEdge(i, i + 1, -coupling, xxF, yyF, 0);
      }
      if (boundary === "periodic" && n > 2) {
        addEdge(n - 1, 0, -coupling, xxF, yyF, 0);
      }
      if (field !== 0) {
        for (let i = 0; i < n; i++) {
          terms.push({ paulis: { [i]: "Z" }, coefficient: -field, label: "Z field" });
        }
      }
    } else if (model === "j1_j2") {
      // H = J_1 \sum (S_i · S_{i+1}) + J_2 \sum (S_i · S_{i+2})
      for (let i = 0; i < n - 1; i++) {
        addEdge(i, i + 1, coupling, 1.0, 1.0, 1.0);
      }
      if (boundary === "periodic" && n > 2) {
        addEdge(n - 1, 0, coupling, 1.0, 1.0, 1.0);
      }
      // Next-nearest neighbor J_2
      for (let i = 0; i < n - 2; i++) {
        addEdge(i, i + 2, j2Coupling, 1.0, 1.0, 1.0);
      }
      if (boundary === "periodic" && n > 3) {
        addEdge(n - 2, 0, j2Coupling, 1.0, 1.0, 1.0);
        addEdge(n - 1, 1, j2Coupling, 1.0, 1.0, 1.0);
      }
    } else if (model === "dimerized") {
      // H = J \sum (1 + (-1)^i \delta) (S_i · S_{i+1})
      for (let i = 0; i < n - 1; i++) {
        const factor = 1.0 + (i % 2 === 0 ? dimerizationDelta : -dimerizationDelta);
        addEdge(i, i + 1, coupling * factor, 1.0, 1.0, 1.0);
      }
      if (boundary === "periodic" && n > 2) {
        const factor = 1.0 + ((n - 1) % 2 === 0 ? dimerizationDelta : -dimerizationDelta);
        addEdge(n - 1, 0, coupling * factor, 1.0, 1.0, 1.0);
      }
    } else if (model === "kitaev_chain") {
      // Jordan-Wigner mapped Kitaev chain:
      // H = -(t+Δ)/2 \sum X_i X_{i+1} - (t-Δ)/2 \sum Y_i Y_{i+1} - μ/2 \sum Z_i
      const xxF = -(coupling + pairingDelta) / 2;
      const yyF = -(coupling - pairingDelta) / 2;
      for (let i = 0; i < n - 1; i++) {
        addEdge(i, i + 1, 1.0, xxF, yyF, 0);
      }
      if (boundary === "periodic" && n > 2) {
        addEdge(n - 1, 0, 1.0, xxF, yyF, 0);
      }
      if (chemicalPotential !== 0) {
        for (let i = 0; i < n; i++) {
          terms.push({ paulis: { [i]: "Z" }, coefficient: -chemicalPotential / 2, label: "Z chem" });
        }
      }
    }

    return {
      ...currentGraph,
      n_qubits: n,
      terms,
    };
  }

  async function generateLattice() {
    setBusy("preview");
    setError(null);
    try {
      const g = await previewLattice("spin-lattice", { dimension, sizes: activeDimensions, boundary });
      setGraph(g);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  async function generateHamiltonianModel() {
    setBusy("hamiltonian");
    setError(null);
    try {
      const g = graph ?? (await previewLattice("spin-lattice", { dimension, sizes: activeDimensions, boundary }));
      if (!graph) setGraph(g);

      if (model === "ising" || model === "heisenberg" || model === "xxz") {
        const payload = {
          dimension,
          sizes: activeDimensions,
          boundary,
          model,
          coupling,
          field,
          anisotropy,
        };
        const h = await buildHamiltonian("spin-lattice", payload);
        setHamiltonian(h);
      } else {
        const h = constructPauliTerms(g);
        setHamiltonian(h);
      }
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  async function runExactED() {
    if (!hamiltonian) return;
    setBusy("ground");
    setError(null);
    try {
      const res = await runGroundState(
        {
          terms: hamiltonian.terms,
          n_qubits: hamiltonian.n_qubits,
        },
        setJobProgress
      );
      setGroundResult(res);
      setLatestOutput(res);
      addExperiment({
        label: `Physics · Exact ED · ${model.toUpperCase()} (N=${qubitCount})`,
        source: "physics",
        kind: "ground_state",
        status: "done",
        request: { source: "physics", kind: "ground_state", payload: { terms: hamiltonian.terms } },
        result: res,
        provenance: res.provenance,
      });
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  async function runDMRGSolver() {
    if (!hamiltonian) return;
    setBusy("dmrg");
    setError(null);
    try {
      const res = await runDMRG(
        {
          terms: hamiltonian.terms,
          n_qubits: hamiltonian.n_qubits,
          bond_dim: bondDim,
          sweeps,
        },
        setJobProgress
      );
      setDmrgResult(res);
      setLatestOutput(res);
      addExperiment({
        label: `Physics · DMRG · ${model.toUpperCase()} (χ=${bondDim})`,
        source: "physics",
        kind: "dmrg",
        status: "done",
        request: { source: "physics", kind: "dmrg", payload: { terms: hamiltonian.terms, bond_dim: bondDim, sweeps } },
        result: res,
        provenance: res.provenance,
      });
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  async function runTEBDEvolution() {
    if (!hamiltonian) return;
    setBusy("tebd");
    setError(null);
    try {
      const res = await runTEBD(
        {
          terms: hamiltonian.terms,
          n_qubits: hamiltonian.n_qubits,
          dt,
          steps,
          bond_dim: bondDim,
        },
        setJobProgress
      );
      setTebdResult(res);
      setLatestOutput(res);
      addExperiment({
        label: `Physics · TEBD · ${model.toUpperCase()} (steps=${steps})`,
        source: "physics",
        kind: "tebd",
        status: "done",
        request: { source: "physics", kind: "tebd", payload: { terms: hamiltonian.terms, dt, steps } },
        result: res,
        provenance: res.provenance,
      });
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  // Exports
  function exportPythonModel() {
    const code = `# Quantum Circuit Platform - Rigorous Many-Body Physics Model
# Model: ${model.toUpperCase()} on ${activeDimensions.join("x")} lattice (${boundary} boundary)
import numpy as np

n_spins = ${hamiltonian?.n_qubits ?? qubitCount}
coupling_J = ${coupling}
transverse_field_h = ${field}
anisotropy_delta = ${anisotropy}
j2_coupling = ${j2Coupling}
dimerization_delta = ${dimerizationDelta}

pauli_terms = ${JSON.stringify(hamiltonian?.terms ?? [], null, 2)}

print(f"Loaded {len(pauli_terms)} Pauli terms for {n_spins}-spin Hamiltonian")
`;
    const blob = new Blob([code], { type: "text/x-python" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `spin_physics_${model}_${activeDimensions.join("x")}.py`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function exportLatexTable() {
    const eVal = dmrgEnergy ?? groundEnergy ?? 0;
    const tex = `% Quantum Many-Body Physics Publication Summary
\\begin{table}[htbp]
\\centering
\\caption{Simulation of ${model.toUpperCase()} Model on ${activeDimensions.join("$\\times$")} Lattice}
\\begin{tabular}{ll|ll}
\\hline\\hline
\\textbf{Parameter} & \\textbf{Value} & \\textbf{Observable} & \\textbf{Value} \\\\
\\hline
Lattice Dimension & ${dimension}D & Boundary Condition & ${boundary} \\\\
Spins ($N$) & ${qubitCount} & Pauli Operators & ${hamiltonian?.terms.length ?? 0} \\\\
Coupling $J$ & ${coupling.toFixed(4)} & Tuning Parameter & ${field.toFixed(4)} \\\\
Ground Energy $E_0$ & ${Number(eVal).toFixed(7)} & Energy per spin $e_0$ & ${(Number(eVal) / siteCount).toFixed(7)} \\\\
Excitation Gap $\\Delta E$ & ${energyGap != null ? energyGap.toFixed(6) : "N/A"} & Central Charge $c$ & ${getPhaseInsight().centralCharge} \\\\
Phase Regime & \\multicolumn{3}{l}{${getPhaseInsight().title}} \\\\
Order Parameter & \\multicolumn{3}{l}{${getPhaseInsight().orderParameter}} \\\\
\\hline\\hline
\\end{tabular}
\\end{table}
`;
    const blob = new Blob([tex], { type: "text/x-tex" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `spin_physics_${model}_${activeDimensions.join("x")}.tex`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // Solvers metrics
  const dmrgEnergy = dmrgResult?.ground_energy;
  const groundEnergy = groundResult?.ground_energy;
  const lowestEigenvalues = Array.isArray(groundResult?.lowest_eigenvalues)
    ? (groundResult.lowest_eigenvalues as number[])
    : null;
  const energyGap =
    lowestEigenvalues && lowestEigenvalues.length > 1
      ? Math.abs(lowestEigenvalues[1] - lowestEigenvalues[0])
      : null;

  const energies =
    tebdResult && Array.isArray(tebdResult.energies)
      ? tebdResult.energies.map(Number).filter(Number.isFinite)
      : [];
  const bondDimHistory =
    tebdResult && Array.isArray(tebdResult.bond_dim_history)
      ? tebdResult.bond_dim_history.map(Number).filter(Number.isFinite)
      : [];

  return (
    <main style={shell}>
      {/* Top Header & Navigation */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 16,
          borderBottom: "1px solid #1e293b",
          paddingBottom: 12,
          flexWrap: "wrap",
          gap: 12,
        }}
      >
        <div>
          <h2 style={{ margin: "0 0 4px 0", fontSize: 18, fontWeight: 700, color: "#f8fafc" }}>
            Many-Body Quantum Physics Laboratory
          </h2>
          <div style={{ fontSize: 12, color: "#94a3b8" }}>
            Quantum spin lattices, topological phases, Jordan-Wigner fermions, Bethe Ansatz limits, and Trotterized real-time dynamics.
          </div>
        </div>

        {/* Preset Selector */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <select
            defaultValue=""
            onChange={(e) => {
              if (e.target.value) {
                applyPreset(e.target.value as PhysicsPreset);
                e.target.value = "";
              }
            }}
            style={{
              padding: "5px 10px",
              borderRadius: 4,
              border: "1px solid #475569",
              background: "#1e293b",
              color: "#38bdf8",
              fontSize: 12,
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            <option value="" disabled>
              Load Physical Preset...
            </option>
            <option value="ising_crit">1D Ising Critical Point (L=8, h=J=1, c=1/2 CFT)</option>
            <option value="ising_ferro">1D Ising Ferromagnet (L=8, h=0.25J, Z₂ broken)</option>
            <option value="ising_para">1D Quantum Paramagnet (L=8, h=2.5J, polarized)</option>
            <option value="heisenberg_afm">1D Heisenberg AFM (L=8, J=1, gapless spinons)</option>
            <option value="xxz_luttinger">1D XXZ Luttinger Liquid (L=10, Δ=0.5, c=1)</option>
            <option value="j1_j2_mg">Majumdar-Ghosh Dimer VBS (L=8, J₂/J₁=0.5)</option>
            <option value="dimerized_ssh">Topological Dimerized Chain (L=8, δ=0.4)</option>
            <option value="kitaev_topo">Topological Kitaev Chain (L=8, Majorana edge modes)</option>
          </select>

          <div
            style={{
              padding: "4px 10px",
              background: "#1e293b",
              borderRadius: 4,
              border: "1px solid #334155",
              fontSize: 12,
              color: "#cbd5e1",
              fontFamily: "monospace",
            }}
          >
            <strong>{qubitCount}</strong> spins ·{" "}
            <span style={{ color: hamiltonian ? "#4ade80" : "#f59e0b" }}>
              {hamiltonian ? `${hamiltonian.terms.length} Pauli terms` : "No Hamiltonian"}
            </span>
          </div>
        </div>
      </div>

      {/* Navigation Sub-Tabs */}
      <div style={{ display: "flex", gap: 4, marginBottom: 16, borderBottom: "1px solid #1e293b", paddingBottom: 8, flexWrap: "wrap" }}>
        {[
          { id: "model", label: "Model & Lattice Geometry" },
          { id: "solvers", label: "Ground State & Spectrum (DMRG / ED)", badge: dmrgEnergy != null || groundEnergy != null ? "Solved" : undefined },
          { id: "dynamics", label: "Quench Dynamics (TEBD)", badge: energies.length ? `${energies.length} steps` : undefined },
          { id: "phase", label: "Phase Diagram & Criticality" },
          { id: "ctmrg", label: "2D Infinite CTMRG" },
        ].map((t) => (
          <button
            key={t.id}
            onClick={() => setPhysicsTab(t.id as PhysicsTab)}
            style={{
              padding: "6px 14px",
              borderRadius: 4,
              border: physicsTab === t.id ? "1px solid #38bdf8" : "1px solid transparent",
              background: physicsTab === t.id ? "#1e293b" : "transparent",
              color: physicsTab === t.id ? "#38bdf8" : "#94a3b8",
              fontSize: 13,
              fontWeight: 600,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <span>{t.label}</span>
            {t.badge && (
              <span style={{ fontSize: 10, padding: "1px 5px", borderRadius: 4, background: "#334155", color: "#e2e8f0" }}>
                {t.badge}
              </span>
            )}
          </button>
        ))}
      </div>

      {error && (
        <div
          style={{
            padding: "10px 14px",
            background: "#450a0a",
            border: "1px solid #ef4444",
            borderRadius: 6,
            color: "#fca5a5",
            fontSize: 12,
            marginBottom: 16,
          }}
        >
          {error}
        </div>
      )}

      {/* 1. MODEL & LATTICE GEOMETRY TAB */}
      {physicsTab === "model" && (
        <div style={{ display: "grid", gridTemplateColumns: "minmax(340px, 1fr) minmax(380px, 1.2fr)", gap: 16 }}>
          {/* Controls Card */}
          <div style={card}>
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 14, color: "#f8fafc" }}>
              Hamiltonian & Parameter Configuration
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 14 }}>
              <label style={fieldStyle}>
                Hamiltonian Model
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value as ModelName)}
                  style={{ ...inputStyle, width: "100%" }}
                >
                  <option value="ising">Transverse-Field Ising (TFIM)</option>
                  <option value="heisenberg">Heisenberg (XXX)</option>
                  <option value="xxz">Anisotropic XXZ</option>
                  <option value="xy">XY Model (Transverse)</option>
                  <option value="j1_j2">Frustrated J1-J2 Chain</option>
                  <option value="dimerized">Dimerized Spin-Peierls</option>
                  <option value="kitaev_chain">Topological Kitaev Chain</option>
                </select>
              </label>

              <label style={fieldStyle}>
                Lattice Dimension
                <select
                  value={dimension}
                  onChange={(e) => setDimension(Number(e.target.value) as 1 | 2)}
                  style={{ ...inputStyle, width: "100%" }}
                >
                  <option value={1}>1D Chain</option>
                  <option value={2}>2D Square Panjara</option>
                </select>
              </label>

              <label style={fieldStyle}>
                Spin Chain Length Lx
                <input
                  type="number"
                  min={2}
                  max={64}
                  value={sizes[0]}
                  onChange={(e) =>
                    setSizes([Math.max(2, Math.min(64, Number(e.target.value) || 2)), sizes[1], sizes[2]])
                  }
                  style={{ ...inputStyle, width: "100%" }}
                />
              </label>

              {dimension === 2 && (
                <label style={fieldStyle}>
                  Grid Width Ly
                  <input
                    type="number"
                    min={2}
                    max={16}
                    value={sizes[1]}
                    onChange={(e) =>
                      setSizes([sizes[0], Math.max(2, Math.min(16, Number(e.target.value) || 2)), sizes[2]])
                    }
                    style={{ ...inputStyle, width: "100%" }}
                  />
                </label>
              )}

              <label style={fieldStyle}>
                Boundary Condition
                <select
                  value={boundary}
                  onChange={(e) => setBoundary(e.target.value as Boundary)}
                  style={{ ...inputStyle, width: "100%" }}
                >
                  <option value="open">Open (OBC)</option>
                  <option value="periodic">Periodic (PBC)</option>
                </select>
              </label>

              <label style={fieldStyle}>
                Coupling Constant J₁
                <input
                  type="number"
                  step="0.1"
                  value={coupling}
                  onChange={(e) => setCoupling(Number(e.target.value) || 0)}
                  style={{ ...inputStyle, width: "100%" }}
                />
              </label>

              {(model === "ising" || model === "heisenberg" || model === "xxz" || model === "xy") && (
                <label style={fieldStyle}>
                  Transverse Field h
                  <input
                    type="number"
                    step="0.05"
                    value={field}
                    onChange={(e) => setField(Number(e.target.value) || 0)}
                    style={{ ...inputStyle, width: "100%" }}
                  />
                </label>
              )}

              {(model === "xxz" || model === "xy") && (
                <label style={fieldStyle}>
                  Anisotropy Parameter Δ / γ
                  <input
                    type="number"
                    step="0.1"
                    value={anisotropy}
                    onChange={(e) => setAnisotropy(Number(e.target.value) || 0)}
                    style={{ ...inputStyle, width: "100%" }}
                  />
                </label>
              )}

              {model === "j1_j2" && (
                <label style={fieldStyle}>
                  Frustrated J₂ Coupling
                  <input
                    type="number"
                    step="0.05"
                    value={j2Coupling}
                    onChange={(e) => setJ2Coupling(Number(e.target.value) || 0)}
                    style={{ ...inputStyle, width: "100%" }}
                  />
                </label>
              )}

              {model === "dimerized" && (
                <label style={fieldStyle}>
                  Dimerization Factor δ
                  <input
                    type="number"
                    step="0.05"
                    value={dimerizationDelta}
                    onChange={(e) => setDimerizationDelta(Number(e.target.value) || 0)}
                    style={{ ...inputStyle, width: "100%" }}
                  />
                </label>
              )}

              {model === "kitaev_chain" && (
                <>
                  <label style={fieldStyle}>
                    Pairing Potential Δ_sc
                    <input
                      type="number"
                      step="0.1"
                      value={pairingDelta}
                      onChange={(e) => setPairingDelta(Number(e.target.value) || 0)}
                      style={{ ...inputStyle, width: "100%" }}
                    />
                  </label>
                  <label style={fieldStyle}>
                    Chemical Potential μ
                    <input
                      type="number"
                      step="0.1"
                      value={chemicalPotential}
                      onChange={(e) => setChemicalPotential(Number(e.target.value) || 0)}
                      style={{ ...inputStyle, width: "100%" }}
                    />
                  </label>
                </>
              )}
            </div>

            <div style={{ display: "flex", gap: 10, marginTop: 14, flexWrap: "wrap" }}>
              <Button variant="primary" onClick={() => void generateHamiltonianModel()} disabled={busy !== null}>
                {busy === "hamiltonian" ? "Building..." : "Build Spin Hamiltonian"}
              </Button>
              <Button variant="secondary" onClick={() => void generateLattice()} disabled={busy !== null}>
                {busy === "preview" ? "Previewing..." : "Preview Lattice"}
              </Button>
              <Button variant="ghost" onClick={exportPythonModel} disabled={!hamiltonian}>
                Export Python Code
              </Button>
              <Button variant="ghost" onClick={exportLatexTable} disabled={!hamiltonian}>
                Export LaTeX Table
              </Button>
            </div>
          </div>

          {/* Theoretical Insight & Lattice Visual */}
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {(() => {
              const insight = getPhaseInsight();
              return (
                <div style={{ ...card, padding: 14, borderLeft: "4px solid #38bdf8" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                    <strong style={{ fontSize: 14, color: "#f8fafc" }}>{insight.title}</strong>
                    <span
                      style={{
                        fontSize: 10,
                        padding: "2px 6px",
                        borderRadius: 4,
                        background: "#1e293b",
                        color: "#38bdf8",
                        border: "1px solid #334155",
                      }}
                    >
                      {insight.badge}
                    </span>
                  </div>
                  <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.4, marginBottom: 6 }}>
                    {insight.description}
                  </div>
                  <div style={{ fontSize: 11, color: "#64748b", display: "grid", gap: 2, fontFamily: "monospace" }}>
                    <div>Order Parameter: <span style={{ color: "#e2e8f0" }}>{insight.orderParameter}</span></div>
                    <div>CFT Central Charge: <span style={{ color: "#4ade80" }}>{insight.centralCharge}</span></div>
                    <div>Energy Gap: <span style={{ color: "#38bdf8" }}>{insight.gap}</span></div>
                  </div>
                </div>
              );
            })()}

            {graph && (
              <div style={card}>
                <div style={{ fontSize: 13, fontWeight: 600, color: "#94a3b8", marginBottom: 8 }}>
                  Spin Network Geometry ({graph.sites.length} sites, {graph.edges.length} bonds)
                </div>
                <LatticeCanvas graph={graph} />
              </div>
            )}
          </div>
        </div>
      )}

      {/* 2. GROUND STATE & SPECTRUM TAB */}
      {physicsTab === "solvers" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={card}>
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 12, color: "#f8fafc" }}>
              Variational Ground State Solvers (DMRG & Exact Diagonalization)
            </div>
            <div style={{ display: "flex", gap: 16, alignItems: "flex-end", flexWrap: "wrap", marginBottom: 16 }}>
              <label style={fieldStyle}>
                Bond Dimension χ
                <input
                  type="number"
                  min={1}
                  max={64}
                  value={bondDim}
                  onChange={(e) => setBondDim(Number(e.target.value) || 16)}
                  style={inputStyle}
                />
              </label>
              <label style={fieldStyle}>
                DMRG Sweeps
                <input
                  type="number"
                  min={1}
                  max={30}
                  value={sweeps}
                  onChange={(e) => setSweeps(Number(e.target.value) || 4)}
                  style={inputStyle}
                />
              </label>
              <Button variant="primary" onClick={() => void runDMRGSolver()} disabled={busy !== null || !hamiltonian}>
                {busy === "dmrg" ? "Running DMRG..." : "Run DMRG Optimizer"}
              </Button>
              <Button
                variant="ghost"
                onClick={() => void runExactED()}
                disabled={busy !== null || !hamiltonian || qubitCount > 12}
              >
                {busy === "ground" ? "Diagonalizing..." : "Exact Diagonalization (N ≤ 12)"}
              </Button>
            </div>

            {/* Results Grid */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                gap: 12,
                padding: 14,
                background: "#0b0f19",
                borderRadius: 6,
                border: "1px solid #1e293b",
                marginBottom: 16,
              }}
            >
              <div>
                <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase" }}>DMRG Ground Energy E₀</div>
                <div style={{ fontSize: 24, fontWeight: 700, color: "#4ade80", fontFamily: "monospace" }}>
                  {dmrgEnergy != null ? Number(dmrgEnergy).toFixed(7) : "—"}
                </div>
                {dmrgEnergy != null && (
                  <div style={{ fontSize: 11, color: "#94a3b8" }}>Per site: {(Number(dmrgEnergy) / siteCount).toFixed(6)}</div>
                )}
              </div>

              <div>
                <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase" }}>Exact ED Ground Energy</div>
                <div style={{ fontSize: 24, fontWeight: 700, color: "#38bdf8", fontFamily: "monospace" }}>
                  {groundEnergy != null ? Number(groundEnergy).toFixed(7) : "—"}
                </div>
                {groundEnergy != null && (
                  <div style={{ fontSize: 11, color: "#94a3b8" }}>Per site: {(Number(groundEnergy) / siteCount).toFixed(6)}</div>
                )}
              </div>

              <div>
                <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase" }}>Excitation Gap ΔE = E₁ - E₀</div>
                <div
                  style={{
                    fontSize: 24,
                    fontWeight: 700,
                    color: energyGap != null && energyGap > 0.05 ? "#f59e0b" : "#38bdf8",
                    fontFamily: "monospace",
                  }}
                >
                  {energyGap != null ? energyGap.toFixed(6) : "—"}
                </div>
                <div style={{ fontSize: 11, color: "#94a3b8" }}>
                  {energyGap != null
                    ? energyGap < 0.05
                      ? "Gapless critical excitation"
                      : "Gapped phase"
                    : "Available via Exact ED"}
                </div>
              </div>

              <div>
                <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase" }}>Analytical / Bethe Limit</div>
                <div style={{ fontSize: 16, fontWeight: 600, color: "#cbd5e1", fontFamily: "monospace", marginTop: 4 }}>
                  {model === "heisenberg"
                    ? ((-0.443147 * siteCount).toFixed(5) + " (Bethe)")
                    : model === "j1_j2" && j2Coupling === 0.5
                    ? ((-0.375 * siteCount).toFixed(5) + " (MG Exact)")
                    : "Model specific"}
                </div>
                <div style={{ fontSize: 11, color: "#64748b" }}>Infinite system extrapolation</div>
              </div>
            </div>

            {/* Excitation Ladder SVG if ED available */}
            {lowestEigenvalues && lowestEigenvalues.length > 1 && (
              <div style={{ background: "#0b0f19", border: "1px solid #1e293b", borderRadius: 6, padding: 14 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8", marginBottom: 8, fontFamily: "monospace" }}>
                  LOW-LYING ENERGY EIGENVALUE SPECTRUM LADDER (N={qubitCount})
                </div>
                <svg viewBox="0 0 540 140" style={{ width: "100%", height: 140, display: "block" }}>
                  {(() => {
                    const eMin = lowestEigenvalues[0];
                    const eMax = lowestEigenvalues[lowestEigenvalues.length - 1];
                    const span = Math.max(1e-9, eMax - eMin);
                    return (
                      <>
                        {/* Axes */}
                        <line x1="60" y1="20" x2="60" y2="120" stroke="#334155" strokeWidth="1.5" />
                        <line x1="60" y1="120" x2="500" y2="120" stroke="#334155" strokeWidth="1.5" />
                        <text x="50" y="15" fill="#64748b" fontSize="10" fontFamily="monospace" textAnchor="end">E_k</text>

                        {/* Energy Levels */}
                        {lowestEigenvalues.slice(0, 8).map((val, k) => {
                          const yPos = 120 - ((val - eMin) / span) * 95;
                          const isGround = k === 0;
                          return (
                            <g key={k}>
                              <line
                                x1="80"
                                y1={yPos}
                                x2="480"
                                y2={yPos}
                                stroke={isGround ? "#4ade80" : "#38bdf8"}
                                strokeWidth={isGround ? "2" : "1.2"}
                                strokeDasharray={isGround ? undefined : "3 2"}
                              />
                              <text
                                x="75"
                                y={yPos + 3}
                                textAnchor="end"
                                fill={isGround ? "#4ade80" : "#94a3b8"}
                                fontSize="9"
                                fontFamily="monospace"
                              >
                                {val.toFixed(4)}
                              </text>
                              <text
                                x="485"
                                y={yPos + 3}
                                fill={isGround ? "#4ade80" : "#64748b"}
                                fontSize="9"
                                fontFamily="monospace"
                              >
                                {isGround ? "E₀ (Ground)" : `E_${k}`}
                              </text>
                            </g>
                          );
                        })}
                      </>
                    );
                  })()}
                </svg>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 3. QUENCH DYNAMICS (TEBD) TAB */}
      {physicsTab === "dynamics" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={card}>
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 12, color: "#f8fafc" }}>
              Real-Time Quantum Quench Dynamics (Trotterized TEBD)
            </div>
            <div style={{ display: "flex", gap: 16, alignItems: "flex-end", flexWrap: "wrap", marginBottom: 16 }}>
              <label style={fieldStyle}>
                Time Step dt
                <input
                  type="number"
                  step="0.01"
                  value={dt}
                  onChange={(e) => setDt(Number(e.target.value) || 0.05)}
                  style={inputStyle}
                />
              </label>
              <label style={fieldStyle}>
                Evolution Steps
                <input
                  type="number"
                  min={1}
                  max={500}
                  value={steps}
                  onChange={(e) => setSteps(Number(e.target.value) || 20)}
                  style={inputStyle}
                />
              </label>
              <Button variant="primary" onClick={() => void runTEBDEvolution()} disabled={busy !== null || !hamiltonian}>
                {busy === "tebd" ? "Evolving MPS..." : "Run TEBD Simulation"}
              </Button>
            </div>

            {energies.length > 0 && (
              <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                <div>
                  <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 6, fontFamily: "monospace" }}>
                    ENERGY TRAJECTORY E(t) OVER EVOLUTION TIME
                  </div>
                  <EnergyChart values={energies} />
                </div>

                {bondDimHistory.length > 0 && (
                  <div style={{ background: "#080d1a", border: "1px solid #334155", borderRadius: 8, padding: 12 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, fontFamily: "monospace", color: "#94a3b8", marginBottom: 6 }}>
                      <span>ENTANGLEMENT GROWTH: Bond Dimension χ(t)</span>
                      <span>Initial χ = {bondDimHistory[0]} → Final χ = {bondDimHistory[bondDimHistory.length - 1]}</span>
                    </div>
                    {/* Compact Bond Dim Growth Bar */}
                    <div style={{ display: "flex", gap: 2, alignItems: "flex-end", height: 60, padding: "4px 0" }}>
                      {bondDimHistory.map((dimVal, idx) => {
                        const maxDim = Math.max(...bondDimHistory, 1);
                        const hPct = Math.max(8, (dimVal / maxDim) * 100);
                        return (
                          <div
                            key={idx}
                            title={`Step ${idx}: χ = ${dimVal}`}
                            style={{
                              flex: 1,
                              height: `${hPct}%`,
                              background: "#38bdf8",
                              borderRadius: 2,
                              opacity: 0.6 + (dimVal / maxDim) * 0.4,
                            }}
                          />
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* 4. PHASE DIAGRAM & CRITICALITY TAB */}
      {physicsTab === "phase" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={card}>
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 8, color: "#f8fafc" }}>
              Theoretical Phase Diagram & Quantum Critical Scaling
            </div>
            <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 14 }}>
              Analytical order parameters, conformal invariance, and universality classes for the active model.
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 14 }}>
              <div style={{ padding: 14, background: "#0b0f19", border: "1px solid #1e293b", borderRadius: 6 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: "#38bdf8", marginBottom: 4 }}>
                  Quantum Critical Point (QCP)
                </div>
                <div style={{ fontSize: 11, color: "#94a3b8", lineHeight: 1.5 }}>
                  {model === "ising" && "Critical parameter: h = J = 1.0. The gap closes as Δ ~ 2|h - J| with dynamic critical exponent z = 1."}
                  {model === "heisenberg" && "SU(2) symmetric gapless point with power-law spin correlations decay."}
                  {model === "xxz" && "BKT transition at Δ = 1. Gapless Luttinger liquid for -1 < Δ ≤ 1."}
                  {model === "j1_j2" && "Transition from gapless to gapped dimerized phase occurs at (J₂/J₁)_c ≈ 0.2411. Exact VBS at 0.5."}
                  {model === "kitaev_chain" && "Topological phase boundary occurs at |μ| = 2t. Majorana zero modes exist inside the gap."}
                  {model !== "ising" && model !== "heisenberg" && model !== "xxz" && model !== "j1_j2" && model !== "kitaev_chain" && "Continuous tuning parameter phase boundary."}
                </div>
              </div>

              <div style={{ padding: 14, background: "#0b0f19", border: "1px solid #1e293b", borderRadius: 6 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: "#4ade80", marginBottom: 4 }}>
                  Conformal Field Theory (CFT) Central Charge
                </div>
                <div style={{ fontSize: 11, color: "#94a3b8", lineHeight: 1.5 }}>
                  Calabrese-Cardy bipartite entanglement scaling:
                  <div style={{ fontFamily: "monospace", color: "#f8fafc", margin: "4px 0" }}>
                    S_vN(x) = (c/6) ln[(L/π) sin(π x/L)] + s₀
                  </div>
                  Active Model Central Charge: <strong style={{ color: "#4ade80" }}>{getPhaseInsight().centralCharge}</strong>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 5. 2D INFINITE CTMRG TAB */}
      {physicsTab === "ctmrg" && (
        <div style={card}>
          <CTMRGLabPanel
            dimensions={activeDimensions}
            model={model === "ising" || model === "heisenberg" || model === "xxz" ? model : "ising"}
            coupling={coupling}
            field={field}
            anisotropy={anisotropy}
            onResult={(res, req, label) => {
              setLatestOutput(res);
              addExperiment({
                label: `Physics · CTMRG · ${label}`,
                source: "physics",
                kind: "ctmrg",
                status: "done",
                request: req,
                result: res,
                provenance: res.provenance,
              });
            }}
          />
        </div>
      )}

      {/* Background job progress indicator */}
      {jobProgress && busy !== null && (
        <div style={{ marginTop: 12, fontSize: 11, color: "#38bdf8", fontFamily: "monospace" }}>
          STATUS: {jobProgress.status} ({jobProgress.job_id}) · PROGRESS: {Math.round(jobProgress.progress * 100)}%
        </div>
      )}
    </main>
  );
}
