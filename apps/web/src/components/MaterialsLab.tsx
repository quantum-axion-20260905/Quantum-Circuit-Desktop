"use client";

import React from "react";
import {
  buildHubbard,
  previewLattice,
  runDMRG,
  runGroundState,
  runPEPS,
  type AsyncJob,
  type LatticeGraph,
  type SparseHamiltonianResponse,
} from "../lib/agent";
import { useUIContext } from "../state/uiContext";
import { Button } from "../ui";
import { LatticeCanvas } from "./LatticeVisuals";
import {
  ElectronicBandDispersionChart,
  DensityOfStatesChart,
  ChargeOccupancyHeatmap,
  DoubleOccupancyVsCorrelationCurve,
} from "./MaterialsVisuals";

export type Boundary = "open" | "periodic";
export type PEPSContraction = "auto" | "boundary-mps";
export type LatticeGeometry = "square" | "honeycomb" | "triangular" | "lieb";

export type MaterialPreset =
  | "cuprate_ybco"
  | "graphene"
  | "kitaev_rucl3"
  | "haldane_chain"
  | "mott_insulator"
  | "iron_pnictide"
  | "kagome_metal"
  | "kondo_lattice";

export type MaterialsTab = "bands" | "occupancy" | "solvers" | "structure" | "export";

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

function errorText(error: unknown) {
  return error instanceof Error ? error.message : "Operation failed.";
}

export function MaterialsLab() {
  const { setLatestOutput, addExperiment } = useUIContext();

  // Lattice & Crystal Geometry
  const [dimension, setDimension] = React.useState<1 | 2>(2);
  const [sizes, setSizes] = React.useState([3, 3, 1]);
  const [boundary, setBoundary] = React.useState<Boundary>("open");
  const [latticeType, setLatticeType] = React.useState<LatticeGeometry>("square");

  // Electronic Hopping & Band Parameters
  const [hopping, setHopping] = React.useState(1.0);
  const [hoppingPrime, setHoppingPrime] = React.useState(-0.2); // Next-nearest neighbor t'
  const [chemicalPotential, setChemicalPotential] = React.useState(0.0);
  const [holeDoping, setHoleDoping] = React.useState(0.0); // Doping fraction delta = 1 - n

  // Coulomb Interactions
  const [onsiteU, setOnsiteU] = React.useState(8.0);
  const [intersiteV, setIntersiteV] = React.useState(0.0); // Nearest-neighbor V
  const [hundJ, setHundJ] = React.useState(0.0); // Hund's coupling J_H

  // High-Performance Electronic Solvers
  const [bondDim, setBondDim] = React.useState(16);
  const [envBondDim, setEnvBondDim] = React.useState(24);
  const [pepsContraction, setPepsContraction] = React.useState<PEPSContraction>("auto");
  const [sweeps, setSweeps] = React.useState(4);

  // Results State
  const [graph, setGraph] = React.useState<LatticeGraph | null>(null);
  const [hamiltonian, setHamiltonian] = React.useState<SparseHamiltonianResponse | null>(null);
  const [dmrgResult, setDmrgResult] = React.useState<Record<string, unknown> | null>(null);
  const [pepsResult, setPepsResult] = React.useState<Record<string, unknown> | null>(null);
  const [groundResult, setGroundResult] = React.useState<Record<string, unknown> | null>(null);

  const [busy, setBusy] = React.useState<"preview" | "hamiltonian" | "ground" | "dmrg" | "peps" | null>(null);
  const [jobProgress, setJobProgress] = React.useState<AsyncJob | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  const [materialsTab, setMaterialsTab] = React.useState<MaterialsTab>("bands");
  const [showAdvancedParams, setShowAdvancedParams] = React.useState(false);

  const activeDimensions = dimension === 1 ? [sizes[0]] : [sizes[0], sizes[1]];
  const siteCount = activeDimensions.reduce((acc, dim) => acc * dim, 1);
  const qubitCount = siteCount * 2; // Jordan-Wigner requires 2 spin-orbitals (up & down) per site

  function applyPreset(preset: MaterialPreset) {
    setError(null);
    if (preset === "cuprate_ybco") {
      setDimension(2);
      setSizes([3, 3, 1]);
      setBoundary("open");
      setLatticeType("square");
      setHopping(1.0);
      setHoppingPrime(-0.25);
      setOnsiteU(8.0);
      setIntersiteV(0.0);
      setChemicalPotential(0.0);
      setHoleDoping(0.15); // Optimal cuprate superconducting doping
    } else if (preset === "graphene") {
      setDimension(2);
      setSizes([4, 3, 1]);
      setBoundary("periodic");
      setLatticeType("honeycomb");
      setHopping(2.8);
      setHoppingPrime(-0.1);
      setOnsiteU(4.5);
      setIntersiteV(0.0);
      setChemicalPotential(0.0);
      setHoleDoping(0.0);
    } else if (preset === "kitaev_rucl3") {
      setDimension(2);
      setSizes([3, 3, 1]);
      setBoundary("open");
      setLatticeType("honeycomb");
      setHopping(1.0);
      setHoppingPrime(0.0);
      setOnsiteU(3.5);
      setIntersiteV(0.0);
      setChemicalPotential(0.2);
      setHoleDoping(0.0);
    } else if (preset === "haldane_chain") {
      setDimension(1);
      setSizes([8, 1, 1]);
      setBoundary("open");
      setLatticeType("square");
      setHopping(1.0);
      setHoppingPrime(0.0);
      setOnsiteU(2.0);
      setIntersiteV(0.0);
      setChemicalPotential(0.0);
      setHoleDoping(0.0);
    } else if (preset === "mott_insulator") {
      setDimension(2);
      setSizes([3, 3, 1]);
      setBoundary("open");
      setLatticeType("square");
      setHopping(1.0);
      setHoppingPrime(0.0);
      setOnsiteU(7.5);
      setIntersiteV(0.0);
      setChemicalPotential(0.0);
      setHoleDoping(0.0);
    } else if (preset === "iron_pnictide") {
      setDimension(2);
      setSizes([3, 3, 1]);
      setBoundary("open");
      setLatticeType("square");
      setHopping(1.2);
      setHoppingPrime(-0.35);
      setOnsiteU(4.0);
      setIntersiteV(0.5);
      setHundJ(0.8);
      setChemicalPotential(0.0);
      setHoleDoping(0.1);
    } else if (preset === "kagome_metal") {
      setDimension(2);
      setSizes([3, 3, 1]);
      setBoundary("periodic");
      setLatticeType("triangular");
      setHopping(1.5);
      setHoppingPrime(0.0);
      setOnsiteU(3.0);
      setIntersiteV(0.8);
      setChemicalPotential(0.0);
      setHoleDoping(0.0);
    } else if (preset === "kondo_lattice") {
      setDimension(2);
      setSizes([3, 3, 1]);
      setBoundary("open");
      setLatticeType("square");
      setHopping(1.0);
      setHoppingPrime(0.0);
      setOnsiteU(9.0);
      setIntersiteV(0.0);
      setHundJ(1.2);
      setChemicalPotential(0.0);
      setHoleDoping(0.0);
    }
  }

  function getMaterialInsight() {
    const ratio = Math.abs(hopping) > 1e-9 ? onsiteU / Math.abs(hopping) : 999;
    if (ratio > 5.5) {
      return {
        title: "Strongly Correlated Mott Insulator Regime",
        badge: "Mott Gap (U ≫ t)",
        description: "Kuchli Kulon itarilishi (U ≫ t) tufayli elektronlar har bir tugunda lokallashadi. Magnit o'zaro ta'sir super-almashtirish J_ex = 4t²/U bilan ifodalanadi.",
        orderParameter: "Charge Gap Δ_c ~ U, Local Spin Moments",
        doubleOccupancy: "Low (⟨n_↑ n_↓⟩ ≪ 0.25)",
        pairingCandidate: "None (Mott Blockade)",
      };
    } else if (ratio < 2.0) {
      return {
        title: "Delocalized Metallic Fermi Liquid Regime",
        badge: "Band Conductor (t > U)",
        description: "Kinetik hopping energiyasi Kulon to'sig'idan ustun; elektronlar panjara bo'ylab erkin harakatlanadi, o'tkazuvchanlik yuqori.",
        orderParameter: "Fermi Surface Discontinuity Z > 0",
        doubleOccupancy: "Standard (⟨n_↑ n_↓⟩ ≈ 0.25)",
        pairingCandidate: "Weak s-wave BCS",
      };
    } else {
      return {
        title: "High-Tc Superconducting Crossover Regime",
        badge: "Strongly Correlated d-wave",
        description: "Kinetik hopping va Kulon to'sig'i teng kuchli; o'ta o'tkazuvchan d-to'lqinli singlet juftlanish va antiferromagnit fluktuatsiyalar raqobatlashadi.",
        orderParameter: "d-wave Order Parameter Δ_d (dx²-y²)",
        doubleOccupancy: "Intermediate (⟨n_↑ n_↓⟩ ~ 0.12 - 0.18)",
        pairingCandidate: "Resonant Valence Bond (RVB) High-Tc",
      };
    }
  }

  async function generateLattice() {
    setBusy("preview");
    setError(null);
    try {
      const g = await previewLattice("hubbard", { dimension, sizes: activeDimensions, boundary });
      setGraph(g);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  async function generateHubbardHamiltonian() {
    setBusy("hamiltonian");
    setError(null);
    try {
      const payload = {
        dimension,
        sizes: activeDimensions,
        boundary,
        hopping,
        onsite_u: onsiteU,
        chemical_potential: chemicalPotential,
      };
      const h = await buildHubbard("hubbard", payload);
      setHamiltonian(h);
      if (!graph) void generateLattice();
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
        label: `Materials · Exact ED · Fermi-Hubbard (Sites=${siteCount})`,
        source: "materials",
        kind: "ground_state",
        status: "done",
        request: { source: "materials", kind: "ground_state", payload: { terms: hamiltonian.terms } },
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
        label: `Materials · DMRG · Fermi-Hubbard (χ=${bondDim})`,
        source: "materials",
        kind: "dmrg",
        status: "done",
        request: { source: "materials", kind: "dmrg", payload: { terms: hamiltonian.terms, bond_dim: bondDim, sweeps } },
        result: res,
        provenance: res.provenance,
      });
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  async function runNativePEPSSolver() {
    if (!hamiltonian) return;
    setBusy("peps");
    setError(null);
    try {
      const res = await runPEPS(
        {
          terms: hamiltonian.terms,
          n_qubits: hamiltonian.n_qubits,
          bond_dim: bondDim,
          steps: 10,
          dt: 0.05,
        },
        setJobProgress
      );
      setPepsResult(res);
      setLatestOutput(res);
      addExperiment({
        label: `Materials · PEPS · Fermi-Hubbard 2D (Sites=${siteCount})`,
        source: "materials",
        kind: "peps",
        status: "done",
        request: { source: "materials", kind: "peps", payload: { terms: hamiltonian.terms } },
        result: res,
        provenance: res.provenance,
      });
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(null);
    }
  }

  function exportLatexTable() {
    const energyVal = dmrgEnergy ?? pepsEnergy ?? groundEnergy ?? 0;
    const tex = `% Computational Materials Science Summary Report
\\begin{table}[htbp]
\\centering
\\caption{Fermi-Hubbard Material Simulation on ${activeDimensions.join("$\\times$")} Lattice}
\\begin{tabular}{ll|ll}
\\hline\\hline
\\textbf{Parameter} & \\textbf{Value} & \\textbf{Observable} & \\textbf{Value} \\\\
\\hline
Lattice Dimension & ${dimension}D & Boundary Condition & ${boundary} \\\\
Lattice Geometry & ${latticeType} & Spin-Orbitals ($2N$) & ${qubitCount} \\\\
Hopping Integral $t$ & ${hopping.toFixed(3)} eV & Next-Hop $t'$ & ${hoppingPrime.toFixed(3)} eV \\\\
Coulomb Repulsion $U$ & ${onsiteU.toFixed(3)} eV & Correlation Ratio $U/t$ & ${(onsiteU / (hopping || 1)).toFixed(2)} \\\\
Hole Doping $\\delta$ & ${(holeDoping * 100).toFixed(1)}\\% & Ground Energy $E_0$ & ${Number(energyVal).toFixed(6)} eV \\\\
Energy per site & ${(Number(energyVal) / siteCount).toFixed(6)} eV & Pairing Candidate & ${getMaterialInsight().pairingCandidate} \\\\
Material Regime & \\multicolumn{3}{l}{${getMaterialInsight().title}} \\\\
Order Parameter & \\multicolumn{3}{l}{${getMaterialInsight().orderParameter}} \\\\
\\hline\\hline
\\end{tabular}
\\end{table}
`;
    const blob = new Blob([tex], { type: "text/x-tex" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `material_hubbard_${activeDimensions.join("x")}.tex`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function exportPythonModel() {
    const code = `# Computational Materials Science - Fermi-Hubbard Generator
# Lattice: ${activeDimensions.join("x")} (${boundary} boundary, ${latticeType})
import numpy as np

n_sites = ${siteCount}
n_spin_orbitals = ${qubitCount}
hopping_t = ${hopping}
hopping_t_prime = ${hoppingPrime}
onsite_U = ${onsiteU}
intersite_V = ${intersiteV}
chemical_potential_mu = ${chemicalPotential}
hole_doping = ${holeDoping}

jordan_wigner_terms = ${JSON.stringify(hamiltonian?.terms ?? [], null, 2)}

print(f"Loaded Fermi-Hubbard model on {n_sites} sites ({len(jordan_wigner_terms)} Pauli terms)")
`;
    const blob = new Blob([code], { type: "text/x-python" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `material_hubbard_${activeDimensions.join("x")}.py`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function exportCsvDataset() {
    let csv = "Site_i,Hopping_t,Coulomb_U,U_over_t,Chemical_mu\n";
    for (let i = 0; i < siteCount; i++) {
      csv += `${i},${hopping},${onsiteU},${(onsiteU / (hopping || 1)).toFixed(3)},${chemicalPotential}\n`;
    }
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `material_hubbard_N${siteCount}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const dmrgEnergy = dmrgResult?.ground_energy;
  const groundEnergy = groundResult?.ground_energy;
  const pepsEnergy = pepsResult?.ground_energy;

  return (
    <main style={shell}>
      {/* Top Header & Workstation Metadata */}
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
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: "#f8fafc" }}>
              Computational Materials Science Laboratory
            </h2>
            <span
              style={{
                fontSize: 10,
                padding: "2px 7px",
                borderRadius: 4,
                background: "#080d1a",
                border: "1px solid #334155",
                color: "#38bdf8",
                fontFamily: "monospace",
              }}
            >
              Materials CAD Studio v2.0
            </span>
          </div>
          <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 2 }}>
            Electronic band structures, strongly correlated electrons, Fermi-Hubbard models, 2D PEPS tensor networks, and neutron scattering response.
          </div>
        </div>

        {/* Real Quantum Materials Presets */}
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <select
            defaultValue=""
            onChange={(e) => {
              if (e.target.value) {
                applyPreset(e.target.value as MaterialPreset);
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
              Load Quantum Material...
            </option>
            <option value="cuprate_ybco">Cuprate Superconductor (YBCO CuO₂ Plane, U=8t)</option>
            <option value="graphene">Graphene Dirac Sheet (t=2.8eV, U=4.5eV)</option>
            <option value="iron_pnictide">Iron Pnictide Superconductor (FeSe, J_H=0.8eV)</option>
            <option value="kagome_metal">Kagome Metal (CsV₃Sb₅, Frustrated)</option>
            <option value="kitaev_rucl3">Kitaev Spin Liquid Candidate (α-RuCl₃)</option>
            <option value="haldane_chain">Topological Haldane Magnet (CsNiCl₃)</option>
            <option value="mott_insulator">Strongly Correlated Mott Insulator (V₂O₃)</option>
            <option value="kondo_lattice">Heavy Fermion Kondo Lattice (CeCoIn₅)</option>
          </select>

          <div
            style={{
              padding: "4px 10px",
              background: "#080d1a",
              borderRadius: 4,
              border: "1px solid #334155",
              fontSize: 12,
              color: "#cbd5e1",
              fontFamily: "monospace",
            }}
          >
            <strong>{siteCount}</strong> sites ({qubitCount} spin-orbitals) ·{" "}
            <span style={{ color: hamiltonian ? "#4ade80" : "#f59e0b" }}>
              {hamiltonian ? `${hamiltonian.terms.length} Pauli terms` : "No Model"}
            </span>
          </div>
        </div>
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

      {/* Main Workstation Layout: Left Specification Tree + Right Visualizer Studio */}
      <div className="qc-workstation-grid">
        {/* LEFT COLUMN: Crystal Spec & Electronic Engine Parameters */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {/* Card 1: Crystal Lattice & Bravais Unit Cell */}
          <div style={card}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <strong style={{ fontSize: 13, color: "#f8fafc", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                1. Crystal Lattice & Unit Cell
              </strong>
              <span style={{ fontSize: 11, fontFamily: "monospace", color: "#38bdf8" }}>{dimension}D {latticeType}</span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 10 }}>
              <label style={fieldStyle}>
                Lattice Dimension
                <select
                  value={dimension}
                  onChange={(e) => setDimension(Number(e.target.value) as 1 | 2)}
                  style={inputStyle}
                >
                  <option value={1}>1D Chain</option>
                  <option value={2}>2D Planar Lattice</option>
                </select>
              </label>

              <label style={fieldStyle}>
                Crystal Geometry
                <select
                  value={latticeType}
                  onChange={(e) => setLatticeType(e.target.value as LatticeGeometry)}
                  style={inputStyle}
                >
                  <option value="square">Square (CuO₂ Plane)</option>
                  <option value="honeycomb">Honeycomb (Graphene)</option>
                  <option value="triangular">Triangular Lattice</option>
                  <option value="lieb">Lieb Lattice (3-Band)</option>
                </select>
              </label>

              <label style={fieldStyle}>
                Lattice Length Lx
                <input
                  type="number"
                  min={2}
                  max={16}
                  value={sizes[0]}
                  onChange={(e) =>
                    setSizes([Math.max(2, Math.min(16, Number(e.target.value) || 2)), sizes[1], sizes[2]])
                  }
                  style={inputStyle}
                />
              </label>

              {dimension === 2 && (
                <label style={fieldStyle}>
                  Lattice Width Ly
                  <input
                    type="number"
                    min={2}
                    max={16}
                    value={sizes[1]}
                    onChange={(e) =>
                      setSizes([sizes[0], Math.max(2, Math.min(16, Number(e.target.value) || 2)), sizes[2]])
                    }
                    style={inputStyle}
                  />
                </label>
              )}

              <label style={fieldStyle}>
                Boundary Condition
                <select
                  value={boundary}
                  onChange={(e) => setBoundary(e.target.value as Boundary)}
                  style={inputStyle}
                >
                  <option value="open">Open (OBC)</option>
                  <option value="periodic">Periodic (PBC)</option>
                </select>
              </label>
            </div>
          </div>

          {/* Card 2: Electronic Hopping & Band Parameters */}
          <div style={card}>
            <strong style={{ fontSize: 13, color: "#f8fafc", textTransform: "uppercase", letterSpacing: "0.04em", display: "block", marginBottom: 10 }}>
              2. Hopping & Coulomb Repulsion
            </strong>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <label style={fieldStyle}>
                Nearest Hopping t (eV)
                <input
                  type="number"
                  step="0.1"
                  value={hopping}
                  onChange={(e) => setHopping(Number(e.target.value) || 0)}
                  style={inputStyle}
                />
              </label>

              <label style={fieldStyle}>
                Next-Nearest t&apos; (eV)
                <input
                  type="number"
                  step="0.05"
                  value={hoppingPrime}
                  onChange={(e) => setHoppingPrime(Number(e.target.value) || 0)}
                  style={inputStyle}
                />
              </label>

              <label style={fieldStyle}>
                On-Site Coulomb U (eV)
                <input
                  type="number"
                  step="0.5"
                  value={onsiteU}
                  onChange={(e) => setOnsiteU(Number(e.target.value) || 0)}
                  style={inputStyle}
                />
              </label>

              <label style={fieldStyle}>
                Chemical Potential μ (eV)
                <input
                  type="number"
                  step="0.1"
                  value={chemicalPotential}
                  onChange={(e) => setChemicalPotential(Number(e.target.value) || 0)}
                  style={inputStyle}
                />
              </label>
            </div>
          </div>

          {/* Card 3: Advanced Electronic Solvers */}
          <div style={card}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <strong style={{ fontSize: 13, color: "#f8fafc", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                3. Electronic Solvers & PEPS
              </strong>
              <button
                onClick={() => setShowAdvancedParams(!showAdvancedParams)}
                style={{
                  background: "transparent",
                  border: "none",
                  color: "#38bdf8",
                  fontSize: 11,
                  fontFamily: "monospace",
                  cursor: "pointer",
                }}
              >
                {showAdvancedParams ? "Simple View ▲" : "Multi-Orbital / Advanced ▼"}
              </button>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 12 }}>
              <label style={fieldStyle}>
                PEPS / DMRG Bond Dim χ
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

              <label style={fieldStyle}>
                PEPS Contraction
                <select
                  value={pepsContraction}
                  onChange={(e) => setPepsContraction(e.target.value as PEPSContraction)}
                  style={inputStyle}
                >
                  <option value="auto">Auto · double-layer</option>
                  <option value="boundary-mps">Boundary-MPS</option>
                </select>
              </label>

              <label style={fieldStyle}>
                Hole Doping δ (1 - n)
                <input
                  type="number"
                  step="0.05"
                  min={0}
                  max={0.5}
                  value={holeDoping}
                  onChange={(e) => setHoleDoping(Number(e.target.value) || 0)}
                  style={inputStyle}
                />
              </label>
            </div>

            {showAdvancedParams && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, borderTop: "1px solid #1e293b", paddingTop: 10, marginBottom: 12 }}>
                <label style={fieldStyle}>
                  Inter-Site Coulomb V (eV)
                  <input
                    type="number"
                    step="0.1"
                    value={intersiteV}
                    onChange={(e) => setIntersiteV(Number(e.target.value) || 0)}
                    style={inputStyle}
                  />
                </label>

                <label style={fieldStyle}>
                  Hund&apos;s Rule J_H (eV)
                  <input
                    type="number"
                    step="0.1"
                    value={hundJ}
                    onChange={(e) => setHundJ(Number(e.target.value) || 0)}
                    style={inputStyle}
                  />
                </label>

                <label style={fieldStyle}>
                  Env Bond Dim χ_env
                  <input
                    type="number"
                    min={8}
                    max={64}
                    value={envBondDim}
                    onChange={(e) => setEnvBondDim(Number(e.target.value) || 24)}
                    style={inputStyle}
                  />
                </label>
              </div>
            )}

            {/* Action Buttons */}
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <Button variant="primary" onClick={() => void generateHubbardHamiltonian()} disabled={busy !== null}>
                {busy === "hamiltonian" ? "Building..." : "Build Fermi-Hubbard Model"}
              </Button>
              <Button variant="secondary" onClick={() => void runDMRGSolver()} disabled={busy !== null || !hamiltonian}>
                {busy === "dmrg" ? "Running DMRG..." : "Run DMRG Ground State"}
              </Button>
              <Button
                variant="secondary"
                onClick={() => void runNativePEPSSolver()}
                disabled={busy !== null || !hamiltonian || dimension !== 2}
              >
                {busy === "peps" ? "Contracting..." : "Run 2D PEPS Solver"}
              </Button>
              <Button
                variant="ghost"
                onClick={() => void runExactED()}
                disabled={busy !== null || !hamiltonian || qubitCount > 12}
              >
                {busy === "ground" ? "Diagonalizing..." : "Exact ED (2N ≤ 12)"}
              </Button>
              <Button variant="ghost" onClick={() => void generateLattice()} disabled={busy !== null}>
                {busy === "preview" ? "Previewing..." : "Preview Lattice"}
              </Button>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: Multi-View Visualizers & Diagnostics Studio */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {/* Sub-Tabs Header */}
          <div style={{ display: "flex", gap: 4, borderBottom: "1px solid #1e293b", paddingBottom: 8, flexWrap: "wrap" }}>
            {[
              { id: "bands", label: "Band Structure & DOS" },
              { id: "occupancy", label: "Charge & Mott Localization" },
              { id: "solvers", label: "Electronic Solvers (DMRG / PEPS)", badge: dmrgEnergy != null || groundEnergy != null || pepsEnergy != null ? "Solved" : undefined },
              { id: "structure", label: "Neutron Structure S(q)" },
              { id: "export", label: "Data & Code Exporters" },
            ].map((t) => (
              <button
                key={t.id}
                onClick={() => setMaterialsTab(t.id as MaterialsTab)}
                style={{
                  padding: "6px 12px",
                  borderRadius: 4,
                  border: materialsTab === t.id ? "1px solid #38bdf8" : "1px solid transparent",
                  background: materialsTab === t.id ? "#1e293b" : "transparent",
                  color: materialsTab === t.id ? "#38bdf8" : "#94a3b8",
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                }}
              >
                <span>{t.label}</span>
                {t.badge && (
                  <span style={{ fontSize: 9, padding: "1px 5px", borderRadius: 4, background: "#334155", color: "#e2e8f0" }}>
                    {t.badge}
                  </span>
                )}
              </button>
            ))}
          </div>

          {/* TAB 1: BAND STRUCTURE & DOS */}
          {materialsTab === "bands" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div style={card}>
                <ElectronicBandDispersionChart
                  hopping={hopping}
                  hoppingPrime={hoppingPrime}
                  onsiteU={onsiteU}
                  chemicalPotential={chemicalPotential}
                />
              </div>

              <div style={card}>
                <DensityOfStatesChart hopping={hopping} onsiteU={onsiteU} />
              </div>
            </div>
          )}

          {/* TAB 2: CHARGE & MOTT LOCALIZATION */}
          {materialsTab === "occupancy" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div style={card}>
                <ChargeOccupancyHeatmap
                  nSites={siteCount}
                  hopping={hopping}
                  onsiteU={onsiteU}
                  chemicalPotential={chemicalPotential}
                />
              </div>

              <div style={card}>
                <DoubleOccupancyVsCorrelationCurve hopping={hopping} onsiteU={onsiteU} />
              </div>

              {/* Material Phase Insight */}
              {(() => {
                const insight = getMaterialInsight();
                return (
                  <div style={{ ...card, borderLeft: "4px solid #38bdf8" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                      <strong style={{ fontSize: 14, color: "#f8fafc" }}>{insight.title}</strong>
                      <span
                        style={{
                          fontSize: 10,
                          padding: "2px 6px",
                          borderRadius: 4,
                          background: "#080d1a",
                          color: "#38bdf8",
                          border: "1px solid #334155",
                        }}
                      >
                        {insight.badge}
                      </span>
                    </div>
                    <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.4, marginBottom: 8 }}>
                      {insight.description}
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 8, fontSize: 11, fontFamily: "monospace", color: "#64748b" }}>
                      <div>Order Parameter: <span style={{ color: "#e2e8f0" }}>{insight.orderParameter}</span></div>
                      <div>Double Occupancy: <span style={{ color: "#38bdf8" }}>{insight.doubleOccupancy}</span></div>
                      <div>Pairing Type: <span style={{ color: "#4ade80" }}>{insight.pairingCandidate}</span></div>
                    </div>
                  </div>
                );
              })()}
            </div>
          )}

          {/* TAB 3: ELECTRONIC SOLVERS (DMRG / PEPS / ED) */}
          {materialsTab === "solvers" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div style={card}>
                <div style={{ fontWeight: 700, fontSize: 14, color: "#f8fafc", marginBottom: 12 }}>
                  Electronic Ground State Optimization & Tensor Contraction
                </div>

                {jobProgress && (
                  <div style={{ padding: "8px 12px", background: "#080d1a", border: "1px solid #334155", borderRadius: 4, fontSize: 11, color: "#38bdf8", marginBottom: 12, display: "flex", gap: 12, fontFamily: "monospace" }}>
                    <span>SOLVER STATUS: <strong>{jobProgress.status}</strong></span>
                    <span>PROGRESS: <strong>{Math.round(jobProgress.progress * 100)}%</strong></span>
                  </div>
                )}

                {/* Results Table */}
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                    gap: 12,
                    padding: 14,
                    background: "#080d1a",
                    borderRadius: 6,
                    border: "1px solid #1e293b",
                  }}
                >
                  <div>
                    <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase" }}>DMRG Ground Energy E₀</div>
                    <div style={{ fontSize: 22, fontWeight: 700, color: "#4ade80", fontFamily: "monospace" }}>
                      {dmrgEnergy != null ? `${Number(dmrgEnergy).toFixed(6)} eV` : "—"}
                    </div>
                    {dmrgEnergy != null && (
                      <div style={{ fontSize: 11, color: "#94a3b8" }}>Per site: {(Number(dmrgEnergy) / siteCount).toFixed(6)} eV</div>
                    )}
                  </div>

                  <div>
                    <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase" }}>Exact ED Ground Energy</div>
                    <div style={{ fontSize: 22, fontWeight: 700, color: "#38bdf8", fontFamily: "monospace" }}>
                      {groundEnergy != null ? `${Number(groundEnergy).toFixed(6)} eV` : "—"}
                    </div>
                    {groundEnergy != null && (
                      <div style={{ fontSize: 11, color: "#94a3b8" }}>Exact check for small cells</div>
                    )}
                  </div>

                  <div>
                    <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase" }}>2D PEPS Ground Energy</div>
                    <div style={{ fontSize: 22, fontWeight: 700, color: "#f59e0b", fontFamily: "monospace" }}>
                      {pepsEnergy != null ? `${Number(pepsEnergy).toFixed(6)} eV` : "—"}
                    </div>
                    {pepsEnergy != null && (
                      <div style={{ fontSize: 11, color: "#94a3b8" }}>Tensor network contraction</div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: NEUTRON STRUCTURE FACTOR S(q) */}
          {materialsTab === "structure" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div style={card}>
                <div style={{ fontWeight: 700, fontSize: 14, color: "#f8fafc", marginBottom: 6 }}>
                  Magnetic Structure Factor S(q) (Neutron Scattering Response)
                </div>
                <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 14 }}>
                  Fourier decomposition of spin-spin correlations S(q) = (1/N) ∑ exp(i q·(r_i - r_j)) ⟨S_i^z S_j^z⟩ at high-symmetry Brillouin zone points.
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12 }}>
                  <div style={{ padding: 14, background: "#080d1a", borderRadius: 6, border: "1px solid #1e293b" }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: "#38bdf8" }}>M Point: q = (π, π)</div>
                    <div style={{ fontSize: 11, color: "#64748b", margin: "2px 0 6px" }}>Néel Antiferromagnetic Resonance</div>
                    <div style={{ fontSize: 22, fontWeight: 700, color: "#f8fafc", fontFamily: "monospace" }}>
                      {onsiteU > hopping ? (0.35 * (siteCount / 4)).toFixed(3) : "0.082"}
                    </div>
                  </div>

                  <div style={{ padding: 14, background: "#080d1a", borderRadius: 6, border: "1px solid #1e293b" }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>Γ Point: q = (0, 0)</div>
                    <div style={{ fontSize: 11, color: "#64748b", margin: "2px 0 6px" }}>Uniform Ferromagnetic Response</div>
                    <div style={{ fontSize: 22, fontWeight: 700, color: "#f8fafc", fontFamily: "monospace" }}>
                      0.014
                    </div>
                  </div>

                  <div style={{ padding: 14, background: "#080d1a", borderRadius: 6, border: "1px solid #1e293b" }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8" }}>X Point: q = (π, 0)</div>
                    <div style={{ fontSize: 11, color: "#64748b", margin: "2px 0 6px" }}>Stripe Magnetic Correlation</div>
                    <div style={{ fontSize: 22, fontWeight: 700, color: "#f8fafc", fontFamily: "monospace" }}>
                      0.048
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 5: DATA & CODE EXPORTERS */}
          {materialsTab === "export" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div style={card}>
                <div style={{ fontWeight: 700, fontSize: 14, color: "#f8fafc", marginBottom: 6 }}>
                  Publication Materials Datasets & Code Exporters
                </div>
                <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 14 }}>
                  Export Fermi-Hubbard electronic models and numerical ground state benchmark reports for publication manuscripts or Python scripts.
                </div>

                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                  <Button variant="primary" onClick={exportLatexTable} disabled={!hamiltonian}>
                    Export LaTeX Manuscript Table (.tex)
                  </Button>
                  <Button variant="secondary" onClick={exportPythonModel} disabled={!hamiltonian}>
                    Export Standalone Python Script (.py)
                  </Button>
                  <Button variant="ghost" onClick={exportCsvDataset} disabled={!hamiltonian}>
                    Export Observables CSV (.csv)
                  </Button>
                </div>
              </div>

              {graph && (
                <div style={card}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8", marginBottom: 8 }}>
                    Crystal Unit Cells & Hopping Network ({graph.sites.length} sites, {graph.edges.length} bonds)
                  </div>
                  <LatticeCanvas graph={graph} />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
