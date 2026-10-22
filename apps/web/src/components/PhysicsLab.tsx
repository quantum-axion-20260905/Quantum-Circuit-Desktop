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
} from "../lib/agent";
import { useUIContext } from "../state/uiContext";
import { Button } from "../ui";
import { EnergyChart, LatticeCanvas } from "./LatticeVisuals";
import { CTMRGLabPanel } from "./CTMRGLabPanel";

type ModelName = "ising" | "heisenberg" | "xxz";
type Boundary = "open" | "periodic";
type PhysicsPreset = "ising_crit" | "ising_ferro" | "heisenberg_afm" | "xxz_luttinger";
type PhysicsTab = "model" | "solvers" | "dynamics" | "ctmrg";

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
  const [coupling, setCoupling] = React.useState(1.0);
  const [field, setField] = React.useState(0.5);
  const [anisotropy, setAnisotropy] = React.useState(1.0);

  const [bondDim, setBondDim] = React.useState(16);
  const [sweeps, setSweeps] = React.useState(4);
  const [dt, setDt] = React.useState(0.05);
  const [steps, setSteps] = React.useState(20);

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
      setAnisotropy(1.0);
    } else if (preset === "ising_ferro") {
      setDimension(1);
      setSizes([8, 1, 1]);
      setBoundary("open");
      setModel("ising");
      setCoupling(1.0);
      setField(0.2);
      setAnisotropy(1.0);
    } else if (preset === "heisenberg_afm") {
      setDimension(1);
      setSizes([8, 1, 1]);
      setBoundary("open");
      setModel("heisenberg");
      setCoupling(1.0);
      setField(0.0);
      setAnisotropy(1.0);
    } else if (preset === "xxz_luttinger") {
      setDimension(1);
      setSizes([10, 1, 1]);
      setBoundary("open");
      setModel("xxz");
      setCoupling(1.0);
      setField(0.0);
      setAnisotropy(0.5);
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
          orderParameter: "⟨Z_i⟩ ≠ 0 (Long-range order)",
        };
      } else if (ratio > 1.05) {
        return {
          title: "Quantum Paramagnetic Phase",
          badge: "Disordered / Polarized",
          description: "Ko'ndalang magnit maydoni g'olib; spinlar X-o'qi bo'ylab qutblangan, energetik mass gap mavjud.",
          orderParameter: "⟨X_i⟩ ≈ 1, ⟨Z_i⟩ = 0",
        };
      } else {
        return {
          title: "Quantum Critical Point (h = J)",
          badge: "Conformal Invariant (c = 1/2 CFT)",
          description: "Massasiz Majorana fermionlari va konform maydon nazariyasi bilan tavsiflanuvchi uzluksiz fazaviy o'tish nuqtasi.",
          orderParameter: "Power-law ⟨Z_0 Z_r⟩ ~ r^(-1/4)",
        };
      }
    } else if (model === "heisenberg") {
      if (coupling > 0) {
        return {
          title: "Antiferromagnetic Phase (Singlet / Néel)",
          badge: "SU(2) Invariant",
          description: "Qo'shni spinlar antiparallel o'zaro ta'sirlashadi; 1D zanjirda gapless spinon to'lqinlari kuzatiladi.",
          orderParameter: "Staggered ⟨(-1)^i S_i^z⟩",
        };
      } else {
        return {
          title: "Ferromagnetic State",
          badge: "Fully Polarized",
          description: "Barcha spinlar bir yo'nalishga qutblangan; asosiy holat to'liq ferromagnit ko'p spinli holatdir.",
          orderParameter: "Total Spin S = N/2",
        };
      }
    } else {
      if (anisotropy > 1.05) {
        return {
          title: "Ising-like Gapped Antiferromagnet",
          badge: "Discrete Z2 Broken",
          description: "Z-anizotropiya z-o'qi bo'ylab antiferromagnit tartib hosil qiladi, qo'zg'alishlar energetik tirqishga ega.",
          orderParameter: "Néel Order with Exponential Decay",
        };
      } else if (anisotropy < -1.05) {
        return {
          title: "Ferromagnetic Phase",
          badge: "Anisotropic FM",
          description: "Z-yo'nalishidagi kuchli ferromagnit o'zaro ta'sir spinlarni to'liq qutblaydi.",
          orderParameter: "Ferromagnetic Polarization",
        };
      } else {
        return {
          title: "Tomonaga–Luttinger Liquid (XY Critical)",
          badge: "Gapless Critical (c = 1)",
          description: "Kvant anizotropik suyuqligi; korrelyatsiyalar algebraik (power-law) pasayadi, tezkor spinon to'lqinlari mavjud.",
          orderParameter: "Algebraic Decay, Central Charge c = 1",
        };
      }
    }
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

  async function generateHamiltonian() {
    setBusy("hamiltonian");
    setError(null);
    try {
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
      const res = await runGroundState({
        terms: hamiltonian.terms,
        n_qubits: hamiltonian.n_qubits,
      }, setJobProgress);
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
      const res = await runDMRG({
        terms: hamiltonian.terms,
        n_qubits: hamiltonian.n_qubits,
        bond_dim: bondDim,
        sweeps,
      }, setJobProgress);
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
      const res = await runTEBD({
        terms: hamiltonian.terms,
        n_qubits: hamiltonian.n_qubits,
        dt,
        steps,
        bond_dim: bondDim,
      }, setJobProgress);
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

  function exportPythonModel() {
    const code = `# Quantum Circuit Platform - Many-Body Spin Physics Exporter
# Model: ${model.toUpperCase()} on ${activeDimensions.join("x")} lattice (${boundary} boundary)
import numpy as np

n_spins = ${hamiltonian?.n_qubits ?? qubitCount}
coupling_J = ${coupling}
transverse_field_h = ${field}
anisotropy_delta = ${anisotropy}

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

  const dmrgEnergy = dmrgResult?.ground_energy;
  const groundEnergy = groundResult?.ground_energy;
  const energies = tebdResult && Array.isArray(tebdResult.energies)
    ? tebdResult.energies.map(Number).filter(Number.isFinite)
    : [];

  return (
    <main style={shell}>
      {/* Top Header & Navigation */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, borderBottom: "1px solid #1e293b", paddingBottom: 12, flexWrap: "wrap", gap: 12 }}>
        <div>
          <h2 style={{ margin: "0 0 4px 0", fontSize: 18, fontWeight: 700, color: "#f8fafc" }}>
            Many-Body Quantum Physics Laboratory
          </h2>
          <div style={{ fontSize: 12, color: "#94a3b8" }}>
            Spin lattice Hamiltonians, quantum phase transitions, critical dynamics, and 2D infinite thermodynamics.
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
            <option value="" disabled>Load Spin Preset...</option>
            <option value="ising_crit">1D Ising Critical Point (L=8, h=J=1)</option>
            <option value="ising_ferro">1D Ising Ferromagnet (L=8, h=0.2J)</option>
            <option value="heisenberg_afm">1D Heisenberg Antiferromagnet (L=8, J=1)</option>
            <option value="xxz_luttinger">1D XXZ Luttinger Liquid (L=10, Δ=0.5)</option>
          </select>

          <div style={{ padding: "4px 10px", background: "#1e293b", borderRadius: 4, border: "1px solid #334155", fontSize: 12, color: "#cbd5e1" }}>
            <strong>{qubitCount}</strong> spins · <span style={{ color: hamiltonian ? "#4ade80" : "#f59e0b" }}>{hamiltonian ? `${hamiltonian.terms.length} terms` : "No Hamiltonian"}</span>
          </div>
        </div>
      </div>

      {/* Navigation Sub-Tabs */}
      <div style={{ display: "flex", gap: 4, marginBottom: 16, borderBottom: "1px solid #1e293b", paddingBottom: 8 }}>
        {[
          { id: "model", label: "Model & Geometry" },
          { id: "solvers", label: "Ground State (DMRG / ED)", badge: dmrgEnergy != null || groundEnergy != null ? "Solved" : undefined },
          { id: "dynamics", label: "Quench Dynamics (TEBD)", badge: energies.length ? `${energies.length} pts` : undefined },
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
        <div style={{ padding: "10px 14px", background: "#450a0a", border: "1px solid #ef4444", borderRadius: 6, color: "#fca5a5", fontSize: 12, marginBottom: 16 }}>
          {error}
        </div>
      )}

      {/* 1. MODEL & GEOMETRY TAB */}
      {physicsTab === "model" && (
        <div style={{ display: "grid", gridTemplateColumns: "minmax(320px, 1fr) minmax(380px, 1.2fr)", gap: 16 }}>
          {/* Controls Card */}
          <div style={card}>
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 14, color: "#f8fafc" }}>
              Spin Hamiltonian Specification
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 14 }}>
              <label style={fieldStyle}>
                Hamiltonian Model
                <select value={model} onChange={(e) => setModel(e.target.value as ModelName)} style={{ ...inputStyle, width: "100%" }}>
                  <option value="ising">Transverse-field Ising</option>
                  <option value="heisenberg">Heisenberg (XXX)</option>
                  <option value="xxz">Anisotropic XXZ</option>
                </select>
              </label>

              <label style={fieldStyle}>
                Lattice Dimension
                <select value={dimension} onChange={(e) => setDimension(Number(e.target.value) as 1 | 2)} style={{ ...inputStyle, width: "100%" }}>
                  <option value={1}>1D Chain</option>
                  <option value={2}>2D Square Grid</option>
                </select>
              </label>

              <label style={fieldStyle}>
                Spin Chain Length L
                <input
                  type="number"
                  min={2}
                  max={64}
                  value={sizes[0]}
                  onChange={(e) => setSizes([Math.max(2, Math.min(64, Number(e.target.value) || 2)), sizes[1], sizes[2]])}
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
                    onChange={(e) => setSizes([sizes[0], Math.max(2, Math.min(16, Number(e.target.value) || 2)), sizes[2]])}
                    style={{ ...inputStyle, width: "100%" }}
                  />
                </label>
              )}

              <label style={fieldStyle}>
                Boundary Condition
                <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={{ ...inputStyle, width: "100%" }}>
                  <option value="open">Open (OBC)</option>
                  <option value="periodic">Periodic (PBC)</option>
                </select>
              </label>

              <label style={fieldStyle}>
                Coupling Constant J
                <input type="number" step="0.1" value={coupling} onChange={(e) => setCoupling(Number(e.target.value) || 0)} style={{ ...inputStyle, width: "100%" }} />
              </label>

              {model === "ising" && (
                <label style={fieldStyle}>
                  Transverse Field h
                  <input type="number" step="0.05" value={field} onChange={(e) => setField(Number(e.target.value) || 0)} style={{ ...inputStyle, width: "100%" }} />
                </label>
              )}

              {model === "xxz" && (
                <label style={fieldStyle}>
                  Anisotropy Δ
                  <input type="number" step="0.1" value={anisotropy} onChange={(e) => setAnisotropy(Number(e.target.value) || 0)} style={{ ...inputStyle, width: "100%" }} />
                </label>
              )}
            </div>

            <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
              <Button variant="primary" onClick={() => void generateHamiltonian()} disabled={busy !== null}>
                {busy === "hamiltonian" ? "Building..." : "Build Spin Hamiltonian"}
              </Button>
              <Button variant="ghost" onClick={exportPythonModel} disabled={!hamiltonian}>
                Export Python Code
              </Button>
            </div>
          </div>

          {/* Theoretical Insight & Lattice Canvas */}
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {/* Phase Insight */}
            {(() => {
              const insight = getPhaseInsight();
              return (
                <div style={{ ...card, padding: 14, borderLeft: "4px solid #38bdf8" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                    <strong style={{ fontSize: 14, color: "#f8fafc" }}>{insight.title}</strong>
                    <span style={{ fontSize: 10, padding: "2px 6px", borderRadius: 4, background: "#1e293b", color: "#38bdf8", border: "1px solid #334155" }}>
                      {insight.badge}
                    </span>
                  </div>
                  <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.4, marginBottom: 6 }}>
                    {insight.description}
                  </div>
                  <div style={{ fontSize: 11, color: "#64748b" }}>
                    Order Parameter: <code style={{ color: "#e2e8f0" }}>{insight.orderParameter}</code>
                  </div>
                </div>
              );
            })()}

            {/* Lattice Graph Visual */}
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

      {/* 2. GROUND STATE SOLVERS TAB */}
      {physicsTab === "solvers" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={card}>
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 12, color: "#f8fafc" }}>
              Variational Ground State Optimization (DMRG & Exact Diagonalization)
            </div>
            <div style={{ display: "flex", gap: 16, alignItems: "flex-end", flexWrap: "wrap", marginBottom: 16 }}>
              <label style={fieldStyle}>
                Bond Dimension χ
                <input type="number" min={1} max={64} value={bondDim} onChange={(e) => setBondDim(Number(e.target.value) || 16)} style={inputStyle} />
              </label>
              <label style={fieldStyle}>
                DMRG Sweeps
                <input type="number" min={1} max={30} value={sweeps} onChange={(e) => setSweeps(Number(e.target.value) || 4)} style={inputStyle} />
              </label>
              <Button variant="primary" onClick={() => void runDMRGSolver()} disabled={busy !== null || !hamiltonian}>
                {busy === "dmrg" ? "Running DMRG..." : "Run DMRG Optimizer"}
              </Button>
              <Button variant="ghost" onClick={() => void runExactED()} disabled={busy !== null || !hamiltonian || qubitCount > 12}>
                {busy === "ground" ? "Diagonalizing..." : "Exact Diagonalization (N ≤ 12)"}
              </Button>
            </div>

            {/* Result Readouts */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12, padding: 14, background: "#0b0f19", borderRadius: 6, border: "1px solid #1e293b" }}>
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
                  <div style={{ fontSize: 11, color: "#94a3b8" }}>Exact benchmark for N ≤ 12</div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 3. QUENCH DYNAMICS (TEBD) TAB */}
      {physicsTab === "dynamics" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div style={card}>
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 12, color: "#f8fafc" }}>
              Real-Time Quench Dynamics (TEBD)
            </div>
            <div style={{ display: "flex", gap: 16, alignItems: "flex-end", flexWrap: "wrap", marginBottom: 16 }}>
              <label style={fieldStyle}>
                Time Step dt
                <input type="number" step="0.01" value={dt} onChange={(e) => setDt(Number(e.target.value) || 0.05)} style={inputStyle} />
              </label>
              <label style={fieldStyle}>
                Number of Steps
                <input type="number" min={1} max={500} value={steps} onChange={(e) => setSteps(Number(e.target.value) || 20)} style={inputStyle} />
              </label>
              <Button variant="primary" onClick={() => void runTEBDEvolution()} disabled={busy !== null || !hamiltonian}>
                {busy === "tebd" ? "Evolving..." : "Run TEBD Simulation"}
              </Button>
            </div>

            {energies.length > 0 && (
              <div>
                <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 6 }}>Energy Trajectory E(t) over Time Steps:</div>
                <EnergyChart values={energies} />
              </div>
            )}
          </div>
        </div>
      )}

      {/* 4. 2D INFINITE CTMRG TAB */}
      {physicsTab === "ctmrg" && (
        <div style={card}>
          <CTMRGLabPanel
            dimensions={activeDimensions}
            model={model}
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
        <div style={{ marginTop: 12, fontSize: 11, color: "#94a3b8" }}>
          Status: {jobProgress.status} ({jobProgress.job_id})
        </div>
      )}
    </main>
  );
}
