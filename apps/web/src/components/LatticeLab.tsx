"use client";

import React from "react";
import { buildHamiltonian, buildHubbard, cancelAsyncJob, previewLattice, runDMRG, runExpectation, runGroundState, runPEPS, runTEBD, type AgentResult, type AsyncJob, type LatticeGraph, type SparseHamiltonianResponse } from "../lib/agent";
import type { ReplayRequest } from "../lib/runHistory";
import { editorToIr } from "../ir/converters";
import { irToAgentTNPayload } from "../ir/agentMapping";
import { useCircuit } from "../state/circuitStore";
import { useUIContext } from "../state/uiContext";
import { syncPhysicsStudy } from "../lib/projectStore";
import { Button } from "../ui";
import { ComputeProgress } from "./ComputeProgress";
import { EnergyChart, LatticeCanvas } from "./LatticeVisuals";
import { buildPhysicsStudyManifest, buildPhysicsStudyVariants, type PhysicsStudyMode } from "../lib/physicsStudy";
import type { JsonObject } from "../lib/agent";
import { PhysicsConvergenceStudy, type PhysicsStudyRow } from "./PhysicsConvergenceStudy";
import { ConvergenceDiagnostics } from "./ConvergenceDiagnostics";
import { CTMRGLabPanel } from "./CTMRGLabPanel";

type ModelName = "ising" | "heisenberg" | "xxz";
type Boundary = "open" | "periodic";
type MaterialName = "spin" | "hubbard";
type PEPSContraction = "auto" | "boundary-mps";

const shell: React.CSSProperties = { maxWidth: "100%", width: "100%", margin: 0, padding: "12px 16px 40px", color: "var(--qc-dark-text)", boxSizing: "border-box" };
const card: React.CSSProperties = { border: "1px solid var(--qc-dark-border)", borderRadius: "var(--qc-radius-lg)", background: "var(--qc-dark-surface)", padding: 20, boxShadow: "var(--qc-shadow-lg)" };
const fieldStyle: React.CSSProperties = { display: "grid", gap: 6, color: "var(--qc-dark-text-muted)", fontSize: 12 };
const input: React.CSSProperties = { width: 88, border: "1px solid var(--qc-dark-border-strong)", borderRadius: "var(--qc-radius-md)", background: "var(--qc-dark-canvas)", color: "var(--qc-dark-text)", padding: "7px 10px" };

function errorText(error: unknown) {
  return error instanceof Error ? error.message : "Operation failed.";
}

export function LatticeLab() {
  const { nQubits, ops, setNQubits } = useCircuit();
  const { setLatestOutput, addExperiment } = useUIContext();
  const [dimension, setDimension] = React.useState<1 | 2 | 3>(2);
  const [sizes, setSizes] = React.useState([4, 4, 2]);
  const [boundary, setBoundary] = React.useState<Boundary>("open");
  const [material, setMaterial] = React.useState<MaterialName>("spin");
  const [model, setModel] = React.useState<ModelName>("ising");
  const [coupling, setCoupling] = React.useState(1);
  const [field, setField] = React.useState(0.5);
  const [anisotropy, setAnisotropy] = React.useState(1);
  const [hopping, setHopping] = React.useState(1);
  const [onsiteU, setOnsiteU] = React.useState(0);
  const [chemicalPotential, setChemicalPotential] = React.useState(0);
  const [bondDim, setBondDim] = React.useState(16);
  const [dt, setDt] = React.useState(0.05);
  const [steps, setSteps] = React.useState(20);
  const [pepsContraction, setPepsContraction] = React.useState<PEPSContraction>("auto");
  const [boundaryBondDim, setBoundaryBondDim] = React.useState(16);
  const [sweeps, setSweeps] = React.useState(4);
  const [graph, setGraph] = React.useState<LatticeGraph | null>(null);
  const [hamiltonian, setHamiltonian] = React.useState<SparseHamiltonianResponse | null>(null);
  const [energyResult, setEnergyResult] = React.useState<Record<string, unknown> | null>(null);
  const [tebdResult, setTebdResult] = React.useState<Record<string, unknown> | null>(null);
  const [dmrgResult, setDmrgResult] = React.useState<Record<string, unknown> | null>(null);
  const [pepsResult, setPepsResult] = React.useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = React.useState<"preview" | "hamiltonian" | "energy" | "ground" | "dmrg" | "tebd" | "peps" | "study" | null>(null);
  const [jobProgress, setJobProgress] = React.useState<AsyncJob | null>(null);
  const [canceling, setCanceling] = React.useState(false);
  const cancelRequestedRef = React.useRef(false);
  const [groundResult, setGroundResult] = React.useState<Record<string, unknown> | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const retryRef = React.useRef<(() => void) | null>(null);
  const [labTab, setLabTab] = React.useState<"overview" | "solvers" | "dynamics" | "ctmrg" | "convergence" | "all">("overview");

  function applyModelPreset(type: "ising1d" | "heisenberg2d" | "hubbard2d" | "xxz1d") {
    setError(null);
    if (type === "ising1d") {
      setDimension(1);
      setSizes([8, 4, 2]);
      setBoundary("open");
      setMaterial("spin");
      setModel("ising");
      setCoupling(1);
      setField(0.5);
      setAnisotropy(1);
    } else if (type === "heisenberg2d") {
      setDimension(2);
      setSizes([4, 4, 2]);
      setBoundary("open");
      setMaterial("spin");
      setModel("heisenberg");
      setCoupling(1);
      setField(0.0);
      setAnisotropy(1);
    } else if (type === "hubbard2d") {
      setDimension(2);
      setSizes([3, 3, 2]);
      setBoundary("open");
      setMaterial("hubbard");
      setHopping(1);
      setOnsiteU(4);
      setChemicalPotential(0);
    } else if (type === "xxz1d") {
      setDimension(1);
      setSizes([10, 4, 2]);
      setBoundary("open");
      setMaterial("spin");
      setModel("xxz");
      setCoupling(1);
      setField(0.0);
      setAnisotropy(1);
    }
  }

  const [studyMode, setStudyMode] = React.useState<PhysicsStudyMode>("dmrg");
  const [studyRows, setStudyRows] = React.useState<PhysicsStudyRow[]>([]);
  const [studyManifest, setStudyManifest] = React.useState<JsonObject | null>(null);
  const studySyncPendingRef = React.useRef(false);
  const studyStartedAtRef = React.useRef<string | null>(null);
  const studyConfigurationRef = React.useRef<JsonObject>({});

  React.useEffect(() => {
    if (busy !== null || !studySyncPendingRef.current || studyRows.length === 0 || studyRows.some((row) => row.status === "queued" || row.status === "running")) return;
    studySyncPendingRef.current = false;
    const finishedAt = new Date().toISOString();
    const startedAt = studyStartedAtRef.current ?? finishedAt;
    const manifest = buildPhysicsStudyManifest({ mode: studyMode, startedAt, finishedAt, rows: studyRows, nQubits: hamiltonian?.n_qubits ?? nQubits, configuration: studyConfigurationRef.current });
    setStudyManifest(manifest);
    void syncPhysicsStudy({ mode: studyMode, startedAt, finishedAt, rows: studyRows, nQubits: hamiltonian?.n_qubits ?? nQubits, configuration: studyConfigurationRef.current }).catch(() => undefined);
  }, [busy, hamiltonian?.n_qubits, nQubits, studyMode, studyRows]);
  const updateJob = React.useCallback((job: AsyncJob) => setJobProgress(job), []);
  const cancelJob = React.useCallback(async () => {
    const job = jobProgress;
    if (!job || (job.status !== "queued" && job.status !== "running")) return;
    cancelRequestedRef.current = true;
    setCanceling(true);
    try {
      await cancelAsyncJob(job.job_id);
    } catch (e: unknown) {
      cancelRequestedRef.current = false;
      setCanceling(false);
      setError(errorText(e));
    }
  }, [jobProgress]);

  const activeDimensions = sizes.slice(0, dimension);
  const siteCount = activeDimensions.reduce((total, size) => total * size, 1);
  const qubitCount = material === "hubbard" ? siteCount * 2 : siteCount;
  const payload = { dimensions: activeDimensions, boundary };

  function updateSize(index: number, raw: string) {
    const value = Math.max(1, Math.min(16, Number(raw) || 1));
    setSizes((current) => current.map((size, position) => position === index ? value : size));
  }

  async function preview() {
    setBusy("preview"); setError(null);
    try { setGraph(await previewLattice("spin-lattice", payload)); } catch (e: unknown) { setError(errorText(e)); } finally { setBusy(null); }
  }

  async function generateHamiltonian() {
    setBusy("hamiltonian"); setError(null);
    try {
      const result = material === "hubbard"
        ? await buildHubbard("hubbard-materials", { ...payload, hopping, onsite_u: onsiteU, chemical_potential: chemicalPotential })
        : await buildHamiltonian("spin-lattice", { ...payload, model, coupling, field, anisotropy });
      setHamiltonian(result); setGraph(result); setEnergyResult(null); setGroundResult(null); setDmrgResult(null); setTebdResult(null); setPepsResult(null);
    } catch (e: unknown) { setError(errorText(e)); }
    finally { setBusy(null); }
  }

  function circuitPayload() {
    return irToAgentTNPayload(editorToIr(nQubits, ops));
  }

  async function runEnergy() {
    setBusy("energy"); setError(null); setJobProgress(null); setCanceling(false); cancelRequestedRef.current = false;
    retryRef.current = () => { void runEnergy(); };
    let request: ReplayRequest | null = null;
    try {
      if (!hamiltonian) throw new Error("Generate a Hamiltonian first.");
      if (hamiltonian.n_qubits !== nQubits) throw new Error(`Lattice has ${hamiltonian.n_qubits} sites but the circuit has ${nQubits} qubits. Use Adopt lattice size.`);
      const payload = { ...circuitPayload(), backend: "tensor-network", terms: hamiltonian.terms, bond_dim: bondDim, truncation_cutoff: 0 };
      request = { source: "physics", kind: "expectation", payload };
      const result = await runExpectation(payload, updateJob);
      setEnergyResult(result); setLatestOutput(result);
      addExperiment({ label: "Physics · energy expectation", source: request.source, kind: request.kind, status: "done", request, result, provenance: result.provenance });
    } catch (e: unknown) {
      const canceled = cancelRequestedRef.current;
      const message = canceled ? "Job user tomonidan bekor qilindi." : errorText(e);
      if (request) addExperiment({ label: "Physics · energy expectation", source: request.source, kind: request.kind, status: canceled ? "canceled" : "failed", request, error: message });
      setError(message);
    }
    finally { setBusy(null); setCanceling(false); cancelRequestedRef.current = false; }
  }

  async function runEvolution() {
    setBusy("tebd"); setError(null); setJobProgress(null); setCanceling(false); cancelRequestedRef.current = false;
    retryRef.current = () => { void runEvolution(); };
    let request: ReplayRequest | null = null;
    try {
      if (!hamiltonian) throw new Error("Generate a Hamiltonian first.");
      if (hamiltonian.n_qubits !== nQubits) throw new Error(`Hamiltonian has ${hamiltonian.n_qubits} mapped qubits but the circuit has ${nQubits} qubits. Use Adopt lattice size.`);
      // Spin models have one qubit per site. Hubbard is spinful (two mapped
      // qubits per site), so its physical lattice cannot be used as the
      // generic TEBD lattice metadata without violating n_sites == n_qubits.
      const replayPayload = { ...circuitPayload(), n_qubits: nQubits, terms: hamiltonian.terms, ...(material === "spin" ? { lattice: payload } : {}), dt, steps, order: 2, bond_dim: bondDim, truncation_cutoff: 0 };
      request = { source: "physics", kind: "tebd", payload: replayPayload };
      const result = await runTEBD(replayPayload, updateJob);
      setTebdResult(result); setPepsResult(null); setLatestOutput(result);
      addExperiment({ label: "Physics · TEBD evolution", source: request.source, kind: request.kind, status: "done", request, result, provenance: result.provenance });
    } catch (e: unknown) {
      const canceled = cancelRequestedRef.current;
      const message = canceled ? "Job user tomonidan bekor qilindi." : errorText(e);
      if (request) addExperiment({ label: "Physics · TEBD evolution", source: request.source, kind: request.kind, status: canceled ? "canceled" : "failed", request, error: message });
      setError(message);
    }
    finally { setBusy(null); setCanceling(false); cancelRequestedRef.current = false; }
  }

  async function runGround() {
    setBusy("ground"); setError(null); setJobProgress(null); setCanceling(false); cancelRequestedRef.current = false;
    retryRef.current = () => { void runGround(); };
    let request: ReplayRequest | null = null;
    try {
      if (!hamiltonian) throw new Error("Generate a Hamiltonian first.");
      if (hamiltonian.n_qubits > 12) throw new Error("Exact ground-state validation is capped at 12 qubits; use MPS/TEBD for larger systems.");
      const replayPayload = { n_qubits: hamiltonian.n_qubits, terms: hamiltonian.terms, backend: "exact-diagonalization", dtype: "complex64", max_mem_mb: 1024, max_time_ms: 120000 };
      request = { source: "physics", kind: "ground_state", payload: replayPayload };
      const result = await runGroundState(replayPayload, updateJob);
      setGroundResult(result); setLatestOutput(result);
      addExperiment({ label: "Physics · exact ground state", source: request.source, kind: request.kind, status: "done", request, result, provenance: result.provenance });
    } catch (e: unknown) {
      const canceled = cancelRequestedRef.current;
      const message = canceled ? "Job user tomonidan bekor qilindi." : errorText(e);
      if (request) addExperiment({ label: "Physics · exact ground state", source: request.source, kind: request.kind, status: canceled ? "canceled" : "failed", request, error: message });
      setError(message);
    }
    finally { setBusy(null); setCanceling(false); cancelRequestedRef.current = false; }
  }

  async function runVariationalGround() {
    setBusy("dmrg"); setError(null); setJobProgress(null); setCanceling(false); cancelRequestedRef.current = false;
    retryRef.current = () => { void runVariationalGround(); };
    let request: ReplayRequest | null = null;
    try {
      if (!hamiltonian) throw new Error("Generate a Hamiltonian first.");
      const replayPayload = { n_qubits: hamiltonian.n_qubits, terms: hamiltonian.terms, dtype: "complex64", backend: "tensor-network", bond_dim: Math.min(64, bondDim), sweeps, tolerance: 1e-7, max_time_ms: 120000, max_mem_mb: 1024 };
      request = { source: "physics", kind: "dmrg", payload: replayPayload };
      const result = await runDMRG(replayPayload, updateJob);
      setDmrgResult(result); setLatestOutput(result);
      addExperiment({ label: "Physics · DMRG ground state", source: request.source, kind: request.kind, status: "done", request, result, provenance: result.provenance });
    } catch (e: unknown) {
      const canceled = cancelRequestedRef.current;
      const message = canceled ? "Job user tomonidan bekor qilindi." : errorText(e);
      if (request) addExperiment({ label: "Physics · DMRG ground state", source: request.source, kind: request.kind, status: canceled ? "canceled" : "failed", request, error: message });
      setError(message);
    }
    finally { setBusy(null); setCanceling(false); cancelRequestedRef.current = false; }
  }

  async function runNativePEPS() {
    setBusy("peps"); setError(null); setJobProgress(null); setCanceling(false); cancelRequestedRef.current = false;
    retryRef.current = () => { void runNativePEPS(); };
    let request: ReplayRequest | null = null;
    try {
      if (!hamiltonian) throw new Error("Generate a Hamiltonian first.");
      if (!pepsEligible) throw new Error("Native PEPS is available for spin models on bounded 2D/3D lattices (up to 64 sites; admission is checked before compute).");
      if (pepsContraction === "boundary-mps" && (dimension !== 2 || boundary !== "open")) throw new Error("Boundary-MPS requires an open 2D lattice.");
      if (hamiltonian.n_qubits !== nQubits) throw new Error(`Hamiltonian has ${hamiltonian.n_qubits} mapped qubits but the circuit has ${nQubits} qubits. Use Adopt lattice size.`);
      const replayPayload = { n_qubits: nQubits, terms: hamiltonian.terms, lattice: payload, dtype: "complex64", backend: "tensor-network", bond_dim: Math.min(4, Math.max(1, bondDim)), truncation_cutoff: 0, dt, steps: Math.min(steps, 8), order: 2, contraction_method: pepsContraction, boundary_bond_dim: Math.min(64, Math.max(1, boundaryBondDim)), max_contraction_states: 1000000, max_time_ms: 120000, max_mem_mb: 1024 };
      request = { source: "physics", kind: "peps", payload: replayPayload };
      const result = await runPEPS(replayPayload, updateJob);
      setPepsResult(result); setTebdResult(null); setLatestOutput(result);
      addExperiment({ label: "Physics · PEPS evolution", source: request.source, kind: request.kind, status: "done", request, result, provenance: result.provenance });
    } catch (e: unknown) {
      const canceled = cancelRequestedRef.current;
      const message = canceled ? "Job user tomonidan bekor qilindi." : errorText(e);
      if (request) addExperiment({ label: "Physics · PEPS evolution", source: request.source, kind: request.kind, status: canceled ? "canceled" : "failed", request, error: message });
      setError(message);
    }
    finally { setBusy(null); setCanceling(false); cancelRequestedRef.current = false; }
  }

  async function runConvergenceStudy(mode: PhysicsStudyMode) {
    if (!hamiltonian) { setError("Generate a Hamiltonian first."); return; }
    if (mode === "dmrg" && hamiltonian.n_qubits < 2) { setError("DMRG convergence requires at least two qubits."); return; }
    if (mode === "peps" && (!pepsEligible || hamiltonian.n_qubits !== nQubits)) { setError("PEPS convergence is available only for an admitted 2D/3D spin lattice."); return; }
    if (mode === "tebd" && (!tebdReady || hamiltonian.n_qubits !== nQubits)) { setError("TEBD convergence is not available for this Hamiltonian."); return; }

    const variants = buildPhysicsStudyVariants({
      mode,
      nQubits: mode === "peps" ? nQubits : hamiltonian.n_qubits,
      terms: hamiltonian.terms,
      lattice: material === "spin" ? payload : undefined,
      bondDim,
      sweeps,
      steps,
      dt,
      boundaryBondDim,
      truncationCutoff: 0,
    });

    setStudyMode(mode);
    setStudyManifest(null);
    studyConfigurationRef.current = {
      lattice: material === "spin" ? payload : null,
      model: material === "spin" ? model : material,
      hamiltonian_parameters: material === "spin" ? { coupling, field, anisotropy } : { hopping, onsite_u: onsiteU, chemical_potential: chemicalPotential },
      terms: hamiltonian.terms,
      solver_controls: { bond_dim: bondDim, boundary_bond_dim: boundaryBondDim, sweeps, steps, dt, truncation_cutoff: 0 },
    };
    setStudyRows(variants.map((variant, index) => ({ id: `${mode}-${Date.now()}-${index}`, label: variant.label, parameters: variant.parameters, status: "queued", request: variant.payload })));
    studySyncPendingRef.current = true;
    studyStartedAtRef.current = new Date().toISOString();
    setBusy("study"); setError(null); setJobProgress(null); setCanceling(false); cancelRequestedRef.current = false;
    retryRef.current = () => { void runConvergenceStudy(mode); };
    for (const [index, variant] of variants.entries()) {
      if (cancelRequestedRef.current) break;
      setStudyRows((rows) => rows.map((row, rowIndex) => rowIndex === index ? { ...row, status: "running" } : row));
      const request: ReplayRequest = { source: "physics", kind: mode, payload: variant.payload };
      try {
        const result: AgentResult = mode === "dmrg"
          ? await runDMRG(variant.payload, updateJob)
          : mode === "tebd"
            ? await runTEBD(variant.payload, updateJob)
            : await runPEPS(variant.payload, updateJob);
        setStudyRows((rows) => rows.map((row, rowIndex) => rowIndex === index ? { ...row, status: "done", request: variant.payload, result } : row));
        setLatestOutput(result);
        if (mode === "dmrg") setDmrgResult(result);
        if (mode === "tebd") setTebdResult(result);
        if (mode === "peps") setPepsResult(result);
        addExperiment({ label: `Physics · ${mode.toUpperCase()} convergence · ${variant.label}`, source: request.source, kind: request.kind, status: "done", request, result, provenance: result.provenance });
      } catch (e: unknown) {
        const canceled = cancelRequestedRef.current;
        const message = canceled ? "Job user tomonidan bekor qilindi." : errorText(e);
        setStudyRows((rows) => rows.map((row, rowIndex) => rowIndex === index ? { ...row, status: canceled ? "canceled" : "failed", request: variant.payload, error: message } : row));
        addExperiment({ label: `Physics · ${mode.toUpperCase()} convergence · ${variant.label}`, source: request.source, kind: request.kind, status: canceled ? "canceled" : "failed", request, error: message });
        if (canceled) break;
      }
    }
    setBusy(null); setCanceling(false); cancelRequestedRef.current = false;
  }

  const energy = energyResult?.energy;
  const groundEnergy = groundResult?.ground_energy;
  const dmrgEnergy = dmrgResult?.ground_energy;
  const trajectoryResult = pepsResult ?? tebdResult;
  const energies = trajectoryResult && Array.isArray(trajectoryResult.energies) ? trajectoryResult.energies.map(Number).filter(Number.isFinite) : [];
  const warningList = trajectoryResult && Array.isArray(trajectoryResult.warnings) ? trajectoryResult.warnings.map(String) : [];
  const evidenceResult = pepsResult ?? dmrgResult ?? tebdResult ?? energyResult;
  const observableRows = evidenceResult && Array.isArray(evidenceResult.observables)
    ? evidenceResult.observables.filter((value): value is Record<string, unknown> => typeof value === "object" && value !== null).slice(0, 24)
    : [];
  const tebdReady = hamiltonian?.tebd_ready ?? true;
  const pepsEligible = material === "spin" && dimension > 1 && siteCount <= 64;

  return (
    <main style={{ ...shell, background: "radial-gradient(circle at 10% 0%, rgba(14,165,233,.14), transparent 34%), radial-gradient(circle at 90% 8%, rgba(168,85,247,.12), transparent 32%)", minHeight: "100%" }}>
      {/* Compact Professional Studio Sub-Header & Navigation */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 12,
          padding: "6px 12px",
          marginBottom: 16,
          background: "rgba(15, 23, 42, 0.75)",
          backdropFilter: "blur(8px)",
          border: "1px solid rgba(51, 65, 85, 0.6)",
          borderRadius: 10,
          flexWrap: "wrap",
        }}
      >
        {/* Left: Professional Tabs */}
        <div style={{ display: "flex", gap: 4, alignItems: "center", overflowX: "auto" }}>
          {[
            { id: "overview", label: "🌐 Geometry & Model", badge: `${siteCount} sites` },
            { id: "solvers", label: "⚡ Ground State & DMRG", badge: dmrgEnergy != null || groundEnergy != null ? "Solved" : undefined },
            { id: "dynamics", label: "🌊 Dynamics (TEBD / PEPS)", badge: energies.length ? `${energies.length} pts` : undefined },
            { id: "ctmrg", label: "📈 2D Infinite (CTMRG)" },
            { id: "convergence", label: "🔬 Convergence" },
            { id: "all", label: "📑 All Panels" }
          ].map((t) => {
            const isActive = labTab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setLabTab(t.id as typeof labTab)}
                style={{
                  padding: "6px 12px",
                  borderRadius: 6,
                  border: "none",
                  background: isActive ? "#0284c7" : "transparent",
                  color: isActive ? "#ffffff" : "#94a3b8",
                  fontWeight: isActive ? 600 : 500,
                  fontSize: 12,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  transition: "all 120ms ease",
                  whiteSpace: "nowrap",
                }}
              >
                <span>{t.label}</span>
                {t.badge && (
                  <span
                    style={{
                      fontSize: 10,
                      padding: "1px 5px",
                      borderRadius: 999,
                      background: isActive ? "rgba(255,255,255,0.25)" : "rgba(51,65,85,0.6)",
                      color: isActive ? "#ffffff" : "#cbd5e1"
                    }}
                  >
                    {t.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Right: Presets Dropdown & Status Pill */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginLeft: "auto" }}>
          {/* Quick Model Presets Dropdown */}
          <select
            defaultValue=""
            onChange={(e) => {
              if (e.target.value) {
                applyModelPreset(e.target.value as "ising1d" | "heisenberg2d" | "hubbard2d" | "xxz1d");
                e.target.value = "";
              }
            }}
            style={{
              padding: "5px 10px",
              borderRadius: 6,
              border: "1px solid rgba(56, 189, 248, 0.35)",
              background: "rgba(11, 18, 37, 0.9)",
              color: "#7dd3fc",
              fontSize: 12,
              fontWeight: 600,
              cursor: "pointer",
              outline: "none",
            }}
            title="Kvant ko'p zarrali modellar shablonlari"
          >
            <option value="" disabled>⚡ Model Presets...</option>
            <option value="ising1d">🧲 1D Ising Chain (L=8, J=1, h=0.5)</option>
            <option value="heisenberg2d">🔄 2D Heisenberg Grid (4×4, J=1)</option>
            <option value="hubbard2d">⚛️ 2D Fermi-Hubbard (3×3, U=4, t=1)</option>
            <option value="xxz1d">🧬 1D Critical XXZ (L=10, Δ=1)</option>
          </select>

          {/* System & Model Status Pill */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "4px 10px",
              background: "#0b1225",
              borderRadius: 6,
              border: "1px solid rgba(51,65,85,0.6)",
              fontSize: 12,
            }}
          >
            <span style={{ color: "#94a3b8" }}>
              <strong style={{ color: "#38bdf8" }}>{siteCount}</strong> sites ({qubitCount}q)
            </span>
            <span style={{ color: "#334155" }}>|</span>
            <span style={{ color: hamiltonian ? "#4ade80" : "#f59e0b", fontWeight: 600 }}>
              {hamiltonian ? `${hamiltonian.terms.length} terms` : "No Model"}
            </span>
          </div>
        </div>
      </div>

      {/* 4. TAB CONTENTS */}

      {/* OVERVIEW TAB: Lattice Geometry + Hamiltonian Plugin */}
      {(labTab === "overview" || labTab === "all") && (
        <div style={{ display: "grid", gap: 18, marginBottom: 18 }}>
          <section style={{ display: "grid", gridTemplateColumns: "minmax(320px, 0.8fr) minmax(440px, 1.2fr)", gap: 18 }}>
            {/* Lattice Design Card */}
            <div style={card}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ fontWeight: 700, fontSize: 16 }}>Lattice geometry</div>
                <span style={{ fontSize: 11, padding: "3px 8px", borderRadius: 4, background: "#0b1225", color: "#38bdf8" }}>Deterministic Snake</span>
              </div>
              <div style={{ color: "#64748b", fontSize: 12, marginTop: 4 }}>
                Configure 1D chain, 2D grid, or 3D cube geometry with open or periodic boundaries.
              </div>
              <div style={{ display: "grid", gap: 14, marginTop: 18 }}>
                <label style={fieldStyle}>
                  Dimension
                  <select value={dimension} onChange={(e) => setDimension(Number(e.target.value) as 1 | 2 | 3)} style={input}>
                    <option value={1}>1D chain</option>
                    <option value={2}>2D grid</option>
                    <option value={3}>3D grid</option>
                  </select>
                </label>
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                  {activeDimensions.map((size, index) => (
                    <label key={index} style={fieldStyle}>
                      {index === 0 ? "X size" : index === 1 ? "Y size" : "Z size"}
                      <input type="number" min={1} max={16} value={size} onChange={(e) => updateSize(index, e.target.value)} style={input} />
                    </label>
                  ))}
                </div>
                <label style={fieldStyle}>
                  Boundary conditions
                  <select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={input}>
                    <option value="open">Open (OBC)</option>
                    <option value="periodic">Periodic (PBC)</option>
                  </select>
                </label>
                <div style={{ padding: "8px 12px", borderRadius: 6, background: "rgba(11,18,37,0.7)", border: "1px solid rgba(51,65,85,0.4)", color: qubitCount > 64 ? "#fbbf24" : "#67e8f9", fontSize: 12 }}>
                  <strong>{siteCount}</strong> sites · <strong>{qubitCount}</strong> mapped qubits
                  {qubitCount > 64 ? " (preview only above 64 qubits)" : " • ready for simulation"}
                </div>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  <Button variant="accent" onClick={() => void preview()} disabled={busy !== null}>
                    {busy === "preview" ? "Previewing…" : "Preview lattice"}
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={() => { if (qubitCount <= 64) setNQubits(qubitCount); }}
                    disabled={qubitCount > 64 || busy !== null}
                    title="Circuit editor qubitlar sonini ushbu panjaraga tenglashtirish"
                  >
                    Adopt {qubitCount} qubits to circuit
                  </Button>
                </div>
              </div>
            </div>

            {/* Lattice Visualization Canvas Card */}
            <div style={card}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <div style={{ fontWeight: 700, fontSize: 15 }}>Spatial preview & graph layout</div>
                <span style={{ fontSize: 12, color: "#94a3b8" }}>{activeDimensions.join(" × ")}</span>
              </div>
              {graph ? (
                <LatticeCanvas graph={graph} />
              ) : (
                <div style={{ minHeight: 280, display: "grid", placeItems: "center", border: "1px dashed #334155", borderRadius: 12, color: "#64748b", background: "rgba(11,18,37,0.5)" }}>
                  Preview a lattice to inspect its spatial geometry and couplings.
                </div>
              )}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10, marginTop: 14 }}>
                {[
                  ["dimension", graph?.dimension ?? `${dimension}D`],
                  ["sites", graph?.sites.length ?? siteCount],
                  ["edges", graph?.edges.length ?? "—"],
                  ["ordering", graph?.ordering ?? "snake"]
                ].map(([label, value]) => (
                  <div key={label} style={{ padding: "8px 10px", borderRadius: 8, background: "#0b1225", border: "1px solid rgba(51,65,85,.4)" }}>
                    <div style={{ color: "#64748b", fontSize: 10, textTransform: "uppercase" }}>{label}</div>
                    <div style={{ color: "#e0f2fe", fontWeight: 700, marginTop: 2, fontSize: 13 }}>{value}</div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Hamiltonian Plugin Card */}
          <section style={card}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
              <div>
                <div style={{ fontWeight: 700, fontSize: 16 }}>Hamiltonian model synthesizer</div>
                <div style={{ color: "#64748b", fontSize: 12, marginTop: 2 }}>
                  Generate sparse Pauli operators for energy evaluation, TEBD real-time evolution, and variational DMRG.
                </div>
              </div>
              <div style={{ padding: "4px 10px", borderRadius: 6, background: hamiltonian ? "rgba(34,197,94,.1)" : "rgba(234,179,8,.1)", color: hamiltonian ? "#4ade80" : "#facc15", fontSize: 12, fontWeight: 600 }}>
                {hamiltonian ? `✓ ${hamiltonian.terms.length} terms loaded` : "No model generated"}
              </div>
            </div>
            <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "end", marginTop: 16 }}>
              <label style={fieldStyle}>
                Domain
                <select
                  value={material}
                  onChange={(e) => {
                    setMaterial(e.target.value as MaterialName);
                    setHamiltonian(null);
                    setEnergyResult(null);
                    setGroundResult(null);
                    setDmrgResult(null);
                    setTebdResult(null);
                    setPepsResult(null);
                  }}
                  style={input}
                >
                  <option value="spin">Spin lattice</option>
                  <option value="hubbard">Hubbard materials</option>
                </select>
              </label>
              {material === "spin" ? (
                <>
                  <label style={fieldStyle}>
                    Model
                    <select value={model} onChange={(e) => setModel(e.target.value as ModelName)} style={input}>
                      <option value="ising">Transverse Ising</option>
                      <option value="heisenberg">Heisenberg</option>
                      <option value="xxz">XXZ</option>
                    </select>
                  </label>
                  <label style={fieldStyle}>
                    Coupling J
                    <input type="number" step="any" value={coupling} onChange={(e) => setCoupling(Number(e.target.value) || 0)} style={input} />
                  </label>
                  <label style={fieldStyle}>
                    Field h
                    <input type="number" step="any" value={field} onChange={(e) => setField(Number(e.target.value) || 0)} style={input} />
                  </label>
                  <label style={fieldStyle}>
                    Anisotropy Δ
                    <input type="number" step="any" value={anisotropy} onChange={(e) => setAnisotropy(Number(e.target.value) || 0)} style={input} />
                  </label>
                </>
              ) : (
                <>
                  <label style={fieldStyle}>
                    Hopping t
                    <input type="number" step="any" value={hopping} onChange={(e) => setHopping(Number(e.target.value) || 0)} style={input} />
                  </label>
                  <label style={fieldStyle}>
                    On-site U
                    <input type="number" step="any" value={onsiteU} onChange={(e) => setOnsiteU(Number(e.target.value) || 0)} style={input} />
                  </label>
                  <label style={fieldStyle}>
                    Chemical μ
                    <input type="number" step="any" value={chemicalPotential} onChange={(e) => setChemicalPotential(Number(e.target.value) || 0)} style={input} />
                  </label>
                </>
              )}
              <Button variant="accent" onClick={() => void generateHamiltonian()} disabled={busy !== null}>
                {busy === "hamiltonian" ? "Generating…" : "Generate Hamiltonian"}
              </Button>
            </div>
            {hamiltonian ? (
              <details style={{ marginTop: 14, color: "#94a3b8", fontSize: 12 }}>
                <summary style={{ cursor: "pointer", color: "#38bdf8" }}>Inspect sparse Pauli terms ({hamiltonian.terms.length} terms)</summary>
                <pre style={{ maxHeight: 180, overflow: "auto", background: "#0b1225", padding: 12, borderRadius: 10, marginTop: 8 }}>
                  {JSON.stringify(hamiltonian.terms.slice(0, 16), null, 2)}
                </pre>
              </details>
            ) : null}
          </section>
        </div>
      )}

      {/* SOLVERS TAB: DMRG, Exact Diagonalization & Circuit Energy */}
      {(labTab === "solvers" || labTab === "all") && (
        <section style={{ ...card, marginBottom: 18 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, flexWrap: "wrap", gap: 12 }}>
            <div>
              <div style={{ fontWeight: 700, fontSize: 16 }}>Variational & Ground State Solvers</div>
              <p style={{ color: "#64748b", fontSize: 12, margin: "2px 0 0" }}>
                Optimize ground state using DMRG (Density Matrix Renormalization Group) or Exact Diagonalization check.
              </p>
            </div>
            {!hamiltonian && (
              <div style={{ color: "#fbbf24", fontSize: 12 }}>⚠️ Generate Hamiltonian first in the Lattice tab</div>
            )}
          </div>
          <div style={{ display: "flex", gap: 14, alignItems: "end", flexWrap: "wrap", padding: "12px 0" }}>
            <label style={fieldStyle}>
              Bond dimension χ
              <input type="number" min={1} max={64} value={bondDim} onChange={(e) => setBondDim(Math.max(1, Math.min(64, Number(e.target.value) || 1)))} style={input} />
            </label>
            <label style={fieldStyle}>
              DMRG sweeps
              <input type="number" min={1} max={64} value={sweeps} onChange={(e) => setSweeps(Math.max(1, Math.min(64, Number(e.target.value) || 1)))} style={input} />
            </label>
            <Button variant="accent" onClick={() => void runEnergy()} disabled={busy !== null || !hamiltonian || hamiltonian.n_qubits !== nQubits}>
              {busy === "energy" ? "Evaluating…" : "Evaluate MPS Circuit Energy"}
            </Button>
            <Button variant="primary" onClick={() => void runVariationalGround()} disabled={busy !== null || !hamiltonian}>
              {busy === "dmrg" ? "Optimizing…" : "DMRG Ground State"}
            </Button>
            <Button variant="ghost" onClick={() => void runGround()} disabled={busy !== null || !hamiltonian || hamiltonian.n_qubits > 12}>
              {busy === "ground" ? "Diagonalizing…" : "Exact ED (≤12 qubits)"}
            </Button>
          </div>
          <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "baseline", marginTop: 18, padding: "16px", borderRadius: 10, background: "#0b1225", border: "1px solid rgba(51,65,85,.5)" }}>
            {energy != null ? (
              <div>
                <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase" }}>MPS Circuit Energy</div>
                <div style={{ fontSize: 28, fontWeight: 750, color: "#38bdf8" }}>{Number(energy).toFixed(7)}</div>
              </div>
            ) : null}
            {dmrgEnergy != null ? (
              <div>
                <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase" }}>DMRG Ground Energy</div>
                <div style={{ fontSize: 28, fontWeight: 750, color: "#4ade80" }}>{Number(dmrgEnergy).toFixed(7)}</div>
              </div>
            ) : null}
            {groundEnergy != null ? (
              <div>
                <div style={{ fontSize: 11, color: "#64748b", textTransform: "uppercase" }}>Exact ED Ground</div>
                <div style={{ fontSize: 28, fontWeight: 750, color: "#f472b6" }}>{Number(groundEnergy).toFixed(7)}</div>
              </div>
            ) : null}
            {energy == null && dmrgEnergy == null && groundEnergy == null && (
              <div style={{ color: "#64748b", fontSize: 13 }}>No solver run yet. Choose a solver above to compute ground state energy.</div>
            )}
          </div>
        </section>
      )}

      {/* DYNAMICS TAB: TEBD / PEPS Real-Time Evolution */}
      {(labTab === "dynamics" || labTab === "all") && (
        <section style={{ ...card, marginBottom: 18 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <div>
              <div style={{ fontWeight: 700, fontSize: 16 }}>Time Evolution & Dynamics (TEBD / PEPS)</div>
              <p style={{ color: "#64748b", fontSize: 12, margin: "2px 0 0" }}>
                Real-time unitary or imaginary evolution on 1D chains (MPS/TEBD) or 2D/3D lattices (PEPS).
              </p>
            </div>
          </div>
          {hamiltonian && !tebdReady ? (
            <div style={{ color: "#fbbf24", fontSize: 12, marginBottom: 12, padding: 8, background: "rgba(245,158,11,.1)", borderRadius: 6 }}>
              ⚠️ Jordan–Wigner parity string exceeds safe limit; TEBD is disabled for this Hamiltonian.
            </div>
          ) : null}
          <div style={{ display: "flex", gap: 14, alignItems: "end", flexWrap: "wrap", padding: "10px 0" }}>
            <label style={fieldStyle}>
              Time step dt
              <input type="number" step="any" value={dt} onChange={(e) => setDt(Number(e.target.value) || 0.01)} style={input} />
            </label>
            <label style={fieldStyle}>
              Steps
              <input type="number" min={1} max={10000} value={steps} onChange={(e) => setSteps(Math.max(1, Math.min(10000, Number(e.target.value) || 1)))} style={input} />
            </label>
            <label style={fieldStyle}>
              PEPS method
              <select value={pepsContraction} onChange={(e) => setPepsContraction(e.target.value as PEPSContraction)} style={input}>
                <option value="auto">Auto · double-layer</option>
                <option value="boundary-mps">Boundary-MPS · 2D open</option>
              </select>
            </label>
            {pepsContraction === "boundary-mps" && (
              <label style={fieldStyle}>
                Environment χ
                <input type="number" min={1} max={64} value={boundaryBondDim} onChange={(e) => setBoundaryBondDim(Math.max(1, Math.min(64, Number(e.target.value) || 1)))} style={input} />
              </label>
            )}
            <Button variant="danger" onClick={() => void runEvolution()} disabled={busy !== null || !hamiltonian || hamiltonian.n_qubits !== nQubits || !tebdReady}>
              {busy === "tebd" ? "Evolving…" : "Run TEBD (1D)"}
            </Button>
            <Button variant="secondary" onClick={() => void runNativePEPS()} disabled={busy !== null || !hamiltonian || !pepsEligible || hamiltonian.n_qubits !== nQubits || (pepsContraction === "boundary-mps" && (dimension !== 2 || boundary !== "open"))}>
              {busy === "peps" ? "Evolving…" : "Run native PEPS (2D/3D)"}
            </Button>
          </div>
          {tebdResult ? (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10, marginTop: 14, fontSize: 12, padding: 12, background: "#0b1225", borderRadius: 8 }}>
              <div><span style={{ color: "#64748b" }}>TEBD norm²</span><br /><strong style={{ color: "#e0f2fe" }}>{Number(tebdResult.norm2 ?? 0).toFixed(6)}</strong></div>
              <div><span style={{ color: "#64748b" }}>Bond dim used</span><br /><strong style={{ color: "#e0f2fe" }}>{String(tebdResult.bond_dim_used ?? "—")}</strong></div>
              <div><span style={{ color: "#64748b" }}>Discarded weight</span><br /><strong style={{ color: "#e0f2fe" }}>{Number(tebdResult.discarded_weight ?? 0).toExponential(2)}</strong></div>
            </div>
          ) : null}
          {pepsResult ? (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10, marginTop: 14, fontSize: 12, padding: 12, background: "#0b1225", borderRadius: 8 }}>
              <div><span style={{ color: "#64748b" }}>PEPS norm²</span><br /><strong style={{ color: "#e0f2fe" }}>{Number(pepsResult.norm2 ?? 0).toFixed(6)}</strong></div>
              <div><span style={{ color: "#64748b" }}>Bond dim used</span><br /><strong style={{ color: "#e0f2fe" }}>{String(pepsResult.bond_dim_used ?? "—")}</strong></div>
              <div><span style={{ color: "#64748b" }}>Discarded weight</span><br /><strong style={{ color: "#e0f2fe" }}>{Number(pepsResult.discarded_weight ?? 0).toExponential(2)}</strong></div>
            </div>
          ) : null}
          {energies.length ? (
            <div style={{ marginTop: 16 }}>
              <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 8, color: "#cbd5e1" }}>Energy trajectory E(t)</div>
              <EnergyChart values={energies} />
              {warningList.length ? <div style={{ color: "#fbbf24", fontSize: 12, marginTop: 8 }}>{warningList.join(" · ")}</div> : null}
            </div>
          ) : null}
        </section>
      )}

      {/* CTMRG TAB: 2D Infinite Thermodynamics */}
      {(labTab === "ctmrg" || labTab === "all") && material === "spin" && (
        <div style={{ marginBottom: 18 }}>
          <CTMRGLabPanel
            dimensions={activeDimensions}
            model={model}
            coupling={coupling}
            field={field}
            anisotropy={anisotropy}
            onResult={(result, request, label) => {
              setLatestOutput(result);
              addExperiment({ label: `Physics · ${label}`, source: request.source, kind: request.kind, status: "done", request, result, provenance: result.provenance });
            }}
          />
        </div>
      )}

      {/* CONVERGENCE STUDY TAB */}
      {(labTab === "convergence" || labTab === "all") && (
        <div style={{ marginBottom: 18 }}>
          <PhysicsConvergenceStudy
            mode={studyMode}
            rows={studyRows}
            manifest={studyManifest}
            running={busy === "study"}
            disabled={!hamiltonian || busy !== null}
            availableModes={["dmrg", ...(tebdReady ? ["tebd" as const] : []), ...(pepsEligible ? ["peps" as const] : [])]}
            onModeChange={setStudyMode}
            onRun={() => void runConvergenceStudy(studyMode)}
          />
          {pepsResult ? <ConvergenceDiagnostics result={pepsResult as AgentResult} /> : null}
        </div>
      )}

      {/* OBSERVABLES TABLE */}
      {observableRows.length > 0 && (
        <section style={{ ...card, marginTop: 18 }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "baseline" }}>
            <div>
              <div style={{ fontWeight: 700 }}>Structured observables</div>
              <div style={{ color: "#64748b", fontSize: 12, marginTop: 4 }}>Named Pauli expectations are stored with provenance.</div>
            </div>
            <div style={{ color: "#67e8f9", fontSize: 12 }}>{observableRows.length} observables</div>
          </div>
          <div style={{ overflowX: "auto", marginTop: 14 }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ color: "#64748b", textAlign: "left" }}>
                  <th style={{ padding: "8px 10px" }}>Label</th>
                  <th style={{ padding: "8px 10px" }}>Pauli support</th>
                  <th style={{ padding: "8px 10px" }}>Coefficient</th>
                  <th style={{ padding: "8px 10px" }}>Value</th>
                </tr>
              </thead>
              <tbody>
                {observableRows.map((row, index) => (
                  <tr key={`${String(row.label ?? "observable")}-${index}`} style={{ borderTop: "1px solid #1e293b" }}>
                    <td style={{ padding: "8px 10px", color: "#e0f2fe" }}>{String(row.label ?? `Observable ${index + 1}`)}</td>
                    <td style={{ padding: "8px 10px", color: "#94a3b8" }}>
                      {row.paulis && typeof row.paulis === "object"
                        ? Object.entries(row.paulis as Record<string, unknown>).map(([qubit, pauli]) => `${String(pauli)}${qubit}`).join(" · ") || "I"
                        : "—"}
                    </td>
                    <td style={{ padding: "8px 10px" }}>{Number(row.coefficient ?? 1).toFixed(4)}</td>
                    <td style={{ padding: "8px 10px", color: "#67e8f9" }}>{Number(row.value ?? 0).toFixed(8)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {jobProgress ? <ComputeProgress job={jobProgress} dark onCancel={() => void cancelJob()} canceling={canceling} onRetry={() => retryRef.current?.()} retrying={busy !== null} /> : null}

      {error ? (
        <div role="alert" style={{ marginTop: 18, padding: 12, borderRadius: 10, color: "#fecaca", background: "rgba(127,29,29,.35)", border: "1px solid rgba(248,113,113,.3)", fontSize: 13 }}>
          {error}
        </div>
      ) : null}
    </main>
  );

}
