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

const shell: React.CSSProperties = { maxWidth: 1180, width: "100%", margin: "0 auto", padding: "32px 28px 54px", color: "var(--qc-dark-text)" };
const card: React.CSSProperties = { border: "1px solid var(--qc-dark-border)", borderRadius: "var(--qc-radius-xl)", background: "var(--qc-dark-surface)", padding: 20, boxShadow: "var(--qc-shadow-lg)" };
const fieldStyle: React.CSSProperties = { display: "grid", gap: 6, color: "var(--qc-dark-text-muted)", fontSize: 12 };
const input: React.CSSProperties = { width: 82, border: "1px solid var(--qc-dark-border-strong)", borderRadius: "var(--qc-radius-md)", background: "var(--qc-dark-canvas)", color: "var(--qc-dark-text)", padding: "8px 9px" };

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

  return <main style={{ ...shell, background: "radial-gradient(circle at 10% 0%, rgba(14,165,233,.16), transparent 34%), radial-gradient(circle at 90% 8%, rgba(168,85,247,.15), transparent 32%)", minHeight: "100%" }}>
    <div style={{ display: "flex", justifyContent: "space-between", gap: 20, alignItems: "end", marginBottom: 24 }}><div><div style={{ color: "#67e8f9", fontSize: 12, letterSpacing: ".14em", textTransform: "uppercase" }}>{material === "spin" ? "Physics plugins · spin lattice" : "Physics plugins · Hubbard materials"}</div><h1 style={{ fontSize: 30, margin: "8px 0 6px", letterSpacing: "-.04em" }}>Many-body laboratory</h1><p style={{ margin: 0, color: "#94a3b8", maxWidth: 680 }}>Build a 1D, 2D or 3D lattice, generate a sparse Hamiltonian, then evaluate energy or evolve it with GPU MPS/TEBD/PEPS. Every large run is admitted by a memory and contraction-width preflight.</p></div><div style={{ color: "#64748b", fontSize: 12, textAlign: "right" }}>Current circuit<br /><strong style={{ color: "#e2e8f0", fontSize: 18 }}>{nQubits} qubits</strong></div></div>
    <section style={{ display: "grid", gridTemplateColumns: "minmax(270px, .7fr) minmax(420px, 1.3fr)", gap: 18 }}>
      <div style={card}><div style={{ fontWeight: 700, fontSize: 16 }}>Lattice design</div><div style={{ color: "#64748b", fontSize: 12, marginTop: 5 }}>Snake ordering keeps the mapping deterministic for MPS.</div><div style={{ display: "grid", gap: 14, marginTop: 20 }}><label style={fieldStyle}>Dimension<select value={dimension} onChange={(e) => setDimension(Number(e.target.value) as 1 | 2 | 3)} style={input}><option value={1}>1D chain</option><option value={2}>2D grid</option><option value={3}>3D grid</option></select></label><div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>{activeDimensions.map((size, index) => <label key={index} style={fieldStyle}>{index === 0 ? "X" : index === 1 ? "Y" : "Z"}<input type="number" min={1} max={16} value={size} onChange={(e) => updateSize(index, e.target.value)} style={input} /></label>)}</div><label style={fieldStyle}>Boundary<select value={boundary} onChange={(e) => setBoundary(e.target.value as Boundary)} style={input}><option value="open">Open</option><option value="periodic">Periodic</option></select></label><div style={{ color: qubitCount > 64 ? "#fbbf24" : "#67e8f9", fontSize: 13 }}>{siteCount} sites · {qubitCount} mapped qubits{qubitCount > 64 ? " · preview only above 64 qubits" : " · ready for circuit coupling"}</div><div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}><Button variant="accent" onClick={() => void preview()} disabled={busy !== null}>{busy === "preview" ? "Previewing…" : "Preview lattice"}</Button><Button variant="secondary" onClick={() => { if (qubitCount <= 64) setNQubits(qubitCount); }} disabled={qubitCount > 64 || busy !== null}>Adopt {qubitCount} qubits</Button></div></div></div>
      <div style={card}>{graph ? <LatticeCanvas graph={graph} /> : <div style={{ minHeight: 290, display: "grid", placeItems: "center", border: "1px dashed #334155", borderRadius: 14, color: "#64748b" }}>Preview a lattice to inspect its geometry.</div>}<div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10, marginTop: 16 }}>{[["dimension", graph?.dimension ?? "—"], ["sites", graph?.sites.length ?? "—"], ["edges", graph?.edges.length ?? "—"], ["order", graph?.ordering ?? "—"]].map(([label, value]) => <div key={label} style={{ padding: "10px 11px", borderRadius: 10, background: "#0b1225" }}><div style={{ color: "#64748b", fontSize: 10, textTransform: "uppercase" }}>{label}</div><div style={{ color: "#e0f2fe", fontWeight: 700, marginTop: 4, fontSize: 14 }}>{value}</div></div>)}</div></div>
    </section>
    <section style={{ ...card, marginTop: 18 }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
        <div><div style={{ fontWeight: 700, fontSize: 16 }}>Hamiltonian plugin</div><div style={{ color: "#64748b", fontSize: 12, marginTop: 5 }}>Sparse Pauli terms are reusable by energy, TEBD and future domain modules.</div></div>
        <div style={{ color: "#67e8f9", fontSize: 13 }}>{hamiltonian ? `${hamiltonian.terms.length} terms generated` : "No model loaded"}</div>
      </div>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "end", marginTop: 18 }}>
        <label style={fieldStyle}>Domain<select value={material} onChange={(e) => { setMaterial(e.target.value as MaterialName); setHamiltonian(null); setEnergyResult(null); setGroundResult(null); setDmrgResult(null); setTebdResult(null); setPepsResult(null); }} style={input}><option value="spin">Spin lattice</option><option value="hubbard">Hubbard materials</option></select></label>
        {material === "spin" ? <>
          <label style={fieldStyle}>Model<select value={model} onChange={(e) => setModel(e.target.value as ModelName)} style={input}><option value="ising">Transverse Ising</option><option value="heisenberg">Heisenberg</option><option value="xxz">XXZ</option></select></label>
          <label style={fieldStyle}>J<input type="number" step="any" value={coupling} onChange={(e) => setCoupling(Number(e.target.value) || 0)} style={input} /></label>
          <label style={fieldStyle}>Field h<input type="number" step="any" value={field} onChange={(e) => setField(Number(e.target.value) || 0)} style={input} /></label>
          <label style={fieldStyle}>Anisotropy Δ<input type="number" step="any" value={anisotropy} onChange={(e) => setAnisotropy(Number(e.target.value) || 0)} style={input} /></label>
        </> : <>
          <label style={fieldStyle}>Hopping t<input type="number" step="any" value={hopping} onChange={(e) => setHopping(Number(e.target.value) || 0)} style={input} /></label>
          <label style={fieldStyle}>On-site U<input type="number" step="any" value={onsiteU} onChange={(e) => setOnsiteU(Number(e.target.value) || 0)} style={input} /></label>
          <label style={fieldStyle}>Chemical μ<input type="number" step="any" value={chemicalPotential} onChange={(e) => setChemicalPotential(Number(e.target.value) || 0)} style={input} /></label>
        </>}
        <Button variant="accent" onClick={() => void generateHamiltonian()} disabled={busy !== null}>{busy === "hamiltonian" ? "Generating…" : "Generate Hamiltonian"}</Button>
      </div>
      {hamiltonian ? <details style={{ marginTop: 16, color: "#94a3b8", fontSize: 12 }}><summary style={{ cursor: "pointer" }}>Inspect sparse terms</summary><pre style={{ maxHeight: 180, overflow: "auto", background: "#0b1225", padding: 12, borderRadius: 10 }}>{JSON.stringify(hamiltonian.terms.slice(0, 12), null, 2)}</pre></details> : null}
    </section>
    <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(300px,1fr))", gap: 18, marginTop: 18 }}><div style={card}><div style={{ fontWeight: 700 }}>Energy and ground state</div><p style={{ color: "#64748b", fontSize: 12, lineHeight: 1.5 }}>Use MPS for the current circuit, DMRG for a variational ground state, or exact diagonalization as a small-system check.</p><div style={{ display: "flex", gap: 14, alignItems: "end", flexWrap: "wrap" }}><label style={fieldStyle}>Bond dimension<input type="number" min={1} max={64} value={bondDim} onChange={(e) => setBondDim(Math.max(1, Math.min(64, Number(e.target.value) || 1)))} style={input} /></label><label style={fieldStyle}>DMRG sweeps<input type="number" min={1} max={64} value={sweeps} onChange={(e) => setSweeps(Math.max(1, Math.min(64, Number(e.target.value) || 1)))} style={fieldStyle} /></label><Button variant="accent" onClick={() => void runEnergy()} disabled={busy !== null || !hamiltonian || hamiltonian.n_qubits !== nQubits}>{busy === "energy" ? "Running…" : "Evaluate energy"}</Button><Button variant="primary" onClick={() => void runVariationalGround()} disabled={busy !== null || !hamiltonian}>{busy === "dmrg" ? "Optimizing…" : "DMRG ground state"}</Button><Button variant="ghost" onClick={() => void runGround()} disabled={busy !== null || !hamiltonian || hamiltonian.n_qubits > 12}>{busy === "ground" ? "Diagonalizing…" : "Exact ground state"}</Button></div><div style={{ display: "flex", gap: 28, flexWrap: "wrap", alignItems: "baseline", marginTop: 22 }}>{energy != null ? <div style={{ fontSize: 30, fontWeight: 750, color: "#67e8f9" }}>{Number(energy).toFixed(7)}<span style={{ fontSize: 13, color: "#64748b", marginLeft: 8 }}>MPS energy</span></div> : null}{dmrgEnergy != null ? <div style={{ fontSize: 30, fontWeight: 750, color: "#7dd3fc" }}>{Number(dmrgEnergy).toFixed(7)}<span style={{ fontSize: 13, color: "#64748b", marginLeft: 8 }}>DMRG ground</span></div> : null}{groundEnergy != null ? <div style={{ fontSize: 30, fontWeight: 750, color: "#f0abfc" }}>{Number(groundEnergy).toFixed(7)}<span style={{ fontSize: 13, color: "#64748b", marginLeft: 8 }}>exact ground</span></div> : null}</div></div><div style={card}><div style={{ fontWeight: 700 }}>TEBD / PEPS evolution</div><p style={{ color: "#64748b", fontSize: 12, lineHeight: 1.5 }}>Choose the evolution backend that matches the geometry and problem size.</p>{hamiltonian && !tebdReady ? <div style={{ color: "#fbbf24", fontSize: 12, marginBottom: 12 }}>This Hamiltonian has Jordan–Wigner parity strings longer than the safe 64-locality limit; TEBD is disabled.</div> : null}{hamiltonian && tebdReady && hamiltonian.terms.some((term) => Object.keys(term.paulis).length > 2) ? <div style={{ color: "#94a3b8", fontSize: 12, marginBottom: 12 }}>Parity strings use a CX network; compare bond-dimension and time-step convergence.</div> : null}<div style={{ display: "flex", gap: 14, alignItems: "end", flexWrap: "wrap" }}><label style={fieldStyle}>dt<input type="number" step="any" value={dt} onChange={(e) => setDt(Number(e.target.value) || 0.01)} style={input} /></label><label style={fieldStyle}>Steps<input type="number" min={1} max={10000} value={steps} onChange={(e) => setSteps(Math.max(1, Math.min(10000, Number(e.target.value) || 1)))} style={input} /></label><label style={fieldStyle}>PEPS contraction<select value={pepsContraction} onChange={(e) => setPepsContraction(e.target.value as PEPSContraction)} style={input}><option value="auto">Auto · double-layer</option><option value="boundary-mps">Boundary-MPS · 2D open</option></select></label>{pepsContraction === "boundary-mps" ? <label style={fieldStyle}>Environment χ<input type="number" min={1} max={64} value={boundaryBondDim} onChange={(e) => setBoundaryBondDim(Math.max(1, Math.min(64, Number(e.target.value) || 1)))} style={input} /></label> : null}<Button variant="danger" onClick={() => void runEvolution()} disabled={busy !== null || !hamiltonian || hamiltonian.n_qubits !== nQubits || !tebdReady}>{busy === "tebd" ? "Evolving…" : "Run TEBD"}</Button><Button variant="secondary" onClick={() => void runNativePEPS()} disabled={busy !== null || !hamiltonian || !pepsEligible || hamiltonian.n_qubits !== nQubits || (pepsContraction === "boundary-mps" && (dimension !== 2 || boundary !== "open"))}>{busy === "peps" ? "Evolving…" : "Run native PEPS"}</Button></div>{pepsContraction === "boundary-mps" && (dimension !== 2 || boundary !== "open") ? <div style={{ color: "#fbbf24", fontSize: 12, marginTop: 10 }}>Boundary-MPS requires an open 2D lattice; choose Auto for 3D or periodic geometry.</div> : null}{pepsEligible ? <div style={{ color: "#64748b", fontSize: 12, marginTop: 10 }}>2D/3D spin lattice · bounded double-layer contraction · boundary-MPS environment χ is explicit · up to 64 sites</div> : null}{tebdResult ? <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10, marginTop: 18, fontSize: 12 }}><div><span style={{ color: "#64748b" }}>TEBD norm</span><br /><strong>{Number(tebdResult.norm2 ?? 0).toFixed(6)}</strong></div><div><span style={{ color: "#64748b" }}>TEBD bond</span><br /><strong>{String(tebdResult.bond_dim_used ?? "—")}</strong></div><div><span style={{ color: "#64748b" }}>TEBD discarded</span><br /><strong>{Number(tebdResult.discarded_weight ?? 0).toExponential(2)}</strong></div></div> : null}{pepsResult ? <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12, marginTop: 12, fontSize: 12 }}><div><span style={{ color: "#64748b" }}>PEPS norm</span><br /><strong>{Number(pepsResult.norm2 ?? 0).toFixed(6)}</strong></div><div><span style={{ color: "#64748b" }}>PEPS bond</span><br /><strong>{String(pepsResult.bond_dim_used ?? "—")}</strong></div><div><span style={{ color: "#64748b" }}>PEPS discarded</span><br /><strong>{Number(pepsResult.discarded_weight ?? 0).toExponential(2)}</strong></div></div> : null}</div></section>
    <PhysicsConvergenceStudy mode={studyMode} rows={studyRows} manifest={studyManifest} running={busy === "study"} disabled={!hamiltonian || busy !== null} availableModes={["dmrg", ...(tebdReady ? ["tebd" as const] : []), ...(pepsEligible ? ["peps" as const] : [])]} onModeChange={setStudyMode} onRun={() => void runConvergenceStudy(studyMode)} />
    {material === "spin" ? <CTMRGLabPanel dimensions={activeDimensions} model={model} coupling={coupling} field={field} anisotropy={anisotropy} onResult={(result, request, label) => { setLatestOutput(result); addExperiment({ label: `Physics · ${label}`, source: request.source, kind: request.kind, status: "done", request, result, provenance: result.provenance }); }} /> : null}
    {observableRows.length ? <section style={{ ...card, marginTop: 18 }}><div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "baseline" }}><div><div style={{ fontWeight: 700 }}>Structured observables</div><div style={{ color: "#64748b", fontSize: 12, marginTop: 4 }}>Named Pauli expectations are stored with the run artifact and provenance.</div></div><div style={{ color: "#67e8f9", fontSize: 12 }}>{observableRows.length}{evidenceResult && Array.isArray(evidenceResult.observables) && evidenceResult.observables.length > observableRows.length ? " shown" : " observables"}</div></div><div style={{ overflowX: "auto", marginTop: 14 }}><table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}><thead><tr style={{ color: "#64748b", textAlign: "left" }}><th style={{ padding: "8px 10px" }}>Label</th><th style={{ padding: "8px 10px" }}>Pauli support</th><th style={{ padding: "8px 10px" }}>Coefficient</th><th style={{ padding: "8px 10px" }}>Value</th></tr></thead><tbody>{observableRows.map((row, index) => <tr key={`${String(row.label ?? "observable")}-${index}`} style={{ borderTop: "1px solid #1e293b" }}><td style={{ padding: "8px 10px", color: "#e0f2fe" }}>{String(row.label ?? `Observable ${index + 1}`)}</td><td style={{ padding: "8px 10px", color: "#94a3b8" }}>{row.paulis && typeof row.paulis === "object" ? Object.entries(row.paulis as Record<string, unknown>).map(([qubit, pauli]) => `${String(pauli)}${qubit}`).join(" · ") || "I" : "—"}</td><td style={{ padding: "8px 10px" }}>{Number(row.coefficient ?? 1).toFixed(4)}</td><td style={{ padding: "8px 10px", color: "#67e8f9" }}>{Number(row.value ?? 0).toFixed(8)}</td></tr>)}</tbody></table></div></section> : null}
    {pepsResult ? <ConvergenceDiagnostics result={pepsResult as AgentResult} /> : null}
    {jobProgress ? <ComputeProgress job={jobProgress} dark onCancel={() => void cancelJob()} canceling={canceling} onRetry={() => retryRef.current?.()} retrying={busy !== null} /> : null}
    {energies.length ? <section style={{ ...card, marginTop: 18 }}><div style={{ fontWeight: 700, marginBottom: 12 }}>Energy trajectory</div><EnergyChart values={energies} />{warningList.length ? <div style={{ color: "#fbbf24", fontSize: 12, marginTop: 12 }}>{warningList.join(" · ")}</div> : null}</section> : null}
    {error ? <div role="alert" style={{ marginTop: 18, padding: 12, borderRadius: 10, color: "#fecaca", background: "rgba(127,29,29,.35)", border: "1px solid rgba(248,113,113,.3)", fontSize: 13 }}>{error}</div> : null}
  </main>;
}
