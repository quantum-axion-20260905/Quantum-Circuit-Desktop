"use client";

import React, { useState } from "react";

// ============================================================================
// 1. Potential Energy Surface (PES) Dissociation Curve
// ============================================================================

export type PESPoint = {
  r: number; // Bond distance in Angstroms
  hfEnergy: number; // Hartree-Fock energy (Ha)
  fciEnergy: number; // Full CI exact energy (Ha)
  vqeEnergy?: number; // VQE measured energy (Ha)
};

export type PESChartProps = {
  moleculeName: string;
  data: PESPoint[];
  currentR: number;
  equilibriumR?: number;
  dissociationEnergy?: number;
  onSelectR?: (r: number) => void;
};

export function PotentialEnergySurfaceChart({
  moleculeName,
  data,
  currentR,
  equilibriumR = 0.74,
  dissociationEnergy = 0.174,
  onSelectR,
}: PESChartProps) {
  const [hoveredPoint, setHoveredPoint] = useState<PESPoint | null>(null);

  if (data.length === 0) {
    return (
      <div style={{ color: "#94a3b8", padding: 20, textAlign: "center" }}>
        No PES data available.
      </div>
    );
  }

  const width = 640;
  const height = 280;
  const padL = 60;
  const padR = 24;
  const padT = 32;
  const padB = 40;

  const minR = Math.min(...data.map((d) => d.r));
  const maxR = Math.max(...data.map((d) => d.r));
  const allEnergies = data.flatMap((d) => [d.hfEnergy, d.fciEnergy, d.vqeEnergy].filter((v): v is number => v != null));
  const minE = Math.min(...allEnergies) - 0.05;
  const maxE = Math.max(...allEnergies) + 0.05;

  const toX = (r: number) => padL + ((r - minR) / (maxR - minR || 1)) * (width - padL - padR);
  const toY = (e: number) => height - padB - ((e - minE) / (maxE - minE || 1)) * (height - padT - padB);

  // Generate SVG polyline paths
  const hfPath = data.map((d) => `${toX(d.r)},${toY(d.hfEnergy)}`).join(" ");
  const fciPath = data.map((d) => `${toX(d.r)},${toY(d.fciEnergy)}`).join(" ");
  const vqePoints = data.filter((d) => d.vqeEnergy != null);

  const curX = toX(currentR);

  return (
    <div
      style={{
        border: "1px solid #334155",
        borderRadius: 8,
        background: "#111827",
        padding: 16,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 12,
          flexWrap: "wrap",
          gap: 8,
        }}
      >
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#f8fafc" }}>
            Potential Energy Surface (PES) · {moleculeName} Dissociation
          </div>
          <div style={{ fontSize: 11, color: "#94a3b8" }}>
            Born-Oppenheimer potential $E(R)$ · Re = {equilibriumR.toFixed(3)} Å · De ≈ {dissociationEnergy.toFixed(3)} Ha ({ (dissociationEnergy * 27.2114).toFixed(2) } eV)
          </div>
        </div>
        <div style={{ display: "flex", gap: 12, fontSize: 11 }}>
          <span style={{ color: "#94a3b8", display: "inline-flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 12, height: 2, background: "#94a3b8", display: "inline-block" }} /> HF Mean-field
          </span>
          <span style={{ color: "#38bdf8", display: "inline-flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 12, height: 2, background: "#38bdf8", display: "inline-block" }} /> Full CI (Exact)
          </span>
          <span style={{ color: "#34d399", display: "inline-flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#34d399", display: "inline-block" }} /> VQE Ansatz
          </span>
        </div>
      </div>

      <div style={{ width: "100%", overflowX: "auto" }}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          style={{ width: "100%", height: "auto", display: "block" }}
        >
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((pct) => {
            const eVal = minE + pct * (maxE - minE);
            const y = toY(eVal);
            return (
              <g key={`ygrid-${pct}`}>
                <line x1={padL} y1={y} x2={width - padR} y2={y} stroke="#1e293b" strokeDasharray="3,3" />
                <text x={padL - 8} y={y + 4} fill="#64748b" fontSize="10" textAnchor="end">
                  {eVal.toFixed(2)}
                </text>
              </g>
            );
          })}

          {/* R ticks */}
          {data.filter((_, i) => i % Math.max(1, Math.floor(data.length / 6)) === 0).map((d) => {
            const x = toX(d.r);
            return (
              <g key={`xgrid-${d.r}`}>
                <line x1={x} y1={height - padB} x2={x} y2={height - padB + 5} stroke="#475569" />
                <text x={x} y={height - padB + 16} fill="#64748b" fontSize="10" textAnchor="middle">
                  {d.r.toFixed(2)}
                </text>
              </g>
            );
          })}

          {/* Hartree-Fock Curve */}
          <polyline
            points={hfPath}
            fill="none"
            stroke="#64748b"
            strokeWidth="1.5"
            strokeDasharray="4,4"
          />

          {/* Full CI Exact Curve */}
          <polyline
            points={fciPath}
            fill="none"
            stroke="#38bdf8"
            strokeWidth="2.5"
            strokeLinecap="round"
          />

          {/* VQE Points */}
          {vqePoints.map((d) => (
            <circle
              key={`vqe-${d.r}`}
              cx={toX(d.r)}
              cy={toY(d.vqeEnergy!)}
              r="4"
              fill="#34d399"
              stroke="#065f46"
              strokeWidth="1.5"
            />
          ))}

          {/* Active Operating Bond Distance Marker Line */}
          <line
            x1={curX}
            y1={padT}
            x2={curX}
            y2={height - padB}
            stroke="#f59e0b"
            strokeWidth="1.5"
            strokeDasharray="2,2"
          />
          <text x={curX} y={padT - 6} fill="#f59e0b" fontSize="10" textAnchor="middle" fontWeight="bold">
            R = {currentR.toFixed(2)} Å
          </text>

          {/* Interactive clickable overlay columns */}
          {data.map((d) => {
            const x = toX(d.r);
            return (
              <rect
                key={`click-col-${d.r}`}
                x={x - 12}
                y={padT}
                width={24}
                height={height - padT - padB}
                fill="transparent"
                style={{ cursor: "pointer" }}
                onMouseEnter={() => setHoveredPoint(d)}
                onMouseLeave={() => setHoveredPoint(null)}
                onClick={() => onSelectR?.(d.r)}
              />
            );
          })}

          {/* Axis Labels */}
          <text x={width / 2} y={height - 8} fill="#94a3b8" fontSize="11" textAnchor="middle">
            Nuclear Separation R (Å)
          </text>
          <text
            x={16}
            y={height / 2}
            fill="#94a3b8"
            fontSize="11"
            textAnchor="middle"
            transform={`rotate(-90, 16, ${height / 2})`}
          >
            Electronic Energy E (Hartree)
          </text>
        </svg>
      </div>

      {/* Hover Info bar */}
      <div
        style={{
          marginTop: 8,
          fontSize: 11,
          color: "#94a3b8",
          display: "flex",
          justifyContent: "space-between",
          background: "#090d16",
          padding: "6px 12px",
          borderRadius: 4,
          border: "1px solid #1e293b",
        }}
      >
        {hoveredPoint ? (
          <div>
            <span style={{ color: "#38bdf8", fontWeight: 600 }}>R = {hoveredPoint.r.toFixed(3)} Å</span>
            {" · "}
            <span>Full CI: {hoveredPoint.fciEnergy.toFixed(5)} Ha</span>
            {" · "}
            <span>HF: {hoveredPoint.hfEnergy.toFixed(5)} Ha</span>
            {hoveredPoint.vqeEnergy != null && (
              <span style={{ color: "#34d399", fontWeight: 600 }}>
                {" · "}VQE: {hoveredPoint.vqeEnergy.toFixed(5)} Ha (Δ = {((hoveredPoint.vqeEnergy - hoveredPoint.fciEnergy) * 1000).toFixed(2)} mHa)
              </span>
            )}
          </div>
        ) : (
          <span>Click on the curve or drag the bond distance slider to compute PES point.</span>
        )}
        <span>1 Hartree = 27.2114 eV = 627.5 kcal/mol</span>
      </div>
    </div>
  );
}

// ============================================================================
// 2. Molecular Orbitals Energy Diagram
// ============================================================================

export type OrbitalLevel = {
  id: string;
  name: string; // e.g. 1\sigma_g, 1\sigma_u^*, 2\sigma_g, 1\pi_u
  energy: number; // in eV or Hartree
  occupancy: 0 | 1 | 2; // number of electrons
  symmetry?: string;
};

export type MolecularOrbitalsProps = {
  orbitals: OrbitalLevel[];
  homoLumoGap?: number; // in eV
  title?: string;
};

export function MolecularOrbitalsDiagram({
  orbitals,
  homoLumoGap,
  title = "Molecular Orbital Energy Levels (MO Spectrum)",
}: MolecularOrbitalsProps) {
  const width = 540;
  const height = 280;
  const padT = 32;
  const padB = 30;
  const padL = 60;
  const padR = 100;

  const energies = orbitals.map((o) => o.energy);
  const minE = Math.min(...energies) - 1.5;
  const maxE = Math.max(...energies) + 1.5;

  const toY = (e: number) => height - padB - ((e - minE) / (maxE - minE || 1)) * (height - padT - padB);

  // Find HOMO and LUMO
  const occupied = orbitals.filter((o) => o.occupancy > 0);
  const unoccupied = orbitals.filter((o) => o.occupancy === 0);
  const homo = occupied[occupied.length - 1];
  const lumo = unoccupied[0];
  const calculatedGap = homo && lumo ? Math.abs(lumo.energy - homo.energy) : homoLumoGap;

  return (
    <div
      style={{
        border: "1px solid #334155",
        borderRadius: 8,
        background: "#111827",
        padding: 16,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 10,
          flexWrap: "wrap",
        }}
      >
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#f8fafc" }}>{title}</div>
          <div style={{ fontSize: 11, color: "#94a3b8" }}>
            Energy spectrum · {calculatedGap ? `HOMO-LUMO Gap Δ = ${calculatedGap.toFixed(2)} eV` : "Orbitals"}
          </div>
        </div>
        <div style={{ fontSize: 11, color: "#cbd5e1", display: "flex", gap: 12 }}>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 10, height: 10, background: "#3b82f6", borderRadius: 2 }} /> Occupied
          </span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 10, height: 10, background: "#475569", borderRadius: 2 }} /> Virtual (Unoccupied)
          </span>
        </div>
      </div>

      <div style={{ width: "100%", overflowX: "auto" }}>
        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height: "auto", display: "block" }}>
          {/* Y Axis Grid */}
          {[0, 0.25, 0.5, 0.75, 1].map((pct) => {
            const eVal = minE + pct * (maxE - minE);
            const y = toY(eVal);
            return (
              <g key={`ogrid-${pct}`}>
                <line x1={padL} y1={y} x2={width - padR} y2={y} stroke="#1e293b" strokeDasharray="3,3" />
                <text x={padL - 8} y={y + 4} fill="#64748b" fontSize="10" textAnchor="end">
                  {eVal.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Molecular Orbitals Levels */}
          {orbitals.map((orb, index) => {
            const y = toY(orb.energy);
            const isHomo = orb === homo;
            const isLumo = orb === lumo;
            const isOcc = orb.occupancy > 0;
            const barW = 120;
            const barX = padL + 60 + (index % 2 === 0 ? 0 : 70);

            return (
              <g key={orb.id}>
                {/* Horizontal Level Bar */}
                <line
                  x1={barX}
                  y1={y}
                  x2={barX + barW}
                  y2={y}
                  stroke={isOcc ? "#38bdf8" : "#64748b"}
                  strokeWidth={isHomo || isLumo ? 3.5 : 2}
                  strokeLinecap="round"
                />

                {/* Spin Arrows for Occupancy */}
                {orb.occupancy >= 1 && (
                  <text x={barX + barW / 2 - 8} y={y - 4} fill="#f8fafc" fontSize="12" fontWeight="bold">
                    ↑
                  </text>
                )}
                {orb.occupancy === 2 && (
                  <text x={barX + barW / 2 + 4} y={y + 11} fill="#f8fafc" fontSize="12" fontWeight="bold">
                    ↓
                  </text>
                )}

                {/* Orbital Name Label */}
                <text x={barX + barW + 12} y={y + 4} fill={isOcc ? "#e2e8f0" : "#94a3b8"} fontSize="11" fontFamily="monospace">
                  {orb.name}
                  {isHomo ? " (HOMO)" : ""}
                  {isLumo ? " (LUMO)" : ""}
                </text>

                {/* Energy Value Label */}
                <text x={barX - 8} y={y + 3} fill="#64748b" fontSize="10" textAnchor="end">
                  {orb.energy.toFixed(2)} eV
                </text>
              </g>
            );
          })}

          {/* HOMO-LUMO Gap Arrow & Label */}
          {homo && lumo && (
            <g>
              <line
                x1={width - padR + 20}
                y1={toY(homo.energy)}
                x2={width - padR + 20}
                y2={toY(lumo.energy)}
                stroke="#f59e0b"
                strokeWidth="2"
                markerEnd="url(#arrow)"
              />
              <text
                x={width - padR + 30}
                y={(toY(homo.energy) + toY(lumo.energy)) / 2 + 4}
                fill="#f59e0b"
                fontSize="11"
                fontWeight="bold"
              >
                Δ = {calculatedGap?.toFixed(2)} eV
              </text>
            </g>
          )}

          {/* Y Axis Title */}
          <text
            x={16}
            y={height / 2}
            fill="#94a3b8"
            fontSize="11"
            textAnchor="middle"
            transform={`rotate(-90, 16, ${height / 2})`}
          >
            Orbital Energy (eV)
          </text>
        </svg>
      </div>
    </div>
  );
}

// ============================================================================
// 3. VQE Optimizer Convergence Trace
// ============================================================================

export type VQETraceProps = {
  energies: number[];
  targetEnergy: number; // Exact Full CI target
  chemicalAccuracyMHa?: number; // default 1.6 mHa (1 kcal/mol)
  optimizerName?: string;
};

export function VQEOptimizationTrace({
  energies,
  targetEnergy,
  chemicalAccuracyMHa = 1.6,
  optimizerName = "COBYLA",
}: VQETraceProps) {
  if (energies.length === 0) {
    return (
      <div style={{ color: "#94a3b8", padding: 18, textAlign: "center" }}>
        No VQE trace history recorded yet. Run VQE Optimization to monitor step-by-step convergence.
      </div>
    );
  }

  const width = 560;
  const height = 240;
  const padL = 64;
  const padR = 24;
  const padT = 30;
  const padB = 36;

  const minE = Math.min(...energies, targetEnergy) - 0.005;
  const maxE = Math.max(...energies) + 0.005;

  const toX = (idx: number) => padL + (idx / Math.max(1, energies.length - 1)) * (width - padL - padR);
  const toY = (e: number) => height - padB - ((e - minE) / (maxE - minE || 1)) * (height - padT - padB);

  const points = energies.map((e, i) => `${toX(i)},${toY(e)}`).join(" ");
  const targetY = toY(targetEnergy);
  const chemAccHa = chemicalAccuracyMHa * 0.001;
  const targetPlusChemY = toY(targetEnergy + chemAccHa);

  const finalE = energies[energies.length - 1];
  const deltaE = Math.abs(finalE - targetEnergy);
  const isChemAccurate = deltaE <= chemAccHa;

  return (
    <div
      style={{
        border: "1px solid #334155",
        borderRadius: 8,
        background: "#111827",
        padding: 16,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 10,
          flexWrap: "wrap",
          gap: 6,
        }}
      >
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#f8fafc" }}>
            VQE Optimizer Convergence · {optimizerName}
          </div>
          <div style={{ fontSize: 11, color: "#94a3b8" }}>
            Final Energy = {finalE.toFixed(6)} Ha · ΔE = {(deltaE * 1000).toFixed(3)} mHa
          </div>
        </div>
        <span
          style={{
            padding: "3px 8px",
            borderRadius: 4,
            fontSize: 11,
            fontWeight: 600,
            background: isChemAccurate ? "rgba(16, 185, 129, 0.15)" : "rgba(245, 158, 11, 0.15)",
            color: isChemAccurate ? "#34d399" : "#fbbf24",
            border: `1px solid ${isChemAccurate ? "#059669" : "#d97706"}`,
          }}
        >
          {isChemAccurate ? "Chemical Accuracy Reached (≤ 1 kcal/mol)" : "Approaching Ground State"}
        </span>
      </div>

      <div style={{ width: "100%", overflowX: "auto" }}>
        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height: "auto", display: "block" }}>
          {/* Target Energy Line */}
          <line x1={padL} y1={targetY} x2={width - padR} y2={targetY} stroke="#38bdf8" strokeWidth="1.5" />
          <text x={width - padR} y={targetY - 4} fill="#38bdf8" fontSize="10" textAnchor="end">
            FCI Exact: {targetEnergy.toFixed(6)} Ha
          </text>

          {/* Chemical Accuracy Window (shaded region) */}
          <rect
            x={padL}
            y={targetPlusChemY}
            width={width - padL - padR}
            height={Math.max(2, targetY - targetPlusChemY)}
            fill="rgba(16, 185, 129, 0.12)"
          />
          <line
            x1={padL}
            y1={targetPlusChemY}
            x2={width - padR}
            y2={targetPlusChemY}
            stroke="#10b981"
            strokeWidth="1"
            strokeDasharray="3,3"
          />

          {/* Energy Trajectory Line */}
          <polyline points={points} fill="none" stroke="#f59e0b" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />

          {/* Points */}
          {energies.map((e, i) => (
            <circle key={`vqe-pt-${i}`} cx={toX(i)} cy={toY(e)} r="3" fill="#f59e0b" stroke="#78350f" strokeWidth="1" />
          ))}

          {/* Axes */}
          <line x1={padL} y1={height - padB} x2={width - padR} y2={height - padB} stroke="#475569" strokeWidth="1" />
          <text x={width / 2} y={height - 6} fill="#94a3b8" fontSize="11" textAnchor="middle">
            Optimization Iteration Step
          </text>
          <text
            x={18}
            y={height / 2}
            fill="#94a3b8"
            fontSize="11"
            textAnchor="middle"
            transform={`rotate(-90, 18, ${height / 2})`}
          >
            Energy ⟨H⟩ (Hartree)
          </text>
        </svg>
      </div>
    </div>
  );
}
