"use client";

import React, { useState } from "react";

// ============================================================================
// 1. Surface Code Stabilizer Plaquette Grid (d=3, 17 qubits)
// ============================================================================

export type PlaquetteType = "X" | "Z";

export type SyndromeCheck = {
  id: string;
  type: PlaquetteType;
  x: number;
  y: number;
  syndrome: 0 | 1; // 0 = no error (+1 eigenvalue), 1 = detected error (-1 eigenvalue)
  dataQubits: number[];
};

export type QubitDataNode = {
  id: number;
  x: number;
  y: number;
  hasError: boolean;
  errorType?: "X" | "Z" | "Y";
};

export type SurfaceCodeGridProps = {
  dataQubits: QubitDataNode[];
  syndromes: SyndromeCheck[];
  onToggleQubitError?: (qubitId: number) => void;
  title?: string;
  distance?: number;
};

export function SurfaceCodePlaquetteGrid({
  dataQubits,
  syndromes,
  onToggleQubitError,
  title = "Rotated Surface Code Lattice (d=3)",
  distance = 3,
}: SurfaceCodeGridProps) {
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  const width = 500;
  const height = 400;
  const cx = width / 2;
  const cy = height / 2;
  const scale = 75;

  const toScreen = (x: number, y: number) => ({
    sx: cx + x * scale,
    sy: cy + y * scale,
  });

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
            {title} · Code Distance d = {distance}
          </div>
          <div style={{ fontSize: 11, color: "#94a3b8" }}>
            Click data qubits to inject Pauli errors · Plaquettes detect local syndrome defects
          </div>
        </div>

        <div style={{ display: "flex", gap: 12, fontSize: 11 }}>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
            <span style={{ width: 12, height: 12, background: "rgba(59, 130, 246, 0.3)", border: "1px solid #3b82f6", borderRadius: 2 }} />
            Z-type Plaquette (Vertex)
          </span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
            <span style={{ width: 12, height: 12, background: "rgba(239, 68, 68, 0.3)", border: "1px solid #ef4444", borderRadius: 2 }} />
            X-type Plaquette (Face)
          </span>
          <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
            <span style={{ width: 10, height: 10, borderRadius: "50%", background: "#f8fafc", border: "1px solid #94a3b8" }} />
            Data Qubit
          </span>
        </div>
      </div>

      <div style={{ width: "100%", overflowX: "auto" }}>
        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height: "auto", display: "block" }}>
          {/* Stabilizer Plaquettes (Polygons) */}
          {syndromes.map((syn) => {
            const p = toScreen(syn.x, syn.y);
            const size = scale * 0.72;
            const isError = syn.syndrome === 1;

            return (
              <g key={syn.id}>
                {/* Rotated diamond plaquette */}
                <polygon
                  points={`${p.sx},${p.sy - size} ${p.sx + size},${p.sy} ${p.sx},${p.sy + size} ${p.sx - size},${p.sy}`}
                  fill={
                    isError
                      ? syn.type === "Z"
                        ? "rgba(59, 130, 246, 0.65)"
                        : "rgba(239, 68, 68, 0.65)"
                      : syn.type === "Z"
                      ? "rgba(59, 130, 246, 0.18)"
                      : "rgba(239, 68, 68, 0.18)"
                  }
                  stroke={syn.type === "Z" ? "#3b82f6" : "#ef4444"}
                  strokeWidth={isError ? 2.5 : 1.2}
                />
                {/* Syndrome defect indicator */}
                <text
                  x={p.sx}
                  y={p.sy + 4}
                  fill={isError ? "#ffffff" : syn.type === "Z" ? "#93c5fd" : "#fca5a5"}
                  fontSize={isError ? 13 : 10}
                  fontWeight={isError ? "bold" : "normal"}
                  textAnchor="middle"
                >
                  {isError ? "⚡ -1" : syn.type}
                </text>
              </g>
            );
          })}

          {/* Connectors / Grid Lines */}
          {syndromes.map((syn) => {
            const sp = toScreen(syn.x, syn.y);
            return syn.dataQubits.map((dqId) => {
              const dq = dataQubits.find((q) => q.id === dqId);
              if (!dq) return null;
              const dp = toScreen(dq.x, dq.y);
              return (
                <line
                  key={`${syn.id}-dq-${dqId}`}
                  x1={sp.sx}
                  y1={sp.sy}
                  x2={dp.sx}
                  y2={dp.sy}
                  stroke="#334155"
                  strokeWidth={1}
                  strokeDasharray="2,2"
                />
              );
            });
          })}

          {/* Data Qubits (Circles) */}
          {dataQubits.map((q) => {
            const p = toScreen(q.x, q.y);
            const isHovered = hoveredNode === `dq-${q.id}`;

            return (
              <g
                key={`dq-${q.id}`}
                style={{ cursor: "pointer" }}
                onClick={() => onToggleQubitError?.(q.id)}
                onMouseEnter={() => setHoveredNode(`dq-${q.id}`)}
                onMouseLeave={() => setHoveredNode(null)}
              >
                {/* Glow ring if error injected */}
                {q.hasError && (
                  <circle cx={p.sx} cy={p.sy} r={18} fill="rgba(245, 158, 11, 0.25)" stroke="#f59e0b" strokeWidth={1.5} />
                )}
                {/* Main qubit sphere */}
                <circle
                  cx={p.sx}
                  cy={p.sy}
                  r={12}
                  fill={q.hasError ? "#f59e0b" : isHovered ? "#38bdf8" : "#f8fafc"}
                  stroke="#0f172a"
                  strokeWidth={2}
                />
                {/* Qubit index */}
                <text
                  x={p.sx}
                  y={p.sy + 3.5}
                  fill={q.hasError ? "#ffffff" : "#0f172a"}
                  fontSize={9}
                  fontWeight="bold"
                  textAnchor="middle"
                >
                  {q.hasError ? q.errorType ?? "E" : `D${q.id}`}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div
        style={{
          marginTop: 10,
          background: "#090d16",
          padding: "8px 12px",
          borderRadius: 4,
          border: "1px solid #1e293b",
          fontSize: 11,
          color: "#94a3b8",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          Active Errors:{" "}
          <strong style={{ color: dataQubits.some((q) => q.hasError) ? "#f59e0b" : "#34d399" }}>
            {dataQubits.filter((q) => q.hasError).length} physical qubits
          </strong>
          {" · "}
          Detected Syndromes:{" "}
          <strong style={{ color: syndromes.some((s) => s.syndrome === 1) ? "#ef4444" : "#34d399" }}>
            {syndromes.filter((s) => s.syndrome === 1).length} violations
          </strong>
        </div>
        <span>Click qubit to inject Pauli X/Z error</span>
      </div>
    </div>
  );
}

// ============================================================================
// 2. QEC Logical Error Threshold Curve (PL vs p_phys)
// ============================================================================

export type ThresholdCurveProps = {
  thresholdP?: number; // e.g. 0.01 (1%)
  currentNoiseP: number;
};

export function LogicalThresholdCurve({
  thresholdP = 0.0105,
  currentNoiseP,
}: ThresholdCurveProps) {
  const width = 540;
  const height = 260;
  const padL = 60;
  const padR = 24;
  const padT = 24;
  const padB = 36;

  // Log-log domain: p_phys from 1e-4 to 0.04
  const minP = 0.0001;
  const maxP = 0.04;
  const minPL = 1e-7;
  const maxPL = 0.2;

  const toX = (p: number) => {
    const logMin = Math.log10(minP);
    const logMax = Math.log10(maxP);
    const logVal = Math.log10(Math.max(minP, Math.min(maxP, p)));
    return padL + ((logVal - logMin) / (logMax - logMin)) * (width - padL - padR);
  };

  const toY = (pl: number) => {
    const logMin = Math.log10(minPL);
    const logMax = Math.log10(maxPL);
    const logVal = Math.log10(Math.max(minPL, Math.min(maxPL, pl)));
    return height - padB - ((logVal - logMin) / (logMax - logMin)) * (height - padT - padB);
  };

  // Generate curves for d=3, d=5, d=7: P_L ~ A * (p / p_th)^((d+1)/2)
  const pValues: number[] = [];
  for (let p = minP; p <= maxP; p *= 1.25) {
    pValues.push(p);
  }

  const curveD3 = pValues.map((p) => {
    const pl = Math.min(0.2, 0.03 * Math.pow(p / thresholdP, 2));
    return `${toX(p)},${toY(pl)}`;
  }).join(" ");

  const curveD5 = pValues.map((p) => {
    const pl = Math.min(0.2, 0.03 * Math.pow(p / thresholdP, 3));
    return `${toX(p)},${toY(pl)}`;
  }).join(" ");

  const curveD7 = pValues.map((p) => {
    const pl = Math.min(0.2, 0.03 * Math.pow(p / thresholdP, 4));
    return `${toX(p)},${toY(pl)}`;
  }).join(" ");

  const curX = toX(currentNoiseP);
  const thX = toX(thresholdP);
  const belowThreshold = currentNoiseP < thresholdP;

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
            Fault-Tolerant Error Threshold Curve (P_L vs p_phys)
          </div>
          <div style={{ fontSize: 11, color: "#94a3b8" }}>
            Threshold p_th ≈ {(thresholdP * 100).toFixed(2)}% · Below threshold, larger codes suppress logical errors exponentially
          </div>
        </div>

        <div style={{ display: "flex", gap: 10, fontSize: 11 }}>
          <span style={{ color: "#38bdf8", display: "inline-flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 12, height: 2, background: "#38bdf8", display: "inline-block" }} /> d=3
          </span>
          <span style={{ color: "#34d399", display: "inline-flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 12, height: 2, background: "#34d399", display: "inline-block" }} /> d=5
          </span>
          <span style={{ color: "#a855f7", display: "inline-flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 12, height: 2, background: "#a855f7", display: "inline-block" }} /> d=7
          </span>
        </div>
      </div>

      <div style={{ width: "100%", overflowX: "auto" }}>
        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height: "auto", display: "block" }}>
          {/* Threshold Crossing Line */}
          <line x1={thX} y1={padT} x2={thX} y2={height - padB} stroke="#ef4444" strokeWidth={1.5} strokeDasharray="3,3" />
          <text x={thX + 4} y={padT + 12} fill="#ef4444" fontSize="10" fontWeight="bold">
            p_th = {(thresholdP * 100).toFixed(2)}%
          </text>

          {/* Curves */}
          <polyline points={curveD3} fill="none" stroke="#38bdf8" strokeWidth={2} />
          <polyline points={curveD5} fill="none" stroke="#34d399" strokeWidth={2} />
          <polyline points={curveD7} fill="none" stroke="#a855f7" strokeWidth={2} />

          {/* Operating Point Line */}
          <line x1={curX} y1={padT} x2={curX} y2={height - padB} stroke="#f59e0b" strokeWidth={2} strokeDasharray="2,2" />
          <text x={curX} y={height - padB - 8} fill="#f59e0b" fontSize="10" textAnchor="middle" fontWeight="bold">
            p = {(currentNoiseP * 100).toFixed(2)}%
          </text>

          {/* Axes */}
          <line x1={padL} y1={height - padB} x2={width - padR} y2={height - padB} stroke="#475569" strokeWidth={1} />
          <text x={width / 2} y={height - 8} fill="#94a3b8" fontSize="11" textAnchor="middle">
            Physical Gate Error Rate p_phys
          </text>
          <text
            x={18}
            y={height / 2}
            fill="#94a3b8"
            fontSize="11"
            textAnchor="middle"
            transform={`rotate(-90, 18, ${height / 2})`}
          >
            Logical Error Rate P_L
          </text>
        </svg>
      </div>

      <div
        style={{
          marginTop: 8,
          background: "#090d16",
          padding: "6px 12px",
          borderRadius: 4,
          fontSize: 11,
          display: "flex",
          justifyContent: "space-between",
          color: "#94a3b8",
        }}
      >
        <span>
          Operating Regime:{" "}
          <strong style={{ color: belowThreshold ? "#34d399" : "#ef4444" }}>
            {belowThreshold ? "Fault-Tolerant (Exponential Suppression)" : "Above Threshold (Noise Dominates)"}
          </strong>
        </span>
        <span>Asymptotics: P_L ∝ (p/p_th)^((d+1)/2)</span>
      </div>
    </div>
  );
}

// ============================================================================
// 3. Quantum Process Matrix Heatmap (Chi Matrix)
// ============================================================================

export type ProcessMatrixProps = {
  chi: number[][]; // 4x4 matrix for 1-qubit process {I, X, Y, Z}
  labels?: string[];
  fidelity?: number;
};

export function ProcessTomographyHeatmap({
  chi,
  labels = ["I", "X", "Y", "Z"],
  fidelity = 0.985,
}: ProcessMatrixProps) {
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
        }}
      >
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#f8fafc" }}>
            Quantum Process Tomography · Chi Matrix (χ)
          </div>
          <div style={{ fontSize: 11, color: "#94a3b8" }}>
            Process Fidelity F_pro = {(fidelity * 100).toFixed(2)}\%$ · Pauli basis representation
          </div>
        </div>
        <div style={{ fontSize: 12, fontWeight: 700, color: fidelity > 0.99 ? "#34d399" : "#38bdf8" }}>
          E(ρ) = Σ_mn χ_mn E_m ρ E_n†
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: `40px repeat(${labels.length}, 1fr)`, gap: 4, textAlign: "center" }}>
        <div />
        {labels.map((l) => (
          <div key={`header-${l}`} style={{ fontSize: 11, fontWeight: 700, color: "#94a3b8" }}>
            {l}
          </div>
        ))}

        {chi.map((row, r) => (
          <React.Fragment key={`row-${r}`}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#94a3b8", display: "flex", alignItems: "center", justifyContent: "center" }}>
              {labels[r]}
            </div>
            {row.map((val, c) => {
              const intensity = Math.min(1, Math.max(0, val));
              const isDiagonal = r === c;

              return (
                <div
                  key={`cell-${r}-${c}`}
                  style={{
                    padding: "8px 4px",
                    borderRadius: 4,
                    background: isDiagonal
                      ? `rgba(56, 189, 248, ${0.1 + intensity * 0.85})`
                      : `rgba(239, 68, 68, ${intensity * 0.85})`,
                    border: "1px solid #1e293b",
                    fontSize: 10,
                    fontFamily: "monospace",
                    color: intensity > 0.5 ? "#ffffff" : "#94a3b8",
                  }}
                >
                  {val.toFixed(3)}
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
