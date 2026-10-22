"use client";

import React, { useState } from "react";
import type { AmplitudeInfo } from "../lib/quantumSim";

interface PhaseDisksProps {
  amplitudes: AmplitudeInfo[];
  nQubits: number;
}

const cardStyle: React.CSSProperties = {
  background: "#111827",
  border: "1px solid #334155",
  borderRadius: 8,
  padding: 16,
  color: "#f8fafc",
  fontFamily: "inherit",
  display: "flex",
  flexDirection: "column",
  gap: 12,
};

const labelStyle: React.CSSProperties = {
  fontSize: 11,
  color: "#94a3b8",
  fontWeight: 500,
  textTransform: "uppercase",
  letterSpacing: "0.04em",
};

// Convert complex phase [-pi, pi] to scientific HSL color string
function phaseToHsl(phase: number): string {
  // 0 rad (+1) -> Sky Cyan (195)
  // +pi/2 rad (+i) -> Spring Green (140)
  // pi rad (-1) -> Amber (38)
  // -pi/2 rad (-i) -> Purple/Violet (280)
  const deg = ((phase * 180) / Math.PI + 360) % 360;
  return `hsl(${deg.toFixed(0)}, 85%, 55%)`;
}

export function PhaseDisks({ amplitudes, nQubits }: PhaseDisksProps) {
  const [filterNonZero, setFilterNonZero] = useState<boolean>(false);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  // If multi-qubit (e.g. 5+ qubits = 32+ states), allow filtering
  const displayedAmps = filterNonZero
    ? amplitudes.filter((a) => a.prob > 0.0005)
    : amplitudes.slice(0, 64); // Show up to 64 states max for smooth rendering

  const nonZeroCount = amplitudes.filter((a) => a.prob > 0.0005).length;
  const totalProbability = amplitudes.reduce((sum, a) => sum + a.prob, 0);

  return (
    <div style={cardStyle}>
      {/* Header and Controls */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10, borderBottom: "1px solid #1e293b", paddingBottom: 10 }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: 14, color: "#f8fafc" }}>
            Complex Phase Disks · Phasor Dial Array
          </div>
          <div style={{ fontSize: 11, color: "#94a3b8" }}>
            State probability P_k = |c_k|² (disk area) and complex quantum phase factor c_k = |c_k|e^(iϕ_k) (needle)
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {amplitudes.length > 4 && (
            <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "#94a3b8", cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={filterNonZero}
                onChange={(e) => setFilterNonZero(e.target.checked)}
                style={{ cursor: "pointer" }}
              />
              Filter P_k &gt; 0
            </label>
          )}

          <div style={{ padding: "4px 8px", background: "#0b0f19", border: "1px solid #334155", borderRadius: 4, fontSize: 11, fontFamily: "monospace", color: "#38bdf8" }}>
            {nonZeroCount} active / {1 << nQubits} basis states · ΣP = {totalProbability.toFixed(4)}
          </div>
        </div>
      </div>

      {/* Complex Phase Legend */}
      <div style={{ display: "flex", alignItems: "center", gap: 14, fontSize: 11, color: "#94a3b8", background: "#0b0f19", padding: "6px 12px", borderRadius: 4, border: "1px solid #1e293b", flexWrap: "wrap" }}>
        <span style={labelStyle}>Phase Arg(c):</span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: phaseToHsl(0), display: "inline-block" }} />
          0 rad (+1 real)
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: phaseToHsl(Math.PI / 2), display: "inline-block" }} />
          +π/2 (+i imag)
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: phaseToHsl(Math.PI), display: "inline-block" }} />
          ±π (-1 real)
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: phaseToHsl(-Math.PI / 2), display: "inline-block" }} />
          -π/2 (-i imag)
        </span>
      </div>

      {/* Grid of Phase Disks */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(136px, 1fr))",
          gap: 10,
          maxHeight: 360,
          overflowY: "auto",
          padding: 2,
        }}
      >
        {displayedAmps.map((amp) => {
          const outerR = 34;
          const cx = 42;
          const cy = 42;
          // Radius proportional to amplitude magnitude (sqrt of probability)
          const mag = Math.sqrt(amp.prob);
          const fillR = Math.max(0, outerR * mag);

          // Phasor needle: X = cx + r * cos(phase), Y = cy - r * sin(phase)
          const needleLen = Math.max(10, fillR);
          const nx = cx + needleLen * Math.cos(amp.phase);
          const ny = cy - needleLen * Math.sin(amp.phase);

          const color = phaseToHsl(amp.phase);
          const phaseDeg = ((((amp.phase * 180) / Math.PI) + 360) % 360).toFixed(1);
          const isHovered = hoveredIdx === amp.index;
          const isSignificant = amp.prob > 0.0005;

          return (
            <div
              key={amp.index}
              onMouseEnter={() => setHoveredIdx(amp.index)}
              onMouseLeave={() => setHoveredIdx(null)}
              style={{
                background: isHovered ? "#1e293b" : "#0b0f19",
                border: isHovered ? "1px solid #38bdf8" : isSignificant ? "1px solid #334155" : "1px dashed #1e293b",
                borderRadius: 6,
                padding: "8px 6px",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                transition: "all 120ms ease",
                opacity: isSignificant ? 1 : 0.35,
              }}
            >
              {/* Phasor Dial SVG */}
              <svg width={84} height={84} style={{ overflow: "visible" }}>
                {/* Outer ring */}
                <circle cx={cx} cy={cy} r={outerR} fill="#080d1a" stroke="#334155" strokeWidth="1" />

                {/* Complex plane crosshairs (Re horizontal, Im vertical) */}
                <line x1={cx - outerR} y1={cy} x2={cx + outerR} y2={cy} stroke="#1e293b" strokeWidth="0.8" strokeDasharray="2 2" />
                <line x1={cx} y1={cy - outerR} x2={cx} y2={cy + outerR} stroke="#1e293b" strokeWidth="0.8" strokeDasharray="2 2" />

                {/* 12 clock ticks */}
                {Array.from({ length: 12 }).map((_, t) => {
                  const tickAngle = (t * Math.PI) / 6;
                  const x1 = cx + (outerR - 3) * Math.cos(tickAngle);
                  const y1 = cy - (outerR - 3) * Math.sin(tickAngle);
                  const x2 = cx + outerR * Math.cos(tickAngle);
                  const y2 = cy - outerR * Math.sin(tickAngle);
                  return (
                    <line
                      key={t}
                      x1={x1}
                      y1={y1}
                      x2={x2}
                      y2={y2}
                      stroke="#475569"
                      strokeWidth={t % 3 === 0 ? "1.2" : "0.6"}
                    />
                  );
                })}

                {/* Probability filled disk */}
                {fillR > 0.5 && (
                  <circle
                    cx={cx}
                    cy={cy}
                    r={fillR}
                    fill={color}
                    fillOpacity={0.3}
                    stroke={color}
                    strokeWidth="1.2"
                    style={{ transition: "r 120ms ease, fill 120ms ease" }}
                  />
                )}

                {/* Center Pivot */}
                <circle cx={cx} cy={cy} r="2" fill="#64748b" />

                {/* Phase Pointer / Phasor Hand */}
                {amp.prob > 0.0001 && (
                  <>
                    <line
                      x1={cx}
                      y1={cy}
                      x2={nx}
                      y2={ny}
                      stroke={color}
                      strokeWidth="2"
                      strokeLinecap="round"
                    />
                    <circle cx={nx} cy={ny} r="2.5" fill={color} />
                  </>
                )}
              </svg>

              {/* Basis Ket Label */}
              <div
                style={{
                  fontWeight: 700,
                  fontSize: 12,
                  fontFamily: "monospace",
                  color: isSignificant ? "#f8fafc" : "#64748b",
                  marginTop: 4,
                }}
              >
                |{amp.bitstring}⟩
              </div>

              {/* Probability Readout */}
              <div
                style={{
                  fontSize: 11,
                  fontFamily: "monospace",
                  fontWeight: 600,
                  color: isSignificant ? "#38bdf8" : "#64748b",
                }}
              >
                {(amp.prob * 100).toFixed(2)}%
              </div>

              {/* Cartesian & Polar Complex Readout */}
              {isSignificant && (
                <div style={{ fontSize: 9, fontFamily: "monospace", color: "#94a3b8", textAlign: "center", marginTop: 2 }}>
                  <div>{amp.amplitude.re >= 0 ? "+" : ""}{amp.amplitude.re.toFixed(3)}{amp.amplitude.im >= 0 ? "+" : ""}{amp.amplitude.im.toFixed(3)}i</div>
                  <div>{phaseDeg}° ({(amp.phase / Math.PI).toFixed(2)}π)</div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {displayedAmps.length === 0 && (
        <div style={{ padding: 18, textAlign: "center", color: "#64748b", fontSize: 12, fontFamily: "monospace" }}>
          No basis states matching filter criterion (Pk &gt; 0.0005).
        </div>
      )}
    </div>
  );
}
