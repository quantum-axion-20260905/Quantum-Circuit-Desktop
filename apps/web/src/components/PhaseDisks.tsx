"use client";

import React, { useState } from "react";
import type { AmplitudeInfo } from "../lib/quantumSim";

interface PhaseDisksProps {
  amplitudes: AmplitudeInfo[];
  nQubits: number;
}

// Convert complex phase [-pi, pi] to HSL color string
function phaseToHsl(phase: number): string {
  // 0 rad (real positive) -> 195 (Cyan)
  // pi/2 rad (+i) -> 120 (Green)
  // pi rad (-1) -> 45 (Orange/Gold)
  // -pi/2 rad (-i) -> 280 (Purple)
  const deg = ((phase * 180) / Math.PI + 360) % 360;
  return `hsl(${deg.toFixed(0)}, 85%, 50%)`;
}

export function PhaseDisks({ amplitudes, nQubits }: PhaseDisksProps) {
  const [filterNonZero, setFilterNonZero] = useState<boolean>(false);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  // If there are many qubits (e.g. 5+ qubits = 32+ states), allow showing only states with prob > 0.001
  const displayedAmps = filterNonZero
    ? amplitudes.filter((a) => a.prob > 0.0005)
    : amplitudes.slice(0, 64); // Show up to 64 states max for performance

  const nonZeroCount = amplitudes.filter((a) => a.prob > 0.001).length;

  return (
    <div
      style={{
        background: "#ffffff",
        border: "1px solid #e2e8f0",
        borderRadius: 12,
        padding: 16,
        boxShadow: "0 4px 6px -1px rgba(0,0,0,0.05)",
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      {/* Header & Controls */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 18 }}>⏱️</span>
          <div>
            <div style={{ fontWeight: 700, fontSize: 14, color: "#0f172a" }}>
              Kvant Fazalari Doiralari (Inline Phase Disks / Phasors)
            </div>
            <div style={{ fontSize: 11, color: "#64748b" }}>
              Har bir bazis holatining ehtimolligi (doira maydoni) va fazasi (e^{"iϕ"} soat mili)
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {amplitudes.length > 8 && (
            <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "#475569", cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={filterNonZero}
                onChange={(e) => setFilterNonZero(e.target.checked)}
                style={{ cursor: "pointer" }}
              />
              Faqat faol holatlar (P &gt; 0)
            </label>
          )}
          <span style={{ fontSize: 11, background: "#f1f5f9", padding: "3px 8px", borderRadius: 6, color: "#334155", fontWeight: 600 }}>
            {nonZeroCount} ta superpozitsiya bazisi ({nQubits} qubit)
          </span>
        </div>
      </div>

      {/* Phase Color Wheel Legend */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, fontSize: 11, color: "#64748b", background: "#f8fafc", padding: "6px 12px", borderRadius: 8, border: "1px solid #f1f5f9", flexWrap: "wrap" }}>
        <span style={{ fontWeight: 600, color: "#334155" }}>Faza shkalasi (Arg):</span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
          <span style={{ width: 10, height: 10, borderRadius: "50%", background: phaseToHsl(0), display: "inline-block" }} />
          0 rad (+1)
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
          <span style={{ width: 10, height: 10, borderRadius: "50%", background: phaseToHsl(Math.PI / 2), display: "inline-block" }} />
          +π/2 (+i)
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
          <span style={{ width: 10, height: 10, borderRadius: "50%", background: phaseToHsl(Math.PI), display: "inline-block" }} />
          ±π (-1)
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
          <span style={{ width: 10, height: 10, borderRadius: "50%", background: phaseToHsl(-Math.PI / 2), display: "inline-block" }} />
          -π/2 (-i)
        </span>
      </div>

      {/* Grid of Phase Disks */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(130px, 1fr))",
          gap: 12,
          maxHeight: 340,
          overflowY: "auto",
          padding: 2,
        }}
      >
        {displayedAmps.map((amp) => {
          const outerR = 38;
          const cx = 46;
          const cy = 46;
          // Radius proportional to amplitude magnitude (sqrt of probability)
          const mag = Math.sqrt(amp.prob);
          const fillR = Math.max(0, outerR * mag);

          // Phase angle in standard Cartesian: 0 rad -> right (X+), counter-clockwise
          // In SVG, Y is down, so needle coordinates:
          // X = cx + r * cos(phase)
          // Y = cy - r * sin(phase)
          const needleLen = Math.max(12, fillR);
          const nx = cx + needleLen * Math.cos(amp.phase);
          const ny = cy - needleLen * Math.sin(amp.phase);

          const color = phaseToHsl(amp.phase);
          const phaseDeg = ((amp.phase * 180) / Math.PI).toFixed(0);
          const isHovered = hoveredIdx === amp.index;
          const isSignificant = amp.prob > 0.001;

          return (
            <div
              key={amp.index}
              onMouseEnter={() => setHoveredIdx(amp.index)}
              onMouseLeave={() => setHoveredIdx(null)}
              style={{
                background: isHovered ? "#eff6ff" : isSignificant ? "#ffffff" : "#f8fafc",
                border: isHovered ? "2px solid #2563eb" : isSignificant ? "1px solid #cbd5e1" : "1px dashed #e2e8f0",
                borderRadius: 10,
                padding: "8px 6px",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                transition: "all 120ms ease",
                opacity: isSignificant ? 1 : 0.45,
              }}
            >
              {/* Dial SVG */}
              <svg width={92} height={92} style={{ overflow: "visible" }}>
                {/* Outer ring */}
                <circle cx={cx} cy={cy} r={outerR} fill="#f8fafc" stroke="#cbd5e1" strokeWidth="1.2" />

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
                      stroke="#cbd5e1"
                      strokeWidth={t % 3 === 0 ? "1.5" : "0.75"}
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
                    fillOpacity={0.35}
                    stroke={color}
                    strokeWidth="1.5"
                    style={{ transition: "r 150ms ease, fill 150ms ease" }}
                  />
                )}

                {/* Center Pivot */}
                <circle cx={cx} cy={cy} r="2.5" fill="#475569" />

                {/* Phase Pointer / Clock Hand */}
                {amp.prob > 0.0001 && (
                  <>
                    <line
                      x1={cx}
                      y1={cy}
                      x2={nx}
                      y2={ny}
                      stroke={color}
                      strokeWidth="2.5"
                      strokeLinecap="round"
                    />
                    <circle cx={nx} cy={ny} r="3" fill={color} />
                  </>
                )}
              </svg>

              {/* Basis Ket Label */}
              <div
                style={{
                  fontWeight: 700,
                  fontSize: 12,
                  color: isSignificant ? "#0f172a" : "#64748b",
                  marginTop: 4,
                }}
              >
                |{amp.bitstring}⟩
              </div>

              {/* Probability Readout */}
              <div
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  color: isSignificant ? "#2563eb" : "#94a3b8",
                }}
              >
                {(amp.prob * 100).toFixed(1)}%
              </div>

              {/* Phase Readout */}
              {isSignificant && (
                <div style={{ fontSize: 10, color: "#64748b", marginTop: 2 }}>
                  {phaseDeg}° ({((amp.phase / Math.PI) >= 0 ? "+" : "") + (amp.phase / Math.PI).toFixed(2)}π)
                </div>
              )}
            </div>
          );
        })}
      </div>

      {displayedAmps.length === 0 && (
        <div style={{ padding: 20, textAlign: "center", color: "#94a3b8", fontSize: 13 }}>
          Tanlangan filtr bo‘yicha holatlar topilmadi.
        </div>
      )}
    </div>
  );
}
