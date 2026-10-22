"use client";

import React, { useState, useRef, useCallback, useEffect } from "react";
import type { BlochVector } from "../lib/quantumSim";

interface BlochSphereProps {
  vector: BlochVector;
  qubitIndex: number;
  nQubits: number;
  onSelectQubit?: (q: number) => void;
  allVectors?: BlochVector[];
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
  gap: 14,
};

const subCardStyle: React.CSSProperties = {
  background: "#0b0f19",
  border: "1px solid #1e293b",
  borderRadius: 6,
  padding: 10,
};

const labelStyle: React.CSSProperties = {
  fontSize: 11,
  color: "#94a3b8",
  fontWeight: 500,
  textTransform: "uppercase",
  letterSpacing: "0.04em",
};

const monoValStyle: React.CSSProperties = {
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
  fontSize: 13,
  fontWeight: 600,
  color: "#f8fafc",
};

export function BlochSphere({
  vector,
  qubitIndex,
  nQubits,
  onSelectQubit,
  allVectors,
}: BlochSphereProps) {
  // Camera angles: azimuth (rotation around Z) and elevation (tilt)
  const [azimuth, setAzimuth] = useState<number>(35 * (Math.PI / 180));
  const [elevation, setElevation] = useState<number>(20 * (Math.PI / 180));
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const dragStart = useRef<{ x: number; y: number; az: number; el: number }>({
    x: 0,
    y: 0,
    az: 0,
    el: 0,
  });

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      setIsDragging(true);
      dragStart.current = {
        x: e.clientX,
        y: e.clientY,
        az: azimuth,
        el: elevation,
      };
    },
    [azimuth, elevation]
  );

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      const dx = e.clientX - dragStart.current.x;
      const dy = e.clientY - dragStart.current.y;
      const newAz = dragStart.current.az + dx * 0.012;
      const newEl = Math.max(
        -Math.PI / 2.05,
        Math.min(Math.PI / 2.05, dragStart.current.el + dy * 0.012)
      );
      setAzimuth(newAz);
      setElevation(newEl);
    };

    const handleMouseUp = () => {
      if (isDragging) setIsDragging(false);
    };

    if (isDragging) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
    }
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging]);

  const setViewPreset = (azDeg: number, elDeg: number) => {
    setAzimuth(azDeg * (Math.PI / 180));
    setElevation(elDeg * (Math.PI / 180));
  };

  // 3D Projection parameters
  const CX = 145;
  const CY = 145;
  const R = 100; // SVG sphere radius in px

  // Project 3D (x, y, z) in [-1, 1] to 2D SVG coords
  const project = useCallback(
    (x: number, y: number, z: number) => {
      // 1. Azimuth rotation around Z axis
      const cosA = Math.cos(azimuth);
      const sinA = Math.sin(azimuth);
      const x1 = x * cosA - y * sinA;
      const y1 = x * sinA + y * cosA;
      const z1 = z;

      // 2. Elevation rotation around X axis
      const cosE = Math.cos(elevation);
      const sinE = Math.sin(elevation);
      const x2 = x1;
      const y2 = y1 * cosE - z1 * sinE;
      const z2 = y1 * sinE + z1 * cosE;

      return {
        sx: CX + R * x2,
        sy: CY - R * z2,
        depth: y2, // > 0 is closer to viewer
      };
    },
    [azimuth, elevation]
  );

  // Compute circles for wireframe
  const renderCircle = (plane: "xy" | "xz" | "yz", numPts = 64, zOffset = 0, radius = 1) => {
    const pts: { sx: number; sy: number; depth: number }[] = [];
    for (let i = 0; i <= numPts; i++) {
      const angle = (i / numPts) * 2 * Math.PI;
      let x = 0;
      let y = 0;
      let z = zOffset;
      if (plane === "xy") {
        x = radius * Math.cos(angle);
        y = radius * Math.sin(angle);
      } else if (plane === "xz") {
        x = radius * Math.cos(angle);
        z = radius * Math.sin(angle);
      } else {
        y = radius * Math.cos(angle);
        z = radius * Math.sin(angle);
      }
      pts.push(project(x, y, z));
    }
    return pts.map((p, idx) => `${idx === 0 ? "M" : "L"} ${p.sx.toFixed(1)} ${p.sy.toFixed(1)}`).join(" ");
  };

  // Coordinate axes
  const origin = project(0, 0, 0);
  const axisZPos = project(0, 0, 1.28);
  const axisZNeg = project(0, 0, -1.28);
  const axisXPos = project(1.28, 0, 0);
  const axisXNeg = project(-1.28, 0, 0);
  const axisYPos = project(0, 1.28, 0);
  const axisYNeg = project(0, -1.28, 0);

  // State vector values
  const { x, y, z, r, theta, phi, prob0, prob1 } = vector;
  const tip = project(x, y, z);
  const tipProjXY = project(x, y, 0);
  const tipProjZ = project(0, 0, z);

  // Scientific metrics
  const thetaDeg = ((theta * 180) / Math.PI).toFixed(2);
  const phiDeg = ((((phi * 180) / Math.PI) + 360) % 360).toFixed(2);
  const purity = (1 + r * r) / 2;
  const isPure = r >= 0.98;
  const isMixed = r < 0.95;

  // Von Neumann entropy S = -p+ log2(p+) - p- log2(p-) where p± = (1 ± r)/2
  const pPlus = (1 + Math.min(1, r)) / 2;
  const pMinus = (1 - Math.min(1, r)) / 2;
  const sVN =
    pMinus > 1e-12 && pPlus > 1e-12
      ? -(pPlus * Math.log2(pPlus) + pMinus * Math.log2(pMinus))
      : 0;

  // Density matrix elements
  const rho00 = (1 + z) / 2;
  const rho11 = (1 - z) / 2;
  const rho01Re = x / 2;
  const rho01Im = -y / 2;

  return (
    <div style={cardStyle}>
      {/* Header and Qubit Selector */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10, borderBottom: "1px solid #1e293b", paddingBottom: 10 }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: 14, color: "#f8fafc" }}>
            Bloch Sphere Instrument · S² Quantum State Space
          </div>
          <div style={{ fontSize: 11, color: "#94a3b8" }}>
            Single-qubit density operator ρ = 1/2 (I + r · σ) on qubit q[{qubitIndex}]
          </div>
        </div>

        {/* Qubit Selector Pills */}
        {nQubits > 1 && onSelectQubit && (
          <div style={{ display: "flex", gap: 4, background: "#0b0f19", padding: 3, borderRadius: 6, border: "1px solid #334155" }}>
            {Array.from({ length: nQubits }).map((_, idx) => (
              <button
                key={idx}
                onClick={() => onSelectQubit(idx)}
                style={{
                  padding: "3px 8px",
                  borderRadius: 4,
                  border: "none",
                  fontSize: 11,
                  fontFamily: "monospace",
                  fontWeight: 600,
                  cursor: "pointer",
                  background: idx === qubitIndex ? "#38bdf8" : "transparent",
                  color: idx === qubitIndex ? "#0b0f19" : "#94a3b8",
                  transition: "all 120ms ease",
                }}
              >
                q[{idx}]
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Main 3D Interactive Canvas & Readout Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: 16, alignItems: "start" }}>
        {/* 3D SVG Bloch Sphere Container */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8, alignItems: "center" }}>
          <div
            style={{
              position: "relative",
              width: 290,
              height: 290,
              cursor: isDragging ? "grabbing" : "grab",
              userSelect: "none",
              borderRadius: 6,
              background: "#080d1a",
              border: "1px solid #334155",
            }}
            onMouseDown={handleMouseDown}
          >
            <svg
              width={290}
              height={290}
              viewBox="0 0 290 290"
              style={{ overflow: "visible" }}
            >
              <defs>
                <radialGradient id="sphereSurfaceDark" cx="42%" cy="38%" r="62%">
                  <stop offset="0%" stopColor="#1e293b" stopOpacity="0.45" />
                  <stop offset="80%" stopColor="#0f172a" stopOpacity="0.85" />
                  <stop offset="100%" stopColor="#050811" stopOpacity="0.95" />
                </radialGradient>
              </defs>

              {/* Sphere Body Fill */}
              <circle cx={CX} cy={CY} r={R} fill="url(#sphereSurfaceDark)" stroke="#334155" strokeWidth="1.2" />

              {/* Wireframe Latitudes: θ = π/4 and θ = 3π/4 */}
              <path d={renderCircle("xy", 48, 1 / Math.SQRT2, 1 / Math.SQRT2)} fill="none" stroke="#1e293b" strokeWidth="0.8" strokeDasharray="3 3" />
              <path d={renderCircle("xy", 48, -1 / Math.SQRT2, 1 / Math.SQRT2)} fill="none" stroke="#1e293b" strokeWidth="0.8" strokeDasharray="3 3" />

              {/* Wireframe Meridians: XZ (ϕ = 0), YZ (ϕ = π/2) */}
              <path d={renderCircle("xz", 64)} fill="none" stroke="#334155" strokeWidth="0.8" opacity="0.6" strokeDasharray="4 3" />
              <path d={renderCircle("yz", 64)} fill="none" stroke="#334155" strokeWidth="0.8" opacity="0.6" strokeDasharray="4 3" />

              {/* Equator (XY Plane, θ = π/2) */}
              <path d={renderCircle("xy", 64)} fill="none" stroke="#38bdf8" strokeWidth="1.2" opacity="0.75" strokeDasharray="3 2" />

              {/* Coordinate Axes */}
              {/* Negative Axes (dashed behind) */}
              <line x1={origin.sx} y1={origin.sy} x2={axisZNeg.sx} y2={axisZNeg.sy} stroke="#475569" strokeWidth="1" strokeDasharray="2 2" />
              <line x1={origin.sx} y1={origin.sy} x2={axisXNeg.sx} y2={axisXNeg.sy} stroke="#475569" strokeWidth="1" strokeDasharray="2 2" />
              <line x1={origin.sx} y1={origin.sy} x2={axisYNeg.sx} y2={axisYNeg.sy} stroke="#475569" strokeWidth="1" strokeDasharray="2 2" />

              {/* Positive Axes (solid) */}
              {/* Z-Axis (|0> / |1>) */}
              <line x1={origin.sx} y1={origin.sy} x2={axisZPos.sx} y2={axisZPos.sy} stroke="#38bdf8" strokeWidth="1.5" />
              <text x={axisZPos.sx} y={axisZPos.sy - 8} textAnchor="middle" fill="#38bdf8" fontSize="11" fontWeight="700" fontFamily="monospace">|0⟩ (+Z)</text>
              <text x={axisZNeg.sx} y={axisZNeg.sy + 14} textAnchor="middle" fill="#64748b" fontSize="10" fontWeight="600" fontFamily="monospace">|1⟩ (-Z)</text>

              {/* X-Axis (|+> / |->) */}
              <line x1={origin.sx} y1={origin.sy} x2={axisXPos.sx} y2={axisXPos.sy} stroke="#f87171" strokeWidth="1.2" />
              <text x={axisXPos.sx + 4} y={axisXPos.sy + 4} fill="#f87171" fontSize="10" fontWeight="600" fontFamily="monospace">|+⟩ (+X)</text>
              <text x={axisXNeg.sx - 6} y={axisXNeg.sy + 4} fill="#64748b" fontSize="9" fontWeight="600" textAnchor="end" fontFamily="monospace">|-⟩</text>

              {/* Y-Axis (|+i> / |-i>) */}
              <line x1={origin.sx} y1={origin.sy} x2={axisYPos.sx} y2={axisYPos.sy} stroke="#4ade80" strokeWidth="1.2" />
              <text x={axisYPos.sx + 4} y={axisYPos.sy + 4} fill="#4ade80" fontSize="10" fontWeight="600" fontFamily="monospace">|+i⟩ (+Y)</text>
              <text x={axisYNeg.sx - 6} y={axisYNeg.sy + 4} fill="#64748b" fontSize="9" fontWeight="600" textAnchor="end" fontFamily="monospace">|-i⟩</text>

              {/* Orthogonal projection guides for vector */}
              {r > 0.05 && (
                <>
                  {/* Radial line in XY plane from origin to (x, y, 0) */}
                  <line x1={origin.sx} y1={origin.sy} x2={tipProjXY.sx} y2={tipProjXY.sy} stroke="#38bdf8" strokeWidth="1" strokeDasharray="2 2" opacity="0.6" />
                  {/* Altitude line from (x, y, 0) to tip (x, y, z) */}
                  <line x1={tipProjXY.sx} y1={tipProjXY.sy} x2={tip.sx} y2={tip.sy} stroke="#e2e8f0" strokeWidth="1" strokeDasharray="2 2" opacity="0.7" />
                  {/* Line from tip to Z axis (0, 0, z) */}
                  <line x1={tip.sx} y1={tip.sy} x2={tipProjZ.sx} y2={tipProjZ.sy} stroke="#64748b" strokeWidth="1" strokeDasharray="2 2" opacity="0.5" />
                  {/* Projected point in XY plane */}
                  <circle cx={tipProjXY.sx} cy={tipProjXY.sy} r="2.5" fill="#38bdf8" opacity="0.75" />
                </>
              )}

              {/* State Vector Line */}
              <line
                x1={origin.sx}
                y1={origin.sy}
                x2={tip.sx}
                y2={tip.sy}
                stroke={isMixed ? "#f59e0b" : "#38bdf8"}
                strokeWidth="2.8"
                strokeLinecap="round"
              />

              {/* Center Pivot */}
              <circle cx={origin.sx} cy={origin.sy} r="3" fill="#64748b" />

              {/* State Vector Tip */}
              <circle
                cx={tip.sx}
                cy={tip.sy}
                r={isMixed ? 5 : 6}
                fill={isMixed ? "#f59e0b" : "#38bdf8"}
                stroke="#0f172a"
                strokeWidth="2"
              />
            </svg>

            {/* Orbit Instruction */}
            <div
              style={{
                position: "absolute",
                top: 8,
                left: 8,
                fontSize: 10,
                color: "#64748b",
                fontFamily: "monospace",
                pointerEvents: "none",
              }}
            >
              DRAG TO ORBIT · AZ {((azimuth * 180) / Math.PI).toFixed(0)}° EL {((elevation * 180) / Math.PI).toFixed(0)}°
            </div>
          </div>

          {/* Preset Camera Viewpoints */}
          <div style={{ display: "flex", gap: 4, flexWrap: "wrap", justifyContent: "center" }}>
            <button
              onClick={() => setViewPreset(35, 20)}
              style={{
                padding: "3px 8px",
                fontSize: 11,
                borderRadius: 4,
                border: "1px solid #334155",
                background: "#0b0f19",
                color: "#94a3b8",
                cursor: "pointer",
              }}
            >
              Default 3D
            </button>
            <button
              onClick={() => setViewPreset(0, 89)}
              style={{
                padding: "3px 8px",
                fontSize: 11,
                borderRadius: 4,
                border: "1px solid #334155",
                background: "#0b0f19",
                color: "#94a3b8",
                cursor: "pointer",
              }}
            >
              +Z (Poles)
            </button>
            <button
              onClick={() => setViewPreset(0, 0)}
              style={{
                padding: "3px 8px",
                fontSize: 11,
                borderRadius: 4,
                border: "1px solid #334155",
                background: "#0b0f19",
                color: "#94a3b8",
                cursor: "pointer",
              }}
            >
              +X (Hadamard)
            </button>
            <button
              onClick={() => setViewPreset(90, 0)}
              style={{
                padding: "3px 8px",
                fontSize: 11,
                borderRadius: 4,
                border: "1px solid #334155",
                background: "#0b0f19",
                color: "#94a3b8",
                cursor: "pointer",
              }}
            >
              +Y (Phase)
            </button>
          </div>
        </div>

        {/* Scientific Readouts & Density Operator Table */}
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {/* State Purity Classification Badge */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", ...subCardStyle }}>
            <div>
              <div style={labelStyle}>State Classification</div>
              <div style={{ fontWeight: 700, fontSize: 13, color: isPure ? "#4ade80" : "#f59e0b" }}>
                {isPure ? "Pure Quantum State (r = 1.0000)" : "Mixed / Entangled Subsystem (r < 1.0000)"}
              </div>
            </div>
            <div style={{ textAlign: "right" }}>
              <div style={labelStyle}>Von Neumann Entropy S(ρ)</div>
              <div style={{ ...monoValStyle, color: sVN < 1e-4 ? "#4ade80" : "#f59e0b" }}>
                {sVN.toFixed(4)} bits
              </div>
            </div>
          </div>

          {/* Coordinate Grid: Cartesian (x,y,z) & Spherical (θ,ϕ) */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            <div style={subCardStyle}>
              <div style={labelStyle}>Bloch Vector r = (rx, ry, rz)</div>
              <div style={{ ...monoValStyle, color: "#38bdf8", marginTop: 2 }}>
                [{x.toFixed(4)}, {y.toFixed(4)}, {z.toFixed(4)}]
              </div>
              <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>
                |r| = {r.toFixed(4)} · Purity γ = {purity.toFixed(4)}
              </div>
            </div>

            <div style={subCardStyle}>
              <div style={labelStyle}>Spherical Coordinates (θ, ϕ)</div>
              <div style={{ ...monoValStyle, marginTop: 2 }}>
                θ = {thetaDeg}° ({(theta / Math.PI).toFixed(3)}π)
              </div>
              <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 2 }}>
                ϕ = {phiDeg}° ({(phi / Math.PI).toFixed(3)}π)
              </div>
            </div>
          </div>

          {/* Basis Probabilities & Gauge */}
          <div style={subCardStyle}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 6 }}>
              <span>
                <strong style={{ color: "#38bdf8" }}>P(|0⟩) = (1+z)/2:</strong>{" "}
                <span style={monoValStyle}>{(prob0 * 100).toFixed(2)}%</span>
              </span>
              <span>
                <strong style={{ color: "#94a3b8" }}>P(|1⟩) = (1-z)/2:</strong>{" "}
                <span style={monoValStyle}>{(prob1 * 100).toFixed(2)}%</span>
              </span>
            </div>
            {/* Scientific calibrated bar */}
            <div style={{ height: 8, width: "100%", background: "#1e293b", borderRadius: 4, overflow: "hidden", display: "flex" }}>
              <div style={{ width: `${(prob0 * 100).toFixed(2)}%`, background: "#38bdf8" }} />
              <div style={{ width: `${(prob1 * 100).toFixed(2)}%`, background: "#475569" }} />
            </div>
          </div>

          {/* Reduced 2x2 Density Operator Matrix */}
          <div style={subCardStyle}>
            <div style={labelStyle}>Reduced Density Operator ρ = 1/2 (I + r · σ)</div>
            <div style={{ marginTop: 6, display: "inline-block", background: "#050811", border: "1px solid #1e293b", borderRadius: 4, padding: "6px 12px", fontFamily: "monospace", fontSize: 12 }}>
              <div>
                [ {rho00.toFixed(4)} , {rho01Re >= 0 ? "+" : ""}{rho01Re.toFixed(4)}{rho01Im >= 0 ? "+" : ""}{rho01Im.toFixed(4)}i ]
              </div>
              <div>
                [ {rho01Re >= 0 ? "+" : ""}{rho01Re.toFixed(4)}{-rho01Im >= 0 ? "+" : ""}{(-rho01Im).toFixed(4)}i , {rho11.toFixed(4)} ]
              </div>
            </div>
          </div>

          {/* State Ket Expansion */}
          {isPure && (
            <div style={{ fontSize: 11, color: "#64748b", fontFamily: "monospace", background: "#0b0f19", padding: "6px 10px", borderRadius: 4, border: "1px solid #1e293b" }}>
              {`|ψ⟩ = cos(${(theta / 2).toFixed(3)})|0⟩ + e^(i${phi.toFixed(3)}) sin(${(theta / 2).toFixed(3)})|1⟩`}
            </div>
          )}
        </div>
      </div>

      {/* Multi-Qubit Mini Preview Strip */}
      {allVectors && allVectors.length > 1 && (
        <div style={{ borderTop: "1px solid #1e293b", paddingTop: 10 }}>
          <div style={{ ...labelStyle, marginBottom: 8 }}>
            All Qubits Geometric States ({allVectors.length} total)
          </div>
          <div style={{ display: "flex", gap: 8, overflowX: "auto", paddingBottom: 4 }}>
            {allVectors.map((v, i) => {
              const miniR = 20;
              const miniCx = 24;
              const miniCy = 24;
              const mx = miniCx + miniR * v.x * 0.85;
              const my = miniCy - miniR * v.z * 0.85;
              const isCur = i === qubitIndex;
              return (
                <div
                  key={i}
                  onClick={() => onSelectQubit && onSelectQubit(i)}
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    cursor: "pointer",
                    padding: "4px 8px",
                    borderRadius: 4,
                    border: isCur ? "1px solid #38bdf8" : "1px solid #334155",
                    background: isCur ? "#1e293b" : "#0b0f19",
                    transition: "all 120ms ease",
                  }}
                >
                  <svg width={48} height={48}>
                    <circle cx={miniCx} cy={miniCy} r={miniR} fill="#080d1a" stroke="#334155" strokeWidth="1" />
                    <line x1={miniCx} y1={miniCy - miniR} x2={miniCx} y2={miniCy + miniR} stroke="#475569" strokeWidth="0.8" strokeDasharray="2 2" />
                    <line x1={miniCx - miniR} y1={miniCy} x2={miniCx + miniR} y2={miniCy} stroke="#475569" strokeWidth="0.8" strokeDasharray="2 2" />
                    <line x1={miniCx} y1={miniCy} x2={mx} y2={my} stroke={v.r >= 0.95 ? "#38bdf8" : "#f59e0b"} strokeWidth="1.8" />
                    <circle cx={mx} cy={my} r="2.5" fill={v.r >= 0.95 ? "#38bdf8" : "#f59e0b"} />
                  </svg>
                  <span style={{ fontSize: 10, fontFamily: "monospace", fontWeight: 700, color: isCur ? "#38bdf8" : "#94a3b8", marginTop: 2 }}>
                    q[{i}]
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
