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

export function BlochSphere({
  vector,
  qubitIndex,
  nQubits,
  onSelectQubit,
  allVectors,
}: BlochSphereProps) {
  // Camera angles: azimuth (alpha, rotation around Z) and elevation (beta, tilt)
  const [azimuth, setAzimuth] = useState<number>(35 * (Math.PI / 180));
  const [elevation, setElevation] = useState<number>(20 * (Math.PI / 180));
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const dragStart = useRef<{ x: number; y: number; az: number; el: number }>({
    x: 0,
    y: 0,
    az: 0,
    el: 0,
  });

  const svgRef = useRef<SVGSVGElement | null>(null);

  // Mouse drag handlers for 3D orbital rotation
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
      // 0.01 radians per pixel
      const newAz = dragStart.current.az + dx * 0.012;
      const newEl = Math.max(
        -Math.PI / 2.2,
        Math.min(Math.PI / 2.2, dragStart.current.el + dy * 0.012)
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

  const resetView = () => {
    setAzimuth(35 * (Math.PI / 180));
    setElevation(20 * (Math.PI / 180));
  };

  // 3D Projection parameters
  const CX = 140;
  const CY = 140;
  const R = 95; // SVG sphere radius in px

  // Project 3D (x, y, z) where x,y,z in [-1, 1] to 2D SVG (sx, sy, depth)
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
  const renderCircle = (plane: "xy" | "xz" | "yz", numPts = 64) => {
    const pts: { sx: number; sy: number; depth: number }[] = [];
    for (let i = 0; i <= numPts; i++) {
      const angle = (i / numPts) * 2 * Math.PI;
      let x = 0;
      let y = 0;
      let z = 0;
      if (plane === "xy") {
        x = Math.cos(angle);
        y = Math.sin(angle);
      } else if (plane === "xz") {
        x = Math.cos(angle);
        z = Math.sin(angle);
      } else {
        y = Math.cos(angle);
        z = Math.sin(angle);
      }
      pts.push(project(x, y, z));
    }
    return pts.map((p, idx) => `${idx === 0 ? "M" : "L"} ${p.sx.toFixed(1)} ${p.sy.toFixed(1)}`).join(" ");
  };

  // Coordinate axes
  const origin = project(0, 0, 0);
  const axisZPos = project(0, 0, 1.25);
  const axisZNeg = project(0, 0, -1.25);
  const axisXPos = project(1.25, 0, 0);
  const axisXNeg = project(-1.25, 0, 0);
  const axisYPos = project(0, 1.25, 0);
  const axisYNeg = project(0, -1.25, 0);

  // State vector tip
  const { x, y, z, r, theta, phi, prob0, prob1 } = vector;
  const tip = project(x, y, z);
  const tipProjXY = project(x, y, 0);

  // Formatted values
  const thetaDeg = ((theta * 180) / Math.PI).toFixed(1);
  const phiDeg = ((phi * 180) / Math.PI).toFixed(1);
  const thetaPi = (theta / Math.PI).toFixed(2);
  const phiPi = (phi / Math.PI).toFixed(2);
  const isPure = r >= 0.99;
  const isMixed = r < 0.95;

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
        fontFamily: "inherit",
      }}
    >
      {/* Header and Qubit Selector */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 18 }}>🌐</span>
          <div>
            <div style={{ fontWeight: 700, fontSize: 14, color: "#0f172a" }}>
              3D Blox Sferasi (Bloch Sphere)
            </div>
            <div style={{ fontSize: 11, color: "#64748b" }}>
              Qubit q[{qubitIndex}] ning geometrik kvant holat fazosi
            </div>
          </div>
        </div>

        {/* Qubit Selector Pills */}
        {nQubits > 1 && onSelectQubit && (
          <div style={{ display: "flex", gap: 4, background: "#f1f5f9", padding: 3, borderRadius: 6 }}>
            {Array.from({ length: nQubits }).map((_, idx) => (
              <button
                key={idx}
                onClick={() => onSelectQubit(idx)}
                style={{
                  padding: "3px 8px",
                  borderRadius: 4,
                  border: "none",
                  fontSize: 11,
                  fontWeight: 600,
                  cursor: "pointer",
                  background: idx === qubitIndex ? "#2563eb" : "transparent",
                  color: idx === qubitIndex ? "#ffffff" : "#475569",
                  transition: "all 120ms ease",
                }}
              >
                q[{idx}]
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Main 3D Interactive Canvas & Readout */}
      <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap", justifyContent: "center" }}>
        {/* 3D SVG Bloch Sphere */}
        <div
          style={{
            position: "relative",
            width: 280,
            height: 280,
            cursor: isDragging ? "grabbing" : "grab",
            userSelect: "none",
            borderRadius: 12,
            background: "radial-gradient(circle at 50% 50%, #f8fafc 0%, #f1f5f9 100%)",
            border: "1px solid #cbd5e1",
            boxShadow: "inset 0 2px 4px rgba(0,0,0,0.03)",
          }}
          onMouseDown={handleMouseDown}
        >
          <svg
            ref={svgRef}
            width={280}
            height={280}
            viewBox="0 0 280 280"
            style={{ overflow: "visible" }}
          >
            <defs>
              {/* Radial gradient for sphere fill */}
              <radialGradient id="sphereGrad" cx="40%" cy="35%" r="65%">
                <stop offset="0%" stopColor="#ffffff" stopOpacity="0.8" />
                <stop offset="70%" stopColor="#e0e7ff" stopOpacity="0.3" />
                <stop offset="100%" stopColor="#c7d2fe" stopOpacity="0.45" />
              </radialGradient>
              {/* Glow filter for state vector */}
              <filter id="vectorGlow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="2" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
              </filter>
            </defs>

            {/* Translucent Sphere Body */}
            <circle cx={CX} cy={CY} r={R} fill="url(#sphereGrad)" stroke="#94a3b8" strokeWidth="1.2" strokeDasharray="3 3" />

            {/* Wireframe Circles: Equator (XY) and Meridians */}
            <path d={renderCircle("xy")} fill="none" stroke="#60a5fa" strokeWidth="1.2" opacity="0.6" strokeDasharray="2 2" />
            <path d={renderCircle("xz")} fill="none" stroke="#cbd5e1" strokeWidth="0.8" opacity="0.4" />
            <path d={renderCircle("yz")} fill="none" stroke="#cbd5e1" strokeWidth="0.8" opacity="0.4" />

            {/* Coordinate Axes */}
            {/* Z axis (|0> / |1>) */}
            <line x1={axisZNeg.sx} y1={axisZNeg.sy} x2={axisZPos.sx} y2={axisZPos.sy} stroke="#64748b" strokeWidth="1.4" />
            <text x={axisZPos.sx} y={axisZPos.sy - 8} textAnchor="middle" fill="#0f172a" fontSize="11" fontWeight="700">|0⟩ (+Z)</text>
            <text x={axisZNeg.sx} y={axisZNeg.sy + 14} textAnchor="middle" fill="#0f172a" fontSize="11" fontWeight="700">|1⟩ (-Z)</text>

            {/* X axis (|+> / |->) */}
            <line x1={axisXNeg.sx} y1={axisXNeg.sy} x2={axisXPos.sx} y2={axisXPos.sy} stroke="#94a3b8" strokeWidth="1" strokeDasharray="3 2" />
            <text x={axisXPos.sx + 6} y={axisXPos.sy + 4} fill="#475569" fontSize="10" fontWeight="600">|+⟩ (X)</text>

            {/* Y axis (|i> / |-i>) */}
            <line x1={axisYNeg.sx} y1={axisYNeg.sy} x2={axisYPos.sx} y2={axisYPos.sy} stroke="#94a3b8" strokeWidth="1" strokeDasharray="3 2" />
            <text x={axisYPos.sx + 6} y={axisYPos.sy + 4} fill="#475569" fontSize="10" fontWeight="600">|i⟩ (Y)</text>

            {/* Projection onto XY plane (dotted guide lines) */}
            {r > 0.05 && (
              <>
                <line x1={origin.sx} y1={origin.sy} x2={tipProjXY.sx} y2={tipProjXY.sy} stroke="#3b82f6" strokeWidth="1.2" strokeDasharray="2 2" opacity="0.7" />
                <line x1={tipProjXY.sx} y1={tipProjXY.sy} x2={tip.sx} y2={tip.sy} stroke="#ef4444" strokeWidth="1.2" strokeDasharray="2 2" opacity="0.7" />
              </>
            )}

            {/* State Vector Line */}
            <line
              x1={origin.sx}
              y1={origin.sy}
              x2={tip.sx}
              y2={tip.sy}
              stroke={isMixed ? "#d97706" : "#2563eb"}
              strokeWidth="3.2"
              strokeLinecap="round"
              filter="url(#vectorGlow)"
            />

            {/* Center origin point */}
            <circle cx={origin.sx} cy={origin.sy} r="3" fill="#64748b" />

            {/* Tip Arrowhead / Dot */}
            <circle
              cx={tip.sx}
              cy={tip.sy}
              r={isMixed ? "5.5" : "6.5"}
              fill={isMixed ? "#f59e0b" : "#2563eb"}
              stroke="#ffffff"
              strokeWidth="2"
              filter="url(#vectorGlow)"
            />
          </svg>

          {/* Quick Reset Orbit Button */}
          <button
            onClick={resetView}
            title="3D ko'rish burchagini qayta tiklash"
            style={{
              position: "absolute",
              bottom: 8,
              right: 8,
              padding: "3px 7px",
              fontSize: 10,
              borderRadius: 4,
              border: "1px solid #cbd5e1",
              background: "rgba(255,255,255,0.85)",
              color: "#475569",
              cursor: "pointer",
            }}
          >
            ↺ Reset 3D
          </button>

          {/* Drag instruction overlay */}
          <div
            style={{
              position: "absolute",
              top: 8,
              left: 8,
              fontSize: 10,
              color: "#94a3b8",
              pointerEvents: "none",
            }}
          >
            🖱 3D aylantirish uchun torting
          </div>
        </div>

        {/* State Metrics & Probabilities Card */}
        <div style={{ flex: 1, minWidth: 240, display: "flex", flexDirection: "column", gap: 10 }}>
          {/* State status badge */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: "#475569" }}>Holat Turi:</span>
            {isPure ? (
              <span
                style={{
                  fontSize: 11,
                  fontWeight: 700,
                  padding: "2px 8px",
                  borderRadius: 999,
                  background: "#dcfce7",
                  color: "#15803d",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 4,
                }}
              >
                ● Sof Holat (Pure State, r ≈ 1)
              </span>
            ) : (
              <span
                style={{
                  fontSize: 11,
                  fontWeight: 700,
                  padding: "2px 8px",
                  borderRadius: 999,
                  background: "#fef3c7",
                  color: "#b45309",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 4,
                }}
              >
                ⚠ Aralash/Chalkashgan (Entangled, r &lt; 1)
              </span>
            )}
          </div>

          {/* Probabilities P(|0>) and P(|1>) */}
          <div style={{ background: "#f8fafc", padding: 10, borderRadius: 8, border: "1px solid #e2e8f0" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
              <span style={{ fontWeight: 600, color: "#1e293b" }}>P(|0⟩): {(prob0 * 100).toFixed(1)}%</span>
              <span style={{ fontWeight: 600, color: "#1e293b" }}>P(|1⟩): {(prob1 * 100).toFixed(1)}%</span>
            </div>
            {/* Dual color progress bar */}
            <div style={{ height: 10, width: "100%", background: "#e2e8f0", borderRadius: 5, overflow: "hidden", display: "flex" }}>
              <div style={{ width: `${(prob0 * 100).toFixed(1)}%`, background: "#3b82f6", transition: "width 150ms ease" }} title={`P(|0>) = ${(prob0 * 100).toFixed(1)}%`} />
              <div style={{ width: `${(prob1 * 100).toFixed(1)}%`, background: "#ec4899", transition: "width 150ms ease" }} title={`P(|1>) = ${(prob1 * 100).toFixed(1)}%`} />
            </div>
          </div>

          {/* Coordinates Grid */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, fontSize: 12 }}>
            <div style={{ background: "#f8fafc", padding: "6px 8px", borderRadius: 6, border: "1px solid #e2e8f0" }}>
              <span style={{ color: "#64748b" }}>Qutb burchagi θ:</span>
              <div style={{ fontWeight: 700, color: "#0f172a" }}>{thetaDeg}° <span style={{ fontSize: 10, color: "#94a3b8" }}>({thetaPi}π)</span></div>
            </div>
            <div style={{ background: "#f8fafc", padding: "6px 8px", borderRadius: 6, border: "1px solid #e2e8f0" }}>
              <span style={{ color: "#64748b" }}>Azimut burchagi ϕ:</span>
              <div style={{ fontWeight: 700, color: "#0f172a" }}>{phiDeg}° <span style={{ fontSize: 10, color: "#94a3b8" }}>({phiPi}π)</span></div>
            </div>
            <div style={{ background: "#f8fafc", padding: "6px 8px", borderRadius: 6, border: "1px solid #e2e8f0" }}>
              <span style={{ color: "#64748b" }}>Blox Vektori (x, y, z):</span>
              <div style={{ fontWeight: 700, color: "#0f172a", fontSize: 11 }}>
                ({x.toFixed(2)}, {y.toFixed(2)}, {z.toFixed(2)})
              </div>
            </div>
            <div style={{ background: "#f8fafc", padding: "6px 8px", borderRadius: 6, border: "1px solid #e2e8f0" }}>
              <span style={{ color: "#64748b" }}>Tozalik Radiusi r:</span>
              <div style={{ fontWeight: 700, color: isMixed ? "#b45309" : "#15803d" }}>
                {r.toFixed(3)} / 1.000
              </div>
            </div>
          </div>

          {/* Mathematical state formula hint */}
          <div style={{ fontSize: 11, color: "#64748b", background: "#f1f5f9", padding: "6px 10px", borderRadius: 6 }}>
            |ψ⟩ = cos({(theta / 2).toFixed(2)})|0⟩ + e^{`i(${phi.toFixed(2)})`}sin({(theta / 2).toFixed(2)})|1⟩
          </div>
        </div>
      </div>

      {/* Mini previews for all other qubits if multi-qubit */}
      {allVectors && allVectors.length > 1 && (
        <div style={{ borderTop: "1px solid #e2e8f0", paddingTop: 10 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: "#64748b", marginBottom: 6 }}>
            Barcha Qubitlar Mini-Sferalari:
          </div>
          <div style={{ display: "flex", gap: 10, overflowX: "auto", paddingBottom: 4 }}>
            {allVectors.map((v, i) => {
              const miniR = 22;
              const miniCx = 26;
              const miniCy = 26;
              // Simple projection for mini preview
              const mx = miniCx + miniR * v.x * 0.8;
              const my = miniCy - miniR * v.z * 0.8;
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
                    borderRadius: 6,
                    border: isCur ? "2px solid #2563eb" : "1px solid #e2e8f0",
                    background: isCur ? "#eff6ff" : "#f8fafc",
                    transition: "all 120ms ease",
                  }}
                >
                  <svg width={52} height={52}>
                    <circle cx={miniCx} cy={miniCy} r={miniR} fill="#f1f5f9" stroke="#cbd5e1" strokeWidth="1" />
                    <line x1={miniCx} y1={miniCy - miniR} x2={miniCx} y2={miniCy + miniR} stroke="#94a3b8" strokeWidth="0.8" strokeDasharray="2 2" />
                    <line x1={miniCx - miniR} y1={miniCy} x2={miniCx + miniR} y2={miniCy} stroke="#94a3b8" strokeWidth="0.8" strokeDasharray="2 2" />
                    <line x1={miniCx} y1={miniCy} x2={mx} y2={my} stroke="#2563eb" strokeWidth="2" />
                    <circle cx={mx} cy={my} r="3" fill="#2563eb" />
                  </svg>
                  <span style={{ fontSize: 10, fontWeight: 700, color: isCur ? "#1d4ed8" : "#475569", marginTop: 2 }}>
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
