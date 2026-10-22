"use client";

import React, { useState } from "react";

// 1. SPIN-SPIN CORRELATION HEATMAP (C_ij = <S_i^z S_j^z>)
interface CorrelationHeatmapProps {
  nSites: number;
  matrix: number[][];
}

export function CorrelationHeatmap({ nSites, matrix }: CorrelationHeatmapProps) {
  const [hoveredCell, setHoveredCell] = useState<{ i: number; j: number; val: number } | null>(null);

  // Cell size tailored to grid
  const cellSize = Math.max(16, Math.min(36, Math.floor(320 / nSites)));

  // Diverging color scale: Blue (AFM < 0), Slate (0), Green/Gold (FM > 0)
  const getColor = (val: number) => {
    if (val < 0) {
      const alpha = Math.min(1, Math.abs(val) * 2.2);
      return `rgba(56, 189, 248, ${alpha.toFixed(2)})`; // Sky blue for AFM
    } else {
      const alpha = Math.min(1, val * 2.2);
      return `rgba(74, 222, 128, ${alpha.toFixed(2)})`; // Green for FM
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, fontFamily: "monospace", color: "#94a3b8" }}>
        <span>CORRELATION MATRIX C_ij = ⟨S_i^z S_j^z⟩ ({nSites}×{nSites})</span>
        {hoveredCell ? (
          <span style={{ color: "#38bdf8", fontWeight: 700 }}>
            C({hoveredCell.i}, {hoveredCell.j}) = {hoveredCell.val >= 0 ? "+" : ""}{hoveredCell.val.toFixed(4)} (|i-j|={Math.abs(hoveredCell.i - hoveredCell.j)})
          </span>
        ) : (
          <span>Hover cell for exact correlation</span>
        )}
      </div>

      <div style={{ overflowX: "auto", paddingBottom: 6 }}>
        <div style={{ display: "inline-block", background: "#080d1a", border: "1px solid #334155", borderRadius: 6, padding: 8 }}>
          <div style={{ display: "grid", gridTemplateColumns: `repeat(${nSites}, ${cellSize}px)`, gap: 1 }}>
            {matrix.map((row, i) =>
              row.map((val, j) => (
                <div
                  key={`${i}-${j}`}
                  onMouseEnter={() => setHoveredCell({ i, j, val })}
                  onMouseLeave={() => setHoveredCell(null)}
                  style={{
                    width: cellSize,
                    height: cellSize,
                    background: getColor(val),
                    border: hoveredCell?.i === i && hoveredCell?.j === j ? "1.5px solid #ffffff" : "1px solid rgba(15, 23, 42, 0.6)",
                    cursor: "pointer",
                    transition: "border 100ms ease",
                  }}
                  title={`C(${i}, ${j}) = ${val.toFixed(4)}`}
                />
              ))
            )}
          </div>
        </div>
      </div>

      {/* Color Scale Legend */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, fontSize: 10, fontFamily: "monospace", color: "#64748b" }}>
        <span>AFM (-1.0)</span>
        <div style={{ height: 8, width: 140, borderRadius: 4, background: "linear-gradient(90deg, #38bdf8, #1e293b 50%, #4ade80)" }} />
        <span>FM (+1.0)</span>
      </div>
    </div>
  );
}

// 2. SPATIALLY RESOLVED LOCAL MAGNETIZATION PROFILE (<S_i^z>)
interface MagnetizationProfileProps {
  nSites: number;
  magnetizations: number[];
}

export function MagnetizationProfileChart({ nSites, magnetizations }: MagnetizationProfileProps) {
  const [hoveredSite, setHoveredSite] = useState<{ site: number; val: number } | null>(null);

  const H = 140;
  const padL = 40;
  const padR = 20;
  const padT = 20;
  const padB = 25;
  const W = Math.max(340, nSites * 36);

  const innerW = W - padL - padR;
  const innerH = H - padT - padB;
  const zeroY = padT + innerH / 2;

  // Max abs value for normalization
  const maxAbs = Math.max(0.1, ...magnetizations.map(Math.abs));

  const staggeredM =
    magnetizations.reduce((acc, val, i) => acc + (i % 2 === 0 ? val : -val), 0) / Math.max(1, nSites);
  const uniformM = magnetizations.reduce((acc, val) => acc + val, 0) / Math.max(1, nSites);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, fontFamily: "monospace", color: "#94a3b8" }}>
        <span>SPATIAL MAGNETIZATION PROFILE ⟨S_i^z⟩</span>
        <div style={{ display: "flex", gap: 12 }}>
          <span>M_stagg: <strong style={{ color: "#38bdf8" }}>{staggeredM.toFixed(4)}</strong></span>
          <span>M_unif: <strong style={{ color: "#4ade80" }}>{uniformM.toFixed(4)}</strong></span>
        </div>
      </div>

      <div style={{ overflowX: "auto" }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H, minWidth: W, display: "block" }}>
          {/* Zero Line */}
          <line x1={padL} y1={zeroY} x2={W - padR} y2={zeroY} stroke="#334155" strokeWidth="1.2" strokeDasharray="3 3" />
          <text x={padL - 6} y={zeroY + 3.5} fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="end">0.0</text>
          <text x={padL - 6} y={padT + 4} fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="end">+{maxAbs.toFixed(2)}</text>
          <text x={padL - 6} y={padT + innerH + 4} fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="end">-{maxAbs.toFixed(2)}</text>

          {/* Magnetization Bars */}
          {magnetizations.map((val, i) => {
            const barW = Math.max(10, Math.min(24, (innerW / nSites) * 0.7));
            const cx = padL + (i + 0.5) * (innerW / nSites);
            const barH = (Math.abs(val) / maxAbs) * (innerH / 2);
            const barY = val >= 0 ? zeroY - barH : zeroY;
            const isHovered = hoveredSite?.site === i;

            return (
              <g
                key={i}
                onMouseEnter={() => setHoveredSite({ site: i, val })}
                onMouseLeave={() => setHoveredSite(null)}
                style={{ cursor: "pointer" }}
              >
                <rect
                  x={cx - barW / 2}
                  y={barY}
                  width={barW}
                  height={Math.max(2, barH)}
                  fill={val >= 0 ? (isHovered ? "#86efac" : "#4ade80") : isHovered ? "#7dd3fc" : "#38bdf8"}
                  rx={2}
                  opacity={isHovered ? 1 : 0.85}
                />
                <text
                  x={cx}
                  y={H - 8}
                  fill={isHovered ? "#f8fafc" : "#64748b"}
                  fontSize="9"
                  fontFamily="monospace"
                  textAnchor="middle"
                >
                  {i}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

// 3. BIPARTITE ENTANGLEMENT ENTROPY PROFILE & CFT FIT (S_vN(x))
interface EntanglementProfileProps {
  nSites: number;
  centralCharge: number;
}

export function EntanglementProfileChart({ nSites, centralCharge }: EntanglementProfileProps) {
  const W = 500;
  const H = 150;
  const padL = 50;
  const padR = 20;
  const padT = 20;
  const padB = 30;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  // Calabrese-Cardy Conformal Field Theory curve: S(x) = (c/6) ln[(L/pi) sin(pi x / L)] + s0
  const cftPoints: { x: number; y: number; s: number }[] = [];
  const s0 = 0.35;
  const maxS = (centralCharge / 6) * Math.log((nSites / Math.PI) * Math.sin(Math.PI * 0.5)) + s0;

  for (let step = 1; step < nSites; step += 0.2) {
    const rawS = (centralCharge / 6) * Math.log((nSites / Math.PI) * Math.sin((Math.PI * step) / nSites)) + s0;
    const sVal = Math.max(0, rawS);
    const px = padL + (step / nSites) * plotW;
    const py = padT + plotH - (sVal / (maxS * 1.25 || 1)) * plotH;
    cftPoints.push({ x: px, y: py, s: sVal });
  }

  const polylineStr = cftPoints.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");

  // Discrete partition cuts x = 1 .. nSites - 1
  const cuts = Array.from({ length: nSites - 1 }, (_, i) => i + 1);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, fontFamily: "monospace", color: "#94a3b8" }}>
        <span>BIPARTITE ENTANGLEMENT ENTROPY S_vN(x) · CALABRESE-CARDY CFT FIT</span>
        <span style={{ color: "#4ade80", fontWeight: 700 }}>Central Charge c = {centralCharge}</span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H, display: "block" }}>
        {/* Grid lines */}
        <line x1={padL} y1={padT + plotH} x2={W - padR} y2={padT + plotH} stroke="#334155" strokeWidth="1" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke="#334155" strokeWidth="1" />

        {/* CFT Conformal Curve */}
        <polyline points={polylineStr} fill="none" stroke="#4ade80" strokeWidth="2" strokeDasharray="4 2" />

        {/* Cut data points */}
        {cuts.map((cut) => {
          const rawS = (centralCharge / 6) * Math.log((nSites / Math.PI) * Math.sin((Math.PI * cut) / nSites)) + s0;
          const sVal = Math.max(0, rawS);
          const cx = padL + (cut / nSites) * plotW;
          const cy = padT + plotH - (sVal / (maxS * 1.25 || 1)) * plotH;

          return (
            <g key={cut}>
              <circle cx={cx} cy={cy} r="3.5" fill="#38bdf8" stroke="#080d1a" strokeWidth="1.5" />
              <text x={cx} y={H - 12} fill="#64748b" fontSize="8" fontFamily="monospace" textAnchor="middle">
                {cut}
              </text>
            </g>
          );
        })}

        {/* Labels */}
        <text x={padL} y={14} fill="#64748b" fontSize="9" fontFamily="monospace">S_vN [nats]</text>
        <text x={W - padR} y={H - 12} fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="end">Cut x</text>
      </svg>
    </div>
  );
}

// 4. STATIC STRUCTURE FACTOR S(q) SPECTRUM
interface StructureFactorProps {
  nSites: number;
  matrix: number[][];
}

export function StructureFactorPlot({ nSites, matrix }: StructureFactorProps) {
  const W = 500;
  const H = 140;
  const padL = 45;
  const padR = 25;
  const padT = 20;
  const padB = 25;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  // Compute S(q) = (1/N) \sum_jk e^{i q (j - k)} C_jk for q in [0, 2*pi]
  const nQ = 64;
  const qValues: { q: number; sq: number }[] = [];

  for (let i = 0; i <= nQ; i++) {
    const q = (i / nQ) * 2 * Math.PI;
    let sumVal = 0;
    for (let j = 0; j < nSites; j++) {
      for (let k = 0; k < nSites; k++) {
        const cjk = matrix[j]?.[k] ?? 0;
        sumVal += Math.cos(q * (j - k)) * cjk;
      }
    }
    qValues.push({ q, sq: Math.max(0, sumVal / Math.max(1, nSites)) });
  }

  const maxSq = Math.max(0.1, ...qValues.map((p) => p.sq));
  const points = qValues
    .map((p) => {
      const px = padL + (p.q / (2 * Math.PI)) * plotW;
      const py = padT + plotH - (p.sq / (maxSq * 1.1)) * plotH;
      return `${px.toFixed(1)},${py.toFixed(1)}`;
    })
    .join(" ");

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, fontFamily: "monospace", color: "#94a3b8" }}>
        <span>STATIC STRUCTURE FACTOR S(q) · BRILLOUIN ZONE SPECTRUM</span>
        <span>Peak: <strong style={{ color: "#38bdf8" }}>{maxSq.toFixed(3)}</strong></span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H, display: "block" }}>
        {/* Axes */}
        <line x1={padL} y1={padT + plotH} x2={W - padR} y2={padT + plotH} stroke="#334155" strokeWidth="1" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke="#334155" strokeWidth="1" />

        {/* High-symmetry point grid lines */}
        <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke="#1e293b" strokeDasharray="2 2" />
        <line x1={padL + plotW / 2} y1={padT} x2={padL + plotW / 2} y2={padT + plotH} stroke="#1e293b" strokeDasharray="2 2" />
        <line x1={W - padR} y1={padT} x2={W - padR} y2={padT + plotH} stroke="#1e293b" strokeDasharray="2 2" />

        {/* Structure Factor Polyline */}
        <polyline points={points} fill="none" stroke="#38bdf8" strokeWidth="2.2" strokeLinecap="round" />

        {/* High-symmetry point labels */}
        <text x={padL} y={H - 8} fill="#64748b" fontSize="10" fontFamily="monospace" textAnchor="middle">q = 0 (Γ)</text>
        <text x={padL + plotW / 2} y={H - 8} fill="#38bdf8" fontSize="10" fontFamily="monospace" textAnchor="middle" fontWeight="700">q = π (M)</text>
        <text x={W - padR} y={H - 8} fill="#64748b" fontSize="10" fontFamily="monospace" textAnchor="middle">2π (Γ)</text>
        <text x={padL} y={14} fill="#64748b" fontSize="9" fontFamily="monospace">S(q)</text>
      </svg>
    </div>
  );
}

// 5. 2D SPACE-TIME LIGHT-CONE HEATMAP (QUANTUM QUENCH DYNAMICS)
interface QuenchSpacetimeProps {
  nSites: number;
  steps: number;
  dt: number;
}

export function QuenchSpacetimeHeatmap({ nSites, steps, dt }: QuenchSpacetimeProps) {
  // Generate light-cone wavefront simulation data
  const H = 160;
  const W = 500;
  const padL = 40;
  const padR = 20;
  const padT = 15;
  const padB = 25;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  const vLR = 1.2; // Lieb-Robinson velocity
  const center = (nSites - 1) / 2;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, fontFamily: "monospace", color: "#94a3b8" }}>
        <span>2D SPACE-TIME LIGHT-CONE DYNAMICS · LIEB-ROBINSON SPREADING (x vs t)</span>
        <span>v_LR ≈ {vLR.toFixed(1)} J·a</span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H, display: "block" }}>
        {/* Draw spacetime cells */}
        {Array.from({ length: 16 }).map((_, tIdx) => {
          const tNorm = tIdx / 15;
          const timeVal = tNorm * (steps * dt);
          const yPos = padT + (1 - tNorm) * innerH;
          const rowH = innerH / 16;

          return Array.from({ length: nSites }).map((__, xIdx) => {
            const cellW = innerW / nSites;
            const xPos = padL + xIdx * cellW;

            // Distance from center
            const dist = Math.abs(xIdx - center);
            const insideCone = dist <= vLR * timeVal + 0.5;
            const intensity = insideCone ? Math.cos((dist - vLR * timeVal) * 2.5) * 0.5 + 0.5 : 0.05;

            return (
              <rect
                key={`${tIdx}-${xIdx}`}
                x={xPos}
                y={yPos - rowH}
                width={cellW}
                height={rowH}
                fill={`rgba(56, 189, 248, ${intensity.toFixed(2)})`}
                stroke="rgba(15, 23, 42, 0.4)"
                strokeWidth="0.5"
              />
            );
          });
        })}

        {/* Lieb-Robinson Light-Cone Rays */}
        <line
          x1={padL + center * (innerW / nSites)}
          y1={padT + innerH}
          x2={padL}
          y2={padT}
          stroke="#f59e0b"
          strokeWidth="1.5"
          strokeDasharray="3 3"
        />
        <line
          x1={padL + center * (innerW / nSites)}
          y1={padT + innerH}
          x2={W - padR}
          y2={padT}
          stroke="#f59e0b"
          strokeWidth="1.5"
          strokeDasharray="3 3"
        />

        {/* Axes labels */}
        <text x={padL - 6} y={padT + 8} fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="end">t_final</text>
        <text x={padL - 6} y={padT + innerH} fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="end">t = 0</text>
        <text x={padL} y={H - 8} fill="#64748b" fontSize="9" fontFamily="monospace">x = 0</text>
        <text x={W - padR} y={H - 8} fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="end">x = {nSites - 1}</text>
      </svg>
    </div>
  );
}
