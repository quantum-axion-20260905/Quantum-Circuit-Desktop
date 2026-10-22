"use client";

import React, { useState } from "react";
import type { LatticeGraph } from "../lib/agent";

function projectGraph(graph: LatticeGraph) {
  const raw = graph.sites.map((site) => {
    const [x = 0, y = 0, z = 0] = site.coordinate;
    return { index: site.index, x: x + z * 0.45, y: y - z * 0.32, depth: z, coord: site.coordinate };
  });
  const minX = Math.min(...raw.map((site) => site.x), 0);
  const maxX = Math.max(...raw.map((site) => site.x), 1);
  const minY = Math.min(...raw.map((site) => site.y), 0);
  const maxY = Math.max(...raw.map((site) => site.y), 1);
  const width = Math.max(1, maxX - minX);
  const height = Math.max(1, maxY - minY);
  return new Map(
    raw.map((site) => [
      site.index,
      {
        x: 48 + ((site.x - minX) / width) * 440,
        y: 40 + ((maxY - site.y) / height) * 200,
        depth: site.depth,
        coord: site.coord,
      },
    ])
  );
}

export function LatticeCanvas({ graph }: { graph: LatticeGraph }) {
  const [selectedSite, setSelectedSite] = useState<number | null>(null);
  const points = projectGraph(graph);

  // Find neighbors of selected site
  const neighborSet = new Set<number>();
  if (selectedSite !== null) {
    graph.edges.forEach((e) => {
      if (e.source === selectedSite) neighborSet.add(e.target);
      if (e.target === selectedSite) neighborSet.add(e.source);
    });
  }

  return (
    <div
      style={{
        borderRadius: 8,
        overflow: "hidden",
        background: "#080d1a",
        border: "1px solid #334155",
        position: "relative",
      }}
    >
      {/* Top Header Scale / Metadata */}
      <div
        style={{
          position: "absolute",
          top: 10,
          left: 12,
          display: "flex",
          gap: 12,
          fontSize: 11,
          fontFamily: "monospace",
          color: "#94a3b8",
          pointerEvents: "none",
        }}
      >
        <span>
          LATTICE: <strong style={{ color: "#38bdf8" }}>{graph.dimension}D</strong> ({graph.sites.length} sites, {graph.edges.length} bonds)
        </span>
        {selectedSite !== null && (
          <span style={{ color: "#4ade80" }}>
            SITE [{selectedSite}]: Coord [{points.get(selectedSite)?.coord.join(", ")}] · {neighborSet.size} neighbors
          </span>
        )}
      </div>

      <svg
        viewBox="0 0 540 280"
        role="img"
        aria-label={`${graph.dimension}D lattice with ${graph.sites.length} sites`}
        style={{ width: "100%", height: 280, display: "block" }}
      >
        {/* Background Grid Pattern */}
        <defs>
          <pattern id="latticeGrid" width="20" height="20" patternUnits="userSpaceOnUse">
            <line x1="0" y1="0" x2="20" y2="0" stroke="#1e293b" strokeWidth="0.5" strokeDasharray="1 3" />
            <line x1="0" y1="0" x2="0" y2="20" stroke="#1e293b" strokeWidth="0.5" strokeDasharray="1 3" />
          </pattern>
        </defs>
        <rect width="540" height="280" fill="url(#latticeGrid)" />

        {/* Coordinate Reference Axes in bottom-left */}
        <g transform="translate(24, 256)">
          {/* X axis */}
          <line x1="0" y1="0" x2="28" y2="0" stroke="#64748b" strokeWidth="1.2" />
          <polygon points="28,0 24,-2.5 24,2.5" fill="#64748b" />
          <text x="32" y="3" fill="#64748b" fontSize="9" fontFamily="monospace">x</text>

          {/* Y axis */}
          <line x1="0" y1="0" x2="0" y2="-28" stroke="#64748b" strokeWidth="1.2" />
          <polygon points="0,-28 -2.5,-24 2.5,-24" fill="#64748b" />
          <text x="-3" y="-32" fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="middle">y</text>
        </g>

        {/* Lattice Edges / Bonds */}
        {graph.edges.map((edge, idx) => {
          const source = points.get(edge.source);
          const target = points.get(edge.target);
          if (!source || !target) return null;

          const isConnectedToSelected =
            selectedSite !== null &&
            (edge.source === selectedSite || edge.target === selectedSite);

          return (
            <line
              key={`edge-${idx}-${edge.source}-${edge.target}`}
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              stroke={isConnectedToSelected ? "#38bdf8" : "#334155"}
              strokeWidth={isConnectedToSelected ? "2.2" : "1.4"}
              strokeLinecap="round"
            />
          );
        })}

        {/* Lattice Sites / Nodes */}
        {graph.sites.map((site) => {
          const point = points.get(site.index);
          if (!point) return null;

          const isSelected = selectedSite === site.index;
          const isNeighbor = neighborSet.has(site.index);

          return (
            <g
              key={`site-${site.index}`}
              onClick={() => setSelectedSite(isSelected ? null : site.index)}
              style={{ cursor: "pointer" }}
            >
              <title>{`Site ${site.index}: [${site.coordinate.join(", ")}]`}</title>
              {/* Outer selection ring */}
              {isSelected && (
                <circle
                  cx={point.x}
                  cy={point.y}
                  r="15"
                  fill="none"
                  stroke="#38bdf8"
                  strokeWidth="1.5"
                  strokeDasharray="2 2"
                />
              )}
              {/* Node Body */}
              <circle
                cx={point.x}
                cy={point.y}
                r="11"
                fill={isSelected ? "#38bdf8" : isNeighbor ? "#1e293b" : "#0f172a"}
                stroke={isSelected ? "#f8fafc" : isNeighbor ? "#4ade80" : "#475569"}
                strokeWidth="1.5"
              />
              {/* Node Index Number */}
              <text
                x={point.x}
                y={point.y + 3.5}
                textAnchor="middle"
                fontSize="9"
                fontWeight="700"
                fontFamily="monospace"
                fill={isSelected ? "#080d1a" : "#f1f5f9"}
              >
                {site.index}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export function EnergyChart({ values }: { values: number[] }) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (values.length < 2) {
    return (
      <div style={{ color: "#64748b", fontSize: 12, fontFamily: "monospace", padding: 12 }}>
        Execute real-time TEBD or DMRG sweeps to generate energy trajectory.
      </div>
    );
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(1e-12, max - min);

  // SVG Chart Dimensions
  const W = 540;
  const H = 220;
  const padLeft = 70;
  const padRight = 30;
  const padTop = 30;
  const padBottom = 35;
  const plotW = W - padLeft - padRight;
  const plotH = H - padTop - padBottom;

  // Coordinate transforms
  const getX = (idx: number) => padLeft + idx * (plotW / Math.max(1, values.length - 1));
  const getY = (val: number) => padTop + plotH - ((val - min) / span) * plotH;

  const points = values.map((val, idx) => `${getX(idx).toFixed(1)},${getY(val).toFixed(1)}`).join(" ");

  // 4 horizontal grid lines
  const gridLines = [0, 0.333, 0.666, 1].map((ratio) => {
    const val = min + ratio * span;
    const yPos = padTop + plotH - ratio * plotH;
    return { val, yPos };
  });

  const lastEnergy = values[values.length - 1];
  const deltaE = Math.abs(lastEnergy - values[0]);

  return (
    <div
      style={{
        borderRadius: 8,
        padding: "12px 14px",
        background: "#080d1a",
        border: "1px solid #334155",
      }}
    >
      {/* Chart Metadata Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8, fontSize: 11, fontFamily: "monospace", color: "#94a3b8" }}>
        <div>
          TRAJECTORY: <strong style={{ color: "#f8fafc" }}>E(t) Evolution</strong> ({values.length} data points)
        </div>
        <div style={{ display: "flex", gap: 12 }}>
          <span>E_min: <strong style={{ color: "#4ade80" }}>{min.toFixed(6)} eV</strong></span>
          <span>E_final: <strong style={{ color: "#38bdf8" }}>{lastEnergy.toFixed(6)} eV</strong></span>
          <span>ΔE: <strong style={{ color: "#cbd5e1" }}>{deltaE.toExponential(2)}</strong></span>
        </div>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", height: H, display: "block", overflow: "visible" }}
      >
        {/* Horizontal Grid Lines with Y-axis Labels */}
        {gridLines.map(({ val, yPos }, i) => (
          <g key={i}>
            <line x1={padLeft} y1={yPos} x2={W - padRight} y2={yPos} stroke="#1e293b" strokeWidth="1" strokeDasharray="2 3" />
            <text
              x={padLeft - 8}
              y={yPos + 3.5}
              textAnchor="end"
              fill="#64748b"
              fontSize="10"
              fontFamily="monospace"
            >
              {val.toFixed(4)}
            </text>
          </g>
        ))}

        {/* Asymptotic Ground State Limit Line (dashed green) */}
        <line
          x1={padLeft}
          y1={getY(min)}
          x2={W - padRight}
          y2={getY(min)}
          stroke="#10b981"
          strokeWidth="1.2"
          strokeDasharray="4 4"
          opacity="0.8"
        />

        {/* Data Polyline */}
        <polyline
          points={points}
          fill="none"
          stroke="#38bdf8"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Data Points */}
        {values.map((val, idx) => {
          const cx = getX(idx);
          const cy = getY(val);
          const isHovered = hoveredIdx === idx;
          return (
            <circle
              key={idx}
              cx={cx}
              cy={cy}
              r={isHovered ? 4.5 : 2.5}
              fill={isHovered ? "#38bdf8" : "#0f172a"}
              stroke="#38bdf8"
              strokeWidth="1.5"
              onMouseEnter={() => setHoveredIdx(idx)}
              onMouseLeave={() => setHoveredIdx(null)}
              style={{ cursor: "pointer" }}
            >
              <title>{`Step ${idx}: ${val.toFixed(6)} eV`}</title>
            </circle>
          );
        })}

        {/* X-axis ticks */}
        <line x1={padLeft} y1={padTop + plotH} x2={W - padRight} y2={padTop + plotH} stroke="#334155" strokeWidth="1" />
        <text x={padLeft} y={H - 10} fill="#64748b" fontSize="10" fontFamily="monospace">t = 0</text>
        <text x={padLeft + plotW / 2} y={H - 10} fill="#64748b" fontSize="10" fontFamily="monospace" textAnchor="middle">t_step</text>
        <text x={W - padRight} y={H - 10} fill="#64748b" fontSize="10" fontFamily="monospace" textAnchor="end">t = final</text>

        {/* Y-axis label */}
        <text x={padLeft} y={16} fill="#94a3b8" fontSize="10" fontFamily="monospace" fontWeight="600">E(t) [eV]</text>
      </svg>
    </div>
  );
}
