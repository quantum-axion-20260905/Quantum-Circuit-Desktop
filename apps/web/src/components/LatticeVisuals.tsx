"use client";

import React from "react";
import type { LatticeGraph } from "../lib/agent";

function projectGraph(graph: LatticeGraph) {
  const raw = graph.sites.map((site) => {
    const [x = 0, y = 0, z = 0] = site.coordinate;
    return { index: site.index, x: x + z * 0.45, y: y - z * 0.32, depth: z };
  });
  const minX = Math.min(...raw.map((site) => site.x), 0);
  const maxX = Math.max(...raw.map((site) => site.x), 1);
  const minY = Math.min(...raw.map((site) => site.y), 0);
  const maxY = Math.max(...raw.map((site) => site.y), 1);
  const width = Math.max(1, maxX - minX);
  const height = Math.max(1, maxY - minY);
  return new Map(raw.map((site) => [site.index, {
    x: 34 + ((site.x - minX) / width) * 472,
    y: 34 + ((maxY - site.y) / height) * 210,
    depth: site.depth,
  }]));
}

export function LatticeCanvas({ graph }: { graph: LatticeGraph }) {
  const points = projectGraph(graph);
  const maxDepth = Math.max(1, ...graph.sites.map((site) => site.coordinate[2] ?? 0));
  return <div style={{ borderRadius: 14, overflow: "hidden", background: "linear-gradient(145deg,#0b1225,#111c36 55%,#172554)", border: "1px solid rgba(96,165,250,.25)" }}>
    <svg viewBox="0 0 540 280" role="img" aria-label={`${graph.dimension}D lattice with ${graph.sites.length} sites`} style={{ width: "100%", display: "block" }}>
      <defs>
        <linearGradient id="latticeLine" x1="0" x2="1"><stop stopColor="#38bdf8" stopOpacity=".2" /><stop offset="1" stopColor="#a78bfa" stopOpacity=".7" /></linearGradient>
        <radialGradient id="latticeNode"><stop stopColor="#f8fafc" /><stop offset=".35" stopColor="#67e8f9" /><stop offset="1" stopColor="#2563eb" /></radialGradient>
      </defs>
      <rect width="540" height="280" fill="transparent" />
      {graph.edges.map((edge) => {
        const source = points.get(edge.source); const target = points.get(edge.target);
        return source && target ? <line key={`${edge.source}-${edge.target}`} x1={source.x} y1={source.y} x2={target.x} y2={target.y} stroke="url(#latticeLine)" strokeWidth="2.5" /> : null;
      })}
      {graph.sites.map((site) => {
        const point = points.get(site.index);
        if (!point) return null;
        const color = point.depth / maxDepth;
        return <g key={site.index}><title>{`site ${site.index}: [${site.coordinate.join(", ")}]`}</title><circle cx={point.x} cy={point.y} r="11" fill="url(#latticeNode)" opacity={0.78 + color * 0.22} /><text x={point.x} y={point.y + 3.5} textAnchor="middle" fontSize="8" fontWeight="700" fill="#082f49">{site.index}</text></g>;
      })}
    </svg>
  </div>;
}

export function EnergyChart({ values }: { values: number[] }) {
  if (values.length < 2) return <div style={{ color: "#64748b", fontSize: 13 }}>Run TEBD to see the energy trajectory.</div>;
  const min = Math.min(...values); const max = Math.max(...values); const span = Math.max(1e-12, max - min);
  const points = values.map((value, index) => `${20 + index * (500 / Math.max(1, values.length - 1))},${220 - ((value - min) / span) * 180}`).join(" ");
  return <div style={{ borderRadius: 14, padding: "12px 12px 4px", background: "#0b1225", border: "1px solid rgba(148,163,184,.15)" }}><svg viewBox="0 0 540 250" style={{ width: "100%", display: "block" }}><line x1="20" y1="220" x2="520" y2="220" stroke="#334155" /><polyline points={points} fill="none" stroke="#67e8f9" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />{values.map((value, index) => <circle key={index} cx={20 + index * (500 / Math.max(1, values.length - 1))} cy={220 - ((value - min) / span) * 180} r="4" fill="#f0abfc" />)}<text x="20" y="242" fill="#64748b" fontSize="11">t = 0</text><text x="468" y="242" fill="#64748b" fontSize="11">t = final</text><text x="26" y="28" fill="#cbd5e1" fontSize="12">E(t)</text></svg></div>;
}
