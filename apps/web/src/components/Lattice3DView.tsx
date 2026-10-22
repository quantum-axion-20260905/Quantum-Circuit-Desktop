"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";

export type Site3D = {
  id: number;
  x: number;
  y: number;
  z?: number;
  label?: string;
  spin?: number; // e.g. +0.5, -0.5
  charge?: number; // e.g. 0.0 - 2.0
  color?: string;
  element?: string; // e.g. 'H', 'Li', 'C', 'O', 'Cu'
  radius?: number;
};

export type Bond3D = {
  source: number;
  target: number;
  weight?: number;
  type?: "single" | "double" | "triple" | "hopping";
  color?: string;
};

export type Lattice3DProps = {
  sites: Site3D[];
  bonds: Bond3D[];
  title?: string;
  subtitle?: string;
  height?: number;
  colorMode?: "spin" | "charge" | "element" | "monochrome";
  showLabels?: boolean;
};

// CPK elemental color table & atomic radii (relative)
const ELEMENT_COLORS: Record<string, { color: string; radius: number }> = {
  H: { color: "#f8fafc", radius: 10 },
  LI: { color: "#a855f7", radius: 16 },
  BE: { color: "#14b8a6", radius: 15 },
  B: { color: "#f97316", radius: 14 },
  C: { color: "#64748b", radius: 14 },
  N: { color: "#3b82f6", radius: 13 },
  O: { color: "#ef4444", radius: 13 },
  F: { color: "#22c55e", radius: 12 },
  CU: { color: "#eab308", radius: 18 },
  RU: { color: "#06b6d4", radius: 17 },
  FE: { color: "#f59e0b", radius: 17 },
  V: { color: "#ec4899", radius: 17 },
  CE: { color: "#8b5cf6", radius: 18 },
  DEFAULT: { color: "#38bdf8", radius: 14 },
};

export function Lattice3DView({
  sites,
  bonds,
  title = "3D Crystal / Molecular Geometry",
  subtitle = "Interactive spatial projection · Click & drag to rotate · Scroll to zoom",
  height = 360,
  colorMode = "element",
  showLabels = true,
}: Lattice3DProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Rotation angles (radians)
  const [rotX, setRotX] = useState<number>(0.4);
  const [rotY, setRotY] = useState<number>(-0.6);
  const [zoom, setZoom] = useState<number>(1.0);
  const [autoRotate, setAutoRotate] = useState<boolean>(false);
  const [hoveredSite, setHoveredSite] = useState<Site3D | null>(null);

  const [isDragging, setIsDragging] = useState(false);
  const isDraggingRef = useRef(false);
  const lastMouseRef = useRef({ x: 0, y: 0 });

  // Reset to isometric
  const handleReset = useCallback(() => {
    setRotX(0.4);
    setRotY(-0.6);
    setZoom(1.0);
  }, []);

  // Top view
  const handleTopView = useCallback(() => {
    setRotX(Math.PI / 2);
    setRotY(0);
  }, []);

  // Front view
  const handleFrontView = useCallback(() => {
    setRotX(0);
    setRotY(0);
  }, []);

  // Mouse interaction handlers
  const onMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    isDraggingRef.current = true;
    lastMouseRef.current = { x: e.clientX, y: e.clientY };
  };

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDraggingRef.current) {
      const dx = e.clientX - lastMouseRef.current.x;
      const dy = e.clientY - lastMouseRef.current.y;
      setRotY((prev) => prev + dx * 0.008);
      setRotX((prev) => Math.max(-Math.PI / 2, Math.min(Math.PI / 2, prev + dy * 0.008)));
      lastMouseRef.current = { x: e.clientX, y: e.clientY };
    } else {
      // Hit testing for hover
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const hit = findHoveredSite(mouseX, mouseY, canvas.width, canvas.height);
      setHoveredSite(hit);
    }
  };

  const onMouseUp = () => {
    setIsDragging(false);
    isDraggingRef.current = false;
  };

  const onWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.92 : 1.08;
    setZoom((prev) => Math.max(0.4, Math.min(3.5, prev * factor)));
  };

  // Compute normalized bounding box of sites
  const { normalizedSites } = React.useMemo(() => {
    if (sites.length === 0) return { normalizedSites: [] };

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    sites.forEach((s) => {
      const z = s.z ?? 0;
      if (s.x < minX) minX = s.x;
      if (s.x > maxX) maxX = s.x;
      if (s.y < minY) minY = s.y;
      if (s.y > maxY) maxY = s.y;
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    });

    const spanX = Math.max(0.001, maxX - minX);
    const spanY = Math.max(0.001, maxY - minY);
    const spanZ = Math.max(0.001, maxZ - minZ);
    const maxSpan = Math.max(spanX, spanY, spanZ, 1);

    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const cz = (minZ + maxZ) / 2;

    const norm = sites.map((s) => ({
      ...s,
      nx: ((s.x - cx) / maxSpan) * 160,
      ny: ((s.y - cy) / maxSpan) * 160,
      nz: (((s.z ?? 0) - cz) / maxSpan) * 160,
    }));

    return { normalizedSites: norm };
  }, [sites]);

  // Hit-testing helper
  const findHoveredSite = (
    mx: number,
    my: number,
    w: number,
    h: number
  ): Site3D | null => {
    const cx = w / 2;
    const cy = h / 2;
    const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
    const cosY = Math.cos(rotY), sinY = Math.sin(rotY);

    for (let i = normalizedSites.length - 1; i >= 0; i--) {
      const s = normalizedSites[i];
      // 3D rotation
      const x1 = s.nx * cosY - s.nz * sinY;
      const z1 = s.nx * sinY + s.nz * cosY;
      const y2 = s.ny * cosX - z1 * sinX;
      const z2 = s.ny * sinX + z1 * cosX;

      const fov = 350;
      const scale = (fov / (fov + z2)) * zoom;
      const screenX = cx + x1 * scale;
      const screenY = cy + y2 * scale;
      const radius = getSiteRadius(s) * scale;

      const dist = Math.hypot(mx - screenX, my - screenY);
      if (dist <= Math.max(radius, 12)) {
        return sites[s.id] ?? null;
      }
    }
    return null;
  };

  // Determine site display color
  const getSiteColor = useCallback((s: Site3D): string => {
    if (s.color) return s.color;
    if (colorMode === "element" && s.element) {
      const el = s.element.toUpperCase();
      return ELEMENT_COLORS[el]?.color ?? ELEMENT_COLORS.DEFAULT.color;
    }
    if (colorMode === "spin" && s.spin != null) {
      return s.spin > 0 ? "#3b82f6" : "#ef4444"; // Blue up, Red down
    }
    if (colorMode === "charge" && s.charge != null) {
      // Heatmap 0 -> 2
      const c = Math.max(0, Math.min(2, s.charge));
      if (c <= 1) {
        const t = c;
        return `rgb(${Math.round(20 + 200 * t)}, ${Math.round(40 + 120 * t)}, 240)`;
      } else {
        const t = c - 1;
        return `rgb(240, ${Math.round(160 * (1 - t))}, ${Math.round(40 * (1 - t))})`;
      }
    }
    return "#38bdf8";
  }, [colorMode]);

  const getSiteRadius = useCallback((s: Site3D): number => {
    if (s.radius) return s.radius;
    if (colorMode === "element" && s.element) {
      const el = s.element.toUpperCase();
      return ELEMENT_COLORS[el]?.radius ?? 14;
    }
    return 14;
  }, [colorMode]);

  // Auto rotation animation
  useEffect(() => {
    if (!autoRotate) return;
    const interval = setInterval(() => {
      setRotY((prev) => prev + 0.012);
    }, 25);
    return () => clearInterval(interval);
  }, [autoRotate]);

  // Main 3D Rendering loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const cx = width / 2;
    const cy = height / 2;

    ctx.clearRect(0, 0, width, height);

    // Dark grid background
    ctx.fillStyle = "#090d16";
    ctx.fillRect(0, 0, width, height);

    // Subtle 3D coordinate floor grid
    ctx.strokeStyle = "rgba(51, 65, 85, 0.35)";
    ctx.lineWidth = 1;

    const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
    const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
    const fov = 350;

    // Helper: project 3D point to 2D screen
    const project = (x: number, y: number, z: number) => {
      const x1 = x * cosY - z * sinY;
      const z1 = x * sinY + z * cosY;
      const y2 = y * cosX - z1 * sinX;
      const z2 = y * sinX + z1 * cosX;
      const scale = (fov / (fov + z2)) * zoom;
      return {
        screenX: cx + x1 * scale,
        screenY: cy + y2 * scale,
        depth: z2,
        scale,
      };
    };

    // Draw floor grid square
    const floorY = 90;
    const gridSize = 120;
    const gridStep = 40;
    ctx.beginPath();
    for (let gx = -gridSize; gx <= gridSize; gx += gridStep) {
      const p1 = project(gx, floorY, -gridSize);
      const p2 = project(gx, floorY, gridSize);
      ctx.moveTo(p1.screenX, p1.screenY);
      ctx.lineTo(p2.screenX, p2.screenY);
    }
    for (let gz = -gridSize; gz <= gridSize; gz += gridStep) {
      const p1 = project(-gridSize, floorY, gz);
      const p2 = project(gridSize, floorY, gz);
      ctx.moveTo(p1.screenX, p1.screenY);
      ctx.lineTo(p2.screenX, p2.screenY);
    }
    ctx.stroke();

    // Projected sites
    const projectedSites = normalizedSites.map((s) => {
      const p = project(s.nx, s.ny, s.nz);
      return {
        ...s,
        ...p,
        baseColor: getSiteColor(s),
        baseRadius: getSiteRadius(s),
      };
    });

    // Create render list of items (bonds + spheres) sorted by depth for Painter's algorithm
    type RenderItem =
      | { type: "bond"; depth: number; b: Bond3D; p1: (typeof projectedSites)[0]; p2: (typeof projectedSites)[0] }
      | { type: "site"; depth: number; s: (typeof projectedSites)[0] };

    const renderItems: RenderItem[] = [];

    // Map bonds
    bonds.forEach((b) => {
      const p1 = projectedSites.find((ps) => ps.id === b.source);
      const p2 = projectedSites.find((ps) => ps.id === b.target);
      if (p1 && p2) {
        renderItems.push({
          type: "bond",
          depth: (p1.depth + p2.depth) / 2,
          b,
          p1,
          p2,
        });
      }
    });

    // Map sites
    projectedSites.forEach((s) => {
      renderItems.push({
        type: "site",
        depth: s.depth,
        s,
      });
    });

    // Sort ascending: deepest (largest z2) rendered first
    renderItems.sort((a, b) => b.depth - a.depth);

    // Render sorted elements
    renderItems.forEach((item) => {
      if (item.type === "bond") {
        const { p1, p2, b } = item;
        const bondWidth = Math.max(2, 3.5 * ((p1.scale + p2.scale) / 2));

        // Depth-dependent bond lighting
        const alpha = Math.max(0.2, Math.min(0.9, 0.6 + (item.depth < 0 ? 0.3 : -0.2)));
        ctx.strokeStyle = b.color ?? `rgba(148, 163, 184, ${alpha})`;
        ctx.lineWidth = bondWidth;
        ctx.beginPath();
        ctx.moveTo(p1.screenX, p1.screenY);
        ctx.lineTo(p2.screenX, p2.screenY);
        ctx.stroke();

        // If double bond, draw secondary parallel line
        if (b.type === "double") {
          const dx = p2.screenX - p1.screenX;
          const dy = p2.screenY - p1.screenY;
          const len = Math.hypot(dx, dy) || 1;
          const nx = (-dy / len) * 4;
          const ny = (dx / len) * 4;
          ctx.beginPath();
          ctx.moveTo(p1.screenX + nx, p1.screenY + ny);
          ctx.lineTo(p2.screenX + nx, p2.screenY + ny);
          ctx.stroke();
        }
      } else {
        const s = item.s;
        const radius = Math.max(4, s.baseRadius * s.scale);
        const isHovered = hoveredSite?.id === s.id;

        // Radial lighting gradient for realistic 3D sphere illusion
        const grad = ctx.createRadialGradient(
          s.screenX - radius * 0.35,
          s.screenY - radius * 0.35,
          radius * 0.1,
          s.screenX,
          s.screenY,
          radius
        );
        grad.addColorStop(0, "#ffffff");
        grad.addColorStop(0.3, s.baseColor);
        grad.addColorStop(1, "#030712");

        ctx.beginPath();
        ctx.arc(s.screenX, s.screenY, radius, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();

        // Shading / rim stroke
        ctx.strokeStyle = isHovered ? "#38bdf8" : "rgba(255, 255, 255, 0.25)";
        ctx.lineWidth = isHovered ? 2.5 : 1;
        ctx.stroke();

        // Optional Atom Label / Spin symbol
        if (showLabels) {
          ctx.font = `${Math.max(9, Math.round(11 * s.scale))}px monospace`;
          ctx.fillStyle = "#ffffff";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";

          let labelText = s.element ?? (s.label ? s.label : `S${s.id}`);
          if (colorMode === "spin" && s.spin != null) {
            labelText = s.spin > 0 ? "↑" : "↓";
          }
          ctx.fillText(labelText, s.screenX, s.screenY);
        }
      }
    });

    // 3D Axis Compass in bottom-left
    const axisLen = 28;
    const axOrigin = { x: 42, y: height - 42 };
    const axes = [
      { dx: 1, dy: 0, dz: 0, color: "#ef4444", label: "X" },
      { dx: 0, dy: 1, dz: 0, color: "#22c55e", label: "Y" },
      { dx: 0, dy: 0, dz: 1, color: "#3b82f6", label: "Z" },
    ];
    axes.forEach((axis) => {
      const p = project(axis.dx * axisLen, axis.dy * axisLen, axis.dz * axisLen);
      const px = axOrigin.x + (p.screenX - cx);
      const py = axOrigin.y + (p.screenY - cy);
      ctx.strokeStyle = axis.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(axOrigin.x, axOrigin.y);
      ctx.lineTo(px, py);
      ctx.stroke();

      ctx.fillStyle = axis.color;
      ctx.font = "10px sans-serif";
      ctx.fillText(axis.label, px + 4, py + 2);
    });
  }, [
    normalizedSites,
    bonds,
    rotX,
    rotY,
    zoom,
    colorMode,
    showLabels,
    hoveredSite,
    getSiteColor,
    getSiteRadius,
  ]);

  return (
    <div
      style={{
        border: "1px solid #334155",
        borderRadius: 8,
        background: "#0b0f19",
        overflow: "hidden",
        position: "relative",
      }}
    >
      {/* Top Header & Toolbar */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "10px 14px",
          background: "#111827",
          borderBottom: "1px solid #1e293b",
          flexWrap: "wrap",
          gap: 8,
        }}
      >
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color: "#f8fafc" }}>
            {title}
          </div>
          <div style={{ fontSize: 11, color: "#94a3b8" }}>{subtitle}</div>
        </div>

        {/* View Controls Toolbar */}
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <button
            type="button"
            onClick={() => setAutoRotate(!autoRotate)}
            style={{
              padding: "4px 8px",
              fontSize: 11,
              fontWeight: 600,
              borderRadius: 4,
              border: "1px solid #475569",
              background: autoRotate ? "#0284c7" : "#1e293b",
              color: "#f8fafc",
              cursor: "pointer",
            }}
          >
            {autoRotate ? "Pause Spin" : "Auto Spin"}
          </button>
          <button
            type="button"
            onClick={handleFrontView}
            style={{
              padding: "4px 8px",
              fontSize: 11,
              fontWeight: 500,
              borderRadius: 4,
              border: "1px solid #475569",
              background: "#1e293b",
              color: "#cbd5e1",
              cursor: "pointer",
            }}
          >
            XY Front
          </button>
          <button
            type="button"
            onClick={handleTopView}
            style={{
              padding: "4px 8px",
              fontSize: 11,
              fontWeight: 500,
              borderRadius: 4,
              border: "1px solid #475569",
              background: "#1e293b",
              color: "#cbd5e1",
              cursor: "pointer",
            }}
          >
            XZ Top
          </button>
          <button
            type="button"
            onClick={handleReset}
            style={{
              padding: "4px 8px",
              fontSize: 11,
              fontWeight: 500,
              borderRadius: 4,
              border: "1px solid #475569",
              background: "#1e293b",
              color: "#cbd5e1",
              cursor: "pointer",
            }}
          >
            Reset 3D
          </button>
        </div>
      </div>

      {/* Main 3D Canvas */}
      <canvas
        ref={canvasRef}
        width={720}
        height={height}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
        onWheel={onWheel}
        style={{
          width: "100%",
          height: `${height}px`,
          display: "block",
          cursor: isDragging ? "grabbing" : "grab",
        }}
      />

      {/* Inspection Footer Tooltip */}
      <div
        style={{
          padding: "6px 12px",
          background: "#090d16",
          borderTop: "1px solid #1e293b",
          fontSize: 11,
          color: "#94a3b8",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          {hoveredSite ? (
            <span style={{ color: "#38bdf8", fontWeight: 600 }}>
              Inspecting site #{hoveredSite.id} [{hoveredSite.element ?? `S${hoveredSite.id}`}] · Pos: (
              {hoveredSite.x.toFixed(2)}, {hoveredSite.y.toFixed(2)}
              {hoveredSite.z != null ? `, ${hoveredSite.z.toFixed(2)}` : ""})
              {hoveredSite.spin != null ? ` · Spin Sz: ${hoveredSite.spin > 0 ? "+1/2" : "-1/2"}` : ""}
              {hoveredSite.charge != null ? ` · Density n: ${hoveredSite.charge.toFixed(3)}` : ""}
            </span>
          ) : (
            <span>Hover over any site sphere to inspect local quantum coordinates.</span>
          )}
        </div>
        <div>
          Zoom: {(zoom * 100).toFixed(0)}% · Sites: {sites.length} · Bonds: {bonds.length}
        </div>
      </div>
    </div>
  );
}
