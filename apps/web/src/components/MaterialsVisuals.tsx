"use client";

import React, { useState } from "react";

// 1. ELECTRONIC BAND DISPERSION E(k) ALONG HIGH-SYMMETRY PATH
interface BandDispersionProps {
  hopping: number;
  hoppingPrime: number;
  onsiteU: number;
  chemicalPotential: number;
}

export function ElectronicBandDispersionChart({
  hopping,
  hoppingPrime,
  onsiteU,
  chemicalPotential,
}: BandDispersionProps) {
  const [hoveredK, setHoveredK] = useState<{ label: string; energy: number } | null>(null);

  const W = 520;
  const H = 180;
  const padL = 45;
  const padR = 25;
  const padT = 20;
  const padB = 30;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  // Bandwidth W = 8t
  const bandwidth = 8 * Math.abs(hopping || 1);
  const mottGap = Math.max(0, onsiteU - bandwidth * 0.75);
  const isMott = onsiteU > 4.5 * Math.abs(hopping || 1);

  // High-symmetry path: Gamma (0,0) -> X (pi, 0) -> M (pi, pi) -> Gamma (0,0)
  // Step 1: Gamma to X (kx in [0, pi], ky = 0)
  // Step 2: X to M (kx = pi, ky in [0, pi])
  // Step 3: M to Gamma (kx = ky in [pi, 0])
  const nSeg = 30;
  const pathPoints: { kx: number; ky: number; pathFrac: number; label?: string }[] = [];

  for (let i = 0; i <= nSeg; i++) {
    const f = i / nSeg;
    pathPoints.push({
      kx: f * Math.PI,
      ky: 0,
      pathFrac: f / 3,
      label: i === 0 ? "Γ (0,0)" : i === nSeg ? "X (π,0)" : undefined,
    });
  }
  for (let i = 1; i <= nSeg; i++) {
    const f = i / nSeg;
    pathPoints.push({
      kx: Math.PI,
      ky: f * Math.PI,
      pathFrac: 1 / 3 + f / 3,
      label: i === nSeg ? "M (π,π)" : undefined,
    });
  }
  for (let i = 1; i <= nSeg; i++) {
    const f = i / nSeg;
    pathPoints.push({
      kx: (1 - f) * Math.PI,
      ky: (1 - f) * Math.PI,
      pathFrac: 2 / 3 + f / 3,
      label: i === nSeg ? "Γ (0,0)" : undefined,
    });
  }

  // Energy scale min/max
  const maxE = 4 * Math.abs(hopping) + Math.abs(hoppingPrime) * 4 + onsiteU * 0.6 + 1.0;
  const minE = -maxE;
  const span = maxE - minE;

  const getY = (eVal: number) => padT + plotH - ((eVal - minE) / span) * plotH;
  const getX = (frac: number) => padL + frac * plotW;

  // Calculate dispersion points: E(k) = -2t (cos kx + cos ky) - 4t' cos kx cos ky - mu
  const upperBand: { x: number; y: number; e: number }[] = [];
  const lowerBand: { x: number; y: number; e: number }[] = [];

  pathPoints.forEach((p) => {
    const bareE =
      -2 * hopping * (Math.cos(p.kx) + Math.cos(p.ky)) -
      4 * hoppingPrime * Math.cos(p.kx) * Math.cos(p.ky) -
      chemicalPotential;

    const xPos = getX(p.pathFrac);

    if (isMott) {
      // Split into Lower and Upper Hubbard bands
      const lhbE = bareE - mottGap / 2;
      const uhbE = bareE + mottGap / 2;
      lowerBand.push({ x: xPos, y: getY(lhbE), e: lhbE });
      upperBand.push({ x: xPos, y: getY(uhbE), e: uhbE });
    } else {
      lowerBand.push({ x: xPos, y: getY(bareE), e: bareE });
    }
  });

  const lowerPoly = lowerBand.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const upperPoly = upperBand.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");

  const fermiY = getY(0);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, fontFamily: "monospace", color: "#94a3b8" }}>
        <span>ELECTRONIC BAND DISPERSION E(k) · HIGH-SYMMETRY PATH</span>
        {hoveredK ? (
          <span style={{ color: "#38bdf8", fontWeight: 700 }}>
            {hoveredK.label}: {hoveredK.energy >= 0 ? "+" : ""}{hoveredK.energy.toFixed(3)} eV
          </span>
        ) : (
          <span>
            {isMott ? `Mott Gap Δ_c ≈ ${mottGap.toFixed(2)} eV` : "Metallic Fermi Surface (Gapless)"}
          </span>
        )}
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H, display: "block" }}>
        {/* Fermi Level E_F = 0 (dashed line) */}
        <line x1={padL} y1={fermiY} x2={W - padR} y2={fermiY} stroke="#f59e0b" strokeWidth="1.2" strokeDasharray="4 3" />
        <text x={padL - 6} y={fermiY + 3.5} fill="#f59e0b" fontSize="9" fontFamily="monospace" textAnchor="end">E_F</text>

        {/* High-symmetry vertical dividers */}
        <line x1={getX(0)} y1={padT} x2={getX(0)} y2={padT + plotH} stroke="#334155" strokeWidth="1" />
        <line x1={getX(1 / 3)} y1={padT} x2={getX(1 / 3)} y2={padT + plotH} stroke="#334155" strokeWidth="1" strokeDasharray="2 2" />
        <line x1={getX(2 / 3)} y1={padT} x2={getX(2 / 3)} y2={padT + plotH} stroke="#334155" strokeWidth="1" strokeDasharray="2 2" />
        <line x1={getX(1)} y1={padT} x2={getX(1)} y2={padT + plotH} stroke="#334155" strokeWidth="1" />

        {/* Lower Band Polyline */}
        <polyline points={lowerPoly} fill="none" stroke="#38bdf8" strokeWidth="2.2" strokeLinecap="round" />

        {/* Upper Hubbard Band Polyline (if Mott) */}
        {isMott && upperPoly && (
          <polyline points={upperPoly} fill="none" stroke="#4ade80" strokeWidth="2.2" strokeLinecap="round" />
        )}

        {/* Data points for hover */}
        {lowerBand.filter((_, idx) => idx % 6 === 0).map((pt, idx) => (
          <circle
            key={`l-${idx}`}
            cx={pt.x}
            cy={pt.y}
            r="3"
            fill="#38bdf8"
            onMouseEnter={() => setHoveredK({ label: "Lower Band", energy: pt.e })}
            onMouseLeave={() => setHoveredK(null)}
            style={{ cursor: "pointer" }}
          />
        ))}

        {isMott &&
          upperBand.filter((_, idx) => idx % 6 === 0).map((pt, idx) => (
            <circle
              key={`u-${idx}`}
              cx={pt.x}
              cy={pt.y}
              r="3"
              fill="#4ade80"
              onMouseEnter={() => setHoveredK({ label: "Upper Hubbard Band", energy: pt.e })}
              onMouseLeave={() => setHoveredK(null)}
              style={{ cursor: "pointer" }}
            />
          ))}

        {/* Path Label Ticks */}
        <text x={getX(0)} y={H - 8} fill="#94a3b8" fontSize="10" fontFamily="monospace" textAnchor="middle">Γ</text>
        <text x={getX(1 / 3)} y={H - 8} fill="#94a3b8" fontSize="10" fontFamily="monospace" textAnchor="middle">X (π,0)</text>
        <text x={getX(2 / 3)} y={H - 8} fill="#94a3b8" fontSize="10" fontFamily="monospace" textAnchor="middle">M (π,π)</text>
        <text x={getX(1)} y={H - 8} fill="#94a3b8" fontSize="10" fontFamily="monospace" textAnchor="middle">Γ</text>

        {/* Y Axis Label */}
        <text x={padL} y={14} fill="#64748b" fontSize="9" fontFamily="monospace">E(k) [eV]</text>
      </svg>
    </div>
  );
}

// 2. DENSITY OF STATES D(E) CHART (DOS)
interface DensityOfStatesProps {
  hopping: number;
  onsiteU: number;
}

export function DensityOfStatesChart({ hopping, onsiteU }: DensityOfStatesProps) {
  const W = 520;
  const H = 150;
  const padL = 45;
  const padR = 25;
  const padT = 20;
  const padB = 25;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const isMott = onsiteU > 4.5 * Math.abs(hopping || 1);
  const bandwidth = 8 * Math.abs(hopping || 1);

  // Energy samples across [-bandwidth, bandwidth]
  const nPts = 60;
  const dosPoints: { e: number; dos: number }[] = [];
  const maxE = Math.max(bandwidth * 0.7, onsiteU * 0.7 + 2);

  for (let i = 0; i <= nPts; i++) {
    const e = -maxE + (i / nPts) * 2 * maxE;
    let val = 0;

    if (isMott) {
      // Split into two peaks: LHB at -U/2 and UHB at +U/2
      const centerL = -onsiteU / 2;
      const centerU = onsiteU / 2;
      const w = bandwidth * 0.35;
      const gL = Math.exp(-Math.pow((e - centerL) / w, 2));
      const gU = Math.exp(-Math.pow((e - centerU) / w, 2));
      val = gL + gU;
    } else {
      // 2D Square tight-binding DOS with Van Hove singularity at 0
      const absE = Math.abs(e);
      if (absE < bandwidth / 2) {
        val = Math.log(bandwidth / Math.max(0.08, absE)) * 0.5;
      } else {
        val = Math.max(0, 0.6 - (absE - bandwidth / 2) * 0.4);
      }
    }
    dosPoints.push({ e, dos: val });
  }

  const maxDos = Math.max(0.1, ...dosPoints.map((p) => p.dos));
  const polyStr = dosPoints
    .map((p) => {
      const px = padL + ((p.e + maxE) / (2 * maxE)) * plotW;
      const py = padT + plotH - (p.dos / maxDos) * plotH;
      return `${px.toFixed(1)},${py.toFixed(1)}`;
    })
    .join(" ");

  const zeroX = padL + (0.5 * plotW);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, fontFamily: "monospace", color: "#94a3b8" }}>
        <span>ELECTRONIC DENSITY OF STATES D(E) · SPECTRAL WEIGHT</span>
        <span>{isMott ? "Mott Gap at Fermi Level E=0" : "Van Hove Singularity at E=0"}</span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H, display: "block" }}>
        {/* Fermi level vertical line at E = 0 */}
        <line x1={zeroX} y1={padT} x2={zeroX} y2={padT + plotH} stroke="#f59e0b" strokeWidth="1.2" strokeDasharray="3 2" />
        <text x={zeroX} y={padT - 6} fill="#f59e0b" fontSize="9" fontFamily="monospace" textAnchor="middle">E_F = 0</text>

        {/* Axes */}
        <line x1={padL} y1={padT + plotH} x2={W - padR} y2={padT + plotH} stroke="#334155" strokeWidth="1" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke="#334155" strokeWidth="1" />

        {/* DOS Polyline */}
        <polyline points={polyStr} fill="none" stroke="#38bdf8" strokeWidth="2.2" strokeLinecap="round" />

        {/* Axis Ticks */}
        <text x={padL} y={H - 6} fill="#64748b" fontSize="9" fontFamily="monospace">-{maxE.toFixed(1)} eV</text>
        <text x={zeroX} y={H - 6} fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="middle">0.0 eV</text>
        <text x={W - padR} y={H - 6} fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="end">+{maxE.toFixed(1)} eV</text>
        <text x={padL} y={14} fill="#64748b" fontSize="9" fontFamily="monospace">D(E)</text>
      </svg>
    </div>
  );
}

// 3. CHARGE & DOUBLE OCCUPANCY LATTICE HEATMAP
interface ChargeOccupancyHeatmapProps {
  nSites: number;
  hopping: number;
  onsiteU: number;
  chemicalPotential: number;
}

export function ChargeOccupancyHeatmap({
  nSites,
  hopping,
  onsiteU,
  chemicalPotential,
}: ChargeOccupancyHeatmapProps) {
  const [selectedSite, setSelectedSite] = useState<number | null>(null);

  // Compute local densities n_up, n_down, and double occupancy d = <n_up n_down>
  const uOverT = Math.abs(hopping) > 1e-9 ? onsiteU / Math.abs(hopping) : 999;
  const isMott = uOverT > 4.5;

  const siteData = Array.from({ length: nSites }, (_, idx) => {
    // Alternating Néel spin density at half-filling
    const stagger = idx % 2 === 0 ? 1 : -1;
    const spinPol = isMott ? 0.38 * stagger : 0.08 * stagger;
    const nUp = 0.5 + spinPol + chemicalPotential * 0.05;
    const nDown = 0.5 - spinPol + chemicalPotential * 0.05;
    const doubleOcc = isMott ? Math.max(0.02, 0.25 / (1 + uOverT * 0.8)) : 0.22;
    return { idx, nUp, nDown, nTot: nUp + nDown, doubleOcc };
  });

  const cols = Math.min(6, nSites);
  const cellSize = Math.max(28, Math.min(48, Math.floor(320 / cols)));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, fontFamily: "monospace", color: "#94a3b8" }}>
        <span>LOCAL ELECTRON DENSITY & DOUBLE OCCUPANCY d_i = ⟨n_↑ n_↓⟩</span>
        {selectedSite !== null ? (
          <span style={{ color: "#38bdf8", fontWeight: 700 }}>
            Site {selectedSite}: n_↑={siteData[selectedSite]?.nUp.toFixed(3)}, n_↓={siteData[selectedSite]?.nDown.toFixed(3)}, d={siteData[selectedSite]?.doubleOcc.toFixed(3)}
          </span>
        ) : (
          <span>Select site to view orbital occupancies</span>
        )}
      </div>

      <div style={{ overflowX: "auto", paddingBottom: 4 }}>
        <div style={{ display: "inline-block", background: "#080d1a", border: "1px solid #334155", borderRadius: 6, padding: 10 }}>
          <div style={{ display: "grid", gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`, gap: 4 }}>
            {siteData.map((s) => {
              const isSel = selectedSite === s.idx;
              // Color based on double occupancy: green (low d, Mott localized) to amber (standard d ~ 0.25)
              const hue = isMott ? 195 : 140;
              const opacity = Math.min(1, Math.max(0.3, s.doubleOcc * 4));

              return (
                <div
                  key={s.idx}
                  onClick={() => setSelectedSite(isSel ? null : s.idx)}
                  style={{
                    width: cellSize,
                    height: cellSize,
                    borderRadius: 4,
                    background: `hsla(${hue}, 85%, 50%, ${opacity.toFixed(2)})`,
                    border: isSel ? "2px solid #ffffff" : "1px solid #334155",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    cursor: "pointer",
                    fontSize: 9,
                    fontFamily: "monospace",
                    fontWeight: 700,
                    color: "#f8fafc",
                    transition: "all 120ms ease",
                  }}
                  title={`Site ${s.idx}: n=${s.nTot.toFixed(2)}, d=${s.doubleOcc.toFixed(3)}`}
                >
                  <div>#{s.idx}</div>
                  <div style={{ fontSize: 8, opacity: 0.85 }}>d:{s.doubleOcc.toFixed(2)}</div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

// 4. DOUBLE OCCUPANCY VS CORRELATION RATIO CURVE (d vs U/t)
interface DoubleOccupancyCurveProps {
  hopping: number;
  onsiteU: number;
}

export function DoubleOccupancyVsCorrelationCurve({ hopping, onsiteU }: DoubleOccupancyCurveProps) {
  const W = 520;
  const H = 140;
  const padL = 45;
  const padR = 25;
  const padT = 20;
  const padB = 25;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const currentRatio = Math.abs(hopping) > 1e-9 ? onsiteU / Math.abs(hopping) : 999;
  const maxRatio = 14;

  // Theoretical Mott localization curve: d(U/t) = 0.25 / (1 + 0.35*(U/t)^1.4)
  const nPts = 40;
  const curvePoints: { r: number; d: number }[] = [];
  for (let i = 0; i <= nPts; i++) {
    const r = (i / nPts) * maxRatio;
    const dVal = 0.25 / (1 + 0.35 * Math.pow(r, 1.4));
    curvePoints.push({ r, d: dVal });
  }

  const polyStr = curvePoints
    .map((p) => {
      const px = padL + (p.r / maxRatio) * plotW;
      const py = padT + plotH - (p.d / 0.28) * plotH;
      return `${px.toFixed(1)},${py.toFixed(1)}`;
    })
    .join(" ");

  // Active operating point
  const clampedR = Math.min(maxRatio, currentRatio);
  const activeD = 0.25 / (1 + 0.35 * Math.pow(clampedR, 1.4));
  const activeX = padL + (clampedR / maxRatio) * plotW;
  const activeY = padT + plotH - (activeD / 0.28) * plotH;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, fontFamily: "monospace", color: "#94a3b8" }}>
        <span>MOTT SUPPRESSION OF DOUBLE OCCUPANCY ⟨D⟩ = ⟨n_↑ n_↓⟩ vs U/t</span>
        <span>
          Current Operating Point: <strong style={{ color: "#38bdf8" }}>U/t = {currentRatio.toFixed(2)}</strong> (⟨D⟩ ≈ {activeD.toFixed(3)})
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: H, display: "block" }}>
        {/* Axes */}
        <line x1={padL} y1={padT + plotH} x2={W - padR} y2={padT + plotH} stroke="#334155" strokeWidth="1" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + plotH} stroke="#334155" strokeWidth="1" />

        {/* Free Electron Limit (d = 0.25 dashed line) */}
        <line x1={padL} y1={padT + plotH - (0.25 / 0.28) * plotH} x2={W - padR} y2={padT + plotH - (0.25 / 0.28) * plotH} stroke="#64748b" strokeWidth="1" strokeDasharray="3 3" />
        <text x={padL - 6} y={padT + plotH - (0.25 / 0.28) * plotH + 3.5} fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="end">0.25</text>

        {/* Mott curve */}
        <polyline points={polyStr} fill="none" stroke="#38bdf8" strokeWidth="2.2" strokeLinecap="round" />

        {/* Active operating point indicator */}
        <circle cx={activeX} cy={activeY} r="5" fill="#f59e0b" stroke="#ffffff" strokeWidth="1.5" />
        <line x1={activeX} y1={activeY} x2={activeX} y2={padT + plotH} stroke="#f59e0b" strokeWidth="1" strokeDasharray="2 2" />

        {/* Ticks and labels */}
        <text x={padL} y={H - 6} fill="#64748b" fontSize="9" fontFamily="monospace">U/t = 0</text>
        <text x={padL + plotW / 2} y={H - 6} fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="middle">U/t = {(maxRatio / 2).toFixed(0)}</text>
        <text x={W - padR} y={H - 6} fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="end">U/t = {maxRatio}</text>
        <text x={padL} y={14} fill="#64748b" fontSize="9" fontFamily="monospace">⟨D⟩</text>
      </svg>
    </div>
  );
}
