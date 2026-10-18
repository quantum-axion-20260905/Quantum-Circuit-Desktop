// Local real-time quantum statevector & density matrix simulator
// Supports 1-qubit and 2-qubit gates, computes exact Bloch vectors (theta, phi, r) and phase amplitudes

import type { GateOp } from "../state/circuitStore";

export type Complex = { re: number; im: number };

export function cZero(): Complex {
  return { re: 0, im: 0 };
}

export function cOne(): Complex {
  return { re: 1, im: 0 };
}

export function cAdd(a: Complex, b: Complex): Complex {
  return { re: a.re + b.re, im: a.im + b.im };
}

export function cSub(a: Complex, b: Complex): Complex {
  return { re: a.re - b.re, im: a.im - b.im };
}

export function cMul(a: Complex, b: Complex): Complex {
  return {
    re: a.re * b.re - a.im * b.im,
    im: a.re * b.im + a.im * b.re,
  };
}

export function cMag2(a: Complex): number {
  return a.re * a.re + a.im * a.im;
}

export function cPhase(a: Complex): number {
  return Math.atan2(a.im, a.re);
}

export type BlochVector = {
  qubit: number;
  x: number;
  y: number;
  z: number;
  r: number;
  theta: number; // polar angle [0, pi]
  phi: number;   // azimuthal angle [-pi, pi]
  prob0: number;
  prob1: number;
};

export type AmplitudeInfo = {
  index: number;
  bitstring: string;
  amplitude: Complex;
  prob: number;
  phase: number; // in radians [-pi, pi]
};

export type SimState = {
  nQubits: number;
  statevector: Complex[];
  blochVectors: BlochVector[];
  amplitudes: AmplitudeInfo[];
};

// 2x2 unitary matrices for standard gates
type Matrix2x2 = [
  [Complex, Complex],
  [Complex, Complex]
];

function gateMatrix(name: string, theta?: number): Matrix2x2 {
  const INV_SQRT2 = 1 / Math.SQRT2;
  switch (name) {
    case "h":
      return [
        [{ re: INV_SQRT2, im: 0 }, { re: INV_SQRT2, im: 0 }],
        [{ re: INV_SQRT2, im: 0 }, { re: -INV_SQRT2, im: 0 }]
      ];
    case "x":
      return [
        [{ re: 0, im: 0 }, { re: 1, im: 0 }],
        [{ re: 1, im: 0 }, { re: 0, im: 0 }]
      ];
    case "y":
      return [
        [{ re: 0, im: 0 }, { re: 0, im: -1 }],
        [{ re: 0, im: 1 }, { re: 0, im: 0 }]
      ];
    case "z":
      return [
        [{ re: 1, im: 0 }, { re: 0, im: 0 }],
        [{ re: 0, im: 0 }, { re: -1, im: 0 }]
      ];
    case "s":
      return [
        [{ re: 1, im: 0 }, { re: 0, im: 0 }],
        [{ re: 0, im: 0 }, { re: 0, im: 1 }]
      ];
    case "t": {
      return [
        [{ re: 1, im: 0 }, { re: 0, im: 0 }],
        [{ re: 0, im: 0 }, { re: Math.cos(Math.PI / 4), im: Math.sin(Math.PI / 4) }]
      ];
    }
    case "rx": {
      const angle = theta ?? 0;
      const c = Math.cos(angle / 2);
      const s = Math.sin(angle / 2);
      return [
        [{ re: c, im: 0 }, { re: 0, im: -s }],
        [{ re: 0, im: -s }, { re: c, im: 0 }]
      ];
    }
    case "ry": {
      const angle = theta ?? 0;
      const c = Math.cos(angle / 2);
      const s = Math.sin(angle / 2);
      return [
        [{ re: c, im: 0 }, { re: -s, im: 0 }],
        [{ re: s, im: 0 }, { re: c, im: 0 }]
      ];
    }
    case "rz": {
      const angle = theta ?? 0;
      const c = Math.cos(angle / 2);
      const s = Math.sin(angle / 2);
      return [
        [{ re: c, im: -s }, { re: 0, im: 0 }],
        [{ re: 0, im: 0 }, { re: c, im: s }]
      ];
    }
    default:
      return [
        [{ re: 1, im: 0 }, { re: 0, im: 0 }],
        [{ re: 0, im: 0 }, { re: 1, im: 0 }]
      ];
  }
}

export function simulateCircuitLocal(nQubits: number, ops: GateOp[]): SimState {
  const boundedQubits = Math.max(1, Math.min(10, nQubits));
  const dim = 1 << boundedQubits;
  let state: Complex[] = new Array(dim);
  for (let i = 0; i < dim; i++) {
    state[i] = i === 0 ? cOne() : cZero();
  }

  // Sort operations column-by-column
  const sortedOps = [...ops]
    .filter((op) => op.target < boundedQubits && (op.control === undefined || op.control < boundedQubits))
    .sort((a, b) => a.col - b.col);

  for (const op of sortedOps) {
    const nextState = [...state];
    const t = op.target;
    const c = op.control;

    if (op.name === "cx" && c !== undefined && c !== t) {
      // CNOT
      const cBit = 1 << (boundedQubits - 1 - c);
      const tBit = 1 << (boundedQubits - 1 - t);
      for (let i = 0; i < dim; i++) {
        if ((i & cBit) !== 0 && (i & tBit) === 0) {
          const partner = i | tBit;
          const tmp = nextState[i];
          nextState[i] = nextState[partner];
          nextState[partner] = tmp;
        }
      }
    } else if (op.name === "cz" && c !== undefined && c !== t) {
      // CZ
      const cBit = 1 << (boundedQubits - 1 - c);
      const tBit = 1 << (boundedQubits - 1 - t);
      for (let i = 0; i < dim; i++) {
        if ((i & cBit) !== 0 && (i & tBit) !== 0) {
          nextState[i] = { re: -nextState[i].re, im: -nextState[i].im };
        }
      }
    } else {
      // Single-qubit unitary
      const U = gateMatrix(op.name, op.theta);
      const tBit = 1 << (boundedQubits - 1 - t);
      for (let i = 0; i < dim; i++) {
        if ((i & tBit) === 0) {
          const i0 = i;
          const i1 = i | tBit;
          const a0 = state[i0];
          const a1 = state[i1];

          // next[i0] = U00*a0 + U01*a1
          nextState[i0] = cAdd(cMul(U[0][0], a0), cMul(U[0][1], a1));
          // next[i1] = U10*a0 + U11*a1
          nextState[i1] = cAdd(cMul(U[1][0], a0), cMul(U[1][1], a1));
        }
      }
    }
    state = nextState;
  }

  // 1. Amplitudes info
  const amplitudes: AmplitudeInfo[] = [];
  for (let i = 0; i < dim; i++) {
    const amp = state[i];
    const prob = cMag2(amp);
    const phase = cPhase(amp);
    const bitstring = i.toString(2).padStart(boundedQubits, "0");
    amplitudes.push({ index: i, bitstring, amplitude: amp, prob, phase });
  }

  // 2. Compute reduced density matrix and Bloch vector for each qubit
  const blochVectors: BlochVector[] = [];
  for (let q = 0; q < boundedQubits; q++) {
    const qBit = 1 << (boundedQubits - 1 - q);
    let rho00 = 0;
    let rho11 = 0;
    let rho01_re = 0;
    let rho01_im = 0;

    for (let i = 0; i < dim; i++) {
      if ((i & qBit) === 0) {
        const i0 = i;
        const i1 = i | qBit;
        const a0 = state[i0];
        const a1 = state[i1];

        rho00 += cMag2(a0);
        rho11 += cMag2(a1);

        // a0 * conj(a1)
        rho01_re += a0.re * a1.re + a0.im * a1.im;
        rho01_im += a0.im * a1.re - a0.re * a1.im;
      }
    }

    const x = 2 * rho01_re;
    const y = -2 * rho01_im;
    const z = rho00 - rho11;
    const r = Math.min(1.0, Math.sqrt(x * x + y * y + z * z));

    // Polar theta [0, pi] and azimuthal phi [-pi, pi]
    const theta = r > 1e-7 ? Math.acos(Math.max(-1, Math.min(1, z / (r || 1)))) : 0;
    const phi = Math.atan2(y, x);

    blochVectors.push({
      qubit: q,
      x,
      y,
      z,
      r,
      theta,
      phi,
      prob0: rho00,
      prob1: rho11,
    });
  }

  return {
    nQubits: boundedQubits,
    statevector: state,
    blochVectors,
    amplitudes,
  };
}

export const simulateCircuit = simulateCircuitLocal;
