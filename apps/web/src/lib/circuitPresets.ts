import type { GateOp } from "../state/circuitStore";

export type PresetCircuit = {
  id: string;
  name: string;
  description: string;
  nQubits: number;
  ops: GateOp[];
};

export const CIRCUIT_PRESETS: PresetCircuit[] = [
  {
    id: "bell-state",
    name: "Bell State (|Φ⁺⟩)",
    description: "2-qubit maximally entangled Bell state: |00⟩ + |11⟩",
    nQubits: 2,
    ops: [
      { id: "p-h0", name: "h", target: 0, col: 0, x: 120, y: 60 },
      { id: "p-cx01", name: "cx", control: 0, target: 1, col: 1, x: 260, y: 120 },
    ],
  },
  {
    id: "ghz-state",
    name: "GHZ State (3-qubit)",
    description: "3-qubit Greenberger–Horne–Zeilinger state: |000⟩ + |111⟩",
    nQubits: 3,
    ops: [
      { id: "ghz-h0", name: "h", target: 0, col: 0, x: 120, y: 60 },
      { id: "ghz-cx01", name: "cx", control: 0, target: 1, col: 1, x: 260, y: 120 },
      { id: "ghz-cx12", name: "cx", control: 1, target: 2, col: 2, x: 400, y: 180 },
    ],
  },
  {
    id: "superposition",
    name: "Full Superposition (4-qubit)",
    description: "Hadamard gate on all 4 qubits creates uniform 2^4 = 16 states",
    nQubits: 4,
    ops: [
      { id: "sup-h0", name: "h", target: 0, col: 0, x: 120, y: 60 },
      { id: "sup-h1", name: "h", target: 1, col: 0, x: 120, y: 120 },
      { id: "sup-h2", name: "h", target: 2, col: 0, x: 120, y: 180 },
      { id: "sup-h3", name: "h", target: 3, col: 0, x: 120, y: 240 },
    ],
  },
  {
    id: "quantum-teleportation",
    name: "Quantum Teleportation",
    description: "Teleport an arbitrary quantum state using an entangled pair and classical Bell measurement",
    nQubits: 3,
    ops: [
      { id: "tel-init", name: "rx", target: 0, theta: 1.234, col: 0, x: 120, y: 60 },
      { id: "tel-h1", name: "h", target: 1, col: 1, x: 260, y: 120 },
      { id: "tel-cx12", name: "cx", control: 1, target: 2, col: 2, x: 400, y: 180 },
      { id: "tel-cx01", name: "cx", control: 0, target: 1, col: 3, x: 540, y: 120 },
      { id: "tel-h0", name: "h", target: 0, col: 4, x: 680, y: 60 },
      { id: "tel-cz02", name: "cz", control: 0, target: 2, col: 5, x: 820, y: 180 },
      { id: "tel-cx12b", name: "cx", control: 1, target: 2, col: 6, x: 960, y: 180 },
    ],
  },
  {
    id: "param-ansatz",
    name: "VQE Parameter Ansatz",
    description: "Parameterized rotations with entangling gates for variational studies",
    nQubits: 2,
    ops: [
      { id: "vqe-ry0", name: "ry", target: 0, parameter: "theta", col: 0, x: 120, y: 60 },
      { id: "vqe-ry1", name: "ry", target: 1, parameter: "phi", col: 0, x: 120, y: 120 },
      { id: "vqe-cx01", name: "cx", control: 0, target: 1, col: 1, x: 260, y: 120 },
      { id: "vqe-rz1", name: "rz", target: 1, theta: 0.785, col: 2, x: 400, y: 120 },
    ],
  },
  {
    id: "grover-2q",
    name: "Grover's Search (2-qubit)",
    description: "Amplifies target state |11⟩ to 100% using phase inversion oracle and diffusion",
    nQubits: 2,
    ops: [
      // 1. Equal Superposition
      { id: "grv-h0", name: "h", target: 0, col: 0, x: 120, y: 60 },
      { id: "grv-h1", name: "h", target: 1, col: 0, x: 120, y: 120 },
      // 2. Oracle for |11> (Controlled-Z flips phase of |11>)
      { id: "grv-cz", name: "cz", control: 0, target: 1, col: 1, x: 260, y: 120 },
      // 3. Diffusion: H -> X -> CZ -> X -> H
      { id: "grv-dh0", name: "h", target: 0, col: 2, x: 400, y: 60 },
      { id: "grv-dh1", name: "h", target: 1, col: 2, x: 400, y: 120 },
      { id: "grv-dx0", name: "x", target: 0, col: 3, x: 540, y: 60 },
      { id: "grv-dx1", name: "x", target: 1, col: 3, x: 540, y: 120 },
      { id: "grv-dcz", name: "cz", control: 0, target: 1, col: 4, x: 680, y: 120 },
      { id: "grv-dx2", name: "x", target: 0, col: 5, x: 820, y: 60 },
      { id: "grv-dx3", name: "x", target: 1, col: 5, x: 820, y: 120 },
      { id: "grv-dh2", name: "h", target: 0, col: 6, x: 960, y: 60 },
      { id: "grv-dh3", name: "h", target: 1, col: 6, x: 960, y: 120 },
    ],
  },
  {
    id: "qft-3q",
    name: "Quantum Fourier Transform (3-qubit)",
    description: "Discrete Fourier Transform on 3 qubits with controlled phase rotations and SWAP",
    nQubits: 3,
    ops: [
      { id: "qft-h0", name: "h", target: 0, col: 0, x: 120, y: 60 },
      { id: "qft-s01", name: "s", target: 1, col: 1, x: 260, y: 120 },
      { id: "qft-cz01", name: "cz", control: 0, target: 1, col: 2, x: 400, y: 120 },
      { id: "qft-t02", name: "t", target: 2, col: 3, x: 540, y: 180 },
      { id: "qft-cz02", name: "cz", control: 0, target: 2, col: 4, x: 680, y: 180 },
      { id: "qft-h1", name: "h", target: 1, col: 5, x: 820, y: 120 },
      { id: "qft-s12", name: "s", target: 2, col: 6, x: 960, y: 180 },
      { id: "qft-cz12", name: "cz", control: 1, target: 2, col: 7, x: 1100, y: 180 },
      { id: "qft-h2", name: "h", target: 2, col: 8, x: 1240, y: 180 },
      // SWAP q0 and q2
      { id: "qft-sw1", name: "cx", control: 0, target: 2, col: 9, x: 1380, y: 180 },
      { id: "qft-sw2", name: "cx", control: 2, target: 0, col: 10, x: 1520, y: 60 },
      { id: "qft-sw3", name: "cx", control: 0, target: 2, col: 11, x: 1660, y: 180 },
    ],
  },
  {
    id: "deutsch-jozsa",
    name: "Deutsch–Jozsa Algorithm (3-qubit)",
    description: "Determines if a quantum oracle is constant or balanced in a single evaluation",
    nQubits: 3,
    ops: [
      // Ancilla initialization to |1>
      { id: "dj-x2", name: "x", target: 2, col: 0, x: 120, y: 180 },
      // Hadamard on all 3 qubits
      { id: "dj-h0", name: "h", target: 0, col: 1, x: 260, y: 60 },
      { id: "dj-h1", name: "h", target: 1, col: 1, x: 260, y: 120 },
      { id: "dj-h2", name: "h", target: 2, col: 1, x: 260, y: 180 },
      // Balanced Oracle: CNOTs targeting ancilla q2
      { id: "dj-cx02", name: "cx", control: 0, target: 2, col: 2, x: 400, y: 180 },
      { id: "dj-cx12", name: "cx", control: 1, target: 2, col: 3, x: 540, y: 180 },
      // Hadamard interference on input qubits
      { id: "dj-fh0", name: "h", target: 0, col: 4, x: 680, y: 60 },
      { id: "dj-fh1", name: "h", target: 1, col: 4, x: 680, y: 120 },
    ],
  },
  {
    id: "w-state",
    name: "W-State (|W⟩ 3-qubit)",
    description: "3-qubit entangled state: 1/√3 (|001⟩ + |010⟩ + |100⟩) with maximum entanglement robustness",
    nQubits: 3,
    ops: [
      { id: "w-ry0", name: "ry", target: 0, theta: 1.911, col: 0, x: 120, y: 60 },
      { id: "w-cx01", name: "cx", control: 0, target: 1, col: 1, x: 260, y: 120 },
      { id: "w-h1", name: "h", target: 1, col: 2, x: 400, y: 120 },
      { id: "w-cx12", name: "cx", control: 1, target: 2, col: 3, x: 540, y: 180 },
      { id: "w-x0", name: "x", target: 0, col: 4, x: 680, y: 60 },
    ],
  },
];
