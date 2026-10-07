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
];
