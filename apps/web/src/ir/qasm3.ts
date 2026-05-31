import type { GateName, IrNode } from "./ir";

function fmtFloat(x: number) {
  // Keep output stable for versioning; avoid scientific for common values.
  if (Number.isInteger(x)) return `${x}.0`;
  const s = x.toPrecision(12);
  return s.replace(/\.?0+$/, "");
}

function requireTheta(name: GateName, node: IrNode) {
  if (name === "rx" || name === "ry" || name === "rz") {
    if (typeof node.theta !== "number") throw new Error(`${name} requires theta`);
    return node.theta;
  }
  return undefined;
}

export function emitQasm3(nQubits: number, nodes: IrNode[]) {
  const lines: string[] = [];
  lines.push("OPENQASM 3;");
  lines.push('include "stdgates.inc";');
  lines.push(`qubit[${nQubits}] q;`);
  lines.push("");

  const ops = [...nodes].sort((a, b) => (a.col - b.col) || (a.y - b.y) || (a.x - b.x));
  for (const op of ops) {
    const name = op.name;
    if (name === "h" || name === "x") {
      lines.push(`${name} q[${op.target}];`);
      continue;
    }
    if (name === "rx" || name === "ry" || name === "rz") {
      const theta = requireTheta(name, op)!;
      lines.push(`${name}(${fmtFloat(theta)}) q[${op.target}];`);
      continue;
    }
    if (name === "cx" || name === "cz") {
      if (typeof op.control !== "number") throw new Error(`${name} requires control`);
      lines.push(`${name} q[${op.control}], q[${op.target}];`);
      continue;
    }
    // exhaustive guard
    const _never: never = name;
    throw new Error(`Unsupported gate: ${_never}`);
  }

  return lines.join("\n");
}

export function parseMinimalQasm3(qasm: string): { nQubits: number; ops: Array<{ name: GateName; target: number; control?: number; theta?: number }> } {
  // Very small parser for our own emitter output; not a full OpenQASM3 parser.
  // If parsing fails, caller should fall back to ui metadata.
  const lines = qasm
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("//"));

  let nQubits = 0;
  for (const line of lines) {
    const m = line.match(/^qubit\[(\d+)\]\s+q\s*;\s*$/);
    if (m) nQubits = Number(m[1]);
  }
  if (!nQubits) throw new Error("Failed to find qubit register declaration");

  const ops: Array<{ name: GateName; target: number; control?: number; theta?: number }> = [];
  for (const line of lines) {
    let m = line.match(/^(h|x)\s+q\[(\d+)\]\s*;\s*$/);
    if (m) {
      ops.push({ name: m[1] as GateName, target: Number(m[2]) });
      continue;
    }
    m = line.match(/^(rx|ry|rz)\(([-+0-9.eE]+)\)\s+q\[(\d+)\]\s*;\s*$/);
    if (m) {
      ops.push({ name: m[1] as GateName, theta: Number(m[2]), target: Number(m[3]) });
      continue;
    }
    m = line.match(/^(cx|cz)\s+q\[(\d+)\]\s*,\s*q\[(\d+)\]\s*;\s*$/);
    if (m) {
      ops.push({ name: m[1] as GateName, control: Number(m[2]), target: Number(m[3]) });
      continue;
    }
  }
  return { nQubits, ops };
}
