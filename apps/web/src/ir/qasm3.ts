import type { GateName, IrNode } from "./ir";

function fmtFloat(x: number) {
  // Keep output stable for versioning; avoid scientific for common values.
  if (Number.isInteger(x)) return `${x}.0`;
  const s = x.toPrecision(12);
  return s.replace(/\.?0+$/, "");
}

function requireAngle(name: GateName, node: IrNode) {
  if (name === "rx" || name === "ry" || name === "rz") {
    if (typeof node.theta === "number") return fmtFloat(node.theta);
    if (typeof node.parameter === "string") return node.parameter;
    throw new Error(`${name} requires theta or parameter`);
  }
  return undefined;
}

export function emitQasm3(nQubits: number, nodes: IrNode[]) {
  const lines: string[] = [];
  lines.push("OPENQASM 3;");
  lines.push('include "stdgates.inc";');
  lines.push(`qubit[${nQubits}] q;`);
  const parameters = [...new Set(nodes.map((node) => node.parameter).filter((value): value is string => Boolean(value)))];
  for (const parameter of parameters) lines.push(`input float ${parameter};`);
  lines.push("");

  const ops = [...nodes].sort((a, b) => (a.col - b.col) || (a.y - b.y) || (a.x - b.x));
  for (const op of ops) {
    const name = op.name;
    if (name === "h" || name === "x") {
      lines.push(`${name} q[${op.target}];`);
      continue;
    }
    if (name === "rx" || name === "ry" || name === "rz") {
      const theta = requireAngle(name, op)!;
      lines.push(`${name}(${theta}) q[${op.target}];`);
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

type ParsedQasmOp = { name: GateName; target: number; control?: number; theta?: number; parameter?: string };

function numericExpression(expression: string): number | undefined {
  const source = expression.replace(/\s+/g, "");
  let cursor = 0;

  function primary(): number | undefined {
    if (source[cursor] === "+" || source[cursor] === "-") {
      const sign = source[cursor++] === "-" ? -1 : 1;
      const value = primary();
      return value === undefined ? undefined : sign * value;
    }
    if (source[cursor] === "(") {
      cursor += 1;
      const value = additive();
      if (source[cursor] !== ")") return undefined;
      cursor += 1;
      return value;
    }
    if (source.slice(cursor, cursor + 2).toLowerCase() === "pi") {
      cursor += 2;
      return Math.PI;
    }
    const match = source.slice(cursor).match(/^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?/);
    if (!match) return undefined;
    cursor += match[0].length;
    return Number(match[0]);
  }

  function multiplicative(): number | undefined {
    let value = primary();
    if (value === undefined) return undefined;
    while (source[cursor] === "*" || source[cursor] === "/") {
      const operator = source[cursor++];
      const rhs = primary();
      if (rhs === undefined) return undefined;
      value = operator === "*" ? value * rhs : value / rhs;
    }
    return value;
  }

  function additive(): number | undefined {
    let value = multiplicative();
    if (value === undefined) return undefined;
    while (source[cursor] === "+" || source[cursor] === "-") {
      const operator = source[cursor++];
      const rhs = multiplicative();
      if (rhs === undefined) return undefined;
      value = operator === "+" ? value + rhs : value - rhs;
    }
    return value;
  }

  const value = additive();
  return cursor === source.length && value !== undefined && Number.isFinite(value) ? value : undefined;
}

function parseAngle(expression: string): { theta?: number; parameter?: string } {
  const numeric = numericExpression(expression);
  if (numeric !== undefined) return { theta: numeric };
  if (/^[A-Za-z][A-Za-z0-9_]*$/.test(expression.trim())) return { parameter: expression.trim() };
  throw new Error(`Unsupported angle expression: ${expression}`);
}

export function parseMinimalQasm3(qasm: string): { nQubits: number; ops: ParsedQasmOp[] } {
  // Deliberately strict parser for the unitary OpenQASM 3 subset supported by
  // the agent. It accepts standard headers, input parameters, expressions,
  // arbitrary single quantum-register names, and both comment styles.
  const lines = qasm
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .split(/\r?\n/)
    .map((l) => l.trim())
    .map((l) => l.replace(/\/\/.*$/, "").trim())
    .filter(Boolean);

  let nQubits = 0;
  let registerName = "q";
  for (const line of lines) {
    const m = line.match(/^qubit(?:\[(\d+)\])?\s+([A-Za-z_][A-Za-z0-9_]*)\s*;\s*$/);
    if (m) {
      nQubits = m[1] ? Number(m[1]) : 1;
      registerName = m[2];
    }
  }
  if (!nQubits) throw new Error("Failed to find qubit register declaration");

  const register = registerName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const ops: ParsedQasmOp[] = [];
  for (const line of lines) {
    if (/^OPENQASM\s+3(?:\.0)?\s*;$/i.test(line) || /^include\s+"stdgates\.inc"\s*;$/i.test(line) || /^input\s+float(?:\[[^\]]+\])?\s+[A-Za-z_][A-Za-z0-9_]*\s*;$/i.test(line)) continue;
    if (/^barrier\b.*;$/i.test(line)) continue;
    let m = line.match(new RegExp(`^(h|x)\\s+${register}\\[(\\d+)\\]\\s*;\\s*$`, "i"));
    if (m) {
      ops.push({ name: m[1] as GateName, target: Number(m[2]) });
      continue;
    }
    m = line.match(new RegExp(`^(rx|ry|rz)\\((.+)\\)\\s+${register}\\[(\\d+)\\]\\s*;\\s*$`, "i"));
    if (m) {
      ops.push({ name: m[1].toLowerCase() as GateName, ...parseAngle(m[2]), target: Number(m[3]) });
      continue;
    }
    m = line.match(new RegExp(`^(cx|cz)\\s+${register}\\[(\\d+)\\]\\s*,\\s*${register}\\[(\\d+)\\]\\s*;\\s*$`, "i"));
    if (m) {
      ops.push({ name: m[1].toLowerCase() as GateName, control: Number(m[2]), target: Number(m[3]) });
      continue;
    }
    throw new Error(`Unsupported OpenQASM 3 statement: ${line}`);
  }
  return { nQubits, ops };
}
