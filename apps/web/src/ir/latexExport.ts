import type { GateOp } from "../state/circuitStore";

/**
 * Generate publication-ready LaTeX Quantikz code for a quantum circuit.
 */
export function circuitToQuantikz(nQubits: number, ops: GateOp[]): string {
  const maxCol = ops.reduce((m, o) => Math.max(m, o.col), -1);
  const totalCols = maxCol + 1;

  // Initialize a grid of wires for each qubit
  const grid: string[][] = Array.from({ length: nQubits }, () =>
    Array.from({ length: totalCols }, () => "\\qw")
  );

  // Group multi-qubit and single-qubit gates
  for (const op of ops) {
    const col = op.col;
    if (col < 0 || col >= totalCols) continue;

    if (op.name === "cx") {
      const c = op.control ?? 0;
      const t = op.target;
      const dist = t - c;
      grid[c][col] = `\\ctrl{${dist}}`;
      grid[t][col] = "\\targ{}";
    } else if (op.name === "cz") {
      const c = op.control ?? 0;
      const t = op.target;
      const dist = t - c;
      grid[c][col] = `\\ctrl{${dist}}`;
      grid[t][col] = "\\gate{Z}";
    } else if (op.name === "rx" || op.name === "ry" || op.name === "rz") {
      const param = op.parameter ? op.parameter : op.theta !== undefined ? `${op.theta.toFixed(3)}` : "\\theta";
      grid[op.target][col] = `\\gate{R_${op.name[1].toUpperCase()}(${param})}`;
    } else {
      grid[op.target][col] = `\\gate{${op.name.toUpperCase()}}`;
    }
  }

  const wireLines = grid.map(
    (row, idx) => `  \\lstick{\\ket{0}_{${idx}}} & ` + row.join(" & ") + " & \\qw \\\\"
  );

  return `% Requires \\usepackage{tikz} and \\usetikzlibrary{quantikz2}
\\begin{quantikz}
${wireLines.join("\n")}
\\end{quantikz}
`;
}

/**
 * Generate CSV representation of circuit operations.
 */
export function circuitToCsv(nQubits: number, ops: GateOp[]): string {
  const headers = "id,name,target,control,theta,parameter,column";
  const rows = ops.map((op) => [
    op.id,
    op.name,
    op.target,
    op.control !== undefined ? op.control : "",
    op.theta !== undefined ? op.theta : "",
    op.parameter ?? "",
    op.col,
  ].join(","));

  return [headers, ...rows].join("\n");
}
