import { z } from "zod";

export const GateNameSchema = z.enum(["h", "x", "rx", "ry", "rz", "cx", "cz"]);
export type GateName = z.infer<typeof GateNameSchema>;

export const IrNodeSchema = z.object({
  id: z.string().min(1),
  name: GateNameSchema,
  target: z.number().int().nonnegative(),
  control: z.number().int().nonnegative().optional(),
  theta: z.number().finite().optional(),
  parameter: z.string().regex(/^[A-Za-z][A-Za-z0-9_]*$/).optional(),
  col: z.number().int().nonnegative().default(0),
  moment: z.number().int().nonnegative().optional(),
  sequence: z.number().int().nonnegative().optional(),
  x: z.number().finite(),
  y: z.number().finite()
});
export type IrNode = z.infer<typeof IrNodeSchema>;

export const CircuitIrV1Schema = z.object({
  qasm: z.string().min(1),
  ui: z.object({
    version: z.literal(1),
    source_format: z.string().default("qasm3"),
    n_qubits: z.number().int().min(1).max(64),
    nodes: z.array(IrNodeSchema)
  })
}).superRefine((value, ctx) => {
  for (const [index, node] of value.ui.nodes.entries()) {
    if (node.target >= value.ui.n_qubits) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, path: ["ui", "nodes", index, "target"], message: "target exceeds n_qubits" });
    }
    if ((node.name === "cx" || node.name === "cz") && node.control === undefined) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, path: ["ui", "nodes", index, "control"], message: `${node.name} requires control` });
    }
    if (node.control !== undefined && node.control >= value.ui.n_qubits) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, path: ["ui", "nodes", index, "control"], message: "control exceeds n_qubits" });
    }
    if ((node.name === "cx" || node.name === "cz") && node.control === node.target) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, path: ["ui", "nodes", index], message: "control and target must differ" });
    }
    if ((node.name === "rx" || node.name === "ry" || node.name === "rz") && node.theta === undefined && node.parameter === undefined) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, path: ["ui", "nodes", index, "theta"], message: `${node.name} requires theta or parameter` });
    }
    if (node.theta !== undefined && node.parameter !== undefined) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, path: ["ui", "nodes", index], message: "provide theta or parameter, not both" });
    }
    if (!(node.name === "rx" || node.name === "ry" || node.name === "rz") && node.parameter !== undefined) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, path: ["ui", "nodes", index, "parameter"], message: `${node.name} does not accept a parameter` });
    }
  }
});
export type CircuitIrV1 = z.infer<typeof CircuitIrV1Schema>;

export function validateCircuitIrV1(input: unknown): CircuitIrV1 {
  return CircuitIrV1Schema.parse(input);
}
