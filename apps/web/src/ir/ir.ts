import { z } from "zod";

export const GateNameSchema = z.enum(["h", "x", "rx", "ry", "rz", "cx", "cz"]);
export type GateName = z.infer<typeof GateNameSchema>;

export const IrNodeSchema = z.object({
  id: z.string().min(1),
  name: GateNameSchema,
  target: z.number().int().nonnegative(),
  control: z.number().int().nonnegative().optional(),
  theta: z.number().finite().optional(),
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
});
export type CircuitIrV1 = z.infer<typeof CircuitIrV1Schema>;

export function validateCircuitIrV1(input: unknown): CircuitIrV1 {
  return CircuitIrV1Schema.parse(input);
}
