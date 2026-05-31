# Module: Circuit IR

Source of truth: OpenQASM 3 + `ui_metadata.json`.

## Responsibilities
- IR schema (JSON Schema)
- Validation
- Converters:
  - Web editor model ↔ IR
  - IR ↔ Agent job payloads

## Planned artifacts
- `schemas/circuit_ir.schema.json`
- `schemas/ui_metadata.schema.json`
- TypeScript types in web (generated or hand-written)

