# ADR-0002: IR = OpenQASM 3 + UI metadata

## Context
UI, agent backends, backend storage o‘rtasida bitta source of truth kerak.

## Decision
Circuit semantics OpenQASM 3 bilan, UI layout/annotations alohida JSON metadata’da saqlanadi.

## Alternatives
- Custom JSON-only IR
- Qiskit internal circuit as canonical

## Consequences
- Import/export aniq bo‘ladi.
- UI metadata schema’ni barqaror qilish kerak.

