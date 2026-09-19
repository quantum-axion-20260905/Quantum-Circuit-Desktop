# Quantum Circuit Desktop v0.8.0-alpha.3

## Boundary-MPS transfer fixed-point diagnostic

This prerelease adds the next bounded Phase 4 research instrument:

- dynamic CTMRG payloads can request a finite-width row-transfer boundary-MPS
  fixed-point diagnostic;
- each unit-cell period reports normalized residual, Rayleigh quotient,
  discarded weight, and retained boundary bond dimension;
- the result is carried through the structured research gate and remains
  separate from the finite open-patch boundary-MPS reference;
- preflight prices the extra transfer cycles and documents that the
  independent contraction is a `cpu-reference` diagnostic.

## Validation evidence

- 197/197 agent regression tests passed.
- Product D=1 CUDA dynamic probe converged with zero boundary residual.
- Random D=2 seed83 CUDA dynamic probe produced boundary residual `8.31e-3`
  and unresolved CTMRG transfer gap `1.81e-6`; the research result correctly
  remained `needs_review`.
- No production admission was widened; the generic entangled solver remains
  experimental.
