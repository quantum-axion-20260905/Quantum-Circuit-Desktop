# Quantum Circuit Desktop v0.8.0-alpha.5

## Boundary-MPS paired-gauge covariance replay

This prerelease adds a separate Phase 4 diagnostic for the open-boundary
transport problem:

- `/jobs/ctmrg/boundary-mps-transfer-gauge-covariance` compares the original
  tensor, a deliberately incorrect raw-gauge control, and a paired-gauge
  replay with inverse-transpose fused top/side/bottom boundary vectors;
- the same study is a first-class unified async kind;
- the result reports finite-patch deltas, transfer residuals, discarded weight,
  frame-dependent Rayleigh diagnostics, and the per-point covariance gate;
- preflight prices all three replay runs per grid point and keeps the
  independent boundary calculation labelled `cpu-reference`;
- the raw gauged control is preserved as a negative control so changing the
  boundary condition cannot be mistaken for solver improvement.

## Validation evidence

- 203/203 agent regression tests passed.
- RTX 3060 unified async smoke completed on assigned device `0` with `done`
  status and schema `quantum-circuit/boundary-mps-transfer-gauge-covariance-study-v1`.
- Random D=2 seed 83, width `[1,2]`, boundary χ `[1,4]`: 3/4 transported
  covariance points passed; width 2 / χ 1 remains review-only with a relative
  finite-patch delta of about `2.87e-1`.
- The raw-gauge negative control remains strongly non-invariant (relative
  deltas about `1.61–3.14`), demonstrating that the transported boundary is
  doing real work.
- This does not promote generic entangled CTMRG to production: fixed-point,
  transfer-gap, χ-convergence, and variational gates remain separate.
