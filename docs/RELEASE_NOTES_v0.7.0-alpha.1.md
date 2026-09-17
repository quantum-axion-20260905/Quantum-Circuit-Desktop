# Quantum Circuit Desktop v0.7.0-alpha.1

## Finite 2D boundary-MPS research slice

This prerelease advances Phase 2 with a bounded, open-rectangular 2D PEPS
boundary-MPS contraction path. It is intended for reproducible research
experiments within the declared limits; it is not yet the final v0.7.0
production release.

- Dedicated atomic boundary-environment checkpoint format with request
  fingerprints, cancellation, resume, and per-row diagnostics.
- 2×2 and 3×3 exact validation against opt_einsum and an independent
  enumeration reference, including local observables and environment-`χ`
  convergence evidence.
- Real CUDA agreement and bounded 3×3 Ising/Heisenberg smoke studies without
  dense `2**N` statevector materialization.
- Frontend finite-2D convergence studies now vary environment `χ_env` and
  display boundary truncation, method, limitations, and convergence status.

## Validation

- 79 agent tests passed, including 5 real-CUDA agreement/smoke tests.
- Python compile, frontend lint, and frontend production build passed.

## Declared limits

The v0.7.0 acceptance gate remains open. CTMRG/boundary-MPS infinite systems,
periodic PEPS, 3D production contraction, high-entanglement full updates,
symmetry-aware solvers, and production chemistry remain future phases.
