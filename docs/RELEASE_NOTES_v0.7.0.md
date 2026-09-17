# Quantum Circuit Desktop v0.7.0

## Finite 2D boundary-MPS research release

This release closes the Phase 2 finite open-2D boundary-MPS acceptance gate
within the declared limits. It provides a reproducible 2D PEPS workflow for
bounded research experiments; it is not a universal tensor-network or
materials/chemistry production solver.

- Dedicated atomic boundary-environment checkpoint format with request
  fingerprints, cancellation, resume, and per-row diagnostics.
- 2×2 and 3×3 exact validation against opt_einsum and an independent
  enumeration reference, including local observables and environment-`χ`
  convergence evidence.
- Explicit separation of physical PEPS truncation from boundary-environment
  truncation in result artifacts and frontend diagnostics.
- Real CUDA agreement and bounded 3×3 Ising/Heisenberg studies without dense
  `2**N` statevector materialization.
- Frontend finite-2D convergence studies vary `χ_env`, preserve replayable
  payloads, and display method, limitations, error budgets, and convergence
  verdicts.

## Validation

- 79 agent tests passed, including 5 real-CUDA agreement/smoke tests.
- Python compile, frontend lint, and frontend production build passed.
- Live desktop verification completed a 3×3 `D=4` / `χ_env={8,16}` study,
  correctly returning `Needs review` for the finite-bond error budget, and
  replayed the stored boundary-MPS payload successfully.

## Declared limits

This release covers finite open rectangular 2D PEPS only for boundary-MPS.
Periodic PEPS, CTMRG/iPEPS, high-entanglement full updates, symmetry-aware
solvers, production-scale 3D contraction, and production fermionic
materials/chemistry remain later phases.
