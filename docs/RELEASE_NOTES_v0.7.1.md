# Quantum Circuit Desktop v0.7.1

## Spin-lattice vertical slice

This maintenance release closes the Phase 3 spin-lattice workflow within the
declared bounded MPS/finite-PEPS limits:

`lattice -> Hamiltonian -> backend/preflight -> GPU run -> bounded convergence
study -> structured observables/provenance -> replay/history -> export`.

- Ising, Heisenberg, and XXZ builders are covered by the same domain plugin.
- DMRG, TEBD, and finite-2D boundary-MPS/PEPS results now expose a common
  structured observable artifact with labels, sparse Pauli support,
  coefficients, and values.
- Async results persist their preflight admission report alongside provenance.
- Convergence studies preserve every point's full request, result, failure
  message, and numerical uncertainty. Failed and canceled points remain
  visible instead of being silently dropped.
- The frontend shows energy range, half-range numerical uncertainty,
  truncation/norm diagnostics, named observables, and exports a versioned
  `quantum-circuit/physics-study-v1` JSON artifact.
- Package and agent versions are aligned at `0.7.1`.

## Validation evidence

- 80 agent tests passed; Python compile, frontend lint, and production build
  passed.
- Bounded real CUDA checks completed for 4-site Ising, Heisenberg, and XXZ
  using both DMRG and TEBD.
- Bounded real CUDA 2x2 open Ising PEPS boundary-MPS check completed with
  `materializes_statevector=false`.
- Live desktop workflow completed a 4-site Ising DMRG convergence study with
  three points, visible uncertainty, named observables, provenance-backed
  history, and the export-artifact control.

## Declared limits

This release is a complete spin-lattice vertical slice, not a universal
many-body solver. CTMRG/iPEPS, full-update PEPS, high-entanglement symmetric
solvers, production-scale 3D contraction, and production chemistry remain
later phases. A `Needs review` convergence verdict is an honest numerical
result, not a failure hidden by the UI.
