# Testing strategy

## Acceptance tests (system-level)
- Agent GPU detection ok
- GPU MPS sanity (Bell, GHZ, non-adjacent CX, 64/256-qubit low-bond runs)
- MPS truncation diagnostics (`bond_dim`, `norm2`, `discarded_weight`)
- Phase 1 spin-chain references: 4-site transverse Ising, Heisenberg, and XXZ
  DMRG agree with exact diagonalization within the declared dtype envelope
- CPU/GPU agreement for MPS observables, DMRG, and TEBD on bounded 4-qubit
  cases; real CUDA tests are skipped explicitly on CPU-only machines
- DMRG automatically performs a small exact-energy cross-check when feasible
  and reports an explicit skip reason above the bounded size/memory limit
- TEBD timestep, norm-drift, bond-growth, discarded-weight histories, and
  cooperative cancellation are covered by contract tests
- Finite-2D boundary-MPS: 2×2 and 3×3 exact double-layer agreement, `χ_env`
  convergence, row diagnostics, cancellation, and checkpoint/resume
- Bounded real-CUDA PEPS boundary-MPS agreement with the CPU path
- Physics plugin contracts: 1D/2D/3D lattice graphs, sparse Hamiltonians,
  observable energy, TEBD trajectories, native PEPS, DMRG convergence, and
  swap-aware preflight
- Web can call agent and render results
- IR round-trip (export/import), experiment replay, and JSON artifact export
  feedback in the desktop-facing web surface

## Unit tests (module-level)
- IR validators and converters
- Agent backend determinism with fixed seeds
- Cross-backend numerical agreement against the independent reference backend
- DMRG stopping classification using energy delta, local residual, and variance
- Async job journal recovery after agent restart
- Editor model operations (undo/redo, move, insert)

