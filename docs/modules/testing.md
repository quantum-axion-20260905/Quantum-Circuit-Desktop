# Testing strategy

## Acceptance tests (system-level)
- Agent GPU detection ok
- GPU MPS sanity (Bell, GHZ, non-adjacent CX, 64/256-qubit low-bond runs)
- MPS truncation diagnostics (`bond_dim`, `norm2`, `discarded_weight`)
- Physics plugin contracts: 1D/2D/3D lattice graphs, sparse Hamiltonians,
  observable energy, TEBD trajectories, native PEPS, DMRG convergence, and
  swap-aware preflight
- Web can call agent and render results
- IR round-trip (export/import)

## Unit tests (module-level)
- IR validators and converters
- Agent backend determinism with fixed seeds
- Cross-backend numerical agreement against the independent reference backend
- Async job journal recovery after agent restart
- Editor model operations (undo/redo, move, insert)

