# Quantum Circuit Desktop v0.6.0

## 1D tensor-network research slice

This release closes the bounded 1D MPS/DMRG/TEBD acceptance gate. It is a
research-ready 1D workflow within declared bond, sweep, timestep, memory, and
entanglement limits; it is not a universal 2D/3D or chemistry solver.

- DMRG reports energy, norm, variance, residual, bond/truncation evidence,
  convergence classification, and an automatic exact cross-check up to 8
  qubits when feasible.
- TEBD preserves point-level norm drift, bond growth, discarded-weight history,
  timestep, and truncation diagnostics, including cancellation at safe
  boundaries.
- CPU/reference coverage includes transverse Ising, Heisenberg, and XXZ
  chains; bounded real-CUDA agreement covers MPS observables, DMRG, and TEBD.
- Replayable local history and JSON export feedback remain available from the
  desktop-facing UI.
- TDVP and VUMPS are explicit planned capabilities. Requests are rejected
  instead of being silently redirected to another solver.

## Known limits

Finite-2D boundary-MPS/CTMRG, high-entanglement symmetry-aware solvers,
large-3D contraction, and production fermionic chemistry remain later roadmap
phases. Every result must still be checked across bond dimension, sweeps or
time step, truncation, and an independent reference where feasible.
