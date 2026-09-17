# Module: Local Compute Agent

FastAPI service running locally. GPU/TN compute lives here.

## Responsibilities
- Hardware detection + GPU smoke tests
- Job API contract
- Backend registry (MPS, exact tensor contraction, statevector, sampling, etc.)
- Algorithm-level capability catalog for DMRG/TEBD and planned TDVP/VUMPS
- Reproducibility metadata (seed, versions, device, SHA-256 fingerprints)
- Resource budgets, current free-GPU-memory guard, and cancellation
- Unified async execution envelope, cooperative cancellation, and GPU reservation
- Independent reference-vs-GPU/TN validation for small circuits

## Current endpoints
- `GET /hardware`
- `GET /capabilities`
- `POST /jobs/bench_matmul`
- `POST /jobs/simulate`
- `POST /jobs/sample`
- `POST /jobs/cross_validate`
- `POST /jobs/tn_estimate`
- `POST /jobs/tn_amplitudes`
- `POST /jobs/sweep`
- `GET /plugins`
- `POST /plugins/{plugin_id}/lattice`
- `POST /plugins/{plugin_id}/hamiltonian`
- `POST /plugins/{plugin_id}/fermion_mapping`
- `POST /plugins/{plugin_id}/hubbard`
- `POST /jobs/expectation`
- `POST /jobs/cross_validate_observables`
- `POST /jobs/tebd`
- `POST /jobs/ground_state`
- `POST /jobs/dmrg`
- `POST /jobs/peps`
- `POST /async/jobs` with `kind` set to `run`, `sample`, `simulate`,
  `bench_matmul`, `expectation`, `tebd`, `ground_state`, `dmrg`, `peps`, `sweep`,
  `cross_validate`, `tn_estimate`, or `tn_amplitudes`
- `GET /queue`

## Domain-plugin boundary

The reusable execution kernel is kept below `qc_agent/plugins`: MPS runtime
operations and observable contractions live in `qc_agent/core`, while CUDA
statevector/MPS and CPU reference implementations stay in `qc_agent/backends`.
Domain plugins only provide metadata, lattice/builders, and scientific terms.
The built-in `spin-lattice` plugin currently provides Ising, Heisenberg, and
XXZ Hamiltonians on deterministic 1D/2D/3D rectangular lattices. A future
fermion-mapping or materials plugin can register the same boundary without
changing the simulator kernels. The built-in `hubbard-materials` plugin now
generates spinful Hubbard terms and maps them to Jordan–Wigner Pauli strings.

Fermion mapping responses preserve complex coefficients in `complex_terms` and
only expose `terms` as expectation-ready when the residual imaginary part is
below the declared Hermiticity tolerance.

`POST /jobs/expectation` evaluates sparse Pauli observables and returns each
term expectation, total energy, norm, truncation diagnostics, and provenance.
`POST /jobs/cross_validate_observables` is a bounded validation instrument: it
compares MPS observable values and total energy with the independent CPU
reference, returning per-term errors, tolerance status, norm, bond dimension,
and truncation evidence. It is limited to the reference-circuit size and is
not a production-size dense solver.
`POST /jobs/tebd` evolves bounded-locality Pauli Hamiltonians with first- or
second-order Suzuki-Trotter steps and returns energy/observable trajectories.
For strings longer than two sites it uses a parity-CX network, so its runtime
and truncation diagnostics should be checked more carefully.
`POST /jobs/ground_state` is a small dense GPU eigensolver for exact ground
energy validation; it is intentionally capped at 12 qubits. `POST /jobs/dmrg`
is a finite two-site variational MPS solver with an iterative Lanczos local
eigensolver by default, optional dense validation mode, sweep history,
tolerance, residual/variance-aware stopping, bond and truncation diagnostics.
The `/capabilities` response also exposes algorithm-level method descriptors:
DMRG and TEBD are executable when the tensor-network GPU backend is admitted;
TDVP and VUMPS are registered as separate `planned` capabilities. Requests for
those methods are rejected explicitly and are never silently redirected to
TEBD or DMRG.
`POST /jobs/peps` is a native rectangular 2D/3D finite PEPS simple-update path;
it uses an opt_einsum boundary contraction when available and a bounded
virtual-bond enumeration fallback, rejecting requests that exceed the
contraction budget. The larger 2D/3D path remains a snake-ordered MPS, so its
bond-dimension and time-step convergence must still be checked.

## Tensor-network execution modes

The default tensor-network method is a GPU-native matrix product state (MPS)
simulator. It supports samples and selected amplitudes without allocating a
`2**n` statevector. Set `bond_dim` to bound entanglement (for example `2`,
`8`, or `16`); the response reports `bond_dim_used`, `norm2`, and cumulative
`discarded_weight` so an approximate run is auditable. A nonzero discarded
weight means that the requested bond dimension truncated the state.

For small circuits, `tn_method: "contraction"` keeps the exact opt_einsum
tensor-network path available for selected amplitudes and validation. It is
not the scalable default and still requires CUDA plus opt_einsum.

Sampling can also use `noise` with one- and two-qubit depolarizing channels
plus independent readout flips. Noise is evaluated as unitary Pauli
trajectories per shot, so the result is a statistical sample distribution;
selected amplitudes with active noise are rejected rather than mislabeled as
exact amplitudes.

Parametric rotations may use `parameter` instead of `theta`. `POST /jobs/sweep`
accepts a finite Cartesian product in `parameter_values`, runs every point
through the same backend/preflight/provenance path, and returns per-point
results or errors. Sweeps are capped at 256 points and execute sequentially
by default; `parallel_devices` can distribute independent points across
distinct GPUs on the legacy sync endpoint. The unified async envelope
intentionally requires `parallel_devices=1` until a multi-device sweep
scheduler is introduced.

## Unified asynchronous lifecycle

Use `POST /async/jobs` with an envelope such as:

```json
{
  "kind": "dmrg",
  "payload": {
    "n_qubits": 8,
    "terms": [{"paulis": {"0": "Z"}, "coefficient": 1.0}],
    "bond_dim": 8,
    "budget": {"max_mem_mb": 1024, "queue_timeout_ms": 120000}
  }
}
```

Poll `GET /async/jobs/{job_id}` and cancel with
`POST /async/jobs/{job_id}/cancel`. GPU jobs reserve estimated memory with
headroom before entering the runner; failure waits or times out without
touching unrelated processes or host RAM.

Example low-bond sampling request:

```json
{
  "n_qubits": 256,
  "gates": [],
  "backend": "tensor-network",
  "result_type": "samples",
  "tn_method": "mps",
  "bond_dim": 2,
  "shots": 128,
  "seed": 7
}
```

