# Compute architecture and staged research roadmap

This document defines the target computation platform before adding the next
large tensor-network backends. The product is a local-first desktop research
workbench. The embedded web UI is only a presentation layer; numerical work is
owned by the Python/CUDA agent and the Tauri process owns transport, storage,
and lifecycle.

## What “universal research workbench” means here

The target is not a solver that is optimal for every problem. No tensor-network
method can avoid exponential cost when entanglement or contraction width grows
without bound. The target is a composable tool that:

- chooses a representation and algorithm explicitly;
- estimates resource and contraction cost before allocation;
- reports approximation, truncation, convergence, and validation evidence;
- checkpoints and resumes long calculations;
- lets a domain plugin add a model without changing numerical kernels; and
- rejects unsupported or scientifically unsafe requests instead of silently
  returning an answer with the wrong semantics.

## Eight architectural layers

### 0. Desktop/runtime layer

Tauri starts and supervises the local compute agent, selects the loopback port,
keeps the agent token private, persists projects/runs/studies in SQLite, and
exposes cancellation, progress, export, and recovery to the UI.

### 1. Tensor execution layer

One backend-neutral contract for NumPy reference execution and CuPy GPU
execution. It owns dtype, device, stream synchronization, SVD/eigh/QR,
einsum/contraction dispatch, memory accounting, and deterministic seeds. No
physics algorithm should import CUDA-specific policy directly.

### 2. Representations layer

The canonical objects are:

- dense statevector and dense operator for small reference systems;
- MPS and MPO for 1D and snake-ordered low-entanglement workloads;
- finite PEPS for rectangular 2D/3D lattices;
- boundary-MPS environments for finite PEPS contraction;
- corner/edge environments for CTMRG and iPEPS;
- 3D boundary tensor networks and coarse-grained tensors; and
- block-sparse/symmetry-aware variants of the above.

Representations expose common operations such as norm, local expectation,
overlap, apply operator, truncate, canonicalize, serialize checkpoint, and
estimate memory. Algorithms consume these protocols instead of concrete tensor
classes.

### 3. Operator/model layer

Hamiltonians and observables use a common operator IR:

`Model -> OperatorSum -> MPO/local terms -> backend representation`.

This layer contains Pauli sums, sparse local terms, MPO construction, fermionic
operators, particle-number/spin metadata, boundary conditions, lattice graphs,
and observable definitions. A new domain should build this IR, not call DMRG or
PEPS internals directly.

### 4. Algorithm layer

The planned engines are:

| Family | Engines | Primary use |
| --- | --- | --- |
| 1D ground state | DMRG, finite-temperature DMRG, VUMPS | chains and MPO models |
| 1D dynamics | TEBD, TDVP, MPO evolution | short/long time evolution |
| finite 2D | simple update, boundary-MPS contraction, full update, variational PEPS | finite lattices |
| infinite 2D | iPEPS + CTMRG, transfer-matrix analysis | phase diagrams and thermodynamic limit |
| 3D approximate | boundary tensor contraction, HOTRG/TRG, cluster contraction | bounded exploratory 3D work |
| chemistry/materials | fermionic MPS/PEPS, MPO chemistry Hamiltonians, active-space workflows | Hubbard and electronic structure |

Every engine returns the same result envelope: energies, observables, norm,
variance when available, truncation/error estimates, convergence history,
resource usage, checkpoint id, and provenance.

### 5. Symmetry and high-entanglement layer

This is a cross-cutting layer rather than a separate solver. It adds U(1), Z2,
spin and particle-number sectors, block-sparse tensors, charge-conserving
operators, canonical gauges, full-update optimization, and tangent-space
methods. It is essential for high-entanglement and chemistry workloads.

### 6. Admission, convergence, and validation layer

Before compute, the system performs:

- GPU/RAM/time/contraction-width admission;
- method-specific feasibility checks;
- exact small-system cross-validation;
- bond dimension, timestep, sweep, environment-dimension and cutoff studies;
- uncertainty/status classification (`converged`, `needs_review`, `rejected`);
- reproducible seeds, hardware snapshot, request/result hashes; and
- structured failure reasons.

This layer must prevent an approximate result from being presented as a final
scientific answer.

### 7. Orchestration and domain plugin layer

The registry selects an algorithm by capabilities and resource policy. Plugins
provide lattice models, fermion mappings, Hubbard/materials models, chemistry
input readers, parameter sweeps, and domain-specific observables. Long jobs
use bounded queues, exclusive GPU reservations, checkpoint/resume, progress,
cancellation, and durable Study manifests.

## Algorithm count versus architecture count

There are eight architectural layers, but approximately twelve major compute
engines in the research roadmap:

1. dense/reference statevector;
2. MPS expectation/sampling;
3. DMRG;
4. TEBD;
5. TDVP/VUMPS;
6. MPO construction/evolution;
7. finite PEPS simple update;
8. finite PEPS boundary-MPS;
9. full-update/variational PEPS;
10. CTMRG/iPEPS;
11. 3D HOTRG/boundary tensor contraction;
12. fermionic/symmetry-aware chemistry engines.

They should share the same tensor runtime, operator IR, preflight, result
schema, provenance, and checkpoint interface. This is what keeps the codebase
scalable rather than creating twelve disconnected backends.

## Build order and completion gates

### Stage A — freeze contracts

Define representation protocols, operator IR, result envelope, checkpoint
format, capability registry, and resource-budget schema. Add contract tests
before changing numerical implementations.

### Stage B — strengthen the 1D core

Finish MPS/MPO canonicalization, variance, TDVP/VUMPS interfaces, symmetry
hooks, and exact/reference cross-checks. Existing DMRG and TEBD remain usable
throughout this stage.

### Stage C — finite 2D boundary-MPS

Implement environment sweeps, truncation, local observables, checkpointing, and
bond/environment convergence. This is the next highest-value layer after the
current simple-update PEPS path.

### Stage D — CTMRG/iPEPS and full update

Add corner/edge environments, transfer-matrix diagnostics, variational/full
update, and automatic environment-dimension convergence. Benchmark against
known Ising/Heisenberg limits.

### Stage E — high entanglement and symmetry

Add block-sparse U(1)/Z2 tensors, better gauges, TDVP/full-update optimization,
and failure-safe escalation of bond/environment dimensions.

### Stage F — 3D approximate contraction

Add HOTRG or a boundary tensor engine only after 2D environment machinery is
stable. Every 3D mode must expose a conservative cross-section bound and a
clear approximate-result label.

### Stage G — materials and chemistry

Add fermionic tensor objects, symmetry sectors, MPO Hamiltonians, FCIDUMP/Molden
readers, active-space workflows, and independent references. Hubbard is the
first domain; general chemistry follows after fermionic validation.

### Stage H — desktop release and research validation

Compile and package Tauri, bundle the agent/runtime, add migration and backup
tests for SQLite, run benchmark suites on clean machines, and publish a
reproducible research report for every supported method.

## Definition of done for each module

A module is not complete when one example runs. It is complete when it has:

1. a stable capability and payload contract;
2. CPU/reference tests on tiny systems;
3. GPU tests within a declared memory budget;
4. a known-answer or independent-backend comparison;
5. convergence and truncation diagnostics;
6. checkpoint/restart and cancellation tests;
7. provenance and durable result storage; and
8. documentation stating where the method is useful and where it is unsafe.

## Honest scope

The current implementation already provides the desktop/runtime foundation,
GPU agent, plugin registry, MPS/DMRG/TEBD, bounded finite PEPS, double-layer
contraction, preflight, provenance, convergence studies, and local history.
The largest missing scientific layers are finite boundary-MPS, CTMRG/full
update, symmetry-aware high-entanglement algorithms, and fermionic chemistry.
Those can be added without redesigning the desktop or frontend because they
will implement the shared contracts above.

## Product mission and target users

### Mission

Provide a local-first, GPU-accelerated desktop environment in which a
researcher can define a lattice or electronic model, select a numerically
appropriate tensor-network method, run a bounded calculation, inspect evidence
of accuracy, and replay or export the result without writing backend glue code.

### Primary users

- condensed-matter researchers studying spin systems and lattice Hamiltonians;
- computational materials researchers working with Hubbard-like models;
- quantum-algorithm researchers validating circuits and low-entanglement
  dynamics; and
- developers adding a new model or contraction method as a plugin.

### Non-goals

- claiming exact answers for arbitrary 2D/3D or volume-law-entangled systems;
- replacing a general-purpose quantum chemistry package in the first release;
- silently falling back from a requested method to a different approximation;
- hiding truncation, convergence, or resource failures behind a polished UI; or
- treating a browser server as the product runtime. The browser UI is embedded
  in the desktop shell for presentation.

## Architectural invariants

These rules apply to every future backend:

1. Numerical kernels do not depend on React, Tauri, Django, or HTTP.
2. Domain plugins produce the common operator/model IR and never call a solver
   implementation directly.
3. Every compute request is validated and admitted before GPU allocation.
4. Every approximate result states its method, bond/environment dimension,
   truncation, norm/variance diagnostics, and convergence status.
5. CPU reference code is independent enough to catch shared GPU algorithm bugs.
6. A failed or canceled job is a first-class durable state, not a missing row.
7. Checkpoints are versioned and include request hash, algorithm config, tensor
   representation metadata, dtype, device, and random seed.
8. A new backend must be selectable through the capability registry without
   modifying existing frontend algorithm branches.
9. Host RAM is never reclaimed or repurposed by the agent; GPU reservations are
   explicit and bounded.
10. A scientific result and its evidence are stored together and are replayable.

## Shared contracts

### Request contract

Every solver request has these logical fields, even if a public endpoint uses a
specialized payload:

```text
ProblemSpec
  model_id, geometry, operator_ir, initial_state, observables
AlgorithmSpec
  method, representation, bond_dim, environment_dim, cutoff, steps, sweeps
ResourceBudget
  device, max_gpu_mb, max_host_mb, max_time_ms, queue_timeout_ms
ExecutionSpec
  dtype, seed, checkpoint_policy, cancellation_policy
```

Specialized payloads may add fields, but they must map to this canonical shape
before admission. This prevents each plugin from inventing incompatible memory,
progress, or provenance semantics.

### Result contract

```text
ResearchResult
  status: done | needs_review | failed | canceled
  method, representation, problem_sha256, result_sha256
  energies, observables, norm2, variance
  truncation: discarded_weight, cutoff, max_bond_dim
  convergence: history, criterion, converged, residual
  resources: device, peak_gpu_mb, elapsed_ms
  checkpoint: last_id, resumable
  warnings, limitations
  provenance: request, backend, software, hardware, seed, timestamps
```

The JSON envelope is stable across algorithms. An algorithm may add a typed
`details` object, but consumers must not need to understand solver-specific
fields to display status, diagnostics, or provenance.

### Representation protocol

Every state or environment representation should eventually implement:

```text
initialize(problem, execution) -> State
norm2() -> scalar
expectation(observable) -> scalar
apply(operator) -> State
truncate(policy) -> TruncationReport
estimate_resources() -> ResourceEstimate
checkpoint(writer) -> CheckpointRef
restore(reader, ref) -> State
```

Not every operation is meaningful for every representation. Unsupported
operations must return a typed capability error, not a silent conversion.

### Plugin protocol

A domain plugin registers:

```text
plugin_id, version
models and geometry validators
operator builders
observable builders
supported representations and algorithms
preflight estimator
reference fixtures and benchmark cases
```

The plugin is responsible for model semantics. The core is responsible for
tensor arithmetic, admission, execution, provenance, and persistence.

## Backend status matrix

| Capability | Current status | Target evidence |
| --- | --- | --- |
| Dense/reference | usable for small validation | independent cross-check suite |
| MPS expectation/sampling | usable | larger MPO and symmetry coverage |
| DMRG | usable for bounded 1D and prototypes | variance, checkpoint, finite-temperature |
| TEBD | usable for controlled short dynamics | TDVP/VUMPS and long-time stability |
| Finite PEPS simple update | usable for bounded 2D/3D exploration | compare with boundary-MPS/full update |
| PEPS double-layer contraction | implemented for bounded width | environment-aware contraction path |
| Finite boundary-MPS | planned next | 2D convergence and exact small references |
| CTMRG/iPEPS | planned | Ising/Heisenberg phase benchmarks |
| Full-update PEPS | planned | energy/gradient convergence evidence |
| 3D approximate contraction | planned | HOTRG/boundary-width benchmark suite |
| Fermionic Hubbard | mapping/prototype level | symmetry-aware MPO and reference energies |
| General chemistry | not production | FCIDUMP/active-space/reference workflow |
| Desktop packaging | source-level shell | clean-machine Tauri build and bundled agent |

## Benchmark and validation matrix

Every major method gets a permanent benchmark family:

| Area | Tiny correctness | Scaling | Scientific evidence |
| --- | --- | --- | --- |
| MPS/DMRG | 2–8 qubit exact diagonalization | 32–128 site chains | energy, variance, discarded weight |
| TEBD/TDVP | analytic one/two-site dynamics | increasing time and χ | timestep, norm drift, χ study |
| finite PEPS | 2×2 and 3×3 exact/reference | 5×5, 8×8 bounded width | χ/environment/cutoff study |
| CTMRG | known Ising limits | environment dimension | correlation length and phase observables |
| 3D | 2×2×2 exact small case | cross-section sweep | contraction width and truncation study |
| Hubbard | two-site/two-mode references | 2×2, 3×2 clusters | energy, particle number, spin symmetry |
| chemistry | few-orbital exact reference | active-space sweep | comparison with independent package |

The benchmark runner must record machine, GPU memory, dtype, seed, git
revision, request hash, result hash, runtime, peak memory, and convergence
history. “Pass” means both numerical tolerance and diagnostic criteria pass.

## Resource and safety policy

### GPU

- Compute jobs use an exclusive GPU reservation by default.
- Admission uses estimated peak memory plus allocator headroom.
- A job may be queued, rejected, or canceled; it may not bypass admission.
- Device selection and reservation are persisted in job logs.

### Host RAM

- No backend may allocate a dense `2**n` object outside an explicitly bounded
  reference mode.
- Large tensor allocations must be estimated before execution.
- Checkpoint writes are streamed or size-bounded.
- The agent never terminates or modifies unrelated host processes.

### Numerical safety

- Complex dtype and truncation policy are explicit.
- Norm drift, variance, residual, discarded weight, and environment residual
  are reported where applicable.
- NaN/Inf detection terminates the job with a structured diagnostic.
- Approximate results are labeled `needs_review` until the configured
  convergence criteria pass.

## Dependency order

The following dependencies are intentional:

```text
Tensor runtime
  -> representations (MPS/MPO/PEPS/environments)
  -> operator IR and model plugins
  -> algorithms
  -> admission/convergence/benchmark adapters
  -> async orchestration/checkpoint
  -> desktop commands and frontend views
```

Symmetry belongs below algorithms and above tensor representations as a shared
capability. Chemistry and materials belong above the operator IR. CTMRG and
boundary-MPS must reuse the same contraction and truncation policy rather than
creating a second tensor runtime.

## Milestone exit criteria

### M0 — architecture freeze

- this document is the source of truth;
- payload/result/checkpoint contracts are versioned;
- capability registry and error taxonomy are defined;
- contract tests cover backend selection and unsupported operations.

### M1 — 1D research baseline

- DMRG and TEBD retain current behavior;
- MPO and variance APIs exist;
- exact small-system validation is automated;
- checkpoint/resume works for a canceled or interrupted run.

### M2 — useful finite 2D

- boundary-MPS computes PEPS norm and local observables;
- environment bond dimension and truncation are reported;
- 2D benchmark table is reproducible;
- frontend shows method and convergence evidence without solver-specific code.

### M3 — thermodynamic 2D/high entanglement

- CTMRG/iPEPS and full-update paths are available;
- transfer-matrix/correlation-length diagnostics exist;
- U(1)/Z2 symmetry path is tested;
- results agree with reference limits on benchmark models.

### M4 — bounded 3D and materials

- 3D contraction is approximate but width-admitted and checkpointable;
- Hubbard/fermionic MPO path preserves particle-number/spin metadata;
- independent cluster references and convergence studies are available.

### M5 — desktop research release

- Tauri compiles on a clean supported machine;
- Python/CUDA runtime installation is documented or bundled;
- SQLite migration/export/recovery is tested;
- benchmark report and supported-scope statement ship with the release.

## Current execution plan

Status at the architecture-freeze checkpoint:

- [x] target product, scope, limitations, and layer boundaries documented;
- [x] backend families and dependency order documented;
- [x] shared request/result/representation/plugin contracts drafted;
- [x] benchmark, safety, and module acceptance criteria documented;
- [x] implement versioned representation/result/checkpoint contracts;
- [x] add M0 contract tests and capability errors;
- [ ] complete M1 1D core hardening;
- [ ] implement M2 finite boundary-MPS.

The next implementation unit is **M0 followed by M1 contract hardening**, then
**M2 finite boundary-MPS**. No CTMRG, 3D, or chemistry implementation should
be started by copying PEPS code before the shared representation, result, and
checkpoint contracts are in place.

Progress is tracked in this document by updating the backend status matrix and
milestone exit criteria after each tested module. This file is the authoritative
purpose and roadmap for the compute platform; code, frontend, and desktop work
must not silently diverge from it.
