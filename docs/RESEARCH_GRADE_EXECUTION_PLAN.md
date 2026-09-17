# Research-grade execution plan

Status: active execution plan for the `v0.5.x` line

Current active slice: **Phase 1, 1D MPS/DMRG/TEBD 1.0**. The convention,
observable-reference, and DMRG stopping-classification items are complete. The
next item is a bounded convergence-study helper.

This is the operational plan for turning Quantum Circuit Desktop into a
reliable tensor-network research workbench. It is intentionally narrower than
the universal long-term vision in `COMPUTE_ARCHITECTURE.md`: one capability is
finished and validated before the next large solver is opened.

## 1. What “45%” means

The current percentage is a scope estimate against the long-term universal
platform, not a measure of code quality or effort. The project already has a
solid platform foundation and a useful narrow product. The remaining work is
mostly scientific breadth and release hardening:

| Capability | Current state | Target before calling it complete |
| --- | --- | --- |
| Contracts, registry, preflight, provenance | implemented | stable compatibility and migration tests |
| 1D MPS/DMRG/TEBD | usable, incomplete hardening | research-ready bounded 1D workflow |
| Finite 2D boundary-MPS | experimental open-2D implementation | validated finite-2D workflow |
| CTMRG/iPEPS | planned | tested infinite-2D workflow |
| Symmetry/high entanglement | planned hooks | block-sparse validated solvers |
| 3D contraction | bounded exploratory PEPS only | explicit approximate 3D backend |
| Fermionic materials/chemistry | mapping and small Hubbard prototypes | reproducible domain workflow |
| Tauri desktop distribution | shell exists, native build still pending | clean-machine packaged release |

“100%” always means 100% of one supported capability. It does not mean that
one solver is correct for every lattice, entanglement regime, or chemistry
problem.

## 2. Rules that prevent architectural drift

1. Work on one active phase only. New ideas go into the backlog unless they
   are required by that phase's acceptance gate.
2. Numerical kernels never import frontend, Django, or CUDA policy directly.
   They consume representation, operator, budget, and result contracts.
3. A domain plugin builds a model and renders a result; it does not implement
   its own DMRG, contraction, checkpoint, or provenance logic.
4. Every approximation reports bond/environment dimension, discarded weight,
   norm/variance when available, convergence status, resource estimate, and
   limitations.
5. A requested backend is never silently replaced by another backend. If the
   capability is unsupported, preflight rejects it with a structured reason.
6. GPU runs are bounded by preflight. Large tests are opt-in and must declare
   a memory/time budget; the default test suite remains small and RAM-safe.
7. A phase is not complete because an example runs. It needs a contract,
   reference test, GPU smoke test, convergence evidence, failure behavior,
   provenance, and documentation.

## 3. Execution phases

### Phase 0 — platform contracts and control plane

Status: mostly complete in `v0.5.0`.

Purpose: keep future backends interchangeable and make approximate results
scientifically inspectable.

Delivered:

- versioned representation, result, checkpoint, and resource-budget contracts;
- backend/plugin capability registry and preflight admission;
- shared `ResearchResult` envelope with convergence and truncation reports;
- GPU/CPU execution separation, provenance, replayable run history, and
  bounded async jobs;
- MPS/MPO, PEPS, DMRG, TEBD, and boundary-MPS adapters using shared metadata.

Remaining maintenance:

- keep the architecture checklist synchronized with implementation status;
- add compatibility/migration tests when a contract changes;
- keep unsupported capabilities explicit in `/plugins` and the UI.

Exit gate: no backend-specific result shape is required by the frontend, and a
new solver can register without changing the core job lifecycle.

### Phase 1 — 1D MPS/DMRG/TEBD 1.0

This is the immediate active phase. It is the first capability that should be
brought close to 100% because it is scientifically useful, bounded, and gives
the rest of the project its strongest reference workflow.

Scope:

- finite MPS canonicalization and stable truncation policy;
- DMRG ground-state optimization with sweep convergence;
- TEBD short-time evolution with timestep and bond convergence;
- MPO expectation values, local observables, energy, variance, and norm;
- explicit checkpoint/resume and cancellation at safe boundaries;
- deterministic CPU reference paths and GPU CuPy paths;
- resource estimates before allocation and result classification.

Implementation order:

1. **Done:** audit current MPS tensor conventions and make left/right canonical
   forms explicit in `mps_conventions.py` and tests. The validator now locks
   `(left_bond, physical, right_bond)`, open boundary dimensions, adjacent
   bond compatibility, and QR isometry residuals.
2. **Done:** finish observable coverage and compare MPS/MPO results with dense
   exact references on small systems. The reusable
   `cross_validate_mps_observables` helper and
   `/jobs/cross_validate_observables` validation route report per-observable
   error, energy error, norm, bond dimension, and truncation evidence.
3. **Done:** add sweep-level DMRG stopping rules based on energy and
   residual/variance. `converged` now requires energy delta, local solver
   residual, and variance when the variance is affordable; dtype-aware
   numerical floors prevent false rejection from `complex64` round-off.
4. Add a bounded convergence-study helper for bond dimension, sweeps, cutoff,
   and timestep. Persist every point as a replayable run.
5. Add checkpoint/resume and cancellation tests. A resumed run must report the
   parent checkpoint and produce equivalent results within declared tolerance.
6. Add TDVP/VUMPS interfaces only after the finite MPS contracts are stable;
   they should register as separate capabilities, not be mixed into DMRG.

Latest evidence for completed convention and observable-reference items:

- focused MPS suite: 15 tests passed;
- full agent suite: 63 tests passed;
- bounded CUDA smoke: 8 qubits on device 0, norm drift below `2e-6`, left and
  right isometry residuals below `1e-6`.
- bounded CUDA observable cross-validation: 4 qubits, passed, maximum
  observable error `1.11e-16`, energy error `2.78e-17`, norm² `1.0`.
- bounded CUDA DMRG stopping smoke: 2-qubit Heisenberg passed with energy
  `-2.9999995`, variance `1.91e-6`, residual `4.14e-7`, and an effective
  `complex64` variance tolerance of `8.58e-6`.

Acceptance gate for Phase 1:

- 1D transverse Ising, Heisenberg, and XXZ examples pass CPU reference tests;
- GPU and CPU results agree within documented dtype tolerance;
- energy, norm, variance, and discarded weight are visible in the result;
- bond/sweep convergence classifies stable and unstable runs correctly;
- TEBD reports timestep, norm drift, bond growth, and truncation diagnostics;
- checkpoint/resume, cancellation, replay, and JSON artifact export pass;
- a small exact diagonalization cross-check is automatic when feasible;
- frontend can run and inspect the complete workflow without backend-specific
  conditionals.

Release target: `v0.6.0` (1D research-ready, not universal).

### Phase 2 — finite 2D boundary-MPS 1.0

Status: implementation exists but remains experimental. Do not label it
production-ready until this gate is complete.

Scope:

- finite open rectangular 2D PEPS double-layer contraction;
- local observables and norm/overlap, not only global norm;
- environment bond dimension `chi` convergence studies;
- boundary sweep diagnostics per row/column;
- explicit boundary checkpoint/resume;
- exact small-lattice comparisons and an independent contraction path;
- clear rejection for periodic, 3D, and unsupported geometry.

Acceptance gate:

- 2x2 and 3x3 random PEPS agree with an exact reference before truncation;
- increasing `chi` produces a measurable convergence report;
- truncation and boundary errors are not confused with physical bond errors;
- local observable results survive replay and checkpoint restoration;
- a 2D Ising/Heisenberg smoke study is useful without materializing a dense
  `2**N` statevector;
- frontend shows method, environment `chi`, limitations, and convergence.

Release target: `v0.7.0` (finite 2D research-ready within declared limits).

### Phase 3 — one complete domain vertical slice

This phase closes one end-to-end research workflow instead of adding breadth.
The first slice is spin-lattice:

`lattice model -> Hamiltonian -> backend selection -> preflight -> run ->
 convergence study -> result artifact -> replay/export`.

Required domain behavior:

- Ising, Heisenberg, and XXZ model builders;
- 1D DMRG/TEBD and finite-2D boundary-MPS selection where supported;
- parameter sweeps with failed points and uncertainty visible;
- structured observables and provenance;
- frontend comparison of two runs without duplicating backend logic.

Exit gate: a researcher can reproduce a documented spin-lattice result from a
fresh project using only the desktop workflow and the exported artifact.

Release target: `v0.7.x` maintenance release if Phase 2 and Phase 3 land
together; otherwise keep the phase in the `v0.7` train.

### Phase 4 — CTMRG/iPEPS and full-update 2D

Only start after Phases 1–3 pass their gates. This is the first genuinely
high-risk scientific expansion.

Scope:

- single-site and small unit-cell iPEPS contracts;
- corner and edge environment tensors;
- CTMRG iterations with environment-dimension convergence;
- transfer-matrix and correlation-length diagnostics;
- simple-update versus full-update comparison;
- known Ising/Heisenberg limits and symmetry of observables.

Non-goals for the first CTMRG release:

- arbitrary unit cells without explicit resource estimates;
- silently using finite boundary-MPS in place of CTMRG;
- calling one converged environment a proof of physical correctness.

Acceptance gate: environment `chi`, truncation, CTMRG residual, correlation
length, and physical observable convergence are all reported and benchmarked.

Release target: `v0.8.0` experimental infinite-2D research backend.

### Phase 5 — symmetry and high-entanglement support

Scope:

- U(1), Z2, particle-number, and spin-charge sector metadata;
- block-sparse MPS/MPO/PEPS tensors;
- charge-conserving operator construction and validation;
- better gauge fixing and stable high-bond optimization;
- automatic escalation policy that stops at a declared budget.

Acceptance gate: symmetric and dense paths agree on tiny systems, forbidden
sector transitions are rejected, and memory estimates include block structure.

Release target: `v0.9.0` as capability-specific experimental support.

### Phase 6 — bounded 3D contraction

Scope:

- explicit 3D cross-section and contraction-width estimate;
- HOTRG/TRG or a boundary-tensor backend selected by benchmark evidence;
- small 3D Ising/Heisenberg references;
- conservative admission that protects GPU memory and host RAM;
- result status that clearly says approximate exploratory 3D.

This phase must not be sold as a general large-3D solver. If width or
entanglement exceeds the budget, preflight must reject the job.

### Phase 7 — fermionic materials and chemistry

Scope:

- fermionic operator IR and validated Jordan–Wigner/Bravyi–Kitaev mappings;
- particle-number and spin sectors;
- Hubbard and active-space model workflows;
- FCIDUMP/Molden-style input after the operator IR is stable;
- independent reference comparison and domain observables.

Acceptance gate: mapping signs/phases, particle number, small exact energies,
and replayable inputs are all validated. A small Hubbard prototype is not
called production chemistry.

### Phase 8 — desktop research release

Scope:

- Tauri native build and agent lifecycle on a clean Windows machine;
- offline local-first workflow and SQLite migration/backup tests;
- cancellation/restart/recovery from the desktop shell;
- packaged runtime, versioned artifacts, logs, and reproducible examples;
- accessibility and error-state audit without adding marketing UI.

Release gate: the supported capability matrix is documented, unsupported
features are disabled or rejected, and a fresh user can install, run, inspect,
and export a bounded research calculation.

## 4. Current execution order

The only active implementation phase should be:

1. Phase 1: 1D MPS/DMRG/TEBD 1.0.
2. Phase 2: finish boundary-MPS.
3. Phase 3: close the spin-lattice vertical slice.
4. Phase 4: CTMRG/iPEPS.
5. Phase 5: symmetry/high entanglement.
6. Phase 6: bounded 3D.
7. Phase 7: fermionic materials/chemistry.
8. Phase 8: desktop release hardening.

Frontend work continues only when it exposes an already-supported backend
capability, improves diagnostics, or closes an acceptance gate. It must not
invent controls for planned algorithms.

## 5. Definition of done checklist

Every solver or domain plugin must have all of these before it is marked
research-ready:

- [ ] versioned request and result contract;
- [ ] capability registration and preflight limits;
- [ ] tiny CPU/reference tests;
- [ ] bounded GPU smoke test;
- [ ] independent or exact comparison where feasible;
- [ ] convergence/truncation diagnostics;
- [ ] checkpoint/resume and cancellation behavior;
- [ ] provenance, replay, and artifact export;
- [ ] failure/rejection tests;
- [ ] frontend result view with limitations;
- [ ] documentation of useful and unsafe regimes;
- [ ] release note and reproducible example.

## 6. Goal-mode operating protocol

When this document is used as the goal-mode brief, the agent should:

1. Read this file, `COMPUTE_ARCHITECTURE.md`, the current git status, and the
   active phase's tests before editing code.
2. Work only on the first incomplete item in the active phase unless a
   dependency or failing test requires a different order.
3. Make small, reviewable changes and preserve existing user changes.
4. Run the smallest relevant tests after each slice, then the full suite at
   the phase gate.
5. Update this document's status, the relevant module documentation, and the
   release notes when a gate changes.
6. Do not claim a phase complete from a single successful run. Record the
   reference, convergence, resource, and limitation evidence.
7. Keep GPU jobs bounded and avoid large RAM allocations unless the user has
   explicitly confirmed that system RAM is available.
8. Commit coherent slices. Tag a release only after the phase gate and full
   validation pass.
9. If blocked, record the exact failing contract, test, or environment
   dependency here instead of opening an unrelated backend.

The next goal-mode task is therefore: **complete Phase 1, item 4 — add a
bounded convergence-study helper for bond dimension, sweeps, cutoff, and
timestep.**

## 7. Progress accounting

Progress should be reported in three separate numbers:

- **Platform completion:** contracts, orchestration, UI, provenance, and
  packaging foundation.
- **Capability completion:** how close the active solver is to its own gate.
- **Universal roadmap completion:** breadth against all planned solver families.

This prevents a large planned feature such as CTMRG from making a completed
and useful DMRG workflow look unfinished, while also preventing a polished UI
from hiding missing scientific validation.
