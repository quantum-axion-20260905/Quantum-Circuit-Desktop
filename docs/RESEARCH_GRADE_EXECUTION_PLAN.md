# Research-grade execution plan

Status: Phase 3 acceptance gate passed for the declared spin-lattice scope;
the result is released as `v0.7.1`.

Current completed slice: **Phase 3, spin-lattice vertical slice**. Phase 1 is
complete and released as `v0.6.0`; Phase 2 is released as `v0.7.0`; Phase 3 is
released as `v0.7.1` within the bounded MPS and finite open-2D limits
documented below.

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
| CTMRG/iPEPS | bounded experimental 1x1–2x2 slice | gauge/χ-converged infinite-2D workflow |
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
4. **Done:** add a bounded convergence-study helper for bond dimension, sweeps,
   cutoff, and timestep. `apps/web/src/lib/physicsStudy.ts` owns the typed
   three-point cap, safe numeric bounds, deterministic variants, and shared
   truncation-cutoff payload.
5. **Done:** add checkpoint/resume and cancellation tests. DMRG now checkpoints
   atomically at completed sweep boundaries, validates method/dtype/problem
   fingerprint on resume, and exposes the checkpoint manifest in the result.
   Cancellation stops before returning a partial success.
6. **Done:** add separate TDVP/VUMPS interfaces and capability registration.
   `core/mps_methods.py` defines typed solver protocols, while the backend
   registry exposes `mps-tdvp` and `mps-vumps` as distinct planned capabilities.
   The capability API makes their unavailable status explicit and rejects them
   without silently falling back to TEBD or DMRG.
7. **Done:** audited every Phase 1 acceptance-gate item, added the remaining
   reference/GPU/TEBD/export evidence, and ran the full bounded gate. The
   release is now ready for final diff review and tagging as `v0.6.0`.

Latest Phase 1 acceptance evidence:

- focused MPS suite: 15 tests passed;
- full agent suite: 73 tests passed;
- CPU reference coverage: 4-site transverse Ising, Heisenberg, and XXZ DMRG
  energies agree with exact diagonalization within `5e-5`, with normalized
  states and converged stopping classification;
- bounded real-CUDA agreement: 3 tests passed for MPS observables, Heisenberg
  DMRG, and TEBD using the documented complex64 agreement envelope;
- bounded CUDA smoke: 8 qubits on device 0, norm drift below `2e-6`, left and
  right isometry residuals below `1e-6`.
- bounded CUDA observable cross-validation: 4 qubits, passed, maximum
  observable error `1.11e-16`, energy error `2.78e-17`, norm² `1.0`.
- bounded CUDA DMRG stopping smoke: 2-qubit Heisenberg passed with energy
  `-2.9999995`, variance `1.91e-6`, residual `4.14e-7`, and an effective
  `complex64` variance tolerance of `8.58e-6`.
- frontend convergence helper: maximum three bounded serial points with
  bond/sweep, timestep, and truncation-cutoff fields centralized; lint and
  production build passed.
- checkpoint/resume and cancellation: CPU equivalence/cancellation tests
  passed; bounded CUDA 2→3 sweep resume passed with matching problem
  fingerprint and norm² `1.00000024`.
- TEBD diagnostics: timestep, point-level norm²/norm drift, bond growth, and
  cumulative discarded-weight histories are returned; CPU and CUDA tests plus
  cooperative-cancellation tests passed.
- automatic exact cross-check: a 4-qubit DMRG run performed and passed the
  CUDA exact-energy comparison with absolute error below `1e-4`; a 9-qubit
  run reported the explicit bounded-size skip reason without dense allocation.
- algorithm capability catalog: `/capabilities` reports executable DMRG/TEBD
  plus separate planned TDVP/VUMPS entries; unsupported-method rejection tests
  confirm there is no silent solver substitution.
- desktop-facing UI workflow: Physics Lab generated a 1D Hamiltonian, ran GPU
  DMRG and TEBD, rendered convergence diagnostics, replayed a stored run, and
  executed JSON export with visible export feedback.

Acceptance gate for Phase 1:

- [x] 1D transverse Ising, Heisenberg, and XXZ examples pass CPU reference tests;
- [x] GPU and CPU results agree within documented dtype tolerance;
- [x] energy, norm, variance, and discarded weight are visible in the result;
- [x] bond/sweep convergence classifies stable and unstable runs correctly;
- [x] TEBD reports timestep, norm drift, bond growth, and truncation diagnostics;
- [x] checkpoint/resume, cancellation, replay, and JSON artifact export pass;
- [x] a small exact diagonalization cross-check is automatic when feasible;
- [x] frontend can run and inspect the complete workflow without backend-specific
  conditionals.

Release target: `v0.6.0` (1D research-ready, not universal).

### Phase 2 — finite 2D boundary-MPS 1.0

Status: complete within the declared finite open-2D scope. Boundary
environment checkpoints are versioned separately from physical MPS
checkpoints, 3×3 exact comparisons pass, local-observable replay and resume
are verified, and the UI studies environment `χ` with an honest verdict.

Scope:

- finite open rectangular 2D PEPS double-layer contraction;
- local observables and norm/overlap, not only global norm;
- environment bond dimension `chi` convergence studies;
- boundary sweep diagnostics per row/column;
- explicit boundary checkpoint/resume;
- exact small-lattice comparisons and an independent contraction path;
- clear rejection for periodic, 3D, and unsupported geometry.

Implementation order:

1. **Done:** isolate the boundary-MPS environment representation and preserve
   its arbitrary fused physical dimensions in a dedicated atomic checkpoint
   format.
2. **Done:** validate 2×2 and 3×3 open rectangular contractions against the
   independent opt_einsum double-layer path, including local observables.
3. **Done:** expose per-row environment bond usage, cumulative boundary
   discarded weight, request fingerprints, and resume/cancellation semantics.
4. **Done:** make the frontend finite-2D study vary environment `χ_env` and
   show boundary truncation in the shared diagnostics surface.
5. **Done:** added an independent finite-2D enumeration/reference path,
   reproducible bounded 3×3 Ising/Heisenberg GPU studies, and completed the
   live frontend replay/final audit.

Latest Phase 2 evidence:

- focused backend physics suite: 28 tests passed;
- real CUDA agreement/smoke suite: 5 tests passed, including bounded 3×3
  Ising and Heisenberg runs without dense statevector materialization;
- full agent suite: 78 tests passed, plus Python compile and frontend lint/build;
- 3×3 `χ_env=16` boundary-MPS agrees with both opt_einsum and exact
  enumeration within the test tolerance, while `χ_env=1` reports nonzero
  boundary truncation;
- cancellation/resume now restores the boundary environment for a local
  observable checkpoint, with request fingerprint and per-row diagnostics.
- live desktop verification: a 3×3 `D=4` frontend study completed all three
  `χ_env`/`dt` points, returned the expected `Needs review` verdict for its
  finite-bond error, and the stored boundary-MPS payload replayed successfully
  from local history.

Acceptance gate:

- [x] 2x2 and 3x3 random PEPS agree with an exact reference before truncation;
- [x] increasing `chi` produces a measurable convergence report;
- [x] truncation and boundary errors are not confused with physical bond errors;
- [x] local observable results survive replay and checkpoint restoration;
- [x] a 2D Ising/Heisenberg smoke study is useful without materializing a dense
  `2**N` statevector;
- [x] frontend shows method, environment `chi`, limitations, and convergence.

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

Phase 3 gate status: **passed in `v0.7.1`**.

- [x] Ising, Heisenberg, and XXZ builders use one spin-lattice plugin contract.
- [x] DMRG/TEBD and finite-2D boundary-MPS selection share preflight evidence.
- [x] Bounded studies preserve successful, failed, and canceled point evidence.
- [x] Energy range, half-range numerical uncertainty, truncation, and norm
  diagnostics are visible.
- [x] Structured observables, provenance, replay history, and JSON artifact
  export are connected end to end.
- [x] Live desktop workflow and bounded CUDA smoke tests reproduce the
  documented spin-lattice slice.

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

Current implementation status (unreleased Phase 4 work):

- bounded one-site through 2x2 CTMRG, χ studies, checkpoints, independent
  finite-PEPS references, resident-tensor Torch/CUDA autograd, and a bounded
  implicit adjoint path are implemented;
- the unrolled and implicit Torch full-update line-search states are now
  opt-in checkpoint/resume capable. Resume validates the scientific request
  hash, optimizer method, dtype, tensor shapes, iteration history, and
  evaluation budget; environments are recomputed deterministically from the
  restored tensors;
- an opt-in paired virtual-gauge probe compares energy and observables after an
  exact virtual gauge transformation;
- convergence uses a gauge-invariant boundary singular-spectrum residual while
  retaining raw basis drift as a diagnostic;
- result envelopes now expose a separate CTMRG research-gate verdict for
  convergence, independent reference, virtual gauge, truncation policy,
  transfer gap, and adjoint residual; execution success is not admission;
- D=1 finite-difference/adjoint and CUDA endpoint gates pass, but generic D=2
  cells can still fail transfer-gap, gauge-sensitivity, or adjoint-residual
  gates and remain `needs_review`;
- the default complex backward path still uses a frozen eigenprojector; an
  opt-in differentiable-eigh policy now fixes the sampled D=2 complex128
  gradient gate, but paired-gauge drift remains and this is not yet the
  `v0.8.0` research-grade release.

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

1. Phase 2: complete finite 2D boundary-MPS gate — done in `v0.7.0`.
2. Phase 3: close the spin-lattice vertical slice — done in `v0.7.1`.
3. Phase 4: CTMRG/iPEPS.
4. Phase 5: symmetry/high entanglement.
5. Phase 6: bounded 3D.
6. Phase 7: fermionic materials/chemistry.
7. Phase 8: desktop release hardening.

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

The Phase 2 and Phase 3 goal-mode tasks are complete. The active goal is now
Phase 4 CTMRG/iPEPS admission, and it must preserve the same contract-first
gates. Finite boundary-MPS must not be called CTMRG, and bounded CTMRG must
not be called a production large-3D solver.

## 7. Progress accounting

Progress should be reported in three separate numbers:

- **Platform completion:** contracts, orchestration, UI, provenance, and
  packaging foundation.
- **Capability completion:** how close the active solver is to its own gate.
- **Universal roadmap completion:** breadth against all planned solver families.

This prevents a large planned feature such as CTMRG from making a completed
and useful DMRG workflow look unfinished, while also preventing a polished UI
from hiding missing scientific validation.
