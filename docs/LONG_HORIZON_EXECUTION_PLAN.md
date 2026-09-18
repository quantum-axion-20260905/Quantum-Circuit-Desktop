# Long-horizon execution plan

## Purpose

This is the operating plan for continuing Quantum Circuit Desktop over a long
period without losing architectural direction or confusing an experimental
backend with a production solver.

The project advances one capability at a time. A phase is complete only when
its request/result contract, reference checks, bounded GPU test, diagnostics,
frontend path, replay/export path, documentation, and release evidence all
exist. A phase can be useful before it is complete, but it is not promoted to
the next maturity level until its gate passes.

The authoritative current release is v0.7.1: a bounded spin-lattice vertical
slice covering Ising, Heisenberg, XXZ, MPS/DMRG/TEBD, and finite open-2D
boundary-MPS/PEPS.

## Long-term target

The target is not an unbounded universal solver. It is a capability-oriented
research desktop in which each domain exposes:

domain model -> representation -> backend resolver -> preflight -> solver ->
reference/convergence evidence -> structured result -> replay/export.

Each backend must declare where it is useful, where it is approximate, and
where it must reject the request.

## Execution order

### Phase 4 — CTMRG/iPEPS and infinite 2D

Goal: add a real infinite-2D tensor-network backend instead of treating finite
boundary-MPS as an infinite solver.

Work packets:

1. Define iPEPS site/unit-cell, environment, gauge, and checkpoint contracts.
2. Implement a bounded single-site and small unit-cell CTMRG environment.
3. Add environment-chi convergence, residual, correlation length, and
   observable diagnostics.
4. Validate against small Ising/Heisenberg limits and symmetry expectations.
5. Add frontend controls only after preflight and result contracts are stable.

Gate:

- CTMRG residual and environment truncation are visible.
- Results agree with an independent reference in small declared cases.
- Increasing environment chi produces an honest convergence study.
- Unsupported unit cells and widths are rejected before GPU work.

Target release: v0.8.0 experimental infinite-2D backend.

### Phase 5 — symmetry and high-entanglement support

Goal: make bond dimension useful beyond small dense tensors.

Work packets:

1. Define charge-sector metadata and operator selection rules.
2. Implement U(1) first, then Z2; defer additional symmetries until the
   sector contract is stable.
3. Add block-sparse MPS/MPO tensors, charge-preserving gates, and diagnostics.
4. Compare symmetric and dense paths on tiny systems.
5. Add memory estimates that account for block structure, not only dense shape.

Gate:

- Forbidden sector transitions are rejected.
- Symmetric and dense results agree within declared tolerance.
- Sector metadata survives checkpoint, replay, and export.
- High-entanglement requests stop at a declared budget rather than silently
  degrading.

Target release: v0.9.0 capability-specific symmetry backend.

### Phase 6 — bounded 3D contraction

Goal: support useful small and low-entanglement 3D experiments with an honest
approximation boundary.

Work packets:

1. Define 3D cross-section and contraction-width estimates.
2. Benchmark HOTRG/TRG versus boundary-tensor alternatives on tiny lattices.
3. Select one backend using measured error, memory, and GPU-time evidence.
4. Add 2x2x2 and small 3D Ising/Heisenberg references.
5. Add explicit approximate-3D warnings and strict admission limits.

Gate:

- Preflight predicts width, memory, and time before scheduling.
- The selected backend has an independent small-system comparison.
- Results expose truncation and geometry-ordering dependence.
- Large or high-entanglement cases are rejected, not marketed as production.

Target release: bounded exploratory 3D capability.

### Phase 7 — fermionic materials and chemistry

Goal: turn the existing operator/mapping prototypes into a reproducible
materials workflow.

Work packets:

1. Stabilize the fermionic operator IR and Hermiticity/sign conventions.
2. Validate Jordan-Wigner first; add Bravyi-Kitaev only after the IR is stable.
3. Add particle-number and spin-sector metadata.
4. Build small Hubbard and active-space workflows with exact references.
5. Add domain observables, input fingerprints, replay, and export.

Gate:

- Mapping phases and signs agree with an independent reference.
- Particle number and symmetry sectors are measured and reported.
- Small Hubbard/active-space energies match exact diagonalization within the
  declared limits.
- No prototype is labelled production chemistry without reference evidence.

Target release: reproducible small-to-medium fermionic domain workflow.

### Phase 8 — desktop research release

Goal: make the supported capability matrix installable and dependable on a
clean Windows machine.

Work packets:

1. Build and test the Tauri shell with the managed local agent lifecycle.
2. Verify offline-first operation, SQLite migrations, backup, and recovery.
3. Test cancellation, restart, failed jobs, and artifact export from the
   packaged desktop.
4. Add clean-machine installation and version upgrade checks.
5. Publish a capability matrix, reproducible examples, and release artifacts.

Gate:

- A fresh user can install, run, inspect, replay, and export a bounded
  calculation without developer tooling.
- Native and browser development paths use the same request/result contracts.
- Unsupported capabilities are visibly disabled or rejected.
- Logs, versions, and exported artifacts are sufficient for reproduction.

Target release: first supported desktop research distribution.

## Work cycle for every phase

Every phase follows the same sequence:

1. Freeze the scientific question and the supported limits.
2. Define request, result, provenance, checkpoint, and failure contracts.
3. Implement the core representation independently from the domain plugin.
4. Add CPU/reference tests before increasing GPU scale.
5. Run bounded GPU smoke tests with explicit memory and time budgets.
6. Connect one frontend workflow to the already-supported backend.
7. Test replay, cancellation, failure visibility, and artifact export.
8. Update the roadmap, release notes, and capability matrix.
9. Commit a coherent slice, tag only after the gate, and push the release.

No phase skips directly from implementation to marketing/demo status.

## Resource policy

- GPU jobs are exclusive, bounded, and admitted through preflight.
- Host RAM is protected by conservative memory budgets and serialized study
  points; no large Cartesian sweep is allowed by default.
- Real GPU tests start small and increase only when the previous evidence is
  stable.
- A Needs review result remains visible and is never converted to a success
  label by presentation code.
- Dense statevector materialization must be declared in the result and
  rejected when the backend's memory contract cannot support it.

## Architecture rule

The numerical core remains independent from domain plugins:

core representation -> solver method -> backend registry -> domain plugin ->
desktop/frontend adapter.

New domains may add builders, observables, and adapters, but they must not
duplicate tensor contraction, provenance, preflight, checkpoint, or result
logic. New solver families receive their own reference and acceptance contract
before being exposed in the UI.

## Phase 4 progress

The first Phase 4 foundation and numerical contraction packet is implemented
but not released:

- iPEPS unit-cell and translational interaction request validation exists;
- CTMRG resource preflight prices double-layer tensors and corner/edge
  environments without dense statevector assumptions;
- the backend registry exposes CTMRG as an explicit tensor-network method and
  rejects accidental solver substitution;
- contract and preflight tests cover unit-cell bounds, site validation, and
  statevector-free estimates.
- a bounded one-site CTMRG core now performs corner/edge growth, Hermitian
  projector truncation, normalization, residual tracking, and local Pauli
  contraction on CPU/reference arrays and the same ``xp`` seam used by CUDA;
- the synchronous and unified async lifecycle now exposes ``ctmrg`` with
  explicit preflight admission and no fallback to finite boundary-MPS;
- the first numerical reference tests verify product-state Z/X expectations,
  interaction energy, finite residuals, and statevector-free resources.
- arbitrary complex one-site tensors can now be imported with an explicit
  ``(physical, up, down, left, right)`` shape contract and virtual bond
  dimension;
- CTMRG environments can be persisted atomically and resumed with a hashed
  scientific-problem manifest, dtype/shape checks, and convergence history;
- nearest-neighbor horizontal and vertical two-site Pauli expectations now
  use an explicit two-site CTM contraction; unsupported longer displacements
  remain visible as withheld values rather than product-of-averages guesses.
- periodic unit-cell sweeps now support 2x2 cells with four independent
  environments and atomic multi-environment checkpoint/resume;
- a separate product-coordinate-descent optimizer provides a real D=1
  variational mean-field baseline, with energy history and explicit
  non-entangled limitations.
- a bounded imaginary-time simple-update baseline now applies one- and
  two-site Pauli gates, performs SVD truncation at the requested virtual bond
  dimension, and reports discarded weight and bond-dimension history;
- a bounded full-update coordinate baseline now re-evaluates CTMRG energy for
  tensor candidates, with strict parameter/evaluation limits and explicit
  optimization history.
- CTMRG results now report a bounded transfer-spectrum diagnostic, per-site
  correlation lengths, and environment spectra in both the top-level result
  and the shared `ResearchResult` contract;
- a replayable environment-dimension convergence helper runs independent
  chi=1..8-point contractions from the same tensor ansatz and records energy,
  residual, correlation length, and statevector-free resource evidence.
- the same study is now exposed through `/jobs/ctmrg/convergence` with an
  explicit nested problem contract and preflight at the largest requested
  environment dimension, so the future desktop surface does not need solver
  internals.

The CTMRG solver now supports one-site, 2-site checkerboard, and bounded 2x2
periodic cells with imported tensors and multi-environment checkpoint/resume.
The current optimizers are a D=1 mean-field baseline, a bounded simple update,
and a bounded CTMRG-feedback coordinate baseline. The latter is not yet a
scalable automatic-differentiation/full ground-state solver. Stronger
reference calculations and frontend controls are still incomplete. They must
land and pass their own gate before Phase 4 can be called complete or tagged
as a release.

## Current next packet

The next implementation packet is Phase 4 variational research-grade convergence:

- compare one-site through 2x2 unit-cell environments and checkpoint/resume
  against finite PEPS/reference product states;
- replace the bounded coordinate baseline with scalable gradient/automatic
  differentiation or an equivalent research-validated full-update solver;
- add energy/variance evidence and expose the environment-dimension study
  through the shared study/provenance API;
- compare CTMRG local contractions against finite PEPS/reference product
  states, including independent small-system references;
- add a variational/simple-update packet only after the contraction gate stays
  stable;
- keep frontend controls hidden until these numerical acceptance tests pass.

This plan is the source of truth for long-running work. The codebase, release
notes, and frontend should be updated to match it after every accepted phase.
