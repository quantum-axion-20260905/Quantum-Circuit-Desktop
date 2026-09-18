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

## Long-running operating agreement

This file is the single source of truth for an extended autonomous work
session. The active packet is always the first incomplete packet below; later
phases are design constraints and backlog, not permission to open unrelated
features.

### Active status — 2026-09-18

- Active phase: Phase 4, CTMRG/iPEPS research admission.
- Current release: `v0.7.1`; the next release is not tagged until the Phase 4
  gate passes.
- Current useful capability: bounded product-state CTMRG and the canonical
  D=2 GHZ transfer-fixed-point reference both have declared gates; generic
  entangled `D>1` results remain `needs_review`.
- Current scientific blocker: the paired virtual-gauge energy drift remains
  nonzero for the generic single-boundary entangled path. The source-ordered
  one-site full-SVD path now preserves the D=1 product limit and passes the
  D=2 GHZ independent-reference gate, but it is still experimental and not
  production-ready until broader gauge-invariance and χ-convergence gates
  pass. This is an environment/gauge-stability problem, not something to hide
  with looser tolerances or UI wording.
- A new opt-in `symmetry-ensemble` policy now averages two deterministic
  boundary fixed points. It restores the narrow canonical GHZ gate on CPU
  `complex128` (`Δgauge≈2.54e-5`) and CUDA `complex128` (`Δgauge≈1e-15`), but
  the sector spread is intentionally reported (`Δ<Z>=2.0`) and CUDA
  `complex64` remains `needs_review`. The generic D=2/random and 2x2 gates are
  still incomplete, so this does not promote entangled CTMRG to production.
- The next isolated candidate is now explicit: a bounded 1x1–2x2
  `pairwise-polar-balance` preconditioner reduces the GHZ virtual-leg Gram
  mismatch from about `0.4803` to `0.0247` and keeps the exact finite-torus
  reference invariant, but the CTMRG paired-gauge observable drift remains
  about `1.0`. It is therefore diagnostic-only and is not admitted into an
  optimizer or a production gate.
- Current resource rule: GPU experiments stay small and bounded; the agent
  must not inspect, stop, or compete with unrelated high-RAM training jobs.

- The follow-up χ/iteration packet is recorded in
  `docs/evidence/ctmrg_chi_convergence_2026-09-18.json` at commit `a632ced`.
  Every sampled random D=2 1x1 and 2x2 CUDA point remained unconverged; χ
  changes did not form a stable monotone sequence, and the 2x2 finite-torus
  reference errors stayed large. The convergence-study API now preserves
  sector policy, sector count, and per-point sector spread. This packet does
  not widen admission; the next target is a stronger boundary-MPS/CTMRG
  fixed-point strategy with explicit residual, χ-convergence, and finite-
  reference gates.

### Four-step delivery ladder for the active phase

The following ladder is the working sequence. Each step produces a commit and
evidence before the next step starts.

1. **Contract and runtime closure.** Carry the research-gate verdict through
   the convergence-study, async-job, export, replay, and desktop API surfaces.
   Verify the live local server against the same contracts used by tests.
2. **Gauge-stable contraction.** Investigate a covariant environment basis or
   preconditioner in isolated experiments first. Promote it only if finite
   reference energy/observables, paired-gauge invariance, residuals, and
   chi/iteration convergence all improve or stay within declared budgets.
3. **Variational entangled solver.** Re-run differentiable truncation and
   implicit-adjoint gates on small entangled cells, then add bounded optimizer
   convergence and checkpoint evidence. No generic `D>1` production label is
   allowed before this step passes.
4. **Research release and desktop surface.** Expose only the admitted
   capability in the desktop UI, add replay/export/failure/cancellation QA,
   update the capability matrix, run the full bounded gate, then tag and push
   the release.

### Checkpoint discipline

After every implementation slice the agent records: changed contracts,
focused tests, full-suite result when applicable, GPU memory/time evidence,
known limitations, git revision, and the exact next packet. A failed gate is
evidence and remains visible; it is not converted into a passing result by
changing presentation code. If a packet does not improve a declared gate, it
is reverted or kept as an explicitly experimental branch of the architecture.

### Long-session cadence

Every extended work session follows the same six-step loop so the project can
continue safely across goal-mode runs and context changes:

1. **Resume:** read this plan, inspect the working tree, current release, last
   evidence artifact, and the first incomplete packet. Do not reopen completed
   phases unless a regression is demonstrated.
2. **Scope:** select exactly one implementation packet. New ideas are recorded
   as backlog items; they do not silently enlarge the active packet.
3. **Implement:** keep domain plugins, numerical kernels, admission gates,
   persistence, and desktop presentation on their existing seams. Add a
   contract before adding a control or a backend branch.
4. **Verify:** run focused tests first, then the relevant backend suite. Use a
   small GPU smoke only when it answers a declared numerical or resource
   question. Large campaigns wait for the packet gate.
5. **Decide:** mark the packet `passed`, `needs_review`, or `blocked` with the
   measured reason. A completed async job is not a scientific pass.
6. **Checkpoint:** update the plan/evidence, commit only coherent changes,
   push the commit, and write the exact next packet. The next session starts
   from that checkpoint rather than from memory.

Resource guardrails are part of the cadence: unrelated high-RAM processes are
never inspected, stopped, or competed with; GPU probes remain bounded by the
declared preflight estimate and a short wall-clock budget; and a test that
cannot run within those limits is reduced or deferred, not forced through the
machine.

### Frontend timing rule

The frontend may be polished continuously for capabilities already admitted,
but new solver controls wait until the backend contract, preflight, result
diagnostics, replay, and at least one independent reference are stable. This
keeps the desktop useful during the long build without creating controls for
algorithms that are not yet scientifically supported.

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
- the study is also a first-class `ctmrg_convergence` unified async job with
  GPU admission, progress, cancellation/timeout callbacks, durable result
  artifacts, and provenance; a live 3-point GPU job has been verified.
- D=1 CTMRG runs now receive an independent NumPy finite-product-supercell
  comparison, including energy error, local observable error, finite-cell
  second moment, and variance; D>1 runs explicitly report that this reference
  is unavailable rather than projecting an entangled tensor onto a product
  ansatz.
- the 2x2 product path has an independent finite-PEPS double-layer cross-check
  for a nearest-neighbor observable; this remains a declared product/reference
  gate, not evidence for arbitrary entangled iPEPS.
- full-update now has an explicit bounded finite-difference-gradient strategy
  with a line search, evaluation budget, gradient diagnostics, and separate
  result method; it is a bridge for small tensors and is still not the
  scalable automatic-differentiation solver required for the release gate.
- a deterministic SPSA/simultaneous-perturbation full-update baseline now
  estimates a dense real/imaginary tensor direction with two objective
  evaluations per step, independent of parameter count; its budget,
  gradient scale, line search, and approximate/non-AD limitation are exposed
  in the result and preflight contract. Its deterministic direction stream is
  bit-mixed and its D=1 product-limit acceptance test reaches the known
  `E=-1.2` reference within the declared budget, while the stochastic
  convergence flag remains separate from that numerical result.
- CTMRG environment initialization now uses a deterministic full-support
  regularization so symmetry-degenerate D=2 transfer sectors do not produce
  accidental zero-over-zero two-site observables; unresolved transfer gaps
  are reported as an unavailable correlation length rather than a huge finite
  sentinel.
- a narrow analytic GHZ transfer-fixed-point reference now validates the
  canonical one-site virtual-D=2 tensor (`<Z>=0`, nearest-neighbor
  `<ZZ>=1`) without pretending it is a generic entangled-iPEPS reference;
  complex64 entangled runs expose a precision warning and recommend
  complex128 for reference-quality observables.
- generic virtual-D≤2 tensors now receive an independent exact finite-periodic
  PEPS double-layer comparison for every admitted 1x1–2x2 unit cell, with
  finite-size variance, explicit unit-cell/lattice metadata, and
  thermodynamic-limit limitations; disagreement remains `needs_review` instead
  of being hidden as a successful infinite-lattice result.
- environment-χ studies now expose point-to-point energy, local-observable, and
  interaction deltas together with raw boundary-basis residual, virtual-leg
  conditioning, a reference summary, and pass count, so a study cannot be
  reduced to an energy-only plot.
- the lattice domain plugin now builds periodic iPEPS CTMRG payloads for
  Ising, Heisenberg, and XXZ unit cells without putting model logic into the
  numerical core; 1x1 and 2x2 product-limit tests pass the independent
  reference gate.
- the product reference family now covers all 2x2 periodic Heisenberg bond
  components as well as the earlier Ising and 1x1 Heisenberg limits, while
  preserving the explicit finite-product limitation.
- the spin-lattice plugin exposes that builder as a first-class `build_ctmrg`
  domain action, keeping future material/chemistry builders on the same
  plugin seam instead of adding model branches to the solver.
- the desktop/frontend lattice workflow now exposes a deliberately bounded
  CTMRG panel for periodic 1x1–2x2 spin cells, with explicit initial state,
  environment-chi, iteration controls, cancellation, energy/variance/
  correlation-length/reference diagnostics, and a replayable chi study;
  larger geometries remain visibly disabled rather than being presented as
  production infinite-2D support.
- a standalone bounded GPU campaign runner now records all admitted cell
  shapes, χ-study histories, hardware/memory snapshots, request/result hashes,
  git revision, and explicit product-versus-entangled reference outcomes in a
  versioned JSON artifact.
- a bounded analytic finite-torus tensor-gradient optimizer now contracts an
  exact four-site reference, reports finite-reference energy/variance, and
  passes real/imaginary gradient checks against central differences; its CUDA
  path is verified on the RTX 3060. It is explicitly an initializer/reference,
  not an infinite-lattice CTMRG variational proof.
- the finite-torus gradient path now has a separate optimizer-state checkpoint
  format with request/dtype/shape validation, iteration/history/evaluation
  persistence, and partial-resume equivalence tests; unsupported optimizer
  checkpoint combinations remain rejected before execution.
- all CTMRG-feedback optimizer policies now share a bounded CTMRG objective
  seam; inner candidate evaluations can pass backend-resident tensors directly
  without flattening them through the public host-side `tensor_data` request,
  preserving a clean insertion point for a future AD or implicit-gradient
  backend and reducing unnecessary GPU-to-host copies.
- all three bounded CTMRG-feedback optimizer paths (coordinate,
  finite-difference, and SPSA) now have separate optimizer-state checkpoint
  contracts with exact method/request/dtype/shape validation and
  partial-resume equivalence tests on CPU and CUDA; they remain experimental
  numerical baselines rather than variational convergence proofs.
- an optional Torch/CUDA autograd backend now differentiates a bounded,
  unrolled CTMRG environment through the resident tensor seam; CPU gradient
  checks and a real RTX 3060 end-to-end smoke pass, while the result explicitly
  remains experimental and is not labelled an implicit fixed-point solver.
- the Torch path is isolated in `requirements-autodiff.txt`, keeps the default
  CuPy desktop install lightweight, and uses a bounded evaluation budget;
  small transfer-spectrum diagnostics fall back to host LAPACK if optional
  Torch/CuPy CUDA libraries expose incompatible Windows solver symbols.
- a bounded implicit CTMRG adjoint path now exists behind an explicit optimizer
  contract; its D=1 gradient matches central differences and its CUDA smoke
  reports transfer-gap, fixed-point, and adjoint residual diagnostics. Complex
  truncated eigenspaces currently use a frozen eigenprojector in backward mode,
  so the path remains `needs_review` until entangled gauge and gap gates pass.
- an opt-in paired virtual-gauge validation probe now reruns the final tensor
  through CTMRG and reports energy/observable deltas; the independent finite
  PEPS reference is gauge invariant, while a random D=2 truncated environment
  correctly remains review-only when its gauge delta exceeds tolerance.
- every CTMRG result now also reports non-mutating virtual-leg Gram spectra,
  rank estimates, and conditioning diagnostics. This is the admission seam
  for future paired PEPS preconditioning, not a claim that local whitening is
  already a valid canonicalization.
- a dedicated CUDA D=2 gradient gate now records central-difference and paired
  gauge evidence for both complex64 and complex128. Both cases currently
  remain needs_review, so the frozen-projector gradient is not promoted by
  a small residual alone.
- CTMRG results now carry a separate research-gate verdict: convergence,
  independent reference, virtual-gauge, truncation-gradient, transfer-gap,
  and adjoint-residual gates are individually reported; a completed job is no
  longer confused with production admission.
- an explicit differentiable-eigh truncation mode now exists behind the Torch
  unrolled/implicit optimizer contract. It is opt-in, fails on non-finite
  gradients instead of falling back silently, and is still experimental until
  the same D=2 gradient/gauge/reference gate passes.
- the first gate run shows differentiable-eigh complex128 fixes the sampled
  gradient error, while the paired virtual-gauge energy drift remains
  nonzero; the next packet must therefore solve environment gauge stability,
  not merely expose another derivative mode.
- CTMRG convergence now uses a boundary-basis-invariant singular-spectrum
  residual, while retaining the raw corner/edge entry residual as a separate
  diagnostic; this prevents harmless retained-basis rotations from being
  misreported as physical non-convergence.
- an opt-in full-SVD CTMRG projector policy now has its own resident-tensor
  contraction seam and resource preflight estimate. Its source-ordered
  one-site projector/absorption seam now uses
  the standard `R.T @ R_tilde` SVD construction and explicit directional
  ket/bra contractions. A shared full-support initialization floor prevents
  complex128 degenerate sectors from selecting a false symmetry-broken branch.
  The D=1 product limit and canonical D=2 GHZ independent reference pass on
  CPU; a bounded CUDA complex64 smoke also passes. The paired virtual-gauge
  gate still fails and the multi-site full-SVD path remains experimental.

The CTMRG solver now supports one-site, 2-site checkerboard, and bounded 2x2
periodic cells with imported tensors and multi-environment checkpoint/resume.
The current optimizers are a D=1 mean-field baseline, a bounded simple update,
bounded CTMRG-feedback coordinate/finite-difference/SPSA baselines, an
analytic finite-torus gradient reference path, and an optional Torch
unrolled-autograd CTMRG path, and a bounded implicit-adjoint path. The latter
two use a frozen truncation projector for complex-eigenspace stability and are
not yet scalable, gauge-validated infinite-CTMRG full ground-state solvers.
Stronger infinite-objective validation and broader frontend coverage are still
incomplete.
They must land and pass their own gate before Phase 4 can be called complete or tagged
as a release. The current frontend panel is an admitted experimental path,
not evidence that the full variational solver is complete.

The entangled-reference gate is intentionally narrow: generic virtual-bond
dimensions above one still require broader independent references and a
scalable variational optimizer before they can be promoted to production
research claims.

## Current next packet

The next implementation packet is Phase 4 variational research-grade convergence:

- keep the recorded GPU χ/iteration campaign and the new D=2 gradient gate as
  permanent replay artifacts; the product shapes pass their finite references,
  while generic entangled cells remain review-only;
- replace the frozen eigenprojector with a differentiable truncation policy,
  then rerun the central-difference, paired-gauge, χ/iteration, and finite
  periodic-reference gates on small entangled cells;
- keep the source-ordered full-SVD quarter/index and biorthogonal absorption
  gate as a permanent regression; the canonical GHZ reference now passes, but
  the policy remains out of optimizer/desktop production controls until the
  paired-gauge and broader entangled gates pass;
- validate the new covariant/symmetry-sector environment ensemble on random
  D=2 and 2x2 cells, including sector spread, χ/iteration convergence, and
  finite-periodic references; keep the policy opt-in until those gates pass;
- strengthen the implicit fixed-point/adjoint path with the same truncation
  policy, transfer-spectrum gap checks, and a reproducible backward-error
  budget before calling the optimizer research-grade; the current adjoint
  implementation is the bounded scaffold, not the completed gate;
- carry energy/variance/reference error and the new point-to-point observable
  deltas through the shared study/provenance and export APIs;
- use the now-resumable Torch line-search state for longer campaigns; strict
  request/dtype/shape/method/history checks remain mandatory, and the CTMRG
  fixed-point environment is deliberately recomputed on resume rather than
  serialized as optimizer state;
- validate the new bond-aware 1x1–2x2 PEPS gauge-preconditioning contract on
  2x2 reference cells, then add a symmetry-sector-preserving environment
  fixed-point strategy before any gauge transform is allowed to alter an
  optimization path;
- compare CTMRG local contractions against finite PEPS/reference product
  states across more than the current nearest-neighbor product gate, including
  small Ising/Heisenberg reference observables;
- add a variational/simple-update packet only after the contraction gate stays
  stable;
- keep the frontend surface limited to the admitted 1x1–2x2 contract until
  the stronger numerical acceptance tests pass; expand it only with matching
  preflight, diagnostics, replay, and reference evidence.

This plan is the source of truth for long-running work. The codebase, release
notes, and frontend should be updated to match it after every accepted phase.
