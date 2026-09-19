# Phase 4 large-work plan: covariant CTMRG environment

## 1. Purpose

The next major implementation is a gauge-covariant environment/fixed-point
strategy for the bounded iPEPS/CTMRG backend. The current solver can contract
useful product limits and narrow GHZ references, but a generic entangled D=2
cell still changes its energy under an exact paired virtual-gauge transform.
Increasing `chi` or the iteration count alone does not remove that drift.

This work must make the truncated environment map respect the declared PEPS
gauge convention before any generic entangled result is called production
research output.

The authoritative acceptance question is:

> Does the same physical iPEPS, represented in two exactly paired virtual
> gauges, produce the same declared energy/observables and consistent fixed
> point diagnostics within the dtype and truncation budget?

This plan deliberately does not promise a universal infinite-2D solver. It
targets a bounded, reproducible D=2 research capability first.

## 2. Scope and hard limits

In scope:

- one-site and existing 1x1–2x2 periodic unit cells;
- physical dimension 2 and virtual dimension 1–2 for the first gate;
- CPU NumPy reference and CUDA CuPy execution;
- Torch unrolled and implicit objective paths using the same environment map;
- full provenance, residuals, truncation, gauge, replay, and export metadata.

Out of scope until this gate passes:

- silently changing the default projector for all existing users;
- optimizer admission for generic entangled cells;
- large 3D contraction, high-entanglement chemistry, or frontend marketing
  controls;
- claiming that a finite boundary-MPS result is an infinite-lattice proof;
- large GPU campaigns or host-RAM-heavy sweeps.

Resource policy remains strict: every GPU run is bounded by preflight, the
number of points is small and explicit, and unrelated high-RAM processes are
not inspected, stopped, or competed with.

## 3. Design principles

1. The gauge convention is a first-class numerical contract, not a UI flag.
2. Environment transport, truncation, normalization, and observables must use
   the same bond orientation and inverse/conjugate convention.
3. A gauge transform must be transported through the environment or the
   environment must be rebuilt by a covariant rule; comparing two unrelated
   boundary initializations is not a gauge proof.
4. Residuals are reported in a basis-invariant form, but raw basis residuals
   remain visible for debugging.
5. A candidate is admitted only if it improves the complete gate set. A lower
   energy or a lower local Gram mismatch alone is insufficient.
6. CuPy, NumPy, and Torch paths must share the same logical map. A backend
   implementation may differ in kernels, not in tensor ordering or stopping
   semantics.
7. Experimental strategies are opt-in and labelled `needs_review` until the
   full evidence packet passes.

## 4. Work packets

### Packet A — freeze the gauge and environment contracts

Status: **passed on 2026-09-19**. The baseline map identity and ordering are
implemented in `agent/qc_agent/core/ctmrg_environment.py`; synchronous results,
research envelopes, and CTMRG checkpoints now carry the map, and resume rejects
a mismatched map. Evidence: `docs/evidence/ctmrg_environment_contract_2026-09-19.json`.

Deliverables:

- document the exact virtual-leg transform for ket, bra, incoming, and
  outgoing indices;
- define how an environment corner/edge transforms under each leg gauge;
- add a versioned environment-map identifier to CTMRG diagnostics and
  checkpoints;
- define a typed internal seam for `initialize -> sweep -> normalize ->
  residual -> observable` so candidate maps can be compared without touching
  HTTP or frontend code;
- add negative checks for incompatible tensor/environment orientation.

Acceptance:

- D=1 product behavior is unchanged;
- paired finite PEPS reference remains invariant to numerical tolerance;
- an environment-map mismatch is rejected or reported, never silently mixed.

### Packet B — implement one covariant candidate

Status: **in progress**. The first candidate is implemented and measured, but
it remains opt-in `needs_review`; the admission gate is intentionally not
passed.

The current candidate is `diagonal-bond-balance`. It applies an exact
periodic `X / X^-T` pairing using a bounded diagonal metric update, accepts
only local and global mismatch reductions that do not worsen the paired Gram
conditioning, and records its diagnostics in the result envelope. It is useful
as a conditioning baseline, but it does not remove off-diagonal or
environment-fixed-point gauge sensitivity. Evidence:
`docs/evidence/ctmrg_diagonal_bond_balance_2026-09-19.json`.

The same packet now also contains an explicit transport primitive:
`transport_ctm_environment` applies the fused `G tensor G*` double-layer map
and inverse-transpose edge action. Its resident-environment contraction gate
passes at approximately `2e-15` relative error for a random complex128 D=2
tensor. This is an algebraic seam for replay and transported fixed-point
experiments, not yet a covariant truncation algorithm. Evidence:
`docs/evidence/ctmrg_environment_transport_2026-09-19.json`.

For the truncation boundary itself, `transport_biorthogonal_boundary_basis`
now provides the exact dual pair `K @ P` and `K**(-H) @ P`. CPU complex128
and CUDA complex64 checks pass. The missing scientific step is deriving the
directional enlarged-boundary `K` generated by each absorption, then using
the dual pair without re-orthogonalizing it away. Evidence:
`docs/evidence/ctmrg_boundary_basis_transport_2026-09-19.json`.

An opt-in `biorthogonal-bilinear` projector was wired through the 1x1 backend
with the raw dual-span rule. It materially improves the generic random D=2
gate: seeds 29 and 41 pass `1e-4`, while seed 17 remains at `3.17e-4`. The
GHZ reference energy remains correct, but its paired-gauge probe drifts by
approximately `1.0`; the fixed-point controller marks that run
`converged=false`. This is a promising candidate, not a production path.
Evidence:
`docs/evidence/ctmrg_bilinear_projector_candidate_2026-09-19.json`.

The fixed-point controller now has a candidate-specific safety gate: the
bilinear path cannot report `converged` from the invariant spectrum alone; it
requires a resolved transfer gap, and a failed paired-gauge probe forces
`converged=false`. The raw boundary-basis residual remains visible as a
diagnostic rather than being treated as a sufficient invariant criterion.
Unresolved transfer degeneracy is classified explicitly as
`degenerate-needs-review`; ordinary residual/gauge failure is classified as
`unconverged`.
The same guard is applied after symmetry-sector aggregation, so an ensemble
cannot hide a failed paired-gauge probe behind a sector-wise small residual.

The bounded random D=2 gate improved for seeds 17/29/41: seeds 29 and 41
pass the declared `1e-4` paired-gauge tolerance, while seed 17 remains at
`3.17e-4`. Therefore the default projector, optimization paths, and
production admission remain unchanged. The next C2 iteration must target
environment transport or a minimal-canonical reduced-boundary construction
rather than adding another local tensor-only heuristic.

A bounded 2x2 baseline campaign is also recorded. Both half-density and
full-SVD execute with the declared site ordering, and the resident-environment
transport gate passes, but paired-gauge deltas remain `0.068--0.203` at
`chi=2`; this confirms the limitation is not only a 1x1 ordering bug.
Evidence: `docs/evidence/ctmrg_2x2_gauge_baseline_2026-09-19.json`.

The directional enlarged-boundary map is now derived and tested against the
actual one-site left/right/top/bottom absorption index order. All four maps
match to below `7e-16` in complex128. This is the first concrete `K` contract
for the boundary-basis transport primitive; it is not yet a solver admission
because the dual projector update and its discarded-weight estimate remain to
be implemented. Evidence:
`docs/evidence/ctmrg_directional_boundary_map_2026-09-19.json`.

The corresponding inverse-transpose bilinear projector-pair transport is now
tested as well: the projected grown edge transforms only by its declared
middle virtual gauge, with the retained boundary indices preserved. This
separates edge covariance from the still-open corner-basis update and is now
the seam used by the raw dual-span candidate.
The directional map and projector-pair transport are now exposed together as
a tested four-direction seam; it still does not infer a gauge or choose a
retained subspace. The same seam now runs inside the bilinear CTMRG gauge
validation result: a bounded GHZ run passes all four directions with maximum
relative error `7.31e-16`. Evidence:
`docs/evidence/ctmrg_directional_boundary_map_2026-09-19.json`.
The same runtime gate passes on CUDA complex64 at `6.42e-7` maximum relative
error (within its declared `1e-6` tolerance), while the separate fresh
fixed-point gauge gate remains false as expected.

A bounded chi probe for seed 17 was run at chi=2 and chi=3. In CPU
complex128, both points remain unconverged and the normalized residual rises
from approximately `1.61e-1` to `2.30e-1`. The corresponding CUDA complex64
probe keeps the resident-environment transport gate green, but its paired-
gauge delta remains approximately `1.72e-2` at both chi values. Increasing
chi alone is therefore not a credible admission strategy. Evidence:
`docs/evidence/ctmrg_bilinear_chi_gpu_gate_2026-09-19.json`.

An in-memory comparison of three projector normalizations confirms that the
raw dual span is currently the least-bad bounded option: overlap-inverse
normalization produces `3.0e-2--1.56e-1` gauge drift, while an SVD-balanced
dual span remains at `2.8e-4--1.07e-3` in the same six-iteration fixture.
Neither passes the `1e-4` gate, so the SVD variant was not admitted to the
code path. Evidence:
`docs/evidence/ctmrg_bilinear_projector_variant_probe_2026-09-19.json`.

An extended eight-iteration retained-subspace probe confirms the same result:
QR and biorthogonal-QR subspaces worsen the gauge gate, while SVD balancing
and the leading singular subspace of `L @ R.T` only change residual scale and
leave the paired-gauge drift unchanged; a similarity-eigenvector variant is
materially worse. The next strategy is therefore to track directional
boundary-basis gauge state through the sweep, rather than selecting a
projector from one local corner pair.
The next candidate must therefore change the reduced-boundary object itself,
not merely orthogonalize or renormalize the current corner span. Evidence:
`docs/evidence/ctmrg_bilinear_subspace_probe_2026-09-19.json`.

The solver now exposes an internal transported-environment gauge diagnostic.
On the GHZ fixture, fresh initialization drift is approximately `1.0`, while
the explicitly transported resident environment gives `1.1e-13`; on the
generic random fixture, transported deltas remain `1.13e-4--3.62e-4`.
Therefore transport removes a degenerate-sector initialization artifact but
does not solve the generic covariant truncation problem. Evidence:
`docs/evidence/ctmrg_transported_gauge_probe_2026-09-19.json`.
The result also records `initialization_sensitive` and the transported/fresh
drift ratio, so downstream replay and UI layers do not have to infer whether
transport actually helped from two unrelated numbers.

Packet C4’s current baseline consistency gate also passes: the Torch unrolled
and implicit CTMRG suite is `12/12`, including six bounded GPU tests, central
differences, adjoint residuals, and checkpoint round trips. Both paths route
their directional moves through the same `payload.ctmrg_projector` seam. The
bilinear candidate remains blocked from optimization by design until its
paired-gauge and chi gates pass. Evidence:
`docs/evidence/ctmrg_phase4_c4_autodiff_2026-09-19.json`.

Implement exactly one candidate first, selected from the existing numerical
seams after a small derivation:

- transport the boundary basis with explicit left/right inverse maps;
- build truncation projectors from a gauge-covariant reduced boundary object;
- use biorthogonal/SVD projectors where the boundary map is non-Hermitian;
- normalize corners, edges, and the transfer operator with explicit scale
  metadata;
- keep the current half-density and full-SVD paths available as baselines;
- expose the candidate behind an opt-in internal policy, not the default.

The implementation must preserve tensor shapes and site ordering for 1x1,
2x1, 1x2, and 2x2 cells. Every candidate update must return discarded weight,
condition information, raw residual, invariant residual, and the map version.

### Packet C — deterministic fixed-point behavior

Add a bounded fixed-point controller around the candidate map:

- deterministic initialization and optional transported initialization;
- residual comparison in invariant spectra plus a raw-basis diagnostic;
- damping only as an explicit control, with the same value in NumPy/CuPy/Torch;
- transfer-gap and near-degeneracy detection;
- bounded multi-start only for diagnostic studies, with sector spread reported;
- stop conditions that distinguish `converged`, `unconverged`, and
  `degenerate/needs_review`.

No convergence result may be promoted merely because the energy stopped moving.

### Packet D — make autodiff use the same map

- route Torch unrolled sweeps through the candidate environment map;
- route the implicit fixed-point/adjoint path through exactly the same map;
- keep differentiable truncation explicit and reject unsupported degeneracy;
- report forward residual, transfer gap, adjoint residual, truncation policy,
  and map version in the optimization result;
- validate central differences on small entangled tensors before any optimizer
  run is treated as evidence.

### Packet E — evidence campaign, small and reproducible

Run only the following bounded matrix first:

| Family | Cell | dtype | chi | Purpose |
| --- | --- | --- | --- | --- |
| product | 1x1 | complex64/128 | 1–4 | exact regression |
| canonical GHZ | 1x1 | complex128 | 2–4 | degenerate-sector behavior |
| random D=2, seeds 17/29/41 | 1x1 | complex128 | 2–4 | gauge and chi gates |
| random D=2, seeds 17/29 | 2x2 | complex128 | 2–3 | unit-cell ordering |
| one CUDA random seed | 1x1 | complex64 | 2 | bounded GPU parity |

For each point record:

- energy, observables, variance, fixed-point residual, raw residual;
- truncation/discarded weight and transfer gap;
- paired-gauge energy/observable deltas;
- chi/iteration differences;
- finite periodic reference error where available;
- boundary-MPS comparison only as an independent diagnostic;
- GPU memory/time and exact request fingerprint.

### Packet F — admission decision

The candidate is `passed` only if all applicable gates pass:

- D=1 energy/reference regression remains exact;
- canonical GHZ reference remains within its declared tolerance;
- random D=2 complex128 paired-gauge drift is `<=1e-4` for all three 1x1
  seeds at the declared converged point;
- the 2x2 sample does not show a new orientation/order-dependent failure;
- increasing chi produces a stable, documented convergence trend rather than
  a cherry-picked point;
- discarded weight and transfer-gap diagnostics are finite and interpretable;
- differentiable-eigh central-difference error is `<=1e-3` on the selected
  complex128 components;
- implicit adjoint residual is within its declared tolerance and the transfer
  gap is resolved;
- no condition safeguard is bypassed and no warning is suppressed.

If any gate fails, the candidate remains opt-in `needs_review`, the evidence is
kept, and the next iteration changes one numerical assumption at a time.

## 5. API and architecture changes

The public request contract should gain a versioned environment-policy field
only after Packet A defines its semantics. The field must be validated by
preflight and included in:

- synchronous CTMRG results;
- async job manifests;
- checkpoints and resume validation;
- convergence-study points and aggregate summaries;
- replay/export artifacts;
- frontend capability metadata.

The numerical candidate belongs in `agent/qc_agent/core/`, not in the server or
domain plugins. The server should only resolve, preflight, schedule, and
serialize it. The spin/materials plugins should continue to provide models and
observables without duplicating environment math.

## 6. Verification cadence

Each packet follows this order:

1. CPU unit/reference test;
2. focused CUDA smoke with a small tensor;
3. relevant backend suite;
4. evidence JSON with exact revision and limits;
5. plan/evaluation update;
6. coherent commit and GitHub push.

The active packet is never silently widened. Frontend controls are added only
after Packet F passes; until then the UI may show diagnostics and the explicit
`needs_review` status but must not present the candidate as a production
solver.

## 7. Expected checkpoints

This is one large goal, but it is intentionally divided into checkpoints:

- C1: contract and baseline tests;
- C2: candidate map on CPU product/GHZ;
- C3: CUDA parity and fixed-point diagnostics;
- C4: autodiff/implicit consistency;
- C5: random D=2 and 2x2 evidence;
- C6: admission decision and release-note update.

The goal may continue across multiple sessions. A checkpoint is complete only
when its tests and evidence are committed; a green job without a scientific
gate is not completion.

### Current checkpoint ledger (2026-09-19)

| Checkpoint | State | Evidence / meaning |
| --- | --- | --- |
| C1 | passed | Versioned environment contract and map replay checks are committed. |
| C2 | needs_review | The integrated reduced-overlap covariant selector and its four-direction transported replay pass, but the candidate remains opt-in until fresh fixed-point gates pass. |
| C3 | partial | Fixed-point classification, sector-ensemble guard, chi diagnostics, transported probes, and a covariant selector replay are implemented; initialization/residual admission remains open. |
| C4 | baseline passed | Torch unrolled/implicit map consistency is `12/12`, including six bounded GPU tests; the bilinear candidate remains blocked from optimization. |
| C5 | incomplete | The 1x1 and bounded 2x2 covariant onsite/two-site replays plus checkpoint resume are green, but fresh random D=2 gauge drift, the invariant residual plateau, and full fixed-point interaction/frame semantics still fail the production gates. |
| C6 | not reached | No production promotion or release claim is allowed until C5 passes or the candidate failure is formally closed with a replacement strategy. |

### Replay checkpoint — retained corner-basis covariance (2026-09-19)

Commit `028a387` adds `directional_sweep_covariance_replay` to the diagnostic
result. It executes the real one-site bilinear move sequence from an explicitly
transported resident environment and records factor-map, projector, moved-edge,
and contraction deltas before/after each direction.

Commit `bb68eff` adds the explicit split-basis primitive used by that replay:
the primal corner form `K·P` and dual edge form `K⁻ᵀ·P` are carried as separate
objects with independent rule and conditioning diagnostics. This is the
correct architectural seam for a tracked state, but it is not yet wired into
the production move or retained-subspace selector.

Commit `f8b5dea` adds a bounded tracked-basis prototype replay. On CPU
complex128 random D=2 seeds 17/29/41 all four directions pass with maximum
factor error `2.23e-15`; on CUDA complex64 seed 17 all four pass with maximum
moved-edge error `5.28e-7`. This is strong algebraic evidence for the split
state, but it remains `validated_prototype_not_integrated` until normalization,
damping, retained-subspace selection, checkpointing, and 2x2 semantics are
implemented.

The bounded replacement probe also closes two tempting shortcuts: damping
values `0.25, 0.5, 0.75, 1.0` leave at least one random-seed drift above
`1e-4`, and half-density/full-SVD with pairwise or diagonal preconditioning
remain above the gate. This ruled out local normalization and conditioning
heuristics; the resulting covariant reduced-boundary selector is now the
opt-in candidate described below. Evidence:
`docs/evidence/ctmrg_phase4_replacement_probe_2026-09-19.json`.

A separate four-sweep SVD-root probe also rejects plain, `S^(1/2)`, and
`S^(-1/2)` left/right subspace scalings: all three variants retain fresh-gauge
drift near `0.759--0.804` on seeds 17/29/41. Root scaling is therefore not the
missing fix; the selector must change the reduced-boundary object itself.
Evidence: `docs/evidence/ctmrg_svd_root_probe_2026-09-19.json`.

### Covariant reduced-boundary selector checkpoint (2026-09-19)

Commit `0e5f908` adds the opt-in `covariant-bilinear` environment map
(`ctmrg-covariant-bilinear-v1`). It selects a primal/dual retained pair from
the invariant reduced overlap `S = L.T @ R`, using an SVD for a genuinely
reduced frame and a transposed linear solve for the currently full-rank
`chi` path. The selector enforces `P.T @ Q = I`, rejects rank loss instead of
using a pseudoinverse, and is wired into the real one-site `_ctm_move` path.

The integrated four-direction replay now calls that real move rather than a
hand-built prototype. CPU complex128 replay passes for seeds 17/29/41. The
bounded CUDA complex64 smoke also passes, with maximum covariance-edge error
`4.42e-7` against the declared `1e-6` gate. Full regression is now `167/167`;
commit `dcedbe8` also rejects the unsupported covariant sector-ensemble
combination at payload validation instead of allowing a runtime rank failure.
Evidence: `docs/evidence/ctmrg_covariant_reduced_boundary_selector_2026-09-19.json`.

The current 1x1 checkpoint path also round-trips the covariant map and resumes
from the checkpointed environment with the same map id. Commit `1f944f8` also
enables the covariant policy for bounded 2x2 cells and runs the actual periodic
unit-cell sweep. Commit `0024a04` adds the bounded two-site interaction replay
to that same gate. Commit `cf34524` adds a versioned per-site/per-direction
retained-frame manifest, digest validation on resume, and CUDA-safe checkpoint
host/device conversion.
Commit `655aa2a` makes the covariant complex128 initialization floor explicit
(`1e-9`) and reports it in the result; the baseline and complex64 floor remain
`1e-6` for degenerate-sector stability.
Commit `af1d437` carries that policy through checkpoint metadata and resume
results so fresh and resumed runs remain directly comparable.
Commit `4e94591` exposes the transfer ratio, explicit gap, and eigenvalue
magnitudes per unit-cell site; this makes the unresolved fixed-point gate
quantitative rather than a missing diagnostic.
Because a multi-site boundary can be represented in a different retained
internal frame after transport, its replay gate compares normalized onsite
observables while retaining raw component error as a diagnostic. CPU
complex128 2x2 runs at 2/4/8 iterations pass this observable replay with
maximum errors remain below `4e-15`; the raw component discrepancy is about
`1.21--1.24` and the complex128 invariant residual now remains near `1.6e-8`.
The same CPU
replay now checks a horizontal `Z⊗Z` interaction with error
`1.4e-16--2.0e-16`; a bounded CUDA complex64 2x2 smoke passes onsite replay at
`1.15e-7` and interaction replay at `4.84e-8`, while raw component error is
`1.21`. The new 2x2 checkpoint manifest round-trips all 16 selector summaries
and its digest on CPU and CUDA; the bounded CUDA resume also passes the
interaction replay at `2.98e-8`. This is a real bounded research capability,
but not full multi-site production admission: fixed-point interaction
covariance and convergence remain separate gates. The next implementation is
therefore fixed-point/longer-sweep validation, not a looser raw-component
tolerance.

At the current 2x2 complex128 point the four transfer gaps are approximately
`2.0e-9`, so the declared finite correlation length is correctly withheld and
the fixed-point classification remains `degenerate-needs-review`.

Several bounded replacement probes were rejected with evidence rather than
silently folded into the candidate: full-SVD increases 1x1 drift to `0.323`,
the SVD-root selector breaks the one-site replay, and global preconditioning or
under-damping worsens 2x2 drift in at least one tested regime. The selected
v1 full-rank linear-solve frame remains the least-wrong map; the next change
must address the transfer-sector representation itself.
Invariant overlap factorization probes (left/right SVD bases, polar basis, and
reduced-overlap scaling) were also measured; the bases that lower residual
break the one-site replay, while the replay-safe choices do not improve the
transfer gap. Their exact values are recorded in the evidence packet.

The bounded longer-sweep campaign confirms that the remaining gate is
algorithmic rather than a missing replay tolerance. CPU complex128 2x2 at 16
iterations reduces fresh-gauge drift to `1.48e-5` and holds the invariant
residual near `1.6e-8`; CUDA complex64 at 2/4/8 iterations remains near
`1.6e-5` because it keeps the more conservative `1e-6` floor. All bounded
onsite and `Z⊗Z` replays stay green while the fixed-point classification
remains `converged=false`. Evidence for this campaign is stored alongside the
selector record.

This closes the local retained-boundary covariance seam, not Phase 4
admission. Fresh paired-gauge drift at four iterations is still
`2.51e-3--4.96e-3` on the three CPU random seeds and `1.15e-2` on the bounded
CUDA smoke. At 24 CPU iterations the drift falls to
`1.49e-9, 4.56e-11, 1.29e-13`, while the complex128 invariant environment
residual plateaus near `1.6e-8`; the transfer gap remains unresolved, so the
fixed-point classification remains `needs_review`.
The next numerical gate is covariant initialization/fixed-point convergence,
not another projector normalization shortcut.

### Dynamic retained-sector contract checkpoint (2026-09-19)

Commit `d77e136` adds `select_covariant_dynamic_boundary_frame`, the first
implementation seam for the v2 rectangular environment state. It inspects the
invariant reduced overlap `S = L.T @ R`, estimates the supported numerical
rank, and returns the largest admissible primal/dual frame up to the requested
dimension. A rank-deficient request is reduced explicitly; it is never hidden
behind a pseudoinverse or a regularizer. The report records the requested
dimension, retained dimension, rank estimate, threshold, biorthogonal error,
and an explicit `fixed_chi_admission: false` marker.

The full-rank path is delegated to the replay-tested v1 selector, preserving
the transposed linear-solve normalization. The reduced path uses the same
invariant-overlap SVD construction at the effective rank and returns a
rectangular frame. Three unit tests cover full-rank equivalence, explicit
rank-one reduction, and empty-sector rejection; the full regression is now
`170/170`.

This is an architectural contract, not yet a production solver upgrade: the
current `CTMEnvironment` stores square fixed-`chi` corners/edges and therefore
cannot consume a changing retained dimension. The next packet must introduce
an immutable rectangular boundary-state type, directional shape validation,
checkpoint serialization/digest support, and a bounded one-site move that
uses this state. Only after that state survives the existing covariance replay
and transfer-gap gates can dynamic retention be enabled in the public CTMRG
policy.

Commit `a3c69e8` now adds the immutable `DynamicCTMEnvironment` and
`BoundaryDimensions` contracts. Corners and edges validate against explicit
top/left/bottom/right dimensions, and `shape_manifest()` provides a stable
schema for the upcoming checkpoint layer. The current square environment is
unchanged; the new type is intentionally not silently substituted into old
runs. Three additional shape-contract tests keep the full regression at
`172/172`.

Commit `b123152` completes the checkpoint side of this seam with separate
`save_dynamic_ctm_checkpoint`/`load_dynamic_ctm_checkpoint` functions. The
dynamic format has its own representation id, persists the directional shape
manifest, and verifies a canonical SHA-256 digest before reconstructing the
state. A round-trip test covers tensor values, dimensions, and digest
agreement; the full regression is now `173/173`. The dynamic state still has
no public move kernel, so checkpoint support alone does not promote it to a
solver capability.

Commit `7fd8e62` adds the first contraction kernel for this state:
`apply_dynamic_covariant_bilinear_move`. It consumes enlarged left/right
factors and a grown edge, applies the invariant primal/dual selector, and
returns the projected corners and edge with their actual output shapes. A
rank-deficient synthetic case proves that the edge becomes rectangular
(`1 x d2 x 1`) instead of being padded or silently pseudoinverted; the full
regression is now `174/174`. This is a reusable directional kernel, not yet a
complete periodic CTMRG sweep: neighboring corner dimension propagation and
normalization remain the next integration gate. A bounded CUDA complex64
smoke also passes on device 0 with the same `1 x d2 x 1` output and
`5.96e-8` biorthogonal overlap error.

Commit `3da4d92` closes that next gate for a bounded one-site path. The new
`apply_dynamic_ctm_move` mirrors the existing left/right/top/bottom index
contractions, transposes only the corners whose legacy square notation hid an
axis reversal, and updates explicit top/left/bottom/right dimensions after
each move. `run_dynamic_ctm_sweep` now executes all four directions with
shape-preserving normalization. A random complex128 1x1 sweep reduces all
four sides from `2` to `1`, and its paired-gauge replay preserves the
normalized `Z` observable below `1e-10`; the full regression is `176/176`.
The same bounded CUDA complex64 sweep passes on device 0 with maximum
biorthogonal overlap error `5.96e-8`.

This is the first real dynamic boundary sweep, but it is still not the public
periodic solver: observable/interaction adapters, multi-site periodic state
propagation, convergence diagnostics, and public checkpoint resume remain
separate admission gates.

Commit `04c147b` adds `run_dynamic_ctmrg_one_site`, an experimental runner
around that sweep. It returns a serializable research-result envelope plus the
resident dynamic state, reports spectral convergence points, evaluates
normalized onsite and nearest-neighbor one-site-cell interactions, and keeps
the `needs_review`/limitation metadata explicit. The product reference gives
`Z=1`, `ZZ=1`, and energy `1.5` in complex128; the bounded CUDA complex64
smoke returns the same `Z=1` result and final dimensions `1x1x1x1`.
This makes the dynamic path usable for controlled research probes, while
multi-site periodic propagation and public payload admission remain blocked
by design until their evidence gates are implemented.

Commit `7597545` adds the periodic `run_dynamic_ctm_cell_sweep` path for
bounded 1x1--2x2 cells. It follows the existing neighbor ordering and uses
the neighboring environment's directional edge/layer for each absorption,
then reconstructs the rectangular corner orientation explicitly. A 2x2
complex128 replay executes 16 moves, reduces all four site boundaries to
dimension one, and preserves every tested onsite Z and horizontal ZZ value
under the paired virtual gauge below `1e-10`. CUDA complex64 also passes the
16-move sweep with maximum onsite delta `1.79e-7` and interaction delta
`2.24e-8`.

This closes the bounded dynamic contraction/replay gate, but not public
admission: a multi-site dynamic result runner, convergence/transfer-gap
diagnostics, and public checkpoint-resume integration still need to be wired
and audited before the candidate can replace the existing square path.

Commit `cd477f0` adds `run_dynamic_ctmrg_cell`, the bounded 1x1--2x2 result
runner. It aggregates per-site norms and onsite terms, evaluates periodic
cell interactions, emits convergence points plus transfer ratios/gaps and
eigenvalue magnitudes, and returns all dynamic states for checkpointing. The
2x2 product reference reports `Z=1`, `ZZ=1`, energy `1.5`, four transfer
gaps equal to `1.0`, and `needs_review` status as required for an experimental
path. The CUDA complex64 product smoke returns the same values.

The remaining admission work is now concentrated in state persistence for a
list of dynamic environments, resume-time digest validation, and a fresh
paired-gauge multi-site result replay after resume. Public payload policy and
optimization remain intentionally unchanged until those gates pass.

Commit `1a50944` completes the list-state checkpoint layer. A 2x2 checkpoint
now stores site-ordered manifests and one digest per environment while
preserving the single-site metadata format for compatibility. Load validates
every digest, reconstructs the directional dimensions, and returns the same
ordered list. A runner→checkpoint→load test recomputes the horizontal ZZ
observable as `1.0`; the CUDA complex64 round-trip does the same. The full
regression is now `181/181`.

The dynamic state can therefore be resumed safely at the bounded 2x2 data
structure level. A public resume API and a paired-gauge replay that starts
from the resumed multi-site state are still the final evidence gates for this
packet.

Commit `db1aa19` closes the paired-gauge resume gate at the dynamic state
level. Two independently checkpointed 2x2 states (original and exactly
paired-gauged) are loaded back, then their four normalized onsite Z values and
horizontal ZZ interaction are compared; complex128 deltas remain below
`1e-10`. The full regression is now `182/182`. This proves checkpoint
integrity plus replay consistency for the bounded dynamic state, but it does
not yet authorize the public payload path or optimization.

Commit `61b2032` exposes `run_dynamic_ctmrg_payload` as the explicit backend
entry point. It reuses the existing payload tensor/interaction contracts,
builds fresh dynamic environments, persists the final multi-site state when a
checkpoint path is supplied, and resumes from that state with a safe effective
retained dimension. The CPU payload test preserves energy `0.5` across fresh
and resumed runs; the CUDA complex64 payload smoke does the same and restores
four site digests. The full regression is now `183/183`.

This is an explicit experimental API, not a silent replacement for
`run_ctmrg`: optimization, sector ensembles, and the default public route
remain blocked until a final policy/admission review covers the dynamic result
path. The opt-in `/jobs/ctmrg/dynamic` endpoint is intentionally separate and
returns the same `needs_review` diagnostics as the backend.

Commit `0de5ed1` hardens resume admission by comparing the current payload
request SHA-256 and unit-cell metadata with the checkpoint before any dynamic
move is executed. A mismatch is rejected explicitly; the regression remains
`183/183`.

Commit `b46f5bb` adds the explicit `/jobs/ctmrg/dynamic` server route. It uses
the same GPU preflight and provenance wrapper as the existing CTMRG endpoint,
but keeps dynamic execution opt-in and preserves its `needs_review` result
status and diagnostics. Route registration is covered in the API contract
tests; the full regression is now `185/185`. The default `/jobs/ctmrg` route
and optimization paths remain unchanged.

Commit `b09a955` adds a negative entangled 2x2 admission test. A deterministic
random D=2 payload reaches a residual near `1.6e-8`, but its four transfer gaps
remain approximately `2e-9`; the runner therefore reports
`degenerate-needs-review`, `converged=false`, and finite energy rather than
promoting the result. The full regression is `184/184`. This is the required
scientific behavior: the dynamic backend can produce a bounded result, but it
does not label an unresolved transfer sector production-ready.

- For comparison, CPU complex128 canonical GHZ on the raw candidate: left/right/bottom pass, top fails with corner
  factor relative error `5.7097e-1` and moved-edge error `1.5037`.
- For comparison, CPU random D=2 seeds 17/29/41 on the raw candidate: replay remains red, with maximum factor error
  `0.342–1.770` and moved-edge error `0.794–4.800`.
- For comparison, CUDA complex64 seed 17 on the raw candidate: local directional transport remains green, but the
  replay is red (`1.0917` factor, `2.4609` moved-edge).

The raw replay remains a negative control, not a promotion. The new covariant
selector replay carries the reduced primal/dual state through every real move
and passes the bounded CPU/CUDA 1x1 and 2x2 onsite/two-site observable
covariance gates; fresh initialization, fixed-point convergence, and
multi-site checkpoint semantics still block admission.

Commit `812af88` (rectangular-frame hardening) generalizes the dynamic selector
to genuinely rectangular left/right factor column counts. A multi-site
periodic boundary can otherwise reach an opaque low-level contraction error
when one retained row/column sector shrinks before its neighbor. The selector
now performs the same invariant-overlap SVD on the rectangular reduced overlap,
and the cell move validates shared neighboring boundary dimensions before any
`einsum`; incompatible frames are rejected explicitly as a reviewable 422
condition. The periodic cell sweep now also restarts from the last complete
cell boundary at the minimum admissible shared sector when a heterogeneous
per-site reduction is detected, and marks that result
`synchronized-sector-needs-review`. CPU tests cover the rectangular
biorthogonal frame, explicit neighbor compatibility, and the restart path.
This closes a correctness hole in the dynamic shape contract, but the
conservative fallback is not a production fixed-point proof and higher-sector
comparison remains required.

The dynamic result envelope now also carries a structured research gate with
separate checks for bounded-cell scope, residual convergence, complete energy,
resolved transfer gaps, shared retained-sector integrity, and public promotion.
The last gate is intentionally false while the endpoint remains experimental;
clients therefore cannot confuse a finite contraction with an admitted
production observable. Product, unresolved-entangled, and synchronized-sector
tests all assert the corresponding gate behavior.

The backend registry exposes this route as a separate `ipeps-ctmrg-dynamic`
experimental capability. The ordinary `ctmrg` resolver remains pinned to the
existing square path, so adding future boundary-MPS or material-specific
variants can follow the same explicit capability pattern without silently
changing a user's solver.

Dynamic payload execution now runs the existing independent references in the
same bounded order as the square path: finite product, analytic GHZ, then the
exact finite 2x2 PEPS contraction where applicable. The selected reference,
energy/observable error, and pass/fail state are persisted in the result and
checkpoint-resumable envelope. A random D=2 negative probe is finite and
measurable but fails the finite-torus comparison by about `2.64e-2`, so its
independent-reference gate remains red.

The dynamic invariant-residual path also now uses backend-native singular-value
APIs for Torch (`linalg.svdvals`) while retaining the NumPy/CuPy path. A Torch
CPU payload parity test is green; the CUDA server path remains CuPy-native and
bounded by the same preflight guard.

The explicit payload can now request the existing finite-cylinder boundary-MPS
cross-check. Its patch/bond diagnostics and observable error are returned as a
separate validation record and become a research gate only when requested. A
bounded 2x2 product probe passes this cross-check; it remains finite-boundary
evidence and does not replace the unresolved transfer-sector fixed-point gate.

The dynamic backend now also exposes a fresh-point convergence study at
`/jobs/ctmrg/dynamic/convergence`. It reports energy/observable deltas,
retained dimensions, transfer-gap minima, reference errors, synchronization
restarts, and per-point research gates. The bounded CPU and CUDA `[1, 2]`
studies show the intended diagnostic split: the product energy and finite
reference are stable, while the `chi=2` transfer sector remains unresolved.

Commit `c9536f3` adds the next diagnostic seam: the dynamic payload records an
optional deterministic initialization-sector seed, and
`/jobs/ctmrg/dynamic/sectors` compares up to eight fresh seeds under the same
bounded GPU preflight. The study reports seed-wise energy, residual,
transfer-gap, retained-shape, reference, and research-gate records. It is
deliberately not an ensemble average and never chooses a physically preferred
sector automatically. CPU regression is now `193/193`; the bounded CUDA
complex64 endpoint smoke with seeds `[null, 1]` passed preflight and provenance,
while both points correctly remained `needs_review`. The study admission also
now prices the full sequential point count instead of admitting only the
single-point estimate, and rejects the request when the aggregate estimate
exceeds the caller's time budget. The CPU regression is `194/194` after this
budgeting fix.

Commit `30e6f5a` adds the first sector-aware transfer fixed-point diagnostic.
The opt-in dynamic payload can repeatedly apply a complete unit-cell row
transfer to a bounded finite-width boundary-MPS, reporting normalized period
residuals, Rayleigh quotients, discarded weight, and retained bond dimension.
The product D=1 CUDA probe converges with zero residual; the random D=2 seed83
probe reports residual `8.31e-3` and remains `needs_review`, alongside the
unresolved CTMRG transfer gap. The diagnostic is explicitly marked
`cpu-reference` for its independent boundary contraction and cannot promote an
infinite-lattice result by itself. Full CPU regression is now `197/197`.

Commit `6021779` promotes the fixed-point diagnostic into a bounded
width/boundary-chi convergence study at
`/jobs/ctmrg/boundary-mps-transfer-convergence` and the unified async API.
The product D=1 grid `[width 1,2] × [boundary chi 1,2]` completes all four
points with zero residual under aggregate preflight; the study still returns
`needs_review` because finite-cylinder convergence is not an infinite-lattice
proof. The async job lifecycle was exercised end-to-end and returned the
versioned convergence-study schema.

### Current Phase 4 disposition

The bounded dynamic boundary capability is complete as an opt-in research
instrument, not as a generic entangled production solver. The evidence packet
now fully characterizes the remaining failure: the covariant reduced frame is
replay-safe, finite references are available, but the full-rank `chi=2`
transfer sector can remain degenerate even when local energy is numerically
stable. The next numerical strategy is therefore explicit: develop a
sector-aware transfer fixed-point/boundary-MPS method, validate it first on the
existing `[1, 2]` convergence study and finite-cylinder reference, then rerun
the paired-gauge and optimizer gates. The first fixed-point diagnostic now
exists, but width/chi convergence and gauge covariance are still required. No
exists, and the width/chi study contract is now available; the next evidence
packet must run that grid on the random D=2 probe and compare it against CTMRG
transfer gaps and paired-gauge observables. No local projector heuristic will
be promoted in the meantime.

## 8. Definition of done

Done means the covariant candidate either passes Packet F and is promoted as a
bounded research capability, or its failure is fully characterized and the
next numerical strategy is explicitly chosen. In both cases the old baseline
remains reproducible, the capability matrix is honest, and no generic
entangled result is labelled production-ready without the required evidence.
