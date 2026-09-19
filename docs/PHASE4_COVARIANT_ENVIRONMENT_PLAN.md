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
leave the paired-gauge drift unchanged.
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
| C2 | needs_review | Directional K maps and bilinear projector seams pass algebraic tests, but the raw dual-span candidate is still opt-in. |
| C3 | partial | Fixed-point classification, sector-ensemble guard, chi diagnostics, and transported-environment probes are implemented; generic gauge covariance is not admitted. |
| C4 | baseline passed | Torch unrolled/implicit map consistency is `12/12`, including six bounded GPU tests; the bilinear candidate remains blocked from optimization. |
| C5 | incomplete | Random D=2 and existing 2x2 evidence still fail the production gauge/chi gates. |
| C6 | not reached | No production promotion or release claim is allowed until C5 passes or the candidate failure is formally closed with a replacement strategy. |

## 8. Definition of done

Done means the covariant candidate either passes Packet F and is promoted as a
bounded research capability, or its failure is fully characterized and the
next numerical strategy is explicitly chosen. In both cases the old baseline
remains reproducible, the capability matrix is honest, and no generic
entangled result is labelled production-ready without the required evidence.
