# Real-world tensor-network evaluation

Date: 2026-09-18

Bu report kichik, nazorat qilinadigan real GPU runlar bilan tool qaysi muammolarga amalda mos kelishini o‘lchaydi. Maqsad marketing benchmark emas: convergence, truncation va reference error ko‘rinishi kerak.

## Test environment

- GPU: NVIDIA GeForce RTX 3060, 12 GB VRAM; test boshida agent taxminan 11.8 GB free VRAM ko‘rsatdi.
- System RAM: test boshida taxminan 3.7 GB free; dense katta diagonalizatsiya ataylab bajarilmadi.
- Compute agent: CuPy 14.2.0, async GPU backends.
- Exact reference faqat 8 qubitgacha ishlatildi.

## Results

| Case | Backend/config | Result | Interpretation |
| --- | --- | --- | --- |
| 1D transverse Ising, 8q | DMRG, bond=8, sweeps=4 | `E=-7.640588328`; exact reference `-7.640592575`; absolute error ≈ `4.25e-6`; discarded weight `5.74e-9`; converged | Low-entanglement 1D ground-state work is already quantitatively useful. |
| 1D Ising, 8q | TEBD, bond=8, `dt=.02`, 5 steps | norm² `0.9999949`; energy drift `6.03e-5`; discarded weight `3.42e-10` | Good for short-time dynamics; `dt` and step convergence must be checked. |
| 2D Heisenberg, 3×3 | Snake-ordered MPS/DMRG, bond=4, sweeps=3 | discarded weight `0.2655`; not converged | Not production-grade as a quantitative 2D ground state at this bond. Increase bond/sweeps or use PEPS/reference. |
| 2D Heisenberg, 3×3 | Native PEPS, bond=2, 3 steps | norm² `0.99712`; discarded weight `0.00337`; energy `10.2 → 10.164` | Useful exploratory short evolution, not a converged ground-state answer. |
| 3D Ising, 2×2×2 | Native PEPS, bond=2, 2 steps | norm² `0.9999995`; discarded weight `3.66e-10`; energy `-12 → -11.999988` | Small 3D proof-of-use; larger 3D claims require convergence studies. |
| 2D Ising, 5×5 | Native PEPS, double-layer GPU contraction, bond=2, 1 step | norm² `0.9999974`; energy `-40 → -39.999254`; runtime ≈ `1.34 s` | A 25-site 2D workload now runs without a `2**25` statevector; still an approximate simple-update result. |
| 3D Heisenberg, 2×2×2 | Native PEPS, double-layer GPU contraction, bond=2, 1 step | norm² `1.0000006`; energy `2.6000000 → 2.6000015`; discarded weight `8.6e-10` | Practical bounded 3D exploratory calculation; width grows rapidly with cross-section. |
| 2×2 Hubbard, U=4 | JW mapping + DMRG, 8q, bond=4, sweeps=3 | `E=-2.7142579`; discarded weight `0.00559`; not converged | Good for materials-method prototyping and diagnostics; not yet a final production ground-state solver at this budget. |
| Two-mode fermion hopping | Jordan–Wigner mapping | 2 qubits, XX/YY terms, imaginary coefficient `0`, expectation-ready | Mapping layer is usable for small fermionic models. |

Frontend study smoke test: 1D transverse-Ising, 4 qubits, DMRG points `χ=8/sweeps=4`, `χ=16/sweeps=4` and `χ=16/sweeps=8` all returned `E=-3.42703423`, discarded weight `0` and maximum norm drift `5.96e-8`. The three points were persisted as backend `Run` records.

The same UI study for TEBD returned `E=-2.99900469 → -2.99971583` when `dt` was halved, with maximum norm drift `1.43e-5`; the UI correctly marks this as `Needs review` because the energy spread is `7.11e-4`. This is the intended behavior: a trajectory is not silently promoted to a converged result.

For a 2×2 PEPS smoke study, the UI compared `χ=2`, `χ=4` and `χ=4, dt/2`; it returned energy spread `2.22e-3`, maximum discarded weight `1.66e-5` and norm drift `2.99e-4`, therefore also `Needs review`. This confirms that the bounded PEPS path is usable for exploration while still exposing non-convergence.

## Phase 4 CTMRG/iPEPS evidence

The bounded CTMRG path was exercised directly on the RTX 3060 without an
unbounded statevector allocation. The finite-reference gradient row below is
the explicit exception: it materializes only the admitted four-site reference
vector and labels that fact in the result:

| Case | Configuration | Result | Interpretation |
| --- | --- | --- | --- |
| 2×2 periodic Ising product cell | `χ=2`, 4 CTMRG iterations | `E=-8.0`, variance `0`, product reference passed; χ=1/2 study both `-8.0` | The admitted product-limit 2D contract is reproducible and numerically stable. |
| Canonical D=2 GHZ iPEPS | `complex128`, `χ=2`, 16 iterations | `E=0.9999963`, `<Z>=-3.7e-6`, `<ZZ>=1.0`, analytic GHZ reference passed, correlation length unresolved | One named entangled transfer fixed point is validated; the degeneracy correctly prevents a finite ξ claim. |
| Generic D=2 iPEPS | `complex128`, `χ=2`, 3 iterations | CTMRG `E=0.4457706`; exact 2×2 torus reference `E=0.7112282`; max error `0.5033`; reference failed | This is useful diagnostic evidence, not a production answer. The current infinite-environment approximation needs larger χ/iterations and broader validation. |
| Generic D=2, 2×2 unit-cell iPEPS | RTX 3060 GPU, `complex128`, `χ=2`, 3 iterations | `E=0.3000`, independent finite 2×2 periodic reference performed; max error `0.03531`; reference failed | The multi-site reference path works on CUDA without a statevector and correctly keeps this entangled result in review. |
| D=2 finite-reference gradient update | RTX 3060 GPU, `complex128`, 6 steps, 13 objective evaluations | exact finite 2×2 objective `-3.9994893`, variance `1.02e-3`; CTMRG per-cell `E=-0.9998723`; finite reference passed | A real analytic tensor gradient is now validated for the bounded four-site reference. It is an initializer/reference path, not an infinite-lattice full-update claim. |
| D=1 SPSA full-update | `complex128`, one deterministic direction, 16 steps, 49 objective evaluations | energy `0 → -1.1999990`; known D=1 product reference `-1.2` | The parameter-independent update can reach a declared product limit without a statevector; its stochastic-gradient convergence flag remains explicit and it is still a non-AD baseline. |

The generic entangled cases are intentionally recorded as failed reference
comparisons. This is a release-quality behavior: the tool surfaces the mismatch
and keeps the result in `Needs review` instead of presenting a plausible number
as a converged infinite-lattice result.

The χ-study result now also carries point-to-point energy, local-observable and
interaction deltas, raw basis drift, virtual-leg conditioning, plus a
reference pass summary. This makes environment convergence evidence auditable
beyond a single energy column.

The same cases can now be replayed through the bounded campaign runner. The
recorded artifact [ctmrg_gpu_campaign_2026-09-18.json](C:/Users/shaxz/OneDrive/Dokumenty/Quantum-Circuit-Desktop/docs/evidence/ctmrg_gpu_campaign_2026-09-18.json)
covers all four admitted product cell shapes at χ=1/2/4 plus one generic D=2
2×2 entangled case. The four product cases pass all three finite-reference
points on the RTX 3060; the entangled case is intentionally recorded as
review-only until the variational full-update gate is complete.

The finite-gradient optimizer checkpoint was also exercised on CUDA: a step-2
optimizer state resumed at iteration 2 and matched an independent fresh
six-step run exactly to the recorded `1e-10` comparison threshold. This
checkpoint is specific to the finite-reference optimizer. The newer Torch
unrolled and implicit infinite-CTMRG line-search states are also resumable
under their stricter request/method/history contract; their environments are
recomputed on resume.

After the objective-seam refactor, a bounded CUDA SPSA run on the D=1 X-X
product limit used the resident-tensor CTMRG objective for seven evaluations,
returned a complete energy of `-1.0`, and preserved the explicit
`deterministic-simultaneous-perturbation` experimental label. The smoke run did
not populate the public host-side `tensor_data` field for inner candidates.

The same bounded SPSA path was also checkpointed on CUDA after two iterations
and resumed to four iterations. The resumed and fresh runs agreed within
`2.5e-7` in energy, with `start_iteration=2`; the checkpoint is still an
experimental infinite-objective baseline, not a variational convergence claim.

The coordinate and finite-difference CTMRG-feedback baselines were also
checkpointed and resumed on CUDA from iteration 1. Both matched their fresh
two-iteration runs exactly in the bounded D=1 case (`abs_delta=0`), confirming
that all three current feedback policies have resumable state contracts.

The optional Torch/CUDA autograd path was then exercised end-to-end on the same
RTX 3060 with a bounded D=1 X-X product objective (`χ=1`, one CTMRG iteration,
one optimizer step, eight objective evaluations allowed). It returned
`method=ipeps-full-update-autodiff-ctmrg`, `E=-1.0`, finite gradients, and
`gradient_backend=torch-autograd-unrolled-ctmrg`; measured free VRAM remained
about 11.1 GB after cleanup. This validates the runtime seam and GPU execution,
not a production claim: the objective is an unrolled bounded environment, not
yet an implicit CTMRG fixed-point/thermodynamic-limit gradient.

The bounded implicit-adjoint path was also run through the same CUDA API on the
D=1 X-X limit. It returned `method=ipeps-full-update-implicit-ctmrg`,
`E=-1.0`, `transfer_gap=1.0`, and zero measured adjoint residual for the
one-dimensional environment. A complex D=2 random-cell probe correctly
reported an unresolved transfer gap and nonzero adjoint residual instead of
claiming convergence. The implementation therefore passes the product-limit
runtime gate but remains `Needs review` for generic entangled cells; complex
truncation currently uses a frozen eigenprojector in backward mode.

A dedicated reproducible D=2 CUDA gradient gate confirms that this is a real
scientific limitation rather than only a precision warning. With the frozen
projector, complex64 reached residual 7.06e-6 but had maximum sampled
autodiff-vs-central-difference error 1.74e-2 and gauge energy drift 1.70e-2;
complex128 reached residual 4.12e-10 but still had gradient error 1.23e-1 and
gauge drift 3.68e-2. The opt-in differentiable-eigh policy fixes the gradient
component at complex128 (maximum error 1.60e-10) and reduces the complex64
error to 2.08e-3, but both still retain gauge drift 3.68e-2. The gate
therefore remains needs_review because the environment is not yet gauge
stable; its replayable evidence is
[ctmrg_gradient_gate_2026-09-18.json](C:/Users/shaxz/OneDrive/Dokumenty/Quantum-Circuit-Desktop/docs/evidence/ctmrg_gradient_gate_2026-09-18.json).
This separates the truncation-derivative problem from the remaining
gauge-sensitive environment problem, not from host-RAM pressure.

The result contract now makes this distinction machine-readable through
research_gate. The bounded D=1 product contract can pass its declared gates,
while entangled D=2 output remains needs_review unless the virtual-gauge gate
passes; the numerical result is never silently replaced by the finite
reference.

An opt-in paired virtual-gauge probe was added to the CTMRG request contract.
On a deterministic random D=2 cell (`χ=2`, four iterations), the exact finite
PEPS reference stayed invariant while the truncated CTMRG energy/observable
delta reached `5.68e-2` against a `1e-4` tolerance. The result surfaces this as
`gauge_validation.passed=false` and a warning, rather than treating the
truncated value as gauge-independent evidence.

The CTMRG convergence criterion was also hardened to compare normalized
corner/edge singular spectra rather than raw retained-basis entries. On the
canonical GHZ D=2 transfer fixed point, the physical residual converges below
`4e-10` while the raw basis residual remains about `1.0`; both values are now
reported so basis rotation is not confused with physical divergence.

The one-site full-SVD projector path was then aligned with the standard
source-ordered `R.T @ R_tilde` construction and explicit directional
ket/bra absorption. With a shared `1e-6` full-support initialization floor,
the complex128 canonical GHZ case (`D=2`, `χ=2`) reaches `E=1.000000000009`,
`<Z>=9.0e-12`, and `<ZZ>=1.0`; the independent GHZ reference passes. A
bounded CUDA complex64 smoke reaches the same `<ZZ>=1.0` result with a
`2,688`-byte estimated peak tensor footprint. The paired virtual-gauge probe
still reports an approximately `1.0` energy/observable drift, so this is a
validated reference/projector slice, not production entangled CTMRG.

The next isolated packet adds an opt-in `pairwise-polar-balance` preconditioner
for bounded 1x1–2x2 periodic cells. On the gauged GHZ tensor it reduces the vertical and
horizontal virtual-leg Gram mismatch from about `0.4803` to `0.0247`, lowers
the reported condition estimate from about `2.16` to `1.73`, and preserves the
independent finite-torus reference energy. It does not yet restore the CTMRG
paired-gauge observable: the gauged run still selects a symmetry-broken
`<Z>≈-1` sector. The candidate therefore remains diagnostic-only and is not
used by optimization or production admission.
The same opt-in path also completed a bounded Torch CUDA `complex64` smoke
with `E=1.00000003` and `<ZZ>=1.0`; CUDA execution does not change its
diagnostic-only admission status.

To address the GHZ transfer degeneracy directly, the opt-in
`symmetry-ensemble` policy now contracts two deterministic boundary fixed
points and averages them while reporting their spread. The canonical GHZ gate
then reaches `E=1`, `<Z>=0`, `<ZZ>=1`, and paired-gauge delta `2.54e-5` on CPU
`complex128`; the same CUDA `complex128` run reaches delta `1.0e-15`. The
sector observable spread is intentionally visible (`Δ<Z>=2.0`), and the
CUDA `complex64` version remains `needs_review` with delta `6.06e-2`. This is
a useful degenerate-sector diagnostic, not yet a generic entangled production
solver.

The broader bounded campaign keeps that distinction honest: a random D=2
1x1 cell still has paired-gauge drift about `1.09` and remains `needs_review`,
while a random 2x2 cell at `χ=2` is not yet converged and has gauge drift about
`0.247`. The ensemble is therefore a targeted fixed-point strategy for
degenerate sectors, not a universal replacement for χ-converged CTMRG.

### χ/iteration convergence packet

The follow-up CUDA sweep is recorded in
`docs/evidence/ctmrg_chi_convergence_2026-09-18.json`. For the random D=2 1x1
cell, all sampled `χ∈{2,3,4}` / iteration points remained unconverged and
the energy/observable sequence was not monotone. For the random 2x2 cell,
`χ=2..3` points also remained unconverged, with large error against the exact
finite 2x2 torus reference. The convergence-study API now carries the sector
spread through every point and reports a study-level maximum, so a small
between-χ energy delta cannot conceal a boundary-sector instability.

Decision: keep `symmetry-ensemble` opt-in and diagnostic-only. Generic
entangled CTMRG is not admitted as a production solver until a stronger
boundary-MPS/CTMRG fixed-point path passes residual, χ-convergence,
finite-reference, and paired-gauge gates together.

### Damped fixed-point update

The CTMRG contract now exposes an opt-in `environment_damping` control. Its
bounded CUDA evidence is recorded in
`docs/evidence/ctmrg_damped_fixed_point_2026-09-18.json`. For a random D=2
cell, damping `0.1` reduced the reported spectrum residual from `0.109` to
`0.047` at 64 iterations and to `0.026` at 200 iterations, but the solver
still did not converge and the sector spread became large. The result is
useful as a fixed-point diagnostic and is forwarded through the spin-lattice
plugin; it is not enabled as an optimizer or production policy.

### Independent finite-cylinder boundary-MPS reference

The CTMRG backend now has an opt-in `boundary_mps_reference` diagnostic. It
repeats the declared unit cell on a finite open patch and contracts rows as
MPOs against a boundary MPS, retaining discarded-weight and boundary-bond
diagnostics. Evidence is recorded in
`docs/evidence/ctmrg_boundary_mps_reference_2026-09-18.json`: the D=1 product
limit matches exactly, but the random D=2 CUDA run has energy discrepancies of
about `7.3e-2`–`8.7e-2` and substantial discarded weight. This is independent
diagnostic evidence, not an infinite-lattice reference or a replacement for
CTMRG; patch-size and boundary-bond convergence are still required.

The first bounded patch/bond study is recorded in
`docs/evidence/ctmrg_boundary_mps_convergence_2026-09-18.json`. Six CUDA
points over patches `2x2`–`4x4` and boundary bonds `2`–`4` move materially;
the best sampled energy error is about `7.65e-3`, while other points exceed
`0.29`, and discarded weight is non-monotone. The backend therefore reports
the study as diagnostic-only and does not use a single finite patch to promote
the infinite CTMRG gate.

The Torch unrolled and implicit full-update paths now have an explicit,
opt-in optimizer checkpoint contract. A CPU resume-vs-fresh regression matches
the final energy for both paths; checkpoint manifests bind the state to the
scientific request hash, optimizer method, dtype, tensor shapes, history, and
evaluation budget. The checkpoint stores the resident tensor cell and scalar
optimizer history only; CTMRG environments are recomputed on resume, so this
feature improves campaign/replay reliability without turning an unresolved
entangled CTMRG result into a production claim.

## Phase 2 finite-2D boundary-MPS slice

The bounded GPU boundary-MPS path was exercised on open 3×3 spin lattices
with physical bond `D=2`, environment `χ=4`, one second-order evolution step,
and `complex64` tensors. It contracts rows through the PEPS environment without
materializing a `2**9` statevector:

| Model | Energy | Norm² | Physical discarded weight | Boundary discarded weight | Environment |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2D Ising, `h=0.3` | `-12.0000 → -11.9996731` | `0.9999762` | `2.46e-5` | `1.57e-18` | `χ=4`, 3 rows |
| 2D Heisenberg, `h=0.3` | `9.3000 → 9.2970641` | `0.9997075` | `1.79e-4` | `2.03e-8` | `χ=4`, 3 rows |

These are useful bounded exploratory runs, not converged 2D ground-state
claims. The separate physical truncation and boundary-environment truncation
figures show why the Heisenberg point requires a larger physical bond and
additional convergence studies. On 3×3 random PEPS tensors, `χ=16` matched
both the opt_einsum double-layer path and the independent virtual-bond
enumeration reference within `1e-3`; `χ=1` reported nonzero environment
discarded weight and a larger reference error.

The refreshed frontend 3×3 transverse-Ising study used `D=4` with
`χ_env=8`, `χ_env=16`, and `χ_env=16, dt/2`. All three points completed through
the boundary-MPS route. It reported energies `-12.01339825`, `-11.99953264`,
and `-11.99536543`, energy spread `1.80e-2`, maximum total discarded weight
`9.76e-3`, and maximum norm error `7.49e-4`; the UI correctly returned
`Needs review`. The same `χ_env=16` payload was then replayed from local
history and completed successfully, preserving the boundary method and
environment dimension.

## Practical positioning

### Quvonchli ishlaydigan yo‘nalishlar

- 1D spin chains: Ising, Heisenberg, XXZ; ground-state energy, observables and controlled dynamics.
- Low-entanglement circuit/tensor-network validation.
- Short-time TEBD experiments with explicit `dt`, bond and truncation diagnostics.
- Small 2D/3D lattice prototyping where PEPS approximation is acceptable and bond/contraction limits are visible.
- Small Hubbard/Jordan–Wigner experiments for method development and comparison.

### Hozircha ehtiyot bo‘lish kerak bo‘lgan yo‘nalishlar

- 2D critical/highly entangled ground states with small bond dimension.
- Large 3D systems: native PEPS contraction is intentionally bounded.
- Hubbard/materials production results without bond/sweep convergence and an independent reference.
- Long-time dynamics without time-step, norm-drift and discarded-weight studies.
- “One run” natijasini ilmiy haqiqat sifatida qabul qilish. Tool warning beradi, lekin final workflow convergence sweep talab qiladi.

## What counts as a usable result

1. Preflight accepts the request within memory/time limits.
2. Norm² stays near one and discarded weight is reported.
3. DMRG energy and discarded weight stabilize across sweeps/bond dimensions.
4. TEBD/PEPS trajectory is stable under smaller `dt` and larger bond dimension.
5. Small systems are cross-checked with exact diagonalization when feasible.
6. Request, backend, hardware snapshot, provenance and result artifact remain replayable.

TEBD artifacts now preserve `dt`, per-point `norm2`/`norm_drift`, bond-dimension
growth, cumulative discarded-weight history, and a structured truncation
diagnostic. This keeps timestep and bond-dimension convergence inspectable in
the exported/replayed result rather than leaving it as a UI-only heuristic.

The current product therefore has real utility in narrow but meaningful research workflows. The current build now persists replayable local history, point-level backend `Run/RunArtifact` records and aggregate `Study` manifests, while the frontend exposes diagnostics, bounded physics convergence studies and A/B comparison. The PEPS path now scales past the old 16-site statevector limit, subject to double-layer boundary width and GPU preflight; CTMRG is now an experimental bounded infinite-2D path with explicit entangled-reference evidence, while generic high-entanglement convergence, resumable multi-step campaigns, and production-grade large-3D methods remain next-stage work.
