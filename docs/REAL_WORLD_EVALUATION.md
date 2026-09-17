# Real-world tensor-network evaluation

Date: 2026-09-17

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

The current product therefore has real utility in narrow but meaningful research workflows. The current build now persists replayable local history, point-level backend `Run/RunArtifact` records and aggregate `Study` manifests, while the frontend exposes diagnostics, bounded physics convergence studies and A/B comparison. The PEPS production path now scales past the old 16-site statevector limit, subject to double-layer boundary width and GPU preflight; CTMRG/boundary-MPS environments, resumable server-side campaigns, and production-grade large-3D methods remain separate next-stage work.
