# Quantum Circuit Desktop v0.8.0-alpha.4

## Boundary-MPS width/chi convergence study

This prerelease extends the Phase 4 transfer fixed-point diagnostic into a
bounded convergence contract:

- `/jobs/ctmrg/boundary-mps-transfer-convergence` compares a width × boundary-
  bond-dimension grid;
- the same study is available through the unified async job lifecycle;
- each point reports residual, Rayleigh quotient, discarded weight, retained
  bond dimension, and the independent `cpu-reference` identity;
- aggregate study time is admitted across all sequential points, while peak
  memory remains a per-point estimate;
- the study remains diagnostic-only and does not promote unresolved CTMRG
  transfer sectors.

## Validation evidence

- 200/200 agent regression tests passed.
- CUDA bounded product grid `[width 1,2] × [boundary chi 1,2]` completed four
  points with zero residual and aggregate preflight.
- Unified async runtime completed with `done` status and the versioned study
  schema.
- Random D=2 production admission remains `needs_review` pending the next
  width/chi/gauge evidence packet.
