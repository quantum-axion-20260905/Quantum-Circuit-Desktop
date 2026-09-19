# Quantum Circuit Desktop v0.8.0-alpha.2

## Study-budget hardening

This prerelease carries forward the opt-in dynamic CTMRG sector diagnostics
from alpha.1 and fixes study admission so sequential points are priced against
the caller's aggregate time budget:

- CTMRG environment-convergence, dynamic-convergence, and dynamic-sector
  routes now expose point and total time estimates;
- a study is rejected when the total bounded estimate exceeds its time budget;
- the GPU memory estimate remains a peak estimate because points execute
  sequentially, while the time estimate scales with the number of points.

## Validation evidence

- 194/194 agent regression tests passed.
- CUDA complex64 dynamic-sector endpoint smoke passed with two seeds,
  aggregate preflight timing, provenance, and bounded 2x2/chi=2 execution.
- The scientific transfer-gap gate remains `needs_review` for unresolved
  generic entangled sectors; no production admission was widened.

See [`docs/RELEASE_NOTES_v0.8.0-alpha.1.md`](RELEASE_NOTES_v0.8.0-alpha.1.md)
for the initial sector-study contract and declared limits.
