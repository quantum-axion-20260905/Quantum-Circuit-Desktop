# Quantum Circuit Desktop v0.8.0-alpha.1

## Phase 4 CTMRG research instrument

This prerelease adds the bounded dynamic CTMRG/iPEPS diagnostics needed to
investigate transfer-sector sensitivity without silently promoting unresolved
entangled results:

- deterministic initialization-sector seeds are part of the dynamic payload,
  checkpoint metadata, and research-result provenance;
- `/jobs/ctmrg/dynamic/sectors` compares up to eight fresh seed points under
  the same GPU preflight and memory budget;
- each point reports energy, residual, transfer gaps, retained shapes,
  independent-reference status, and the structured research gate;
- the study is explicitly not a symmetry-sector ensemble average and never
  selects a physically preferred fixed point automatically;
- the dynamic convergence study and the sector study remain opt-in routes;
  the ordinary CTMRG route is unchanged.

## Validation evidence

- 193/193 agent regression tests passed.
- CUDA complex64 sector endpoint smoke passed with seeds `[null, 1]`, GPU
  preflight, tensor-network provenance, and bounded 2x2/chi=2 execution.
- Both sector points correctly remained `needs_review`; this is expected
  because the generic entangled transfer-gap admission gate is unresolved.
- Evidence is recorded in
  `docs/evidence/ctmrg_dynamic_boundary_frame_contract_2026-09-19.json`.

## Declared limits

This is an alpha research instrument, not a universal production solver. It
does not yet provide a generic high-entanglement CTMRG fixed-point proof,
large 3D contraction, production chemistry, or automatic physical-sector
selection. Results with unresolved transfer gaps must remain under review.
