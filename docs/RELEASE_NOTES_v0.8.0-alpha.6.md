# Quantum Circuit Desktop v0.8.0-alpha.6

## Gauge-tracked boundary-MPS compression

This prerelease extends the Phase 4 paired-gauge replay with an opt-in
tracked-frame compression strategy:

- boundary-MPS physical output legs are unframed before SVD truncation and
  transported back afterward;
- the newly formed internal MPS bonds remain in the fresh reference frame,
  avoiding invalid fixed-size bond-gauge assumptions after χ reduction;
- the gauge study reports both ordinary transported replay and tracked-frame
  replay, so truncation improvements remain attributable and auditable;
- sync and unified async preflight now price all four replay branches per
  point;
- production admission remains disabled until this strategy is integrated with
  a general CTMRG retained-subspace/fixed-point policy.

## Validation evidence

- 203/203 agent regression tests passed.
- Random D=2 seed83 width `[1,2]`, boundary χ `[1,4]`: ordinary transported
  replay passes 3/4 points; tracked-frame replay passes 4/4, including the
  former width 2 / χ 1 point (`1.57e-15` finite-patch relative delta).
- Raw gauged all-ones boundaries remain a negative control with large drift.
- The bounded RTX 3060 async contract remains device-admitted and produces the
  versioned gauge-covariance artifact; the independent reference is still
  labelled `cpu-reference`.
