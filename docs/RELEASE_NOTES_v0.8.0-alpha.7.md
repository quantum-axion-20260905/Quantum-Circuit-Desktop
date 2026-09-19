# Quantum Circuit Desktop v0.8.0-alpha.7

## Invariant transfer-spectrum reference

The Phase 4 boundary replay now includes a small exact reference for widths
one and two:

- it builds the finite row-period transfer matrix without MPS truncation;
- it compares the dominant transfer-eigenvalue magnitude for the original,
  raw-gauge, and transported-gauge controls;
- the transported spectrum is gauge invariant while the raw all-ones control
  remains intentionally non-invariant;
- the reference is bounded to width `<=2` and never materializes a large
  many-body state or changes production admission policy.

## Validation evidence

- 203/203 agent regression tests passed.
- Random D=2 seed83 width 1/2 dense reference reports transported-spectrum
  covariance at machine precision for all four grid points; the raw control
  remains non-invariant.
- The tracked-frame MPS replay remains 4/4, including width 2 / χ 1, while
  the ordinary native-frame replay remains transparently 3/4.
- Generic entangled CTMRG remains `needs_review` until transfer-gap,
  environment-χ, and variational gates are integrated with this evidence.
