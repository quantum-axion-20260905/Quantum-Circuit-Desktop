# Module: Local Compute Agent

FastAPI service running locally. GPU/TN compute lives here.

## Responsibilities
- Hardware detection + GPU smoke tests
- Job API contract
- Backend registry (TN, statevector, sampling, etc.)
- Reproducibility logs (seed, versions, device)
- Resource budgets + cancellation (later)

## Current endpoints
- `GET /hardware`
- `POST /jobs/bench_matmul`
- `POST /jobs/tn_estimate`
- `POST /jobs/tn_amplitudes`

