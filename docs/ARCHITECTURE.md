# Quantum Circuit Platform Architecture

## 1. System Overview

The project is a local-first desktop quantum workflow. The target runtime is
split into three cooperating layers:

1. Desktop shell (`apps/desktop`, Tauri): embedded UI, local transport, SQLite,
   agent lifecycle, and export.
2. Embedded UI (`apps/web`, Next.js): circuit authoring, run control, and result
   visualization; it is not the numerical runtime.
3. Compute Agent (`agent`, FastAPI + CuPy): GPU-backed simulation and tensor-network jobs.

The Django API (`backend`, Django + DRF) is an optional queryable project
service for development or shared history. Desktop mode remains useful without
it because runs and studies are persisted locally.

The complete target compute design and staged implementation contract lives in
[`COMPUTE_ARCHITECTURE.md`](COMPUTE_ARCHITECTURE.md); that file is the source of
truth for new numerical layers.

Primary flow:

1. User edits a circuit in the web editor.
2. UI serializes to IR/OpenQASM and saves a `CircuitVersion` in backend.
3. UI submits compute jobs to the agent through the unified async envelope.
4. Saved circuit versions and, when integrated by a host workflow, run metadata
   and artifacts are persisted by the backend.
5. The agent provenance/job journal remains the source of truth for active
   local compute; backend run records provide queryable project history.

## 2. Repository Layout

- `apps/web`: frontend and editor UX
  - `app/globals.css` + `src/ui/*`: central semantic design tokens and reusable UI primitives; page components must consume these instead of introducing new repeated colors, spacing, or control styles.
  - `src/components/CircuitEditor.tsx`: circuit canvas and gate placement.
  - `src/components/ResearchWorkspace.tsx`: circuit runs, preflight, results, and validation.
  - `src/components/LatticeLab.tsx`: lattice/Hamiltonian/TEBD/DMRG/PEPS workflows.
  - `src/components/LatticeVisuals.tsx`: reusable lattice and trajectory visualizations.
  - `src/ir/*`: IR model, QASM conversion, agent payload mapping.
- `backend`: Django service
  - `projects/models.py`: `Project`, `CircuitVersion`, `Run`, `RunArtifact`, `Study`.
  - `projects/views.py`: DRF viewsets + run query filters + stale-run reconciliation.
  - `qc_backend/settings.py`: environment-driven runtime config.
- `agent`: FastAPI compute service
  - `qc_agent/server.py`: HTTP API, async job dispatch, endpoint contracts.
  - `qc_agent/core/`: backend-independent MPS, DMRG, PEPS, and observable contractions.
  - `qc_agent/plugins/`: domain registry, sparse Pauli terms, lattice builders, fermion mapping, and evolution payloads.
  - `qc_agent/backends/statevector.py`: CuPy statevector simulation/sampling.
  - `qc_agent/backends/mps.py`: GPU-native bounded-bond MPS simulation and sampling.
  - `qc_agent/backends/tn.py`: exact opt_einsum tensor-network estimate/amplitude routines.
  - `qc_agent/backends/registry.py`: capability catalog and safe backend resolution.
  - `qc_agent/backends/limits.py`: shared qubit/memory limits used before allocation.
  - `qc_agent/provenance.py`: canonical request/circuit/result hashes and execution metadata.
  - `qc_agent/jobs.py`: bounded worker pool, cooperative cancellation, GPU admission broker, and disk-backed job journal.

Domain code is intentionally above the execution kernel. The registry exposes a
small plugin boundary (`PluginInfo`, lattice preview, Hamiltonian builder), so
TEBD, fermion mappings, energy functionals, and materials-specific builders can
be added without coupling them to CUDA tensor operations. The current
`spin-lattice` plugin supports rectangular 1D/2D/3D graphs and sparse Ising,
Heisenberg, and XXZ terms. The `hubbard-materials` plugin builds spinful
Hubbard Hamiltonians and reuses the Jordan–Wigner mapper to produce the same
sparse Pauli contract consumed by observables and TEBD.

## 3. Data Model

Core entities (backend):

- `Project`: logical container.
- `CircuitVersion`: immutable saved circuit (`qasm` + UI metadata JSON).
- `Run`: one compute execution request/response with lifecycle fields.
- `RunArtifact`: normalized output blobs (`counts`, `amplitudes`, `estimate`, `raw`).
- `Study`: durable aggregate manifest for a bounded multi-run campaign; point-level provenance remains in `Run` and `RunArtifact`.

Run lifecycle states:

- `running`: accepted and in progress.
- `done`: completed successfully.
- `failed`: terminal error.
- `canceled`: user-canceled async execution.

## 4. API Boundaries

### Backend (`/api/*`)

- CRUD for projects, versions, runs, artifacts.
- Run listing supports filters: `version`, `project`, `kind`, `status`.
- Stale running rows are reconciled during run queries when result/error already exists.

### Agent (`127.0.0.1:8788`)

- `GET /health`: liveness.
- `GET /hardware`: GPU capability and device metadata.
- Sync jobs:
  - `POST /jobs/sample`
  - `POST /jobs/simulate`
  - `POST /jobs/cross_validate` (small independent CPU ↔ GPU/TN check)
  - `POST /jobs/tn_estimate`
  - `POST /jobs/tn_amplitudes`
  - `POST /jobs/sweep`
  - `POST /jobs/bench_matmul`
  - `POST /jobs/expectation`
  - `POST /jobs/tebd`
  - `POST /jobs/ground_state`
  - `POST /jobs/dmrg`
  - `POST /jobs/peps`
  - `GET /plugins`
  - `POST /plugins/{plugin_id}/lattice`
  - `POST /plugins/{plugin_id}/hamiltonian`
  - `POST /plugins/{plugin_id}/fermion_mapping`
  - `POST /plugins/{plugin_id}/hubbard`
- Async jobs:
  - `POST /async/jobs` with `{kind, payload}` for all long-running operations
    (`bench_matmul` included)
  - `POST /async/{kind}`
  - `GET /async/jobs/{job_id}` (404 when missing)
  - `POST /async/jobs/{job_id}/cancel`
  - `GET /queue` for worker and GPU reservation telemetry

## 5. Compute Path and GPU Semantics

- CuPy is the required execution backend for compute endpoints.
- `require_gpu` guards endpoints and fails fast if CUDA is unavailable.
- The default tensor-network backend uses CuPy-native MPS with an explicit bond-dimension budget.
- Physics workloads use the same MPS runtime through the plugin layer; TEBD
  reports energy trajectories, observable trajectories, norm, bond dimension,
  discarded weight, and a preflight resource estimate. Long Pauli strings use
  a parity-CX network and have an explicit locality cap.
- 2D/3D rectangular lattices can use the existing snake-ordered MPS path for
  larger low-entanglement workloads, or the native finite PEPS simple-update
  backend for bounded-width contractions. PEPS uses explicit virtual bonds and
  a bra/ket double-layer opt_einsum contraction for norms and local
  observables; it does not materialize a `2**n` statevector on the production
  path. Enumeration remains a bounded compatibility mode. Preflight prices
  `D**2` boundary width, local tensor memory, GPU headroom and time before a
  run is admitted.
- DMRG is a separate finite two-site variational MPS backend with local
  effective Hamiltonian eigensolves and sweep-level convergence history.
- Exact small-circuit tensor contraction uses `opt_einsum` (and optional `cotengra` optimizer).
- The registry exposes MPS sampling, selected amplitudes, and estimates; results include truncation and norm diagnostics.
- Statevector allocations are bounded at 20 qubits and checked against the
  requested budget and current free GPU memory before execution.
- Async job manager is bounded-worker by configuration (`QC_AGENT_JOB_WORKERS`, default `1`), with exclusive per-GPU reservations and queue timeouts.
- Every async operation uses the same queued/running/done/failed/canceled lifecycle. Long TEBD/DMRG/PEPS loops expose cooperative cancellation checkpoints.
- Sampling/async requests include budget limits (`max_qubits`, `max_shots`, `max_mem_mb`).
- Noise is implemented as explicit shot-based Pauli trajectories; parameter sweeps materialize named rotations and reuse the same backend resolver, preflight, and provenance path for every point. Independent sweep points may be distributed across distinct local CUDA devices when available.

## 6.1 Research provenance

Every successful compute response contains a `provenance` object with:

- canonical SHA-256 hashes for the complete scientific request (`problem_sha256`),
  circuit, and result;
- requested and resolved backend names;
- agent/Python/platform versions, seed, timestamps, elapsed time;
- the hardware snapshot captured by the agent.

`POST /jobs/cross_validate` compares selected amplitudes from the independent
CPU reference implementation with the GPU tensor-network backend. It is capped
at 12 qubits so validation remains a lightweight correctness check.

## 6. Reliability and Consistency Controls

- Backend run reconciliation normalizes historical inconsistent statuses.
- Agent missing-job lookup now returns proper HTTP 404.
- Frontend persists seed/device/request/result to support replay and debugging.
- Artifacts are saved separately for query-friendly history.
- Circuit versions are append-only and expose a canonical SHA-256 fingerprint;
  changing a circuit requires creating a new version.

## 7. Configuration Model

Backend config is environment-driven:

- `DJANGO_SECRET_KEY`
- `DJANGO_DEBUG` (`1`/`0`)
- `DJANGO_ALLOWED_HOSTS` (comma-separated)
- `DJANGO_CORS_ALLOW_ALL` (`1`/`0`)
- `DJANGO_CORS_ALLOWED_ORIGINS` (comma-separated)
- `DJANGO_DATABASE_URL` (PostgreSQL URL for production; SQLite remains the local default)
- `DJANGO_REQUIRE_AUTH` (`1` enables DRF authentication requirements)
- `DJANGO_ENV=production` requires a non-default `DJANGO_SECRET_KEY`

Compute-agent hardening/configuration:

- `QC_AGENT_ENV=production` requires `QC_AGENT_TOKEN` for non-health requests.
- `QC_AGENT_CORS_ALLOWED_ORIGINS` controls browser origins; `QC_AGENT_CORS_ALLOW_ALL=1`
  is intended only for isolated development.
- `NEXT_PUBLIC_QC_AGENT_TOKEN` lets a browser build call a token-protected local
  agent; desktop builds receive the token through the Tauri transport.
- `QC_AGENT_JOB_WORKERS` is clamped to a bounded worker count; the default is one.
- `QC_AGENT_JOB_STATE_DIR` relocates the durable local job journal.

Runtime profiles:

- `configs/runtime.profiles.json` defines `dev` and `research` profiles.
- `scripts/start-stack.ps1` applies profile env vars and starts all local services.
- `scripts/watch-stack.ps1` provides auto-restart watchdog behavior for `agent`, `backend`, and `web`.

Default behavior remains developer-friendly, but production can be hardened without code changes.

## 8. Current Constraints

- Async jobs are journaled locally; a process restart preserves terminal jobs and
  marks unfinished jobs as interrupted. The resource broker is deliberately
  process-local; a shared multi-machine queue still requires Redis/PostgreSQL
  worker deployment.
- SQLite is default backend DB; suitable for local dev, limited for concurrent production.
- No CPU fallback path for GPU compute endpoints.
- Agent token authentication is required when `QC_AGENT_ENV=production`; DRF authentication can be enforced with `DJANGO_REQUIRE_AUTH=1`.
- The Tauri shell is scaffolded, but native Rust compilation and packaged-agent
  distribution still need a dedicated release validation pass.
- Native PEPS and finite DMRG are available with bounded scopes: PEPS uses
  simple-update plus double-layer opt_einsum contraction, and DMRG uses an
  iterative Lanczos local solver (dense `eigh` is an explicit validation mode). CTMRG/full PEPS environments, excited-state
  targeting, finite-temperature DMRG, and broader chemistry/materials models
  still need dedicated implementations. The 12-qubit exact GPU eigensolver
  remains a validation reference, not a scalable ground-state solver.

## 9. Recommended Production Hardening

1. Move backend DB to PostgreSQL and add migration/backup policy.
2. Add authentication and service-to-service authorization.
3. Add structured logging and centralized metrics (run latency, failure rate, GPU memory).
4. Persist async job state externally (Redis/DB queue) for restart safety.
5. Add smoke/integration tests for web -> backend -> agent run lifecycle.
6. Deploy the async envelope behind Redis/PostgreSQL workers when multi-machine
   scheduling is required; the local broker is not a distributed lock.
