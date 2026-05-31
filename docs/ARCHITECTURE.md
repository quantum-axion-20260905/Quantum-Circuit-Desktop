# Quantum Circuit Platform Architecture

## 1. System Overview

The project is a local-first quantum workflow split into three runtime layers:

1. Web UI (`apps/web`, Next.js): circuit authoring, run control, result visualization.
2. Backend API (`backend`, Django + DRF): durable storage for projects, versions, runs, and artifacts.
3. Compute Agent (`agent`, FastAPI + CuPy): GPU-backed simulation and tensor-network jobs.

Primary flow:

1. User edits a circuit in the web editor.
2. UI serializes to IR/OpenQASM and saves a `CircuitVersion` in backend.
3. UI submits compute jobs to agent (sync or async).
4. UI persists run metadata and output back to backend.
5. UI reads run/artifact history for replay and inspection.

## 2. Repository Layout

- `apps/web`: frontend and editor UX
  - `src/components/CircuitEditor.tsx`: circuit canvas and gate placement.
  - `src/components/ExplainPanel.tsx`: compute controls, run orchestration, history.
  - `src/ir/*`: IR model, QASM conversion, agent payload mapping.
- `backend`: Django service
  - `projects/models.py`: `Project`, `CircuitVersion`, `Run`, `RunArtifact`.
  - `projects/views.py`: DRF viewsets + run query filters + stale-run reconciliation.
  - `qc_backend/settings.py`: environment-driven runtime config.
- `agent`: FastAPI compute service
  - `qc_agent/server.py`: HTTP API, async job dispatch, endpoint contracts.
  - `qc_agent/backends/statevector.py`: CuPy statevector simulation/sampling.
  - `qc_agent/backends/tn.py`: tensor-network estimate/amplitude routines.
  - `qc_agent/jobs.py`: in-memory async job manager.

## 3. Data Model

Core entities (backend):

- `Project`: logical container.
- `CircuitVersion`: immutable saved circuit (`qasm` + UI metadata JSON).
- `Run`: one compute execution request/response with lifecycle fields.
- `RunArtifact`: normalized output blobs (`counts`, `amplitudes`, `estimate`, `raw`).

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
  - `POST /jobs/tn_estimate`
  - `POST /jobs/tn_amplitudes`
  - `POST /jobs/bench_matmul`
- Async jobs:
  - `POST /async/{kind}`
  - `GET /async/jobs/{job_id}` (404 when missing)
  - `POST /async/jobs/{job_id}/cancel`

## 5. Compute Path and GPU Semantics

- CuPy is the required execution backend for compute endpoints.
- `require_gpu` guards endpoints and fails fast if CUDA is unavailable.
- Tensor-network routines use `opt_einsum` (and optional `cotengra` optimizer).
- Async job manager is single-worker by default for deterministic GPU context behavior.
- Sampling/async requests include budget limits (`max_qubits`, `max_shots`, `max_mem_mb`).

## 6. Reliability and Consistency Controls

- Backend run reconciliation normalizes historical inconsistent statuses.
- Agent missing-job lookup now returns proper HTTP 404.
- Frontend persists seed/device/request/result to support replay and debugging.
- Artifacts are saved separately for query-friendly history.

## 7. Configuration Model

Backend config is environment-driven:

- `DJANGO_SECRET_KEY`
- `DJANGO_DEBUG` (`1`/`0`)
- `DJANGO_ALLOWED_HOSTS` (comma-separated)
- `DJANGO_CORS_ALLOW_ALL` (`1`/`0`)
- `DJANGO_CORS_ALLOWED_ORIGINS` (comma-separated)

Runtime profiles:

- `configs/runtime.profiles.json` defines `dev` and `research` profiles.
- `scripts/start-stack.ps1` applies profile env vars and starts all local services.
- `scripts/watch-stack.ps1` provides auto-restart watchdog behavior for `agent`, `backend`, and `web`.

Default behavior remains developer-friendly, but production can be hardened without code changes.

## 8. Current Constraints

- Agent jobs are in-memory; process restart loses live job registry.
- SQLite is default backend DB; suitable for local dev, limited for concurrent production.
- No CPU fallback path for GPU compute endpoints.
- AuthN/AuthZ is not enabled on backend or agent APIs.
- Tauri desktop shell is not scaffolded yet; current orchestration is script-based.

## 9. Recommended Production Hardening

1. Move backend DB to PostgreSQL and add migration/backup policy.
2. Add authentication and service-to-service authorization.
3. Add structured logging and centralized metrics (run latency, failure rate, GPU memory).
4. Persist async job state externally (Redis/DB queue) for restart safety.
5. Add smoke/integration tests for web -> backend -> agent run lifecycle.
