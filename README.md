# Quantum Circuit Desktop 0.8.0-alpha.4

This repo is a monorepo for a local-first quantum circuit and tensor-network
research workbench. The deliverable is a Tauri desktop application; the React
surface is embedded in that shell and the Python/CUDA agent owns numerical work:

- `apps/web`: Next.js circuit editor + visualization
- `backend`: Django REST API (projects, versions, sharing)
- `agent`: Local Python compute agent (CuPy statevector + GPU MPS/exact TN + CPU reference)
- `apps/desktop`: Tauri 2 desktop shell with Rust local storage and managed local agent

## Docs

- Master plan: [`docs/PLAN.md`](docs/PLAN.md)
- Long-horizon execution plan: [`docs/LONG_HORIZON_EXECUTION_PLAN.md`](docs/LONG_HORIZON_EXECUTION_PLAN.md)
- Compute architecture and staged research roadmap: [`docs/COMPUTE_ARCHITECTURE.md`](docs/COMPUTE_ARCHITECTURE.md)
- Active research-grade execution plan: [`docs/RESEARCH_GRADE_EXECUTION_PLAN.md`](docs/RESEARCH_GRADE_EXECUTION_PLAN.md)
- v0.8.0-alpha.4 release notes: [`docs/RELEASE_NOTES_v0.8.0-alpha.4.md`](docs/RELEASE_NOTES_v0.8.0-alpha.4.md)
- v0.8.0-alpha.3 release notes: [`docs/RELEASE_NOTES_v0.8.0-alpha.3.md`](docs/RELEASE_NOTES_v0.8.0-alpha.3.md)
- v0.8.0-alpha.2 release notes: [`docs/RELEASE_NOTES_v0.8.0-alpha.2.md`](docs/RELEASE_NOTES_v0.8.0-alpha.2.md)
- v0.8.0-alpha.1 release notes: [`docs/RELEASE_NOTES_v0.8.0-alpha.1.md`](docs/RELEASE_NOTES_v0.8.0-alpha.1.md)
- v0.7.1 release notes: [`docs/RELEASE_NOTES_v0.7.1.md`](docs/RELEASE_NOTES_v0.7.1.md)
- v0.7.0 release notes: [`docs/RELEASE_NOTES_v0.7.0.md`](docs/RELEASE_NOTES_v0.7.0.md)
- v0.7.0-alpha.1 release notes: [`docs/RELEASE_NOTES_v0.7.0-alpha.1.md`](docs/RELEASE_NOTES_v0.7.0-alpha.1.md)
- v0.6.0 release notes: [`docs/RELEASE_NOTES_v0.6.0.md`](docs/RELEASE_NOTES_v0.6.0.md)
- Module breakdown: [`docs/modules/README.md`](docs/modules/README.md)

## Prereqs (Windows)

- Node.js (already detected)
- Python 3.11+ (you have 3.14)
- Rust toolchain + WebView2 for the desktop build

## Desktop development

The desktop shell keeps the React renderer but routes desktop operations through Rust commands.
Without CUDA it uses the deterministic `reference-cpu` backend, so the full design → preflight → run → validation → export workflow can be tested on any machine.

```powershell
npm install
npm -w apps/desktop install
npm -w apps/desktop run tauri dev
```

Set `QC_AGENT_COMMAND` when using a packaged agent executable. In development the shell starts `python agent/app.py` automatically, allocates a free local port, and routes all renderer agent calls through the Tauri command boundary.

## Getting started

### One-command local stack

Use profile-driven orchestration scripts:

```powershell
.\scripts\start-stack.ps1 -Profile research -WithWatcher
.\scripts\status-stack.ps1
.\scripts\stop-stack.ps1
```

Profiles are stored in `configs/runtime.profiles.json`.
`research` is the recommended default for local desktop research workflows.

### Web UI

PowerShell (npm is blocked as a `.ps1` on some systems, so run npm via `node npm-cli.js`):

```powershell
$npmCli = Join-Path (Split-Path (Get-Command node).Source) 'node_modules\npm\bin\npm-cli.js'
node $npmCli install
node $npmCli run dev -w apps/web
```

The web UI can call the local compute agent at `http://127.0.0.1:8788` (buttons in the right panel).

For a static production preview (also used by the Tauri bundle):

```powershell
npm -w apps/web run build
npm -w apps/web run start
```

The editor can export/import the versioned IR document (`.qcir.json`) and export
the corresponding OpenQASM 3 text. Import accepts both formats, using the
minimal OpenQASM 3 parser for circuits emitted by this project.
In browser mode, `Save local` first writes an immutable `CircuitVersion` to the
Django API and falls back to a downloadable IR file when the API is offline;
the desktop shell uses its local SQLite store.

### Backend API

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
python backend/manage.py migrate
python backend/manage.py runserver
```

### Local compute agent

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r agent/requirements.txt
python agent/app.py
```

Agent runs on `http://127.0.0.1:8788`.

Async job metadata is journaled under `agent/.runtime/jobs/` by default (override
with `QC_AGENT_JOB_STATE_DIR`). Terminal jobs can therefore be inspected after
an agent restart; an unfinished GPU job is marked as interrupted and is not
silently reported as successful.

Long-running operations share the durable `POST /async/jobs` envelope. Set
`QC_AGENT_JOB_WORKERS` to a small bounded value when multiple GPUs are
available; each GPU job is admitted through a reservation broker and waits for
capacity instead of competing with unrelated GPU/RAM processes. `GET /queue`
shows queue and reservation telemetry.

By default the agent accepts browser requests only from the local web origins
(`http://127.0.0.1:3000` and `http://localhost:3000`). Override this with
`QC_AGENT_CORS_ALLOWED_ORIGINS`, or set `QC_AGENT_CORS_ALLOW_ALL=1` only for
isolated development.

Note:
- Backend remains DRF (`backend`).
- Compute agent is FastAPI (`agent`) and is intentionally separate from DRF for CUDA/TN workloads.

Quick tests:

```powershell
Invoke-RestMethod http://127.0.0.1:8788/hardware | ConvertTo-Json -Depth 6
Invoke-RestMethod -Method Post http://127.0.0.1:8788/jobs/simulate -ContentType 'application/json' -Body '{\"n_qubits\": 8, \"gates\": [{\"name\": \"h\", \"target\": 0}, {\"name\": \"rx\", \"target\": 0, \"theta\": 1.234}] }' | ConvertTo-Json -Depth 6
Invoke-RestMethod -Method Post http://127.0.0.1:8788/jobs/bench_matmul -ContentType 'application/json' -Body '{\"size\": 1024, \"iters\": 10, \"dtype\": \"fp16\"}' | ConvertTo-Json -Depth 6
Invoke-RestMethod -Method Post http://127.0.0.1:8788/jobs/tn_amplitudes -ContentType 'application/json' -Body '{\"n_qubits\":2, \"gates\": [{\"name\":\"h\",\"target\":0},{\"name\":\"cx\",\"control\":0,\"target\":1}], \"bitstrings\":[\"00\",\"11\"] }' | ConvertTo-Json -Depth 8
Invoke-RestMethod -Method Post http://127.0.0.1:8788/jobs/tn_estimate -ContentType 'application/json' -Body '{\"n_qubits\":24, \"gates\": [{\"name\":\"h\",\"target\":0},{\"name\":\"cx\",\"control\":0,\"target\":1}] }' | ConvertTo-Json -Depth 8
```

Notes:
- `tn_amplitudes` computes `amp(b) = <b| U |0...0>` via GPU MPS by default; set `tn_method: "contraction"` for the exact opt_einsum path on small circuits.
- MPS sampling and selected amplitudes are bounded by `bond_dim`, report `norm2` and `discarded_weight`, and can run far beyond statevector qubit limits when entanglement stays low.
- `POST /jobs/sweep` evaluates named rotation parameters; `noise` enables shot-based Pauli depolarizing/readout trajectories for realistic sampling experiments.
- `bench_matmul` with `fp16` is a quick Tensor Core smoke test.

Physics Lab endpoints:

- `GET /plugins` lists domain modules. The built-in `spin-lattice` plugin builds
  Ising, Heisenberg, and XXZ Hamiltonians on 1D/2D/3D rectangular lattices.
- `POST /plugins/spin-lattice/lattice` previews site coordinates and edges.
- `POST /plugins/spin-lattice/hamiltonian` returns a sparse Pauli Hamiltonian.
- `POST /plugins/hubbard-materials/hubbard` builds a spinful Hubbard model and
  maps it to Jordan–Wigner Pauli terms.
- `POST /plugins/hubbard-materials/fermion_mapping` maps ordered fermion
  creation/annihilation products while preserving complex coefficients.
- `POST /jobs/expectation` evaluates sparse observables and total energy.
- `POST /jobs/cross_validate_observables` compares bounded MPS observables and
  energy against the independent CPU reference on small systems.
- `POST /jobs/tebd` runs bounded-bond MPS time evolution for bounded-locality
  Pauli strings and returns energy/observable trajectories with
  truncation/provenance diagnostics.
- `POST /jobs/ground_state` computes an exact small-system ground energy on GPU
  for validating spin/Hubbard Hamiltonians; it is capped at 12 qubits.
- `POST /jobs/dmrg` runs finite two-site variational DMRG over the MPS backend
  with sweep history, convergence and truncation diagnostics.
- `POST /jobs/peps` runs native finite 2D/3D PEPS simple-update evolution for
  one- and two-site lattice terms with bounded opt_einsum boundary contraction
  (and an explicit virtual-bond enumeration fallback).
- `POST /plugins/spin-lattice/ctmrg` builds a bounded periodic 1x1–2x2 iPEPS
  problem for Ising, Heisenberg, or XXZ spin cells.
- `POST /jobs/ctmrg` contracts the admitted CTMRG environment and reports
  residual, correlation length, variance, and independent reference status.
- `POST /jobs/ctmrg/convergence` runs a bounded environment-chi study from the
  same tensor ansatz; `ctmrg` and `ctmrg_convergence` are also available through
  the unified async job route and frontend replay history.

For UI and service integrations, prefer `POST /async/jobs` with `kind` set to
`run`, `expectation`, `tebd`, `ground_state`, `dmrg`, `peps`, `ctmrg`,
`ctmrg_convergence`, `sweep`, or `bench_matmul`, then poll the returned job
id. The synchronous routes remain compatibility endpoints.

The 2D/3D MPS path remains useful for larger low-entanglement calculations.
Native PEPS is available for small bounded contractions, while DMRG provides
variational MPS ground states; neither replaces production CTMRG/PEPS or
large-scale multi-state DMRG yet.
