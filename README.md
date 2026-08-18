# Quantum Circuit (Web + Local Compute)

This repo is a monorepo for a research-grade quantum circuit tool:

- `apps/web`: Next.js circuit editor + visualization
- `backend`: Django REST API (projects, versions, sharing)
- `agent`: Local Python compute agent (Qiskit/Aer + future TN/GPU backends)
- `apps/desktop`: Tauri 2 desktop shell with Rust local storage and managed local agent

## Docs

- Master plan: `D:\Programming\Quantum Circuit\docs\PLAN.md`
- Module breakdown: `D:\Programming\Quantum Circuit\docs\modules\README.md`

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

Set `QC_AGENT_COMMAND` when using a packaged agent executable. In development the shell starts `python agent/app.py` automatically.

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

Note:
- Backend remains DRF (`backend`).
- Compute agent is FastAPI (`agent`) and is intentionally separate from DRF for CUDA/TN workloads.

Quick tests:

```powershell
Invoke-RestMethod http://127.0.0.1:8788/hardware | ConvertTo-Json -Depth 6
Invoke-RestMethod -Method Post http://127.0.0.1:8788/jobs/simulate -ContentType 'application/json' -Body '{\"n_qubits\": 24, \"gates\": [{\"name\": \"h\", \"target\": 0}, {\"name\": \"rx\", \"target\": 0, \"theta\": 1.234}] }' | ConvertTo-Json -Depth 6
Invoke-RestMethod -Method Post http://127.0.0.1:8788/jobs/bench_matmul -ContentType 'application/json' -Body '{\"size\": 1024, \"iters\": 10, \"dtype\": \"fp16\"}' | ConvertTo-Json -Depth 6
Invoke-RestMethod -Method Post http://127.0.0.1:8788/jobs/tn_amplitudes -ContentType 'application/json' -Body '{\"n_qubits\":2, \"gates\": [{\"name\":\"h\",\"target\":0},{\"name\":\"cx\",\"control\":0,\"target\":1}], \"bitstrings\":[\"00\",\"11\"] }' | ConvertTo-Json -Depth 8
Invoke-RestMethod -Method Post http://127.0.0.1:8788/jobs/tn_estimate -ContentType 'application/json' -Body '{\"n_qubits\":24, \"gates\": [{\"name\":\"h\",\"target\":0},{\"name\":\"cx\",\"control\":0,\"target\":1}] }' | ConvertTo-Json -Depth 8
```

Notes:
- `tn_amplitudes` computes `amp(b) = <b| U |0...0>` via tensor-network contraction on GPU (CuPy).
- `bench_matmul` with `fp16` is a quick Tensor Core smoke test.
