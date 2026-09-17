# Quantum Circuit Tool — Master Plan

Bu hujjat: loyiha yo‘l xaritasi (roadmap), arxitektura qarorlari, modul chegaralari va ketma-ket ish rejasi uchun “single source of truth”.

Frontendning aniq backlogi, UI ownership va design system kontrakti uchun [FRONTEND_ROADMAP.md](FRONTEND_ROADMAP.md) asosiy hujjat hisoblanadi.

## 0) Maqsad (Product definition)

Research-grade quantum circuit tool:
- Katta circuitlar bilan ishlaydi (editor + viz).
- Local-first compute (RTX/GPU) bilan arzon va tez.
- Reproducibility (run log + versiyalar) bilan ilmiy ishga mos.
- Pluggable compute backend (TN, statevector, sampling, keyin noise).
- Web demo + Desktop (Tauri) orqali real ish muhiti.

## 1) Asosiy prinsiplar

1) **Local-first compute**: default hisoblash user agent’da (RTX). Server compute optional.
2) **Reproducibility by design**: har job’da seed, versiya, config, device info loglanadi.
3) **Single IR (source of truth)**: UI/agent/backend hammasi bitta circuit formatdan ishlaydi.
4) **Scalable UI**: virtual render + LOD, “full matrix” faqat kichik n.
5) **Modularity**: web/backend/agent/desktop bir-biriga minimal coupling bilan bog‘lanadi.

## 2) Komponentlar (High-level architecture)

**A) Web UI (Next.js)**
- Circuit editor (drag/drop, large circuits)
- Explain/Math panel (KaTeX/LaTeX)
- Local agent’ga compute job yuborish
- Project/version UI (server bilan)

**B) Local Compute Agent (FastAPI, Python)**
- GPU detection + CUDA smoke tests
- Compute jobs: TN estimate, TN amplitudes, statevector (small n), sampling (keyin)
- Backend registry (pluggable backends)
- Budget/limits + cancellation (keyin)

**C) Backend API (Django DRF)**
- Auth, projects, versions, sharing
- Run logs + artifacts metadata
- (Optional) server-side jobs (enterprise/self-host)

**D) Desktop (Tauri)**
- Web UI reuse
- Agent auto-start + port management
- Offline/self-host ready packaging

## 3) Circuit IR (Single source of truth)

### 3.1 Format
- **OpenQASM 3**: circuit semantics (quantum ops)
- `ui_metadata.json`: UI layout + annotations + display settings

### 3.2 Minimal schema (v1)
- `qasm`: string (OpenQASM 3)
- `ui`:
  - `nodes`: `{id, type, x, y, gateRef}`
  - `wires`: qubits + classical bits
  - `annotations`: text/math blocks, references
  - `view`: zoom/grid/theme

### 3.3 Versioning model
- CircuitVersion DAG (parent pointer) + message + author + timestamps.
- Diff: IR-level (qasm changes + metadata changes).

## 4) Compute API contract (UI ↔ Agent)

Agent base: `http://127.0.0.1:<port>`

### 4.1 Endpoints (v1)
- `GET /health`
- `GET /hardware`
- `POST /jobs/bench_matmul`
- `POST /jobs/tn_estimate`
- `POST /jobs/tn_amplitudes`

### 4.2 Job payload standard
Har job’ga umumiy header qo‘shiladi (keyin):
- `job_id` (client or server generated)
- `seed`
- `budget`: `{max_time_ms, max_mem_mb}`
- `backend`: `{name, version}`

### 4.3 Artifacts standard
Job natijasi quyidagilarni qaytarishi kerak:
- `metrics` (time, memory est, path stats)
- `artifacts` (json blobs: amplitudes, histograms, compiled ir)
- `logs` (structured events)

## 5) Editor (scalable architecture)

### 5.1 Data model
MVP’da ops list bo‘lishi mumkin, lekin core model:
- `CircuitTimeline` yoki `CircuitColumns` (moments) + stable ids
- Multi-qubit ops (cx/cz/swap) topologies
- Symbolic params (theta, phi, lambda) + sweeps (keyin)

### 5.2 Rendering
- Virtualization (faqat viewport) + LOD
- Multi-qubit gate lines (control/target)
- Zoom/pan optimized (canvas/WebGL)

### 5.3 UX
- Undo/Redo
- Multi-select, copy/paste
- Snapping, alignment
- Keyboard shortcuts

## 6) Deployment strategy

### 6.1 Web demo
- UI + explain + local agent integration
- Project/version server optional

### 6.2 Desktop (Tauri)
- Agent auto-start
- Offline mode
- Update channel (keyin)

### 6.3 Self-host (labs)
- DRF + Postgres + object storage
- Optional shared compute workers

## 7) Milestones (acceptance tests bilan)

**M0 (done):** GPU+TN agent endpoints + web’dan chaqirish MVP.
- Acceptance: RTX3060’da `/hardware`, `/bench_matmul`, `/tn_amplitudes` ishlaydi.

**M1:** IR v1 + save/load + versioning.
- Acceptance: export/import QASM3 round-trip.

**M2:** Editor core scalability + cx/cz viz.
- Acceptance: 2000+ gate smooth zoom/scroll.

**M3:** Reproducible compute pipeline (runs + artifacts + logs).
- Acceptance: bir circuit run’ini boshqa mashinada replay.

**M4:** Desktop (Tauri) one-click.
- Acceptance: agent+UI offline ishlaydi.

## 8) Modul xaritasi (Module boundaries)

1) `apps/web` — UI, editor, explain, agent client.
2) `agent` — compute service, backend registry, GPU/TN backends.
3) `backend` — DRF API, versioning, sharing, logs.
4) `apps/desktop` — Tauri shell (UI reuse + agent lifecycle).
5) `docs` — specs, ADR, schemas.

## 9) Ishni ketma-ket bajarish (Execution order)

1) IR v1 spec + validator + converters (web↔ir, ir↔agent payload).
2) Agent: job schema + backend registry + structured logs.
3) Backend: version DAG + artifacts index.
4) Editor: scalable renderer + proper multi-qubit UX.
5) Desktop: tauri init + bundling + auto agent.

## 10) Decision log (qisqa)

**D1:** Local-first compute (agent) — server compute optional.
**D2:** IR = OpenQASM3 + UI metadata (JSON).

