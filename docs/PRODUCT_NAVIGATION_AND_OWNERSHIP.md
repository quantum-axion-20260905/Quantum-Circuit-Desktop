# Product Navigation and Ownership Map

## Purpose

This document defines:

1. Which page/tab is responsible for which user job.
2. What the full product needs to deliver for research-grade usage.
3. Where each requirement belongs in the UI.
4. Which needs require creating a new page/tab.

This is the working contract for implementation sequencing.

## Global Navigation (Top Tabs)

Current top tabs:

1. `Design`
2. `Run`
3. `Results`
4. `Theory`
5. `Code`
6. `Experiments`

## Page Responsibilities

### 1) Design
Primary job:
- Build and edit quantum circuits quickly and safely.

Owns:
- Circuit canvas and gate operations.
- Qubit/layer layout controls.
- Gate-level visual overlays (from analysis state).

Does not own:
- Job orchestration configuration.
- Full result interpretation.
- Export workflows.

Success criteria:
- User can create/modify circuit without leaving tab.
- Canvas never overflows or becomes unusable.
- Selected analysis overlays are understandable.

### 2) Run
Primary job:
- Configure and execute compute jobs against current circuit.

Owns:
- Backend/agent/GPU connection health.
- Run mode selection (`sample`, `tn_estimate`, `tn_amplitudes`).
- Parameters (`shots`, optimizer, budgets, seeds).
- Start/stop/progress/phase.

Does not own:
- Deep result comparison and reporting.
- Long-form theory explanations.

Success criteria:
- User knows exactly which button starts which run.
- User always sees current run phase and failure reason.

### 3) Results
Primary job:
- Understand what simulation produced and whether it is valid.

Owns:
- Result summary cards.
- Validation checks and warnings.
- Artifact views (counts, amplitudes, complexity).
- A/B compare metrics.

Does not own:
- New run execution controls.
- Experiment queue authoring.

Success criteria:
- User can answer: “Is this result valid?” and “How is it different from previous?”

### 4) Theory
Primary job:
- Explain the physics/math of the current circuit and output.

Owns:
- Gate math references.
- State evolution explanation.
- Context-aware formulas tied to selected region/run.

Does not own:
- Execution state or run management.

Success criteria:
- Explanations are contextual, not static text blocks.
- Research users can derive/verify claims.

### 5) Code
Primary job:
- Interoperate with external toolchains.

Owns:
- OpenQASM view/edit.
- Qiskit/Cirq code generation.
- Round-trip import/export.

Does not own:
- Numeric result interpretation.

Success criteria:
- User can copy runnable code with minimal edits.
- Importing code/QASM updates circuit predictably.

### 6) Experiments
Primary job:
- Run systematic studies (batch/sweep/reproducibility).

Owns:
- Experiment queue.
- Sweep builders.
- Run lineage.
- Bundle export (manifest + diagnostics + artifacts).

Does not own:
- Core circuit editing.

Success criteria:
- User can run many controlled variants and reproduce any run.

## Cross-Cutting Panels

### Right Inspector
Role:
- Context details for selected object (gate/run/result row).

Should contain:
- Info / Metrics / Warnings tabs.

### Bottom Workbench
Role:
- Long outputs and operational tools.

Should contain:
- Run log, artifacts table, compare detail table, notes.

## Full Product Requirement List

### A. Circuit Authoring
- Stable drag/drop, collision-safe gate placement.
- Multi-qubit gate ergonomics.
- Undo/redo reliability.
- Large-circuit navigation.

### B. Compute Orchestration
- Health checks (`UI -> DRF -> Agent -> GPU`).
- Clear run controls and phase tracking.
- Queue execution with cancel/retry.
- Pre-flight cost estimate.

### C. Scientific Correctness
- Validation checks per output type.
- Explicit warnings for incomplete/invalid outputs.
- Deterministic run metadata (seed, environment snapshot).

### D. Analysis and Comparison
- Counts distance metrics.
- Amplitude deltas.
- Complexity deltas.
- Run-to-run compare summary and detail.

### E. Reproducibility
- Export bundle with lineage.
- Re-run exact configuration.
- Sweep grouping and parent-child run mapping.

### F. Interoperability
- OpenQASM export/import.
- Qiskit/Cirq export.
- Versioned code snapshots tied to runs.

### G. Operational Reliability
- Service watchdog and restart policy.
- Failure diagnostics surface.
- Persistent queue state (future milestone).

## Requirement-to-Page Assignment

### Design
- A. Circuit Authoring

### Run
- B. Compute Orchestration (interactive single runs)
- G. Operational Reliability (status visibility)

### Results
- C. Scientific Correctness
- D. Analysis and Comparison

### Theory
- Contextual explanation layer for A/C/D outputs

### Code
- F. Interoperability

### Experiments
- B. Queue orchestration (batch)
- E. Reproducibility
- G. Reliability (queue continuity, retry policy)

## New Page/Tab Proposals

If scope grows beyond current tabs, add these only:

1. `Admin` (optional, internal)
- Service/process controls, debug toggles, environment diagnostics.
- Keep out of researcher-facing default flow.

2. `Datasets` (optional, future)
- Saved benchmark circuits and reference outputs.
- Only if library grows large enough to justify dedicated management.

No additional tab should be added until one of these two becomes necessary.

## Implementation Sequence

### Phase 1 (now)
- Enforce ownership boundaries in current tabs.
- Move controls/content to correct tab responsibility.
- Remove duplicated controls across inspector/workbench.

### Phase 2
- Complete Results + Experiments feature depth.
- Add robust compare and validation UX.

### Phase 3
- Complete Theory contextualization and Code round-trip maturity.

## Decision Rules

When adding a feature:

1. Identify user job first.
2. Place feature in owning tab from this document.
3. If feature spans tabs, keep control in owner tab and expose read-only reflection elsewhere.
4. Do not add new tab unless current tab model cannot host the job cleanly.

