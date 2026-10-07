from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException

from ...cuda import require_gpu
from ...models import TNPayload
from ...plugins.models import (
    CTMRGBoundaryMPSStudyPayload,
    CTMRGBoundaryMPSTransferGaugeStudyPayload,
    CTMRGBoundaryMPSTransferStudyPayload,
    CTMRGConvergenceStudyPayload,
    CTMRGDynamicSectorStudyPayload,
    CTMRGPayload,
    DMRGPayload,
    ExpectationPayload,
    GroundStatePayload,
    ObservableCrossValidatePayload,
    PEPSPayload,
    TEBDPayload,
)
from ...core.ctmrg import run_ctmrg, run_ctmrg_convergence_study
from ...core.ctmrg_boundary_mps import (
    run_boundary_mps_convergence_study,
    run_boundary_mps_transfer_gauge_covariance_study,
    run_boundary_mps_transfer_convergence_study,
    tensors_from_payload,
)
from ...core.ctmrg_dynamic import (
    run_dynamic_ctmrg_convergence_study,
    run_dynamic_ctmrg_payload,
    run_dynamic_ctmrg_sector_study,
)
from ...core.dmrg import run_dmrg
from ...core.ground_state import exact_ground_state
from ...core.mps_runtime import MPSRuntime
from ...core.observables import (
    cross_validate_mps_observables,
    evolve_statevector,
    reference_expectation,
    statevector_expectation,
)
from ...core.peps import run_peps
from ...plugins.tebd import run_tebd
from ...backends.preflight import (
    estimate_ctmrg as preflight_ctmrg,
    estimate_dmrg as preflight_dmrg,
    estimate_ground_state as preflight_ground_state,
    estimate_peps as preflight_peps,
    estimate_tebd as preflight_tebd,
)
from ...services.common import (
    cp,
    _gpu_free_mb,
    _hardware_snapshot,
    _require_feasible_gpu_run,
    _resolve_or_http,
    _scale_ctmrg_study_preflight,
    _sync_gpu_guard,
    _with_run_provenance,
)

router = APIRouter(tags=["physics"])


def _observable_result(
    payload: ExpectationPayload,
    resolved: str,
    *,
    progress_cb: Any = None,
    cancel_cb: Any = None,
) -> dict[str, Any]:
    if resolved == "reference":
        result = reference_expectation(payload)
    elif resolved == "statevector":
        require_gpu(cp)
        state = evolve_statevector(cp, payload)
        values = statevector_expectation(state, cp, payload.terms, payload.n_qubits)
        norm2 = float(cp.sum(cp.abs(state) ** 2).get())
        result = {"backend": "cupy-statevector-observable", "method": "statevector-observable", "norm2": norm2, "values": values}
    else:
        require_gpu(cp)
        runtime = MPSRuntime(cp, TNPayload(
            n_qubits=payload.n_qubits,
            gates=payload.gates,
            dtype=payload.dtype,
            bond_dim=payload.bond_dim,
            truncation_cutoff=payload.truncation_cutoff,
        ), progress_cb=progress_cb, cancel_cb=cancel_cb)
        values = runtime.expectation(payload.terms)
        runtime.sync()
        result = {
            "backend": "tensor-network-mps-observable",
            "method": "mps-observable",
            "norm2": runtime.norm2(),
            "values": values,
            "bond_dim_requested": payload.bond_dim,
            "bond_dim_used": runtime.bond_dim_used,
            "discarded_weight": runtime.discarded_weight,
            "approximate": runtime.discarded_weight > 1e-12,
        }
    terms = [term.model_dump(mode="json") for term in payload.terms]
    values = result.pop("values")
    energy = sum(term.coefficient * value for term, value in zip(payload.terms, values))
    result.update({
        "status": "done",
        "n_qubits": payload.n_qubits,
        "terms": terms,
        "expectations": [
            {"label": term.label, "coefficient": term.coefficient, "value": value}
            for term, value in zip(payload.terms, values)
        ],
        "energy": energy,
        "warnings": (["bond dimension truncated entanglement; inspect discarded_weight and norm2"] if result.get("approximate") else []),
    })
    return result


@router.post("/jobs/expectation")
@_sync_gpu_guard
def jobs_expectation(payload: ExpectationPayload) -> dict[str, Any]:
    if any(gate.parameter is not None for gate in payload.gates):
        raise HTTPException(status_code=422, detail="expectation requires materialized rotation parameters; use /jobs/sweep first")
    resolved = _resolve_or_http(payload.backend, "expectation")
    started_at = time.perf_counter()
    if resolved != "reference":
        require_gpu(cp)
        _require_feasible_gpu_run(payload, resolved)
    try:
        result = _observable_result(payload, resolved)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _with_run_provenance(result, payload, requested_backend=payload.backend, resolved_backend=resolved, started_at=started_at)


@router.post("/jobs/cross_validate_observables")
@_sync_gpu_guard
def jobs_cross_validate_observables(payload: ObservableCrossValidatePayload) -> dict[str, Any]:
    """Compare bounded MPS observables with the independent CPU reference."""
    if any(gate.parameter is not None for gate in payload.gates):
        raise HTTPException(status_code=422, detail="observable validation requires materialized rotation parameters")
    require_gpu(cp)
    _require_feasible_gpu_run(payload, "tensor-network")
    started_at = time.perf_counter()
    try:
        result = cross_validate_mps_observables(cp, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _with_run_provenance(
        result,
        payload,
        requested_backend="tensor-network",
        resolved_backend="observable-cross-validation",
        started_at=started_at,
    )


@router.post("/jobs/tebd")
@_sync_gpu_guard
def jobs_tebd(payload: TEBDPayload) -> dict[str, Any]:
    if any(gate.parameter is not None for gate in payload.gates):
        raise HTTPException(status_code=422, detail="TEBD requires materialized rotation parameters")
    resolved = _resolve_or_http("tensor-network", "evolve")
    require_gpu(cp)
    if payload.lattice is not None and payload.lattice.n_sites != payload.n_qubits:
        raise HTTPException(status_code=422, detail="lattice site count must equal n_qubits")
    snapshot = _hardware_snapshot()
    report = preflight_tebd(payload, gpu_free_mb=_gpu_free_mb(snapshot))
    if not report.get("feasible", False):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "TEBD request rejected by preflight budget",
                "warnings": report.get("warnings", []),
                "estimated_peak_memory_mb": report.get("estimated_peak_memory_mb"),
                "estimated_time_ms": report.get("estimated_time_ms"),
            },
        )
    started_at = time.perf_counter()
    try:
        result = run_tebd(cp, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["preflight"] = report
    return _with_run_provenance(result, payload, requested_backend="tensor-network", resolved_backend=resolved, started_at=started_at)


@router.post("/jobs/ground_state")
@_sync_gpu_guard
def jobs_ground_state(payload: GroundStatePayload) -> dict[str, Any]:
    resolved = _resolve_or_http(payload.backend, "ground_state")
    require_gpu(cp)
    report = preflight_ground_state(payload, gpu_free_mb=_gpu_free_mb(_hardware_snapshot()))
    if not report.get("feasible", False):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "ground-state request rejected by preflight budget",
                "warnings": report.get("warnings", []),
                "estimated_peak_memory_mb": report.get("estimated_peak_memory_mb"),
                "estimated_time_ms": report.get("estimated_time_ms"),
            },
        )
    started_at = time.perf_counter()
    try:
        result = exact_ground_state(cp, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["preflight"] = report
    return _with_run_provenance(
        result,
        payload,
        requested_backend=payload.backend,
        resolved_backend=resolved,
        started_at=started_at,
    )


@router.post("/jobs/dmrg")
@_sync_gpu_guard
def jobs_dmrg(payload: DMRGPayload) -> dict[str, Any]:
    resolved = _resolve_or_http(payload.backend, "dmrg")
    require_gpu(cp)
    report = preflight_dmrg(payload, gpu_free_mb=_gpu_free_mb(_hardware_snapshot()))
    if not report.get("feasible", False):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "DMRG request rejected by preflight budget",
                "warnings": report.get("warnings", []),
                "estimated_peak_memory_mb": report.get("estimated_peak_memory_mb"),
                "estimated_time_ms": report.get("estimated_time_ms"),
            },
        )
    started_at = time.perf_counter()
    try:
        result = run_dmrg(cp, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["preflight"] = report
    return _with_run_provenance(
        result, payload, requested_backend=payload.backend, resolved_backend=resolved, started_at=started_at,
    )


@router.post("/jobs/peps")
@_sync_gpu_guard
def jobs_peps(payload: PEPSPayload) -> dict[str, Any]:
    resolved = _resolve_or_http(payload.backend, "peps")
    require_gpu(cp)
    report = preflight_peps(payload, gpu_free_mb=_gpu_free_mb(_hardware_snapshot()))
    if not report.get("feasible", False):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "PEPS request rejected by preflight budget",
                "warnings": report.get("warnings", []),
                "estimated_peak_memory_mb": report.get("estimated_peak_memory_mb"),
                "estimated_time_ms": report.get("estimated_time_ms"),
            },
        )
    started_at = time.perf_counter()
    try:
        result = run_peps(cp, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["preflight"] = report
    return _with_run_provenance(
        result, payload, requested_backend=payload.backend, resolved_backend=resolved, started_at=started_at,
    )


@router.post("/jobs/ctmrg")
@_sync_gpu_guard
def jobs_ctmrg(payload: CTMRGPayload) -> dict[str, Any]:
    """Run the bounded iPEPS CTMRG contraction path."""
    resolved = _resolve_or_http(payload.backend, "ctmrg")
    require_gpu(cp)
    report = preflight_ctmrg(payload, gpu_free_mb=_gpu_free_mb(_hardware_snapshot()))
    if not report.get("feasible", False):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "CTMRG request rejected by preflight budget",
                "warnings": report.get("warnings", []),
                "estimated_peak_memory_mb": report.get("estimated_peak_memory_mb"),
                "estimated_time_ms": report.get("estimated_time_ms"),
            },
        )
    started_at = time.perf_counter()
    try:
        result = run_ctmrg(cp, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["preflight"] = report
    return _with_run_provenance(
        result, payload, requested_backend=payload.backend, resolved_backend=resolved, started_at=started_at,
    )


@router.post("/jobs/ctmrg/dynamic")
@_sync_gpu_guard
def jobs_ctmrg_dynamic(payload: CTMRGPayload) -> dict[str, Any]:
    """Run the explicit experimental rectangular-boundary CTMRG path."""
    resolved = _resolve_or_http(payload.backend, "ctmrg")
    require_gpu(cp)
    report = preflight_ctmrg(payload, gpu_free_mb=_gpu_free_mb(_hardware_snapshot()))
    if not report.get("feasible", False):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "dynamic CTMRG request rejected by preflight budget",
                "warnings": report.get("warnings", []),
                "estimated_peak_memory_mb": report.get("estimated_peak_memory_mb"),
                "estimated_time_ms": report.get("estimated_time_ms"),
            },
        )
    started_at = time.perf_counter()
    try:
        result, _ = run_dynamic_ctmrg_payload(cp, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["preflight"] = report
    return _with_run_provenance(
        result, payload, requested_backend=payload.backend, resolved_backend=resolved, started_at=started_at,
    )


@router.post("/jobs/ctmrg/convergence")
@_sync_gpu_guard
def jobs_ctmrg_convergence(payload: CTMRGConvergenceStudyPayload) -> dict[str, Any]:
    """Run independent bounded CTMRG points at requested environment chi values."""
    resolved = _resolve_or_http(payload.backend, "ctmrg")
    require_gpu(cp)
    max_payload = payload.problem.model_copy(update={
        "environment_bond_dim": max(payload.environment_bond_dims),
        "max_time_ms": payload.max_time_ms,
        "max_mem_mb": payload.max_mem_mb,
    })
    report = preflight_ctmrg(max_payload, gpu_free_mb=_gpu_free_mb(_hardware_snapshot()))
    report = _scale_ctmrg_study_preflight(
        report,
        point_count=len(payload.environment_bond_dims),
        max_time_ms=payload.max_time_ms,
    )
    if not report.get("feasible", False):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "CTMRG convergence study rejected by preflight budget",
                "warnings": report.get("warnings", []),
                "estimated_peak_memory_mb": report.get("estimated_peak_memory_mb"),
                "estimated_time_ms": report.get("estimated_time_ms"),
            },
        )
    started_at = time.perf_counter()
    try:
        result = run_ctmrg_convergence_study(cp, payload.problem, payload.environment_bond_dims)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["preflight"] = report
    return _with_run_provenance(
        result, payload, requested_backend=payload.backend, resolved_backend=resolved, started_at=started_at,
    )


@router.post("/jobs/ctmrg/dynamic/convergence")
@_sync_gpu_guard
def jobs_ctmrg_dynamic_convergence(payload: CTMRGConvergenceStudyPayload) -> dict[str, Any]:
    """Compare fresh bounded dynamic CTMRG points across environment dimensions."""
    resolved = _resolve_or_http(payload.backend, "ctmrg")
    require_gpu(cp)
    max_payload = payload.problem.model_copy(update={
        "environment_bond_dim": max(payload.environment_bond_dims),
        "max_time_ms": payload.max_time_ms,
        "max_mem_mb": payload.max_mem_mb,
    })
    report = preflight_ctmrg(max_payload, gpu_free_mb=_gpu_free_mb(_hardware_snapshot()))
    report["dynamic_study"] = True
    report = _scale_ctmrg_study_preflight(
        report,
        point_count=len(payload.environment_bond_dims),
        max_time_ms=payload.max_time_ms,
    )
    if not report.get("feasible", False):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "dynamic CTMRG convergence study rejected by preflight budget",
                "warnings": report.get("warnings", []),
                "estimated_peak_memory_mb": report.get("estimated_peak_memory_mb"),
                "estimated_time_ms": report.get("estimated_time_ms"),
            },
        )
    started_at = time.perf_counter()
    try:
        result = run_dynamic_ctmrg_convergence_study(
            cp,
            payload.problem,
            payload.environment_bond_dims,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["preflight"] = report
    return _with_run_provenance(
        result, payload, requested_backend=payload.backend, resolved_backend=resolved, started_at=started_at,
    )


@router.post("/jobs/ctmrg/dynamic/sectors")
@_sync_gpu_guard
def jobs_ctmrg_dynamic_sectors(payload: CTMRGDynamicSectorStudyPayload) -> dict[str, Any]:
    """Compare fresh dynamic CTMRG initial sectors without averaging them."""
    resolved = _resolve_or_http(payload.backend, "ctmrg")
    require_gpu(cp)
    max_payload = payload.problem.model_copy(update={
        "max_time_ms": payload.max_time_ms,
        "max_mem_mb": payload.max_mem_mb,
    })
    report = preflight_ctmrg(max_payload, gpu_free_mb=_gpu_free_mb(_hardware_snapshot()))
    report["dynamic_sector_study"] = True
    report = _scale_ctmrg_study_preflight(
        report,
        point_count=len(payload.initialization_seeds),
        max_time_ms=payload.max_time_ms,
    )
    if not report.get("feasible", False):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "dynamic CTMRG sector study rejected by preflight budget",
                "warnings": report.get("warnings", []),
                "estimated_peak_memory_mb": report.get("estimated_peak_memory_mb"),
                "estimated_time_ms": report.get("estimated_time_ms"),
            },
        )
    started_at = time.perf_counter()
    try:
        result = run_dynamic_ctmrg_sector_study(
            cp,
            payload.problem,
            payload.initialization_seeds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["preflight"] = report
    return _with_run_provenance(
        result, payload, requested_backend=payload.backend, resolved_backend=resolved, started_at=started_at,
    )


@router.post("/jobs/ctmrg/boundary-mps-convergence")
@_sync_gpu_guard
def jobs_ctmrg_boundary_mps_convergence(payload: CTMRGBoundaryMPSStudyPayload) -> dict[str, Any]:
    """Compare the independent finite-cylinder boundary-MPS diagnostic."""
    resolved = _resolve_or_http(payload.backend, "ctmrg")
    require_gpu(cp)
    estimate_payload = payload.problem.model_copy(update={
        "boundary_mps_reference": True,
        "boundary_mps_width": max(width for width, _ in payload.patch_sizes),
        "boundary_mps_height": max(height for _, height in payload.patch_sizes),
        "boundary_mps_bond_dim": max(payload.boundary_bond_dims),
        "max_time_ms": payload.max_time_ms,
        "max_mem_mb": payload.max_mem_mb,
    })
    report = preflight_ctmrg(estimate_payload, gpu_free_mb=_gpu_free_mb(_hardware_snapshot()))
    report["study_points"] = len(payload.patch_sizes) * len(payload.boundary_bond_dims)
    if not report.get("feasible", False):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "boundary-MPS convergence study rejected by preflight budget",
                "warnings": report.get("warnings", []),
                "estimated_peak_memory_mb": report.get("estimated_peak_memory_mb"),
                "estimated_time_ms": report.get("estimated_time_ms"),
            },
        )
    started_at = time.perf_counter()
    problem = payload.problem.model_copy(update={
        "boundary_mps_reference": False,
        "max_time_ms": payload.max_time_ms,
        "max_mem_mb": payload.max_mem_mb,
    })
    try:
        ctmrg_result = run_ctmrg(cp, problem)
        study = run_boundary_mps_convergence_study(
            tensors_from_payload(problem),
            problem,
            ctmrg_energy=float(ctmrg_result["energy"]),
            ctmrg_onsite=[float(item["value"]) for item in ctmrg_result["observables"]],
            ctmrg_interactions=[item["value"] for item in ctmrg_result["interactions"]],
            patch_sizes=[tuple(pair) for pair in payload.patch_sizes],
            boundary_bond_dims=payload.boundary_bond_dims,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result = {
        "status": "done",
        "method": "ctmrg-with-finite-cylinder-boundary-mps-study",
        "ctmrg": ctmrg_result,
        "boundary_mps": study,
        "preflight": report,
    }
    return _with_run_provenance(
        result, payload, requested_backend=payload.backend, resolved_backend=resolved, started_at=started_at,
    )


@router.post("/jobs/ctmrg/boundary-mps-transfer-convergence")
@_sync_gpu_guard
def jobs_ctmrg_boundary_mps_transfer_convergence(payload: CTMRGBoundaryMPSTransferStudyPayload) -> dict[str, Any]:
    """Compare bounded boundary-MPS row-transfer fixed points across controls."""
    resolved = _resolve_or_http(payload.backend, "ctmrg")
    require_gpu(cp)
    estimate_payload = payload.problem.model_copy(update={
        "boundary_mps_transfer_fixed_point": True,
        "boundary_mps_width": max(payload.widths),
        "boundary_mps_bond_dim": max(payload.boundary_bond_dims),
        "boundary_mps_transfer_cycles": payload.cycles,
        "max_time_ms": payload.max_time_ms,
        "max_mem_mb": payload.max_mem_mb,
    })
    report = preflight_ctmrg(estimate_payload, gpu_free_mb=_gpu_free_mb(_hardware_snapshot()))
    report = _scale_ctmrg_study_preflight(
        report,
        point_count=len(payload.widths) * len(payload.boundary_bond_dims),
        max_time_ms=payload.max_time_ms,
    )
    report["boundary_mps_transfer_study"] = True
    if not report.get("feasible", False):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "boundary-MPS transfer convergence study rejected by preflight budget",
                "warnings": report.get("warnings", []),
                "estimated_peak_memory_mb": report.get("estimated_peak_memory_mb"),
                "estimated_time_ms": report.get("estimated_time_ms"),
            },
        )
    started_at = time.perf_counter()
    try:
        study = run_boundary_mps_transfer_convergence_study(
            tensors_from_payload(payload.problem),
            payload.problem.unit_cell,
            widths=payload.widths,
            boundary_bond_dims=payload.boundary_bond_dims,
            cycles=payload.cycles,
            cutoff=payload.cutoff,
            tolerance=payload.tolerance,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    study["preflight"] = report
    return _with_run_provenance(
        study, payload, requested_backend=payload.backend, resolved_backend=resolved, started_at=started_at,
    )


@router.post("/jobs/ctmrg/boundary-mps-transfer-gauge-covariance")
@_sync_gpu_guard
def jobs_ctmrg_boundary_mps_transfer_gauge_covariance(
    payload: CTMRGBoundaryMPSTransferGaugeStudyPayload,
) -> dict[str, Any]:
    """Replay row-transfer boundaries under a paired virtual gauge."""
    resolved = _resolve_or_http(payload.backend, "ctmrg")
    require_gpu(cp)
    estimate_payload = payload.problem.model_copy(update={
        "boundary_mps_transfer_fixed_point": True,
        "boundary_mps_width": max(payload.widths),
        "boundary_mps_bond_dim": max(payload.boundary_bond_dims),
        "boundary_mps_transfer_cycles": payload.cycles,
        "max_time_ms": payload.max_time_ms,
        "max_mem_mb": payload.max_mem_mb,
    })
    report = preflight_ctmrg(estimate_payload, gpu_free_mb=_gpu_free_mb(_hardware_snapshot()))
    report = _scale_ctmrg_study_preflight(
        report,
        point_count=4 * len(payload.widths) * len(payload.boundary_bond_dims),
        max_time_ms=payload.max_time_ms,
    )
    report["boundary_mps_transfer_gauge_study"] = True
    report["gauge_replay_runs_per_point"] = 4
    if not report.get("feasible", False):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "boundary-MPS transfer gauge study rejected by preflight budget",
                "warnings": report.get("warnings", []),
                "estimated_peak_memory_mb": report.get("estimated_peak_memory_mb"),
                "estimated_time_ms": report.get("estimated_time_ms"),
            },
        )
    started_at = time.perf_counter()
    try:
        study = run_boundary_mps_transfer_gauge_covariance_study(
            tensors_from_payload(payload.problem),
            payload.problem.unit_cell,
            widths=payload.widths,
            boundary_bond_dims=payload.boundary_bond_dims,
            cycles=payload.cycles,
            cutoff=payload.cutoff,
            tolerance=payload.tolerance,
            gauge_tolerance=payload.gauge_tolerance,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    study["preflight"] = report
    return _with_run_provenance(
        study, payload, requested_backend=payload.backend, resolved_backend=resolved, started_at=started_at,
    )
