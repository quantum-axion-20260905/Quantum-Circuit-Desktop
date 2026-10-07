from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
import math
import os
import threading
import time
from typing import Any, Literal

from fastapi import HTTPException

from ..cuda import add_cuda_dll_dirs, hardware_info, require_gpu
from ..jobs import JobManager, ResourceRequest
from ..metrics import metrics
from ..models import RunPayload, SweepPayload, TNGate, TNPayload
from ..provenance import with_provenance
from ..backends.registry import resolve_run_backend
from ..backends.reference import MAX_REFERENCE_QUBITS, run as reference_run
from ..backends.preflight import estimate as preflight_estimate
from ..backends.tn import estimate as tn_estimate

add_cuda_dll_dirs()

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None

try:
    import opt_einsum as oe
except Exception:  # pragma: no cover
    oe = None

try:
    import cotengra as ctg
except Exception:  # pragma: no cover
    ctg = None

AGENT_VERSION = "0.8.0-alpha.7"


def _hardware_snapshot() -> dict[str, Any]:
    return hardware_info(cp)


def _gpu_available() -> bool:
    return bool(hardware_info(cp).get("gpu", {}).get("available"))


def _gpu_free_mb(snapshot: dict[str, Any]) -> float | None:
    device = snapshot.get("gpu", {}).get("device0", {})
    value = device.get("free_global_mem")
    return float(value) / (1024 * 1024) if isinstance(value, (int, float)) else None


# Central JobManager instance
jobs = JobManager(
    resource_provider=_hardware_snapshot,
    max_workers=None,
)

_RESOURCE_ALREADY_HELD: ContextVar[bool] = ContextVar("qc_resource_already_held", default=False)
_SWEEP_CANCEL: ContextVar[Any] = ContextVar("qc_sweep_cancel", default=None)


def _sync_gpu_guard(handler):
    """Serialize compatibility GPU routes with the async admission broker."""

    @wraps(handler)
    def guarded(payload, *args, **kwargs):
        if _RESOURCE_ALREADY_HELD.get():
            return handler(payload, *args, **kwargs)
        requested = getattr(payload, "backend", None)
        if requested == "reference" or (requested == "auto" and not _gpu_available()):
            return handler(payload, *args, **kwargs)
        timeout_s = max(1.0, float(getattr(payload, "max_time_ms", 120000)) / 1000.0)
        event = threading.Event()
        request = ResourceRequest(kind="cuda", memory_mb=1.0, exclusive=True)
        try:
            lease = jobs.broker.acquire(request, event, timeout_s=timeout_s)
        except TimeoutError as exc:
            raise HTTPException(status_code=429, detail="GPU is busy; retry or use POST /async/jobs") from exc
        try:
            if cp is not None and lease.device_id is not None:
                with cp.cuda.Device(lease.device_id):
                    return handler(payload, *args, **kwargs)
            return handler(payload, *args, **kwargs)
        finally:
            jobs.broker.release(lease)

    return guarded


def _resolve_or_http(
    requested: str,
    operation: Literal["samples", "selected_amplitudes", "estimate", "simulate", "expectation", "evolve", "ground_state", "dmrg", "peps", "ctmrg"],
) -> str:
    try:
        snapshot = _hardware_snapshot()
        return resolve_run_backend(
            requested,
            operation,
            gpu_available=bool(snapshot.get("gpu", {}).get("available")),
            tensor_network_available=bool(snapshot.get("gpu", {}).get("available")),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _with_run_provenance(
    result: dict[str, Any],
    payload: Any,
    *,
    requested_backend: str,
    resolved_backend: str,
    started_at: float,
    seed: int | None = None,
) -> dict[str, Any]:
    elapsed_seconds = max(0.0, time.perf_counter() - started_at)
    return with_provenance(
        result,
        payload,
        requested_backend=requested_backend,
        resolved_backend=resolved_backend,
        device=_hardware_snapshot(),
        started_at=started_at,
        started_wall_at=time.time() - elapsed_seconds,
        agent_version=AGENT_VERSION,
        seed=seed,
    )


def _scale_ctmrg_study_preflight(
    report: dict[str, Any],
    *,
    point_count: int,
    max_time_ms: int,
) -> dict[str, Any]:
    """Price sequential CTMRG study points against the caller's full budget."""

    points = max(1, int(point_count))
    point_time = report.get("estimated_time_ms")
    report["study_points"] = points
    report["estimated_point_time_ms"] = point_time
    if point_time is None:
        return report
    total_time = max(1, int(math.ceil(float(point_time) * points)))
    report["estimated_time_ms"] = total_time
    report["estimated_total_time_ms"] = total_time
    if total_time > int(max_time_ms):
        warning = f"estimated CTMRG study time {total_time} ms exceeds time budget {int(max_time_ms)} ms"
        report.setdefault("warnings", []).append(warning)
        report.setdefault("blocking_warnings", []).append(warning)
        report["status"] = "rejected"
        report["feasible"] = False
    return report


def _reference_run(payload: RunPayload) -> dict[str, Any]:
    if payload.n_qubits > MAX_REFERENCE_QUBITS:
        raise HTTPException(status_code=422, detail=f"reference-cpu supports at most {MAX_REFERENCE_QUBITS} qubits")
    try:
        return reference_run(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _materialize_sweep(payload: SweepPayload, assignment: dict[str, float]) -> RunPayload:
    data = payload.model_dump(mode="json")
    materialized_gates = []
    for raw_gate in data["gates"]:
        gate = dict(raw_gate)
        parameter = gate.pop("parameter", None)
        if parameter is not None:
            gate["theta"] = assignment[parameter]
        materialized_gates.append(TNGate(**gate))
    data["gates"] = materialized_gates
    data.pop("parameter_values", None)
    return RunPayload(**data)


def _contract_path_metrics(payload: Any) -> dict[str, Any] | None:
    """Build a TN contraction path without executing the contraction."""
    if getattr(payload, "tn_method", "mps") != "contraction":
        return None
    if cp is None or oe is None:
        raise HTTPException(status_code=422, detail="tn_method=contraction requires CUDA and opt_einsum")
    try:
        estimate_result = tn_estimate(cp, oe, ctg, TNPayload(**payload.model_dump(mode="json")))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"tensor-network path estimation failed: {exc}") from exc
    return {
        "path_cost": estimate_result.get("opt_cost"),
        "largest_intermediate": estimate_result.get("largest_intermediate"),
    }


def _unresolved_parameters(payload: Any) -> list[str]:
    return sorted({
        str(gate.parameter)
        for gate in getattr(payload, "gates", [])
        if getattr(gate, "parameter", None) is not None
    })


def _require_materialized_parameters(payload: Any) -> None:
    unresolved = _unresolved_parameters(payload)
    if unresolved:
        raise HTTPException(
            status_code=422,
            detail=(
                f"unresolved rotation parameters: {', '.join(unresolved)}; "
                "use POST /jobs/sweep with parameter_values"
            ),
        )


def _require_feasible_gpu_run(payload: Any, backend_name: str) -> None:
    _require_materialized_parameters(payload)
    if backend_name == "tensor-network" and getattr(payload, "tn_method", "mps") == "contraction" and oe is None:
        raise HTTPException(status_code=422, detail="tn_method=contraction requires opt_einsum")
    snapshot = _hardware_snapshot()
    path_metrics = _contract_path_metrics(payload) if backend_name == "tensor-network" else None
    report = preflight_estimate(
        payload,
        cotengra_available=ctg is not None,
        backend_name=backend_name,
        gpu_free_mb=_gpu_free_mb(snapshot),
        path_metrics=path_metrics,
    )
    if not report.get("feasible", False):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "run rejected by preflight budget",
                "warnings": report.get("warnings", []),
                "estimated_peak_memory_mb": report.get("estimated_peak_memory_mb"),
                "estimated_time_ms": report.get("estimated_time_ms"),
            },
        )


def _job_view(job: Any) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "kind": job.kind,
        "status": job.status,
        "progress": job.progress,
        "seed": job.seed,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error": job.error or None,
        "metrics": job.metrics,
        "artifacts": job.artifacts,
        "logs": job.logs,
        "request": job.request,
        "resource": job.resource,
    }

