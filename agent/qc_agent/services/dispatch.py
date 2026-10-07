from __future__ import annotations

from itertools import product
import time
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import ValidationError

from ..api.contracts import AsyncBudget, AsyncKind, AsyncSubmission
from ..cuda import require_gpu
from ..jobs import ResourceRequest
from ..models import (
    BenchPayload,
    CrossValidatePayload,
    PreflightPayload,
    RunPayload,
    SamplePayload,
    SimulatePayload,
    SweepPayload,
    TNPayload,
)
from ..plugins.models import (
    CTMRGBoundaryMPSStudyPayload,
    CTMRGBoundaryMPSTransferGaugeStudyPayload,
    CTMRGBoundaryMPSTransferStudyPayload,
    CTMRGConvergenceStudyPayload,
    CTMRGPayload,
    DMRGPayload,
    ExpectationPayload,
    GroundStatePayload,
    PEPSPayload,
    TEBDPayload,
)
from ..backends.bench import bench_matmul
from ..backends.mps import (
    amplitudes as mps_amplitudes,
    estimate as mps_estimate,
    sample as mps_sample,
)
from ..backends.preflight import (
    estimate as preflight_estimate,
    estimate_ctmrg as preflight_ctmrg,
    estimate_dmrg as preflight_dmrg,
    estimate_ground_state as preflight_ground_state,
    estimate_peps as preflight_peps,
    estimate_tebd as preflight_tebd,
)
from ..backends.statevector import sample as sv_sample, simulate
from ..backends.tn import amplitudes as tn_amplitudes, estimate as tn_estimate
from ..core.ctmrg import run_ctmrg, run_ctmrg_convergence_study
from ..core.ctmrg_boundary_mps import (
    run_boundary_mps_convergence_study,
    run_boundary_mps_transfer_gauge_covariance_study,
    run_boundary_mps_transfer_convergence_study,
    tensors_from_payload,
)
from ..core.dmrg import run_dmrg
from ..core.ground_state import exact_ground_state
from ..core.peps import run_peps
from ..plugins.tebd import run_tebd
from .common import (
    cp,
    ctg,
    jobs,
    oe,
    _contract_path_metrics,
    _gpu_free_mb,
    _hardware_snapshot,
    _job_view,
    _materialize_sweep,
    _reference_run,
    _require_feasible_gpu_run,
    _require_materialized_parameters,
    _resolve_or_http,
    _scale_ctmrg_study_preflight,
    _with_run_provenance,
    _RESOURCE_ALREADY_HELD,
    _SWEEP_CANCEL,
)
from ..api.routers.physics import _observable_result
from ..api.routers.circuits import jobs_cross_validate, jobs_sweep


def _async_parse(kind: AsyncKind, raw: dict[str, Any]) -> Any:
    payload_types: dict[AsyncKind, type[Any]] = {
        "run": RunPayload,
        "sample": SamplePayload,
        "simulate": SimulatePayload,
        "bench_matmul": BenchPayload,
        "expectation": ExpectationPayload,
        "tebd": TEBDPayload,
        "ground_state": GroundStatePayload,
        "dmrg": DMRGPayload,
        "peps": PEPSPayload,
        "ctmrg": CTMRGPayload,
        "ctmrg_convergence": CTMRGConvergenceStudyPayload,
        "ctmrg_boundary_mps_convergence": CTMRGBoundaryMPSStudyPayload,
        "ctmrg_boundary_mps_transfer_convergence": CTMRGBoundaryMPSTransferStudyPayload,
        "ctmrg_boundary_mps_transfer_gauge_covariance": CTMRGBoundaryMPSTransferGaugeStudyPayload,
        "tn_estimate": TNPayload,
        "tn_amplitudes": TNPayload,
        "sweep": SweepPayload,
        "cross_validate": CrossValidatePayload,
    }
    try:
        return payload_types[kind](**raw)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


def _async_backend(kind: AsyncKind, payload: Any) -> tuple[str, str]:
    if kind == "bench_matmul":
        return "statevector", "simulate"
    if kind in ("sample", "simulate"):
        return "statevector", "samples" if kind == "sample" else "simulate"
    if kind == "tn_estimate":
        return "tensor-network", "estimate"
    if kind == "tn_amplitudes":
        return "tensor-network", "selected_amplitudes"
    elif kind == "cross_validate":
        return "tensor-network", "selected_amplitudes"
    if kind == "sweep":
        operation = "samples" if payload.result_type == "samples" else "selected_amplitudes"
        return _resolve_or_http(payload.backend, operation), operation
    if kind == "tebd":
        return _resolve_or_http(getattr(payload, "backend", "auto"), "evolve"), "evolve"
    if kind == "dmrg":
        return _resolve_or_http(payload.backend, "dmrg"), "dmrg"
    if kind == "peps":
        return _resolve_or_http(getattr(payload, "backend", "auto"), "peps"), "peps"
    if kind in (
        "ctmrg",
        "ctmrg_convergence",
        "ctmrg_boundary_mps_convergence",
        "ctmrg_boundary_mps_transfer_convergence",
        "ctmrg_boundary_mps_transfer_gauge_covariance",
    ):
        return _resolve_or_http(getattr(payload, "backend", "auto"), "ctmrg"), "ctmrg"
    if kind == "ground_state":
        return _resolve_or_http(payload.backend, "ground_state"), "ground_state"
    if kind == "expectation":
        return _resolve_or_http(payload.backend, "expectation"), "expectation"
    operation = "samples" if payload.result_type == "samples" else "selected_amplitudes"
    return _resolve_or_http(payload.backend, operation), operation


def _async_preflight(kind: AsyncKind, payload: Any, resolved: str, budget: dict[str, Any]) -> dict[str, Any]:
    snapshot = _hardware_snapshot()
    free_mb = _gpu_free_mb(snapshot) if resolved in ("statevector", "tensor-network", "exact-diagonalization") else None
    max_time_ms = int(budget.get("max_time_ms", getattr(payload, "max_time_ms", 120000)))
    max_mem_mb = float(budget.get("max_mem_mb", getattr(payload, "max_mem_mb", 4096)))
    max_qubits = budget.get("max_qubits")
    if max_qubits is not None and hasattr(payload, "n_qubits") and int(payload.n_qubits) > int(max_qubits):
        raise HTTPException(status_code=422, detail=f"n_qubits {payload.n_qubits} exceeds budget max_qubits {max_qubits}")
    max_shots = budget.get("max_shots")
    if max_shots is not None and hasattr(payload, "shots") and int(payload.shots) > int(max_shots):
        raise HTTPException(status_code=422, detail=f"shots {payload.shots} exceeds budget max_shots {max_shots}")
    bounded_payload = payload.model_copy(update={"max_time_ms": max_time_ms, "max_mem_mb": max_mem_mb}) if hasattr(payload, "model_copy") else payload

    if kind == "bench_matmul":
        bytes_per_value = 2 if payload.dtype == "fp16" else 4
        estimated_peak = (float(payload.size) * float(payload.size) * bytes_per_value * 3.5) / (1024 * 1024)
        estimated_ms = int(1 + (float(payload.size) ** 3 * float(payload.iters)) / 200_000_000)
        warnings: list[str] = []
        if estimated_peak > max_mem_mb:
            warnings.append(f"estimated matmul memory {estimated_peak:.1f} MB exceeds memory budget")
        if free_mb is not None and estimated_peak > free_mb * 0.70:
            warnings.append(f"estimated matmul memory {estimated_peak:.1f} MB exceeds 70% of currently free GPU memory")
        if estimated_ms > max_time_ms:
            warnings.append(f"estimated matmul time {estimated_ms} ms exceeds time budget")
        blocking_warnings = [warning for warning in warnings if "exceeds" in warning]
        report = {
            "status": "ready" if not blocking_warnings else "rejected",
            "feasible": not blocking_warnings,
            "backend": resolved,
            "method": "gpu-matmul-bound",
            "size": payload.size,
            "iters": payload.iters,
            "dtype": payload.dtype,
            "estimated_peak_memory_mb": round(estimated_peak, 3),
            "estimated_time_ms": estimated_ms,
            "warnings": warnings,
        }
    elif kind == "sweep":
        if payload.parallel_devices != 1:
            raise HTTPException(status_code=422, detail="unified async sweeps currently require parallel_devices=1 so GPU admission remains exclusive")
        names = list(payload.parameter_values)
        first_assignment = {name: payload.parameter_values[name][0] for name in names}
        first = _materialize_sweep(payload, first_assignment)
        points = len(list(product(*(payload.parameter_values[name] for name in names))))
        return _async_preflight("run", first, resolved, budget) | {"sweep_points": points}
    elif kind == "cross_validate":
        base = PreflightPayload(
            n_qubits=payload.n_qubits,
            gates=payload.gates,
            dtype=payload.dtype,
            backend=resolved,
            result_type="selected_amplitudes",
            bond_dim=payload.bond_dim,
            truncation_cutoff=payload.truncation_cutoff,
            max_time_ms=max_time_ms,
            max_mem_mb=max_mem_mb,
        )
        report = preflight_estimate(base, cotengra_available=ctg is not None, backend_name=resolved, gpu_free_mb=free_mb)
    elif kind == "ground_state":
        report = preflight_ground_state(bounded_payload, gpu_free_mb=free_mb)
    elif kind == "dmrg":
        report = preflight_dmrg(bounded_payload, gpu_free_mb=free_mb)
    elif kind == "peps":
        report = preflight_peps(bounded_payload, gpu_free_mb=free_mb)
    elif kind == "ctmrg":
        report = preflight_ctmrg(bounded_payload, gpu_free_mb=free_mb)
    elif kind == "ctmrg_convergence":
        max_payload = payload.problem.model_copy(update={
            "environment_bond_dim": max(payload.environment_bond_dims),
            "max_time_ms": max_time_ms,
            "max_mem_mb": max_mem_mb,
        })
        report = preflight_ctmrg(max_payload, gpu_free_mb=free_mb)
        report["study_points"] = len(payload.environment_bond_dims)
        report["environment_bond_dims"] = list(payload.environment_bond_dims)
    elif kind == "ctmrg_boundary_mps_convergence":
        estimate_payload = payload.problem.model_copy(update={
            "boundary_mps_reference": True,
            "boundary_mps_width": max(width for width, _ in payload.patch_sizes),
            "boundary_mps_height": max(height for _, height in payload.patch_sizes),
            "boundary_mps_bond_dim": max(payload.boundary_bond_dims),
            "max_time_ms": max_time_ms,
            "max_mem_mb": max_mem_mb,
        })
        report = preflight_ctmrg(estimate_payload, gpu_free_mb=free_mb)
        report["study_points"] = len(payload.patch_sizes) * len(payload.boundary_bond_dims)
    elif kind == "ctmrg_boundary_mps_transfer_convergence":
        estimate_payload = payload.problem.model_copy(update={
            "boundary_mps_transfer_fixed_point": True,
            "boundary_mps_width": max(payload.widths),
            "boundary_mps_bond_dim": max(payload.boundary_bond_dims),
            "boundary_mps_transfer_cycles": payload.cycles,
            "max_time_ms": max_time_ms,
            "max_mem_mb": max_mem_mb,
        })
        report = preflight_ctmrg(estimate_payload, gpu_free_mb=free_mb)
        report = _scale_ctmrg_study_preflight(
            report,
            point_count=len(payload.widths) * len(payload.boundary_bond_dims),
            max_time_ms=max_time_ms,
        )
        report["boundary_mps_transfer_study"] = True
    elif kind == "ctmrg_boundary_mps_transfer_gauge_covariance":
        estimate_payload = payload.problem.model_copy(update={
            "boundary_mps_transfer_fixed_point": True,
            "boundary_mps_width": max(payload.widths),
            "boundary_mps_bond_dim": max(payload.boundary_bond_dims),
            "boundary_mps_transfer_cycles": payload.cycles,
            "max_time_ms": max_time_ms,
            "max_mem_mb": max_mem_mb,
        })
        report = preflight_ctmrg(estimate_payload, gpu_free_mb=free_mb)
        report = _scale_ctmrg_study_preflight(
            report,
            point_count=4 * len(payload.widths) * len(payload.boundary_bond_dims),
            max_time_ms=max_time_ms,
        )
        report["boundary_mps_transfer_gauge_study"] = True
        report["gauge_replay_runs_per_point"] = 4
    elif kind == "tebd":
        report = preflight_tebd(bounded_payload, gpu_free_mb=free_mb)
    elif resolved == "reference":
        report = {"status": "ready", "feasible": True, "backend": "reference"}
    else:
        if kind == "sample":
            base = PreflightPayload(
                n_qubits=payload.n_qubits,
                gates=payload.gates,
                dtype=payload.dtype,
                backend="statevector",
                result_type="samples",
                shots=payload.shots,
                noise=payload.noise,
                max_time_ms=max_time_ms,
                max_mem_mb=max_mem_mb,
            )
        elif kind == "expectation":
            base = PreflightPayload(
                n_qubits=payload.n_qubits,
                gates=payload.gates,
                dtype=payload.dtype,
                backend=resolved,
                result_type="selected_amplitudes",
                bond_dim=payload.bond_dim,
                truncation_cutoff=payload.truncation_cutoff,
                max_time_ms=max_time_ms,
                max_mem_mb=max_mem_mb,
            )
        elif kind == "simulate":
            base = PreflightPayload(
                n_qubits=payload.n_qubits,
                gates=payload.gates,
                dtype=payload.dtype,
                backend="statevector",
                result_type="samples",
                max_time_ms=max_time_ms,
                max_mem_mb=max_mem_mb,
            )
        else:
            base = PreflightPayload(
                n_qubits=payload.n_qubits,
                gates=payload.gates,
                dtype=payload.dtype,
                backend=resolved,
                result_type="samples" if getattr(payload, "shots", None) is not None else "selected_amplitudes",
                bond_dim=getattr(payload, "bond_dim", 64),
                truncation_cutoff=getattr(payload, "truncation_cutoff", 0.0),
                shots=getattr(payload, "shots", 1024),
                noise=getattr(payload, "noise", None),
                max_time_ms=max_time_ms,
                max_mem_mb=max_mem_mb,
            )
        report = preflight_estimate(
            base,
            cotengra_available=ctg is not None,
            backend_name=resolved,
            gpu_free_mb=free_mb,
            path_metrics=_contract_path_metrics(payload) if resolved == "tensor-network" and hasattr(payload, "tn_method") else None,
        )
    report.setdefault("backend", resolved)
    report.setdefault("estimated_peak_memory_mb", 0.0)
    report.setdefault("estimated_time_ms", max_time_ms)
    if not report.get("feasible", False):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "async job rejected by preflight budget",
                "warnings": report.get("warnings", []),
                "estimated_peak_memory_mb": report.get("estimated_peak_memory_mb"),
                "estimated_time_ms": report.get("estimated_time_ms"),
            },
        )
    return report


def _async_resource(kind: AsyncKind, payload: Any, resolved: str, report: dict[str, Any], budget: dict[str, Any]) -> ResourceRequest:
    if resolved in ("reference", "cpu"):
        return ResourceRequest(kind="cpu")
    requested_device = budget.get("device_id")
    device_id = int(requested_device) if requested_device is not None else None
    estimated_mb = max(1.0, float(report.get("estimated_peak_memory_mb", 1.0)))
    configured_mb = budget.get("reservation_mb")
    memory_mb = max(estimated_mb * 1.15, float(configured_mb)) if configured_mb is not None else estimated_mb * 1.15
    return ResourceRequest(kind="cuda", memory_mb=memory_mb, device_id=device_id, exclusive=True)


def _async_compute(kind: AsyncKind, payload: Any, resolved: str, job: Any) -> dict[str, Any]:
    budget = job.request.get("budget", {}) if isinstance(job.request, dict) else {}
    max_time_ms = int(budget.get("max_time_ms", getattr(payload, "max_time_ms", 120000)))
    deadline = time.perf_counter() + max_time_ms / 1000.0

    def check_active() -> None:
        job.check_canceled()
        if time.perf_counter() > deadline:
            raise TimeoutError(f"job exceeded max_time_ms={max_time_ms}")

    def canceled() -> bool:
        check_active()
        return job.is_canceled()

    check_active()
    if job.seed is not None and cp is not None:
        try:
            cp.random.seed(job.seed)
        except Exception:
            pass

    def progress(value: float, phase: str) -> None:
        check_active()
        job.progress = max(job.progress, min(0.99, 0.05 + 0.94 * float(value)))
        job.log("progress", phase=phase, progress=job.progress)

    started_at = time.perf_counter()
    if kind == "run":
        if resolved == "reference":
            result = _reference_run(payload)
        elif resolved == "statevector":
            result = sv_sample(cp, SamplePayload(**payload.model_dump()), progress_cb=progress, cancel_cb=canceled)
        else:
            tn_payload = TNPayload(**payload.model_dump())
            if payload.result_type == "samples":
                result = mps_sample(cp, payload, progress_cb=progress, cancel_cb=canceled)
            elif tn_payload.tn_method == "mps":
                result = mps_amplitudes(cp, tn_payload, progress_cb=progress, cancel_cb=canceled)
            else:
                result = tn_amplitudes(cp, oe, ctg, tn_payload)
    elif kind == "sample":
        result = sv_sample(cp, payload, progress_cb=progress, cancel_cb=canceled)
    elif kind == "simulate":
        result = simulate(cp, payload, progress_cb=progress, cancel_cb=canceled)
    elif kind == "bench_matmul":
        result = bench_matmul(cp, payload, progress_cb=progress, cancel_cb=canceled)
    elif kind == "expectation":
        result = _observable_result(payload, resolved, progress_cb=progress, cancel_cb=canceled)
    elif kind == "tn_estimate":
        result = mps_estimate(payload) if payload.tn_method == "mps" else tn_estimate(cp, oe, ctg, payload)
    elif kind == "tn_amplitudes":
        result = mps_amplitudes(cp, payload, progress_cb=progress, cancel_cb=canceled) if payload.tn_method == "mps" else tn_amplitudes(cp, oe, ctg, payload)
    elif kind == "sweep":
        resource_token = _RESOURCE_ALREADY_HELD.set(True)
        cancel_token = _SWEEP_CANCEL.set(canceled)
        try:
            result = jobs_sweep(payload)
        finally:
            _SWEEP_CANCEL.reset(cancel_token)
            _RESOURCE_ALREADY_HELD.reset(resource_token)
    elif kind == "cross_validate":
        result = jobs_cross_validate(payload)
    elif kind == "tebd":
        result = run_tebd(cp, payload, progress_cb=progress, cancel_cb=canceled)
    elif kind == "dmrg":
        result = run_dmrg(cp, payload, progress_cb=progress, cancel_cb=canceled)
    elif kind == "peps":
        result = run_peps(cp, payload, progress_cb=progress, cancel_cb=canceled)
    elif kind == "ctmrg":
        result = run_ctmrg(cp, payload, progress_cb=progress, cancel_cb=canceled)
    elif kind == "ctmrg_convergence":
        result = run_ctmrg_convergence_study(
            cp,
            payload.problem,
            payload.environment_bond_dims,
            progress_cb=progress,
            cancel_cb=canceled,
        )
    elif kind == "ctmrg_boundary_mps_convergence":
        problem = payload.problem.model_copy(update={"boundary_mps_reference": False})
        ctmrg_result = run_ctmrg(cp, problem, progress_cb=progress, cancel_cb=canceled)
        result = {
            "status": "done",
            "method": "ctmrg-with-finite-cylinder-boundary-mps-study",
            "ctmrg": ctmrg_result,
            "boundary_mps": run_boundary_mps_convergence_study(
                tensors_from_payload(problem),
                problem,
                ctmrg_energy=float(ctmrg_result["energy"]),
                ctmrg_onsite=[float(item["value"]) for item in ctmrg_result["observables"]],
                ctmrg_interactions=[item["value"] for item in ctmrg_result["interactions"]],
                patch_sizes=[tuple(pair) for pair in payload.patch_sizes],
                boundary_bond_dims=payload.boundary_bond_dims,
            ),
        }
    elif kind == "ctmrg_boundary_mps_transfer_convergence":
        problem = payload.problem.model_copy(update={
            "max_time_ms": max_time_ms,
            "max_mem_mb": float(budget.get("max_mem_mb", getattr(payload.problem, "max_mem_mb", 4096))),
        })
        result = run_boundary_mps_transfer_convergence_study(
            tensors_from_payload(problem),
            problem.unit_cell,
            widths=payload.widths,
            boundary_bond_dims=payload.boundary_bond_dims,
            cycles=payload.cycles,
            cutoff=payload.cutoff,
            tolerance=payload.tolerance,
        )
    elif kind == "ctmrg_boundary_mps_transfer_gauge_covariance":
        problem = payload.problem.model_copy(update={
            "max_time_ms": max_time_ms,
            "max_mem_mb": float(budget.get("max_mem_mb", getattr(payload.problem, "max_mem_mb", 4096))),
        })
        result = run_boundary_mps_transfer_gauge_covariance_study(
            tensors_from_payload(problem),
            problem.unit_cell,
            widths=payload.widths,
            boundary_bond_dims=payload.boundary_bond_dims,
            cycles=payload.cycles,
            cutoff=payload.cutoff,
            tolerance=payload.tolerance,
            gauge_tolerance=payload.gauge_tolerance,
        )
    elif kind == "ground_state":
        result = exact_ground_state(cp, payload)
    else:
        raise ValueError(f"unsupported async kind: {kind}")

    check_active()
    if kind in ("sweep", "cross_validate") and isinstance(result.get("provenance"), dict):
        return {"metrics": {"backend": result.get("backend"), "resolved_backend": resolved}, "artifacts": {"result": result}}
    if isinstance(result, dict) and "preflight" not in result:
        preflight_log = next((entry for entry in reversed(job.logs) if entry.get("event") == "preflight"), None)
        if preflight_log is not None:
            result["preflight"] = {key: value for key, value in preflight_log.items() if key not in {"t", "event"}}
    result = _with_run_provenance(
        result,
        payload,
        requested_backend=getattr(payload, "backend", resolved),
        resolved_backend=resolved,
        started_at=started_at,
        seed=job.seed,
    )
    return {"metrics": {"backend": result.get("backend"), "resolved_backend": resolved}, "artifacts": {"result": result}}


def submit_unified_async(submission: AsyncSubmission) -> dict[str, Any]:
    """Submit any compute operation through the same durable job lifecycle."""
    raw = dict(submission.payload)
    raw_budget = raw.pop("budget", {})
    if not isinstance(raw_budget, dict):
        raise HTTPException(status_code=422, detail="budget must be an object")
    try:
        budget = AsyncBudget(**raw_budget).model_dump(exclude_none=True)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    parsed = _async_parse(submission.kind, raw)
    if submission.kind not in ("tn_estimate", "tn_amplitudes", "sample", "simulate", "sweep"):
        _require_materialized_parameters(parsed)
    resolved, _ = _async_backend(submission.kind, parsed)
    if resolved not in ("reference", "cpu"):
        require_gpu(cp)
    report = _async_preflight(submission.kind, parsed, resolved, budget)
    resource = _async_resource(submission.kind, parsed, resolved, report, budget)
    request_data = parsed.model_dump(mode="json")
    request_data["budget"] = budget
    seed = getattr(parsed, "seed", None)
    job = jobs.create(submission.kind, request_data, seed, resource=resource)
    job.log("preflight", **report)

    def runner(active_job: Any) -> dict[str, Any]:
        active_job.log("budget", **budget)
        active_job.progress = 0.05
        active_job.check_canceled()
        assigned_device = active_job.resource.get("assigned_device") if isinstance(active_job.resource, dict) else None
        if resource.kind == "cuda" and assigned_device is not None and cp is not None:
            with cp.cuda.Device(int(assigned_device)):
                return _async_compute(submission.kind, parsed, resolved, active_job)
        return _async_compute(submission.kind, parsed, resolved, active_job)

    queue_timeout_ms = budget.get("queue_timeout_ms", 120000)
    acquire_timeout_s = max(0.0, float(queue_timeout_ms) / 1000.0) if queue_timeout_ms else None
    jobs.run_async(job, runner, resource=resource, acquire_timeout_s=acquire_timeout_s)
    return _job_view(job)


def submit_async(kind: Literal["tn_estimate", "tn_amplitudes", "sample"], payload: dict[str, Any]) -> dict[str, Any]:
    """Generic async wrapper around existing compute endpoints."""
    require_gpu(cp)
    seed = payload.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except Exception:
            seed = None

    budget = payload.get("budget") or {}
    if not isinstance(budget, dict):
        budget = {}

    try:
        parsed_payload: Any = SamplePayload(**payload) if kind == "sample" else TNPayload(**payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    payload = parsed_payload.model_dump(mode="json")
    payload["budget"] = budget

    preflight_payload = PreflightPayload(
        n_qubits=parsed_payload.n_qubits,
        gates=parsed_payload.gates,
        dtype=parsed_payload.dtype,
        backend="tensor-network" if kind != "sample" else "auto",
        result_type="samples" if kind == "sample" else "selected_amplitudes",
        shots=getattr(parsed_payload, "shots", 1024),
        noise=parsed_payload.noise,
        max_time_ms=int(budget.get("max_time_ms", 120000)),
        max_mem_mb=float(budget.get("max_mem_mb", 4096)),
    )
    resolved = "statevector" if kind == "sample" else "tensor-network"
    _require_feasible_gpu_run(preflight_payload, resolved)

    legacy_report = preflight_estimate(
        preflight_payload,
        cotengra_available=ctg is not None,
        backend_name=resolved,
        gpu_free_mb=_gpu_free_mb(_hardware_snapshot()),
    )
    legacy_resource = ResourceRequest(
        kind="cuda",
        memory_mb=max(1.0, float(legacy_report.get("estimated_peak_memory_mb", 1.0)) * 1.15),
        exclusive=True,
    )
    job = jobs.create(kind=kind, request=payload, seed=seed, resource=legacy_resource)

    def runner(j: Any) -> dict[str, Any]:
        if j.seed is not None and cp is not None:
            try:
                cp.random.seed(j.seed)
            except Exception:
                pass
        j.progress = 0.05
        j.log("budget", **budget)

        def check_limits(n_qubits: int, shots: int | None = None) -> None:
            max_qubits = budget.get("max_qubits")
            if max_qubits is not None and n_qubits > int(max_qubits):
                raise ValueError(f"n_qubits {n_qubits} exceeds budget max_qubits {max_qubits}")
            if shots is not None:
                max_shots = budget.get("max_shots")
                if max_shots is not None and shots > int(max_shots):
                    raise ValueError(f"shots {shots} exceeds budget max_shots {max_shots}")

        def progress(value: float, phase: str) -> None:
            j.check_canceled()
            j.progress = max(j.progress, min(0.99, 0.05 + 0.94 * float(value)))
            j.log("progress", phase=phase, progress=j.progress)

        def canceled() -> bool:
            j.check_canceled()
            return j.is_canceled()

        j.check_canceled()
        started_at = time.perf_counter()
        if kind == "tn_estimate":
            check_limits(parsed_payload.n_qubits)
            progress(0.2, "estimate")
            result = mps_estimate(parsed_payload) if parsed_payload.tn_method == "mps" else tn_estimate(cp, oe, ctg, parsed_payload)
            progress(1.0, "done")
            res = _with_run_provenance(result, parsed_payload, requested_backend="tensor-network", resolved_backend="tensor-network", started_at=started_at, seed=j.seed)
            return {"metrics": {"backend": res.get("backend")}, "artifacts": {"estimate": res}}
        elif kind == "tn_amplitudes":
            check_limits(parsed_payload.n_qubits)
            progress(0.1, "amplitudes")
            result = (
                mps_amplitudes(cp, parsed_payload, progress_cb=progress, cancel_cb=canceled)
                if parsed_payload.tn_method == "mps"
                else tn_amplitudes(cp, oe, ctg, parsed_payload)
            )
            progress(1.0, "done")
            res = _with_run_provenance(result, parsed_payload, requested_backend="tensor-network", resolved_backend="tensor-network", started_at=started_at, seed=j.seed)
            return {"metrics": {"backend": res.get("backend")}, "artifacts": {"amplitudes": res}}
        elif kind == "sample":
            check_limits(parsed_payload.n_qubits, shots=parsed_payload.shots)
            progress(0.1, "sample")
            result = sv_sample(cp, parsed_payload, progress_cb=progress, cancel_cb=canceled)
            progress(1.0, "done")
            res = _with_run_provenance(result, parsed_payload, requested_backend="statevector", resolved_backend="statevector", started_at=started_at, seed=j.seed)
            return {"metrics": {"backend": res.get("backend")}, "artifacts": {"samples": res}}
        raise ValueError(f"unsupported legacy async kind: {kind}")

    jobs.run_async(job, runner, resource=legacy_resource)
    return _job_view(job)


def get_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    return _job_view(job)


def cancel_job(job_id: str) -> dict[str, Any]:
    ok = jobs.cancel(job_id)
    return {"ok": ok}
