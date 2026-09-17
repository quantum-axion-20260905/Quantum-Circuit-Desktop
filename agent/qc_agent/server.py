from __future__ import annotations

from itertools import product
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from functools import wraps
from typing import Any, Literal
import os
import secrets
import threading
import time

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from .cuda import add_cuda_dll_dirs, hardware_info, require_gpu
from .jobs import JobCanceled, JobManager, ResourceRequest
from .models import (
    BenchPayload,
    CrossValidatePayload,
    PreflightPayload,
    RunPayload,
    SamplePayload,
    SimulatePayload,
    SweepPayload,
    TNGate,
    TNPayload,
)
from .plugins.models import (
    DMRGPayload,
    ExpectationPayload,
    GroundStatePayload,
    ObservableCrossValidatePayload,
    PEPSPayload,
    TEBDPayload,
)
from .plugins.registry import catalog as plugin_catalog
from .core.observables import cross_validate_mps_observables, evolve_statevector, mps_expectation_from_tensors, reference_expectation, statevector_expectation
from .core.mps_runtime import MPSRuntime
from .core.ground_state import exact_ground_state
from .core.dmrg import run_dmrg
from .core.peps import run_peps
from .plugins.tebd import run_tebd
from .provenance import with_provenance
from .backends.registry import catalog, resolve_run_backend

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

from .backends.bench import bench_matmul
from .backends.statevector import sample as sv_sample
from .backends.statevector import simulate
from .backends.tn import amplitudes as tn_amplitudes
from .backends.tn import estimate as tn_estimate
from .backends.mps import amplitudes as mps_amplitudes
from .backends.mps import estimate as mps_estimate
from .backends.mps import sample as mps_sample
from .backends.reference import MAX_REFERENCE_QUBITS, run as reference_run
from .backends.preflight import estimate as preflight_estimate
from .backends.preflight import estimate_ground_state as preflight_ground_state
from .backends.preflight import estimate_dmrg as preflight_dmrg
from .backends.preflight import estimate_peps as preflight_peps
from .backends.preflight import estimate_tebd as preflight_tebd
from .api.plugins import router as plugin_router
from .api.contracts import AsyncBudget, AsyncKind, AsyncSubmission
from .metrics import metrics


app = FastAPI(title="Quantum Compute Agent", version="0.5.0")
cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "QC_AGENT_CORS_ALLOWED_ORIGINS",
        "http://127.0.0.1:3000,http://localhost:3000",
    ).split(",")
    if origin.strip()
]
cors_allow_all = os.environ.get("QC_AGENT_CORS_ALLOW_ALL", "0") == "1"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if cors_allow_all else cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def local_token_guard(request, call_next):
    configured = os.environ.get("QC_AGENT_TOKEN")
    production = os.environ.get("QC_AGENT_ENV", "development").lower() == "production"
    if production and not configured and request.url.path not in ("/health", "/docs", "/openapi.json"):
        return JSONResponse(status_code=503, content={"detail": "QC_AGENT_TOKEN is required in production"})
    if configured and request.url.path not in ("/health", "/docs", "/openapi.json"):
        provided = request.headers.get("x-qc-agent-token", "")
        if not secrets.compare_digest(provided, configured):
            return JSONResponse(status_code=401, content={"detail": "invalid agent token"})
    return await call_next(request)


@app.middleware("http")
async def request_metrics(request, call_next):
    started = time.perf_counter()
    method = request.method
    path = request.url.path
    metric_path = path
    if path.startswith("/async/jobs/"):
        metric_path = "/async/jobs/:job_id" if path.count("/") == 3 else "/async/jobs/:job_id/cancel"
    elif path.startswith("/plugins/"):
        parts = path.split("/")
        metric_path = "/plugins/:plugin_id/" + "/".join(parts[3:]) if len(parts) > 3 else "/plugins/:plugin_id"
    try:
        response = await call_next(request)
        metrics.count("http_requests_total", method=method, path=metric_path)
        metrics.count(f"http_responses_{response.status_code}_total", method=method, path=metric_path)
        return response
    except Exception:
        metrics.count("http_exceptions_total", method=method, path=metric_path)
        raise
    finally:
        metrics.observe("http_request_duration", (time.perf_counter() - started) * 1000, method=method, path=metric_path)

def _hardware_snapshot() -> dict[str, Any]:
    return hardware_info(cp)


def _gpu_available() -> bool:
    return bool(hardware_info(cp).get("gpu", {}).get("available"))


def _gpu_free_mb(snapshot: dict[str, Any]) -> float | None:
    device = snapshot.get("gpu", {}).get("device0", {})
    value = device.get("free_global_mem")
    return float(value) / (1024 * 1024) if isinstance(value, (int, float)) else None


# The provider is installed after the CUDA helpers exist. Job admission is
# therefore based on live telemetry and not on a stale startup snapshot.
jobs = JobManager(
    resource_provider=_hardware_snapshot,
    max_workers=None,
)
app.include_router(plugin_router)
_RESOURCE_ALREADY_HELD: ContextVar[bool] = ContextVar("qc_resource_already_held", default=False)
_SWEEP_CANCEL: ContextVar[Any] = ContextVar("qc_sweep_cancel", default=None)


def _sync_gpu_guard(handler):
    """Serialize compatibility GPU routes with the async admission broker.

    The unified async API is preferred because it exposes progress and
    cancellation.  Synchronous routes still need the same device-level lock
    so a legacy caller cannot race an admitted async job.  The route performs
    its own detailed preflight; this guard only reserves a device-level slot.
    """

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


def _resolve_or_http(requested: str, operation: Literal["samples", "selected_amplitudes", "estimate", "simulate", "expectation", "evolve", "ground_state", "dmrg", "peps"]) -> str:
    try:
        snapshot = _hardware_snapshot()
        return resolve_run_backend(
            requested,
            operation,
            gpu_available=bool(snapshot.get("gpu", {}).get("available")),
            # The default MPS backend is CuPy-native. opt_einsum is only
            # required for the explicit tn_method=contraction path.
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
        agent_version=app.version,
        seed=seed,
    )


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


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@app.get("/hardware")
def hardware() -> dict[str, Any]:
    return hardware_info(cp)


@app.get("/capabilities")
def capabilities() -> dict[str, Any]:
    snapshot = _hardware_snapshot()
    gpu = snapshot.get("gpu", {})
    return {
        "backends": [item.__dict__ for item in catalog(
            gpu_available=bool(gpu.get("available")),
            tensor_network_available=bool(gpu.get("available")),
        )],
        "gpu": gpu,
        "features": {
            "cross_backend_validation": bool(gpu.get("available") and oe is not None),
            "noise_trajectories": bool(gpu.get("available")),
            "parameter_sweeps": True,
            "physics_plugins": True,
            "observables": bool(gpu.get("available")),
            "tebd": bool(gpu.get("available")),
            "dmrg": bool(gpu.get("available")),
            "peps": bool(gpu.get("available")),
            "lattice_dimensions": 3,
            "distributed_multi_gpu": int(gpu.get("count", 0) or 0) > 1,
            "unified_async_jobs": True,
            "async_job_kinds": [
                "run", "sample", "simulate", "bench_matmul", "expectation", "tebd", "ground_state",
                "dmrg", "peps", "tn_estimate", "tn_amplitudes", "sweep", "cross_validate",
            ],
        },
        "agent_version": app.version,
        "queue": jobs.snapshot(),
    }


@app.get("/plugins")
def plugins() -> dict[str, Any]:
    return {"plugins": plugin_catalog()}


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


@app.post("/jobs/expectation")
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


@app.post("/jobs/cross_validate_observables")
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


@app.post("/jobs/tebd")
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


@app.post("/jobs/ground_state")
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


@app.post("/jobs/dmrg")
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


@app.post("/jobs/peps")
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


@app.post("/jobs/preflight")
def jobs_preflight(payload: PreflightPayload) -> dict[str, Any]:
    unresolved = _unresolved_parameters(payload)
    if unresolved:
        return {
            "status": "rejected",
            "feasible": False,
            "backend": payload.backend,
            "warnings": [
                f"unresolved rotation parameters: {', '.join(unresolved)}; use /jobs/sweep with parameter_values"
            ],
        }
    operation = "samples" if payload.result_type == "samples" else "selected_amplitudes"
    try:
        resolved = _resolve_or_http(payload.backend, operation)
    except HTTPException as exc:
        return {"status": "rejected", "feasible": False, "backend": payload.backend, "warnings": [str(exc.detail)]}

    snapshot = _hardware_snapshot()
    try:
        path_metrics = _contract_path_metrics(payload) if resolved == "tensor-network" else None
        report = preflight_estimate(
            payload,
            cotengra_available=ctg is not None,
            backend_name=resolved,
            gpu_free_mb=_gpu_free_mb(snapshot) if resolved in ("statevector", "tensor-network") else None,
            path_metrics=path_metrics,
        )
    except HTTPException as exc:
        return {"status": "rejected", "feasible": False, "backend": resolved, "warnings": [str(exc.detail)]}
    warnings = list(report.get("warnings", []))
    if resolved == "reference":
        warnings.append("CPU reference performance is not comparable to CUDA")
    report["backend"] = "reference-cpu" if resolved == "reference" else resolved
    report["resolved_backend"] = resolved
    report["requested_backend"] = payload.backend
    report["warnings"] = warnings
    return report


@app.post("/jobs/run")
@_sync_gpu_guard
def jobs_run(payload: RunPayload) -> dict[str, Any]:
    _require_materialized_parameters(payload)
    operation = "samples" if payload.result_type == "samples" else "selected_amplitudes"
    resolved = _resolve_or_http(payload.backend, operation)
    started_at = time.perf_counter()
    if resolved == "reference":
        result = _reference_run(payload)
        return _with_run_provenance(
            result, payload, requested_backend=payload.backend, resolved_backend=resolved,
            started_at=started_at, seed=payload.seed,
        )

    require_gpu(cp)
    _require_feasible_gpu_run(payload, resolved)
    if resolved == "statevector":
        result = sv_sample(cp, SamplePayload(**payload.model_dump()), progress_cb=None, cancel_cb=None)
    else:
        tn_payload = TNPayload(**payload.model_dump())
        if payload.result_type == "samples":
            if tn_payload.tn_method != "mps":
                raise HTTPException(status_code=422, detail="tensor-network sampling requires tn_method=mps")
            result = mps_sample(cp, payload)
        elif tn_payload.tn_method == "mps":
            result = mps_amplitudes(cp, tn_payload)
        else:
            result = tn_amplitudes(cp, oe, ctg, tn_payload)
    return _with_run_provenance(
        result, payload, requested_backend=payload.backend, resolved_backend=resolved,
        started_at=started_at, seed=payload.seed,
    )


@app.post("/jobs/sweep")
def jobs_sweep(payload: SweepPayload) -> dict[str, Any]:
    """Evaluate a finite Cartesian product of named rotation parameters.

    Points run through the same backend resolver and preflight path. By
    default they are sequential; on a multi-GPU host, parallel_devices can
    distribute independent points across distinct CUDA devices.
    """

    if payload.noise is not None and payload.noise.active and payload.result_type == "selected_amplitudes":
        raise HTTPException(status_code=422, detail="noise sweeps require result_type=samples")
    started_at = time.perf_counter()
    names = list(payload.parameter_values)
    values = list(product(*(payload.parameter_values[name] for name in names)))
    gpu = _hardware_snapshot().get("gpu", {})
    gpu_count = int(gpu.get("count", 0) or 0)
    if payload.parallel_devices > 1 and (cp is None or gpu_count < payload.parallel_devices):
        raise HTTPException(
            status_code=422,
            detail=f"parallel_devices={payload.parallel_devices} requested, but only {gpu_count} CUDA device(s) are available",
        )

    def run_point(item: tuple[int, tuple[float, ...]]) -> dict[str, Any]:
        index, point = item
        assignment = dict(zip(names, point))
        cancel_cb = _SWEEP_CANCEL.get()
        if cancel_cb and cancel_cb():
            raise JobCanceled("job canceled")
        try:
            if payload.parallel_devices > 1:
                with cp.cuda.Device(index % payload.parallel_devices):
                    point_result = jobs_run(_materialize_sweep(payload, assignment))
            else:
                point_result = jobs_run(_materialize_sweep(payload, assignment))
            return {"parameters": assignment, "result": point_result}
        except HTTPException as exc:
            return {"parameters": assignment, "error": exc.detail, "status_code": exc.status_code}
        except Exception as exc:  # keep one bad point from hiding other results
            return {"parameters": assignment, "error": str(exc), "status_code": 500}

    if payload.parallel_devices > 1 and len(values) > 1:
        with ThreadPoolExecutor(max_workers=payload.parallel_devices, thread_name_prefix="qc-sweep") as pool:
            results = list(pool.map(run_point, enumerate(values)))
    else:
        results = [run_point(item) for item in enumerate(values)]
    failed = sum(1 for item in results if "error" in item)

    result = {
        "status": "done" if failed < len(values) else "failed",
        "backend": "parameter-sweep",
        "method": "cartesian-parallel-devices" if payload.parallel_devices > 1 else "cartesian-sequential",
        "result_type": payload.result_type,
        "n_qubits": payload.n_qubits,
        "points": len(values),
        "completed": len(values) - failed,
        "failed": failed,
        "parameter_values": payload.parameter_values,
        "parallel_devices": payload.parallel_devices,
        "results": results,
    }
    return _with_run_provenance(
        result, payload, requested_backend=payload.backend, resolved_backend="parameter-sweep",
        started_at=started_at, seed=payload.seed,
    )


@app.post("/jobs/simulate")
@_sync_gpu_guard
def jobs_simulate(payload: SimulatePayload) -> dict[str, Any]:
    require_gpu(cp)
    _require_feasible_gpu_run(payload, "statevector")
    started_at = time.perf_counter()
    result = simulate(cp, payload)
    return _with_run_provenance(
        result, payload, requested_backend="statevector", resolved_backend="statevector", started_at=started_at,
    )


@app.post("/jobs/sample")
@_sync_gpu_guard
def jobs_sample(payload: SamplePayload) -> dict[str, Any]:
    require_gpu(cp)
    preflight_payload = PreflightPayload(
        n_qubits=payload.n_qubits, gates=payload.gates, dtype=payload.dtype,
        shots=payload.shots, noise=payload.noise,
    )
    _require_feasible_gpu_run(preflight_payload, "statevector")
    started_at = time.perf_counter()
    result = sv_sample(cp, payload)
    return _with_run_provenance(
        result, payload, requested_backend="statevector", resolved_backend="statevector",
        started_at=started_at, seed=payload.seed,
    )


def _job_view(job) -> dict[str, Any]:
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


@app.post("/async/jobs")
def submit_unified_async_route(submission: AsyncSubmission) -> dict[str, Any]:
    # Keep the exact route before the legacy dynamic route so ``/async/jobs``
    # cannot be interpreted as the old ``{kind}`` path parameter.
    return submit_unified_async(submission)


@app.post("/async/{kind}")
def submit_async(kind: Literal["tn_estimate", "tn_amplitudes", "sample"], payload: dict[str, Any]) -> dict[str, Any]:
    """
    Generic async wrapper around existing compute endpoints.
    kind: tn_estimate|tn_amplitudes|sample
    payload: same as sync endpoints, plus optional seed
    """
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

    def runner(j):
        # Deterministic seeding for CuPy random (sampling).
        if j.seed is not None:
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

            max_mem_mb = budget.get("max_mem_mb")
            # TN requests are guarded by the contraction-path estimate above;
            # a full-statevector bound would incorrectly reject useful TN jobs.
            if kind == "sample" and max_mem_mb is not None:
                try:
                    max_mem_mb_f = float(max_mem_mb)
                except Exception:
                    max_mem_mb_f = None
                if max_mem_mb_f is not None and max_mem_mb_f > 0:
                    dtype = payload.get("dtype", "complex64")
                    bytes_per_amp = 8 if dtype == "complex64" else 16
                    est_bytes = (2**n_qubits) * bytes_per_amp
                    if est_bytes / (1024 * 1024) > max_mem_mb_f:
                        raise ValueError(
                            f"estimated statevector memory {(est_bytes/(1024*1024)):.1f} MB exceeds budget max_mem_mb {max_mem_mb_f}"
                        )

        n_qubits = int(payload.get("n_qubits", 0) or 0)
        if kind == "tn_estimate":
            check_limits(n_qubits)
            j.progress = 0.15
            j.log("running", phase="tn_estimate")
            started_at = time.perf_counter()
            out = mps_estimate(parsed_payload) if parsed_payload.tn_method == "mps" else tn_estimate(cp, oe, ctg, parsed_payload)
            out = _with_run_provenance(
                out, parsed_payload, requested_backend="tensor-network", resolved_backend=resolved,
                started_at=started_at, seed=j.seed,
            )
            return {"metrics": {"backend": out.get("backend")}, "artifacts": {"result": out}}
        if kind == "tn_amplitudes":
            check_limits(n_qubits)
            j.progress = 0.15
            j.log("running", phase="tn_amplitudes")
            started_at = time.perf_counter()
            out = mps_amplitudes(cp, parsed_payload) if parsed_payload.tn_method == "mps" else tn_amplitudes(cp, oe, ctg, parsed_payload)
            out = _with_run_provenance(
                out, parsed_payload, requested_backend="tensor-network", resolved_backend=resolved,
                started_at=started_at, seed=j.seed,
            )
            return {"metrics": {"backend": out.get("backend")}, "artifacts": {"result": out}}
        if kind == "sample":
            shots = int(payload.get("shots", 0) or 0)
            check_limits(n_qubits, shots=shots)
            j.progress = 0.12
            j.log("running", phase="statevector_sample")

            def progress_cb(p: float, phase: str):
                j.progress = max(j.progress, min(0.99, float(p)))
                j.log("progress", phase=phase, progress=j.progress)

            started_at = time.perf_counter()
            out = sv_sample(cp, parsed_payload, progress_cb=progress_cb, cancel_cb=lambda: j.cancel_flag)
            out = _with_run_provenance(
                out, parsed_payload, requested_backend="auto", resolved_backend=resolved,
                started_at=started_at, seed=j.seed,
            )
            return {"metrics": {"backend": out.get("backend")}, "artifacts": {"result": out}}
        raise ValueError(f"Unknown kind: {kind}")

    jobs.run_async(job, runner, resource=legacy_resource, acquire_timeout_s=max(1.0, float(budget.get("queue_timeout_ms", 120000)) / 1000.0))
    return _job_view(job)


def _async_parse(kind: AsyncKind, raw: dict[str, Any]) -> Any:
    payload_types: dict[str, Any] = {
        "run": RunPayload,
        "sample": SamplePayload,
        "simulate": SimulatePayload,
        "bench_matmul": BenchPayload,
        "expectation": ExpectationPayload,
        "tebd": TEBDPayload,
        "ground_state": GroundStatePayload,
        "dmrg": DMRGPayload,
        "peps": PEPSPayload,
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
        # TEBD is intentionally tensor-network-only.  TEBDPayload does not
        # expose a user-selectable backend, so async admission must use the
        # same default as the synchronous /jobs/tebd route.
        return _resolve_or_http(getattr(payload, "backend", "tensor-network"), "evolve"), "evolve"
    if kind == "dmrg":
        return _resolve_or_http(payload.backend, "dmrg"), "dmrg"
    if kind == "peps":
        # PEPS has the same fixed backend contract as TEBD.
        return _resolve_or_http(getattr(payload, "backend", "tensor-network"), "peps"), "peps"
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
    elif kind == "ground_state":
        result = exact_ground_state(cp, payload)
    else:
        raise ValueError(f"unsupported async kind: {kind}")

    check_active()
    if kind in ("sweep", "cross_validate") and isinstance(result.get("provenance"), dict):
        return {"metrics": {"backend": result.get("backend"), "resolved_backend": resolved}, "artifacts": {"result": result}}
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


@app.get("/async/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    return _job_view(job)


@app.get("/queue")
def queue_status() -> dict[str, Any]:
    """Expose admission-control state without exposing job payload contents."""
    return jobs.snapshot()


@app.get("/metrics")
def metrics_status() -> dict[str, Any]:
    return metrics.snapshot()


@app.post("/async/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    ok = jobs.cancel(job_id)
    return {"ok": ok}


@app.post("/jobs/bench_matmul")
@_sync_gpu_guard
def jobs_bench(payload: BenchPayload) -> dict[str, Any]:
    require_gpu(cp)
    started_at = time.perf_counter()
    result = bench_matmul(cp, payload)
    return _with_run_provenance(
        result, payload, requested_backend="statevector", resolved_backend="cupy-matmul", started_at=started_at,
    )


@app.post("/jobs/tn_amplitudes")
@_sync_gpu_guard
def jobs_tn_amplitudes(payload: TNPayload) -> dict[str, Any]:
    require_gpu(cp)
    _require_materialized_parameters(payload)
    _require_feasible_gpu_run(payload, "tensor-network")
    started_at = time.perf_counter()
    result = mps_amplitudes(cp, payload) if payload.tn_method == "mps" else tn_amplitudes(cp, oe, ctg, payload)
    return _with_run_provenance(
        result, payload, requested_backend="tensor-network", resolved_backend="tensor-network", started_at=started_at,
    )


@app.post("/jobs/tn_estimate")
@_sync_gpu_guard
def jobs_tn_estimate(payload: TNPayload) -> dict[str, Any]:
    require_gpu(cp)
    _require_materialized_parameters(payload)
    _require_feasible_gpu_run(payload, "tensor-network")
    started_at = time.perf_counter()
    result = mps_estimate(payload) if payload.tn_method == "mps" else tn_estimate(cp, oe, ctg, payload)
    return _with_run_provenance(
        result, payload, requested_backend="tensor-network", resolved_backend="tensor-network", started_at=started_at,
    )


@app.post("/jobs/cross_validate")
@_sync_gpu_guard
def jobs_cross_validate(payload: CrossValidatePayload) -> dict[str, Any]:
    """Compare selected GPU/TN amplitudes with the independent CPU reference.

    This endpoint is intentionally limited to small circuits.  It is a
    validation instrument, not a production-size simulator.
    """

    require_gpu(cp)
    _require_materialized_parameters(payload)
    if payload.tn_method == "contraction" and oe is None:
        raise HTTPException(status_code=500, detail="opt_einsum is not available")
    started_at = time.perf_counter()
    bitstrings = [item.strip().replace("_", "") for item in payload.bitstrings]
    bitstrings = bitstrings or ["0" * payload.n_qubits]
    reference_payload = RunPayload(
        n_qubits=payload.n_qubits,
        gates=payload.gates,
        bitstrings=bitstrings,
        dtype=payload.dtype,
        backend="reference",
        result_type="selected_amplitudes",
    )
    reference = _reference_run(reference_payload)
    tensor_payload = TNPayload(**payload.model_dump())
    tensor_result = (
        mps_amplitudes(cp, tensor_payload)
        if payload.tn_method == "mps"
        else tn_amplitudes(cp, oe, ctg, tensor_payload)
    )
    reference_map = {item["bitstring"]: complex(item["re"], item["im"]) for item in reference["amplitudes"]}
    tensor_map = {item["bitstring"]: complex(item["re"], item["im"]) for item in tensor_result["amplitudes"]}
    comparisons = []
    max_error = 0.0
    for bitstring in bitstrings:
        expected = reference_map[bitstring]
        actual = tensor_map[bitstring]
        error = abs(expected - actual)
        max_error = max(max_error, error)
        comparisons.append({
            "bitstring": bitstring,
            "reference": {"re": expected.real, "im": expected.imag},
            "tensor_network": {"re": actual.real, "im": actual.imag},
            "absolute_error": error,
            "passed": error <= payload.tolerance,
        })
    result = {
        "status": "done",
        "backend": "cross-validation",
        "n_qubits": payload.n_qubits,
        "tolerance": payload.tolerance,
        "max_absolute_error": max_error,
        "passed": max_error <= payload.tolerance,
        "comparisons": comparisons,
        "reference_backend": reference.get("backend"),
        "tested_backend": tensor_result.get("backend"),
    }
    return _with_run_provenance(
        result, payload, requested_backend="tensor-network", resolved_backend="cross-validation", started_at=started_at,
    )
