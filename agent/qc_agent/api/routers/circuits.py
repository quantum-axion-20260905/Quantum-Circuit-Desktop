from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import product
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from ...cuda import require_gpu
from ...jobs import JobCanceled
from ...models import (
    BenchPayload,
    CrossValidatePayload,
    PreflightPayload,
    RunPayload,
    SamplePayload,
    SimulatePayload,
    SweepPayload,
    TNPayload,
)
from ...backends.bench import bench_matmul
from ...backends.mps import (
    amplitudes as mps_amplitudes,
    estimate as mps_estimate,
    sample as mps_sample,
)
from ...backends.preflight import estimate as preflight_estimate
from ...backends.statevector import sample as sv_sample, simulate
from ...backends.tn import amplitudes as tn_amplitudes, estimate as tn_estimate
from ...services.common import (
    cp,
    ctg,
    oe,
    _contract_path_metrics,
    _gpu_free_mb,
    _hardware_snapshot,
    _materialize_sweep,
    _reference_run,
    _require_feasible_gpu_run,
    _require_materialized_parameters,
    _resolve_or_http,
    _SWEEP_CANCEL,
    _sync_gpu_guard,
    _unresolved_parameters,
    _with_run_provenance,
)

router = APIRouter(tags=["circuits"])


@router.post("/jobs/preflight")
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


@router.post("/jobs/run")
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


@router.post("/jobs/sweep")
def jobs_sweep(payload: SweepPayload) -> dict[str, Any]:
    """Evaluate a finite Cartesian product of named rotation parameters."""
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
        except Exception as exc:
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


@router.post("/jobs/simulate")
@_sync_gpu_guard
def jobs_simulate(payload: SimulatePayload) -> dict[str, Any]:
    require_gpu(cp)
    _require_feasible_gpu_run(payload, "statevector")
    started_at = time.perf_counter()
    result = simulate(cp, payload)
    return _with_run_provenance(
        result, payload, requested_backend="statevector", resolved_backend="statevector", started_at=started_at,
    )


@router.post("/jobs/sample")
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


@router.post("/jobs/bench_matmul")
@_sync_gpu_guard
def jobs_bench(payload: BenchPayload) -> dict[str, Any]:
    require_gpu(cp)
    started_at = time.perf_counter()
    result = bench_matmul(cp, payload)
    return _with_run_provenance(
        result, payload, requested_backend="statevector", resolved_backend="cupy-matmul", started_at=started_at,
    )


@router.post("/jobs/tn_amplitudes")
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


@router.post("/jobs/tn_estimate")
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


@router.post("/jobs/cross_validate")
@_sync_gpu_guard
def jobs_cross_validate(payload: CrossValidatePayload) -> dict[str, Any]:
    """Compare selected GPU/TN amplitudes with the independent CPU reference."""
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
