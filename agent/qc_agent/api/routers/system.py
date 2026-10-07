from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ...backends.registry import catalog, experimental_method_catalog, method_catalog
from ...cuda import hardware_info
from ...metrics import metrics
from ...plugins.registry import catalog as plugin_catalog
from ...services.common import (
    AGENT_VERSION,
    cp,
    jobs,
    oe,
    _hardware_snapshot,
)

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@router.get("/hardware")
def hardware() -> dict[str, Any]:
    return hardware_info(cp)


@router.get("/capabilities")
def capabilities() -> dict[str, Any]:
    snapshot = _hardware_snapshot()
    gpu = snapshot.get("gpu", {})
    methods = method_catalog(
        gpu_available=bool(gpu.get("available")),
        tensor_network_available=bool(gpu.get("available")),
    )
    experimental_methods = experimental_method_catalog(
        gpu_available=bool(gpu.get("available")),
        tensor_network_available=bool(gpu.get("available")),
    )
    return {
        "backends": [item.__dict__ for item in catalog(
            gpu_available=bool(gpu.get("available")),
            tensor_network_available=bool(gpu.get("available")),
        )],
        "methods": [item.__dict__ for item in methods],
        "experimental_methods": [item.__dict__ for item in experimental_methods],
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
            "ctmrg": bool(gpu.get("available")),
            "tdvp": any(item.method == "tdvp" and item.available for item in methods),
            "vumps": any(item.method == "vumps" and item.available for item in methods),
            "lattice_dimensions": 3,
            "distributed_multi_gpu": int(gpu.get("count", 0) or 0) > 1,
            "unified_async_jobs": True,
            "async_job_kinds": [
                "run", "sample", "simulate", "bench_matmul", "expectation", "tebd", "ground_state",
                "dmrg", "peps", "ctmrg", "tn_estimate", "tn_amplitudes", "sweep", "cross_validate",
            ],
        },
        "agent_version": AGENT_VERSION,
        "queue": jobs.snapshot(),
    }


@router.get("/plugins")
def plugins() -> dict[str, Any]:
    return {"plugins": plugin_catalog()}


@router.get("/queue")
def queue_status() -> dict[str, Any]:
    """Expose admission-control state without exposing job payload contents."""
    return jobs.snapshot()


@router.get("/metrics")
def metrics_status() -> dict[str, Any]:
    return metrics.snapshot()
