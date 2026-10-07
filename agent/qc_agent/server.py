from __future__ import annotations

import os
import secrets
import time
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .cuda import add_cuda_dll_dirs
from .metrics import metrics
from .api.contracts import AsyncBudget, AsyncKind, AsyncSubmission
from .api.plugins import router as plugin_router
from .api.routers.system import router as system_router
from .api.routers.physics import (
    router as physics_router,
    _observable_result,
    jobs_expectation,
    jobs_cross_validate_observables,
    jobs_tebd,
    jobs_ground_state,
    jobs_dmrg,
    jobs_peps,
    jobs_ctmrg,
    jobs_ctmrg_dynamic,
    jobs_ctmrg_convergence,
    jobs_ctmrg_dynamic_convergence,
    jobs_ctmrg_dynamic_sectors,
    jobs_ctmrg_boundary_mps_convergence,
    jobs_ctmrg_boundary_mps_transfer_convergence,
    jobs_ctmrg_boundary_mps_transfer_gauge_covariance,
)
from .api.routers.circuits import (
    router as circuits_router,
    jobs_preflight,
    jobs_run,
    jobs_sweep,
    jobs_simulate,
    jobs_sample,
    jobs_bench,
    jobs_tn_amplitudes,
    jobs_tn_estimate,
    jobs_cross_validate,
)
from .api.routers.async_jobs import (
    router as async_jobs_router,
    submit_unified_async_route,
    submit_async_route as submit_async,
    get_job_route as get_job,
    cancel_job_route as cancel_job,
)
from .services.common import (
    AGENT_VERSION,
    cp,
    ctg,
    jobs,
    oe,
    _contract_path_metrics,
    _gpu_available,
    _gpu_free_mb,
    _hardware_snapshot,
    _job_view,
    _materialize_sweep,
    _reference_run,
    _require_feasible_gpu_run,
    _require_materialized_parameters,
    _resolve_or_http,
    _scale_ctmrg_study_preflight,
    _sync_gpu_guard,
    _unresolved_parameters,
    _with_run_provenance,
    _RESOURCE_ALREADY_HELD,
    _SWEEP_CANCEL,
)
from .services.dispatch import (
    _async_backend,
    _async_compute,
    _async_parse,
    _async_preflight,
    _async_resource,
    submit_unified_async,
)

add_cuda_dll_dirs()

app = FastAPI(title="Quantum Compute Agent", version=AGENT_VERSION)

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


# Mount all modular routers
app.include_router(system_router)
app.include_router(plugin_router)
app.include_router(physics_router)
app.include_router(circuits_router)
app.include_router(async_jobs_router)

__all__ = [
    "app",
    "jobs",
    "AsyncBudget",
    "AsyncKind",
    "AsyncSubmission",
    "submit_unified_async_route",
    "submit_unified_async",
    "submit_async",
    "get_job",
    "cancel_job",
    "_scale_ctmrg_study_preflight",
    "_async_backend",
    "_async_parse",
    "_async_preflight",
    "_hardware_snapshot",
    "_gpu_available",
    "_gpu_free_mb",
    "_job_view",
    "_observable_result",
    "_reference_run",
    "_materialize_sweep",
    "_require_feasible_gpu_run",
    "_require_materialized_parameters",
    "_resolve_or_http",
    "_with_run_provenance",
    "jobs_expectation",
    "jobs_cross_validate_observables",
    "jobs_tebd",
    "jobs_ground_state",
    "jobs_dmrg",
    "jobs_peps",
    "jobs_ctmrg",
    "jobs_ctmrg_dynamic",
    "jobs_ctmrg_convergence",
    "jobs_ctmrg_dynamic_convergence",
    "jobs_ctmrg_dynamic_sectors",
    "jobs_ctmrg_boundary_mps_convergence",
    "jobs_ctmrg_boundary_mps_transfer_convergence",
    "jobs_ctmrg_boundary_mps_transfer_gauge_covariance",
    "jobs_preflight",
    "jobs_run",
    "jobs_sweep",
    "jobs_simulate",
    "jobs_sample",
    "jobs_bench",
    "jobs_tn_amplitudes",
    "jobs_tn_estimate",
    "jobs_cross_validate",
]
