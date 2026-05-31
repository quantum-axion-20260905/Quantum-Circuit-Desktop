from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .cuda import add_cuda_dll_dirs, hardware_info, require_gpu
from .jobs import JobManager
from .models import BenchPayload, SamplePayload, SimulatePayload, TNPayload

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


app = FastAPI(title="Quantum Compute Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = JobManager()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@app.get("/hardware")
def hardware() -> dict[str, Any]:
    return hardware_info(cp)


@app.post("/jobs/simulate")
def jobs_simulate(payload: SimulatePayload) -> dict[str, Any]:
    require_gpu(cp)
    return simulate(cp, payload)


@app.post("/jobs/sample")
def jobs_sample(payload: SamplePayload) -> dict[str, Any]:
    require_gpu(cp)
    return sv_sample(cp, payload)


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
    }


@app.post("/async/{kind}")
def submit_async(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
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
    payload["budget"] = budget

    job = jobs.create(kind=kind, request=payload, seed=seed)

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
            max_qubits = int(budget.get("max_qubits", 64))
            if n_qubits > max_qubits:
                raise ValueError(f"n_qubits {n_qubits} exceeds budget max_qubits {max_qubits}")

            if shots is not None:
                max_shots = int(budget.get("max_shots", 200000))
                if shots > max_shots:
                    raise ValueError(f"shots {shots} exceeds budget max_shots {max_shots}")

            max_mem_mb = budget.get("max_mem_mb")
            if max_mem_mb is not None:
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
            out = tn_estimate(cp, oe, ctg, TNPayload(**payload))
            return {"metrics": {"backend": out.get("backend")}, "artifacts": {"result": out}}
        if kind == "tn_amplitudes":
            check_limits(n_qubits)
            j.progress = 0.15
            j.log("running", phase="tn_amplitudes")
            out = tn_amplitudes(cp, oe, ctg, TNPayload(**payload))
            return {"metrics": {"backend": out.get("backend")}, "artifacts": {"result": out}}
        if kind == "sample":
            shots = int(payload.get("shots", 0) or 0)
            check_limits(n_qubits, shots=shots)
            j.progress = 0.12
            j.log("running", phase="statevector_sample")

            def progress_cb(p: float, phase: str):
                j.progress = max(j.progress, min(0.99, float(p)))
                j.log("progress", phase=phase, progress=j.progress)

            out = sv_sample(cp, SamplePayload(**payload), progress_cb=progress_cb, cancel_cb=lambda: j.cancel_flag)
            return {"metrics": {"backend": out.get("backend")}, "artifacts": {"result": out}}
        raise ValueError(f"Unknown kind: {kind}")

    jobs.run_async(job, runner)
    return _job_view(job)


@app.get("/async/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    return _job_view(job)


@app.post("/async/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    ok = jobs.cancel(job_id)
    return {"ok": ok}


@app.post("/jobs/bench_matmul")
def jobs_bench(payload: BenchPayload) -> dict[str, Any]:
    require_gpu(cp)
    return bench_matmul(cp, payload)


@app.post("/jobs/tn_amplitudes")
def jobs_tn_amplitudes(payload: TNPayload) -> dict[str, Any]:
    require_gpu(cp)
    return tn_amplitudes(cp, oe, ctg, payload)


@app.post("/jobs/tn_estimate")
def jobs_tn_estimate(payload: TNPayload) -> dict[str, Any]:
    require_gpu(cp)
    return tn_estimate(cp, oe, ctg, payload)
