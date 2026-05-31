from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


JobStatus = str  # queued|running|done|failed|canceled


@dataclass
class Job:
    job_id: str
    kind: str
    status: JobStatus = "queued"
    progress: float = 0.0
    seed: Optional[int] = None
    created_at: str = field(default_factory=_now_iso)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: str = ""
    request: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)

    cancel_flag: bool = False

    def log(self, event: str, **data: Any) -> None:
        self.logs.append({"t": _now_iso(), "event": event, **data})


class JobManager:
    """
    In-memory async job runner for local agent.
    Enough for research-grade reproducibility + progress/cancel UI.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}
        # GPU-safe default: single worker so CuPy global RNG + GPU context stays deterministic.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qc-agent-job")

    def create(self, kind: str, request: Dict[str, Any], seed: Optional[int]) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(job_id=job_id, kind=kind, seed=seed, request=request)
        job.log("created", kind=kind)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job:
            return False
        with self._lock:
            if job.status in ("done", "failed", "canceled"):
                return True
            job.cancel_flag = True
            job.status = "canceled"
            job.progress = min(job.progress, 1.0)
            job.finished_at = _now_iso()
            job.log("canceled")
        return True

    def run_async(self, job: Job, fn: Callable[[Job], Dict[str, Any]]) -> None:
        self._pool.submit(self._run, job, fn)

    def _run(self, job: Job, fn: Callable[[Job], Dict[str, Any]]) -> None:
        with self._lock:
            if job.status == "canceled":
                return
            job.status = "running"
            job.started_at = _now_iso()
            job.progress = 0.01
            job.log("started")

        t0 = time.perf_counter()
        try:
            out = fn(job)
            with self._lock:
                if job.status == "canceled":
                    return
                job.metrics.update(out.get("metrics", {}))
                job.artifacts.update(out.get("artifacts", {}))
                job.progress = 1.0
                job.status = "done"
                job.finished_at = _now_iso()
                job.log("done", elapsed_s=round(time.perf_counter() - t0, 6))
        except Exception as exc:
            with self._lock:
                if job.status == "canceled":
                    return
                job.status = "failed"
                job.progress = 1.0
                job.error = str(exc)
                job.finished_at = _now_iso()
                job.log("failed", error=str(exc))
