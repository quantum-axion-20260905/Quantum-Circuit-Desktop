from __future__ import annotations

import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


JobStatus = str  # queued|running|done|failed|canceled


class JobCanceled(RuntimeError):
    """Raised by a compute callback when a cooperative cancellation is observed."""


@dataclass(frozen=True)
class ResourceRequest:
    """Resources reserved for one job.

    GPU memory is a reservation, not an attempt to reclaim memory from another
    process. The broker may wait for capacity, but it never touches unrelated
    processes or host RAM.
    """

    kind: str = "cpu"
    memory_mb: float = 0.0
    device_id: int | None = None
    exclusive: bool = False


@dataclass(frozen=True)
class ResourceLease:
    kind: str
    device_id: int | None
    memory_mb: float


class ResourceBroker:
    """Local GPU admission control with a provider seam for future workers."""

    def __init__(self, provider: Callable[[], dict[str, Any]] | None = None) -> None:
        self._provider = provider
        self._condition = threading.Condition(threading.Lock())
        self._reserved: dict[int, float] = {}
        self._exclusive: set[int] = set()

    def set_provider(self, provider: Callable[[], dict[str, Any]] | None) -> None:
        with self._condition:
            self._provider = provider
            self._condition.notify_all()

    def _gpu_devices(self) -> list[tuple[int, float]]:
        if self._provider is None:
            return []
        try:
            snapshot = self._provider() or {}
            gpu = snapshot.get("gpu", {})
            if not gpu.get("available"):
                return []
            devices: list[tuple[int, float]] = []
            count = int(gpu.get("count", 0) or 0)
            for device_id in range(max(1, count)):
                device = gpu.get(f"device{device_id}", {}) or {}
                free = device.get("free_global_mem")
                if isinstance(free, (int, float)):
                    devices.append((device_id, float(free) / (1024 * 1024)))
            if not devices:
                free = gpu.get("free_global_mem")
                if isinstance(free, (int, float)):
                    devices.append((0, float(free) / (1024 * 1024)))
            return devices
        except Exception:
            # Telemetry failure must make a CUDA job wait instead of bypassing
            # the safety gate with an optimistic reservation.
            return []

    def acquire(
        self,
        request: ResourceRequest,
        cancel_event: threading.Event,
        *,
        timeout_s: float | None = None,
    ) -> ResourceLease:
        if request.kind != "cuda":
            return ResourceLease(request.kind, request.device_id, max(0.0, request.memory_mb))

        deadline = None if timeout_s is None else time.monotonic() + max(0.0, timeout_s)
        memory_mb = max(0.0, float(request.memory_mb))
        with self._condition:
            while True:
                if cancel_event.is_set():
                    raise JobCanceled("job canceled while waiting for GPU capacity")
                devices = self._gpu_devices()
                if request.device_id is not None:
                    devices = [item for item in devices if item[0] == request.device_id]
                for device_id, free_mb in devices:
                    reserved = self._reserved.get(device_id, 0.0)
                    if device_id in self._exclusive or (request.exclusive and reserved > 0):
                        continue
                    # Keep 10% telemetry headroom for CUDA allocator/workspace
                    # fluctuations while tracking this agent's reservations.
                    available_mb = max(0.0, free_mb * 0.90 - reserved)
                    if memory_mb > available_mb:
                        continue
                    if request.exclusive:
                        self._exclusive.add(device_id)
                    self._reserved[device_id] = reserved + memory_mb
                    return ResourceLease("cuda", device_id, memory_mb)

                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for requested GPU resources")
                wait_s = 0.25 if deadline is None else min(0.25, max(0.01, deadline - time.monotonic()))
                self._condition.wait(wait_s)

    def release(self, lease: ResourceLease | None) -> None:
        if lease is None or lease.kind != "cuda" or lease.device_id is None:
            return
        with self._condition:
            remaining = self._reserved.get(lease.device_id, 0.0) - max(0.0, lease.memory_mb)
            if remaining > 1e-9:
                self._reserved[lease.device_id] = remaining
            else:
                self._reserved.pop(lease.device_id, None)
                self._exclusive.discard(lease.device_id)
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "reserved_gpu_memory_mb": {str(key): round(value, 3) for key, value in self._reserved.items()},
                "exclusive_devices": sorted(self._exclusive),
            }


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
    resource: dict[str, Any] = field(default_factory=dict)

    cancel_flag: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)

    def log(self, event: str, **data: Any) -> None:
        self.logs.append({"t": _now_iso(), "event": event, **data})

    def is_canceled(self) -> bool:
        return self.cancel_flag or self.cancel_event.is_set()

    def check_canceled(self) -> None:
        if self.is_canceled():
            raise JobCanceled("job canceled")


class JobManager:
    """Durable local job manager with bounded workers and resource admission.

    ``QC_AGENT_JOB_WORKERS`` controls the worker count and is clamped to a
    small safe range. The default remains one worker for deterministic local
    CUDA usage; multiple GPUs can run exclusive jobs concurrently.
    """

    def __init__(
        self,
        state_dir: str | os.PathLike[str] | None = None,
        *,
        resource_provider: Callable[[], dict[str, Any]] | None = None,
        max_workers: int | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}
        configured_dir = state_dir or os.environ.get("QC_AGENT_JOB_STATE_DIR")
        self._state_dir = Path(configured_dir) if configured_dir else Path(__file__).resolve().parents[1] / ".runtime" / "jobs"
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._load_persisted_jobs()
        if max_workers is None:
            try:
                configured_workers = int(os.environ.get("QC_AGENT_JOB_WORKERS", "1"))
            except ValueError:
                configured_workers = 1
        else:
            configured_workers = max_workers
        self.max_workers = max(1, min(8, configured_workers))
        self._pool = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="qc-agent-job")
        self._broker = ResourceBroker(resource_provider)

    @property
    def broker(self) -> ResourceBroker:
        return self._broker

    def set_resource_provider(self, provider: Callable[[], dict[str, Any]] | None) -> None:
        self._broker.set_provider(provider)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        return str(value)

    def _job_path(self, job_id: str) -> Path:
        return self._state_dir / f"{job_id}.json"

    def _persist_unlocked(self, job: Job) -> None:
        """Write one journal record atomically without serializing the event."""
        path = self._job_path(job.job_id)
        temporary = path.with_suffix(f".{threading.get_ident()}.json.tmp")
        try:
            data = {key: value for key, value in job.__dict__.items() if key != "cancel_event"}
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(data, handle, ensure_ascii=True, default=self._json_default)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except Exception:
                pass

    def _load_persisted_jobs(self) -> None:
        for path in self._state_dir.glob("*.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                fields = {
                    key: raw[key]
                    for key in (
                        "job_id", "kind", "status", "progress", "seed", "created_at",
                        "started_at", "finished_at", "error", "request", "metrics", "artifacts", "logs", "resource",
                    )
                    if key in raw
                }
                job = Job(**fields)
                was_interrupted = False
                if job.status in ("queued", "running"):
                    job.status = "failed"
                    job.progress = 1.0
                    job.error = "Agent restarted before the job completed."
                    job.finished_at = _now_iso()
                    job.log("interrupted", reason="agent_restart")
                    was_interrupted = True
                self._jobs[job.job_id] = job
                if was_interrupted:
                    self._persist_unlocked(job)
            except Exception:
                continue

    def create(
        self,
        kind: str,
        request: Dict[str, Any],
        seed: Optional[int],
        *,
        resource: ResourceRequest | None = None,
    ) -> Job:
        job_id = uuid.uuid4().hex
        request_resource = resource or ResourceRequest()
        job = Job(
            job_id=job_id,
            kind=kind,
            seed=seed,
            request=request,
            resource={
                "kind": request_resource.kind,
                "memory_mb": request_resource.memory_mb,
                "device_id": request_resource.device_id,
                "exclusive": request_resource.exclusive,
            },
        )
        job.log("created", kind=kind, resource=job.resource)
        with self._lock:
            self._jobs[job_id] = job
            self._persist_unlocked(job)
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
            job.cancel_event.set()
            job.status = "canceled"
            job.finished_at = _now_iso()
            job.log("canceled")
            self._persist_unlocked(job)
        return True

    def run_async(
        self,
        job: Job,
        fn: Callable[[Job], Dict[str, Any]],
        *,
        resource: ResourceRequest | None = None,
        acquire_timeout_s: float | None = None,
    ) -> None:
        request_data = resource or ResourceRequest(**{
            key: value for key, value in job.resource.items()
            if key in {"kind", "memory_mb", "device_id", "exclusive"}
        })
        self._pool.submit(self._run, job, fn, request_data, acquire_timeout_s)

    def _run(
        self,
        job: Job,
        fn: Callable[[Job], Dict[str, Any]],
        request: ResourceRequest,
        acquire_timeout_s: float | None,
    ) -> None:
        lease: ResourceLease | None = None
        try:
            lease = self._broker.acquire(request, job.cancel_event, timeout_s=acquire_timeout_s)
            with self._lock:
                if job.is_canceled():
                    return
                job.status = "running"
                job.started_at = _now_iso()
                job.progress = max(job.progress, 0.01)
                job.metrics["resource_device"] = lease.device_id
                # ``device_id`` is the user's preference (and may be None).
                # ``assigned_device`` is the concrete device selected by the
                # broker and lets the runner establish the correct CUDA
                # context even when the request asked for any available GPU.
                job.resource["assigned_device"] = lease.device_id
                job.log("started", resource_device=lease.device_id)
                self._persist_unlocked(job)

            t0 = time.perf_counter()
            out = fn(job) or {}
            with self._lock:
                if job.is_canceled():
                    return
                job.metrics.update(out.get("metrics", {}))
                job.artifacts.update(out.get("artifacts", {}))
                job.progress = 1.0
                job.status = "done"
                job.finished_at = _now_iso()
                job.log("done", elapsed_s=round(time.perf_counter() - t0, 6))
                self._persist_unlocked(job)
        except JobCanceled as exc:
            with self._lock:
                job.status = "canceled"
                job.progress = min(job.progress, 1.0)
                job.error = str(exc) if str(exc) != "job canceled" else ""
                job.finished_at = job.finished_at or _now_iso()
                job.log("canceled", reason=str(exc))
                self._persist_unlocked(job)
        except TimeoutError as exc:
            with self._lock:
                if job.is_canceled():
                    job.status = "canceled"
                else:
                    job.status = "failed"
                    job.error = str(exc)
                job.progress = 1.0
                job.finished_at = _now_iso()
                job.log("failed", error=str(exc))
                self._persist_unlocked(job)
        except Exception as exc:
            with self._lock:
                if job.is_canceled():
                    return
                job.status = "failed"
                job.progress = 1.0
                job.error = str(exc)
                job.finished_at = _now_iso()
                job.log("failed", error=str(exc))
                self._persist_unlocked(job)
        finally:
            self._broker.release(lease)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counts: dict[str, int] = {}
            for job in self._jobs.values():
                counts[job.status] = counts.get(job.status, 0) + 1
        return {"max_workers": self.max_workers, "jobs": counts, "resources": self._broker.snapshot()}

    def shutdown(self, wait: bool = False) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=not wait)
