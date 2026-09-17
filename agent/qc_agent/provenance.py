from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any


PROVENANCE_SCHEMA = "quantum-circuit/research-run-v2"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=lambda item: item.item() if hasattr(item, "item") else str(item),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def circuit_digest(payload: Any) -> str:
    data = _jsonable(payload)
    if not isinstance(data, dict):
        return sha256_json(data)
    return sha256_json({
        "n_qubits": data.get("n_qubits"),
        "gates": data.get("gates", []),
        "dtype": data.get("dtype"),
    })


def problem_digest(payload: Any) -> str:
    """Hash the complete scientific request, including Hamiltonian/domain data."""
    return sha256_json(_jsonable(payload))


def with_provenance(
    result: dict[str, Any],
    payload: Any,
    *,
    requested_backend: str,
    resolved_backend: str,
    device: dict[str, Any],
    started_at: float,
    agent_version: str,
    seed: int | None = None,
    started_wall_at: float | None = None,
) -> dict[str, Any]:
    clean_result = dict(result)
    result_hash = sha256_json(clean_result)
    request_data = _jsonable(payload)
    raw_elapsed_seconds = time.perf_counter() - started_at
    elapsed_seconds = raw_elapsed_seconds if started_at > 0.0 and 0.0 <= raw_elapsed_seconds <= 86400.0 else 0.0
    # Callers use perf_counter for duration accuracy.  Keep the wall-clock
    # instant separate so replay/audit metadata never serializes a monotonic
    # timer as a Unix timestamp (which appears as a date in 1970).
    if started_wall_at is None:
        if started_at <= 0.0 or raw_elapsed_seconds > 86400.0:
            # This fallback also keeps direct library callers with a synthetic
            # timer value safe; production server calls provide the wall time.
            started_wall_at = time.time()
        else:
            started_wall_at = time.time() - elapsed_seconds
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "request_sha256": sha256_json(request_data),
        "circuit_sha256": circuit_digest(payload),
        "problem_sha256": problem_digest(payload),
        "result_sha256": result_hash,
        "requested_backend": requested_backend,
        "resolved_backend": resolved_backend,
        "backend": clean_result.get("backend", resolved_backend),
        "agent_version": agent_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seed": seed,
        "started_at": datetime.fromtimestamp(started_wall_at, timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_ms": round(elapsed_seconds * 1000, 3),
        "device": device,
    }
    return {**clean_result, "provenance": provenance}
