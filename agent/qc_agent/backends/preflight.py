from __future__ import annotations

import math
from typing import Any

from ..models import PreflightPayload


def estimate(payload: PreflightPayload, cotengra_available: bool = False) -> dict[str, Any]:
    n = payload.n_qubits
    warnings: list[str] = []
    for g in payload.gates:
        if g.target >= n or (g.control is not None and g.control >= n):
            return {"feasible": False, "status": "rejected", "warnings": ["gate index exceeds qubit count"]}
        if g.name in ("rx", "ry", "rz") and g.theta is None:
            return {"feasible": False, "status": "rejected", "warnings": [f"{g.name} requires theta"]}
    two_q = sum(1 for g in payload.gates if g.name in ("cx", "cz"))
    # This is a conservative TN planning estimate, not a statevector memory claim.
    path_cost = max(1, len(payload.gates)) * max(1, 2 ** min(two_q, 20))
    largest = 2 ** min(n, 20)
    peak_mb = largest * (8 if payload.dtype == "complex64" else 16) / (1024 * 1024)
    if payload.optimize == "cotengra" and not cotengra_available:
        warnings.append("cotengra unavailable; optimizer will fall back to auto")
    if peak_mb > payload.max_mem_mb:
        warnings.append(f"estimated intermediate {peak_mb:.1f} MB exceeds memory budget")
    if two_q > 18: warnings.append("high two-qubit interaction count; contraction may be expensive")
    estimated_ms = int(1 + path_cost / 5000)
    feasible = peak_mb <= payload.max_mem_mb and estimated_ms <= payload.max_time_ms
    return {"status": "ready" if feasible else "rejected", "feasible": feasible, "backend": "tensor-network", "n_qubits": n, "path_steps": len(payload.gates), "path_cost": path_cost, "largest_intermediate": largest, "estimated_peak_memory_mb": round(peak_mb, 3), "estimated_time_ms": estimated_ms, "dtype": payload.dtype, "optimize": payload.optimize, "warnings": warnings}
