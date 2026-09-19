from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


AsyncKind = Literal[
    "run", "sample", "simulate", "bench_matmul", "expectation", "tebd", "ground_state",
    "dmrg", "peps", "ctmrg", "ctmrg_convergence", "ctmrg_boundary_mps_convergence", "ctmrg_boundary_mps_transfer_convergence", "ctmrg_boundary_mps_transfer_gauge_covariance", "tn_estimate", "tn_amplitudes", "sweep", "cross_validate",
]


class AsyncSubmission(BaseModel):
    """Stable envelope for every long-running compute operation."""

    kind: AsyncKind
    payload: dict[str, Any]


class AsyncBudget(BaseModel):
    """Admission and queue limits shared by every asynchronous operation."""

    max_qubits: int | None = Field(default=None, ge=1, le=4096)
    max_shots: int | None = Field(default=None, ge=1, le=10_000_000)
    max_time_ms: int | None = Field(default=None, ge=100, le=3_600_000)
    max_mem_mb: float | None = Field(default=None, gt=0, le=1_048_576)
    queue_timeout_ms: int = Field(default=120000, ge=0, le=3_600_000)
    reservation_mb: float | None = Field(default=None, gt=0, le=1_048_576)
    device_id: int | None = Field(default=None, ge=0, le=128)
