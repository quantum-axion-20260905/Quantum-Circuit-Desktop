"""Versioned contracts shared by tensor-network representations and solvers.

The numerical implementations deliberately remain independent from this
module. These small, serializable contracts are the seam that lets MPS, PEPS,
boundary-MPS, CTMRG, and future chemistry engines share admission, results,
and checkpoint semantics.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, Sequence, runtime_checkable


CONTRACT_SCHEMA = "quantum-circuit/research-contracts-v1"
RESULT_SCHEMA = "quantum-circuit/research-result-v1"
CHECKPOINT_SCHEMA = "quantum-circuit/checkpoint-v1"
ResultStatus = Literal["done", "needs_review", "failed", "canceled"]


class CapabilityError(RuntimeError):
    """Typed failure for an unsupported representation/algorithm operation."""


def _positive_number(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return value


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class ResourceBudget:
    """Bounded execution budget shared by synchronous and async solvers."""

    device_id: int | None = None
    max_gpu_mb: float = 4096.0
    max_host_mb: float = 4096.0
    max_time_ms: int = 120_000
    queue_timeout_ms: int = 120_000

    def __post_init__(self) -> None:
        if self.device_id is not None and int(self.device_id) < 0:
            raise ValueError("device_id must be non-negative")
        _positive_number("max_gpu_mb", self.max_gpu_mb)
        _positive_number("max_host_mb", self.max_host_mb)
        _positive_int("max_time_ms", self.max_time_ms)
        _positive_int("queue_timeout_ms", self.queue_timeout_ms)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TruncationReport:
    discarded_weight: float = 0.0
    cutoff: float = 0.0
    max_bond_dim: int | None = None
    max_environment_dim: int | None = None

    def __post_init__(self) -> None:
        for name, value in (("discarded_weight", self.discarded_weight), ("cutoff", self.cutoff)):
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name, value in (("max_bond_dim", self.max_bond_dim), ("max_environment_dim", self.max_environment_dim)):
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when provided")


@dataclass(frozen=True)
class ConvergencePoint:
    iteration: int
    energy: float | None = None
    residual: float | None = None
    norm_drift: float | None = None
    discarded_weight: float | None = None
    bond_dim: int | None = None
    environment_dim: int | None = None

    def __post_init__(self) -> None:
        if int(self.iteration) < 0:
            raise ValueError("convergence iteration must be non-negative")
        for name in ("energy", "residual", "norm_drift", "discarded_weight"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when provided")


@dataclass
class ConvergenceReport:
    converged: bool = False
    criterion: str = ""
    points: list[ConvergencePoint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "criterion": self.criterion,
            "points": [asdict(point) for point in self.points],
            "warnings": list(self.warnings),
        }


@dataclass
class ResearchResult:
    """Stable result envelope for every numerical engine."""

    status: ResultStatus
    method: str
    representation: str
    metrics: dict[str, Any] = field(default_factory=dict)
    truncation: TruncationReport = field(default_factory=TruncationReport)
    convergence: ConvergenceReport = field(default_factory=ConvergenceReport)
    resources: dict[str, Any] = field(default_factory=dict)
    checkpoint: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.method.strip():
            raise ValueError("result method must not be empty")
        if not self.representation.strip():
            raise ValueError("result representation must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESULT_SCHEMA,
            "status": self.status,
            "method": self.method,
            "representation": self.representation,
            "metrics": dict(self.metrics),
            "truncation": asdict(self.truncation),
            "convergence": self.convergence.to_dict(),
            "resources": dict(self.resources),
            "checkpoint": dict(self.checkpoint),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "provenance": dict(self.provenance),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class CheckpointManifest:
    """Metadata needed to decide whether a checkpoint can be resumed safely."""

    checkpoint_id: str
    request_sha256: str
    method: str
    representation: str
    dtype: str
    device: str
    step: int
    created_at: str
    resumable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.checkpoint_id.strip():
            raise ValueError("checkpoint_id must not be empty")
        if len(self.request_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.request_sha256.lower()):
            raise ValueError("request_sha256 must be a 64-character hexadecimal digest")
        if not self.method.strip() or not self.representation.strip():
            raise ValueError("checkpoint method and representation are required")
        if int(self.step) < 0:
            raise ValueError("checkpoint step must be non-negative")
        if not self.created_at.strip():
            raise ValueError("checkpoint created_at is required")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": CHECKPOINT_SCHEMA, **asdict(self)}


@runtime_checkable
class Representation(Protocol):
    """Minimum protocol consumed by generic observables and checkpoint code."""

    def norm2(self) -> float: ...

    def expectation(self, terms: Sequence[Any]) -> list[float]: ...

    def estimate_resources(self) -> dict[str, Any]: ...


@runtime_checkable
class Solver(Protocol):
    """Protocol for future DMRG/PEPS/CTMRG engines."""

    method: str

    def run(self, problem: Any, budget: ResourceBudget) -> ResearchResult: ...
