"""Shared CTMRG objective seam for variational tensor optimizers.

The contraction objective is deliberately kept separate from optimizer policy.
Every candidate is evaluated through the same imported-tensor CTMRG path, with
optimization and environment checkpoints disabled for the inner evaluation.
This keeps coordinate, finite-difference, SPSA, and a future AD/implicit
gradient backend on one scientific objective contract.
"""

from __future__ import annotations

from typing import Any

from ..plugins.models import CTMRGPayload
from ..provenance import sha256_json


def normalize_tensors(xp: Any, tensors: list[Any]) -> list[Any]:
    """Return independently normalized candidate tensors."""

    return [tensor / (xp.linalg.norm(tensor) + 1e-30) for tensor in tensors]


def optimizer_request_sha256(payload: CTMRGPayload) -> str:
    """Hash the scientific optimizer problem, excluding run controls."""

    data = payload.model_dump(
        mode="json",
        exclude={
            "optimization_steps",
            "optimization_tolerance",
            "full_update_max_evaluations",
            "max_time_ms",
            "max_mem_mb",
            "checkpoint_path",
            "resume_from",
            "optimizer_checkpoint_path",
            "optimizer_resume_from",
        },
    )
    return sha256_json(data)


class CTMRGObjective:
    """Bounded evaluator for the infinite-system CTMRG energy objective.

    The current implementation uses the existing CTMRG contraction as the
    oracle. It intentionally exposes no gradient method: finite-difference and
    SPSA remain explicit optimizer policies, while a future differentiable
    backend can implement the same evaluator contract without changing the
    request/result surface.
    """

    def __init__(self, xp: Any, payload: CTMRGPayload, run_ctmrg: Any) -> None:
        self.xp = xp
        self.max_evaluations = int(payload.full_update_max_evaluations)
        self.evaluations = 0
        self._run_ctmrg = run_ctmrg
        self.payload = payload.model_copy(update={
            "tensor_data": None,
            "optimization": "none",
            "checkpoint_path": None,
            "resume_from": None,
            "optimizer_checkpoint_path": None,
            "optimizer_resume_from": None,
        })

    @property
    def remaining_evaluations(self) -> int:
        return max(0, self.max_evaluations - self.evaluations)

    def evaluate(self, candidate: list[Any]) -> dict[str, Any] | None:
        """Evaluate a candidate tensor list, or return ``None`` at the budget."""

        if self.evaluations >= self.max_evaluations:
            return None
        normalized = normalize_tensors(self.xp, [tensor.copy() for tensor in candidate])
        result = self._run_ctmrg(self.xp, self.payload, tensors=normalized)
        self.evaluations += 1
        if not result.get("energy_complete", False):
            raise ValueError("CTMRG objective requires complete nearest-neighbor interaction energy")
        return result
