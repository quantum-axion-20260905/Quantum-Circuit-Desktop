"""Shared finite-MPS storage and canonical-gauge conventions.

All finite MPS tensors in the agent use the open-boundary layout
``(left_bond, physical, right_bond)``.  Keeping this contract in one small
module prevents DMRG, TEBD, observables, and checkpoint code from silently
acquiring different axis orders.
"""

from __future__ import annotations

from typing import Any, Iterable


MPS_TENSOR_AXIS_ORDER = ("left_bond", "physical", "right_bond")
MPS_PHYSICAL_DIM = 2


def validate_mps_tensors(
    tensors: Iterable[Any],
    *,
    physical_dim: int = MPS_PHYSICAL_DIM,
    require_open_boundaries: bool = True,
) -> list[tuple[int, int, int]]:
    """Validate and return the finite-MPS shape contract.

    The function intentionally checks only representation invariants.  It does
    not check normalization or canonical gauge because those are state
    properties and can legitimately change after an operator is applied.
    """

    items = list(tensors)
    if not items:
        raise ValueError("finite MPS must contain at least one tensor")

    shapes: list[tuple[int, int, int]] = []
    for index, tensor in enumerate(items):
        shape = tuple(int(value) for value in getattr(tensor, "shape", ()))
        if len(shape) != 3:
            raise ValueError(
                f"MPS tensor {index} must use axis order {MPS_TENSOR_AXIS_ORDER}; got shape {shape}"
            )
        left, physical, right = shape
        if min(shape) < 1:
            raise ValueError(f"MPS tensor {index} has an empty bond or physical axis: {shape}")
        if physical != int(physical_dim):
            raise ValueError(
                f"MPS tensor {index} physical dimension must be {physical_dim}; got {physical}"
            )
        shapes.append((left, physical, right))

    if require_open_boundaries and (shapes[0][0] != 1 or shapes[-1][2] != 1):
        raise ValueError(
            "finite open MPS must have left boundary 1 and right boundary 1; "
            f"got {shapes[0][0]} and {shapes[-1][2]}"
        )

    for index, (current, following) in enumerate(zip(shapes, shapes[1:])):
        if current[2] != following[0]:
            raise ValueError(
                f"MPS bond mismatch between sites {index} and {index + 1}: "
                f"right={current[2]} != left={following[0]}"
            )
    return shapes


def _host_scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _frobenius_error(xp: Any, residual: Any) -> float:
    return float(_host_scalar(xp.linalg.norm(residual)))


def left_isometry_error(xp: Any, tensor: Any) -> float:
    """Return ``||A†A-I||`` for a left-canonical tensor."""

    left, physical, right = (int(value) for value in tensor.shape)
    matrix = tensor.reshape(left * physical, right)
    identity = xp.eye(right, dtype=tensor.dtype)
    return _frobenius_error(xp, matrix.conj().T @ matrix - identity)


def right_isometry_error(xp: Any, tensor: Any) -> float:
    """Return ``||AA†-I||`` for a right-canonical tensor."""

    left, physical, right = (int(value) for value in tensor.shape)
    matrix = tensor.reshape(left, physical * right)
    identity = xp.eye(left, dtype=tensor.dtype)
    return _frobenius_error(xp, matrix @ matrix.conj().T - identity)


def canonical_form_report(
    xp: Any,
    tensors: Iterable[Any],
    *,
    orthogonality_center: int | None = None,
) -> dict[str, Any]:
    """Report canonical-gauge residuals without changing the tensors.

    If a center is supplied, sites strictly to its left are expected to be
    left-isometric and sites strictly to its right right-isometric.  The
    center itself is intentionally excluded because it carries the remaining
    norm/Schmidt weight.
    """

    items = list(tensors)
    shapes = validate_mps_tensors(items)
    if orthogonality_center is None:
        left_sites = range(len(items))
        right_sites = range(len(items))
    else:
        center = int(orthogonality_center)
        if center < 0 or center >= len(items):
            raise ValueError(f"orthogonality center is outside the MPS: {center}")
        left_sites = range(0, center)
        right_sites = range(center + 1, len(items))

    left_errors = [left_isometry_error(xp, items[index]) for index in left_sites]
    right_errors = [right_isometry_error(xp, items[index]) for index in right_sites]
    return {
        "axis_order": list(MPS_TENSOR_AXIS_ORDER),
        "physical_dim": int(shapes[0][1]),
        "tensor_shapes": [list(shape) for shape in shapes],
        "orthogonality_center": orthogonality_center,
        "left_sites": len(left_errors),
        "right_sites": len(right_errors),
        "left_isometry_max_error": max(left_errors, default=0.0),
        "right_isometry_max_error": max(right_errors, default=0.0),
    }
