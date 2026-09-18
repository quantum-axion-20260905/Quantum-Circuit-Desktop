"""Bounded virtual-gauge probes for CTMRG result validation.

The probe applies paired inverse-transpose/invertible transformations on
periodic virtual bonds.  It is a diagnostic, not a gauge-fixing algorithm:
the exact finite PEPS contraction should be invariant, while a truncated
environment can expose gauge sensitivity that must remain visible to users.
"""

from __future__ import annotations

import math
from typing import Any


_VIRTUAL_LEGS = ("up", "down", "left", "right")


def _host_array(value: Any) -> Any:
    """Copy a small diagnostic array to the host without assuming one backend."""

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "get"):
        return value.get()
    return value


def virtual_leg_conditioning_report(
    xp: Any,
    tensors: list[Any],
    *,
    eigenvalue_floor: float = 1e-12,
) -> dict[str, Any]:
    """Report virtual-leg Gram spectra used by a future PEPS gauge preconditioner.

    This deliberately does not mutate tensors.  A PEPS has no universal local
    canonical form like a finite open-boundary MPS; a whitening transform must
    be paired across every virtual bond and validated against an independent
    contraction.  Returning the spectra first gives optimizers and the UI a
    backend-neutral conditioning contract without making that scientific leap.
    """

    if not tensors:
        return {
            "performed": False,
            "reason": "tensor cell is empty",
            "method": "virtual-leg-gram-spectrum",
            "legs": [],
            "well_conditioned": False,
        }
    reports: list[dict[str, Any]] = []
    for site, tensor in enumerate(tensors):
        if getattr(tensor, "ndim", None) != 5:
            return {
                "performed": False,
                "reason": "virtual-leg conditioning requires rank-5 iPEPS tensors",
                "method": "virtual-leg-gram-spectrum",
                "legs": [],
                "well_conditioned": False,
            }
        for axis, leg in enumerate(_VIRTUAL_LEGS, start=1):
            moved = tensor.movedim(axis, 0) if hasattr(tensor, "movedim") else xp.moveaxis(tensor, axis, 0)
            matrix = moved.reshape((int(moved.shape[0]), -1))
            gram = matrix @ matrix.conj().T
            eigenvalues = _host_array(xp.linalg.eigvalsh(gram))
            eigenvalues = sorted(
                max(0.0, float(value.real if hasattr(value, "real") else value))
                for value in eigenvalues
            )
            largest = eigenvalues[-1] if eigenvalues else 0.0
            smallest = eigenvalues[0] if eigenvalues else 0.0
            effective_floor = max(float(eigenvalue_floor), largest * float(eigenvalue_floor))
            condition_number = (
                largest / smallest
                if smallest > effective_floor
                else None
            )
            reports.append({
                "site": int(site),
                "leg": leg,
                "dimension": int(tensor.shape[axis]),
                "eigenvalues": eigenvalues,
                "trace": float(sum(eigenvalues)),
                "minimum_eigenvalue": smallest,
                "maximum_eigenvalue": largest,
                "condition_number": condition_number,
                "rank_estimate": int(sum(value > effective_floor for value in eigenvalues)),
                "well_conditioned": bool(smallest > effective_floor),
            })
    return {
        "performed": True,
        "method": "virtual-leg-gram-spectrum",
        "eigenvalue_floor": float(eigenvalue_floor),
        "legs": reports,
        "well_conditioned": bool(all(item["well_conditioned"] for item in reports)),
        "mutated_tensors": False,
        "limitations": [
            "this is a conditioning diagnostic, not PEPS canonicalization",
            "a future preconditioner must pair inverse transforms across every virtual bond",
            "conditioning does not establish CTMRG gauge invariance or thermodynamic convergence",
        ],
    }


def _moveaxis(xp: Any, value: Any, source: int, destination: int) -> Any:
    """Move one tensor axis without importing a backend-specific helper."""

    if hasattr(value, "movedim"):
        return value.movedim(source, destination)
    return xp.moveaxis(value, source, destination)


def _leg_gram(xp: Any, tensor: Any, axis: int) -> Any:
    moved = _moveaxis(xp, tensor, axis, 0)
    matrix = moved.reshape((int(moved.shape[0]), -1))
    gram = matrix @ xp.conj(matrix).T
    return 0.5 * (gram + xp.conj(gram).T)


def _hermitian_sqrt(xp: Any, matrix: Any, *, inverse: bool, eigenvalue_floor: float) -> Any:
    values, vectors = xp.linalg.eigh(0.5 * (matrix + xp.conj(matrix).T))
    values = xp.real(values)
    scale = max(float(max(_host_array(values))), 1e-30) if int(values.shape[0]) else 1.0
    floor = max(float(eigenvalue_floor) * scale, 1e-30)
    values = xp.maximum(values, xp.full_like(values, floor))
    factors = 1.0 / xp.sqrt(values) if inverse else xp.sqrt(values)
    return (vectors * factors[None, :]) @ xp.conj(vectors).T


def _pair_balance_transform(
    xp: Any,
    lower_gram: Any,
    upper_gram: Any,
    *,
    eigenvalue_floor: float,
) -> Any:
    """Return X for lower<-X and upper<-inv(X).T pair balancing.

    The positive solution balances ``X G_lower X†`` against
    ``X⁻ᵀ G_upper X⁻*``.  It is a bounded local gauge choice, not a PEPS
    canonical-form theorem; callers must keep it opt-in and reference-tested.
    """

    lower_root = _hermitian_sqrt(
        xp, lower_gram, inverse=False, eigenvalue_floor=eigenvalue_floor
    )
    lower_inverse_root = _hermitian_sqrt(
        xp, lower_gram, inverse=True, eigenvalue_floor=eigenvalue_floor
    )
    middle = lower_root @ upper_gram @ lower_root
    squared = lower_inverse_root @ _hermitian_sqrt(
        xp, middle, inverse=False, eigenvalue_floor=eigenvalue_floor
    ) @ lower_inverse_root
    squared = 0.5 * (squared + xp.conj(squared).T)
    return _hermitian_sqrt(xp, squared, inverse=False, eigenvalue_floor=eigenvalue_floor)


def _pair_metric(xp: Any, tensor: Any, lower_axis: int, upper_axis: int) -> tuple[float, float]:
    lower = _leg_gram(xp, tensor, lower_axis)
    upper = _leg_gram(xp, tensor, upper_axis)
    delta = _host_array(xp.linalg.norm(lower - upper))
    scale = max(_host_array(xp.linalg.norm(lower)), _host_array(xp.linalg.norm(upper)), 1e-30)
    values = []
    for gram in (lower, upper):
        values.extend(float(value.real if hasattr(value, "real") else value) for value in _host_array(xp.linalg.eigvalsh(gram)))
    positive = [max(value, 0.0) for value in values]
    condition = max(positive) / max(min(positive), 1e-30) if positive else math.inf
    return float(delta / scale), float(condition)


def _apply_leg_transform(xp: Any, tensor: Any, axis: int, matrix: Any) -> Any:
    if getattr(xp, "__name__", "") == "torch":
        transformed = xp.tensordot(matrix, tensor, dims=([1], [axis]))
    else:
        transformed = xp.tensordot(matrix, tensor, axes=([1], [axis]))
    return _moveaxis(xp, transformed, 0, axis)


def pairwise_virtual_gauge_preconditioner(
    xp: Any,
    tensors: list[Any],
    *,
    unit_cell: tuple[int, int] = (1, 1),
    iterations: int = 4,
    eigenvalue_floor: float = 1e-10,
) -> tuple[list[Any], dict[str, Any]]:
    """Apply a bounded bond-aware paired polar-balance candidate.

    Every periodic right/left and down/up bond receives ``X`` on one endpoint
    and the exact inverse-transpose on the other.  This preserves the exact
    finite PEPS contraction for 1x1 through 2x2 cells.  It is still a bounded
    local conditioning candidate, not a PEPS canonical-form theorem.
    """

    nx, ny = (int(value) for value in unit_cell)
    if (nx, ny) not in ((1, 1), (2, 1), (1, 2), (2, 2)):
        return list(tensors), {
            "performed": False,
            "method": "pairwise-polar-balance",
            "reason": "the bounded candidate supports only 1x1 through 2x2 periodic cells",
            "mutated_tensors": False,
        }
    if len(tensors) != nx * ny:
        return list(tensors), {
            "performed": False,
            "method": "pairwise-polar-balance",
            "reason": "tensor count does not match the declared periodic unit cell",
            "mutated_tensors": False,
        }
    if int(iterations) < 1:
        raise ValueError("pairwise virtual-gauge preconditioner iterations must be positive")

    def site_index(x: int, y: int) -> int:
        return (x % nx) + nx * (y % ny)

    bonds: list[tuple[str, int, int, int, int]] = []
    for y in range(ny):
        for x in range(nx):
            source = site_index(x, y)
            bonds.append(("horizontal", source, site_index(x + 1, y), 4, 3))
            bonds.append(("vertical", source, site_index(x, y + 1), 2, 1))

    working = list(tensors)
    mutated = False

    def bond_metric(
        source_tensor: Any,
        target_tensor: Any,
        source_axis: int,
        target_axis: int,
    ) -> tuple[float, float]:
        source_gram = _leg_gram(xp, source_tensor, source_axis)
        target_gram = _leg_gram(xp, target_tensor, target_axis)
        delta = _host_array(xp.linalg.norm(source_gram - target_gram))
        scale = max(
            _host_array(xp.linalg.norm(source_gram)),
            _host_array(xp.linalg.norm(target_gram)),
            1e-30,
        )
        values = []
        for gram in (source_gram, target_gram):
            values.extend(
                float(value.real if hasattr(value, "real") else value)
                for value in _host_array(xp.linalg.eigvalsh(gram))
            )
        positive = [max(value, 0.0) for value in values]
        condition = max(positive) / max(min(positive), 1e-30) if positive else math.inf
        return float(delta / scale), float(condition)

    def metrics(current: list[Any]) -> list[tuple[float, float]]:
        return [
            bond_metric(current[source], current[target], source_axis, target_axis)
            for _, source, target, source_axis, target_axis in bonds
        ]

    before_metrics = metrics(working)

    for _ in range(int(iterations)):
        for _, source, target, source_axis, target_axis in bonds:
            current_metric = bond_metric(
                working[source], working[target], source_axis, target_axis
            )
            if current_metric[0] <= 1e-12:
                continue
            transform = _pair_balance_transform(
                xp,
                _leg_gram(xp, working[source], source_axis),
                _leg_gram(xp, working[target], target_axis),
                eigenvalue_floor=eigenvalue_floor,
            )
            working[source] = _apply_leg_transform(xp, working[source], source_axis, transform)
            working[target] = _apply_leg_transform(
                xp,
                working[target],
                target_axis,
                xp.linalg.inv(transform).T,
            )
            mutated = True
    after_metrics = metrics(working)
    vertical_before = [metric for bond, metric in zip(bonds, before_metrics) if bond[0] == "vertical"]
    vertical_after = [metric for bond, metric in zip(bonds, after_metrics) if bond[0] == "vertical"]
    horizontal_before = [metric for bond, metric in zip(bonds, before_metrics) if bond[0] == "horizontal"]
    horizontal_after = [metric for bond, metric in zip(bonds, after_metrics) if bond[0] == "horizontal"]
    return working, {
        "performed": True,
        "method": "pairwise-polar-balance",
        "iterations": int(iterations),
        "eigenvalue_floor": float(eigenvalue_floor),
        "vertical_pair_delta_before": max((metric[0] for metric in vertical_before), default=0.0),
        "vertical_pair_delta_after": max((metric[0] for metric in vertical_after), default=0.0),
        "horizontal_pair_delta_before": max((metric[0] for metric in horizontal_before), default=0.0),
        "horizontal_pair_delta_after": max((metric[0] for metric in horizontal_after), default=0.0),
        "condition_number_before": max((metric[1] for metric in before_metrics), default=0.0),
        "condition_number_after": max((metric[1] for metric in after_metrics), default=0.0),
        "bond_metrics": [
            {
                "orientation": orientation,
                "source_site": int(source),
                "target_site": int(target),
                "source_axis": int(source_axis),
                "target_axis": int(target_axis),
                "delta_before": float(before[0]),
                "delta_after": float(after[0]),
                "condition_before": float(before[1]),
                "condition_after": float(after[1]),
            }
            for (orientation, source, target, source_axis, target_axis), before, after
            in zip(bonds, before_metrics, after_metrics)
        ],
        "mutated_tensors": mutated,
        "exact_periodic_pairing": True,
        "limitations": [
            "bounded 1x1 through 2x2 candidate; it is not a general PEPS canonical form",
            "independent finite-PEPS/reference and paired-gauge gates remain mandatory",
            "not admitted into optimization paths until those gates pass",
        ],
    }


def paired_virtual_gauge(xp: Any, tensors: list[Any]) -> list[Any]:
    """Apply a deterministic, invertible paired gauge to each unit-cell tensor."""

    if not tensors:
        return []
    dtype = tensors[0].dtype
    device = getattr(tensors[0], "device", None)
    matrix = xp.asarray([[1.2 + 0.1j, 0.2 - 0.1j], [0.0 + 0.2j, 0.8 - 0.05j]], dtype=dtype)
    if device is not None and getattr(xp, "__name__", "") == "torch":
        matrix = matrix.to(device=device)
    inverse_transpose = xp.linalg.inv(matrix).T
    return [
        xp.einsum(
            "sUDLR,uU,dD,lL,rR->sudlr",
            tensor,
            inverse_transpose,
            matrix,
            inverse_transpose,
            matrix,
        )
        for tensor in tensors
    ]


def gauge_validation_result(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    tolerance: float,
    virtual_bond_dim: int,
) -> dict[str, Any]:
    """Summarize energy/observable changes under the paired virtual gauge."""

    observable_deltas = [
        abs(float(left.get("value", 0.0)) - float(right.get("value", 0.0)))
        for left, right in zip(before.get("observables", []), after.get("observables", []))
    ]
    interaction_deltas = [
        abs(float(left.get("value", 0.0)) - float(right.get("value", 0.0)))
        for left, right in zip(before.get("interactions", []), after.get("interactions", []))
        if left.get("value") is not None and right.get("value") is not None
    ]
    energy_delta = abs(float(before["energy"]) - float(after["energy"]))
    max_delta = max([energy_delta, *observable_deltas, *interaction_deltas], default=0.0)
    return {
        "performed": True,
        "transform": "paired-invertible-virtual-gauge",
        "virtual_bond_dim": int(virtual_bond_dim),
        "energy_before": float(before["energy"]),
        "energy_after": float(after["energy"]),
        "energy_abs_delta": energy_delta,
        "observable_max_abs_delta": max(observable_deltas, default=0.0),
        "interaction_max_abs_delta": max(interaction_deltas, default=0.0),
        "max_abs_delta": max_delta,
        "tolerance": float(tolerance),
        "passed": bool(max_delta <= float(tolerance)),
        "limitations": [
            "this is a validation probe, not a canonical-gauge or gauge-fixing solver",
            "a failed truncated-CTMRG probe indicates needs_review and does not invalidate the exact finite-PEPS reference",
        ],
    }
