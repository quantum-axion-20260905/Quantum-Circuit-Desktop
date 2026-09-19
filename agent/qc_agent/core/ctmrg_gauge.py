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
    accepted_transform_count = 0
    rejected_transform_count = 0

    def condition_not_worse(candidate: float, current: float) -> bool:
        if math.isinf(current):
            return not math.isinf(candidate) or candidate <= current
        return candidate <= current * (1.0 + 1e-9)

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
            inverse_transform = xp.linalg.inv(transform).T
            candidate_source = _apply_leg_transform(
                xp, working[source], source_axis, transform
            )
            candidate_target = _apply_leg_transform(
                xp,
                candidate_source if target == source else working[target],
                target_axis,
                inverse_transform,
            )
            candidate_metric = bond_metric(
                candidate_source if target != source else candidate_target,
                candidate_target,
                source_axis,
                target_axis,
            )
            if (
                candidate_metric[0] < current_metric[0] - 1e-12
                and condition_not_worse(candidate_metric[1], current_metric[1])
            ):
                working[source] = candidate_source if target != source else candidate_target
                working[target] = candidate_target
                accepted_transform_count += 1
                mutated = True
            else:
                rejected_transform_count += 1
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
        "accepted_transform_count": int(accepted_transform_count),
        "rejected_transform_count": int(rejected_transform_count),
        "acceptance_rule": "bond mismatch must decrease without increasing the paired Gram condition number",
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


def diagonal_bond_balance_preconditioner(
    xp: Any,
    tensors: list[Any],
    *,
    unit_cell: tuple[int, int] = (1, 1),
    iterations: int = 4,
    eigenvalue_floor: float = 1e-10,
) -> tuple[list[Any], dict[str, Any]]:
    """Apply a conservative diagonal virtual-bond balancing candidate.

    For a periodic bond with lower/target Gram diagonals ``a`` and ``b`` the
    diagonal gauge ``X=diag((b/a)**1/4)`` makes the two diagonal metrics agree
    to first order under the exact paired action ``X`` / ``X**(-T)``.  A
    candidate is accepted only when the full Hermitian bond mismatch decreases
    and the paired Gram condition number does not worsen.  This is an
    intentionally small, backend-neutral canonicalization probe: it does not
    claim to construct the minimal PEPS canonical form and remains opt-in.
    """

    nx, ny = (int(value) for value in unit_cell)
    if (nx, ny) not in ((1, 1), (2, 1), (1, 2), (2, 2)):
        return list(tensors), {
            "performed": False,
            "method": "diagonal-bond-balance",
            "reason": "the bounded candidate supports only 1x1 through 2x2 periodic cells",
            "mutated_tensors": False,
        }
    if len(tensors) != nx * ny:
        return list(tensors), {
            "performed": False,
            "method": "diagonal-bond-balance",
            "reason": "tensor count does not match the declared periodic unit cell",
            "mutated_tensors": False,
        }
    if int(iterations) < 1:
        raise ValueError("diagonal bond-balance iterations must be positive")

    def site_index(x: int, y: int) -> int:
        return (x % nx) + nx * (y % ny)

    bonds: list[tuple[str, int, int, int, int]] = []
    for y in range(ny):
        for x in range(nx):
            source = site_index(x, y)
            bonds.append(("horizontal", source, site_index(x + 1, y), 4, 3))
            bonds.append(("vertical", source, site_index(x, y + 1), 2, 1))

    def bond_metric(
        current: list[Any],
        source: int,
        target: int,
        source_axis: int,
        target_axis: int,
    ) -> tuple[float, float]:
        source_gram = _leg_gram(xp, current[source], source_axis)
        target_gram = _leg_gram(xp, current[target], target_axis)
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

    def condition_not_worse(candidate: float, current: float) -> bool:
        if math.isinf(current):
            return not math.isinf(candidate) or candidate <= current
        return candidate <= current * (1.0 + 1e-9)

    def metrics(current: list[Any]) -> list[tuple[float, float]]:
        return [
            bond_metric(current, source, target, source_axis, target_axis)
            for _, source, target, source_axis, target_axis in bonds
        ]

    def score(current: list[Any]) -> tuple[float, float]:
        current_metrics = metrics(current)
        return (
            float(sum(metric[0] for metric in current_metrics)),
            float(max((metric[1] for metric in current_metrics), default=0.0)),
        )

    working = list(tensors)
    before_metrics = metrics(working)
    score_before = (
        float(sum(metric[0] for metric in before_metrics)),
        float(max((metric[1] for metric in before_metrics), default=0.0)),
    )
    accepted_transform_count = 0
    rejected_transform_count = 0
    mutated = False

    for _ in range(int(iterations)):
        for _, source, target, source_axis, target_axis in bonds:
            current_metric = bond_metric(working, source, target, source_axis, target_axis)
            source_gram = _leg_gram(xp, working[source], source_axis)
            target_gram = _leg_gram(xp, working[target], target_axis)
            source_diag = xp.maximum(xp.real(xp.diag(source_gram)), 1e-30)
            target_diag = xp.maximum(xp.real(xp.diag(target_gram)), 1e-30)
            scale = xp.power(target_diag / source_diag, 0.25)
            transform = xp.diag(scale)
            inverse_transform = xp.diag(1.0 / scale)
            candidate_source = _apply_leg_transform(
                xp, working[source], source_axis, transform
            )
            candidate_target = _apply_leg_transform(
                xp,
                candidate_source if target == source else working[target],
                target_axis,
                inverse_transform,
            )
            candidate = list(working)
            candidate[source] = candidate_source if target != source else candidate_target
            candidate[target] = candidate_target
            candidate_metric = bond_metric(candidate, source, target, source_axis, target_axis)
            current_score = score(working)
            candidate_score = score(candidate)
            if (
                candidate_metric[0] < current_metric[0] - 1e-12
                and candidate_score[0] < current_score[0] - 1e-12
                and condition_not_worse(candidate_score[1], current_score[1])
            ):
                working = candidate
                accepted_transform_count += 1
                mutated = True
            else:
                rejected_transform_count += 1

    after_metrics = metrics(working)
    score_after = (
        float(sum(metric[0] for metric in after_metrics)),
        float(max((metric[1] for metric in after_metrics), default=0.0)),
    )
    vertical_before = [metric for bond, metric in zip(bonds, before_metrics) if bond[0] == "vertical"]
    vertical_after = [metric for bond, metric in zip(bonds, after_metrics) if bond[0] == "vertical"]
    horizontal_before = [metric for bond, metric in zip(bonds, before_metrics) if bond[0] == "horizontal"]
    horizontal_after = [metric for bond, metric in zip(bonds, after_metrics) if bond[0] == "horizontal"]
    return working, {
        "performed": True,
        "method": "diagonal-bond-balance",
        "iterations": int(iterations),
        "eigenvalue_floor": float(eigenvalue_floor),
        "balance_power": 0.25,
        "vertical_pair_delta_before": max((metric[0] for metric in vertical_before), default=0.0),
        "vertical_pair_delta_after": max((metric[0] for metric in vertical_after), default=0.0),
        "horizontal_pair_delta_before": max((metric[0] for metric in horizontal_before), default=0.0),
        "horizontal_pair_delta_after": max((metric[0] for metric in horizontal_after), default=0.0),
        "total_pair_delta_before": score_before[0],
        "total_pair_delta_after": score_after[0],
        "condition_number_before": max((metric[1] for metric in before_metrics), default=0.0),
        "condition_number_after": max((metric[1] for metric in after_metrics), default=0.0),
        "accepted_transform_count": int(accepted_transform_count),
        "rejected_transform_count": int(rejected_transform_count),
        "acceptance_rule": "local and total Hermitian bond mismatch must decrease without increasing the global paired Gram condition number",
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
            "diagonal metric balancing is not a general PEPS canonical form",
            "it does not remove off-diagonal or long-range gauge sensitivity",
            "independent finite-PEPS/reference and paired-gauge gates remain mandatory",
            "not admitted into optimization or production paths until those gates pass",
        ],
    }


def paired_virtual_gauge(xp: Any, tensors: list[Any]) -> list[Any]:
    """Apply a deterministic, invertible paired gauge to each unit-cell tensor."""

    if not tensors:
        return []
    matrices = paired_virtual_gauge_matrices(xp, tensors[0])
    up, down, left, right = matrices
    return [
        xp.einsum(
            "sUDLR,uU,dD,lL,rR->sudlr",
            tensor,
            up,
            down,
            left,
            right,
        )
        for tensor in tensors
    ]


def paired_virtual_gauge_matrices(xp: Any, tensor: Any) -> tuple[Any, Any, Any, Any]:
    """Return the explicit ``(up, down, left, right)`` virtual gauges.

    The returned tuple is the exact convention used by
    :func:`paired_virtual_gauge`: down/right receive ``G`` and up/left receive
    ``G**(-T)``.  Keeping the matrices available lets the environment
    transport probe use the same convention instead of reconstructing a
    second, potentially inconsistent gauge rule.
    """

    dtype = tensor.dtype
    device = getattr(tensor, "device", None)
    matrix = xp.asarray(
        [[1.2 + 0.1j, 0.2 - 0.1j], [0.0 + 0.2j, 0.8 - 0.05j]],
        dtype=dtype,
    )
    if device is not None and getattr(xp, "__name__", "") == "torch":
        matrix = matrix.to(device=device)
    inverse_transpose = xp.linalg.inv(matrix).T
    return inverse_transpose, matrix, inverse_transpose, matrix


def fused_double_layer_gauge(xp: Any, virtual_gauge: Any) -> Any:
    """Lift a ket virtual gauge to the fused ket/bra double-layer leg."""

    return xp.kron(virtual_gauge, xp.conj(virtual_gauge))


def directional_boundary_gauge_map(
    xp: Any,
    virtual_gauges: tuple[Any, Any, Any, Any],
    boundary_dim: int,
    direction: str,
) -> dict[str, Any]:
    """Return the directional enlarged-boundary gauge factors.

    The one-site CTM absorption code forms two enlarged corner factors and a
    three-leg grown edge.  Their index order is direction-dependent.  This
    helper records the exact maps in that order so a future covariant move can
    transport a dual projector instead of guessing a Kronecker ordering.

    Returned keys are ``corner_left``, ``corner_right``, ``grown_row``,
    ``grown_middle``, and ``grown_col``.  For a factor ``A`` with a row index,
    the transformed factor is ``A' = map @ A``; for a three-leg grown edge the
    transformed slice is ``grown'[:,m,:] = row @ grown[:,m,:] @ col.T`` and
    the middle index is transformed by ``grown_middle``.
    """

    if direction not in {"left", "right", "top", "bottom"}:
        raise ValueError(f"unsupported directional boundary gauge {direction!r}")
    if int(boundary_dim) < 1:
        raise ValueError("boundary_dim must be positive")
    up, down, left, right = virtual_gauges
    fused = {
        "up": fused_double_layer_gauge(xp, up),
        "down": fused_double_layer_gauge(xp, down),
        "left": fused_double_layer_gauge(xp, left),
        "right": fused_double_layer_gauge(xp, right),
    }
    identity_kwargs = {"dtype": fused["up"].dtype}
    if getattr(xp, "__name__", "") == "torch":
        identity_kwargs["device"] = fused["up"].device
    boundary_identity = xp.eye(int(boundary_dim), **identity_kwargs)

    def boundary_factor(value: Any) -> Any:
        return xp.kron(boundary_identity, value)

    def inverse_edge(value: Any) -> Any:
        return xp.linalg.inv(value).T

    if direction in {"left", "right"}:
        edge_axis = "up"
        opposite_axis = "down"
        middle_axis = "right" if direction == "left" else "left"
    else:
        edge_axis = "left"
        opposite_axis = "right"
        middle_axis = "down" if direction == "top" else "up"
    return {
        "corner_left": boundary_factor(inverse_edge(fused[edge_axis])),
        "corner_right": boundary_factor(inverse_edge(fused[opposite_axis])),
        "grown_row": boundary_factor(fused[edge_axis]),
        "grown_middle": fused[middle_axis],
        "grown_col": boundary_factor(fused[opposite_axis]),
        "direction": direction,
        "boundary_dim": int(boundary_dim),
        "fused_virtual_dim": int(fused["up"].shape[0]),
    }


def transport_ctm_environment(
    xp: Any,
    environment: Any,
    tensor: Any,
    *,
    virtual_gauges: tuple[Any, Any, Any, Any] | None = None,
) -> Any:
    """Transport edge tensors under an explicit paired virtual gauge.

    The double layer transforms on a fused virtual leg as ``G ⊗ G*``.  Its
    adjacent CTM edge therefore receives the inverse-transpose action so that
    the ordinary (non-conjugating) contraction of edge and local double layer
    is unchanged.  Corner tensors and the internal boundary basis are kept in
    the same basis; changing that basis is a separate CTM similarity gauge.

    This is an explicit covariant transport primitive, not an automatic
    canonicalizer.  It is useful for fixed-point transport, replay, and a
    paired-gauge reference gate because it makes the algebraic convention
    executable and testable.
    """

    gauges = virtual_gauges or paired_virtual_gauge_matrices(xp, tensor)
    up, down, left, right = gauges

    def fused_inverse_transpose(gauge: Any) -> Any:
        fused = fused_double_layer_gauge(xp, gauge)
        return xp.linalg.inv(fused).T

    edge_transforms = (
        fused_inverse_transpose(up),
        fused_inverse_transpose(right),
        fused_inverse_transpose(down),
        fused_inverse_transpose(left),
    )
    T1 = _apply_leg_transform(xp, environment.T1, 1, edge_transforms[0])
    T2 = _apply_leg_transform(xp, environment.T2, 1, edge_transforms[1])
    T3 = _apply_leg_transform(xp, environment.T3, 1, edge_transforms[2])
    T4 = _apply_leg_transform(xp, environment.T4, 1, edge_transforms[3])
    return type(environment)(
        environment.C1,
        environment.C2,
        environment.C3,
        environment.C4,
        T1,
        T2,
        T3,
        T4,
    )


def transport_biorthogonal_boundary_basis(
    xp: Any,
    projector: Any,
    enlarged_gauge: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """Transport a retained boundary basis as an exact dual pair.

    If an enlarged boundary basis changes by a generally non-unitary matrix
    ``K``, transporting an isometric projector as ``K @ P`` and then
    re-orthogonalizing loses the dual contraction rule.  The covariant pair is
    instead ``P_right = K @ P`` and ``P_left = K**(-H) @ P``.  It satisfies
    ``P_left.H @ P_right = I`` and ``P_left.H @ K = P.H`` up to roundoff.

    This is a truncation building block, not a complete CTMRG move: callers
    still need to derive the correct enlarged-boundary gauge ``K`` for each
    directional absorption and handle rank loss explicitly.
    """

    if getattr(projector, "ndim", None) != 2 or getattr(enlarged_gauge, "ndim", None) != 2:
        raise ValueError("boundary-basis transport requires a rank-2 projector and rank-2 enlarged gauge")
    if int(enlarged_gauge.shape[0]) != int(enlarged_gauge.shape[1]):
        raise ValueError("boundary-basis transport requires a square enlarged gauge")
    if int(enlarged_gauge.shape[0]) != int(projector.shape[0]):
        raise ValueError("enlarged gauge dimension does not match the projector row dimension")
    inverse_adjoint = xp.linalg.inv(xp.conj(enlarged_gauge).T)
    right = enlarged_gauge @ projector
    left = inverse_adjoint @ projector
    overlap = xp.conj(left).T @ right
    dual_rule = xp.conj(left).T @ enlarged_gauge - xp.conj(projector).T
    eye_kwargs = {"dtype": overlap.dtype}
    if getattr(xp, "__name__", "") == "torch":
        eye_kwargs["device"] = overlap.device
    identity = xp.eye(int(overlap.shape[0]), **eye_kwargs)
    overlap_error = _host_array(xp.linalg.norm(overlap - identity))
    dual_rule_error = _host_array(xp.linalg.norm(dual_rule))
    gauge_condition = _host_array(xp.linalg.cond(enlarged_gauge))
    numerical_tolerance = 1e-6 if "64" in str(projector.dtype) else 1e-8
    return left, right, {
        "performed": True,
        "method": "inverse-adjoint-biorthogonal-boundary-transport",
        "projector_rows": int(projector.shape[0]),
        "projector_columns": int(projector.shape[1]),
        "enlarged_gauge_condition_number": float(gauge_condition),
        "dual_overlap_error": float(overlap_error),
        "dual_rule_error": float(dual_rule_error),
        "numerical_tolerance": numerical_tolerance,
        "passed": bool(
            math.isfinite(float(gauge_condition))
            and float(overlap_error) <= numerical_tolerance
            and float(dual_rule_error) <= numerical_tolerance
        ),
        "limitations": [
            "the caller must supply the correct directional enlarged-boundary gauge",
            "rank-deficient or ill-conditioned gauges require a separate admission gate",
            "this does not by itself define a thermodynamic CTMRG fixed point",
        ],
    }


def transport_bilinear_projector_pair(
    xp: Any,
    left_projector: Any,
    right_projector: Any,
    row_gauge: Any,
    column_gauge: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """Transport the two sides of a bilinear grown-edge projection.

    For a grown edge transforming as ``E' = K_row @ E @ K_col.T``, the
    transpose-based projection rule is preserved by
    ``P_left' = K_row**(-T) @ P_left`` and
    ``P_right' = K_col**(-T) @ P_right``.  This is distinct from the
    inverse-adjoint boundary-basis transport above: CTM edge contractions are
    ordinary bilinear tensor contractions, not Hilbert inner products.
    """

    if getattr(left_projector, "ndim", None) != 2 or getattr(right_projector, "ndim", None) != 2:
        raise ValueError("bilinear projector transport requires rank-2 projectors")
    if getattr(row_gauge, "ndim", None) != 2 or getattr(column_gauge, "ndim", None) != 2:
        raise ValueError("bilinear projector transport requires rank-2 row and column gauges")
    if int(row_gauge.shape[0]) != int(row_gauge.shape[1]) or int(column_gauge.shape[0]) != int(column_gauge.shape[1]):
        raise ValueError("bilinear projector transport requires square row and column gauges")
    if int(row_gauge.shape[0]) != int(left_projector.shape[0]):
        raise ValueError("row gauge dimension does not match the left projector")
    if int(column_gauge.shape[0]) != int(right_projector.shape[0]):
        raise ValueError("column gauge dimension does not match the right projector")
    transported_left = xp.linalg.inv(row_gauge).T @ left_projector
    transported_right = xp.linalg.inv(column_gauge).T @ right_projector
    left_error = _host_array(
        xp.linalg.norm(transported_left.T @ row_gauge - left_projector.T)
    )
    right_error = _host_array(
        xp.linalg.norm(transported_right.T @ column_gauge - right_projector.T)
    )
    row_condition = _host_array(xp.linalg.cond(row_gauge))
    column_condition = _host_array(xp.linalg.cond(column_gauge))
    numerical_tolerance = 1e-6 if "64" in str(left_projector.dtype) else 1e-8
    return transported_left, transported_right, {
        "performed": True,
        "method": "inverse-transpose-bilinear-projector-transport",
        "row_condition_number": float(row_condition),
        "column_condition_number": float(column_condition),
        "left_dual_rule_error": float(left_error),
        "right_dual_rule_error": float(right_error),
        "numerical_tolerance": numerical_tolerance,
        "passed": bool(
            math.isfinite(float(row_condition))
            and math.isfinite(float(column_condition))
            and float(left_error) <= numerical_tolerance
            and float(right_error) <= numerical_tolerance
        ),
        "limitations": [
            "this transports a known grown-edge projector pair; it does not select the retained subspace",
            "corner-factor basis transport must be coordinated with the edge pair in a complete CTM move",
            "ill-conditioned row/column gauges require an explicit rank and transfer-gap gate",
        ],
    }


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
