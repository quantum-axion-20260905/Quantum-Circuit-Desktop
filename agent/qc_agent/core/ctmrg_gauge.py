"""Bounded virtual-gauge probes for CTMRG result validation.

The probe applies paired inverse-transpose/invertible transformations on
periodic virtual bonds.  It is a diagnostic, not a gauge-fixing algorithm:
the exact finite PEPS contraction should be invariant, while a truncated
environment can expose gauge sensitivity that must remain visible to users.
"""

from __future__ import annotations

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
