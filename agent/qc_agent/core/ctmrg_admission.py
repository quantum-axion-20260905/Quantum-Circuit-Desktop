"""Scientific admission gates for bounded CTMRG results.

Execution success only means that contractions completed.  This module keeps
that operational state separate from evidence needed before a result can be
used as a declared research input.  It never changes the numerical answer or
silently substitutes a finite reference.
"""

from __future__ import annotations

import math
from typing import Any


def ctmrg_research_gate(
    payload: Any,
    *,
    converged: bool,
    reference_validation: dict[str, Any],
    gauge_validation: dict[str, Any],
    optimization_info: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return explicit bounded evidence gates for one CTMRG result."""

    virtual_bond_dim = int(payload.virtual_bond_dim)
    optimizer = str(getattr(payload, "full_update_optimizer", "coordinate"))
    optimization = str(getattr(payload, "optimization", "none"))
    is_entangled = virtual_bond_dim > 1

    gates: dict[str, dict[str, Any]] = {
        "ctmrg_convergence": {
            "passed": bool(converged),
            "reason": "declared environment residual reached tolerance"
            if converged else
            "environment residual did not reach tolerance",
        },
        "independent_reference": {
            "passed": bool(reference_validation.get("performed") and reference_validation.get("passed")),
            "performed": bool(reference_validation.get("performed")),
            "reference": reference_validation.get("reference"),
            "reason": (
                "independent finite/reference comparison passed"
                if reference_validation.get("performed") and reference_validation.get("passed") else
                "independent reference is unavailable or exceeded its tolerance"
            ),
        },
        "virtual_gauge": {
            "passed": (
                True
                if not is_entangled else
                bool(gauge_validation.get("performed") and gauge_validation.get("passed"))
            ),
            "performed": bool(gauge_validation.get("performed")),
            "reason": (
                "virtual_bond_dim=1 has no non-trivial gauge"
                if not is_entangled else
                "paired virtual-gauge probe passed"
                if gauge_validation.get("performed") and gauge_validation.get("passed") else
                "entangled result lacks a passing gauge-invariance probe"
            ),
        },
    }

    truncation_mode = str(getattr(payload, "full_update_truncation_gradient", "frozen-eigenprojector"))
    if optimization == "full-update" and optimizer in {
        "autodiff-ctmrg-gradient",
        "implicit-ctmrg-gradient",
    } and is_entangled:
        gates["truncation_gradient"] = {
            "passed": truncation_mode == "differentiable-eigh",
            "mode": truncation_mode,
            "reason": (
                "differentiable Hermitian truncation is selected"
                if truncation_mode == "differentiable-eigh" else
                "frozen truncation projector is not an admitted entangled gradient"
            ),
        }
    else:
        gates["truncation_gradient"] = {
            "passed": True,
            "mode": truncation_mode,
            "reason": "gradient truncation gate is not required for this contraction mode",
        }

    if optimization == "full-update" and optimizer == "implicit-ctmrg-gradient":
        transfer_gap = float(optimization_info.get("transfer_gap", 0.0)) if optimization_info else 0.0
        adjoint_residual = float(optimization_info.get("adjoint_residual", math.inf)) if optimization_info else math.inf
        adjoint_tolerance = float(getattr(payload, "full_update_implicit_tolerance", 1e-6))
        gates["transfer_gap"] = {
            "passed": bool(not is_entangled or transfer_gap > 1e-5),
            "value": transfer_gap,
            "reason": "transfer spectrum is resolved" if transfer_gap > 1e-5 else "transfer gap is unresolved",
        }
        gates["adjoint_residual"] = {
            "passed": bool(adjoint_residual <= adjoint_tolerance),
            "value": adjoint_residual,
            "tolerance": adjoint_tolerance,
            "reason": "adjoint backward residual is within tolerance"
            if adjoint_residual <= adjoint_tolerance else
            "adjoint backward residual exceeds tolerance",
        }
    else:
        gates["transfer_gap"] = {"passed": True, "reason": "implicit transfer-gap gate is not required"}
        gates["adjoint_residual"] = {"passed": True, "reason": "implicit adjoint gate is not required"}

    blocking_reasons = [
        str(gate["reason"])
        for gate in gates.values()
        if not bool(gate.get("passed"))
    ]
    all_passed = not blocking_reasons
    return {
        "status": "passed" if all_passed else "needs_review",
        "scope": "bounded-declared-ctmrg-contract",
        "production_ready": bool(all_passed and not is_entangled),
        "virtual_bond_dim": virtual_bond_dim,
        "optimizer": optimizer,
        "gates": gates,
        "blocking_reasons": blocking_reasons,
        "limitations": [
            "a passed bounded gate is not a proof of thermodynamic-limit convergence",
            "entangled cells require a passing virtual-gauge probe before production admission",
        ],
    }


def dynamic_ctmrg_research_gate(
    *,
    unit_cell: tuple[int, int],
    converged: bool,
    residual: float,
    tolerance: float,
    energy_complete: bool,
    transfer_gaps: list[float | None],
    synchronized_sector_retry: bool,
    reference_validation: dict[str, Any] | None = None,
    boundary_mps_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return explicit admission gates for the experimental dynamic path.

    Dynamic retained dimensions are useful for bounded research probes, but a
    completed contraction is not automatically a production result.  Keeping
    these gates structured makes that distinction available to API clients and
    future domain plugins without changing the numerical value.
    """

    finite_gaps = [
        float(value)
        for value in transfer_gaps
        if value is not None and math.isfinite(float(value))
    ]
    transfer_gap_resolved = bool(
        len(finite_gaps) == len(transfer_gaps)
        and bool(finite_gaps)
        and min(finite_gaps) > 1e-5
    )
    residual_ok = bool(math.isfinite(float(residual)) and float(residual) <= float(tolerance))
    bounded_cell = (
        len(unit_cell) == 2
        and 1 <= int(unit_cell[0]) <= 2
        and 1 <= int(unit_cell[1]) <= 2
    )
    reference = reference_validation or {
        "performed": False,
        "passed": False,
        "reason": "independent reference was not requested",
    }
    reference_passed = bool(reference.get("performed") and reference.get("passed"))
    boundary_mps = boundary_mps_validation or {
        "requested": False,
        "performed": False,
        "passed": True,
        "reason": "finite-cylinder boundary-MPS cross-check was not requested",
    }
    boundary_mps_requested = bool(boundary_mps.get("requested"))
    boundary_mps_passed = bool(
        not boundary_mps_requested
        or (boundary_mps.get("performed") and boundary_mps.get("passed"))
    )
    gates: dict[str, dict[str, Any]] = {
        "bounded_cell": {
            "passed": bounded_cell,
            "unit_cell": [int(value) for value in unit_cell],
            "reason": "unit cell is within the bounded dynamic contract"
            if bounded_cell else
            "unit cell exceeds the bounded dynamic contract",
        },
        "environment_convergence": {
            "passed": bool(converged and residual_ok),
            "converged": bool(converged),
            "residual": float(residual),
            "tolerance": float(tolerance),
            "reason": "dynamic environment residual reached tolerance"
            if converged and residual_ok else
            "dynamic environment residual did not reach tolerance",
        },
        "energy_complete": {
            "passed": bool(energy_complete),
            "reason": "all requested observables and interactions were evaluated"
            if energy_complete else
            "one or more requested interactions could not be evaluated",
        },
        "independent_reference": {
            "passed": reference_passed,
            "performed": bool(reference.get("performed")),
            "reference": reference.get("reference"),
            "max_abs_error": reference.get("max_abs_error"),
            "reason": "independent finite/reference comparison passed"
            if reference_passed else
            str(reference.get("reason", "independent reference is unavailable or exceeded its tolerance")),
        },
        "boundary_mps_crosscheck": {
            "passed": boundary_mps_passed,
            "requested": boundary_mps_requested,
            "performed": bool(boundary_mps.get("performed")),
            "max_abs_error": boundary_mps.get("max_abs_error"),
            "reason": "finite-cylinder boundary-MPS comparison passed"
            if boundary_mps_passed and boundary_mps_requested else
            "finite-cylinder boundary-MPS cross-check was not requested"
            if not boundary_mps_requested else
            str(boundary_mps.get("reason", "boundary-MPS cross-check failed")),
        },
        "transfer_gap": {
            "passed": transfer_gap_resolved,
            "minimum": min(finite_gaps) if finite_gaps else None,
            "threshold": 1e-5,
            "reason": "all transfer sectors have a resolved leading gap"
            if transfer_gap_resolved else
            "one or more transfer sectors are unresolved",
        },
        "shared_retained_sector": {
            "passed": not bool(synchronized_sector_retry),
            "synchronized_retry": bool(synchronized_sector_retry),
            "reason": "all sites retained the requested shared sector"
            if not synchronized_sector_retry else
            "a conservative shared-sector restart reduced the requested dimension",
        },
        "public_promotion": {
            "passed": False,
            "reason": "dynamic CTMRG remains opt-in experimental and is not the default public solver",
        },
    }
    blocking_reasons = [
        str(gate["reason"])
        for gate in gates.values()
        if not bool(gate.get("passed"))
    ]
    return {
        "status": "passed" if not blocking_reasons else "needs_review",
        "scope": "bounded-dynamic-ctmrg-contract",
        "production_ready": False,
        "gates": gates,
        "blocking_reasons": blocking_reasons,
        "limitations": [
            "passing bounded gates is not a proof of thermodynamic-limit convergence",
            "the dynamic endpoint remains opt-in until independent fixed-point and optimizer evidence is complete",
        ],
    }
