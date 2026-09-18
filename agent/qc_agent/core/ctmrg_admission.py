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
