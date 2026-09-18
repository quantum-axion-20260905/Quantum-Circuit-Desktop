"""Optional torch-autograd full-update backend for bounded infinite CTMRG.

This backend differentiates an unrolled, bounded CTMRG environment objective.
It is intentionally separate from the CuPy production path because torch is
an optional research dependency and because eigenvector truncation makes the
gradient well-defined only away from transfer-spectrum degeneracies. Results
remain ``needs_review`` until independent convergence and gauge-invariance
gates are completed.
"""

from __future__ import annotations

import math
from typing import Any

from ..plugins.models import CTMRGPayload


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "the CTMRG autograd backend requires optional agent/requirements-autodiff.txt"
        ) from exc
    return torch


def _to_torch(torch: Any, xp: Any, tensor: Any, *, dtype: Any, device: Any) -> Any:
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device=device, dtype=dtype)
    if getattr(xp, "__name__", "") == "cupy":
        return torch.utils.dlpack.from_dlpack(tensor).to(device=device, dtype=dtype)
    return torch.as_tensor(tensor, device=device, dtype=dtype)


def _from_torch(torch: Any, xp: Any, tensor: Any) -> Any:
    detached = tensor.detach()
    if getattr(xp, "__name__", "") == "cupy":
        return xp.from_dlpack(torch.utils.dlpack.to_dlpack(detached))
    if getattr(xp, "__name__", "") == "torch":
        return detached
    return detached.cpu().numpy().copy()


def _normalize(torch: Any, tensors: list[Any]) -> list[Any]:
    return [tensor / (torch.linalg.norm(tensor) + 1e-30) for tensor in tensors]


def _operator_on_device(torch: Any, operator: Any, tensor: Any) -> Any:
    if isinstance(operator, torch.Tensor):
        return operator.to(device=tensor.device)
    return operator


def _renormalize(torch: Any, _xp: Any, environment: Any) -> Any:
    from .ctmrg import CTMEnvironment

    def normalize(value: Any) -> Any:
        scale = torch.amax(torch.abs(value))
        return value / (scale + 1e-30)

    return CTMEnvironment(*(normalize(value) for value in environment.tensors()))


def _onsite_expectation(torch: Any, xp: Any, environment: Any, tensor: Any, term: Any) -> Any:
    from .ctmrg import _double_layer, _environment_contraction, _pauli_matrix

    if len(term.paulis) > 1:
        raise ValueError("autodiff CTMRG onsite terms must act on one unit-cell site")
    label = str(next(iter(term.paulis.values()), "I"))
    operator = _operator_on_device(torch, _pauli_matrix(xp, tensor.dtype, label), tensor)
    numerator = _environment_contraction(xp, environment, _double_layer(xp, tensor, operator))
    denominator = _environment_contraction(xp, environment, _double_layer(xp, tensor))
    return torch.real(numerator / (denominator + 1e-30))


def _interaction_expectation(
    torch: Any,
    xp: Any,
    environments: list[Any],
    tensors: list[Any],
    interaction: Any,
) -> Any:
    from .ctmrg import (
        _double_layer,
        _horizontal_two_site_contraction,
        _horizontal_two_site_contraction_pair,
        _pauli_matrix,
        _vertical_two_site_contraction,
        _vertical_two_site_contraction_pair,
    )

    dx, dy = (int(value) for value in interaction.displacement)
    if (abs(dx), abs(dy)) not in ((1, 0), (0, 1)):
        raise ValueError("autodiff CTMRG supports nearest-neighbor interactions only")
    left_site = int(interaction.left_site)
    right_site = int(interaction.right_site)
    left_tensor = tensors[left_site]
    right_tensor = tensors[right_site]
    left_operator = _operator_on_device(torch, _pauli_matrix(xp, left_tensor.dtype, str(interaction.left_pauli)), left_tensor)
    right_operator = _operator_on_device(torch, _pauli_matrix(xp, right_tensor.dtype, str(interaction.right_pauli)), right_tensor)
    left_layer = _double_layer(xp, left_tensor, left_operator)
    right_layer = _double_layer(xp, right_tensor, right_operator)
    left_identity = _double_layer(xp, left_tensor)
    right_identity = _double_layer(xp, right_tensor)
    if len(tensors) == 1:
        if abs(dx) == 1:
            numerator = _horizontal_two_site_contraction(xp, environments[0], left_layer, right_layer)
            denominator = _horizontal_two_site_contraction(xp, environments[0], left_identity, right_identity)
        else:
            numerator = _vertical_two_site_contraction(xp, environments[0], left_layer, right_layer)
            denominator = _vertical_two_site_contraction(xp, environments[0], left_identity, right_identity)
    else:
        if abs(dx) == 1:
            numerator = _horizontal_two_site_contraction_pair(
                xp, environments[left_site], environments[right_site], left_layer, right_layer
            )
            denominator = _horizontal_two_site_contraction_pair(
                xp, environments[left_site], environments[right_site], left_identity, right_identity
            )
        else:
            numerator = _vertical_two_site_contraction_pair(
                xp, environments[left_site], environments[right_site], left_layer, right_layer
            )
            denominator = _vertical_two_site_contraction_pair(
                xp, environments[left_site], environments[right_site], left_identity, right_identity
            )
    return torch.real(numerator / (denominator + 1e-30))


def differentiable_ctmrg_energy(torch: Any, payload: CTMRGPayload, tensors: list[Any]) -> tuple[Any, dict[str, Any]]:
    """Evaluate a bounded differentiable CTMRG objective as a torch scalar."""

    from .ctmrg import (
        _bottom_move,
        _double_layer,
        _environment_residual,
        _initialize_environment,
        _left_move,
        _right_move,
        _top_move,
        _unit_cell_sweep,
    )

    unit_cell = list(payload.unit_cell)
    if unit_cell not in ([1, 1], [2, 1], [1, 2], [2, 2]):
        raise ValueError("autodiff CTMRG supports unit cells no larger than 2x2")
    if int(payload.physical_bond_dim) != 2:
        raise ValueError("autodiff CTMRG currently supports physical_bond_dim=2 only")
    if len(tensors) != math.prod(unit_cell):
        raise ValueError("autodiff CTMRG tensor count does not match the unit cell")
    if any(len(term.paulis) > 1 for term in payload.terms):
        raise ValueError("autodiff CTMRG supports one-site terms only")
    layers = [_double_layer(torch, tensor) for tensor in tensors]
    chi = int(payload.environment_bond_dim)
    environments = [_initialize_environment(torch, layer, chi) for layer in layers]
    residual = math.inf
    completed = 0
    for iteration in range(1, int(payload.iterations) + 1):
        before = list(environments)
        if len(environments) == 1:
            env = environments[0]
            env, _ = _left_move(torch, env, layers[0], chi)
            env, _ = _right_move(torch, env, layers[0], chi)
            env, _ = _top_move(torch, env, layers[0], chi)
            env, _ = _bottom_move(torch, env, layers[0], chi)
            environments = [_renormalize(torch, torch, env)]
        else:
            environments, _ = _unit_cell_sweep(
                torch,
                environments,
                layers,
                chi,
                unit_cell,
                renormalize=_renormalize,
            )
        residual = max(
            _environment_residual(torch, old, new)
            for old, new in zip(before, environments)
        )
        completed = iteration
        if residual <= float(payload.tolerance):
            break

    onsite_values = [
        _onsite_expectation(torch, torch, environments[int(next(iter(term.paulis), 0))], tensors[int(next(iter(term.paulis), 0))], term)
        for term in payload.terms
    ]
    interaction_values = [
        _interaction_expectation(torch, torch, environments, tensors, interaction)
        for interaction in payload.interactions
    ]
    energy = torch.zeros((), dtype=tensors[0].real.dtype, device=tensors[0].device)
    for term, value in zip(payload.terms, onsite_values):
        energy = energy + float(term.coefficient) * value
    for interaction, value in zip(payload.interactions, interaction_values):
        energy = energy + float(interaction.coefficient) * value
    return energy, {
        "energy": float(energy.detach().cpu()),
        "residual": float(residual),
        "iterations": completed,
        "environment_bond_dim": chi,
        "gradient_backend": "torch-autograd-unrolled-ctmrg",
        "materializes_statevector": False,
    }


def run_autodiff_full_update(
    xp: Any,
    payload: CTMRGPayload,
    tensors: list[Any],
    run_ctmrg: Any,
) -> dict[str, Any]:
    """Run bounded torch-autograd updates through the CTMRG environment."""

    torch = _torch()
    if getattr(xp, "__name__", "") == "cupy" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required when the CTMRG request selects the GPU autograd backend")
    device = torch.device("cuda" if torch.cuda.is_available() and getattr(xp, "__name__", "") == "cupy" else "cpu")
    dtype = torch.complex64 if payload.dtype == "complex64" else torch.complex128
    working = _normalize(torch, [_to_torch(torch, xp, tensor, dtype=dtype, device=device) for tensor in tensors])
    parameter_count = sum(int(tensor.numel()) for tensor in working) * 2
    if parameter_count > int(payload.full_update_max_parameters):
        raise ValueError(
            f"full-update tensor parameter count {parameter_count} exceeds full_update_max_parameters={payload.full_update_max_parameters}"
        )
    max_evaluations = int(payload.full_update_max_evaluations)
    evaluations = 0

    def evaluate(candidate: list[Any], *, with_gradient: bool) -> tuple[Any, dict[str, Any]] | None:
        nonlocal evaluations
        if evaluations >= max_evaluations:
            return None
        values = [tensor if with_gradient else tensor.detach() for tensor in candidate]
        if with_gradient:
            values = [tensor.requires_grad_(True) for tensor in values]
        energy, diagnostics = differentiable_ctmrg_energy(torch, payload, values)
        evaluations += 1
        return energy, diagnostics

    initial = evaluate(working, with_gradient=True)
    if initial is None:
        raise ValueError("autodiff CTMRG evaluation budget must allow an initial objective")
    initial_energy = float(initial[0].detach().cpu())
    current_energy = initial_energy
    history: list[dict[str, Any]] = []
    budget_exhausted = False
    for iteration in range(1, int(payload.optimization_steps) + 1):
        evaluated = evaluate(working, with_gradient=True)
        if evaluated is None:
            budget_exhausted = True
            break
        energy_tensor, diagnostics = evaluated
        gradients = torch.autograd.grad(energy_tensor, working, allow_unused=False)
        gradient_norm = math.sqrt(sum(float(torch.sum(torch.abs(gradient) ** 2).detach().cpu()) for gradient in gradients))
        sweep_start = float(energy_tensor.detach().cpu())
        accepted = None
        accepted_energy = sweep_start
        for scale in (1.0, 0.5, 0.25, 0.125):
            if evaluations >= max_evaluations:
                budget_exhausted = True
                break
            candidate = _normalize(
                torch,
                [tensor.detach() - float(payload.full_update_step) * scale * gradient for tensor, gradient in zip(working, gradients)],
            )
            candidate_result = evaluate(candidate, with_gradient=False)
            if candidate_result is not None:
                candidate_energy = float(candidate_result[0].detach().cpu())
                if candidate_energy < accepted_energy - float(payload.optimization_tolerance):
                    accepted = candidate
                    accepted_energy = candidate_energy
                    break
        if accepted is not None:
            working = [tensor.detach().requires_grad_(True) for tensor in accepted]
            current_energy = accepted_energy
        else:
            current_energy = sweep_start
        improvement = sweep_start - current_energy
        history.append({
            "iteration": iteration,
            "energy": current_energy,
            "improvement": improvement,
            "gradient_norm": gradient_norm,
            "accepted_updates": 1 if accepted is not None else 0,
            "evaluations": evaluations,
            "environment_residual": diagnostics["residual"],
            "evaluation_budget_exhausted": budget_exhausted,
        })
        if budget_exhausted or improvement <= float(payload.optimization_tolerance) or gradient_norm <= float(payload.optimization_tolerance):
            break

    result_tensors = [_from_torch(torch, xp, tensor) for tensor in working]
    return {
        "tensors": result_tensors,
        "iterations": len(history),
        "energy_history": history,
        "initial_energy": initial_energy,
        "final_energy": current_energy,
        "evaluations": evaluations,
        "parameter_count": parameter_count,
        "optimizer": "autodiff-ctmrg-gradient",
        "gradient_backend": "torch-autograd-unrolled-ctmrg",
        "objective": "infinite-ctmrg-unrolled-environment",
        "environment_iterations": int(payload.iterations),
        "materializes_reference_statevector": False,
        "evaluation_budget": max_evaluations,
        "evaluation_budget_exhausted": budget_exhausted,
        "converged": bool(
            history
            and not budget_exhausted
            and (
                history[-1]["improvement"] <= float(payload.optimization_tolerance)
                or history[-1]["gradient_norm"] <= float(payload.optimization_tolerance)
            )
        ),
    }
