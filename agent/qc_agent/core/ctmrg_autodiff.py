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


def _ctmrg_sweep(
    torch: Any,
    payload: CTMRGPayload,
    layers: list[Any],
    environments: list[Any],
    *,
    differentiate_truncation: bool = True,
) -> list[Any]:
    """Apply exactly one differentiable CTMRG environment sweep."""

    from .ctmrg import (
        _bottom_move,
        _left_move,
        _right_move,
        _top_move,
        _unit_cell_sweep,
    )

    chi = int(payload.environment_bond_dim)
    if len(environments) == 1:
        env = environments[0]
        env, _ = _left_move(torch, env, layers[0], chi, differentiate_truncation)
        env, _ = _right_move(torch, env, layers[0], chi, differentiate_truncation)
        env, _ = _top_move(torch, env, layers[0], chi, differentiate_truncation)
        env, _ = _bottom_move(torch, env, layers[0], chi, differentiate_truncation)
        return [_renormalize(torch, torch, env)]
    environments, _ = _unit_cell_sweep(
        torch,
        environments,
        layers,
        chi,
        list(payload.unit_cell),
        renormalize=lambda _xp, env: _renormalize(torch, torch, env),
        differentiate_truncation=differentiate_truncation,
    )
    return environments


def _environment_energy(torch: Any, payload: CTMRGPayload, environments: list[Any], tensors: list[Any]) -> Any:
    """Contract observables against an already supplied environment."""

    onsite_values = [
        _onsite_expectation(
            torch,
            torch,
            environments[int(next(iter(term.paulis), 0))],
            tensors[int(next(iter(term.paulis), 0))],
            term,
        )
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
    return energy


def _pack_environment(torch: Any, environments: list[Any]) -> Any:
    """Pack complex environment tensors into a real differentiable vector."""

    return torch.cat([
        torch.view_as_real(value).reshape(-1)
        for environment in environments
        for value in environment.tensors()
    ])


def _unpack_environment(torch: Any, vector: Any, template: list[Any]) -> list[Any]:
    """Rebuild CTM environments from the real vector used by the adjoint."""

    from .ctmrg import CTMEnvironment

    values: list[Any] = []
    offset = 0
    for environment in template:
        tensors: list[Any] = []
        for value in environment.tensors():
            count = int(value.numel())
            real_view = vector[offset:offset + 2 * count].reshape(tuple(value.shape) + (2,))
            tensors.append(torch.view_as_complex(real_view.contiguous()))
            offset += 2 * count
        values.append(CTMEnvironment(*tensors))
    if offset != int(vector.numel()):
        raise ValueError("implicit CTMRG environment vector shape mismatch")
    return values


def _transfer_gap(torch: Any, environments: list[Any]) -> float:
    """Return the smallest resolved normalized transfer-spectrum gap."""

    gaps: list[float] = []
    for environment in environments:
        transfer = torch.sum(environment.T1, dim=1)
        magnitudes = torch.sort(torch.abs(torch.linalg.eigvals(transfer)), descending=True).values
        if int(magnitudes.numel()) < 2:
            gaps.append(1.0)
            continue
        leading = float(magnitudes[0].detach().cpu())
        subleading = float(magnitudes[1].detach().cpu())
        gaps.append(max(0.0, 1.0 - subleading / max(leading, 1e-30)))
    return min(gaps) if gaps else 0.0


def _fixed_point_environments(torch: Any, payload: CTMRGPayload, layers: list[Any]) -> tuple[list[Any], float, int]:
    """Find a bounded CTMRG fixed point without retaining the forward graph."""

    from .ctmrg import _environment_residual, _initialize_environment

    environments = [_initialize_environment(torch, layer, int(payload.environment_bond_dim)) for layer in layers]
    residual = math.inf
    completed = 0
    with torch.no_grad():
        for iteration in range(1, int(payload.iterations) + 1):
            before = list(environments)
            environments = _ctmrg_sweep(torch, payload, layers, environments, differentiate_truncation=False)
            residual = max(
                _environment_residual(torch, old, new)
                for old, new in zip(before, environments)
            )
            completed = iteration
            if residual <= float(payload.tolerance):
                break
    return [
        type(environment)(*(value.detach() for value in environment.tensors()))
        for environment in environments
    ], float(residual), completed


def implicit_ctmrg_energy_and_gradient(
    torch: Any,
    payload: CTMRGPayload,
    tensors: list[Any],
) -> tuple[Any, list[Any], dict[str, Any]]:
    """Evaluate energy and an adjoint gradient at a bounded CTMRG fixed point.

    The environment fixed point is treated as ``e = F(e, A)``.  The adjoint
    solves ``(I - J_F(e)^T) lambda = dE/de`` by a bounded Neumann iteration,
    then combines the direct tensor derivative with ``lambda^T dF/dA``.  The
    method is intentionally bounded and reports its gap/residual diagnostics;
    it is not admitted as a production thermodynamic-limit proof until those
    diagnostics are independently validated on entangled reference families.
    """

    unit_cell = list(payload.unit_cell)
    if unit_cell not in ([1, 1], [2, 1], [1, 2], [2, 2]):
        raise ValueError("implicit CTMRG supports unit cells no larger than 2x2")
    if int(payload.physical_bond_dim) != 2:
        raise ValueError("implicit CTMRG currently supports physical_bond_dim=2 only")
    if len(tensors) != math.prod(unit_cell):
        raise ValueError("implicit CTMRG tensor count does not match the unit cell")
    if any(len(term.paulis) > 1 for term in payload.terms):
        raise ValueError("implicit CTMRG supports one-site terms only")

    from .ctmrg import _double_layer

    layers = [_double_layer(torch, tensor) for tensor in tensors]
    fixed_environments, fixed_residual, fixed_iterations = _fixed_point_environments(torch, payload, layers)
    environment_vector = _pack_environment(torch, fixed_environments).detach().requires_grad_(True)
    graph_environments = _unpack_environment(torch, environment_vector, fixed_environments)
    energy = _environment_energy(torch, payload, graph_environments, tensors)
    mapped_environments = _ctmrg_sweep(
        torch,
        payload,
        layers,
        graph_environments,
        differentiate_truncation=False,
    )
    mapped_vector = _pack_environment(torch, mapped_environments)

    rhs = torch.autograd.grad(
        energy,
        environment_vector,
        retain_graph=True,
        create_graph=True,
        allow_unused=True,
    )[0]
    if rhs is None:
        rhs = torch.zeros_like(environment_vector)
    lambda_vector = torch.zeros_like(rhs)
    max_adjoint_iterations = int(getattr(payload, "full_update_implicit_iterations", 32))
    adjoint_tolerance = float(getattr(payload, "full_update_implicit_tolerance", 1e-6))
    damping = float(getattr(payload, "full_update_implicit_damping", 0.5))
    adjoint_iterations = 0
    adjoint_delta = math.inf
    for adjoint_iteration in range(1, max_adjoint_iterations + 1):
        jacobian_transpose = torch.autograd.grad(
            mapped_vector,
            environment_vector,
            grad_outputs=lambda_vector,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if jacobian_transpose is None:
            jacobian_transpose = torch.zeros_like(lambda_vector)
        proposed = rhs + jacobian_transpose
        updated = damping * proposed + (1.0 - damping) * lambda_vector
        adjoint_delta = float(torch.linalg.norm(updated - lambda_vector).detach().cpu())
        lambda_vector = updated
        adjoint_iterations = adjoint_iteration
        if adjoint_delta <= adjoint_tolerance:
            break

    jacobian_transpose = torch.autograd.grad(
        mapped_vector,
        environment_vector,
        grad_outputs=lambda_vector,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if jacobian_transpose is None:
        jacobian_transpose = torch.zeros_like(lambda_vector)
    adjoint_residual = float(torch.linalg.norm(lambda_vector - jacobian_transpose - rhs).detach().cpu())
    direct_gradients = torch.autograd.grad(
        energy,
        tensors,
        retain_graph=True,
        allow_unused=True,
    )
    fixed_point_gradients = torch.autograd.grad(
        mapped_vector,
        tensors,
        grad_outputs=lambda_vector,
        retain_graph=False,
        allow_unused=True,
    )
    gradients = [
        (direct if direct is not None else torch.zeros_like(tensor))
        + (fixed_point if fixed_point is not None else torch.zeros_like(tensor))
        for tensor, direct, fixed_point in zip(tensors, direct_gradients, fixed_point_gradients)
    ]
    return energy, gradients, {
        "energy": float(energy.detach().cpu()),
        "fixed_point_residual": fixed_residual,
        "fixed_point_iterations": fixed_iterations,
        "transfer_gap": _transfer_gap(torch, fixed_environments),
        "adjoint_iterations": adjoint_iterations,
        "adjoint_delta": adjoint_delta,
        "adjoint_residual": adjoint_residual,
        "gradient_backend": "torch-autograd-implicit-fixed-point",
        "truncation_gradient": "frozen-eigenprojector",
        "materializes_statevector": False,
    }


def differentiable_ctmrg_energy(torch: Any, payload: CTMRGPayload, tensors: list[Any]) -> tuple[Any, dict[str, Any]]:
    """Evaluate a bounded differentiable CTMRG objective as a torch scalar."""

    from .ctmrg import _double_layer, _environment_residual, _initialize_environment

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
        environments = _ctmrg_sweep(
            torch,
            payload,
            layers,
            environments,
            differentiate_truncation=False,
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
        "truncation_gradient": "frozen-eigenprojector",
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
        "truncation_gradient": "frozen-eigenprojector",
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


def run_implicit_full_update(
    xp: Any,
    payload: CTMRGPayload,
    tensors: list[Any],
    _run_ctmrg: Any,
) -> dict[str, Any]:
    """Run bounded full-update steps using the CTMRG fixed-point adjoint."""

    torch = _torch()
    if getattr(xp, "__name__", "") == "cupy" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required when the CTMRG request selects the GPU implicit backend")
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

    def evaluate(candidate: list[Any]) -> tuple[Any, list[Any], dict[str, Any]] | None:
        nonlocal evaluations
        if evaluations >= max_evaluations:
            return None
        values = [tensor.detach().requires_grad_(True) for tensor in candidate]
        result = implicit_ctmrg_energy_and_gradient(torch, payload, values)
        evaluations += 1
        return result

    initial = evaluate(working)
    if initial is None:
        raise ValueError("implicit CTMRG evaluation budget must allow an initial objective")
    initial_energy = float(initial[0].detach().cpu())
    current_energy = initial_energy
    history: list[dict[str, Any]] = []
    budget_exhausted = False
    last_diagnostics = initial[2]
    for iteration in range(1, int(payload.optimization_steps) + 1):
        evaluated = evaluate(working)
        if evaluated is None:
            budget_exhausted = True
            break
        energy_tensor, gradients, diagnostics = evaluated
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
                [
                    tensor.detach() - float(payload.full_update_step) * scale * gradient
                    for tensor, gradient in zip(working, gradients)
                ],
            )
            candidate_result = evaluate(candidate)
            if candidate_result is not None:
                candidate_energy = float(candidate_result[0].detach().cpu())
                if candidate_energy < accepted_energy - float(payload.optimization_tolerance):
                    accepted = candidate
                    accepted_energy = candidate_energy
                    last_diagnostics = candidate_result[2]
                    break
        if accepted is not None:
            working = [tensor.detach() for tensor in accepted]
            current_energy = accepted_energy
        else:
            current_energy = sweep_start
            last_diagnostics = diagnostics
        improvement = sweep_start - current_energy
        history.append({
            "iteration": iteration,
            "energy": current_energy,
            "improvement": improvement,
            "gradient_norm": gradient_norm,
            "accepted_updates": 1 if accepted is not None else 0,
            "evaluations": evaluations,
            "fixed_point_residual": last_diagnostics["fixed_point_residual"],
            "transfer_gap": last_diagnostics["transfer_gap"],
            "adjoint_residual": last_diagnostics["adjoint_residual"],
            "adjoint_iterations": last_diagnostics["adjoint_iterations"],
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
        "optimizer": "implicit-ctmrg-gradient",
        "gradient_backend": "torch-autograd-implicit-fixed-point",
        "truncation_gradient": "frozen-eigenprojector",
        "objective": "infinite-ctmrg-implicit-fixed-point-environment",
        "environment_iterations": int(payload.iterations),
        "materializes_reference_statevector": False,
        "evaluation_budget": max_evaluations,
        "evaluation_budget_exhausted": budget_exhausted,
        "fixed_point_residual": last_diagnostics["fixed_point_residual"],
        "transfer_gap": last_diagnostics["transfer_gap"],
        "adjoint_residual": last_diagnostics["adjoint_residual"],
        "adjoint_iterations": last_diagnostics["adjoint_iterations"],
        "converged": bool(
            history
            and not budget_exhausted
            and last_diagnostics["adjoint_residual"] <= float(getattr(payload, "full_update_implicit_tolerance", 1e-6))
            and (
                history[-1]["improvement"] <= float(payload.optimization_tolerance)
                or history[-1]["gradient_norm"] <= float(payload.optimization_tolerance)
            )
        ),
    }
