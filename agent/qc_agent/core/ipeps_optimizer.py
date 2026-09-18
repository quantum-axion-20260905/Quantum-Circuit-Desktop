"""Variational product-manifold optimizer for bounded iPEPS baselines.

This is intentionally separate from CTMRG contraction.  It optimizes only
physical two-level vectors with virtual bond dimension one, so its output is a
reproducible mean-field baseline rather than an entangled iPEPS ground state.
The seam can later host simple-update or full-update optimizers without
changing request/result or checkpoint contracts.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from ..plugins.models import CTMRGPayload
from .checkpoints import load_optimizer_checkpoint, save_optimizer_checkpoint
from .contracts import CheckpointManifest
from .ctmrg_objective import CTMRGObjective, normalize_tensors, optimizer_request_sha256


def _host(value: Any) -> Any:
    try:
        return value.get()
    except AttributeError:
        return value


def _pauli(xp: Any, dtype: Any, label: str) -> Any:
    if label == "I":
        return xp.eye(2, dtype=dtype)
    if label == "X":
        return xp.asarray([[0, 1], [1, 0]], dtype=dtype)
    if label == "Y":
        return xp.asarray([[0, -1j], [1j, 0]], dtype=dtype)
    if label == "Z":
        return xp.asarray([[1, 0], [0, -1]], dtype=dtype)
    raise ValueError(f"unsupported Pauli operator {label!r}")


def _normalize(xp: Any, state: Any) -> Any:
    norm = xp.linalg.norm(state)
    if float(_host(norm)) <= 1e-30:
        raise ValueError("product optimizer received a zero-norm local state")
    return state / norm


def _expectation(xp: Any, state: Any, label: str) -> float:
    value = xp.conj(state) @ _pauli(xp, state.dtype, label) @ state
    return float(complex(_host(value)).real)


def _energy(xp: Any, states: list[Any], payload: CTMRGPayload) -> float:
    energy = 0.0
    for term in payload.terms:
        if len(term.paulis) > 1:
            raise ValueError("product optimizer onsite terms must act on one unit-cell site")
        site = next(iter(term.paulis), 0)
        energy += float(term.coefficient) * _expectation(xp, states[site], next(iter(term.paulis.values()), "I"))
    for interaction in payload.interactions:
        energy += float(interaction.coefficient) * _expectation(
            xp, states[interaction.left_site], interaction.left_pauli
        ) * _expectation(xp, states[interaction.right_site], interaction.right_pauli)
    return float(energy)


def _effective_operator(xp: Any, states: list[Any], payload: CTMRGPayload, site: int) -> Any:
    dtype = states[site].dtype
    operator = xp.zeros((2, 2), dtype=dtype)
    for term in payload.terms:
        if len(term.paulis) > 1:
            raise ValueError("product optimizer onsite terms must act on one unit-cell site")
        term_site = next(iter(term.paulis), 0)
        if term_site == site:
            operator = operator + float(term.coefficient) * _pauli(
                xp, dtype, next(iter(term.paulis.values()), "I")
            )
    for interaction in payload.interactions:
        if interaction.left_site == site:
            other = _expectation(xp, states[interaction.right_site], interaction.right_pauli)
            operator = operator + float(interaction.coefficient) * other * _pauli(
                xp, dtype, interaction.left_pauli
            )
        if interaction.right_site == site:
            other = _expectation(xp, states[interaction.left_site], interaction.left_pauli)
            operator = operator + float(interaction.coefficient) * other * _pauli(
                xp, dtype, interaction.right_pauli
            )
    return 0.5 * (operator + xp.conj(operator).T)


def optimize_product_states(
    xp: Any,
    payload: CTMRGPayload,
    initial_states: list[Any],
) -> dict[str, Any]:
    """Coordinate-descent optimization over one normalized qubit per cell site."""

    if int(payload.virtual_bond_dim) != 1:
        raise ValueError("product-coordinate-descent requires virtual_bond_dim=1")
    states = [_normalize(xp, state) for state in initial_states]
    if len(states) != math.prod(payload.unit_cell):
        raise ValueError("product optimizer state count does not match the unit cell")
    history: list[dict[str, Any]] = []
    current_energy = _energy(xp, states, payload)
    initial_energy = current_energy
    for iteration in range(1, int(payload.optimization_steps) + 1):
        max_change = 0.0
        for site in range(len(states)):
            _, vectors = xp.linalg.eigh(_effective_operator(xp, states, payload, site))
            updated = _normalize(xp, vectors[:, 0])
            change = float(_host(xp.linalg.norm(updated - states[site])))
            max_change = max(max_change, change)
            states[site] = updated
        current_energy = _energy(xp, states, payload)
        history.append({
            "iteration": iteration,
            "energy": current_energy,
            "max_state_change": max_change,
        })
        if max_change <= float(payload.optimization_tolerance):
            break
    return {
        "states": states,
        "energy_history": history,
        "initial_energy": initial_energy,
        "final_energy": current_energy,
        "iterations": len(history),
        "converged": bool(history and history[-1]["max_state_change"] <= float(payload.optimization_tolerance)),
    }


def _imaginary_time_gate(xp: Any, dtype: Any, label_left: str, label_right: str, coefficient: float, dt: float) -> Any:
    left = _pauli(xp, dtype, label_left)
    right = _pauli(xp, dtype, label_right)
    operator = xp.kron(left, right)
    identity = xp.eye(4, dtype=dtype)
    return math.cosh(dt * coefficient) * identity - math.sinh(dt * coefficient) * operator


def _apply_one_site_gate(xp: Any, tensor: Any, label: str, coefficient: float, dt: float) -> Any:
    gate = math.cosh(dt * coefficient) * _pauli(xp, tensor.dtype, "I") - math.sinh(
        dt * coefficient
    ) * _pauli(xp, tensor.dtype, label)
    return xp.tensordot(gate, tensor, axes=(1, 0))


def _bond_axes(displacement: list[int]) -> tuple[int, int]:
    dx, dy = (int(value) for value in displacement)
    if (abs(dx), abs(dy)) == (1, 0):
        return (4, 3) if dx > 0 else (3, 4)
    if (abs(dx), abs(dy)) == (0, 1):
        return (2, 1) if dy > 0 else (1, 2)
    raise ValueError("simple-update supports only nearest-neighbor horizontal or vertical interactions")


def _apply_two_site_gate(
    xp: Any,
    tensors: list[Any],
    left_site: int,
    right_site: int,
    displacement: list[int],
    label_left: str,
    label_right: str,
    coefficient: float,
    dt: float,
    max_bond_dim: int,
) -> float:
    if left_site == right_site:
        raise ValueError("simple-update does not support a self-interaction bond")
    left_axis, right_axis = _bond_axes(displacement)
    left_tensor = tensors[left_site]
    right_tensor = tensors[right_site]
    left_remaining = [axis for axis in range(5) if axis != left_axis]
    right_remaining = [axis for axis in range(5) if axis != right_axis]
    left_external = [axis for axis in range(1, 5) if axis != left_axis]
    right_external = [axis for axis in range(1, 5) if axis != right_axis]
    combined = xp.tensordot(left_tensor, right_tensor, axes=(left_axis, right_axis))
    combined = combined.transpose(
        [left_remaining.index(0), len(left_remaining) + right_remaining.index(0)]
        + [left_remaining.index(axis) for axis in left_external]
        + [len(left_remaining) + right_remaining.index(axis) for axis in right_external]
    )
    external_shape = [left_tensor.shape[axis] for axis in left_external] + [
        right_tensor.shape[axis] for axis in right_external
    ]
    gate = _imaginary_time_gate(
        xp, left_tensor.dtype, label_left, label_right, float(coefficient), float(dt)
    )
    physical_size = int(left_tensor.shape[0] * right_tensor.shape[0])
    combined = (gate @ combined.reshape(physical_size, -1)).reshape(
        (left_tensor.shape[0], right_tensor.shape[0], *external_shape)
    )
    left_external_shape = [left_tensor.shape[axis] for axis in left_external]
    right_external_shape = [right_tensor.shape[axis] for axis in right_external]
    left_size = int(left_tensor.shape[0] * math.prod(left_external_shape or (1,)))
    right_size = int(right_tensor.shape[0] * math.prod(right_external_shape or (1,)))
    matrix_order = [0] + list(range(2, 2 + len(left_external))) + [1] + list(
        range(2 + len(left_external), combined.ndim)
    )
    matrix = combined.transpose(matrix_order).reshape(left_size, right_size)
    u, singular, vh = xp.linalg.svd(matrix, full_matrices=False)
    keep = max(1, min(int(max_bond_dim), int(singular.shape[0])))
    weights = xp.abs(singular) ** 2
    total_weight = float(_host(xp.sum(weights)))
    discarded_weight = float(_host(xp.sum(weights[keep:]))) / max(total_weight, 1e-30)
    sqrt_singular = xp.sqrt(singular[:keep])
    left_new = (u[:, :keep] * sqrt_singular).reshape((left_tensor.shape[0], *left_external_shape, keep))
    right_new = (sqrt_singular[:, None] * vh[:keep, :]).reshape((keep, right_tensor.shape[0], *right_external_shape))
    left_order = [0] + left_external + [left_axis]
    right_order = [right_axis, 0] + right_external
    tensors[left_site] = left_new.transpose([left_order.index(axis) for axis in range(5)])
    tensors[right_site] = right_new.transpose([right_order.index(axis) for axis in range(5)])
    return discarded_weight


def run_simple_update(xp: Any, payload: CTMRGPayload, tensors: list[Any]) -> dict[str, Any]:
    """Apply bounded imaginary-time simple-update gates to an iPEPS cell."""

    if int(payload.physical_bond_dim) != 2:
        raise ValueError("simple-update currently supports physical_bond_dim=2 only")
    if any(int(tensor.shape[1]) != int(payload.virtual_bond_dim) for tensor in tensors):
        raise ValueError("simple-update tensor virtual dimensions do not match the request")
    working = [tensor.copy() for tensor in tensors]
    discarded_history: list[float] = []
    bond_dim_history: list[int] = []
    for _ in range(int(payload.optimization_steps)):
        discarded_step = 0.0
        for term in payload.terms:
            if len(term.paulis) > 1:
                raise ValueError("simple-update onsite terms must act on one unit-cell site")
            site = next(iter(term.paulis), 0)
            label = next(iter(term.paulis.values()), "I")
            working[site] = _apply_one_site_gate(
                xp, working[site], label, float(term.coefficient), float(payload.optimization_dt)
            )
        for interaction in payload.interactions:
            discarded_step += _apply_two_site_gate(
                xp,
                working,
                int(interaction.left_site),
                int(interaction.right_site),
                list(interaction.displacement),
                interaction.left_pauli,
                interaction.right_pauli,
                float(interaction.coefficient),
                float(payload.optimization_dt),
                int(payload.virtual_bond_dim),
            )
        for index, tensor in enumerate(working):
            working[index] = tensor / (xp.linalg.norm(tensor) + 1e-30)
        discarded_history.append(float(discarded_step))
        bond_dim_history.append(max(int(tensor.shape[1]) for tensor in working))
        if discarded_step <= float(payload.optimization_tolerance):
            break
    return {
        "tensors": working,
        "iterations": len(discarded_history),
        "discarded_weight_history": discarded_history,
        "bond_dim_history": bond_dim_history,
        "converged": bool(discarded_history and discarded_history[-1] <= float(payload.optimization_tolerance)),
    }


def _run_finite_difference_full_update(
    xp: Any,
    payload: CTMRGPayload,
    tensors: list[Any],
    run_ctmrg: Any,
) -> dict[str, Any]:
    """Run a bounded CTMRG-feedback finite-difference gradient update.

    This is a useful bridge for environments without an automatic
    differentiation tensor runtime.  It estimates the real and imaginary
    tensor gradients from the same CTMRG objective used for the final result,
    then applies a short backtracking line search.  The evaluation budget and
    parameter cap are explicit because this remains an experimental dense
    fallback, not a scalable AD implementation.
    """

    parameter_count = sum(int(tensor.size) for tensor in tensors) * 2
    if parameter_count > int(payload.full_update_max_parameters):
        raise ValueError(
            f"full-update tensor parameter count {parameter_count} exceeds "
            f"full_update_max_parameters={payload.full_update_max_parameters}"
        )
    objective = CTMRGObjective(xp, payload, run_ctmrg)
    state = _restore_optimizer_state(
        xp,
        payload,
        tensors,
        objective,
        expected_method="ipeps-full-update-finite-difference-gradient-ctmrg",
        optimizer_name="finite-difference-gradient",
    )
    request_sha256 = state["request_sha256"]
    working = state["working"]
    max_evaluations = int(payload.full_update_max_evaluations)
    budget_exhausted = False
    start_iteration = state["start_iteration"]
    history: list[dict[str, Any]] = state["history"]
    checkpoint_info: dict[str, Any] = state["checkpoint"]
    initial_energy: float | None = state["initial_energy"]
    current_energy: float | None = state["current_energy"]

    def evaluate(candidate: list[Any]) -> float | None:
        result = objective.evaluate(candidate)
        return None if result is None else float(result["energy"])

    if current_energy is None:
        current_energy = evaluate(working)
        if current_energy is None:
            raise ValueError("full-update evaluation budget must allow an initial CTMRG evaluation")
        initial_energy = current_energy
    epsilon = float(payload.full_update_gradient_epsilon)

    for iteration in range(start_iteration + 1, int(payload.optimization_steps) + 1):
        if objective.evaluations + 2 * parameter_count > max_evaluations:
            budget_exhausted = True
            break
        gradients: list[tuple[int, int, complex]] = []
        gradient_squared_norm = 0.0
        for site in range(len(working)):
            for flat_index in range(int(working[site].size)):
                candidates: list[list[Any]] = []
                for delta in (epsilon, -epsilon, 1j * epsilon, -1j * epsilon):
                    candidate = [tensor.copy() for tensor in working]
                    flat = candidate[site].reshape(-1)
                    flat[flat_index] = flat[flat_index] + delta
                    candidates.append(candidate)
                real_plus = evaluate(candidates[0])
                real_minus = evaluate(candidates[1])
                imag_plus = evaluate(candidates[2])
                imag_minus = evaluate(candidates[3])
                if None in (real_plus, real_minus, imag_plus, imag_minus):
                    budget_exhausted = True
                    break
                real_gradient = (float(real_plus) - float(real_minus)) / (2.0 * epsilon)
                imag_gradient = (float(imag_plus) - float(imag_minus)) / (2.0 * epsilon)
                gradient = complex(real_gradient, imag_gradient)
                gradients.append((site, flat_index, gradient))
                gradient_squared_norm += real_gradient * real_gradient + imag_gradient * imag_gradient
            if budget_exhausted:
                break
        if budget_exhausted:
            history.append({
                "iteration": iteration,
                "energy": current_energy,
                "improvement": 0.0,
                "gradient_norm": math.sqrt(gradient_squared_norm),
                "accepted_updates": 0,
                "evaluations": objective.evaluations,
                "parameter_count": parameter_count,
                "evaluation_budget_exhausted": True,
            })
            break

        gradient_norm = math.sqrt(gradient_squared_norm)
        sweep_start = current_energy
        accepted_energy = current_energy
        accepted_candidate: list[Any] | None = None
        for scale in (1.0, 0.5, 0.25, 0.125):
            if objective.evaluations >= max_evaluations:
                budget_exhausted = True
                break
            candidate = [tensor.copy() for tensor in working]
            for site, flat_index, gradient in gradients:
                flat = candidate[site].reshape(-1)
                flat[flat_index] = flat[flat_index] - float(payload.full_update_step) * scale * gradient
            candidate_energy = evaluate(candidate)
            if candidate_energy is not None and candidate_energy < accepted_energy - float(payload.optimization_tolerance):
                accepted_energy = candidate_energy
                accepted_candidate = normalize_tensors(xp, candidate)
                break
        if accepted_candidate is not None:
            working = accepted_candidate
            current_energy = accepted_energy
        improvement = sweep_start - current_energy
        history.append({
            "iteration": iteration,
            "energy": current_energy,
            "improvement": improvement,
            "gradient_norm": gradient_norm,
            "accepted_updates": 1 if accepted_candidate is not None else 0,
            "evaluations": objective.evaluations,
            "parameter_count": parameter_count,
            "evaluation_budget_exhausted": budget_exhausted,
        })
        checkpoint_info = _save_optimizer_state(
            xp,
            payload,
            objective,
            working,
            request_sha256=request_sha256,
            method="ipeps-full-update-finite-difference-gradient-ctmrg",
            optimizer_name="finite-difference-gradient",
            iteration=iteration,
            history=history,
            initial_energy=float(initial_energy),
            current_energy=float(current_energy),
            parameter_count=parameter_count,
        ) or checkpoint_info
        if budget_exhausted or improvement <= float(payload.optimization_tolerance) or gradient_norm <= float(payload.optimization_tolerance):
            break

    return {
        "tensors": working,
        "iterations": len(history),
        "energy_history": history,
        "initial_energy": float(initial_energy),
        "final_energy": float(current_energy),
        "evaluations": objective.evaluations,
        "parameter_count": parameter_count,
        "optimizer": "finite-difference-gradient",
        "gradient_backend": "bounded-finite-difference",
        "evaluation_budget": max_evaluations,
        "evaluation_budget_exhausted": budget_exhausted,
        "start_iteration": start_iteration,
        "request_sha256": request_sha256,
        "checkpoint": checkpoint_info or {
            "resumable": False,
            "reason": "set optimizer_checkpoint_path to persist and optimizer_resume_from to resume the bounded finite-difference update",
        },
        "converged": bool(
            history
            and not budget_exhausted
            and (
                history[-1]["improvement"] <= float(payload.optimization_tolerance)
                or history[-1]["gradient_norm"] <= float(payload.optimization_tolerance)
            )
        ),
    }


def _rademacher(index: int, iteration: int, direction: int, component: int) -> float:
    """Return a deterministic +/-1 perturbation without global RNG state."""

    value = (
        (int(index) + 1) * 1103515245
        + (int(iteration) + 1) * 12345
        + (int(direction) + 1) * 2246822519
        + (int(component) + 1) * 2654435761
        + 17
    ) & 0xFFFFFFFF
    # Mix high bits before extracting a sign; using only the low bit of the
    # linear combination creates visible alternating patterns for small
    # tensors and defeats the intended simultaneous-perturbation averaging.
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    return 1.0 if value & 0x80000000 else -1.0


def _restore_optimizer_state(
    xp: Any,
    payload: CTMRGPayload,
    tensors: list[Any],
    objective: CTMRGObjective,
    *,
    expected_method: str,
    optimizer_name: str,
) -> dict[str, Any]:
    """Restore a bounded CTMRG-feedback optimizer state after strict checks."""

    request_sha256 = optimizer_request_sha256(payload)
    if not payload.optimizer_resume_from:
        return {
            "request_sha256": request_sha256,
            "working": normalize_tensors(xp, [tensor.copy() for tensor in tensors]),
            "start_iteration": 0,
            "history": [],
            "initial_energy": None,
            "current_energy": None,
            "checkpoint": {},
        }
    manifest, restored_tensors = load_optimizer_checkpoint(
        payload.optimizer_resume_from,
        xp,
        expected_method=expected_method,
    )
    if manifest.get("request_sha256") != request_sha256:
        raise ValueError("optimizer checkpoint does not match the scientific CTMRG problem")
    if manifest.get("dtype") != payload.dtype:
        raise ValueError("optimizer checkpoint dtype does not match the requested dtype")
    metadata = manifest.get("metadata", {})
    if metadata.get("optimizer") != optimizer_name:
        raise ValueError(f"optimizer checkpoint is not a {optimizer_name} state")
    if int(metadata.get("tensor_count", -1)) != len(tensors):
        raise ValueError("optimizer checkpoint tensor count does not match the request")
    if any(tuple(restored.shape) != tuple(current.shape) for restored, current in zip(restored_tensors, tensors)):
        raise ValueError("optimizer checkpoint tensor shapes do not match the request")
    start_iteration = int(manifest.get("step", 0))
    if start_iteration > int(payload.optimization_steps):
        raise ValueError(
            f"optimizer checkpoint already contains {start_iteration} iterations, "
            f"but the requested run only allows {payload.optimization_steps}"
        )
    raw_history = metadata.get("energy_history", [])
    if not isinstance(raw_history, list) or len(raw_history) != start_iteration:
        raise ValueError("optimizer checkpoint energy history is invalid")
    objective.evaluations = int(metadata.get("evaluations", 0))
    if objective.evaluations > int(payload.full_update_max_evaluations):
        raise ValueError("optimizer checkpoint evaluations exceed the requested evaluation budget")
    initial_energy = float(metadata["initial_energy"])
    current_energy = float(metadata.get("current_energy", initial_energy))
    return {
        "request_sha256": request_sha256,
        "working": normalize_tensors(xp, [tensor.copy() for tensor in restored_tensors]),
        "start_iteration": start_iteration,
        "history": [dict(point) for point in raw_history],
        "initial_energy": initial_energy,
        "current_energy": current_energy,
        "checkpoint": manifest,
    }


def _save_optimizer_state(
    xp: Any,
    payload: CTMRGPayload,
    objective: CTMRGObjective,
    working: list[Any],
    *,
    request_sha256: str,
    method: str,
    optimizer_name: str,
    iteration: int,
    history: list[dict[str, Any]],
    initial_energy: float,
    current_energy: float,
    parameter_count: int,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a bounded CTMRG optimizer state when checkpointing is enabled."""

    if not payload.optimizer_checkpoint_path:
        return {}
    metadata = {
        "optimizer": optimizer_name,
        "completed_iterations": int(iteration),
        "evaluations": int(objective.evaluations),
        "initial_energy": float(initial_energy),
        "current_energy": float(current_energy),
        "energy_history": [dict(point) for point in history],
        "tensor_count": len(working),
        "parameter_count": int(parameter_count),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return save_optimizer_checkpoint(
        payload.optimizer_checkpoint_path,
        working,
        CheckpointManifest(
            checkpoint_id=f"{optimizer_name}-ctmrg-{request_sha256[:12]}-iteration-{iteration}",
            request_sha256=request_sha256,
            method=method,
            representation="ipeps-optimizer-state",
            dtype=payload.dtype,
            device="cuda" if hasattr(xp, "cuda") else "cpu",
            step=int(iteration),
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        ),
    )


def _run_spsa_full_update(
    xp: Any,
    payload: CTMRGPayload,
    tensors: list[Any],
    run_ctmrg: Any,
) -> dict[str, Any]:
    """Run a deterministic, bounded simultaneous-perturbation update.

    SPSA estimates the full real/imaginary tensor gradient with two objective
    evaluations per iteration instead of four evaluations per parameter.  It
    is useful for scaling experiments and GPU studies, but remains an
    approximate stochastic-gradient baseline: it is not automatic
    differentiation and does not by itself establish variational convergence.
    """

    parameter_count = sum(int(tensor.size) for tensor in tensors) * 2
    if parameter_count > int(payload.full_update_max_parameters):
        raise ValueError(
            f"full-update tensor parameter count {parameter_count} exceeds "
            f"full_update_max_parameters={payload.full_update_max_parameters}"
        )
    objective = CTMRGObjective(xp, payload, run_ctmrg)
    state = _restore_optimizer_state(
        xp,
        payload,
        tensors,
        objective,
        expected_method="ipeps-full-update-spsa-ctmrg",
        optimizer_name="spsa-gradient",
    )
    request_sha256 = state["request_sha256"]
    working = state["working"]
    max_evaluations = int(payload.full_update_max_evaluations)
    budget_exhausted = False
    start_iteration = state["start_iteration"]
    history: list[dict[str, Any]] = state["history"]
    checkpoint_info: dict[str, Any] = state["checkpoint"]
    initial_energy: float | None = state["initial_energy"]
    current_energy: float | None = state["current_energy"]

    def evaluate(candidate: list[Any]) -> float | None:
        result = objective.evaluate(candidate)
        return None if result is None else float(result["energy"])

    if current_energy is None:
        current_energy = evaluate(working)
        if current_energy is None:
            raise ValueError("full-update evaluation budget must allow an initial CTMRG evaluation")
        initial_energy = current_energy
    epsilon = float(payload.full_update_gradient_epsilon)
    direction_count = int(payload.full_update_spsa_directions)

    for iteration in range(start_iteration + 1, int(payload.optimization_steps) + 1):
        if objective.evaluations + 2 * direction_count > max_evaluations:
            budget_exhausted = True
            break
        averaged_gradient: list[Any] = [xp.zeros_like(tensor) for tensor in working]
        gradient_scales: list[float] = []
        direction_index = 0
        for direction_number in range(direction_count):
            direction: list[Any] = []
            for tensor in working:
                delta = xp.empty_like(tensor)
                flat = delta.reshape(-1)
                for flat_index in range(int(tensor.size)):
                    flat[flat_index] = complex(
                        _rademacher(direction_index, iteration, direction_number, 0),
                        _rademacher(direction_index, iteration, direction_number, 1),
                    )
                    direction_index += 1
                direction.append(delta)
            plus = evaluate([tensor + epsilon * delta for tensor, delta in zip(working, direction)])
            minus = evaluate([tensor - epsilon * delta for tensor, delta in zip(working, direction)])
            if plus is None or minus is None:
                budget_exhausted = True
                break
            gradient_scale = (float(plus) - float(minus)) / (2.0 * epsilon)
            gradient_scales.append(gradient_scale)
            for index, delta in enumerate(direction):
                averaged_gradient[index] = averaged_gradient[index] + gradient_scale * delta
        if budget_exhausted:
            break
        averaged_gradient = [gradient / float(direction_count) for gradient in averaged_gradient]
        gradient_norm = math.sqrt(sum(float(_host(xp.sum(xp.abs(gradient) ** 2))) for gradient in averaged_gradient))
        sweep_start = current_energy
        accepted_energy = current_energy
        accepted_candidate: list[Any] | None = None
        for scale in (1.0, 0.5, 0.25, 0.125):
            if objective.evaluations >= max_evaluations:
                budget_exhausted = True
                break
            candidate = [
                tensor - float(payload.full_update_step) * scale * gradient
                for tensor, gradient in zip(working, averaged_gradient)
            ]
            candidate_energy = evaluate(candidate)
            if candidate_energy is not None and candidate_energy < accepted_energy - float(payload.optimization_tolerance):
                accepted_energy = candidate_energy
                accepted_candidate = normalize_tensors(xp, candidate)
                break
        if accepted_candidate is not None:
            working = accepted_candidate
            current_energy = accepted_energy
        improvement = sweep_start - current_energy
        history.append({
            "iteration": iteration,
            "energy": current_energy,
            "improvement": improvement,
            "gradient_norm": gradient_norm,
            "gradient_scales": gradient_scales,
            "direction_count": direction_count,
            "accepted_updates": 1 if accepted_candidate is not None else 0,
            "evaluations": objective.evaluations,
            "parameter_count": parameter_count,
            "evaluation_budget_exhausted": budget_exhausted,
        })
        checkpoint_info = _save_optimizer_state(
            xp,
            payload,
            objective,
            working,
            request_sha256=request_sha256,
            method="ipeps-full-update-spsa-ctmrg",
            optimizer_name="spsa-gradient",
            iteration=iteration,
            history=history,
            initial_energy=float(initial_energy),
            current_energy=float(current_energy),
            parameter_count=parameter_count,
            extra_metadata={"direction_count": direction_count},
        ) or checkpoint_info
        if budget_exhausted or improvement <= float(payload.optimization_tolerance) or gradient_norm <= float(payload.optimization_tolerance):
            break

    return {
        "tensors": working,
        "iterations": len(history),
        "energy_history": history,
        "initial_energy": float(initial_energy),
        "final_energy": float(current_energy),
        "evaluations": objective.evaluations,
        "parameter_count": parameter_count,
        "optimizer": "spsa-gradient",
        "gradient_backend": "deterministic-simultaneous-perturbation",
        "direction_count": direction_count,
        "evaluation_budget": max_evaluations,
        "evaluation_budget_exhausted": budget_exhausted,
        "start_iteration": start_iteration,
        "request_sha256": request_sha256,
        "checkpoint": checkpoint_info or {
            "resumable": False,
            "reason": "set optimizer_checkpoint_path to persist and optimizer_resume_from to resume the bounded SPSA update",
        },
        "converged": bool(
            history
            and not budget_exhausted
            and (
                history[-1]["improvement"] <= float(payload.optimization_tolerance)
                or history[-1]["gradient_norm"] <= float(payload.optimization_tolerance)
            )
        ),
    }


def run_full_update(xp: Any, payload: CTMRGPayload, tensors: list[Any]) -> dict[str, Any]:
    """Dispatch explicit bounded full-update/reference optimization paths.

    CTMRG-feedback coordinate/finite-difference/SPSA paths use the same
    infinite-system contraction as the final result.  The finite-torus
    analytic-gradient path is intentionally dispatched to a separate module:
    it is an exact four-site reference objective, not an infinite-system
    gradient claim.
    """

    from .ctmrg import run_ctmrg  # lazy import avoids the optimizer/core cycle

    if payload.full_update_optimizer == "finite-torus-gradient":
        from .finite_peps_optimizer import run_finite_torus_gradient

        return run_finite_torus_gradient(xp, payload, tensors)
    if payload.full_update_optimizer == "autodiff-ctmrg-gradient":
        from .ctmrg_autodiff import run_autodiff_full_update

        return run_autodiff_full_update(xp, payload, tensors, run_ctmrg)
    if payload.full_update_optimizer == "implicit-ctmrg-gradient":
        from .ctmrg_autodiff import run_implicit_full_update

        return run_implicit_full_update(xp, payload, tensors, run_ctmrg)
    if payload.full_update_optimizer == "finite-difference-gradient":
        return _run_finite_difference_full_update(xp, payload, tensors, run_ctmrg)
    if payload.full_update_optimizer == "spsa-gradient":
        return _run_spsa_full_update(xp, payload, tensors, run_ctmrg)

    parameter_count = sum(int(tensor.size) for tensor in tensors) * 2
    if parameter_count > int(payload.full_update_max_parameters):
        raise ValueError(
            f"full-update tensor parameter count {parameter_count} exceeds "
            f"full_update_max_parameters={payload.full_update_max_parameters}"
        )
    objective = CTMRGObjective(xp, payload, run_ctmrg)
    state = _restore_optimizer_state(
        xp,
        payload,
        tensors,
        objective,
        expected_method="ipeps-full-update-coordinate-ctmrg",
        optimizer_name="coordinate",
    )
    request_sha256 = state["request_sha256"]
    working = state["working"]
    max_evaluations = int(payload.full_update_max_evaluations)
    start_iteration = state["start_iteration"]
    history: list[dict[str, Any]] = state["history"]
    checkpoint_info: dict[str, Any] = state["checkpoint"]
    initial_energy: float | None = state["initial_energy"]
    current_energy: float | None = state["current_energy"]

    def evaluate(candidate: list[Any]) -> float:
        result = objective.evaluate(candidate)
        if result is None:
            raise ValueError("full-update evaluation budget exhausted")
        return float(result["energy"])

    if current_energy is None:
        current_energy = evaluate(working)
        initial_energy = current_energy
    component_count = parameter_count
    for iteration in range(start_iteration + 1, int(payload.optimization_steps) + 1):
        sweep_start = current_energy
        accepted_updates = 0
        step = float(payload.full_update_step)
        for site in range(len(working)):
            for flat_index in range(int(working[site].size)):
                for component in (0, 1):
                    candidates: list[tuple[float, list[Any]]] = []
                    for sign in (1.0, -1.0):
                        candidate = [tensor.copy() for tensor in working]
                        flat = candidate[site].reshape(-1)
                        delta = step * sign if component == 0 else 1j * step * sign
                        flat[flat_index] = flat[flat_index] + delta
                        candidate = normalize_tensors(xp, candidate)
                        candidates.append((evaluate(candidate), candidate))
                    best_energy, best_candidate = min(candidates, key=lambda item: item[0])
                    if best_energy < current_energy - float(payload.optimization_tolerance):
                        working = best_candidate
                        current_energy = best_energy
                        accepted_updates += 1
        improvement = sweep_start - current_energy
        history.append({
            "iteration": iteration,
            "energy": current_energy,
            "improvement": improvement,
            "accepted_updates": accepted_updates,
            "evaluations": objective.evaluations,
            "parameter_count": component_count,
        })
        checkpoint_info = _save_optimizer_state(
            xp,
            payload,
            objective,
            working,
            request_sha256=request_sha256,
            method="ipeps-full-update-coordinate-ctmrg",
            optimizer_name="coordinate",
            iteration=iteration,
            history=history,
            initial_energy=float(initial_energy),
            current_energy=float(current_energy),
            parameter_count=parameter_count,
        ) or checkpoint_info
        if improvement <= float(payload.optimization_tolerance):
            break
    return {
        "tensors": working,
        "iterations": len(history),
        "energy_history": history,
        "initial_energy": float(initial_energy),
        "final_energy": float(current_energy),
        "evaluations": objective.evaluations,
        "parameter_count": component_count,
        "optimizer": "coordinate",
        "gradient_backend": None,
        "evaluation_budget": max_evaluations,
        "evaluation_budget_exhausted": objective.evaluations >= max_evaluations,
        "start_iteration": start_iteration,
        "request_sha256": request_sha256,
        "checkpoint": checkpoint_info or {
            "resumable": False,
            "reason": "set optimizer_checkpoint_path to persist and optimizer_resume_from to resume the bounded coordinate update",
        },
        "converged": bool(history and history[-1]["improvement"] <= float(payload.optimization_tolerance)),
    }
