"""Analytic finite-torus variational reference for very small iPEPS cells.

This module is intentionally separate from CTMRG.  It contracts an exact
2x2 periodic torus, materializes only the explicitly bounded four-site
reference vector, and differentiates the Rayleigh quotient by reverse
contraction of the tensor network.  It is a research reference/initializer,
not an infinite-lattice full-update solver.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

from ..plugins.models import CTMRGPayload


def _host(value: Any) -> Any:
    try:
        return value.get()
    except AttributeError:
        return value


def _real(value: Any) -> float:
    host = _host(value)
    try:
        host = host.item()
    except AttributeError:
        pass
    return float(complex(host).real)


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


def _cell_site_index(x: int, y: int, unit_cell: list[int]) -> int:
    width, height = (int(value) for value in unit_cell)
    return (int(x) % width) + width * (int(y) % height)


def _torus_site_index(x: int, y: int) -> int:
    return (int(x) % 2) + 2 * (int(y) % 2)


def _site_subscripts(x: int, y: int) -> str:
    horizontal = {(column, row): "abcdefgh"[row * 2 + column] for row in range(2) for column in range(2)}
    vertical = {
        (column, row): "abcdefgh"[4 + row * 2 + column]
        for row in range(2)
        for column in range(2)
    }
    return "".join((
        vertical[(x, (y - 1) % 2)],
        vertical[(x, y)],
        horizontal[((x - 1) % 2, y)],
        horizontal[(x, y)],
    ))


def _operator(xp: Any, dtype: Any, site_paulis: dict[int, str]) -> Any:
    value = xp.asarray([[1.0 + 0.0j]], dtype=dtype)
    for site in range(4):
        value = xp.kron(value, _pauli(xp, dtype, site_paulis.get(site, "I")))
    return value


def _hamiltonian(xp: Any, payload: CTMRGPayload, dtype: Any) -> Any:
    """Build the bounded four-site reference Hamiltonian for the cell pattern."""

    hamiltonian = xp.zeros((16, 16), dtype=dtype)
    width, height = (int(value) for value in payload.unit_cell)
    for term in payload.terms:
        local_site = int(next(iter(term.paulis), 0))
        local_x, local_y = local_site % width, local_site // width
        label = str(next(iter(term.paulis.values()), "I"))
        for row in range(2):
            for column in range(2):
                hamiltonian = hamiltonian + float(term.coefficient) * _operator(
                    xp,
                    dtype,
                    {_torus_site_index(column + local_x, row + local_y): label},
                )
    for interaction in payload.interactions:
        local_left_x = int(interaction.left_site) % width
        local_left_y = int(interaction.left_site) // width
        dx, dy = (int(value) for value in interaction.displacement)
        for row in range(2):
            for column in range(2):
                left = _torus_site_index(column + local_left_x, row + local_left_y)
                right = _torus_site_index(column + local_left_x + dx, row + local_left_y + dy)
                hamiltonian = hamiltonian + float(interaction.coefficient) * _operator(
                    xp,
                    dtype,
                    {
                        left: str(interaction.left_pauli),
                        right: str(interaction.right_pauli),
                    },
                )
    return hamiltonian


def _wavefunction_and_jacobian(xp: Any, payload: CTMRGPayload, tensors: list[Any]) -> tuple[Any, list[Any]]:
    """Contract the four-site torus and its tensor-element Jacobians."""

    virtual = int(payload.virtual_bond_dim)
    dtype = tensors[0].dtype
    state_count = 16
    parameter_size = int(tensors[0].size)
    psi = xp.zeros((state_count,), dtype=dtype)
    jacobians = [xp.zeros((state_count, int(tensor.size)), dtype=dtype) for tensor in tensors]
    physical_configs = list(itertools.product(range(2), repeat=4))

    for state_index, physical in enumerate(physical_configs):
        operands: list[Any] = []
        subscripts: list[str] = []
        site_tensor_indices: list[int] = []
        for y in range(2):
            for x in range(2):
                cell_index = _cell_site_index(x, y, payload.unit_cell)
                site_tensor_indices.append(cell_index)
                operands.append(tensors[cell_index][physical[_torus_site_index(x, y)]])
                subscripts.append(_site_subscripts(x, y))
        psi[state_index] = xp.einsum(
            ",".join(subscripts) + "->",
            *operands,
        )

        for occurrence in range(4):
            x, y = occurrence % 2, occurrence // 2
            cell_index = site_tensor_indices[occurrence]
            target_subscript = subscripts[occurrence]
            other_operands = [value for index, value in enumerate(operands) if index != occurrence]
            other_subscripts = [value for index, value in enumerate(subscripts) if index != occurrence]
            environment = xp.einsum(
                ",".join(other_subscripts) + "->" + target_subscript,
                *other_operands,
            )
            physical_index = physical[_torus_site_index(x, y)]
            start = physical_index * virtual ** 4
            jacobians[cell_index][state_index, start:start + virtual ** 4] += environment.reshape(-1)
    return psi, jacobians


def finite_torus_energy_gradient(
    xp: Any,
    payload: CTMRGPayload,
    tensors: list[Any],
    *,
    with_gradient: bool = True,
) -> dict[str, Any]:
    """Return exact finite-torus energy, variance, and analytic real gradient."""

    if int(payload.physical_bond_dim) != 2:
        raise ValueError("finite-torus-gradient requires physical_bond_dim=2")
    if int(payload.virtual_bond_dim) > 2:
        raise ValueError("finite-torus-gradient is limited to virtual_bond_dim<=2")
    if len(tensors) != math.prod(payload.unit_cell):
        raise ValueError("finite-torus-gradient tensor count does not match the unit cell")
    cell_width, cell_height = (int(value) for value in payload.unit_cell)
    cell_sites = cell_width * cell_height
    if any(len(term.paulis) > 1 for term in payload.terms):
        raise ValueError("finite-torus-gradient supports one-site terms only")
    if any(
        tuple(map(abs, interaction.displacement)) not in ((1, 0), (0, 1))
        for interaction in payload.interactions
    ):
        raise ValueError("finite-torus-gradient supports nearest-neighbor interactions only")
    for interaction in payload.interactions:
        left_x = int(interaction.left_site) % cell_width
        left_y = int(interaction.left_site) // cell_width
        dx, dy = (int(value) for value in interaction.displacement)
        expected_right = _cell_site_index(left_x + dx, left_y + dy, payload.unit_cell)
        if int(interaction.right_site) != expected_right:
            raise ValueError("finite-torus-gradient requires right_site to match left_site plus displacement")
        if int(interaction.left_site) >= cell_sites or int(interaction.right_site) >= cell_sites:
            raise ValueError("finite-torus-gradient interaction site exceeds the unit cell")
    psi, jacobians = _wavefunction_and_jacobian(xp, payload, tensors)
    hamiltonian = _hamiltonian(xp, payload, tensors[0].dtype)
    norm = _real(xp.vdot(psi, psi))
    if not math.isfinite(norm) or norm <= 1e-30:
        raise ValueError("finite-torus-gradient produced a zero or non-finite reference state")
    hpsi = hamiltonian @ psi
    energy = _real(xp.vdot(psi, hpsi)) / norm
    second_moment = _real(xp.vdot(hpsi, hpsi)) / norm
    variance = max(0.0, second_moment - energy * energy)
    result: dict[str, Any] = {
        "energy": energy,
        "energy_second_moment": second_moment,
        "energy_variance": variance,
        "norm": norm,
        "materializes_reference_statevector": True,
        "reference_sites": 4,
    }
    if not with_gradient:
        return result
    residual = hpsi - energy * psi
    gradients: list[Any] = []
    gradient_norm_squared = 0.0
    for jacobian in jacobians:
        inner = xp.conj(jacobian).T @ residual
        real_gradient = 2.0 * xp.real(inner) / norm
        imaginary_gradient = 2.0 * xp.imag(inner) / norm
        gradient = real_gradient + 1j * imaginary_gradient
        gradients.append(gradient.reshape(-1))
        gradient_norm_squared += _real(xp.sum(xp.abs(gradient) ** 2))
    result["gradients"] = gradients
    result["gradient_norm"] = math.sqrt(max(0.0, gradient_norm_squared))
    result["gradient_backend"] = "analytic-finite-torus-reverse-contraction"
    return result


def _normalize_tensors(xp: Any, tensors: list[Any]) -> list[Any]:
    return [tensor / (xp.linalg.norm(tensor) + 1e-30) for tensor in tensors]


def run_finite_torus_gradient(
    xp: Any,
    payload: CTMRGPayload,
    tensors: list[Any],
) -> dict[str, Any]:
    """Optimize the exact finite reference objective with bounded line search."""

    parameter_count = sum(int(tensor.size) for tensor in tensors) * 2
    if parameter_count > int(payload.full_update_max_parameters):
        raise ValueError(
            f"full-update tensor parameter count {parameter_count} exceeds "
            f"full_update_max_parameters={payload.full_update_max_parameters}"
        )
    working = _normalize_tensors(xp, [tensor.copy() for tensor in tensors])
    evaluations = 0
    max_evaluations = int(payload.full_update_max_evaluations)
    budget_exhausted = False

    def evaluate(candidate: list[Any], *, gradient: bool) -> dict[str, Any] | None:
        nonlocal evaluations
        if evaluations >= max_evaluations:
            return None
        normalized = _normalize_tensors(xp, candidate)
        result = finite_torus_energy_gradient(xp, payload, normalized, with_gradient=gradient)
        evaluations += 1
        return result

    current = evaluate(working, gradient=True)
    if current is None:
        raise ValueError("finite-torus-gradient evaluation budget must allow an initial objective")
    initial_energy = float(current["energy"])
    history: list[dict[str, Any]] = []
    for iteration in range(1, int(payload.optimization_steps) + 1):
        gradients = current["gradients"]
        gradient_norm = float(current["gradient_norm"])
        sweep_start = float(current["energy"])
        accepted: list[Any] | None = None
        accepted_result: dict[str, Any] | None = None
        accepted_scale = 0.0
        for scale in (1.0, 0.5, 0.25, 0.125):
            if evaluations >= max_evaluations:
                budget_exhausted = True
                break
            candidate = [
                tensor - float(payload.full_update_step) * scale * gradient.reshape(tensor.shape)
                for tensor, gradient in zip(working, gradients)
            ]
            candidate_result = evaluate(candidate, gradient=False)
            if candidate_result is not None and float(candidate_result["energy"]) < sweep_start - float(payload.optimization_tolerance):
                accepted = _normalize_tensors(xp, candidate)
                accepted_result = candidate_result
                accepted_scale = scale
                break
        if accepted is not None and accepted_result is not None:
            working = accepted
            current = evaluate(working, gradient=True)
            if current is None:
                budget_exhausted = True
                break
        improvement = sweep_start - float(current["energy"])
        history.append({
            "iteration": iteration,
            "energy": float(current["energy"]),
            "energy_variance": float(current["energy_variance"]),
            "improvement": improvement,
            "gradient_norm": gradient_norm,
            "accepted_updates": 1 if accepted is not None else 0,
            "accepted_scale": accepted_scale,
            "evaluations": evaluations,
            "parameter_count": parameter_count,
            "evaluation_budget_exhausted": budget_exhausted,
        })
        if budget_exhausted or improvement <= float(payload.optimization_tolerance) or gradient_norm <= float(payload.optimization_tolerance):
            break

    return {
        "tensors": working,
        "iterations": len(history),
        "energy_history": history,
        "initial_energy": initial_energy,
        "final_energy": float(current["energy"]),
        "final_variance": float(current["energy_variance"]),
        "evaluations": evaluations,
        "parameter_count": parameter_count,
        "optimizer": "finite-torus-gradient",
        "gradient_backend": "analytic-finite-torus-reverse-contraction",
        "objective": "finite-periodic-peps-2x2",
        "reference_sites": 4,
        "materializes_reference_statevector": True,
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
