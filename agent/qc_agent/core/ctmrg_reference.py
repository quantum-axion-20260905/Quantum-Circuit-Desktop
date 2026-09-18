"""Small independent references for bounded CTMRG acceptance evidence.

The reference deliberately covers only product iPEPS tensors (virtual bond
dimension one).  It builds a finite product supercell with NumPy and evaluates
the declared unit-cell Hamiltonian directly.  Entangled tensors are reported
as unavailable rather than being silently approximated by a product state.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..plugins.models import CTMRGPayload


def _pauli(label: str) -> np.ndarray:
    return {
        "I": np.eye(2, dtype=np.complex128),
        "X": np.asarray([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.asarray([[1, 0], [0, -1]], dtype=np.complex128),
    }[label]


def _operator(n_sites: int, paulis: dict[int, str]) -> np.ndarray:
    value = np.asarray([[1.0 + 0.0j]], dtype=np.complex128)
    for site in range(n_sites):
        value = np.kron(value, _pauli(paulis.get(site, "I")))
    return value


def _host(value: Any) -> np.ndarray:
    try:
        return np.asarray(value.get())
    except AttributeError:
        return np.asarray(value)


def finite_product_reference(
    payload: CTMRGPayload,
    tensors: list[Any],
    onsite_values: list[float],
    interaction_values: list[float | None],
    ctmrg_energy: float,
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Compare a product iPEPS contraction with a direct finite supercell.

    A same-site interaction is represented by two copies of the unit cell,
    because an iPEPS displacement can connect a site to its translated copy.
    The direct reference is therefore finite-cell evidence, not an infinite
    lattice variance proof.
    """

    if int(payload.virtual_bond_dim) != 1:
        return {
            "performed": False,
            "reason": "finite product reference is limited to virtual_bond_dim=1",
            "energy_variance": None,
        }
    cell_sites = math.prod(payload.unit_cell)
    if len(tensors) != cell_sites:
        return {"performed": False, "reason": "tensor count does not match the unit cell", "energy_variance": None}
    if any(len(term.paulis) > 1 for term in payload.terms):
        return {"performed": False, "reason": "reference does not support multi-site onsite terms", "energy_variance": None}

    states: list[np.ndarray] = []
    for tensor in tensors:
        vector = _host(tensor[:, 0, 0, 0, 0]).astype(np.complex128, copy=False)
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 1e-14:
            return {"performed": False, "reason": "product tensor contains a zero or non-finite state", "energy_variance": None}
        states.append(vector / norm)

    needs_translated_copy = any(item.left_site == item.right_site for item in payload.interactions)
    reference_states = states + (states if needs_translated_copy else [])
    n_sites = len(reference_states)
    state = reference_states[0]
    for vector in reference_states[1:]:
        state = np.kron(state, vector)

    hamiltonian = np.zeros((2 ** n_sites, 2 ** n_sites), dtype=np.complex128)
    reference_observables: list[float] = []
    for term in payload.terms:
        mapped = {int(site): str(pauli) for site, pauli in term.paulis.items()}
        operator = _operator(n_sites, mapped)
        hamiltonian += float(term.coefficient) * operator
        reference_observables.append(float(np.vdot(state, operator @ state).real))

    reference_interactions: list[float] = []
    for interaction in payload.interactions:
        right_site = int(interaction.right_site)
        if interaction.left_site == interaction.right_site:
            right_site += cell_sites
        mapped = {
            int(interaction.left_site): str(interaction.left_pauli),
            right_site: str(interaction.right_pauli),
        }
        operator = _operator(n_sites, mapped)
        hamiltonian += float(interaction.coefficient) * operator
        reference_interactions.append(float(np.vdot(state, operator @ state).real))

    reference_energy = float(np.vdot(state, hamiltonian @ state).real)
    second_moment = float(np.vdot(state, hamiltonian @ (hamiltonian @ state)).real)
    variance = max(0.0, second_moment - reference_energy * reference_energy)
    observable_errors = [
        abs(float(actual) - expected)
        for actual, expected in zip(onsite_values, reference_observables)
    ]
    interaction_errors = [
        abs(float(actual) - expected)
        for actual, expected in zip(interaction_values, reference_interactions)
        if actual is not None
    ]
    all_errors = [*observable_errors, *interaction_errors, abs(float(ctmrg_energy) - reference_energy)]
    max_error = max(all_errors, default=0.0)
    complete = all(value is not None for value in interaction_values)
    return {
        "performed": True,
        "reference": "finite-product-supercell",
        "reference_sites": n_sites,
        "reference_energy": reference_energy,
        "energy_error": abs(float(ctmrg_energy) - reference_energy),
        "energy_second_moment": second_moment,
        "energy_variance": variance,
        "observable_max_abs_error": max(observable_errors, default=0.0),
        "interaction_max_abs_error": max(interaction_errors, default=0.0),
        "max_abs_error": max_error,
        "energy_complete": complete,
        "passed": bool(complete and max_error <= max(float(tolerance), 1e-5)),
        "tolerance": max(float(tolerance), 1e-5),
        "limitations": [
            "variance is for the declared finite product supercell, not an infinite-lattice variance",
            "translated same-site interactions use a copied unit cell in the reference",
        ],
    }
