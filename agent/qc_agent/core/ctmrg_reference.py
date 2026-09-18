"""Small independent references for bounded CTMRG acceptance evidence.

References are deliberately explicit about their scope: product supercells,
one analytic GHZ transfer fixed point, and a tiny exact 2x2 periodic PEPS
double-layer contraction.  No arbitrary entangled tensor is silently reduced
to a product ansatz or promoted to thermodynamic-limit evidence.
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


def analytic_ghz_reference(
    payload: CTMRGPayload,
    tensors: list[Any],
    onsite_values: list[float],
    interaction_values: list[float | None],
    ctmrg_energy: float,
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate the canonical one-site GHZ transfer fixed point.

    This is intentionally a narrow entangled reference, not a generic
    entangled-iPEPS approximation.  The tensor has two equal virtual sectors
    and only ``A^0_{0000}``/``A^1_{1111}`` non-zero.  Its symmetric infinite
    transfer fixed point has ``<Z>=0`` and nearest-neighbor ``<Z Z>=1``.
    Keeping this reference explicit lets the solver validate one important
    D=2 case without projecting arbitrary entangled tensors onto a product
    ansatz.
    """

    unavailable = {
        "performed": False,
        "reason": "analytic GHZ reference requires a one-site physical-2, virtual-2 tensor",
        "energy_variance": None,
    }
    if len(tensors) != 1 or int(payload.unit_cell[0]) != 1 or int(payload.unit_cell[1]) != 1:
        return unavailable
    if int(payload.physical_bond_dim) != 2 or int(payload.virtual_bond_dim) != 2:
        return unavailable
    tensor = _host(tensors[0]).astype(np.complex128, copy=False)
    expected = np.zeros_like(tensor)
    expected[(0, 0, 0, 0, 0)] = 1.0
    expected[(1, 1, 1, 1, 1)] = 1.0
    scale = float(np.linalg.norm(tensor))
    if scale <= 1e-14 or not np.all(np.isfinite(tensor)):
        return {**unavailable, "reason": "tensor is zero or non-finite"}
    normalized = tensor / scale
    expected /= float(np.linalg.norm(expected))
    # Global phase and harmless overall normalization are gauge freedoms.
    overlap = np.vdot(expected, normalized)
    if abs(overlap) <= 1e-12 or not np.allclose(normalized, overlap / abs(overlap) * expected, atol=1e-6, rtol=1e-6):
        return unavailable

    def one_site(pauli: str) -> float | None:
        if pauli == "I":
            return 1.0
        if pauli in {"X", "Y", "Z"}:
            return 0.0
        return None

    def two_site(left: str, right: str) -> float | None:
        if left == right == "I":
            return 1.0
        if left == right == "Z":
            return 1.0
        if left in {"I", "X", "Y", "Z"} and right in {"I", "X", "Y", "Z"}:
            return 0.0
        return None

    expected_onsite: list[float] = []
    for term in payload.terms:
        if len(term.paulis) > 1 or any(int(site) != 0 for site in term.paulis):
            return unavailable
        value = one_site(str(next(iter(term.paulis.values()), "I")))
        if value is None:
            return unavailable
        expected_onsite.append(value)
    expected_interactions: list[float] = []
    for interaction in payload.interactions:
        if interaction.left_site != 0 or interaction.right_site != 0 or tuple(map(abs, interaction.displacement)) not in ((1, 0), (0, 1)):
            return unavailable
        value = two_site(str(interaction.left_pauli), str(interaction.right_pauli))
        if value is None:
            return unavailable
        expected_interactions.append(value)

    onsite_errors = [abs(float(actual) - expected) for actual, expected in zip(onsite_values, expected_onsite)]
    interaction_errors = [
        abs(float(actual) - expected)
        for actual, expected in zip(interaction_values, expected_interactions)
        if actual is not None
    ]
    complete = all(value is not None for value in interaction_values)
    max_error = max([*onsite_errors, *interaction_errors], default=0.0)
    expected_energy = sum(float(term.coefficient) * value for term, value in zip(payload.terms, expected_onsite))
    expected_energy += sum(
        float(term.coefficient) * value
        for term, value in zip(payload.interactions, expected_interactions)
    )
    max_error = max(max_error, abs(float(ctmrg_energy) - expected_energy))
    return {
        "performed": True,
        "reference": "analytic-ghz-transfer-fixed-point",
        "reference_energy": expected_energy,
        "energy_error": abs(float(ctmrg_energy) - expected_energy),
        "energy_second_moment": None,
        "energy_variance": None,
        "observable_max_abs_error": max(onsite_errors, default=0.0),
        "interaction_max_abs_error": max(interaction_errors, default=0.0),
        "max_abs_error": max_error,
        "energy_complete": complete,
        "passed": bool(complete and max_error <= max(float(tolerance), 1e-5)),
        "tolerance": max(float(tolerance), 1e-5),
        "limitations": [
            "analytic reference is limited to the canonical one-site GHZ transfer fixed point",
            "the reference validates selected local observables, not an infinite-lattice variance",
        ],
    }


def finite_periodic_peps_reference(
    payload: CTMRGPayload,
    tensors: list[Any],
    onsite_values: list[float],
    interaction_values: list[float | None],
    ctmrg_energy: float,
    *,
    size: int = 2,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Compare a small generic iPEPS unit cell against an exact finite torus.

    The finite torus is an independent double-layer einsum reference for
    arbitrary complex tensors with virtual bond dimension up to two.  It
    supports every currently admitted unit cell and repeats the explicit
    tensor pattern on a 2x2 torus.  This is deliberately reported as
    finite-size evidence: agreement does not prove an infinite environment is
    converged, while disagreement is useful evidence that the current
    ``chi``/iteration choice needs review.
    """

    unavailable = {
        "performed": False,
        "reason": "finite periodic reference is limited to physical-2 iPEPS with virtual bond dimension <=2",
        "energy_variance": None,
    }
    cell_x, cell_y = (int(value) for value in payload.unit_cell)
    cell_sites = cell_x * cell_y
    if len(tensors) != cell_sites:
        return unavailable
    if int(payload.physical_bond_dim) != 2 or int(payload.virtual_bond_dim) > 2:
        return unavailable
    if int(size) < 2:
        return {**unavailable, "reason": "finite periodic reference requires a torus size of at least 2"}
    torus_x = max(int(size), cell_x)
    torus_y = max(int(size), cell_y)
    virtual = int(payload.virtual_bond_dim)
    expected_shape = (2, virtual, virtual, virtual, virtual)
    normalized_tensors: list[np.ndarray] = []
    for tensor_value in tensors:
        tensor = _host(tensor_value).astype(np.complex128, copy=False)
        if tensor.shape != expected_shape:
            return {**unavailable, "reason": "tensor shape does not match the declared physical and virtual dimensions"}
        if not np.all(np.isfinite(tensor)) or float(np.linalg.norm(tensor)) <= 1e-14:
            return {**unavailable, "reason": "tensor is zero or non-finite"}
        normalized_tensors.append(tensor)
    if not normalized_tensors:
        return {**unavailable, "reason": "tensor shape does not match the declared physical and virtual dimensions"}

    def site_index(x: int, y: int) -> int:
        return (int(x) % torus_x) + torus_x * (int(y) % torus_y)

    def cell_site_index(x: int, y: int) -> int:
        return (int(x) % cell_x) + cell_x * (int(y) % cell_y)

    def cell_coordinates(site: int) -> tuple[int, int]:
        return int(site) % cell_x, int(site) // cell_x

    for term in payload.terms:
        if len(term.paulis) > 1:
            return {**unavailable, "reason": "finite periodic reference supports one-site terms only"}
        if any(int(site) < 0 or int(site) >= cell_sites for site in term.paulis):
            return {**unavailable, "reason": "finite periodic reference term site exceeds the unit cell"}
    for interaction in payload.interactions:
        if tuple(map(abs, interaction.displacement)) not in ((1, 0), (0, 1)):
            return {**unavailable, "reason": "finite periodic reference supports nearest-neighbor interactions only"}
        if interaction.left_site >= cell_sites or interaction.right_site >= cell_sites:
            return {**unavailable, "reason": "finite periodic reference interaction site exceeds the unit cell"}
        left_x, left_y = cell_coordinates(interaction.left_site)
        dx, dy = (int(value) for value in interaction.displacement)
        expected_right = cell_site_index(left_x + dx, left_y + dy)
        if expected_right != int(interaction.right_site):
            return {
                **unavailable,
                "reason": "finite periodic reference requires interaction right_site to match left_site plus displacement",
            }

    # Each unique torus bond receives one fused ket/bra index.  Sharing the
    # same label between neighboring local tensors performs the exact finite
    # double-layer contraction without constructing a statevector.
    edge_labels = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    required_edges = 2 * torus_x * torus_y
    if required_edges > len(edge_labels):
        return {**unavailable, "reason": "finite periodic reference torus exceeds the independent einsum label budget"}
    horizontal = {(x, y): edge_labels[y * torus_x + x] for y in range(torus_y) for x in range(torus_x)}
    vertical = {
        (x, y): edge_labels[torus_x * torus_y + y * torus_x + x]
        for y in range(torus_y)
        for x in range(torus_x)
    }

    def contract(operator_by_site: dict[int, np.ndarray]) -> float:
        operands: list[np.ndarray] = []
        subscripts: list[str] = []
        for y in range(torus_y):
            for x in range(torus_x):
                site = site_index(x, y)
                operator = operator_by_site.get(site, _pauli("I"))
                tensor = normalized_tensors[cell_site_index(x, y)]
                layer = np.einsum(
                    "sudlr,st,tUDLR->uUdDlLrR",
                    tensor,
                    operator,
                    np.conjugate(tensor),
                    optimize=True,
                ).reshape((int(payload.virtual_bond_dim) ** 2,) * 4)
                operands.append(layer)
                subscripts.append(
                    "".join((
                        vertical[(x, (y - 1) % torus_y)],
                        vertical[(x, y % torus_y)],
                        horizontal[((x - 1) % torus_x, y)],
                        horizontal[(x, y)],
                    ))
                )
        value = np.einsum(",".join(subscripts), *operands, optimize=True)
        return float(np.real_if_close(value).real)

    def operator_map(paulis: dict[int, str]) -> dict[int, np.ndarray]:
        return {int(site): _pauli(str(label)) for site, label in paulis.items()}

    def multiply_maps(left: dict[int, np.ndarray], right: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        identities = _pauli("I")
        return {
            site: left.get(site, identities) @ right.get(site, identities)
            for site in set(left) | set(right)
        }

    base_terms: list[tuple[float, dict[int, np.ndarray]]] = []
    expected_onsite: list[float] = []
    for term in payload.terms:
        term_site = int(next(iter(term.paulis), 0))
        term_x, term_y = cell_coordinates(term_site)
        term_label = str(next(iter(term.paulis.values()), "I"))
        mapped = operator_map({site_index(term_x, term_y): term_label})
        expected_onsite.append(contract(mapped))
        for y in range(torus_y):
            for x in range(torus_x):
                base_terms.append((
                    float(term.coefficient),
                    operator_map({site_index(x + term_x, y + term_y): term_label}),
                ))

    expected_interactions: list[float] = []
    for interaction in payload.interactions:
        dx, dy = (int(value) for value in interaction.displacement)
        left_x, left_y = cell_coordinates(interaction.left_site)
        left = site_index(left_x, left_y)
        right = site_index(left_x + dx, left_y + dy)
        representative = operator_map({
            left: str(interaction.left_pauli),
            right: str(interaction.right_pauli),
        })
        expected_interactions.append(contract(representative))
        for y in range(torus_y):
            for x in range(torus_x):
                base_terms.append((
                    float(interaction.coefficient),
                    operator_map({
                        site_index(x + left_x, y + left_y): str(interaction.left_pauli),
                        site_index(x + left_x + dx, y + left_y + dy): str(interaction.right_pauli),
                    }),
                ))

    total_sites = torus_x * torus_y
    reference_energy_total = sum(coefficient * contract(operators) for coefficient, operators in base_terms)
    reference_energy = reference_energy_total / total_sites
    second_moment_total = 0.0
    for left_coefficient, left_operator in base_terms:
        for right_coefficient, right_operator in base_terms:
            second_moment_total += left_coefficient * right_coefficient * contract(multiply_maps(left_operator, right_operator))
    second_moment = second_moment_total / (total_sites * total_sites)
    variance = max(0.0, float(second_moment - reference_energy * reference_energy))
    observable_errors = [abs(float(actual) - expected) for actual, expected in zip(onsite_values, expected_onsite)]
    interaction_errors = [
        abs(float(actual) - expected)
        for actual, expected in zip(interaction_values, expected_interactions)
        if actual is not None
    ]
    complete = all(value is not None for value in interaction_values)
    max_error = max([*observable_errors, *interaction_errors, abs(float(ctmrg_energy) - reference_energy)], default=0.0)
    return {
        "performed": True,
        "reference": f"finite-periodic-peps-{torus_x}x{torus_y}",
        "reference_sites": total_sites,
        "reference_unit_cell": [cell_x, cell_y],
        "reference_lattice": [torus_x, torus_y],
        "reference_energy": float(reference_energy),
        "energy_error": abs(float(ctmrg_energy) - reference_energy),
        "energy_second_moment": float(second_moment),
        "energy_variance": variance,
        "observable_max_abs_error": max(observable_errors, default=0.0),
        "interaction_max_abs_error": max(interaction_errors, default=0.0),
        "max_abs_error": max_error,
        "energy_complete": complete,
        "passed": bool(complete and max_error <= max(float(tolerance), 1e-5)),
        "tolerance": max(float(tolerance), 1e-5),
        "limitations": [
            "this is an exact finite torus, not a thermodynamic-limit reference",
            "the declared unit-cell tensor pattern is repeated on the finite torus",
            "compare multiple torus sizes and environment chi before drawing infinite-lattice conclusions",
        ],
    }
