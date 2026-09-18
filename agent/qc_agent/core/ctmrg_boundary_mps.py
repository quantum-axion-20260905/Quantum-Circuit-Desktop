"""Bounded finite-cylinder boundary-MPS diagnostics for iPEPS tensors.

This module is deliberately independent from the CTMRG environment update.
It repeats a declared iPEPS unit cell on a finite open patch, contracts rows
as MPOs against a boundary MPS, and returns truncation diagnostics.  The
patch is a finite-size/boundary probe, not an infinite-lattice replacement;
callers must keep its result separate from the CTMRG research gate.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _pauli(label: str) -> np.ndarray:
    value = str(label).upper()
    if value == "I":
        return np.eye(2, dtype=np.complex128)
    if value == "X":
        return np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    if value == "Y":
        return np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    if value == "Z":
        return np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    raise ValueError(f"unsupported Pauli operator {label!r}")


def _host(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach()
    cpu = getattr(value, "cpu", None)
    if cpu is not None:
        value = cpu()
    get = getattr(value, "get", None)
    if get is not None:
        value = get()
    return np.asarray(value)


def _double_layer(tensor: np.ndarray, operator: np.ndarray | None = None) -> np.ndarray:
    op = _pauli("I") if operator is None else operator
    raw = np.einsum(
        "sudlr,st,tUDLR->uUdDlLrR",
        tensor,
        op,
        np.conjugate(tensor),
        optimize=True,
    )
    virtual = int(tensor.shape[1])
    return raw.reshape((virtual * virtual,) * 4)


def tensors_from_payload(payload: Any) -> list[np.ndarray]:
    """Materialize the small declared iPEPS cell for the reference path only."""

    physical = int(payload.physical_bond_dim)
    virtual = int(payload.virtual_bond_dim)
    if physical != 2:
        raise ValueError("finite-cylinder boundary-MPS currently supports physical_bond_dim=2")
    cell_sites = int(payload.unit_cell[0]) * int(payload.unit_cell[1])
    tensor_size = physical * virtual ** 4
    if payload.tensor_data is not None:
        values = [complex(float(real), float(imaginary)) for real, imaginary in payload.tensor_data]
        if len(values) != cell_sites * tensor_size:
            raise ValueError("boundary-MPS tensor_data length does not match the declared unit cell")
        raw = np.asarray(values, dtype=np.complex128)
        return [raw[index * tensor_size:(index + 1) * tensor_size].reshape(
            (physical, virtual, virtual, virtual, virtual)
        ) for index in range(cell_sites)]
    tensors: list[np.ndarray] = []
    for site in range(cell_sites):
        tensor = np.zeros((physical, virtual, virtual, virtual, virtual), dtype=np.complex128)
        if payload.initial_state == "plus":
            amplitudes = [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]
        elif payload.initial_state == "down":
            amplitudes = [0.0, 1.0]
        elif payload.initial_state == "neel" and ((site % int(payload.unit_cell[0])) + (site // int(payload.unit_cell[0]))) % 2:
            amplitudes = [0.0, 1.0]
        else:
            amplitudes = [1.0, 0.0]
        tensor[:, 0, 0, 0, 0] = amplitudes
        tensors.append(tensor)
    return tensors


def _row_mpo(
    layer: np.ndarray,
    *,
    x: int,
    width: int,
    ones: np.ndarray,
) -> np.ndarray:
    """Return one local MPO tensor in ``(left,right,input,output)`` order."""

    operator = np.transpose(layer, (2, 3, 0, 1))
    if width == 1:
        return np.einsum("lrud,l,r->ud", operator, ones, ones, optimize=True)[None, None, :, :]
    if x == 0:
        reduced = np.einsum("lrud,l->rud", operator, ones, optimize=True)
        return reduced[None, :, :, :]
    if x == width - 1:
        reduced = np.einsum("lrud,r->lud", operator, ones, optimize=True)
        return reduced[:, None, :, :]
    return operator


def _apply_row_mpo(boundary: list[np.ndarray], row: list[np.ndarray]) -> list[np.ndarray]:
    if len(boundary) != len(row):
        raise ValueError("boundary-MPS row width mismatch")
    output: list[np.ndarray] = []
    for state, operator in zip(boundary, row):
        if int(state.shape[1]) != int(operator.shape[2]):
            raise ValueError("boundary-MPS vertical bond dimension mismatch")
        product = np.einsum("aib,xyio->axoby", state, operator, optimize=True)
        output.append(product.reshape(
            int(state.shape[0] * operator.shape[0]),
            int(operator.shape[3]),
            int(state.shape[2] * operator.shape[1]),
        ))
    return output


def _compress(
    tensors: list[np.ndarray],
    max_bond_dim: int,
    cutoff: float,
) -> tuple[list[np.ndarray], float, int]:
    compressed = list(tensors)
    discarded_total = 0.0
    max_used = 1
    for index in range(len(compressed) - 1):
        tensor = compressed[index]
        left_dim, physical_dim, right_dim = (int(value) for value in tensor.shape)
        matrix = tensor.reshape(left_dim * physical_dim, right_dim)
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        weights = np.square(np.abs(singular))
        total = float(np.sum(weights))
        keep = min(int(max_bond_dim), int(weights.size))
        if cutoff > 0.0 and weights.size:
            threshold = float(weights[0]) * float(cutoff) ** 2
            keep = min(keep, max(1, int(np.sum(weights >= threshold))))
        discarded_total += float(np.sum(weights[keep:])) / total if total > 0.0 else 0.0
        compressed[index] = u[:, :keep].reshape(left_dim, physical_dim, keep)
        factor = singular[:keep, None] * vh[:keep, :]
        compressed[index + 1] = np.tensordot(factor, compressed[index + 1], axes=(1, 0))
        max_used = max(max_used, keep)
    return compressed, discarded_total, max_used


def _bottom_contract(boundary: list[np.ndarray], ones: np.ndarray) -> complex:
    environment = np.ones((1,), dtype=np.complex128)
    for tensor in boundary:
        reduced = np.einsum("asb,s->ab", tensor, ones, optimize=True)
        environment = np.einsum("a,ab->b", environment, reduced, optimize=True)
    return complex(environment[0])


def contract_patch(
    tensors: list[Any],
    unit_cell: list[int] | tuple[int, int],
    *,
    width: int,
    height: int,
    operators: dict[int, str] | None = None,
    max_bond_dim: int = 16,
    cutoff: float = 0.0,
) -> tuple[complex, dict[str, Any]]:
    """Contract one finite open patch with a boundary-MPS row sweep."""

    if width < 1 or height < 1:
        raise ValueError("boundary-MPS patch dimensions must be positive")
    if max_bond_dim < 1:
        raise ValueError("boundary-MPS boundary bond dimension must be positive")
    cell_x, cell_y = (int(value) for value in unit_cell)
    cell_sites = cell_x * cell_y
    if len(tensors) != cell_sites:
        raise ValueError("boundary-MPS tensor count does not match the declared unit cell")
    normalized = [_host(tensor).astype(np.complex128, copy=False) for tensor in tensors]
    expected_rank = (2, int(normalized[0].shape[1]), int(normalized[0].shape[1]), int(normalized[0].shape[1]), int(normalized[0].shape[1]))
    if any(tuple(tensor.shape) != expected_rank for tensor in normalized):
        raise ValueError("boundary-MPS requires equal physical and virtual tensor shapes in the cell")
    virtual = int(normalized[0].shape[1])
    double_virtual = virtual * virtual
    ones = np.ones((double_virtual,), dtype=np.complex128)
    requested = {int(site): str(label).upper() for site, label in (operators or {}).items()}
    boundary = [np.ones((1, double_virtual, 1), dtype=np.complex128) for _ in range(width)]
    discarded_total = 0.0
    max_used = 1
    rows: list[dict[str, Any]] = []
    for y in range(height):
        row: list[np.ndarray] = []
        for x in range(width):
            cell_site = (x % cell_x) + cell_x * (y % cell_y)
            label = requested.get(y * width + x, "I")
            layer = _double_layer(normalized[cell_site], _pauli(label))
            row.append(_row_mpo(layer, x=x, width=width, ones=ones))
        boundary = _apply_row_mpo(boundary, row)
        boundary, discarded, used = _compress(boundary, int(max_bond_dim), float(cutoff))
        discarded_total += discarded
        max_used = max(max_used, used)
        rows.append({
            "row": y + 1,
            "boundary_bond_dim_used": used,
            "discarded_weight": float(discarded),
        })
    value = _bottom_contract(boundary, ones)
    return value, {
        "method": "finite-cylinder-boundary-mps",
        "patch": [int(width), int(height)],
        "boundary_bond_dim_requested": int(max_bond_dim),
        "boundary_bond_dim_used": int(max_used),
        "discarded_weight": float(discarded_total),
        "rows": rows,
        "rows_completed": int(height),
        "operator_sites": sorted(requested),
        "value_imaginary_abs": abs(float(value.imag)),
    }


def _select_site(
    *,
    cell_site: int,
    cell_x: int,
    cell_y: int,
    width: int,
    height: int,
    displacement: tuple[int, int] = (0, 0),
) -> tuple[int, int] | None:
    center_x, center_y = width // 2, height // 2
    candidates: list[tuple[int, int, int]] = []
    for y in range(height):
        for x in range(width):
            if (x % cell_x) + cell_x * (y % cell_y) != int(cell_site):
                continue
            tx, ty = x + int(displacement[0]), y + int(displacement[1])
            if not (0 <= tx < width and 0 <= ty < height):
                continue
            distance = abs(x - center_x) + abs(y - center_y)
            candidates.append((distance, x, y))
    if not candidates:
        return None
    _, x, y = min(candidates)
    return x, y


def run_boundary_mps_reference(
    tensors: list[Any],
    payload: Any,
    *,
    ctmrg_energy: float,
    ctmrg_onsite: list[float],
    ctmrg_interactions: list[float | None],
) -> dict[str, Any]:
    """Evaluate central local observables using an independent finite patch."""

    width = int(payload.boundary_mps_width)
    height = int(payload.boundary_mps_height)
    cell_x, cell_y = (int(value) for value in payload.unit_cell)
    norm, norm_diag = contract_patch(
        tensors,
        payload.unit_cell,
        width=width,
        height=height,
        max_bond_dim=int(payload.boundary_mps_bond_dim),
        cutoff=float(payload.boundary_mps_cutoff),
    )
    if abs(norm) <= 1e-30:
        return {
            "performed": False,
            "reason": "boundary-MPS finite patch norm is zero",
            "diagnostics": norm_diag,
        }
    onsite: list[float] = []
    onsite_diagnostics: list[dict[str, Any]] = []
    for term in payload.terms:
        site = int(next(iter(term.paulis), 0))
        position = _select_site(
            cell_site=site,
            cell_x=cell_x,
            cell_y=cell_y,
            width=width,
            height=height,
        )
        if position is None:
            return {"performed": False, "reason": "finite patch has no representative onsite site", "diagnostics": norm_diag}
        x, y = position
        label = str(next(iter(term.paulis.values()), "I"))
        numerator, diagnostics = contract_patch(
            tensors,
            payload.unit_cell,
            width=width,
            height=height,
            operators={y * width + x: label},
            max_bond_dim=int(payload.boundary_mps_bond_dim),
            cutoff=float(payload.boundary_mps_cutoff),
        )
        onsite.append(float(np.real(numerator / norm)))
        onsite_diagnostics.append({"position": [x, y], "operator": label, **diagnostics})

    interactions: list[float | None] = []
    interaction_diagnostics: list[dict[str, Any]] = []
    for interaction in payload.interactions:
        dx, dy = (int(value) for value in interaction.displacement)
        position = _select_site(
            cell_site=int(interaction.left_site),
            cell_x=cell_x,
            cell_y=cell_y,
            width=width,
            height=height,
            displacement=(dx, dy),
        )
        if position is None:
            interactions.append(None)
            interaction_diagnostics.append({"performed": False, "reason": "finite patch has no representative interaction bond"})
            continue
        x, y = position
        tx, ty = x + dx, y + dy
        numerator, diagnostics = contract_patch(
            tensors,
            payload.unit_cell,
            width=width,
            height=height,
            operators={
                y * width + x: str(interaction.left_pauli),
                ty * width + tx: str(interaction.right_pauli),
            },
            max_bond_dim=int(payload.boundary_mps_bond_dim),
            cutoff=float(payload.boundary_mps_cutoff),
        )
        interactions.append(float(np.real(numerator / norm)))
        interaction_diagnostics.append({"position": [[x, y], [tx, ty]], **diagnostics})

    expected_energy = sum(
        float(term.coefficient) * value
        for term, value in zip(payload.terms, onsite)
    ) + sum(
        float(interaction.coefficient) * float(value)
        for interaction, value in zip(payload.interactions, interactions)
        if value is not None
    )
    onsite_errors = [abs(float(actual) - float(expected)) for actual, expected in zip(ctmrg_onsite, onsite)]
    interaction_errors = [
        abs(float(actual) - float(expected))
        for actual, expected in zip(ctmrg_interactions, interactions)
        if actual is not None and expected is not None
    ]
    max_error = max(
        [*onsite_errors, *interaction_errors, abs(float(ctmrg_energy) - expected_energy)],
        default=0.0,
    )
    return {
        "performed": True,
        "reference": "finite-cylinder-boundary-mps",
        "reference_patch": [width, height],
        "reference_energy": float(expected_energy),
        "energy_error": abs(float(ctmrg_energy) - expected_energy),
        "observable_max_abs_error": max(onsite_errors, default=0.0),
        "interaction_max_abs_error": max(interaction_errors, default=0.0),
        "max_abs_error": max_error,
        "onsite_values": onsite,
        "interaction_values": interactions,
        "diagnostics": {
            "norm": float(np.real(norm)),
            "norm_imaginary_abs": abs(float(norm.imag)),
            "norm_contraction": norm_diag,
            "onsite_contractions": onsite_diagnostics,
            "interaction_contractions": interaction_diagnostics,
            "boundary_bond_dim_requested": int(payload.boundary_mps_bond_dim),
            "boundary_mps_cutoff": float(payload.boundary_mps_cutoff),
        },
        "limitations": [
            "finite open patch with fixed all-ones virtual boundary vectors, not an infinite-lattice fixed point",
            "central observables retain finite-size and boundary effects",
            "compare patch size and boundary bond dimension before using as scientific evidence",
        ],
    }


def run_boundary_mps_convergence_study(
    tensors: list[Any],
    payload: Any,
    *,
    ctmrg_energy: float,
    ctmrg_onsite: list[float],
    ctmrg_interactions: list[float | None],
    patch_sizes: list[tuple[int, int]],
    boundary_bond_dims: list[int],
) -> dict[str, Any]:
    """Compare the finite-cylinder diagnostic across bounded controls."""

    if not patch_sizes or not boundary_bond_dims:
        raise ValueError("boundary-MPS convergence study requires patch sizes and bond dimensions")
    if len(patch_sizes) * len(boundary_bond_dims) > 8:
        raise ValueError("boundary-MPS convergence study is limited to eight points")
    normalized_sizes = [(int(width), int(height)) for width, height in patch_sizes]
    normalized_bonds = [int(value) for value in boundary_bond_dims]
    if any(width < 2 or height < 2 or width > 16 or height > 16 for width, height in normalized_sizes):
        raise ValueError("boundary-MPS study patch dimensions must be between 2 and 16")
    if any(value < 1 or value > 128 for value in normalized_bonds):
        raise ValueError("boundary-MPS study bond dimensions must be between 1 and 128")

    points: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for width, height in normalized_sizes:
        for bond_dim in normalized_bonds:
            point_payload = payload.model_copy(update={
                "boundary_mps_width": width,
                "boundary_mps_height": height,
                "boundary_mps_bond_dim": bond_dim,
            })
            reference = run_boundary_mps_reference(
                tensors,
                point_payload,
                ctmrg_energy=ctmrg_energy,
                ctmrg_onsite=ctmrg_onsite,
                ctmrg_interactions=ctmrg_interactions,
            )
            reference_energy = reference.get("reference_energy")
            energy_delta = None
            if reference_energy is not None and previous is not None and previous.get("reference_energy") is not None:
                energy_delta = abs(float(reference_energy) - float(previous["reference_energy"]))
            point = {
                "patch": [width, height],
                "boundary_bond_dim": bond_dim,
                "performed": bool(reference.get("performed", False)),
                "reference_energy": reference_energy,
                "energy_error": reference.get("energy_error"),
                "observable_max_abs_error": reference.get("observable_max_abs_error"),
                "interaction_max_abs_error": reference.get("interaction_max_abs_error"),
                "max_abs_error": reference.get("max_abs_error"),
                "discarded_weight": (
                    reference.get("diagnostics", {})
                    .get("norm_contraction", {})
                    .get("discarded_weight")
                ),
                "energy_abs_delta": energy_delta,
                "reference": reference,
            }
            points.append(point)
            previous = point

    return {
        "status": "done",
        "method": "finite-cylinder-boundary-mps-convergence-study",
        "points": points,
        "point_count": len(points),
        "max_abs_error": max(
            (float(point["max_abs_error"]) for point in points if point["max_abs_error"] is not None),
            default=None,
        ),
        "max_discarded_weight": max(
            (float(point["discarded_weight"]) for point in points if point["discarded_weight"] is not None),
            default=None,
        ),
        "limitations": [
            "points are finite open patches with fixed all-ones virtual boundaries",
            "point-to-point deltas diagnose boundary-MPS truncation, not infinite-lattice convergence",
            "compare with CTMRG chi, residual, and paired-gauge evidence",
        ],
    }
