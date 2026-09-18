"""Bounded iPEPS corner-transfer-matrix contraction.

This module deliberately owns only the numerical representation.  Admission,
HTTP lifecycle, provenance, and domain builders stay outside the module so
larger unit cells and domain-specific optimizers can reuse the same result
seam.

The solver supports one-site and two-site checkerboard environments. It does
not optimize the iPEPS tensor and therefore reports ``needs_review``; that
distinction is important for research use and prevents a product-state ansatz
from being presented as a variational ground state.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Iterable

from ..plugins.models import CTMRGPayload, PauliTerm
from ..provenance import sha256_json
from .checkpoints import load_ctm_checkpoint, save_ctm_checkpoint
from .contracts import CheckpointManifest, ConvergencePoint, ConvergenceReport, ResearchResult, TruncationReport
from .ctmrg_reference import finite_product_reference
from .ipeps_optimizer import optimize_product_states, run_full_update, run_simple_update
from .observables import structured_observables


@dataclass(frozen=True)
class CTMEnvironment:
    """Four corners and four edge tensors around one unit-cell tensor.

    Corner tensors have shape ``(chi, chi)`` and edge tensors have shape
    ``(chi, D2, chi)``, where ``D2`` is the double-layer virtual dimension.
    The ordering is ``C1/T1/C2`` at the top, ``C4/T3/C3`` at the bottom,
    with ``T4`` on the left and ``T2`` on the right.
    """

    C1: Any
    C2: Any
    C3: Any
    C4: Any
    T1: Any
    T2: Any
    T3: Any
    T4: Any

    def tensors(self) -> tuple[Any, ...]:
        return (self.C1, self.C2, self.C3, self.C4, self.T1, self.T2, self.T3, self.T4)


def _host(value: Any) -> Any:
    try:
        return value.get()
    except AttributeError:
        return value


def _real(value: Any) -> float:
    return float(complex(_host(value)).real)


def _state_pairs(xp: Any, states: list[Any]) -> list[list[list[float]]]:
    return [
        [[float(complex(value).real), float(complex(value).imag)] for value in _host(state).reshape(-1)]
        for state in states
    ]


def _max_abs(xp: Any, value: Any) -> float:
    return float(_host(xp.max(xp.abs(value))))


def _build_tensors(xp: Any, payload: CTMRGPayload, state_vectors: list[Any] | None = None) -> list[Any]:
    """Create or import all tensors in the explicit unit cell.

    If ``tensor_data`` is supplied it is interpreted as row-major complex
    pairs in the explicit shape ``(physical, up, down, left, right)``.
    Otherwise a deterministic product-state ansatz is built for compatibility
    and the result is marked as non-variational by the caller.
    """

    physical = int(payload.physical_bond_dim)
    if physical != 2:
        raise ValueError("the first CTMRG solver supports physical_bond_dim=2 only")
    dtype = xp.complex64 if payload.dtype == "complex64" else xp.complex128
    virtual = int(payload.virtual_bond_dim)
    cell_sites = math.prod(payload.unit_cell)
    nx = int(payload.unit_cell[0])
    tensor_size = physical * virtual ** 4
    if state_vectors is None and payload.tensor_data is not None:
        values = [complex(float(real), float(imaginary)) for real, imaginary in payload.tensor_data]
        raw = xp.asarray(values, dtype=dtype)
        return [raw[index * tensor_size : (index + 1) * tensor_size].reshape(
            (physical, virtual, virtual, virtual, virtual)
        ) for index in range(cell_sites)]
    tensors: list[Any] = []
    for site in range(cell_sites):
        tensor = xp.zeros((physical, virtual, virtual, virtual, virtual), dtype=dtype)
        if state_vectors is not None:
            tensor[:, 0, 0, 0, 0] = state_vectors[site]
        else:
            if payload.initial_state == "plus":
                amplitudes = [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)]
            else:
                amplitudes = [1.0, 0.0]
                if payload.initial_state == "down":
                    amplitudes = [0.0, 1.0]
                elif payload.initial_state == "neel" and ((site % nx) + (site // nx)) % 2:
                    amplitudes = [0.0, 1.0]
            tensor[:, 0, 0, 0, 0] = xp.asarray(amplitudes, dtype=dtype)
        tensors.append(tensor)
    return tensors


def _build_tensor(xp: Any, payload: CTMRGPayload) -> Any:
    """Compatibility helper for one-site callers and reference tests."""

    return _build_tensors(xp, payload)[0]


def _pauli_matrix(xp: Any, dtype: Any, label: str) -> Any:
    if label == "I":
        return xp.eye(2, dtype=dtype)
    if label == "X":
        return xp.asarray([[0, 1], [1, 0]], dtype=dtype)
    if label == "Y":
        return xp.asarray([[0, -1j], [1j, 0]], dtype=dtype)
    if label == "Z":
        return xp.asarray([[1, 0], [0, -1]], dtype=dtype)
    raise ValueError(f"unsupported Pauli operator {label!r}")


def _double_layer(xp: Any, tensor: Any, operator: Any | None = None) -> Any:
    """Fuse ket/bra virtual legs while contracting the physical leg."""

    physical = tensor.shape[0]
    dtype = tensor.dtype
    op = operator if operator is not None else xp.eye(physical, dtype=dtype)
    raw = xp.einsum(
        "sudlr,st,tUDLR->uUdDlLrR",
        tensor,
        op,
        xp.conj(tensor),
    )
    d2 = tensor.shape[1] * tensor.shape[1]
    return raw.reshape((d2, d2, d2, d2))


def _initialize_environment(xp: Any, double_layer: Any, chi: int) -> CTMEnvironment:
    d2 = int(double_layer.shape[0])
    dtype = double_layer.dtype
    corner = xp.zeros((chi, chi), dtype=dtype)
    diagonal = min(chi, d2)
    corner[xp.arange(diagonal), xp.arange(diagonal)] = 1
    edge = xp.zeros((chi, d2, chi), dtype=dtype)
    for index in range(min(chi, d2)):
        edge[index, :, index] = 1
    return CTMEnvironment(corner, corner.copy(), corner.copy(), corner.copy(), edge, edge.copy(), edge.copy(), edge.copy())


def _ctm_move(xp: Any, left_corner: Any, right_corner: Any, grown_edge: Any, chi: int) -> tuple[Any, Any, Any, float]:
    """Truncate one enlarged boundary with a Hermitian half-system projector."""

    rho = left_corner @ xp.conj(left_corner).T + right_corner @ xp.conj(right_corner).T
    rho = 0.5 * (rho + xp.conj(rho).T)
    eigenvalues, eigenvectors = xp.linalg.eigh(rho)
    keep = min(int(chi), int(eigenvalues.shape[0]))
    projector = eigenvectors[:, -keep:]
    projector = projector[:, ::-1]
    discarded = float(_host(xp.sum(xp.clip(xp.real(eigenvalues[:-keep]), 0, None)))) if keep < eigenvalues.shape[0] else 0.0
    retained = float(_host(xp.sum(xp.clip(xp.real(eigenvalues[-keep:]), 0, None))))
    discarded_weight = discarded / max(discarded + retained, 1e-30)
    projector_dag = xp.conj(projector).T
    new_left = projector_dag @ left_corner
    new_right = projector_dag @ right_corner
    # The first factor is stored as (enlarged, kept) so einsum contracts its
    # conjugate on the enlarged index and emits the kept index.  This is the
    # same P† T P projection used for the corners, but preserves the explicit
    # edge orientation without transposing the middle leg.
    new_edge = xp.einsum("ia,idj,jb->adb", xp.conj(projector), grown_edge, projector)
    return new_left, new_right, new_edge, discarded_weight


def _left_move(xp: Any, env: CTMEnvironment, double_layer: Any, chi: int) -> tuple[CTMEnvironment, float]:
    d2 = int(double_layer.shape[0])
    c1_g = xp.einsum("ab,buc->auc", env.C1, env.T1).reshape(-1, env.T1.shape[2])
    c4_g = xp.einsum("gh,hdi->gdi", env.C4, env.T3).reshape(-1, env.T3.shape[2])
    t4_g = xp.einsum("alg,udlr->augdr", env.T4, double_layer)
    t4_g = t4_g.transpose(0, 1, 4, 2, 3).reshape(c1_g.shape[0], d2, c4_g.shape[0])
    c1, c4, t4, discarded = _ctm_move(xp, c1_g, c4_g, t4_g, chi)
    return CTMEnvironment(c1, env.C2, env.C3, c4, env.T1, env.T2, env.T3, t4), discarded


def _right_move(xp: Any, env: CTMEnvironment, double_layer: Any, chi: int) -> tuple[CTMEnvironment, float]:
    d2 = int(double_layer.shape[0])
    c2_g = xp.einsum("ce,buc->eub", env.C2, env.T1).reshape(-1, env.T1.shape[0])
    c3_g = xp.einsum("im,hdi->mdh", env.C3, env.T3).reshape(-1, env.T3.shape[0])
    t2_g = xp.einsum("erm,udlr->eumdl", env.T2, double_layer)
    t2_g = t2_g.transpose(0, 1, 4, 2, 3).reshape(c2_g.shape[0], d2, c3_g.shape[0])
    c2, c3, t2, discarded = _ctm_move(xp, c2_g, c3_g, t2_g, chi)
    return CTMEnvironment(env.C1, c2, c3, env.C4, env.T1, t2, env.T3, env.T4), discarded


def _top_move(xp: Any, env: CTMEnvironment, double_layer: Any, chi: int) -> tuple[CTMEnvironment, float]:
    d2 = int(double_layer.shape[0])
    c1_g = xp.einsum("ab,alg->blg", env.C1, env.T4).reshape(-1, env.T4.shape[2])
    c2_g = xp.einsum("ce,erm->crm", env.C2, env.T2).reshape(-1, env.T2.shape[2])
    t1_g = xp.einsum("buc,udlr->bcdlr", env.T1, double_layer)
    t1_g = t1_g.transpose(0, 3, 2, 1, 4).reshape(c1_g.shape[0], d2, c2_g.shape[0])
    c1, c2, t1, discarded = _ctm_move(xp, c1_g, c2_g, t1_g, chi)
    return CTMEnvironment(c1, c2, env.C3, env.C4, t1, env.T2, env.T3, env.T4), discarded


def _bottom_move(xp: Any, env: CTMEnvironment, double_layer: Any, chi: int) -> tuple[CTMEnvironment, float]:
    d2 = int(double_layer.shape[0])
    c4_g = xp.einsum("gh,alg->hal", env.C4, env.T4).transpose(0, 2, 1).reshape(-1, env.T4.shape[0])
    c3_g = xp.einsum("im,erm->ire", env.C3, env.T2).reshape(-1, env.T2.shape[0])
    t3_g = xp.einsum("hdi,udlr->hiulr", env.T3, double_layer)
    t3_g = t3_g.transpose(0, 3, 2, 1, 4).reshape(c4_g.shape[0], d2, c3_g.shape[0])
    c4, c3, t3, discarded = _ctm_move(xp, c4_g, c3_g, t3_g, chi)
    return CTMEnvironment(env.C1, env.C2, c3, c4, env.T1, env.T2, t3, env.T4), discarded


def _left_move_two_site(
    xp: Any,
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    neighbor_layer: Any,
    chi: int,
) -> tuple[CTMEnvironment, float]:
    d2 = int(neighbor_layer.shape[0])
    c1_g = xp.einsum("ab,buc->auc", env_self.C1, env_neighbor.T1).reshape(-1, env_neighbor.T1.shape[2])
    c4_g = xp.einsum("gh,hdi->gdi", env_self.C4, env_neighbor.T3).reshape(-1, env_neighbor.T3.shape[2])
    t4_g = xp.einsum("alg,udlr->augdr", env_self.T4, neighbor_layer)
    t4_g = t4_g.transpose(0, 1, 4, 2, 3).reshape(c1_g.shape[0], d2, c4_g.shape[0])
    c1, c4, t4, discarded = _ctm_move(xp, c1_g, c4_g, t4_g, chi)
    return CTMEnvironment(c1, env_self.C2, env_self.C3, c4, env_self.T1, env_self.T2, env_self.T3, t4), discarded


def _right_move_two_site(
    xp: Any,
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    neighbor_layer: Any,
    chi: int,
) -> tuple[CTMEnvironment, float]:
    d2 = int(neighbor_layer.shape[0])
    c2_g = xp.einsum("ce,buc->eub", env_self.C2, env_neighbor.T1).reshape(-1, env_neighbor.T1.shape[0])
    c3_g = xp.einsum("im,hdi->mdh", env_self.C3, env_neighbor.T3).reshape(-1, env_neighbor.T3.shape[0])
    t2_g = xp.einsum("erm,udlr->eumdl", env_self.T2, neighbor_layer)
    t2_g = t2_g.transpose(0, 1, 4, 2, 3).reshape(c2_g.shape[0], d2, c3_g.shape[0])
    c2, c3, t2, discarded = _ctm_move(xp, c2_g, c3_g, t2_g, chi)
    return CTMEnvironment(env_self.C1, c2, c3, env_self.C4, env_self.T1, t2, env_self.T3, env_self.T4), discarded


def _top_move_two_site(
    xp: Any,
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    neighbor_layer: Any,
    chi: int,
) -> tuple[CTMEnvironment, float]:
    d2 = int(neighbor_layer.shape[0])
    c1_g = xp.einsum("ab,alg->blg", env_self.C1, env_neighbor.T4).reshape(-1, env_neighbor.T4.shape[2])
    c2_g = xp.einsum("ce,erm->crm", env_self.C2, env_neighbor.T2).reshape(-1, env_neighbor.T2.shape[2])
    t1_g = xp.einsum("buc,udlr->bcdlr", env_self.T1, neighbor_layer)
    t1_g = t1_g.transpose(0, 3, 2, 1, 4).reshape(c1_g.shape[0], d2, c2_g.shape[0])
    c1, c2, t1, discarded = _ctm_move(xp, c1_g, c2_g, t1_g, chi)
    return CTMEnvironment(c1, c2, env_self.C3, env_self.C4, t1, env_self.T2, env_self.T3, env_self.T4), discarded


def _bottom_move_two_site(
    xp: Any,
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    neighbor_layer: Any,
    chi: int,
) -> tuple[CTMEnvironment, float]:
    d2 = int(neighbor_layer.shape[0])
    c4_g = xp.einsum("gh,alg->hal", env_self.C4, env_neighbor.T4).transpose(0, 2, 1).reshape(-1, env_neighbor.T4.shape[0])
    c3_g = xp.einsum("im,erm->ire", env_self.C3, env_neighbor.T2).reshape(-1, env_neighbor.T2.shape[0])
    t3_g = xp.einsum("hdi,udlr->hiulr", env_self.T3, neighbor_layer)
    t3_g = t3_g.transpose(0, 3, 2, 1, 4).reshape(c4_g.shape[0], d2, c3_g.shape[0])
    c4, c3, t3, discarded = _ctm_move(xp, c4_g, c3_g, t3_g, chi)
    return CTMEnvironment(env_self.C1, env_self.C2, c3, c4, env_self.T1, env_self.T2, t3, env_self.T4), discarded


def _unit_cell_sweep(
    xp: Any,
    environments: list[CTMEnvironment],
    layers: list[Any],
    chi: int,
    unit_cell: list[int],
) -> tuple[list[CTMEnvironment], float]:
    """Run one periodic unit-cell CTM sweep for 2, 3, or 4 tensors.

    The environment associated with a cell site is updated from the tensor
    and environment of its periodic neighbor in each direction.  The
    sequential update order is deterministic and reduces to the standard
    checkerboard two-site sweep for a 2x1 cell.
    """

    nx, ny = (int(value) for value in unit_cell)
    if len(environments) != nx * ny or len(layers) != nx * ny:
        raise ValueError("unit-cell CTMRG sweep tensor/environment count mismatch")

    def site_index(x: int, y: int) -> int:
        return (x % nx) + nx * (y % ny)

    def neighbors(site: int) -> tuple[int, int, int, int]:
        x, y = site % nx, site // nx
        return (
            site_index(x - 1, y),
            site_index(x + 1, y),
            site_index(x, y - 1),
            site_index(x, y + 1),
        )

    working = list(environments)
    discarded_total = 0.0
    moves = (
        (0, _left_move_two_site),
        (1, _right_move_two_site),
        (2, _top_move_two_site),
        (3, _bottom_move_two_site),
    )
    for neighbor_position, move in moves:
        for site in range(len(working)):
            neighbor = neighbors(site)[neighbor_position]
            working[site], discarded = move(xp, working[site], working[neighbor], layers[neighbor], chi)
            discarded_total += discarded
    return [_renormalize(xp, env) for env in working], discarded_total


def _renormalize(xp: Any, env: CTMEnvironment) -> CTMEnvironment:
    def normalize(value: Any) -> Any:
        return value / (_max_abs(xp, value) + 1e-30)

    return CTMEnvironment(*(normalize(value) for value in env.tensors()))


def _environment_residual(xp: Any, before: CTMEnvironment, after: CTMEnvironment) -> float:
    residual = 0.0
    for old, new in zip(before.tensors(), after.tensors()):
        scale = max(_max_abs(xp, old), 1e-30)
        residual = max(residual, _max_abs(xp, new - old) / scale)
    return residual


def _environment_diagnostics(xp: Any, environments: list[CTMEnvironment]) -> dict[str, Any]:
    """Extract bounded transfer-spectrum diagnostics from converged edges."""

    spectra: list[list[float]] = []
    correlation_lengths: list[float | None] = []
    for env in environments:
        transfer = xp.sum(env.T1, axis=1)
        eigenvalues = xp.linalg.eigvals(transfer)
        magnitudes = sorted((float(abs(value)) for value in _host(eigenvalues)), reverse=True)
        leading = magnitudes[0] if magnitudes else 0.0
        subleading = magnitudes[1] if len(magnitudes) > 1 else 0.0
        if leading <= 1e-30 or subleading <= 1e-30:
            correlation_lengths.append(0.0)
        else:
            ratio = min(1.0 - 1e-15, max(0.0, subleading / leading))
            correlation_lengths.append(float(-1.0 / math.log(ratio)) if ratio > 0 else 0.0)
        singular = xp.linalg.svd(env.T1.reshape(env.T1.shape[0], -1), compute_uv=False)
        singular_host = [float(abs(value)) for value in _host(singular)]
        scale = max(singular_host[0] if singular_host else 0.0, 1e-30)
        spectra.append([value / scale for value in singular_host])
    finite_lengths = [value for value in correlation_lengths if value is not None and math.isfinite(value)]
    return {
        "correlation_length": max(finite_lengths) if finite_lengths else None,
        "correlation_lengths_by_site": correlation_lengths,
        "environment_spectrum": spectra,
    }


def _environment_contraction(xp: Any, env: CTMEnvironment, local_tensor: Any) -> Any:
    return xp.einsum(
        "ab,buc,ce,erm,im,hdi,gh,alg,udlr->",
        env.C1,
        env.T1,
        env.C2,
        env.T2,
        env.C3,
        env.T3,
        env.C4,
        env.T4,
        local_tensor,
    )


def _term_expectation(xp: Any, env: CTMEnvironment, tensor: Any, term: PauliTerm) -> float:
    if len(term.paulis) > 1:
        raise ValueError("one-site CTMRG onsite terms must act on at most one unit-cell site")
    pauli = next(iter(term.paulis.values()), "I")
    dtype = tensor.dtype
    op = _pauli_matrix(xp, dtype, str(pauli))
    numerator = _environment_contraction(xp, env, _double_layer(xp, tensor, op))
    denominator = _environment_contraction(xp, env, _double_layer(xp, tensor))
    return _real(numerator / (denominator + 1e-30))


def _horizontal_two_site_contraction(xp: Any, env: CTMEnvironment, left: Any, right: Any) -> Any:
    """Contract two neighboring sites in the horizontal direction."""

    return xp.einsum(
        "ab,buc,cve,ef,fqm,mi,hyi,gwh,gh,azg,uwzx,vyxq->",
        env.C1,
        env.T1,
        env.T1,
        env.C2,
        env.T2,
        env.C3,
        env.T3,
        env.T3,
        env.C4,
        env.T4,
        left,
        right,
    )


def _horizontal_two_site_contraction_pair(
    xp: Any,
    left_env: CTMEnvironment,
    right_env: CTMEnvironment,
    left: Any,
    right: Any,
) -> Any:
    """Contract a horizontal bond with distinct checkerboard environments."""

    return xp.einsum(
        "ab,buc,cve,ef,fqm,mi,hyi,gwh,gh,azg,uwzx,vyxq->",
        left_env.C1,
        left_env.T1,
        right_env.T1,
        right_env.C2,
        right_env.T2,
        right_env.C3,
        right_env.T3,
        left_env.T3,
        left_env.C4,
        left_env.T4,
        left,
        right,
    )


def _vertical_two_site_contraction(xp: Any, env: CTMEnvironment, top: Any, bottom: Any) -> Any:
    """Contract two neighboring sites in the vertical direction."""

    return xp.einsum(
        "ab,buc,ce,erm,mqn,ni,hyi,gh,alg,gkh,uxlr,xykq->",
        env.C1,
        env.T1,
        env.C2,
        env.T2,
        env.T2,
        env.C3,
        env.T3,
        env.C4,
        env.T4,
        env.T4,
        top,
        bottom,
    )


def _vertical_two_site_contraction_pair(
    xp: Any,
    top_env: CTMEnvironment,
    bottom_env: CTMEnvironment,
    top: Any,
    bottom: Any,
) -> Any:
    """Contract a vertical bond with distinct checkerboard environments."""

    return xp.einsum(
        "ab,buc,ce,erm,mqn,ni,hyi,gh,alg,gkh,uxlr,xykq->",
        top_env.C1,
        top_env.T1,
        top_env.C2,
        top_env.T2,
        bottom_env.T2,
        bottom_env.C3,
        bottom_env.T3,
        bottom_env.C4,
        bottom_env.T4,
        top_env.T4,
        top,
        bottom,
    )


def _interaction_expectation(
    xp: Any,
    env: CTMEnvironment,
    tensor: Any,
    displacement: list[int],
    left_pauli: str,
    right_pauli: str,
) -> float | None:
    """Evaluate a nearest-neighbor two-site Pauli expectation from one CTM.

    A two-site density matrix is only defined here for nearest horizontal or
    vertical neighbors of a one-site translational cell.  Unsupported
    displacements return ``None`` so callers cannot mistake a product of local
    averages for a connected two-site observable.
    """

    dx, dy = (int(value) for value in displacement)
    if (abs(dx), abs(dy)) not in ((1, 0), (0, 1)):
        return None
    dtype = tensor.dtype
    left_operator = _pauli_matrix(xp, dtype, left_pauli)
    right_operator = _pauli_matrix(xp, dtype, right_pauli)
    left = _double_layer(xp, tensor, left_operator)
    right = _double_layer(xp, tensor, right_operator)
    identity = _double_layer(xp, tensor)
    if abs(dx) == 1:
        numerator = _horizontal_two_site_contraction(xp, env, left, right)
        denominator = _horizontal_two_site_contraction(xp, env, identity, identity)
    else:
        numerator = _vertical_two_site_contraction(xp, env, left, right)
        denominator = _vertical_two_site_contraction(xp, env, identity, identity)
    return _real(numerator / (denominator + 1e-30))


def _interaction_expectation_cell(
    xp: Any,
    environments: list[CTMEnvironment],
    tensors: list[Any],
    left_site: int,
    right_site: int,
    displacement: list[int],
    left_pauli: str,
    right_pauli: str,
) -> float | None:
    """Evaluate a nearest-neighbor bond for a two-site checkerboard cell."""

    dx, dy = (int(value) for value in displacement)
    if (abs(dx), abs(dy)) not in ((1, 0), (0, 1)):
        return None
    left_tensor = tensors[int(left_site)]
    right_tensor = tensors[int(right_site)]
    dtype = left_tensor.dtype
    left_layer = _double_layer(xp, left_tensor, _pauli_matrix(xp, dtype, left_pauli))
    right_layer = _double_layer(xp, right_tensor, _pauli_matrix(xp, dtype, right_pauli))
    left_identity = _double_layer(xp, left_tensor)
    right_identity = _double_layer(xp, right_tensor)
    if abs(dx) == 1:
        numerator = _horizontal_two_site_contraction_pair(
            xp, environments[int(left_site)], environments[int(right_site)], left_layer, right_layer
        )
        denominator = _horizontal_two_site_contraction_pair(
            xp, environments[int(left_site)], environments[int(right_site)], left_identity, right_identity
        )
    else:
        numerator = _vertical_two_site_contraction_pair(
            xp, environments[int(left_site)], environments[int(right_site)], left_layer, right_layer
        )
        denominator = _vertical_two_site_contraction_pair(
            xp, environments[int(left_site)], environments[int(right_site)], left_identity, right_identity
        )
    return _real(numerator / (denominator + 1e-30))


def _resource_summary(
    xp: Any,
    environments: CTMEnvironment | list[CTMEnvironment],
    tensors: Any | list[Any],
    iterations: int,
    unit_cell: list[int],
) -> dict[str, Any]:
    env_list = environments if isinstance(environments, list) else [environments]
    tensor_list = tensors if isinstance(tensors, list) else [tensors]
    values = sum(int(item.size) for env in env_list for item in env.tensors())
    values += sum(int(tensor.size) for tensor in tensor_list)
    tensor = tensor_list[0]
    itemsize = int(getattr(tensor.dtype, "itemsize", 8))
    return {
        "representation": "ipeps",
        "unit_cell": list(unit_cell),
        "unit_cell_sites": len(tensor_list),
        "physical_bond_dim": int(tensor.shape[0]),
        "virtual_bond_dim": int(tensor.shape[1]),
        "environment_bond_dim_used": max(int(env.C1.shape[0]) for env in env_list),
        "double_layer_virtual_dim": int(env_list[0].T1.shape[1]),
        "tensor_values": values,
        "peak_bytes_estimate": int(math.ceil(values * itemsize * 3.0)),
        "iterations": int(iterations),
        "materializes_statevector": False,
        "device": "cuda" if hasattr(xp, "cuda") else "cpu",
    }


def _problem_sha256(payload: CTMRGPayload) -> str:
    """Fingerprint the scientific tensor problem, excluding run controls."""

    data = payload.model_dump(
        mode="json",
        exclude={"iterations", "tolerance", "max_time_ms", "max_mem_mb", "checkpoint_path", "resume_from"},
    )
    return sha256_json(data)


def _environment_from_arrays(arrays: dict[str, Any]) -> CTMEnvironment:
    return CTMEnvironment(*(arrays[name] for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4")))


def _environments_from_arrays(arrays: dict[str, Any], count: int) -> list[CTMEnvironment]:
    environments: list[CTMEnvironment] = []
    for index in range(int(count)):
        prefix = "" if int(count) == 1 else f"site{index}_"
        environments.append(CTMEnvironment(*(
            arrays[f"{prefix}{name}"]
            for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4")
        )))
    return environments


def run_ctmrg(
    xp: Any,
    payload: CTMRGPayload,
    *,
    progress_cb: Any = None,
    cancel_cb: Any = None,
) -> dict[str, Any]:
    """Run bounded one-site or two-site checkerboard CTMRG contraction."""

    unit_cell = list(payload.unit_cell)
    if unit_cell not in ([1, 1], [2, 1], [1, 2], [2, 2]):
        raise ValueError("the current CTMRG solver supports unit_cell dimensions no larger than 2x2")

    started = time.perf_counter()
    tensors = _build_tensors(xp, payload)
    optimization_info: dict[str, Any] | None = None
    if payload.optimization != "none":
        if payload.optimization == "product-coordinate-descent" and int(payload.virtual_bond_dim) != 1:
            raise ValueError("product-coordinate-descent optimization requires virtual_bond_dim=1")
        if payload.optimization == "product-coordinate-descent":
            initial_states = [tensor[:, 0, 0, 0, 0] for tensor in tensors]
            optimization_info = optimize_product_states(xp, payload, initial_states)
            tensors = _build_tensors(xp, payload, state_vectors=optimization_info["states"])
        elif payload.optimization == "simple-update":
            optimization_info = run_simple_update(xp, payload, tensors)
            tensors = optimization_info["tensors"]
        elif payload.optimization == "full-update":
            optimization_info = run_full_update(xp, payload, tensors)
            tensors = optimization_info["tensors"]
    layers = [_double_layer(xp, tensor) for tensor in tensors]
    chi = int(payload.environment_bond_dim)
    environments = [_initialize_environment(xp, layer, chi) for layer in layers]
    problem_sha256 = _problem_sha256(payload)
    points: list[ConvergencePoint] = []
    discarded_total = 0.0
    residual = math.inf
    start_iteration = 0
    checkpoint_info: dict[str, Any] = {}

    if payload.resume_from:
        manifest, arrays = load_ctm_checkpoint(payload.resume_from, xp)
        if manifest.get("request_sha256") != problem_sha256:
            raise ValueError("CTMRG checkpoint does not match the scientific tensor problem")
        if manifest.get("dtype") != payload.dtype:
            raise ValueError("CTMRG checkpoint dtype does not match the requested dtype")
        metadata = manifest.get("metadata", {})
        for name, expected in (
            ("physical_bond_dim", payload.physical_bond_dim),
            ("virtual_bond_dim", payload.virtual_bond_dim),
            ("environment_bond_dim", payload.environment_bond_dim),
        ):
            if int(metadata.get(name, -1)) != int(expected):
                raise ValueError(f"CTMRG checkpoint {name} does not match the request")
        if list(metadata.get("unit_cell", unit_cell)) != unit_cell:
            raise ValueError("CTMRG checkpoint unit cell does not match the request")
        if int(metadata.get("environment_count", len(tensors))) != len(tensors):
            raise ValueError("CTMRG checkpoint environment count does not match the request")
        start_iteration = int(manifest.get("step", 0))
        if start_iteration > int(payload.iterations):
            raise ValueError(
                f"checkpoint already contains {start_iteration} iterations, but the requested run only allows {payload.iterations}"
            )
        raw_points = metadata.get("convergence_points", [])
        if not isinstance(raw_points, list):
            raise ValueError("CTMRG checkpoint convergence history is invalid")
        points = [ConvergencePoint(**dict(point)) for point in raw_points]
        discarded_total = float(metadata.get("discarded_weight_total", 0.0))
        residual = float(metadata.get("residual", math.inf))
        environments = _environments_from_arrays(arrays, len(tensors))
        checkpoint_info = manifest

    def save_iteration_checkpoint(iteration: int) -> None:
        nonlocal checkpoint_info
        if not payload.checkpoint_path:
            return
        checkpoint_info = save_ctm_checkpoint(
            payload.checkpoint_path,
            environments,
            CheckpointManifest(
                checkpoint_id=f"ctmrg-{problem_sha256[:12]}-iteration-{iteration}",
                request_sha256=problem_sha256,
                method="ipeps-ctmrg-contraction",
                representation="ipeps",
                dtype=payload.dtype,
                device="cuda" if hasattr(xp, "cuda") else "cpu",
                step=iteration,
                created_at=datetime.now(timezone.utc).isoformat(),
                metadata={
                    "completed_iterations": iteration,
                    "residual": float(residual),
                    "discarded_weight_total": float(discarded_total),
                    "convergence_points": [point.__dict__ for point in points],
                    "physical_bond_dim": int(payload.physical_bond_dim),
                    "virtual_bond_dim": int(payload.virtual_bond_dim),
                    "environment_bond_dim": int(payload.environment_bond_dim),
                    "unit_cell": list(unit_cell),
                    "environment_count": len(environments),
                },
            ),
        )

    for iteration in range(start_iteration + 1, int(payload.iterations) + 1):
        if cancel_cb and cancel_cb():
            raise RuntimeError("job canceled")
        before = list(environments)
        if len(environments) == 1:
            env = environments[0]
            env, left_discarded = _left_move(xp, env, layers[0], chi)
            env, right_discarded = _right_move(xp, env, layers[0], chi)
            env, top_discarded = _top_move(xp, env, layers[0], chi)
            env, bottom_discarded = _bottom_move(xp, env, layers[0], chi)
            environments = [_renormalize(xp, env)]
            discarded_sweep = left_discarded + right_discarded + top_discarded + bottom_discarded
        else:
            environments, discarded_sweep = _unit_cell_sweep(xp, environments, layers, chi, unit_cell)
        discarded_total += discarded_sweep
        residual = max(
            _environment_residual(xp, old, new)
            for old, new in zip(before, environments)
        )
        points.append(ConvergencePoint(
            iteration=iteration,
            residual=residual,
            environment_dim=max(int(env.C1.shape[0]) for env in environments),
            discarded_weight=discarded_sweep,
        ))
        save_iteration_checkpoint(iteration)
        if progress_cb:
            progress_cb(iteration / max(1, int(payload.iterations)), "ctmrg-sweep")
        if residual <= float(payload.tolerance):
            break

    norms = [
        _real(_environment_contraction(xp, env, layer))
        for env, layer in zip(environments, layers)
    ]
    norm = sum(norms) / max(1, len(norms))
    onsite_values: list[float] = []
    for term in payload.terms:
        site = next(iter(term.paulis), 0)
        if site >= len(tensors):
            raise ValueError("onsite term index exceeds the enabled CTMRG unit-cell tensors")
        onsite_values.append(_term_expectation(xp, environments[site], tensors[site], term))

    interaction_values: list[float | None] = []
    for interaction in payload.interactions:
        if len(tensors) == 1:
            value = _interaction_expectation(
                xp,
                environments[0],
                tensors[0],
                interaction.displacement,
                interaction.left_pauli,
                interaction.right_pauli,
            )
        else:
            value = _interaction_expectation_cell(
                xp,
                environments,
                tensors,
                interaction.left_site,
                interaction.right_site,
                interaction.displacement,
                interaction.left_pauli,
                interaction.right_pauli,
            )
        interaction_values.append(value)
    interaction_values_available = all(value is not None for value in interaction_values)
    energy = sum(float(term.coefficient) * value for term, value in zip(payload.terms, onsite_values))
    energy += sum(
        float(term.coefficient) * float(value)
        for term, value in zip(payload.interactions, interaction_values)
        if value is not None
    )
    reference_validation = finite_product_reference(
        payload,
        tensors,
        onsite_values,
        interaction_values,
        float(energy),
        tolerance=max(float(payload.tolerance) * 10.0, 1e-6),
    )
    converged = bool(residual <= float(payload.tolerance))
    result_method = (
        "ipeps-full-update-gradient-ctmrg" if payload.optimization == "full-update" and payload.full_update_optimizer == "finite-difference-gradient" else
        "ipeps-full-update-ctmrg" if payload.optimization == "full-update" else
        "ipeps-simple-update-ctmrg" if payload.optimization == "simple-update" else
        "ipeps-ctmrg-product-optimization" if optimization_info is not None else
        "ipeps-ctmrg-contraction"
    )
    warnings = [
        "compare environment_bond_dim and iteration convergence before using values as scientific conclusions",
    ]
    if len(tensors) == 1:
        warnings.insert(0, "CTMRG contraction uses a one-site translational environment")
    else:
        warnings.insert(0, "CTMRG contraction uses a periodic multi-site unit-cell environment")
    if payload.optimization == "simple-update":
        warnings.append("simple-update is an imaginary-time entangled-tensor baseline; compare it against the bounded full-update path before treating energies as variational evidence")
    elif payload.optimization == "full-update":
        if payload.full_update_optimizer == "finite-difference-gradient":
            warnings.append("finite-difference-gradient full-update is a bounded gradient estimate; it is not automatic differentiation and does not scale to large tensors")
        else:
            warnings.append("full-update re-evaluates CTMRG energy for bounded coordinate trials; it is not an automatic-differentiation optimizer")
    elif optimization_info is not None:
        warnings.append("product-coordinate-descent is a variational mean-field baseline with virtual_bond_dim=1; it is not an entangled iPEPS update")
    elif payload.tensor_data is None:
        warnings.append("the tensor is a deterministic product-state ansatz; no variational ground-state optimization was performed")
    else:
        warnings.append("the imported tensor was contracted without variational ground-state optimization")
    if not interaction_values_available:
        warnings.append("one or more interaction displacements are outside the supported nearest-neighbor two-site CTM contraction")
    if reference_validation["performed"] and not reference_validation["passed"]:
        warnings.append("finite product-supercell reference comparison exceeded its declared tolerance")
    elif not reference_validation["performed"]:
        warnings.append(f"independent finite product reference unavailable: {reference_validation['reason']}")
    checkpoint_result = checkpoint_info or {
        "resumable": False,
        "reason": (
            "set checkpoint_path to persist and resume the CTMRG environment"
            if len(tensors) == 1 else
            "set checkpoint_path to persist and resume the multi-site CTMRG environment"
        ),
    }
    limitations = [
        (
            "full-update is a bounded experimental optimization path and is not a scalable automatic-differentiation or full ground-state solver"
            if payload.optimization == "full-update" else
            "no environment-feedback full ground-state optimization"
        ),
        (
            "energy variance is a finite product-supercell diagnostic and is not an infinite-lattice variance proof"
            if reference_validation["performed"] else
            "energy variance and finite-product reference are unavailable for the current entangled tensor"
        ),
        "non-nearest interaction displacements are not yet supported by the two-site-RDM contraction",
    ]
    environment_diagnostics = _environment_diagnostics(xp, environments)
    research_result = ResearchResult(
        status="needs_review",
        method=result_method,
        representation="ipeps",
        metrics={
            "norm": norm,
            "energy": float(energy),
            "energy_complete": interaction_values_available,
            "residual": float(residual),
            "energy_second_moment": reference_validation.get("energy_second_moment"),
            "energy_variance": reference_validation.get("energy_variance"),
            "reference_energy_error": reference_validation.get("energy_error"),
            **environment_diagnostics,
        },
        truncation=TruncationReport(
            discarded_weight=float(discarded_total),
            max_bond_dim=int(payload.virtual_bond_dim),
            max_environment_dim=int(payload.environment_bond_dim),
        ),
        convergence=ConvergenceReport(
            converged=converged,
            criterion="normalized corner/edge environment residual",
            points=points,
            warnings=list(warnings),
        ),
        resources=_resource_summary(xp, environments, tensors, len(points), unit_cell),
        checkpoint=checkpoint_result,
        warnings=list(warnings),
        limitations=limitations,
        details={
            "initial_state": payload.initial_state,
            "environment_shapes": [[list(item.shape) for item in env.tensors()] for env in environments],
            "optimization": (
                {key: value for key, value in optimization_info.items() if key not in {"states", "tensors"}}
                | ({"optimized_state_vectors": _state_pairs(xp, optimization_info["states"])} if "states" in optimization_info else {})
                if optimization_info is not None else {"method": "none"}
            ),
            "correlation_lengths_by_site": environment_diagnostics["correlation_lengths_by_site"],
            "environment_spectrum": environment_diagnostics["environment_spectrum"],
            "reference_validation": reference_validation,
        },
    ).to_dict()
    return {
        "status": "done",
        "backend": "tensor-network-ctmrg",
        "method": result_method,
        "representation": "ipeps",
        "unit_cell": unit_cell,
        "unit_cell_sites": len(tensors),
        "tensor_source": (
            "optimized-product-state" if optimization_info is not None else
            ("imported" if payload.tensor_data is not None else "generated-product-ansatz")
        ),
        "optimization": payload.optimization,
        "optimization_diagnostics": (
            {key: value for key, value in optimization_info.items() if key not in {"states", "tensors"}}
            | ({"optimized_state_vectors": _state_pairs(xp, optimization_info["states"])} if "states" in optimization_info else {})
            if optimization_info is not None else None
        ),
        "physical_bond_dim": int(payload.physical_bond_dim),
        "virtual_bond_dim": int(payload.virtual_bond_dim),
        "environment_bond_dim_requested": int(payload.environment_bond_dim),
        "environment_bond_dim_used": max(int(env.C1.shape[0]) for env in environments),
        "iterations": len(points),
        "converged": converged,
        "residual": float(residual),
        "norm": norm,
        "energy": float(energy),
        "energy_complete": interaction_values_available,
        "energy_second_moment": reference_validation.get("energy_second_moment"),
        "energy_variance": reference_validation.get("energy_variance"),
        "reference_validation": reference_validation,
        "correlation_length": environment_diagnostics["correlation_length"],
        "correlation_lengths_by_site": environment_diagnostics["correlation_lengths_by_site"],
        "environment_spectrum": environment_diagnostics["environment_spectrum"],
        "observables": structured_observables(payload.terms, onsite_values),
        "interactions": [
            {
                "index": index,
                "label": interaction.label or f"{interaction.left_pauli}{interaction.left_site}-{interaction.right_pauli}{interaction.right_site}",
                "coefficient": float(interaction.coefficient),
                "value": float(value) if value is not None else None,
                "displacement": list(interaction.displacement),
                "left_site": int(interaction.left_site),
                "right_site": int(interaction.right_site),
            }
            for index, (interaction, value) in enumerate(zip(payload.interactions, interaction_values))
        ],
        "resource_estimate": _resource_summary(xp, environments, tensors, len(points), unit_cell),
        "research_result": research_result,
        "checkpoint": checkpoint_result,
        "warnings": warnings,
        "time_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def run_ctmrg_convergence_study(
    xp: Any,
    payload: CTMRGPayload,
    environment_bond_dims: list[int],
) -> dict[str, Any]:
    """Run a bounded environment-dimension convergence study.

    Each point is an independent contraction from the same tensor ansatz.  The
    study intentionally disables optimization and checkpoint reuse so that an
    energy delta reflects the requested environment dimension rather than a
    hidden optimizer or a partially evolved state.  This is a diagnostic
    helper, not a claim of thermodynamic convergence.
    """

    if not environment_bond_dims:
        raise ValueError("environment_bond_dims must contain at least one value")
    if len(environment_bond_dims) > 8:
        raise ValueError("environment_bond_dims is limited to eight bounded study points")
    normalized_dims = [int(value) for value in environment_bond_dims]
    if any(value < 1 or value > 128 for value in normalized_dims):
        raise ValueError("environment_bond_dims values must be between 1 and 128")
    if len(set(normalized_dims)) != len(normalized_dims):
        raise ValueError("environment_bond_dims values must be unique")
    if payload.optimization != "none":
        raise ValueError("CTMRG convergence studies require optimization='none'")

    points: list[dict[str, Any]] = []
    previous_energy: float | None = None
    for environment_bond_dim in normalized_dims:
        point_payload = payload.model_copy(update={
            "environment_bond_dim": environment_bond_dim,
            "checkpoint_path": None,
            "resume_from": None,
        })
        result = run_ctmrg(xp, point_payload)
        energy = float(result["energy"])
        points.append({
            "environment_bond_dim": environment_bond_dim,
            "environment_bond_dim_used": int(result["environment_bond_dim_used"]),
            "energy": energy,
            "energy_complete": bool(result["energy_complete"]),
            "energy_delta": None if previous_energy is None else energy - previous_energy,
            "residual": float(result["residual"]),
            "converged": bool(result["converged"]),
            "correlation_length": result["correlation_length"],
            "correlation_lengths_by_site": result["correlation_lengths_by_site"],
            "environment_spectrum": result["environment_spectrum"],
            "energy_second_moment": result["energy_second_moment"],
            "energy_variance": result["energy_variance"],
            "reference_validation": result["reference_validation"],
            "resource_estimate": result["resource_estimate"],
        })
        previous_energy = energy

    return {
        "status": "done",
        "method": "ipeps-ctmrg-environment-convergence-study",
        "optimization": "none",
        "unit_cell": list(payload.unit_cell),
        "unit_cell_sites": math.prod(payload.unit_cell),
        "points": points,
        "materializes_statevector": False,
        "warnings": [
            "points are independent bounded CTMRG contractions from the same tensor ansatz",
            "compare energy, residual, correlation length, and local observables together before drawing physical conclusions",
        ],
    }
