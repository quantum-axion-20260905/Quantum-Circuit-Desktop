"""Bounded iPEPS corner-transfer-matrix contraction.

This module deliberately owns only the numerical representation.  Admission,
HTTP lifecycle, provenance, and domain builders stay outside the module so
larger unit cells and domain-specific optimizers can reuse the same result
seam.

The base solver supports one-site, checkerboard, and small periodic
unit-cell environments. Optional optimizer paths are deliberately explicit:
the finite-torus gradient mode optimizes a bounded four-site reference, while
the other paths remain experimental. Every result therefore retains the
``needs_review`` distinction until the declared infinite-lattice evidence is
available.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Iterable

from ..plugins.models import CTMRGPayload, PauliTerm
from ..provenance import sha256_json
from .checkpoints import load_ctm_checkpoint, save_ctm_checkpoint
from .contracts import CheckpointManifest, ConvergencePoint, ConvergenceReport, ResearchResult, TruncationReport
from .ctmrg_admission import ctmrg_research_gate
from .ctmrg_reference import analytic_ghz_reference, finite_periodic_peps_reference, finite_product_reference
from .ctmrg_gauge import pairwise_virtual_gauge_preconditioner, virtual_leg_conditioning_report
from .ctmrg_projectors import (
    full_svd_projectors,
    standard_full_svd_projectors,
    swap_fused_pair_columns,
    swap_fused_pair_rows,
)
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
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach()
    cpu = getattr(value, "cpu", None)
    if cpu is not None:
        value = cpu()
    try:
        return value.get()
    except AttributeError:
        return value


def _copy(value: Any) -> Any:
    """Copy NumPy/CuPy arrays and preserve graph semantics for torch tensors."""

    clone = getattr(value, "clone", None)
    if clone is not None:
        return clone()
    return value.copy()


def _real(value: Any) -> float:
    return float(complex(_host(value)).real)


def _device_name(xp: Any, value: Any | None = None) -> str:
    device = getattr(value, "device", None)
    device_type = getattr(device, "type", None)
    if device_type in {"cpu", "cuda", "mps"}:
        return str(device_type)
    return "cuda" if hasattr(xp, "cuda") else "cpu"


def _state_pairs(xp: Any, states: list[Any]) -> list[list[list[float]]]:
    return [
        [[float(complex(value).real), float(complex(value).imag)] for value in _host(state).reshape(-1)]
        for state in states
    ]


def _max_abs(xp: Any, value: Any) -> float:
    return float(_host(xp.max(xp.abs(value))))


def _flip(xp: Any, value: Any, axis: int) -> Any:
    """Flip an axis across NumPy, CuPy, and torch namespace conventions."""

    try:
        return xp.flip(value, axis=axis)
    except TypeError:
        return xp.flip(value, dims=(axis,))


def _transpose(xp: Any, value: Any, axes: tuple[int, ...]) -> Any:
    """Transpose across NumPy/CuPy and torch array conventions."""

    if getattr(xp, "__name__", "") == "torch":
        return value.permute(*axes)
    return value.transpose(*axes)


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


def _pauli_matrix(xp: Any, dtype: Any, label: str, device: Any | None = None) -> Any:
    kwargs = {"dtype": dtype}
    if getattr(xp, "__name__", "") == "torch" and device is not None:
        kwargs["device"] = device
    def matrix(values: list[list[Any]]) -> Any:
        if getattr(xp, "__name__", "") == "torch":
            return xp.tensor(values, **kwargs)
        return xp.asarray(values, dtype=dtype)

    if label == "I":
        return xp.eye(2, **kwargs)
    if label == "X":
        return matrix([[0, 1], [1, 0]])
    if label == "Y":
        return matrix([[0, -1j], [1j, 0]])
    if label == "Z":
        return matrix([[1, 0], [0, -1]])
    raise ValueError(f"unsupported Pauli operator {label!r}")


def _double_layer(xp: Any, tensor: Any, operator: Any | None = None) -> Any:
    """Fuse ket/bra virtual legs while contracting the physical leg."""

    physical = tensor.shape[0]
    dtype = tensor.dtype
    if operator is not None:
        op = operator
    elif getattr(xp, "__name__", "") == "torch":
        op = xp.eye(physical, dtype=dtype, device=tensor.device)
    else:
        op = xp.eye(physical, dtype=dtype)
    raw = xp.einsum(
        "sudlr,st,tUDLR->uUdDlLrR",
        tensor,
        op,
        xp.conj(tensor),
    )
    d2 = tensor.shape[1] * tensor.shape[1]
    return raw.reshape((d2, d2, d2, d2))


def _initialize_environment(
    xp: Any,
    double_layer: Any,
    chi: int,
    *,
    sector_seed: int | None = None,
) -> CTMEnvironment:
    """Create a deterministic, full-support CTM boundary.

    A diagonal-only boundary is sufficient for product tensors, but it can
    select a zero-overlap sector when the transfer operator has degenerate
    fixed points (for example a GHZ-like D=2 iPEPS).  A tiny positive
    full-support component regularizes that initialization without changing
    the normalized fixed point; it also prevents an otherwise valid two-site
    observable from becoming an accidental ``0/0`` contraction.
    """
    d2 = int(double_layer.shape[0])
    dtype = double_layer.dtype
    if getattr(xp, "__name__", "") == "torch":
        corner = xp.zeros((chi, chi), dtype=dtype, device=double_layer.device)
        edge = xp.zeros((chi, d2, chi), dtype=dtype, device=double_layer.device)
    else:
        corner = xp.zeros((chi, chi), dtype=dtype)
        edge = xp.zeros((chi, d2, chi), dtype=dtype)
    diagonal = min(chi, d2)
    if sector_seed is None:
        corner[xp.arange(diagonal), xp.arange(diagonal)] = 1
        for index in range(min(chi, d2)):
            edge[index, :, index] = 1
    else:
        # Two deterministic, full-support boundary probes are used by the
        # opt-in symmetry-sector ensemble.  They bias different dominant
        # transfer sectors without importing a backend RNG or mutating global
        # random state.  The ensemble averages the resulting fixed points;
        # this is useful for degenerate GHZ-like transfer spectra but remains
        # experimental until generic symmetry gates pass.
        def arange(count: int) -> Any:
            return (
                xp.arange(count, device=double_layer.device)
                if getattr(xp, "__name__", "") == "torch" else
                xp.arange(count)
            )

        boundary_index = arange(chi)
        edge_index = arange(d2)
        seed = int(sector_seed)
        weights = xp.exp(
            0.8 * xp.sin((boundary_index + 1) * (seed + 1) * 1.17)
            + 0.25 * xp.cos((boundary_index + 1) * (seed + 2))
        )
        if getattr(xp, "__name__", "") == "torch":
            weights = weights.to(dtype=dtype)
        else:
            weights = xp.asarray(weights, dtype=dtype)
        corner_indices = arange(diagonal)
        corner[corner_indices, corner_indices] = weights[:diagonal]
        for index in range(min(chi, d2)):
            edge_values = (
                1.0
                + 0.35 * xp.sin((edge_index + 1) * (index + 1 + seed) * 0.91)
                + 0.1 * xp.cos(edge_index + seed + 1)
            )
            if getattr(xp, "__name__", "") == "torch":
                edge_values = edge_values.to(dtype=dtype)
            else:
                edge_values = xp.asarray(edge_values, dtype=dtype)
            edge[index, :, index] = edge_values
    # Keep a deterministic non-zero overlap with every boundary sector.  The
    # floor is deliberately shared by complex64 and complex128: a much
    # smaller complex128 perturbation lets degenerate SVD sectors choose a
    # different boundary branch on the first sweep, which can turn the
    # canonical D=2 GHZ fixed point into a false symmetry-broken result.
    regularizer = 1e-6
    corner = corner + regularizer * xp.ones_like(corner)
    if sector_seed is None:
        edge = edge + regularizer * xp.ones_like(edge)
    return CTMEnvironment(corner, _copy(corner), _copy(corner), _copy(corner), edge, _copy(edge), _copy(edge), _copy(edge))


def _ctm_move(
    xp: Any,
    left_corner: Any,
    right_corner: Any,
    grown_edge: Any,
    chi: int,
    differentiate_truncation: bool = True,
) -> tuple[Any, Any, Any, float]:
    """Truncate one enlarged boundary with a Hermitian half-system projector."""

    rho = left_corner @ xp.conj(left_corner).T + right_corner @ xp.conj(right_corner).T
    rho = 0.5 * (rho + xp.conj(rho).T)
    eigenvalues, eigenvectors = xp.linalg.eigh(rho)
    keep = min(int(chi), int(eigenvalues.shape[0]))
    projector = eigenvectors[:, -keep:]
    projector = _flip(xp, projector, axis=1)
    if getattr(xp, "__name__", "") == "torch" and not differentiate_truncation:
        # Complex Hermitian eigenspaces carry an arbitrary phase.  A backward
        # pass through a truncated eigenbasis is therefore ill-defined near
        # degeneracies; the implicit research path differentiates the local
        # contraction with the current projector held fixed and reports that
        # approximation explicitly.
        projector = projector.detach()
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


def _left_move(
    xp: Any,
    env: CTMEnvironment,
    double_layer: Any,
    chi: int,
    differentiate_truncation: bool = True,
    tensor: Any | None = None,
    projector_method: str = "half-density",
) -> tuple[CTMEnvironment, float]:
    d2 = int(double_layer.shape[0])
    c1_g = xp.einsum("ab,buc->auc", env.C1, env.T1).reshape(-1, env.T1.shape[2])
    c4_g = xp.einsum("gh,hdi->gdi", env.C4, env.T3).reshape(-1, env.T3.shape[2])
    t4_g = xp.einsum("alg,udlr->augdr", env.T4, double_layer)
    t4_g = _transpose(xp, t4_g, (0, 1, 4, 2, 3)).reshape(c1_g.shape[0], d2, c4_g.shape[0])
    if projector_method == "full-svd":
        if tensor is None:
            raise ValueError("full-svd CTMRG projectors require resident raw tensors")
        projector, dual_projector, discarded = standard_full_svd_projectors(
            xp,
            env,
            tensor,
            chi,
            "left",
            differentiate_truncation=differentiate_truncation,
        )
        virtual = int(tensor.shape[1])
        boundary = int(env.C1.shape[0])
        kept = int(projector.shape[1])
        projector = projector.reshape(boundary, virtual, virtual, kept)
        dual_projector = dual_projector.reshape(boundary, virtual, virtual, kept)
        t1_split = env.T1.reshape(boundary, virtual, virtual, boundary)
        t3_split = env.T3.reshape(boundary, virtual, virtual, boundary)
        t4_split = env.T4.reshape(boundary, virtual, virtual, boundary)
        double_layer_split = double_layer.reshape(
            virtual, virtual, virtual, virtual, virtual, virtual, virtual, virtual
        )
        c1 = xp.einsum("auUp,ab,buUc->pc", dual_projector, env.C1, t1_split)
        c4 = xp.einsum("gdDq,hg,hdDi->qi", projector, env.C4, t3_split)
        t4 = xp.einsum(
            "alLg,uUdDlLrR,auUp,gdDq->prRq",
            t4_split,
            double_layer_split,
            dual_projector,
            projector,
        ).reshape(kept, virtual * virtual, kept)
        return CTMEnvironment(c1, env.C2, env.C3, c4, env.T1, env.T2, env.T3, t4), discarded
    c1, c4, t4, discarded = _ctm_move(xp, c1_g, c4_g, t4_g, chi, differentiate_truncation)
    return CTMEnvironment(c1, env.C2, env.C3, c4, env.T1, env.T2, env.T3, t4), discarded


def _right_move(
    xp: Any,
    env: CTMEnvironment,
    double_layer: Any,
    chi: int,
    differentiate_truncation: bool = True,
    tensor: Any | None = None,
    projector_method: str = "half-density",
) -> tuple[CTMEnvironment, float]:
    d2 = int(double_layer.shape[0])
    c2_g = xp.einsum("ce,buc->eub", env.C2, env.T1).reshape(-1, env.T1.shape[0])
    c3_g = xp.einsum("im,hdi->mdh", env.C3, env.T3).reshape(-1, env.T3.shape[0])
    t2_g = xp.einsum("erm,udlr->eumdl", env.T2, double_layer)
    t2_g = _transpose(xp, t2_g, (0, 1, 4, 2, 3)).reshape(c2_g.shape[0], d2, c3_g.shape[0])
    if projector_method == "full-svd":
        if tensor is None:
            raise ValueError("full-svd CTMRG projectors require resident raw tensors")
        projector, dual_projector, discarded = standard_full_svd_projectors(
            xp,
            env,
            tensor,
            chi,
            "right",
            differentiate_truncation=differentiate_truncation,
        )
        virtual = int(tensor.shape[1])
        boundary = int(env.C1.shape[0])
        kept = int(projector.shape[1])
        projector = projector.reshape(boundary, virtual, virtual, kept)
        dual_projector = dual_projector.reshape(boundary, virtual, virtual, kept)
        t1_split = env.T1.reshape(boundary, virtual, virtual, boundary)
        t3_split = env.T3.reshape(boundary, virtual, virtual, boundary)
        t2_split = env.T2.reshape(boundary, virtual, virtual, boundary)
        double_layer_split = double_layer.reshape(
            virtual, virtual, virtual, virtual, virtual, virtual, virtual, virtual
        )
        c2 = xp.einsum("euUp,ce,buUc->pb", projector, env.C2, t1_split)
        c3 = xp.einsum("mdDq,im,hdDi->qh", dual_projector, env.C3, t3_split)
        t2 = xp.einsum(
            "crRe,uUdDlLrR,euUp,mdDq->prRq",
            t2_split,
            double_layer_split,
            projector,
            dual_projector,
        ).reshape(kept, virtual * virtual, kept)
        return CTMEnvironment(env.C1, c2, c3, env.C4, env.T1, t2, env.T3, env.T4), discarded
    c2, c3, t2, discarded = _ctm_move(xp, c2_g, c3_g, t2_g, chi, differentiate_truncation)
    return CTMEnvironment(env.C1, c2, c3, env.C4, env.T1, t2, env.T3, env.T4), discarded


def _top_move(
    xp: Any,
    env: CTMEnvironment,
    double_layer: Any,
    chi: int,
    differentiate_truncation: bool = True,
    tensor: Any | None = None,
    projector_method: str = "half-density",
) -> tuple[CTMEnvironment, float]:
    d2 = int(double_layer.shape[0])
    c1_g = xp.einsum("ab,alg->blg", env.C1, env.T4).reshape(-1, env.T4.shape[2])
    c2_g = xp.einsum("ce,erm->crm", env.C2, env.T2).reshape(-1, env.T2.shape[2])
    t1_g = xp.einsum("buc,udlr->bcdlr", env.T1, double_layer)
    t1_g = _transpose(xp, t1_g, (0, 3, 2, 1, 4)).reshape(c1_g.shape[0], d2, c2_g.shape[0])
    if projector_method == "full-svd":
        if tensor is None:
            raise ValueError("full-svd CTMRG projectors require resident raw tensors")
        projector, dual_projector, discarded = standard_full_svd_projectors(
            xp,
            env,
            tensor,
            chi,
            "top",
            differentiate_truncation=differentiate_truncation,
        )
        virtual = int(tensor.shape[1])
        boundary = int(env.C1.shape[0])
        kept = int(projector.shape[1])
        projector = projector.reshape(boundary, virtual, virtual, kept)
        dual_projector = dual_projector.reshape(boundary, virtual, virtual, kept)
        t2_split = env.T2.reshape(boundary, virtual, virtual, boundary)
        t4_split = env.T4.reshape(boundary, virtual, virtual, boundary)
        t1_split = env.T1.reshape(boundary, virtual, virtual, boundary)
        double_layer_split = double_layer.reshape(
            virtual, virtual, virtual, virtual, virtual, virtual, virtual, virtual
        )
        c1 = xp.einsum("blLp,ab,alLg->pg", projector, env.C1, t4_split)
        c2 = xp.einsum("crRq,ce,erRm->qm", dual_projector, env.C2, t2_split)
        t1 = xp.einsum(
            "auUc,uUdDlLrR,alLp,crRq->pdDq",
            t1_split,
            double_layer_split,
            projector,
            dual_projector,
        ).reshape(kept, virtual * virtual, kept)
        return CTMEnvironment(c1, c2, env.C3, env.C4, t1, env.T2, env.T3, env.T4), discarded
    c1, c2, t1, discarded = _ctm_move(xp, c1_g, c2_g, t1_g, chi, differentiate_truncation)
    return CTMEnvironment(c1, c2, env.C3, env.C4, t1, env.T2, env.T3, env.T4), discarded


def _bottom_move(
    xp: Any,
    env: CTMEnvironment,
    double_layer: Any,
    chi: int,
    differentiate_truncation: bool = True,
    tensor: Any | None = None,
    projector_method: str = "half-density",
) -> tuple[CTMEnvironment, float]:
    d2 = int(double_layer.shape[0])
    c4_g = _transpose(xp, xp.einsum("gh,alg->hal", env.C4, env.T4), (0, 2, 1)).reshape(-1, env.T4.shape[0])
    c3_g = xp.einsum("im,erm->ire", env.C3, env.T2).reshape(-1, env.T2.shape[0])
    t3_g = xp.einsum("hdi,udlr->hiulr", env.T3, double_layer)
    t3_g = _transpose(xp, t3_g, (0, 3, 2, 1, 4)).reshape(c4_g.shape[0], d2, c3_g.shape[0])
    if projector_method == "full-svd":
        if tensor is None:
            raise ValueError("full-svd CTMRG projectors require resident raw tensors")
        projector, dual_projector, discarded = standard_full_svd_projectors(
            xp,
            env,
            tensor,
            chi,
            "bottom",
            differentiate_truncation=differentiate_truncation,
        )
        virtual = int(tensor.shape[1])
        boundary = int(env.C1.shape[0])
        kept = int(projector.shape[1])
        projector = projector.reshape(boundary, virtual, virtual, kept)
        dual_projector = dual_projector.reshape(boundary, virtual, virtual, kept)
        t2_split = env.T2.reshape(boundary, virtual, virtual, boundary)
        t4_split = env.T4.reshape(boundary, virtual, virtual, boundary)
        t3_split = env.T3.reshape(boundary, virtual, virtual, boundary)
        double_layer_split = double_layer.reshape(
            virtual, virtual, virtual, virtual, virtual, virtual, virtual, virtual
        )
        c4 = xp.einsum("hlLp,gh,alLg->pg", dual_projector, env.C4, t4_split)
        c3 = xp.einsum("irRq,im,erRm->qe", projector, env.C3, t2_split)
        t3 = xp.einsum(
            "hdDi,uUdDlLrR,hlLp,irRq->puUq",
            t3_split,
            double_layer_split,
            dual_projector,
            projector,
        ).reshape(kept, virtual * virtual, kept)
        return CTMEnvironment(env.C1, env.C2, c3, c4, env.T1, env.T2, t3, env.T4), discarded
    c4, c3, t3, discarded = _ctm_move(xp, c4_g, c3_g, t3_g, chi, differentiate_truncation)
    return CTMEnvironment(env.C1, env.C2, c3, c4, env.T1, env.T2, t3, env.T4), discarded


def _left_move_two_site(
    xp: Any,
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    neighbor_layer: Any,
    chi: int,
    differentiate_truncation: bool = True,
    neighbor_tensor: Any | None = None,
    projector_method: str = "half-density",
) -> tuple[CTMEnvironment, float]:
    d2 = int(neighbor_layer.shape[0])
    c1_g = xp.einsum("ab,buc->auc", env_self.C1, env_neighbor.T1).reshape(-1, env_neighbor.T1.shape[2])
    c4_g = xp.einsum("gh,hdi->gdi", env_self.C4, env_neighbor.T3).reshape(-1, env_neighbor.T3.shape[2])
    t4_g = xp.einsum("alg,udlr->augdr", env_self.T4, neighbor_layer)
    t4_g = _transpose(xp, t4_g, (0, 1, 4, 2, 3)).reshape(c1_g.shape[0], d2, c4_g.shape[0])
    if projector_method == "full-svd":
        if neighbor_tensor is None:
            raise ValueError("full-svd CTMRG projectors require resident raw tensors")
        left_projector, right_projector, discarded = full_svd_projectors(
            xp,
            env_self,
            neighbor_tensor,
            chi,
            "left",
            differentiate_truncation=differentiate_truncation,
        )
        c1 = left_projector @ c1_g
        c4 = xp.conj(right_projector).T @ c4_g
        t4 = xp.einsum("ai,idj,jb->adb", left_projector, t4_g, right_projector)
        return CTMEnvironment(c1, env_self.C2, env_self.C3, c4, env_self.T1, env_self.T2, env_self.T3, t4), discarded
    c1, c4, t4, discarded = _ctm_move(xp, c1_g, c4_g, t4_g, chi, differentiate_truncation)
    return CTMEnvironment(c1, env_self.C2, env_self.C3, c4, env_self.T1, env_self.T2, env_self.T3, t4), discarded


def _right_move_two_site(
    xp: Any,
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    neighbor_layer: Any,
    chi: int,
    differentiate_truncation: bool = True,
    neighbor_tensor: Any | None = None,
    projector_method: str = "half-density",
) -> tuple[CTMEnvironment, float]:
    d2 = int(neighbor_layer.shape[0])
    c2_g = xp.einsum("ce,buc->eub", env_self.C2, env_neighbor.T1).reshape(-1, env_neighbor.T1.shape[0])
    c3_g = xp.einsum("im,hdi->mdh", env_self.C3, env_neighbor.T3).reshape(-1, env_neighbor.T3.shape[0])
    t2_g = xp.einsum("erm,udlr->eumdl", env_self.T2, neighbor_layer)
    t2_g = _transpose(xp, t2_g, (0, 1, 4, 2, 3)).reshape(c2_g.shape[0], d2, c3_g.shape[0])
    if projector_method == "full-svd":
        if neighbor_tensor is None:
            raise ValueError("full-svd CTMRG projectors require resident raw tensors")
        left_projector, right_projector, discarded = full_svd_projectors(
            xp,
            env_self,
            neighbor_tensor,
            chi,
            "right",
            differentiate_truncation=differentiate_truncation,
        )
        c2 = left_projector @ c2_g
        c3 = xp.conj(right_projector).T @ c3_g
        t2 = xp.einsum("ai,idj,jb->adb", left_projector, t2_g, right_projector)
        return CTMEnvironment(env_self.C1, c2, c3, env_self.C4, env_self.T1, t2, env_self.T3, env_self.T4), discarded
    c2, c3, t2, discarded = _ctm_move(xp, c2_g, c3_g, t2_g, chi, differentiate_truncation)
    return CTMEnvironment(env_self.C1, c2, c3, env_self.C4, env_self.T1, t2, env_self.T3, env_self.T4), discarded


def _top_move_two_site(
    xp: Any,
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    neighbor_layer: Any,
    chi: int,
    differentiate_truncation: bool = True,
    neighbor_tensor: Any | None = None,
    projector_method: str = "half-density",
) -> tuple[CTMEnvironment, float]:
    d2 = int(neighbor_layer.shape[0])
    c1_g = xp.einsum("ab,alg->blg", env_self.C1, env_neighbor.T4).reshape(-1, env_neighbor.T4.shape[2])
    c2_g = xp.einsum("ce,erm->crm", env_self.C2, env_neighbor.T2).reshape(-1, env_neighbor.T2.shape[2])
    t1_g = xp.einsum("buc,udlr->bcdlr", env_self.T1, neighbor_layer)
    t1_g = _transpose(xp, t1_g, (0, 3, 2, 1, 4)).reshape(c1_g.shape[0], d2, c2_g.shape[0])
    if projector_method == "full-svd":
        if neighbor_tensor is None:
            raise ValueError("full-svd CTMRG projectors require resident raw tensors")
        left_projector, right_projector, discarded = full_svd_projectors(
            xp,
            env_self,
            neighbor_tensor,
            chi,
            "top",
            differentiate_truncation=differentiate_truncation,
        )
        c1 = xp.conj(right_projector).T @ c1_g
        c2 = left_projector @ c2_g
        t1 = xp.einsum("ai,idj,jb->adb", xp.conj(right_projector).T, t1_g, xp.conj(left_projector).T)
        return CTMEnvironment(c1, c2, env_self.C3, env_self.C4, t1, env_self.T2, env_self.T3, env_self.T4), discarded
    c1, c2, t1, discarded = _ctm_move(xp, c1_g, c2_g, t1_g, chi, differentiate_truncation)
    return CTMEnvironment(c1, c2, env_self.C3, env_self.C4, t1, env_self.T2, env_self.T3, env_self.T4), discarded


def _bottom_move_two_site(
    xp: Any,
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    neighbor_layer: Any,
    chi: int,
    differentiate_truncation: bool = True,
    neighbor_tensor: Any | None = None,
    projector_method: str = "half-density",
) -> tuple[CTMEnvironment, float]:
    d2 = int(neighbor_layer.shape[0])
    c4_g = _transpose(xp, xp.einsum("gh,alg->hal", env_self.C4, env_neighbor.T4), (0, 2, 1)).reshape(-1, env_neighbor.T4.shape[0])
    c3_g = xp.einsum("im,erm->ire", env_self.C3, env_neighbor.T2).reshape(-1, env_neighbor.T2.shape[0])
    t3_g = xp.einsum("hdi,udlr->hiulr", env_self.T3, neighbor_layer)
    t3_g = _transpose(xp, t3_g, (0, 3, 2, 1, 4)).reshape(c4_g.shape[0], d2, c3_g.shape[0])
    if projector_method == "full-svd":
        if neighbor_tensor is None:
            raise ValueError("full-svd CTMRG projectors require resident raw tensors")
        left_projector, right_projector, discarded = full_svd_projectors(
            xp,
            env_self,
            neighbor_tensor,
            chi,
            "bottom",
            differentiate_truncation=differentiate_truncation,
        )
        c4 = right_projector @ c4_g
        c3 = xp.conj(left_projector).T @ c3_g
        t3 = xp.einsum("ai,idj,jb->adb", right_projector, t3_g, left_projector)
        return CTMEnvironment(env_self.C1, env_self.C2, c3, c4, env_self.T1, env_self.T2, t3, env_self.T4), discarded
    c4, c3, t3, discarded = _ctm_move(xp, c4_g, c3_g, t3_g, chi, differentiate_truncation)
    return CTMEnvironment(env_self.C1, env_self.C2, c3, c4, env_self.T1, env_self.T2, t3, env_self.T4), discarded


def _unit_cell_sweep(
    xp: Any,
    environments: list[CTMEnvironment],
    layers: list[Any],
    chi: int,
    unit_cell: list[int],
    renormalize: Any = None,
    differentiate_truncation: bool = True,
    tensors: list[Any] | None = None,
    projector_method: str = "half-density",
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
    if projector_method == "full-svd" and (tensors is None or len(tensors) != nx * ny):
        raise ValueError("full-svd CTMRG projectors require resident raw tensors for every unit-cell site")

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
            working[site], discarded = move(
                xp,
                working[site],
                working[neighbor],
                layers[neighbor],
                chi,
                differentiate_truncation,
                None if tensors is None else tensors[neighbor],
                projector_method,
            )
            discarded_total += discarded
    normalize = renormalize or _renormalize
    return [normalize(xp, env) for env in working], discarded_total


def _renormalize(xp: Any, env: CTMEnvironment) -> CTMEnvironment:
    def normalize(value: Any) -> Any:
        return value / (_max_abs(xp, value) + 1e-30)

    return CTMEnvironment(*(normalize(value) for value in env.tensors()))


def _raw_environment_residual(xp: Any, before: CTMEnvironment, after: CTMEnvironment) -> float:
    residual = 0.0
    for old, new in zip(before.tensors(), after.tensors()):
        scale = max(_max_abs(xp, old), 1e-30)
        residual = max(residual, _max_abs(xp, new - old) / scale)
    return residual


def _environment_residual(xp: Any, before: CTMEnvironment, after: CTMEnvironment) -> float:
    """Compare environments modulo their internal boundary-basis gauge.

    CTM truncation can rotate or rephase the retained boundary basis even
    after the physical environment has converged.  Comparing raw tensor
    entries therefore reports false residuals near two.  Singular spectra of
    each corner/edge flattening are invariant under those retained-basis
    unitaries and provide a conservative bounded convergence diagnostic.
    """

    def spectrum(value: Any) -> list[float]:
        matrix = value.reshape(value.shape[0], -1)
        try:
            if getattr(xp, "__name__", "") == "torch":
                singular = xp.linalg.svdvals(matrix)
                host = singular.detach().cpu().tolist()
            else:
                singular = xp.linalg.svd(matrix, compute_uv=False)
                host = _host(singular)
        except Exception:
            import numpy as np

            if getattr(xp, "__name__", "") == "torch":
                host = np.linalg.svd(matrix.detach().cpu().numpy(), compute_uv=False)
            else:
                host = np.linalg.svd(_host(matrix), compute_uv=False)
        values = [float(abs(item)) for item in host]
        scale = max(values[0] if values else 0.0, 1e-30)
        return [item / scale for item in values]

    residual = 0.0
    for old, new in zip(before.tensors(), after.tensors()):
        old_spectrum = spectrum(old)
        new_spectrum = spectrum(new)
        size = max(len(old_spectrum), len(new_spectrum))
        old_spectrum.extend([0.0] * (size - len(old_spectrum)))
        new_spectrum.extend([0.0] * (size - len(new_spectrum)))
        residual = max(
            residual,
            max(abs(left - right) for left, right in zip(old_spectrum, new_spectrum)),
        )
    return residual


def _blend_environment(
    before: CTMEnvironment,
    candidate: CTMEnvironment,
    damping: float,
) -> CTMEnvironment:
    """Under-relax one CTM fixed-point update in its resident backend."""

    alpha = float(damping)
    if alpha >= 1.0:
        return candidate
    one_minus = 1.0 - alpha
    return CTMEnvironment(*(
        one_minus * old + alpha * new
        for old, new in zip(before.tensors(), candidate.tensors())
    ))


def _blend_environments(
    xp: Any,
    before: list[CTMEnvironment],
    candidates: list[CTMEnvironment],
    damping: float,
) -> list[CTMEnvironment]:
    """Apply the same bounded damping to every periodic cell environment."""

    return [
        _renormalize(xp, _blend_environment(old, new, damping))
        for old, new in zip(before, candidates)
    ]


def _environment_diagnostics(xp: Any, environments: list[CTMEnvironment]) -> dict[str, Any]:
    """Extract bounded transfer-spectrum diagnostics from converged edges."""

    def eigenvalues(value: Any) -> Any:
        try:
            return xp.linalg.eigvals(value)
        except Exception:
            # Torch's optional CUDA runtime and CuPy can expose different
            # cuSOLVER symbols in the same Windows process.  Diagnostics are
            # small boundary matrices, so a host fallback keeps the primary
            # contraction on GPU without making the result unavailable.
            import numpy as np

            return np.linalg.eigvals(_host(value))

    def singular_values(value: Any) -> Any:
        try:
            return xp.linalg.svd(value, compute_uv=False)
        except Exception:
            import numpy as np

            return np.linalg.svd(_host(value), compute_uv=False)

    spectra: list[list[float]] = []
    correlation_lengths: list[float | None] = []
    for env in environments:
        transfer = xp.sum(env.T1, axis=1)
        transfer_eigenvalues = eigenvalues(transfer)
        magnitudes = sorted((float(abs(value)) for value in _host(transfer_eigenvalues)), reverse=True)
        leading = magnitudes[0] if magnitudes else 0.0
        subleading = magnitudes[1] if len(magnitudes) > 1 else 0.0
        if leading <= 1e-30 or subleading <= 1e-30:
            correlation_lengths.append(0.0)
        else:
            ratio = max(0.0, subleading / leading)
            # A degenerate leading transfer eigenvalue has no finite
            # correlation length.  Returning a huge finite sentinel is
            # misleading and can be mistaken for a measured scale.
            # Below this gap the bounded environment cannot resolve a
            # reliable finite length; report it as unresolved instead of
            # turning round-off into a giant scientific number.
            if ratio >= 1.0 - 1e-5:
                correlation_lengths.append(None)
            else:
                correlation_lengths.append(float(-1.0 / math.log(ratio)) if ratio > 0 else 0.0)
        singular = singular_values(env.T1.reshape(env.T1.shape[0], -1))
        singular_host = [float(abs(value)) for value in _host(singular)]
        scale = max(singular_host[0] if singular_host else 0.0, 1e-30)
        spectra.append([value / scale for value in singular_host])
    finite_lengths = [value for value in correlation_lengths if value is not None and math.isfinite(value)]
    return {
        "correlation_length": None if any(value is None for value in correlation_lengths) else (
            max(finite_lengths) if finite_lengths else None
        ),
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
    op = _pauli_matrix(xp, dtype, str(pauli), getattr(tensor, "device", None))
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
    left_operator = _pauli_matrix(xp, dtype, left_pauli, getattr(tensor, "device", None))
    right_operator = _pauli_matrix(xp, dtype, right_pauli, getattr(tensor, "device", None))
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
    left_layer = _double_layer(xp, left_tensor, _pauli_matrix(xp, dtype, left_pauli, getattr(left_tensor, "device", None)))
    right_layer = _double_layer(xp, right_tensor, _pauli_matrix(xp, dtype, right_pauli, getattr(right_tensor, "device", None)))
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
    def element_count(value: Any) -> int:
        numel = getattr(value, "numel", None)
        if callable(numel):
            return int(numel())
        size = getattr(value, "size", 0)
        return int(size() if callable(size) else size)

    env_list = environments if isinstance(environments, list) else [environments]
    tensor_list = tensors if isinstance(tensors, list) else [tensors]
    values = sum(element_count(item) for env in env_list for item in env.tensors())
    values += sum(element_count(tensor) for tensor in tensor_list)
    tensor = tensor_list[0]
    itemsize_value = getattr(tensor, "element_size", None)
    itemsize = int(itemsize_value() if callable(itemsize_value) else getattr(tensor.dtype, "itemsize", 8))
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
        "device": _device_name(xp, tensor),
    }


def _problem_sha256(payload: CTMRGPayload) -> str:
    """Fingerprint the scientific tensor problem, excluding run controls."""

    data = payload.model_dump(
        mode="json",
        exclude={
            "iterations", "tolerance", "max_time_ms", "max_mem_mb",
            "checkpoint_path", "resume_from", "optimizer_checkpoint_path",
            "optimizer_resume_from",
        },
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


def _mean_float(values: Iterable[Any]) -> float | None:
    normalized = [float(value) for value in values if value is not None]
    return sum(normalized) / len(normalized) if normalized else None


def _reference_for_sector_average(
    payload: CTMRGPayload,
    tensors: list[Any],
    energy: float,
    onsite_values: list[float],
    interaction_values: list[float | None],
) -> dict[str, Any]:
    tolerance = max(float(payload.tolerance) * 10.0, 1e-6)
    reference = finite_product_reference(
        payload,
        tensors,
        onsite_values,
        interaction_values,
        energy,
        tolerance=tolerance,
    )
    if not reference["performed"]:
        reference = analytic_ghz_reference(
            payload,
            tensors,
            onsite_values,
            interaction_values,
            energy,
            tolerance=tolerance,
        )
    if not reference["performed"]:
        reference = finite_periodic_peps_reference(
            payload,
            tensors,
            onsite_values,
            interaction_values,
            energy,
            tolerance=tolerance,
        )
    return reference


def _aggregate_sector_results(
    xp: Any,
    payload: CTMRGPayload,
    sector_results: list[dict[str, Any]],
    tensors: list[Any],
    *,
    started: float,
) -> dict[str, Any]:
    """Average bounded sector fixed points without hiding their spread."""

    if not sector_results:
        raise ValueError("symmetry-sector ensemble requires at least one sector result")
    result = deepcopy(sector_results[0])
    energy = _mean_float(item.get("energy") for item in sector_results) or 0.0
    observables = deepcopy(sector_results[0].get("observables", []))
    for index, observable in enumerate(observables):
        values = [item.get("observables", [])[index].get("value") for item in sector_results]
        observable["value"] = _mean_float(values)
    interactions = deepcopy(sector_results[0].get("interactions", []))
    for index, interaction in enumerate(interactions):
        values = [item.get("interactions", [])[index].get("value") for item in sector_results]
        interaction["value"] = _mean_float(values)
    def value_range(values: Iterable[Any]) -> float:
        normalized = [float(value) for value in values if value is not None]
        return max(normalized) - min(normalized) if normalized else 0.0

    sector_spread = {
        "energy_abs_range": value_range(item.get("energy") for item in sector_results),
        "observable_max_abs_range": max(
            (
                value_range(item.get("observables", [])[index].get("value") for item in sector_results)
                for index in range(len(observables))
            ),
            default=0.0,
        ),
        "interaction_max_abs_range": max(
            (
                value_range(item.get("interactions", [])[index].get("value") for item in sector_results)
                for index in range(len(interactions))
            ),
            default=0.0,
        ),
    }
    onsite_values = [float(item["value"]) for item in observables]
    interaction_values = [item.get("value") for item in interactions]
    reference_validation = _reference_for_sector_average(
        payload,
        tensors,
        energy,
        onsite_values,
        interaction_values,
    )
    converged = all(bool(item.get("converged")) for item in sector_results)
    residual = max(float(item.get("residual", math.inf)) for item in sector_results)
    raw_residual = max(float(item.get("raw_boundary_basis_residual", math.inf)) for item in sector_results)
    correlation_lengths = [item.get("correlation_length") for item in sector_results]
    finite_lengths = [float(value) for value in correlation_lengths if value is not None]
    warnings: list[str] = []
    for item in sector_results:
        for warning in item.get("warnings", []):
            if warning not in warnings:
                warnings.append(warning)
    ensemble_warning = (
        "symmetry-sector ensemble averages two deterministic boundary fixed points; "
        "sector spread remains diagnostic and does not prove thermodynamic convergence"
    )
    if ensemble_warning not in warnings:
        warnings.append(ensemble_warning)
    spread_warning = (
        "symmetry-sector ensemble fixed-point spread is "
        f"energy={sector_spread['energy_abs_range']:.3e}, "
        f"observables={sector_spread['observable_max_abs_range']:.3e}, "
        f"interactions={sector_spread['interaction_max_abs_range']:.3e}"
    )
    warnings.append(spread_warning)
    gauge_validation = {"performed": False, "reason": "disabled by request"}
    gauge_conditioning = deepcopy(sector_results[0].get("gauge_conditioning", {}))
    gauge_preconditioning = deepcopy(sector_results[0].get("gauge_preconditioning", {}))
    research_gate = ctmrg_research_gate(
        payload,
        converged=converged,
        reference_validation=reference_validation,
        gauge_validation=gauge_validation,
        optimization_info=None,
    )
    result.update({
        "environment_sector_policy": "symmetry-ensemble",
        "environment_sector_count": len(sector_results),
        "environment_sector_spread": sector_spread,
        "energy": float(energy),
        "observables": observables,
        "interactions": interactions,
        "converged": converged,
        "residual": float(residual),
        "raw_boundary_basis_residual": float(raw_residual),
        "reference_validation": reference_validation,
        "gauge_validation": gauge_validation,
        "gauge_conditioning": gauge_conditioning,
        "gauge_preconditioning": gauge_preconditioning,
        "research_gate": research_gate,
        "correlation_length": None if any(value is None for value in correlation_lengths) else (
            max(finite_lengths) if finite_lengths else None
        ),
        "correlation_lengths_by_site": sector_results[0].get("correlation_lengths_by_site", []),
        "warnings": warnings,
        "resource_estimate": {
            **dict(result.get("resource_estimate", {})),
            "sector_ensemble": {
                "sector_count": len(sector_results),
                "sector_time_ms": [float(item.get("time_ms", 0.0)) for item in sector_results],
            },
        },
        "time_ms": round((time.perf_counter() - started) * 1000, 3),
    })
    research_result = deepcopy(result.get("research_result", {}))
    metrics = dict(research_result.get("metrics", {}))
    metrics.update({
        "energy": float(energy),
        "residual": float(residual),
        "raw_boundary_basis_residual": float(raw_residual),
        "gauge_validation_max_abs_delta": None,
        "reference_energy_error": reference_validation.get("energy_error"),
        "energy_variance": reference_validation.get("energy_variance"),
        "environment_sector_spread": sector_spread,
    })
    research_result["metrics"] = metrics
    research_result["warnings"] = list(warnings)
    research_result.setdefault("convergence", {})["converged"] = converged
    research_result["convergence"]["warnings"] = list(warnings)
    research_result.setdefault("details", {}).update({
        "environment_sector_policy": "symmetry-ensemble",
        "environment_sector_count": len(sector_results),
        "environment_sector_spread": sector_spread,
        "reference_validation": reference_validation,
        "gauge_validation": gauge_validation,
        "gauge_conditioning": gauge_conditioning,
        "gauge_preconditioning": gauge_preconditioning,
        "research_gate": research_gate,
    })
    result["research_result"] = research_result
    result.setdefault("research_result", {})
    return result


def run_ctmrg(
    xp: Any,
    payload: CTMRGPayload,
    *,
    progress_cb: Any = None,
    cancel_cb: Any = None,
    tensors: list[Any] | None = None,
    _environment_seed: int | None = None,
) -> dict[str, Any]:
    """Run bounded one-site or two-site checkerboard CTMRG contraction.

    ``tensors`` is an internal optimizer seam: when supplied, the solver uses
    the already resident backend tensors directly instead of serializing a
    candidate through the public ``tensor_data`` request field. This avoids a
    GPU-to-host round trip for every variational objective evaluation while
    preserving the public request contract for ordinary jobs.
    """

    unit_cell = list(payload.unit_cell)
    if unit_cell not in ([1, 1], [2, 1], [1, 2], [2, 2]):
        raise ValueError("the current CTMRG solver supports unit_cell dimensions no larger than 2x2")

    started = time.perf_counter()
    if payload.environment_sector_policy == "symmetry-ensemble" and _environment_seed is None:
        ensemble_tensors = _build_tensors(xp, payload) if tensors is None else list(tensors)
        single_payload = payload.model_copy(update={
            "environment_sector_policy": "single",
            "gauge_validation": False,
        })
        sector_results = [
            run_ctmrg(
                xp,
                single_payload,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
                tensors=ensemble_tensors,
                _environment_seed=seed,
            )
            for seed in (0, 1)
        ]
        ensemble_result = _aggregate_sector_results(
            xp,
            payload,
            sector_results,
            ensemble_tensors,
            started=started,
        )
        if payload.gauge_validation:
            from .ctmrg_gauge import gauge_validation_result, paired_virtual_gauge

            gauged_tensors = paired_virtual_gauge(xp, ensemble_tensors)
            gauged_sector_results = [
                run_ctmrg(
                    xp,
                    single_payload,
                    progress_cb=progress_cb,
                    cancel_cb=cancel_cb,
                    tensors=gauged_tensors,
                    _environment_seed=seed,
                )
                for seed in (0, 1)
            ]
            gauged_ensemble = _aggregate_sector_results(
                xp,
                payload,
                gauged_sector_results,
                gauged_tensors,
                started=started,
            )
            gauge_validation = gauge_validation_result(
                {
                    "energy": ensemble_result["energy"],
                    "observables": ensemble_result["observables"],
                    "interactions": ensemble_result["interactions"],
                },
                {
                    "energy": gauged_ensemble["energy"],
                    "observables": gauged_ensemble["observables"],
                    "interactions": gauged_ensemble["interactions"],
                },
                tolerance=float(payload.gauge_validation_tolerance),
                virtual_bond_dim=int(payload.virtual_bond_dim),
            )
            ensemble_result["gauge_validation"] = gauge_validation
            if not gauge_validation["passed"]:
                ensemble_result["warnings"].append(
                    f"virtual-gauge validation exceeded tolerance: max observable/energy delta {gauge_validation['max_abs_delta']:.3e}"
                )
            ensemble_result["research_gate"] = ctmrg_research_gate(
                payload,
                converged=bool(ensemble_result["converged"]),
                reference_validation=ensemble_result["reference_validation"],
                gauge_validation=gauge_validation,
                optimization_info=None,
            )
            ensemble_result["research_result"]["metrics"]["gauge_validation_max_abs_delta"] = gauge_validation.get("max_abs_delta")
            ensemble_result["research_result"]["warnings"] = list(ensemble_result["warnings"])
            ensemble_result["research_result"]["convergence"]["warnings"] = list(ensemble_result["warnings"])
            ensemble_result["research_result"]["details"]["gauge_validation"] = gauge_validation
            ensemble_result["research_result"]["details"]["research_gate"] = ensemble_result["research_gate"]
        ensemble_result["time_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return ensemble_result
    if tensors is None:
        tensors = _build_tensors(xp, payload)
    else:
        expected_dtype = xp.complex64 if payload.dtype == "complex64" else xp.complex128
        expected_shape = (
            int(payload.physical_bond_dim),
            int(payload.virtual_bond_dim),
            int(payload.virtual_bond_dim),
            int(payload.virtual_bond_dim),
            int(payload.virtual_bond_dim),
        )
        if len(tensors) != math.prod(payload.unit_cell):
            raise ValueError("runtime CTMRG tensor count does not match the unit cell")
        for index, tensor in enumerate(tensors):
            if tuple(int(size) for size in tensor.shape) != expected_shape:
                raise ValueError(f"runtime CTMRG tensor_{index} shape does not match the request")
            if tensor.dtype != expected_dtype:
                raise ValueError(f"runtime CTMRG tensor_{index} dtype does not match the request")
            if not bool(_host(xp.all(xp.isfinite(tensor)))):
                raise ValueError(f"runtime CTMRG tensor_{index} contains non-finite values")
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
    gauge_preconditioning: dict[str, Any] = {
        "performed": False,
        "method": str(payload.gauge_preconditioner),
        "reason": "disabled by request",
    }
    if payload.gauge_preconditioner == "pairwise-polar-balance":
        tensors, gauge_preconditioning = pairwise_virtual_gauge_preconditioner(
            xp,
            tensors,
            unit_cell=(int(unit_cell[0]), int(unit_cell[1])),
            iterations=int(payload.gauge_preconditioner_iterations),
        )
    layers = [_double_layer(xp, tensor) for tensor in tensors]
    chi = int(payload.environment_bond_dim)
    environments = [_initialize_environment(xp, layer, chi, sector_seed=_environment_seed) for layer in layers]
    problem_sha256 = _problem_sha256(payload)
    points: list[ConvergencePoint] = []
    discarded_total = 0.0
    residual = math.inf
    raw_residual = math.inf
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
                device=_device_name(xp, tensors[0] if tensors else None),
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
            env, left_discarded = _left_move(
                xp, env, layers[0], chi, tensor=tensors[0], projector_method=payload.ctmrg_projector
            )
            env, right_discarded = _right_move(
                xp, env, layers[0], chi, tensor=tensors[0], projector_method=payload.ctmrg_projector
            )
            env, top_discarded = _top_move(
                xp, env, layers[0], chi, tensor=tensors[0], projector_method=payload.ctmrg_projector
            )
            env, bottom_discarded = _bottom_move(
                xp, env, layers[0], chi, tensor=tensors[0], projector_method=payload.ctmrg_projector
            )
            candidate = _renormalize(xp, env)
            environments = _blend_environments(
                xp,
                before,
                [candidate],
                float(payload.environment_damping),
            )
            discarded_sweep = left_discarded + right_discarded + top_discarded + bottom_discarded
        else:
            candidates, discarded_sweep = _unit_cell_sweep(
                xp,
                environments,
                layers,
                chi,
                unit_cell,
                tensors=tensors,
                projector_method=payload.ctmrg_projector,
            )
            environments = _blend_environments(
                xp,
                before,
                candidates,
                float(payload.environment_damping),
            )
        discarded_total += discarded_sweep
        residual = max(
            _environment_residual(xp, old, new)
            for old, new in zip(before, environments)
        )
        raw_residual = max(
            _raw_environment_residual(xp, old, new)
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
    if not reference_validation["performed"]:
        reference_validation = analytic_ghz_reference(
            payload,
            tensors,
            onsite_values,
            interaction_values,
            float(energy),
            tolerance=max(float(payload.tolerance) * 10.0, 1e-6),
        )
    if not reference_validation["performed"]:
        reference_validation = finite_periodic_peps_reference(
            payload,
            tensors,
            onsite_values,
            interaction_values,
            float(energy),
            tolerance=max(float(payload.tolerance) * 10.0, 1e-6),
        )
    gauge_conditioning = virtual_leg_conditioning_report(xp, tensors)
    converged = bool(residual <= float(payload.tolerance))
    optimization_consistency_error: float | None = None
    if (
        optimization_info is not None
        and payload.optimization == "full-update"
        and payload.full_update_optimizer != "finite-torus-gradient"
        and optimization_info.get("final_energy") is not None
    ):
        optimization_consistency_error = abs(float(energy) - float(optimization_info["final_energy"]))
        optimization_info["objective_consistency_error"] = optimization_consistency_error
    result_method = (
        "ipeps-full-update-finite-torus-gradient-ctmrg" if payload.optimization == "full-update" and payload.full_update_optimizer == "finite-torus-gradient" else
        "ipeps-full-update-autodiff-ctmrg" if payload.optimization == "full-update" and payload.full_update_optimizer == "autodiff-ctmrg-gradient" else
        "ipeps-full-update-implicit-ctmrg" if payload.optimization == "full-update" and payload.full_update_optimizer == "implicit-ctmrg-gradient" else
        "ipeps-full-update-gradient-ctmrg" if payload.optimization == "full-update" and payload.full_update_optimizer == "finite-difference-gradient" else
        "ipeps-full-update-spsa-ctmrg" if payload.optimization == "full-update" and payload.full_update_optimizer == "spsa-gradient" else
        "ipeps-full-update-ctmrg" if payload.optimization == "full-update" else
        "ipeps-simple-update-ctmrg" if payload.optimization == "simple-update" else
        "ipeps-ctmrg-product-optimization" if optimization_info is not None else
        "ipeps-ctmrg-contraction"
    )
    warnings = [
        "compare environment_bond_dim and iteration convergence before using values as scientific conclusions",
    ]
    if float(payload.environment_damping) < 1.0:
        warnings.append(
            f"CTMRG environment updates use under-relaxation damping={float(payload.environment_damping):.3f}; compare fixed-point residuals across damping values"
        )
    if payload.ctmrg_projector == "full-svd":
        warnings.append(
            "full-svd CTMRG projector is an opt-in entangled research path; it remains needs_review until paired-gauge and independent-reference gates pass"
        )
    if raw_residual > max(float(payload.tolerance) * 10.0, 1e-6) and residual <= float(payload.tolerance):
        warnings.append(
            f"raw boundary-basis residual is {raw_residual:.3e}; convergence uses a gauge-invariant environment spectrum"
        )
    if len(tensors) == 1:
        warnings.insert(0, "CTMRG contraction uses a one-site translational environment")
    else:
        warnings.insert(0, "CTMRG contraction uses a periodic multi-site unit-cell environment")
    if payload.optimization == "simple-update":
        warnings.append("simple-update is an imaginary-time entangled-tensor baseline; compare it against the bounded full-update path before treating energies as variational evidence")
    elif payload.optimization == "full-update":
        truncation_gradient = str(getattr(payload, "full_update_truncation_gradient", "frozen-eigenprojector"))
        if payload.full_update_optimizer == "finite-torus-gradient":
            warnings.append("finite-torus-gradient optimizes an exact bounded 2x2 periodic reference objective; it is not an infinite-lattice variational proof")
        elif payload.full_update_optimizer == "autodiff-ctmrg-gradient":
            warnings.append(
                "torch autograd differentiates a bounded unrolled CTMRG environment with a differentiable Hermitian truncation eigenspace; it is experimental and requires non-degenerate transfer spectra"
                if truncation_gradient == "differentiable-eigh" else
                "torch autograd differentiates a bounded unrolled CTMRG environment with a frozen truncation projector; it is experimental and not yet an implicit fixed-point variational proof"
            )
        elif payload.full_update_optimizer == "implicit-ctmrg-gradient":
            warnings.append(
                "implicit-ctmrg-gradient uses a bounded adjoint fixed-point solve with differentiable Hermitian truncation; validate non-degenerate spectra, transfer gaps, gauge sensitivity, and backward residual before scientific use"
                if truncation_gradient == "differentiable-eigh" else
                "implicit-ctmrg-gradient uses a bounded adjoint fixed-point solve with a frozen truncation projector; validate transfer gaps, gauge sensitivity, and backward residual before scientific use"
            )
            if optimization_info is not None:
                transfer_gap = float(optimization_info.get("transfer_gap", 0.0))
                adjoint_residual = float(optimization_info.get("adjoint_residual", math.inf))
                if transfer_gap <= 1e-5:
                    warnings.append("implicit CTMRG transfer gap is unresolved; the adjoint fixed-point gradient is not reliable for this tensor")
                if adjoint_residual > float(getattr(payload, "full_update_implicit_tolerance", 1e-6)):
                    warnings.append("implicit CTMRG adjoint residual exceeds its declared tolerance; treat optimizer diagnostics as needs_review")
        elif payload.full_update_optimizer == "finite-difference-gradient":
            warnings.append("finite-difference-gradient full-update is a bounded gradient estimate; it is not automatic differentiation and does not scale to large tensors")
        elif payload.full_update_optimizer == "spsa-gradient":
            warnings.append("SPSA full-update uses two deterministic simultaneous-perturbation evaluations per step; it is a scalable approximate gradient baseline, not automatic differentiation or a variational convergence proof")
        else:
            warnings.append("full-update re-evaluates CTMRG energy for bounded coordinate trials; it is not an automatic-differentiation optimizer")
        if optimization_consistency_error is not None and optimization_consistency_error > max(float(payload.optimization_tolerance) * 10.0, 1e-5):
            warnings.append(
                f"full-update objective/final CTMRG energy mismatch is {optimization_consistency_error:.3e}; treat the optimizer result as needs_review"
            )
    elif optimization_info is not None:
        warnings.append("product-coordinate-descent is a variational mean-field baseline with virtual_bond_dim=1; it is not an entangled iPEPS update")
    elif payload.tensor_data is None:
        warnings.append("the tensor is a deterministic product-state ansatz; no variational ground-state optimization was performed")
    else:
        warnings.append("the imported tensor was contracted without variational ground-state optimization")
    if int(payload.virtual_bond_dim) > 1 and payload.dtype == "complex64":
        warnings.append("complex64 entangled iPEPS runs may lose transfer-sector precision; use complex128 for reference-quality observables")
    if payload.gauge_preconditioner != "none":
        warnings.append(
            "pairwise-polar-balance is an opt-in 1x1-2x2 gauge diagnostic; it preserves finite periodic bond pairing but is not admitted into optimization or production paths"
        )
    if gauge_conditioning.get("performed") and not gauge_conditioning.get("well_conditioned", False):
        warnings.append("one or more virtual-leg Gram spectra are rank-deficient or ill-conditioned; gauge preconditioning remains diagnostic-only")
    if not interaction_values_available:
        warnings.append("one or more interaction displacements are outside the supported nearest-neighbor two-site CTM contraction")
    if reference_validation["performed"] and not reference_validation["passed"]:
        warnings.append(f"{reference_validation.get('reference', 'independent reference')} comparison exceeded its declared tolerance")
    elif not reference_validation["performed"]:
        warnings.append(f"independent finite product reference unavailable: {reference_validation['reason']}")
    gauge_validation: dict[str, Any] = {"performed": False, "reason": "disabled by request"}
    if bool(getattr(payload, "gauge_validation", False)):
        if int(payload.virtual_bond_dim) <= 1:
            gauge_validation["reason"] = "virtual_bond_dim=1 has no non-trivial virtual gauge probe"
        elif int(payload.virtual_bond_dim) > 2:
            gauge_validation["reason"] = "the bounded paired gauge probe currently supports virtual_bond_dim<=2"
        else:
            from .ctmrg_gauge import gauge_validation_result, paired_virtual_gauge

            probe_payload = payload.model_copy(update={
                "optimization": "none",
                "checkpoint_path": None,
                "resume_from": None,
                "optimizer_checkpoint_path": None,
                "optimizer_resume_from": None,
                "gauge_validation": False,
            })
            gauged_tensors = paired_virtual_gauge(xp, tensors)
            gauged_result = run_ctmrg(xp, probe_payload, tensors=gauged_tensors)
            gauge_validation = gauge_validation_result(
                {
                    "energy": float(energy),
                    "observables": structured_observables(payload.terms, onsite_values),
                    "interactions": [
                        {"value": value}
                        for value in interaction_values
                    ],
                },
                gauged_result,
                tolerance=float(payload.gauge_validation_tolerance),
                virtual_bond_dim=int(payload.virtual_bond_dim),
            )
            if not gauge_validation["passed"]:
                warnings.append(
                    f"virtual-gauge validation exceeded tolerance: max observable/energy delta {gauge_validation['max_abs_delta']:.3e}"
                )
    research_gate = ctmrg_research_gate(
        payload,
        converged=converged,
        reference_validation=reference_validation,
        gauge_validation=gauge_validation,
        optimization_info=optimization_info,
    )
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
            "finite-torus-gradient optimizes a bounded exact 2x2 reference objective and does not establish infinite-lattice convergence"
            if payload.optimization == "full-update" and payload.full_update_optimizer == "finite-torus-gradient" else
            "autodiff-ctmrg-gradient is a bounded unrolled CTMRG gradient and does not yet provide an implicit fixed-point or thermodynamic-limit variational proof"
            if payload.optimization == "full-update" and payload.full_update_optimizer == "autodiff-ctmrg-gradient" else
            "implicit-ctmrg-gradient is bounded and requires a resolved transfer gap, gauge-invariance evidence, and backward-error validation before production use"
            if payload.optimization == "full-update" and payload.full_update_optimizer == "implicit-ctmrg-gradient" else
            "full-update is a bounded experimental optimization path and is not a scalable automatic-differentiation or full ground-state solver"
            if payload.optimization == "full-update" else
            "no environment-feedback full ground-state optimization"
        ),
        (
            "energy variance is a finite product-supercell diagnostic and is not an infinite-lattice variance proof"
            if reference_validation.get("reference") == "finite-product-supercell" else
            f"energy variance is a finite torus ({'x'.join(str(value) for value in reference_validation.get('reference_lattice', [2, 2]))} periodic-torus) diagnostic and is not an infinite-lattice variance proof"
            if reference_validation.get("reference") == "finite-periodic-peps-2x2" else
            "the selected entangled reference validates local observables but does not provide an infinite-lattice variance"
            if reference_validation["performed"] else
            "energy variance and independent reference are unavailable for the current entangled tensor"
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
            "raw_boundary_basis_residual": float(raw_residual),
            "gauge_validation_max_abs_delta": gauge_validation.get("max_abs_delta"),
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
            "ctmrg_projector": payload.ctmrg_projector,
            "environment_sector_policy": payload.environment_sector_policy,
            "environment_damping": float(payload.environment_damping),
            "environment_shapes": [[list(item.shape) for item in env.tensors()] for env in environments],
            "optimization": (
                {key: value for key, value in optimization_info.items() if key not in {"states", "tensors"}}
                | ({"optimized_state_vectors": _state_pairs(xp, optimization_info["states"])} if "states" in optimization_info else {})
                if optimization_info is not None else {"method": "none"}
            ),
            "correlation_lengths_by_site": environment_diagnostics["correlation_lengths_by_site"],
            "environment_spectrum": environment_diagnostics["environment_spectrum"],
            "reference_validation": reference_validation,
            "gauge_validation": gauge_validation,
            "gauge_conditioning": gauge_conditioning,
            "gauge_preconditioning": gauge_preconditioning,
            "research_gate": research_gate,
        },
    ).to_dict()
    return {
        "status": "done",
        "backend": "tensor-network-ctmrg",
        "method": result_method,
        "representation": "ipeps",
        "ctmrg_projector": payload.ctmrg_projector,
        "environment_sector_policy": payload.environment_sector_policy,
        "environment_damping": float(payload.environment_damping),
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
        "raw_boundary_basis_residual": float(raw_residual),
        "norm": norm,
        "energy": float(energy),
        "energy_complete": interaction_values_available,
        "energy_second_moment": reference_validation.get("energy_second_moment"),
        "energy_variance": reference_validation.get("energy_variance"),
        "reference_validation": reference_validation,
        "gauge_validation": gauge_validation,
        "gauge_conditioning": gauge_conditioning,
        "gauge_preconditioning": gauge_preconditioning,
        "research_gate": research_gate,
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
    *,
    progress_cb: Any = None,
    cancel_cb: Any = None,
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
    previous_observables: list[float] | None = None
    previous_interactions: list[float | None] | None = None
    study_gauge_conditioning: dict[str, Any] | None = None
    for point_index, environment_bond_dim in enumerate(normalized_dims):
        if cancel_cb and cancel_cb():
            raise RuntimeError("job canceled")
        point_payload = payload.model_copy(update={
            "environment_bond_dim": environment_bond_dim,
            "checkpoint_path": None,
            "resume_from": None,
        })
        def point_progress(value: float, phase: str) -> None:
            if progress_cb:
                progress_cb(
                    (point_index + float(value)) / len(normalized_dims),
                    f"chi-{environment_bond_dim}-{phase}",
                )

        result = run_ctmrg(
            xp,
            point_payload,
            progress_cb=point_progress,
            cancel_cb=cancel_cb,
        )
        if study_gauge_conditioning is None:
            study_gauge_conditioning = dict(result["gauge_conditioning"])
        energy = float(result["energy"])
        observables = [float(item["value"]) for item in result["observables"]]
        interactions = [
            None if item["value"] is None else float(item["value"])
            for item in result["interactions"]
        ]
        observable_delta = None
        if previous_observables is not None and len(previous_observables) == len(observables):
            observable_delta = max(
                (abs(current - previous) for current, previous in zip(observables, previous_observables)),
                default=0.0,
            )
        interaction_delta = None
        if previous_interactions is not None and len(previous_interactions) == len(interactions):
            comparable = [
                abs(float(current) - float(previous))
                for current, previous in zip(interactions, previous_interactions)
                if current is not None and previous is not None
            ]
            interaction_delta = max(comparable, default=0.0)
        reference = result["reference_validation"]
        research_gate = result["research_gate"]
        points.append({
            "environment_bond_dim": environment_bond_dim,
            "environment_bond_dim_used": int(result["environment_bond_dim_used"]),
            "environment_sector_policy": result.get("environment_sector_policy", "single"),
            "environment_sector_count": int(result.get("environment_sector_count", 1)),
            "environment_sector_spread": result.get("environment_sector_spread"),
            "energy": energy,
            "energy_complete": bool(result["energy_complete"]),
            "energy_delta": None if previous_energy is None else energy - previous_energy,
            "energy_abs_delta": None if previous_energy is None else abs(energy - previous_energy),
            "observable_max_abs_delta": observable_delta,
            "interaction_max_abs_delta": interaction_delta,
            "residual": float(result["residual"]),
            "raw_boundary_basis_residual": float(result["raw_boundary_basis_residual"]),
            "converged": bool(result["converged"]),
            "correlation_length": result["correlation_length"],
            "correlation_lengths_by_site": result["correlation_lengths_by_site"],
            "environment_spectrum": result["environment_spectrum"],
            "energy_second_moment": result["energy_second_moment"],
            "energy_variance": result["energy_variance"],
            "reference_validation": reference,
            "reference_name": reference.get("reference") if reference.get("performed") else None,
            "reference_passed": bool(reference.get("passed")) if reference.get("performed") else None,
            "reference_max_abs_error": reference.get("max_abs_error"),
            "research_gate": research_gate,
            "research_gate_status": research_gate.get("status"),
            "research_gate_production_ready": bool(research_gate.get("production_ready", False)),
            "research_gate_blocking_reasons": list(research_gate.get("blocking_reasons", [])),
            "gauge_conditioning_well_conditioned": bool(
                result["gauge_conditioning"].get("well_conditioned", False)
            ),
            "resource_estimate": result["resource_estimate"],
        })
        previous_energy = energy
        previous_observables = observables
        previous_interactions = interactions

    reference_names = sorted({
        str(point["reference_name"])
        for point in points
        if point["reference_name"] is not None
    })
    reference_errors = [
        float(point["reference_max_abs_error"])
        for point in points
        if point["reference_max_abs_error"] is not None
    ]
    research_gate_statuses = [str(point["research_gate_status"]) for point in points]
    research_gate_blocking_reasons = sorted({
        str(reason)
        for point in points
        for reason in point["research_gate_blocking_reasons"]
    })
    research_gate_summary = {
        "status": "passed" if all(status == "passed" for status in research_gate_statuses) else "needs_review",
        "scope": "bounded-declared-ctmrg-study-contract",
        "production_ready": bool(all(point["research_gate_production_ready"] for point in points)),
        "points": len(points),
        "passed_points": sum(status == "passed" for status in research_gate_statuses),
        "review_points": sum(status != "passed" for status in research_gate_statuses),
        "blocking_reasons": research_gate_blocking_reasons,
    }
    sector_policies = sorted({
        str(point["environment_sector_policy"])
        for point in points
    })
    sector_spread_fields = (
        "energy_abs_range",
        "observable_max_abs_range",
        "interaction_max_abs_range",
    )
    sector_spread_summary = {
        field: max(
            (
                float(point["environment_sector_spread"].get(field, 0.0))
                for point in points
                if point.get("environment_sector_spread") is not None
            ),
            default=0.0,
        )
        for field in sector_spread_fields
    }

    return {
        "status": "done",
        "method": "ipeps-ctmrg-environment-convergence-study",
        "optimization": "none",
        "unit_cell": list(payload.unit_cell),
        "unit_cell_sites": math.prod(payload.unit_cell),
        "points": points,
        "reference_summary": {
            "reference_names": reference_names,
            "consistent_reference": reference_names[0] if len(reference_names) == 1 else None,
            "performed_points": sum(1 for point in points if point["reference_name"] is not None),
            "passed_points": sum(1 for point in points if point["reference_passed"] is True),
            "max_abs_error": max(reference_errors, default=None),
        },
        "research_gate_summary": research_gate_summary,
        "environment_sector_summary": {
            "policies": sector_policies,
            "sector_counts": sorted({int(point["environment_sector_count"]) for point in points}),
            "max_spread": sector_spread_summary,
        },
        "gauge_conditioning": study_gauge_conditioning or {
            "performed": False,
            "reason": "study has no points",
        },
        "materializes_statevector": False,
        "warnings": [
            "points are independent bounded CTMRG contractions from the same tensor ansatz",
            "compare energy, residual, correlation length, and local observables together before drawing physical conclusions",
            "sector spread is retained per point; a small chi delta does not override a large boundary-sector spread",
        ],
    }
