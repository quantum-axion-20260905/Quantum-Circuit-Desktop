"""Full SVD CTMRG projectors.

The default CTMRG implementation intentionally keeps its small half-density
projector.  This module contains the more expensive full-projector policy as
an isolated numerical seam.  It contracts the four CTM quarters with open
ket/bra virtual legs, constructs biorthogonal projectors from an SVD, and
returns only the matrices needed by the directional absorption routines.

The policy is experimental until the paired-gauge and independent-reference
gates pass for entangled tensors.  Keeping it here prevents a research
experiment from silently changing the established contraction path.
"""

from __future__ import annotations

from typing import Any


def _transpose(xp: Any, value: Any, axes: tuple[int, ...]) -> Any:
    if getattr(xp, "__name__", "") == "torch":
        return value.permute(*axes)
    return value.transpose(axes)


def _host(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach()
    cpu = getattr(value, "cpu", None)
    if cpu is not None:
        value = cpu()
    try:
        return value.numpy()
    except AttributeError:
        try:
            return value.get()
        except AttributeError:
            return value


def _as_float(value: Any) -> float:
    return float(complex(_host(value)).real)


def _split_edge(edge: Any, virtual: int) -> Any:
    if int(edge.shape[1]) != virtual * virtual:
        raise ValueError(
            "full-svd CTMRG projectors require a fused edge dimension equal to virtual_bond_dim**2"
        )
    return edge.reshape(int(edge.shape[0]), virtual, virtual, int(edge.shape[2]))


def _matrix(value: Any) -> Any:
    return value.reshape(
        int(value.shape[0]) * int(value.shape[1]) * int(value.shape[2]),
        int(value.shape[3]) * int(value.shape[4]) * int(value.shape[5]),
    )


def swap_fused_pair_rows(xp: Any, value: Any, boundary_dim: int, virtual: int) -> Any:
    """Convert a [boundary, bra, ket] fused basis to [boundary, ket, bra]."""

    reshaped = value.reshape(boundary_dim, virtual, virtual, int(value.shape[1]))
    return _transpose(xp, reshaped, (0, 2, 1, 3)).reshape(value.shape)


def swap_fused_pair_columns(xp: Any, value: Any, boundary_dim: int, virtual: int) -> Any:
    """Convert a [boundary, bra, ket] fused basis on matrix columns."""

    reshaped = value.reshape(int(value.shape[0]), boundary_dim, virtual, virtual)
    return _transpose(xp, reshaped, (0, 1, 3, 2)).reshape(value.shape)


def _normalize(xp: Any, value: Any) -> Any:
    norm = xp.linalg.norm(value)
    if _as_float(norm) <= 1e-30:
        raise ValueError("full-svd CTMRG quarter contraction has zero norm")
    return value / norm


def _truncated_svd(
    xp: Any,
    matrix: Any,
    chi: int,
    *,
    differentiate_truncation: bool,
) -> tuple[Any, Any, Any, float]:
    U, singular, Vh = xp.linalg.svd(matrix, full_matrices=False)
    keep = min(int(chi), int(singular.shape[0]))
    if keep < 1:
        raise ValueError("full-svd CTMRG projector cannot retain zero singular values")
    if not differentiate_truncation and getattr(xp, "__name__", "") == "torch":
        U = U.detach()
        singular = singular.detach()
        Vh = Vh.detach()
    retained = singular[:keep]
    scale = max(_as_float(singular[0]), 1e-30)
    active = retained > scale * 1e-12
    inverse_sqrt = xp.where(active, 1.0 / xp.sqrt(xp.where(active, retained, 1.0)), 0.0)
    total = xp.sum(xp.abs(singular) ** 2)
    discarded = xp.sum(xp.abs(singular[keep:]) ** 2) if keep < int(singular.shape[0]) else 0.0
    discarded_weight = _as_float(discarded / (total + 1e-30))
    return (
        inverse_sqrt,
        U[:, :keep],
        Vh[:keep, :],
        discarded_weight,
    )


def _quarter_matrices(xp: Any, env: Any, tensor: Any) -> tuple[Any, Any, Any, Any]:
    """Build the four three-leg quarter matrices used by full CTMRG.

    The public tensor order is ``(physical, up, down, left, right)`` and the
    environment edges are fused ket/bra legs.  The contractions below keep
    the ket/bra order explicit while forming the four corner quarters:

    ``top_left, top_right, bottom_left, bottom_right``.
    """

    virtual = int(tensor.shape[1])
    C1, C2, C3, C4, T1, T2, T3, T4 = env.tensors()
    # Convert the fused internal edges to the ket/bra order used by the
    # standard four-quarter CTMRG definitions.  T2/T4 also reverse their
    # corner orientation at the cut; preserving this order matters for a
    # non-Hermitian retained boundary basis.
    top = _split_edge(T1, virtual)              # C1, up ket, up bra, C2
    right = _transpose(xp, _split_edge(T2, virtual), (1, 2, 3, 0))  # r, R, C3, C2
    bottom = _transpose(xp, _split_edge(T3, virtual), (0, 3, 1, 2))  # C4, C3, D, d
    left = _transpose(xp, _split_edge(T4, virtual), (3, 2, 1, 0))  # C4, L, l, C1
    conjugate = xp.conj(tensor)

    # Each result has shape (chi, D, D, chi, D, D).  The first three and
    # last three axes are the row and column spaces of a quarter matrix.
    top_left = xp.einsum(
        "gLla,ab,buUc,sudlr,sUDLR->gdDcrR",
        left,
        C1,
        top,
        tensor,
        conjugate,
    )
    top_right = xp.einsum(
        "ce,buUc,rRme,sudlr,sUDLR->blLmdD",
        C2,
        top,
        right,
        tensor,
        conjugate,
    )
    bottom_left = xp.einsum(
        "hg,hiDd,gLla,sudlr,sUDLR->irRauU",
        C4,
        bottom,
        left,
        tensor,
        conjugate,
    )
    bottom_right = xp.einsum(
        "mi,rRme,hiDd,sudlr,sUDLR->euUhlL",
        C3,
        right,
        bottom,
        tensor,
        conjugate,
    )

    return tuple(
        _normalize(xp, _matrix(value))
        for value in (top_left, top_right, bottom_left, bottom_right)
    )  # type: ignore[return-value]


def standard_full_svd_projectors(
    xp: Any,
    env: Any,
    tensor: Any,
    chi: int,
    direction: str,
    *,
    differentiate_truncation: bool = True,
) -> tuple[Any, Any, float]:
    """Build the standard CTMRG ``P``/``P~`` pair for one-site absorption.

    The four quarter matrices are arranged into the two enlarged corners
    facing the requested cut.  The source CTMRG construction then performs
    ``SVD(R.T @ R_tilde)`` and forms

    ``P = R @ conj(U) @ S**(-1/2)`` and
    ``P_tilde = R_tilde @ V @ S**(-1/2)``.

    Both returned matrices use the unfused convention ``(chi * D**2, chi)``;
    the directional absorption code is responsible for assigning their
    boundary and ket/bra legs.  Keeping this convention explicit prevents a
    transpose or conjugation intended for one move from leaking into another.
    The older ``full_svd_projectors`` function below remains available for the
    multi-site experimental path until its neighbour-specific absorption is
    migrated to the same contract.
    """

    if direction not in {"left", "right", "top", "bottom"}:
        raise ValueError(f"unsupported standard full-svd CTMRG direction {direction!r}")
    top_left, top_right, bottom_left, bottom_right = _quarter_matrices(xp, env, tensor)
    if direction == "left":
        R, R_tilde = top_left, _transpose(xp, bottom_left, (1, 0))
    elif direction == "right":
        R, R_tilde = bottom_right, _transpose(xp, top_right, (1, 0))
    elif direction == "top":
        R, R_tilde = top_right, _transpose(xp, top_left, (1, 0))
    else:
        R, R_tilde = _transpose(xp, bottom_left, (1, 0)), _transpose(xp, bottom_right, (1, 0))

    inverse_sqrt, U, Vh, discarded = _truncated_svd(
        xp,
        _transpose(xp, R, (1, 0)) @ R_tilde,
        chi,
        differentiate_truncation=differentiate_truncation,
    )
    keep = int(inverse_sqrt.shape[0])
    V = _transpose(xp, xp.conj(Vh), (1, 0))
    P = (R @ xp.conj(U[:, :keep])) * inverse_sqrt[None, :]
    P_tilde = (R_tilde @ V[:, :keep]) * inverse_sqrt[None, :]
    return P, P_tilde, discarded


def full_svd_projectors(
    xp: Any,
    env: Any,
    tensor: Any,
    chi: int,
    direction: str,
    *,
    differentiate_truncation: bool = True,
) -> tuple[Any, Any, float]:
    """Return directional biorthogonal CTMRG projectors.

    The returned pair is ordered according to the directional absorption
    helper: for left/right it is ``(first_boundary, second_boundary)``; for
    top/bottom it is ``(left_boundary, right_boundary)``.  Shapes are
    explicitly documented at the call sites in ``ctmrg.py``.
    """

    if direction not in {"left", "right", "top", "bottom"}:
        raise ValueError(f"unsupported full-svd CTMRG direction {direction!r}")
    top_left, top_right, bottom_left, bottom_right = _quarter_matrices(xp, env, tensor)

    if direction == "left":
        upper = top_left @ top_right
        lower = bottom_right @ bottom_left
        product = lower @ upper
        inverse_sqrt, U, Vh, discarded = _truncated_svd(
            xp, product, chi, differentiate_truncation=differentiate_truncation
        )
        first = lower
        second = upper
        # first: new x old, second: old x new
        return (
            U.conj().T * inverse_sqrt[:, None] @ first,
            second @ (Vh.conj().T * inverse_sqrt),
            discarded,
        )

    if direction == "right":
        upper = top_left @ top_right
        lower = bottom_right @ bottom_left
        product = upper @ lower
        inverse_sqrt, U, Vh, discarded = _truncated_svd(
            xp, product, chi, differentiate_truncation=differentiate_truncation
        )
        # first: new x old, second: old x new
        return (
            U.conj().T * inverse_sqrt[:, None] @ upper,
            lower @ (Vh.conj().T * inverse_sqrt),
            discarded,
        )

    left_cut = bottom_left @ top_left
    right_cut = top_right @ bottom_right
    if direction == "top":
        product = left_cut @ right_cut
        inverse_sqrt, U, Vh, discarded = _truncated_svd(
            xp, product, chi, differentiate_truncation=differentiate_truncation
        )
        # first: new x old, second: old x new
        return (
            U.conj().T * inverse_sqrt[:, None] @ left_cut,
            right_cut @ (Vh.conj().T * inverse_sqrt),
            discarded,
        )

    product = right_cut @ left_cut
    inverse_sqrt, U, Vh, discarded = _truncated_svd(
        xp, product, chi, differentiate_truncation=differentiate_truncation
    )
    # first: new x old, second: old x new
    return (
        left_cut @ (Vh.conj().T * inverse_sqrt),
        U.conj().T * inverse_sqrt[:, None] @ right_cut,
        discarded,
    )
