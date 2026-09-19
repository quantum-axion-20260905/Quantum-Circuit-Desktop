"""Rectangular boundary-state contracts for the next CTMRG environment map.

The production CTMRG path currently stores a square fixed-``chi`` environment.
This module defines the shape contract for a future dynamically retained
boundary without changing that path prematurely.  Keeping the contract in a
small backend-neutral module makes NumPy/CuPy/Torch kernels and checkpoint
code depend on the same directional vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


DYNAMIC_BOUNDARY_SCHEMA = "quantum-circuit/ctmrg-dynamic-boundary-v1"
_CORNER_NAMES = ("C1", "C2", "C3", "C4")
_EDGE_NAMES = ("T1", "T2", "T3", "T4")


@dataclass(frozen=True)
class BoundaryDimensions:
    """Retained dimensions on the four directional CTM boundary sides."""

    top: int
    left: int
    bottom: int
    right: int

    def __post_init__(self) -> None:
        for name in ("top", "left", "bottom", "right"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"boundary dimension {name} must be positive")

    @classmethod
    def uniform(cls, dimension: int) -> "BoundaryDimensions":
        return cls(dimension, dimension, dimension, dimension)

    def to_dict(self) -> dict[str, int]:
        return {
            "top": int(self.top),
            "left": int(self.left),
            "bottom": int(self.bottom),
            "right": int(self.right),
        }


@dataclass(frozen=True)
class DynamicCTMEnvironment:
    """A rectangular CTM boundary with explicit directional dimensions.

    Corner ordering follows the existing ``CTMEnvironment`` contraction
    indices: ``C1`` is ``(left, top)``, ``C2`` is ``(top, right)``, ``C3``
    is ``(bottom, right)``, and ``C4`` is ``(left, bottom)``. ``T1``/``T3``
    run along the top/bottom sides and ``T2``/``T4`` along the right/left
    sides. The fused double-layer leg is the middle edge axis and is
    intentionally not constrained here.
    """

    C1: Any
    C2: Any
    C3: Any
    C4: Any
    T1: Any
    T2: Any
    T3: Any
    T4: Any
    dimensions: BoundaryDimensions
    map_id: str = "ctmrg-covariant-dynamic-frame-v1"

    def __post_init__(self) -> None:
        if not self.map_id.strip():
            raise ValueError("dynamic CTM environment map_id must not be empty")
        self.validate_shapes()

    def tensors(self) -> tuple[Any, ...]:
        return (self.C1, self.C2, self.C3, self.C4, self.T1, self.T2, self.T3, self.T4)

    def validate_shapes(self) -> None:
        dims = self.dimensions
        corner_shapes = {
            "C1": (dims.left, dims.top),
            "C2": (dims.top, dims.right),
            "C3": (dims.bottom, dims.right),
            "C4": (dims.left, dims.bottom),
        }
        edge_shapes = {
            "T1": dims.top,
            "T2": dims.right,
            "T3": dims.bottom,
            "T4": dims.left,
        }
        for name, expected in corner_shapes.items():
            tensor = getattr(self, name)
            if getattr(tensor, "ndim", None) != 2:
                raise ValueError(f"dynamic CTM corner {name} must be rank-2")
            actual = tuple(int(size) for size in tensor.shape)
            if actual != expected:
                raise ValueError(f"dynamic CTM corner {name} must have shape {expected}, got {actual}")
        for name, boundary_dim in edge_shapes.items():
            tensor = getattr(self, name)
            if getattr(tensor, "ndim", None) != 3:
                raise ValueError(f"dynamic CTM edge {name} must be rank-3")
            actual = tuple(int(size) for size in tensor.shape)
            if actual[0] != boundary_dim or actual[2] != boundary_dim:
                raise ValueError(
                    f"dynamic CTM edge {name} must have boundary shape ({boundary_dim}, d2, {boundary_dim}), got {actual}"
                )
            if actual[1] <= 0:
                raise ValueError(f"dynamic CTM edge {name} fused dimension must be positive")

    def shape_manifest(self) -> dict[str, Any]:
        return {
            "schema": DYNAMIC_BOUNDARY_SCHEMA,
            "map_id": self.map_id,
            "dimensions": self.dimensions.to_dict(),
            "corners": {
                name: list(int(size) for size in getattr(self, name).shape)
                for name in _CORNER_NAMES
            },
            "edges": {
                name: list(int(size) for size in getattr(self, name).shape)
                for name in _EDGE_NAMES
            },
        }


def dynamic_boundary_manifest_digest(manifest: dict[str, Any]) -> str:
    """Return the canonical digest for a rectangular boundary shape manifest."""

    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def apply_dynamic_covariant_bilinear_move(
    xp: Any,
    left_factor: Any,
    right_factor: Any,
    grown_edge: Any,
    requested_dim: int,
    *,
    relative_singular_floor: float = 1e-12,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Apply one rectangular primal/dual bilinear boundary projection.

    ``left_factor`` and ``right_factor`` are enlarged-boundary maps and
    ``grown_edge`` has matching enlarged indices on its first and third axes.
    The selector chooses the supported retained sector, then applies the
    bilinear rules ``Q.T @ L``, ``P.T @ R``, and ``P.T @ E @ Q``.  If the
    effective rank is smaller than the request, the returned corners are
    rectangular; this is intentional and is the shape transition a future
    directional environment sweep must carry to its neighboring corners.
    """

    if getattr(grown_edge, "ndim", None) != 3:
        raise ValueError("dynamic covariant move requires a rank-3 grown edge")
    if int(grown_edge.shape[0]) != int(left_factor.shape[0]):
        raise ValueError("grown edge left index must match the left boundary factor")
    if int(grown_edge.shape[2]) != int(right_factor.shape[0]):
        raise ValueError("grown edge right index must match the right boundary factor")

    # Local import keeps this module usable as the low-level shape/checkpoint
    # contract while avoiding a package-level dependency cycle.
    from .ctmrg_gauge import select_covariant_dynamic_boundary_frame

    primal, dual, report = select_covariant_dynamic_boundary_frame(
        xp,
        left_factor,
        right_factor,
        requested_dim,
        relative_singular_floor=relative_singular_floor,
    )
    new_left = dual.T @ left_factor
    new_right = primal.T @ right_factor
    new_edge = xp.einsum("ia,idj,jb->adb", primal, grown_edge, dual)
    report = dict(report)
    report.update(
        {
            "move": "dynamic-covariant-bilinear-boundary-projection",
            "input_left_shape": [int(size) for size in left_factor.shape],
            "input_right_shape": [int(size) for size in right_factor.shape],
            "input_edge_shape": [int(size) for size in grown_edge.shape],
            "output_left_shape": [int(size) for size in new_left.shape],
            "output_right_shape": [int(size) for size in new_right.shape],
            "output_edge_shape": [int(size) for size in new_edge.shape],
        }
    )
    return new_left, new_right, new_edge, report


def _transpose(xp: Any, value: Any, axes: tuple[int, ...]) -> Any:
    if getattr(xp, "__name__", "") == "torch":
        return value.permute(*axes)
    return value.transpose(axes)


def _host_scalar(value: Any) -> float:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    try:
        value = value.get()
    except AttributeError:
        pass
    return float(value)


def normalize_dynamic_ctm_environment(xp: Any, environment: DynamicCTMEnvironment) -> DynamicCTMEnvironment:
    """Normalize every rectangular boundary tensor without changing its shape.

    The scale convention intentionally matches the existing square CTMRG
    `_renormalize` path: max-absolute scaling, rather than an L2 norm, keeps
    full-rank dynamic moves numerically comparable to the established map.
    """

    def normalize(value: Any) -> Any:
        scale = max(_host_scalar(xp.max(xp.abs(value))), 1e-30)
        return value / scale

    return DynamicCTMEnvironment(
        *(normalize(value) for value in environment.tensors()),
        dimensions=environment.dimensions,
        map_id=environment.map_id,
    )


def apply_dynamic_ctm_move(
    xp: Any,
    environment: DynamicCTMEnvironment,
    double_layer: Any,
    direction: str,
    requested_dim: int,
    *,
    relative_singular_floor: float = 1e-12,
    normalize: bool = True,
) -> tuple[DynamicCTMEnvironment, dict[str, Any]]:
    """Apply one real directional rectangular CTMRG absorption.

    The contractions mirror the index ordering of the existing square
    ``_left_move``/``_right_move``/``_top_move``/``_bottom_move`` routines.
    Only the retained boundary side is allowed to change; the neighboring
    corner dimensions are carried explicitly by ``BoundaryDimensions``.
    """

    direction = str(direction).lower()
    if direction not in {"left", "right", "top", "bottom"}:
        raise ValueError("dynamic CTMRG direction must be left, right, top, or bottom")
    if getattr(double_layer, "ndim", None) != 4:
        raise ValueError("dynamic CTMRG move requires a rank-4 double layer")
    d2 = int(double_layer.shape[0])
    dims = environment.dimensions

    if direction == "left":
        c1_g = xp.einsum("ab,buc->auc", environment.C1, environment.T1).reshape(
            -1, environment.T1.shape[2]
        )
        c4_g = xp.einsum("gh,hdi->gdi", environment.C4, environment.T3).reshape(
            -1, environment.T3.shape[2]
        )
        t4_g = xp.einsum("alg,udlr->augdr", environment.T4, double_layer)
        t4_g = _transpose(xp, t4_g, (0, 1, 4, 2, 3)).reshape(c1_g.shape[0], d2, c4_g.shape[0])
        c1, c4, t4, report = apply_dynamic_covariant_bilinear_move(
            xp, c1_g, c4_g, t4_g, requested_dim, relative_singular_floor=relative_singular_floor
        )
        next_environment = DynamicCTMEnvironment(
            c1,
            environment.C2,
            environment.C3,
            c4,
            environment.T1,
            environment.T2,
            environment.T3,
            t4,
            dimensions=BoundaryDimensions(
                top=dims.top, left=report["retained_dim"], bottom=dims.bottom, right=dims.right
            ),
            map_id=environment.map_id,
        )
    elif direction == "right":
        c2_g = xp.einsum("ce,buc->eub", environment.C2, environment.T1).reshape(
            -1, environment.T1.shape[0]
        )
        c3_g = xp.einsum("im,hdi->mdh", environment.C3, environment.T3).reshape(
            -1, environment.T3.shape[0]
        )
        t2_g = xp.einsum("erm,udlr->eumdl", environment.T2, double_layer)
        t2_g = _transpose(xp, t2_g, (0, 1, 4, 2, 3)).reshape(c2_g.shape[0], d2, c3_g.shape[0])
        c2, c3, t2, report = apply_dynamic_covariant_bilinear_move(
            xp, c2_g, c3_g, t2_g, requested_dim, relative_singular_floor=relative_singular_floor
        )
        c2 = _transpose(xp, c2, (1, 0))
        c3 = _transpose(xp, c3, (1, 0))
        next_environment = DynamicCTMEnvironment(
            environment.C1,
            c2,
            c3,
            environment.C4,
            environment.T1,
            t2,
            environment.T3,
            environment.T4,
            dimensions=BoundaryDimensions(
                top=dims.top, left=dims.left, bottom=dims.bottom, right=report["retained_dim"]
            ),
            map_id=environment.map_id,
        )
    elif direction == "top":
        c1_g = xp.einsum("ab,alg->blg", environment.C1, environment.T4).reshape(
            -1, environment.T4.shape[2]
        )
        c2_g = xp.einsum("ce,erm->crm", environment.C2, environment.T2).reshape(
            -1, environment.T2.shape[2]
        )
        t1_g = xp.einsum("buc,udlr->bcdlr", environment.T1, double_layer)
        t1_g = _transpose(xp, t1_g, (0, 3, 2, 1, 4)).reshape(c1_g.shape[0], d2, c2_g.shape[0])
        c1, c2, t1, report = apply_dynamic_covariant_bilinear_move(
            xp, c1_g, c2_g, t1_g, requested_dim, relative_singular_floor=relative_singular_floor
        )
        c1 = _transpose(xp, c1, (1, 0))
        next_environment = DynamicCTMEnvironment(
            c1,
            c2,
            environment.C3,
            environment.C4,
            t1,
            environment.T2,
            environment.T3,
            environment.T4,
            dimensions=BoundaryDimensions(
                top=report["retained_dim"], left=dims.left, bottom=dims.bottom, right=dims.right
            ),
            map_id=environment.map_id,
        )
    else:
        c4_g = _transpose(
            xp,
            xp.einsum("gh,alg->hal", environment.C4, environment.T4),
            (0, 2, 1),
        ).reshape(-1, environment.T4.shape[0])
        c3_g = xp.einsum("im,erm->ire", environment.C3, environment.T2).reshape(
            -1, environment.T2.shape[0]
        )
        t3_g = xp.einsum("hdi,udlr->hiulr", environment.T3, double_layer)
        t3_g = _transpose(xp, t3_g, (0, 3, 2, 1, 4)).reshape(c4_g.shape[0], d2, c3_g.shape[0])
        c4, c3, t3, report = apply_dynamic_covariant_bilinear_move(
            xp, c4_g, c3_g, t3_g, requested_dim, relative_singular_floor=relative_singular_floor
        )
        c4 = _transpose(xp, c4, (1, 0))
        next_environment = DynamicCTMEnvironment(
            environment.C1,
            environment.C2,
            c3,
            c4,
            environment.T1,
            environment.T2,
            t3,
            environment.T4,
            dimensions=BoundaryDimensions(
                top=dims.top, left=dims.left, bottom=report["retained_dim"], right=dims.right
            ),
            map_id=environment.map_id,
        )

    report = dict(report)
    report["direction"] = direction
    report["dimensions_before"] = dims.to_dict()
    report["dimensions_after"] = next_environment.dimensions.to_dict()
    if normalize:
        next_environment = normalize_dynamic_ctm_environment(xp, next_environment)
        report["normalized"] = True
    else:
        report["normalized"] = False
    return next_environment, report


def run_dynamic_ctm_sweep(
    xp: Any,
    environment: DynamicCTMEnvironment,
    double_layer: Any,
    requested_dim: int,
    *,
    directions: tuple[str, ...] = ("left", "right", "top", "bottom"),
    relative_singular_floor: float = 1e-12,
    normalize: bool = True,
) -> tuple[DynamicCTMEnvironment, dict[str, Any]]:
    """Run a bounded one-site four-direction dynamic boundary sweep."""

    current = environment
    reports: list[dict[str, Any]] = []
    for direction in directions:
        current, report = apply_dynamic_ctm_move(
            xp,
            current,
            double_layer,
            direction,
            requested_dim,
            relative_singular_floor=relative_singular_floor,
            normalize=normalize,
        )
        reports.append(report)
    return current, {
        "schema": "quantum-circuit/ctmrg-dynamic-sweep-v1",
        "performed": True,
        "directions": list(directions),
        "requested_dim": int(requested_dim),
        "reports": reports,
        "dimensions_initial": environment.dimensions.to_dict(),
        "dimensions_final": current.dimensions.to_dict(),
        "all_passed": bool(all(item.get("passed", False) for item in reports)),
    }
