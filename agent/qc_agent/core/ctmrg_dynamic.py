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
import math
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
    return float(complex(value).real)


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


def _dynamic_environment_from_tensors(
    tensors: tuple[Any, ...],
    *,
    map_id: str,
) -> DynamicCTMEnvironment:
    """Infer directional dimensions from a freshly projected environment."""

    C1, C2, C3, C4, T1, T2, T3, T4 = tensors
    dimensions = BoundaryDimensions(
        top=int(C2.shape[0]),
        left=int(C1.shape[0]),
        bottom=int(C3.shape[0]),
        right=int(C2.shape[1]),
    )
    return DynamicCTMEnvironment(*tensors, dimensions=dimensions, map_id=map_id)


def apply_dynamic_ctm_two_site_move(
    xp: Any,
    environment: DynamicCTMEnvironment,
    neighbor_environment: DynamicCTMEnvironment,
    neighbor_layer: Any,
    direction: str,
    requested_dim: int,
    *,
    relative_singular_floor: float = 1e-12,
    normalize: bool = True,
) -> tuple[DynamicCTMEnvironment, dict[str, Any]]:
    """Apply one periodic neighbor absorption for a dynamic environment.

    The self environment supplies the corners and the directional grown edge;
    the neighboring environment supplies the boundary edge that is absorbed,
    matching the existing periodic ``_unit_cell_sweep`` contract.
    """

    direction = str(direction).lower()
    if direction not in {"left", "right", "top", "bottom"}:
        raise ValueError("dynamic CTMRG direction must be left, right, top, or bottom")
    if getattr(neighbor_layer, "ndim", None) != 4:
        raise ValueError("dynamic two-site CTMRG move requires a rank-4 neighbor layer")
    d2 = int(neighbor_layer.shape[0])

    if direction == "left":
        first = xp.einsum("ab,buc->auc", environment.C1, neighbor_environment.T1)
        second = xp.einsum("gh,hdi->gdi", environment.C4, neighbor_environment.T3)
        grown = xp.einsum("alg,udlr->augdr", environment.T4, neighbor_layer)
        first = first.reshape(-1, neighbor_environment.T1.shape[2])
        second = second.reshape(-1, neighbor_environment.T3.shape[2])
        grown = _transpose(xp, grown, (0, 1, 4, 2, 3)).reshape(first.shape[0], d2, second.shape[0])
        new_first, new_second, new_edge, report = apply_dynamic_covariant_bilinear_move(
            xp, first, second, grown, requested_dim, relative_singular_floor=relative_singular_floor
        )
        tensors = (
            new_first,
            environment.C2,
            environment.C3,
            new_second,
            environment.T1,
            environment.T2,
            environment.T3,
            new_edge,
        )
    elif direction == "right":
        first = xp.einsum("ce,buc->eub", environment.C2, neighbor_environment.T1)
        second = xp.einsum("im,hdi->mdh", environment.C3, neighbor_environment.T3)
        grown = xp.einsum("erm,udlr->eumdl", environment.T2, neighbor_layer)
        first = first.reshape(-1, neighbor_environment.T1.shape[0])
        second = second.reshape(-1, neighbor_environment.T3.shape[0])
        grown = _transpose(xp, grown, (0, 1, 4, 2, 3)).reshape(first.shape[0], d2, second.shape[0])
        new_first, new_second, new_edge, report = apply_dynamic_covariant_bilinear_move(
            xp, first, second, grown, requested_dim, relative_singular_floor=relative_singular_floor
        )
        tensors = (
            environment.C1,
            _transpose(xp, new_first, (1, 0)),
            _transpose(xp, new_second, (1, 0)),
            environment.C4,
            environment.T1,
            new_edge,
            environment.T3,
            environment.T4,
        )
    elif direction == "top":
        first = xp.einsum("ab,alg->blg", environment.C1, neighbor_environment.T4)
        second = xp.einsum("ce,erm->crm", environment.C2, neighbor_environment.T2)
        grown = xp.einsum("buc,udlr->bcdlr", environment.T1, neighbor_layer)
        first = first.reshape(-1, neighbor_environment.T4.shape[2])
        second = second.reshape(-1, neighbor_environment.T2.shape[2])
        grown = _transpose(xp, grown, (0, 3, 2, 1, 4)).reshape(first.shape[0], d2, second.shape[0])
        new_first, new_second, new_edge, report = apply_dynamic_covariant_bilinear_move(
            xp, first, second, grown, requested_dim, relative_singular_floor=relative_singular_floor
        )
        tensors = (
            _transpose(xp, new_first, (1, 0)),
            new_second,
            environment.C3,
            environment.C4,
            new_edge,
            environment.T2,
            environment.T3,
            environment.T4,
        )
    else:
        first = _transpose(
            xp,
            xp.einsum("gh,alg->hal", environment.C4, neighbor_environment.T4),
            (0, 2, 1),
        )
        second = xp.einsum("im,erm->ire", environment.C3, neighbor_environment.T2)
        grown = xp.einsum("hdi,udlr->hiulr", environment.T3, neighbor_layer)
        first = first.reshape(-1, neighbor_environment.T4.shape[0])
        second = second.reshape(-1, neighbor_environment.T2.shape[0])
        grown = _transpose(xp, grown, (0, 3, 2, 1, 4)).reshape(first.shape[0], d2, second.shape[0])
        new_first, new_second, new_edge, report = apply_dynamic_covariant_bilinear_move(
            xp, first, second, grown, requested_dim, relative_singular_floor=relative_singular_floor
        )
        tensors = (
            environment.C1,
            environment.C2,
            new_second,
            _transpose(xp, new_first, (1, 0)),
            environment.T1,
            environment.T2,
            new_edge,
            environment.T4,
        )

    next_environment = _dynamic_environment_from_tensors(tensors, map_id=environment.map_id)
    report = dict(report)
    report["direction"] = direction
    report["neighbor_map_id"] = neighbor_environment.map_id
    report["dimensions_before"] = environment.dimensions.to_dict()
    report["dimensions_after"] = next_environment.dimensions.to_dict()
    if normalize:
        next_environment = normalize_dynamic_ctm_environment(xp, next_environment)
        report["normalized"] = True
    else:
        report["normalized"] = False
    return next_environment, report


def run_dynamic_ctm_cell_sweep(
    xp: Any,
    environments: list[DynamicCTMEnvironment],
    layers: list[Any],
    unit_cell: tuple[int, int],
    requested_dim: int,
    *,
    directions: tuple[str, ...] = ("left", "right", "top", "bottom"),
    relative_singular_floor: float = 1e-12,
    normalize: bool = True,
) -> tuple[list[DynamicCTMEnvironment], dict[str, Any]]:
    """Run the deterministic periodic neighbor sweep for a 1x1--2x2 cell."""

    nx, ny = (int(value) for value in unit_cell)
    if nx < 1 or nx > 2 or ny < 1 or ny > 2:
        raise ValueError("dynamic CTMRG cell sweep supports only 1x1 through 2x2 cells")
    if len(environments) != nx * ny or len(layers) != nx * ny:
        raise ValueError("dynamic CTMRG cell sweep tensor/environment count mismatch")

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

    positions = {"left": 0, "right": 1, "top": 2, "bottom": 3}
    working = list(environments)
    reports: list[dict[str, Any]] = []
    for direction in directions:
        if direction not in positions:
            raise ValueError(f"unsupported dynamic cell-sweep direction {direction!r}")
        for site in range(len(working)):
            neighbor = neighbors(site)[positions[direction]]
            working[site], report = apply_dynamic_ctm_two_site_move(
                xp,
                working[site],
                working[neighbor],
                layers[neighbor],
                direction,
                requested_dim,
                relative_singular_floor=relative_singular_floor,
                normalize=normalize,
            )
            report = dict(report)
            report["site"] = site
            report["neighbor"] = neighbor
            reports.append(report)
    return working, {
        "schema": "quantum-circuit/ctmrg-dynamic-cell-sweep-v1",
        "performed": True,
        "unit_cell": [nx, ny],
        "site_count": len(working),
        "directions": list(directions),
        "requested_dim": int(requested_dim),
        "reports": reports,
        "all_passed": bool(all(item.get("passed", False) for item in reports)),
        "dimensions_final": [environment.dimensions.to_dict() for environment in working],
    }


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


def _host_values(value: Any) -> list[float]:
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
    return [float(abs(item)) for item in value.reshape(-1)]


def _dynamic_environment_residual(xp: Any, before: DynamicCTMEnvironment, after: DynamicCTMEnvironment) -> float:
    """Compare rectangular environments modulo internal boundary bases."""

    residual = 0.0 if before.dimensions == after.dimensions else 1.0

    def spectrum(value: Any) -> list[float]:
        matrix = value.reshape(value.shape[0], -1)
        singular = xp.linalg.svd(matrix, compute_uv=False)
        values = _host_values(singular)
        scale = max(values[0] if values else 0.0, 1e-30)
        return [item / scale for item in values]

    for old, new in zip(before.tensors(), after.tensors()):
        old_spectrum = spectrum(old)
        new_spectrum = spectrum(new)
        size = max(len(old_spectrum), len(new_spectrum))
        old_spectrum.extend([0.0] * (size - len(old_spectrum)))
        new_spectrum.extend([0.0] * (size - len(new_spectrum)))
        residual = max(
            residual,
            max((abs(left - right) for left, right in zip(old_spectrum, new_spectrum)), default=0.0),
        )
    return float(residual)


def run_dynamic_ctmrg_one_site(
    xp: Any,
    tensor: Any,
    environment: DynamicCTMEnvironment,
    *,
    requested_dim: int,
    iterations: int = 4,
    tolerance: float = 1e-8,
    terms: tuple[Any, ...] = (),
    interactions: tuple[Any, ...] = (),
    directions: tuple[str, ...] = ("left", "right", "top", "bottom"),
    relative_singular_floor: float = 1e-12,
    normalize: bool = True,
) -> tuple[dict[str, Any], DynamicCTMEnvironment]:
    """Run the bounded dynamic one-site CTMRG research path.

    This runner deliberately returns the raw ``DynamicCTMEnvironment`` beside
    its serializable result envelope so callers can checkpoint the exact state.
    It supports normalized one-site observables and nearest-neighbor
    one-site-cell interactions through the established contraction routines;
    multi-site periodic cells and optimization admission remain separate.
    """

    if getattr(tensor, "ndim", None) != 5:
        raise ValueError("dynamic one-site CTMRG requires a rank-5 iPEPS tensor")
    if int(tensor.shape[0]) != 2:
        raise ValueError("dynamic one-site CTMRG currently supports physical_bond_dim=2")
    if int(iterations) < 1:
        raise ValueError("dynamic one-site CTMRG iterations must be positive")
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError("dynamic one-site CTMRG tolerance must be finite and positive")

    from .contracts import ConvergencePoint, ConvergenceReport, ResearchResult, TruncationReport
    from .ctmrg import _double_layer, _environment_contraction, _interaction_expectation, _term_expectation

    layer = _double_layer(xp, tensor)
    current = environment
    convergence_points: list[Any] = []
    sweep_reports: list[dict[str, Any]] = []
    converged = False
    discarded_total = 0.0
    for iteration in range(1, int(iterations) + 1):
        before = current
        current, sweep_report = run_dynamic_ctm_sweep(
            xp,
            current,
            layer,
            requested_dim,
            directions=directions,
            relative_singular_floor=relative_singular_floor,
            normalize=normalize,
        )
        residual = _dynamic_environment_residual(xp, before, current)
        discarded = sum(float(item.get("discarded_weight", 0.0)) for item in sweep_report["reports"])
        discarded_total += discarded
        sweep_report = dict(sweep_report)
        sweep_report["iteration"] = iteration
        sweep_report["residual"] = residual
        sweep_reports.append(sweep_report)
        convergence_points.append(ConvergencePoint(
            iteration=iteration,
            residual=residual,
            environment_dim=max(current.dimensions.to_dict().values()),
            discarded_weight=discarded,
        ))
        if residual <= float(tolerance) and iteration > 1:
            converged = True
            break

    norm_value = _host_scalar(_environment_contraction(xp, current, layer))
    observables: list[dict[str, Any]] = []
    energy = 0.0
    energy_complete = True
    for term in terms:
        value = float(_term_expectation(xp, current, tensor, term))
        coefficient = float(getattr(term, "coefficient", 1.0))
        observables.append({
            "paulis": {int(index): str(pauli) for index, pauli in term.paulis.items()},
            "coefficient": coefficient,
            "value": value,
            "contribution": coefficient * value,
        })
        energy += coefficient * value
    interaction_values: list[dict[str, Any]] = []
    for interaction in interactions:
        value = _interaction_expectation(
            xp,
            current,
            tensor,
            list(interaction.displacement),
            str(interaction.left_pauli),
            str(interaction.right_pauli),
        )
        coefficient = float(getattr(interaction, "coefficient", 1.0))
        interaction_values.append({
            "displacement": list(interaction.displacement),
            "left_pauli": str(interaction.left_pauli),
            "right_pauli": str(interaction.right_pauli),
            "coefficient": coefficient,
            "value": None if value is None else float(value),
            "contribution": None if value is None else coefficient * float(value),
        })
        if value is None:
            energy_complete = False
        else:
            energy += coefficient * float(value)

    final_residual = float(convergence_points[-1].residual if convergence_points else math.inf)
    research_result = ResearchResult(
        status="needs_review",
        method="ctmrg-dynamic-covariant-v2",
        representation="ipeps-dynamic-boundary",
        metrics={
            "norm": norm_value,
            "energy": float(energy),
            "energy_complete": bool(energy_complete),
            "residual": final_residual,
            "requested_environment_dim": int(requested_dim),
            "retained_dimensions": current.dimensions.to_dict(),
        },
        truncation=TruncationReport(
            discarded_weight=float(discarded_total),
            max_environment_dim=max(current.dimensions.to_dict().values()),
        ),
        convergence=ConvergenceReport(
            converged=converged,
            classification="converged" if converged else "unconverged",
            criterion=f"dynamic environment spectral residual <= {float(tolerance):.3e}",
            points=convergence_points,
            warnings=[] if converged else ["bounded dynamic one-site sweep did not reach its residual tolerance"],
        ),
        warnings=[
            "dynamic covariant CTMRG is an experimental one-site research path",
            "production admission still requires multi-site periodic and transfer-gap gates",
        ],
        limitations=[
            "this runner supports one-site cells only",
            "dynamic retained dimensions are not yet integrated into optimization or public CTMRG payload policy",
            "a passing normalized observable is not a thermodynamic-limit convergence proof",
        ],
        provenance={
            "sweep_schema": "quantum-circuit/ctmrg-dynamic-sweep-v1",
            "relative_singular_floor": float(relative_singular_floor),
        },
        details={
            "environment_shape_manifest": current.shape_manifest(),
            "sweep_reports": sweep_reports,
        },
    )
    result = {
        "schema": "quantum-circuit/research-result-v1",
        "status": "needs_review",
        "method": "ctmrg-dynamic-covariant-v2",
        "representation": "ipeps-dynamic-boundary",
        "norm": norm_value,
        "energy": float(energy),
        "energy_complete": bool(energy_complete),
        "observables": observables,
        "interactions": interaction_values,
        "converged": converged,
        "residual": final_residual,
        "environment_shape_manifest": current.shape_manifest(),
        "dynamic_sweep": sweep_reports,
        "research_result": research_result.to_dict(),
    }
    return result, current


def run_dynamic_ctmrg_cell(
    xp: Any,
    tensors: list[Any],
    environments: list[DynamicCTMEnvironment],
    unit_cell: tuple[int, int],
    *,
    requested_dim: int,
    iterations: int = 4,
    tolerance: float = 1e-8,
    terms: tuple[Any, ...] = (),
    interactions: tuple[Any, ...] = (),
    directions: tuple[str, ...] = ("left", "right", "top", "bottom"),
    relative_singular_floor: float = 1e-12,
    normalize: bool = True,
) -> tuple[dict[str, Any], list[DynamicCTMEnvironment]]:
    """Run the bounded dynamic periodic 1x1--2x2 research path."""

    nx, ny = (int(value) for value in unit_cell)
    site_count = nx * ny
    if len(tensors) != site_count or len(environments) != site_count:
        raise ValueError("dynamic CTMRG cell runner tensor/environment count mismatch")
    if any(getattr(tensor, "ndim", None) != 5 or int(tensor.shape[0]) != 2 for tensor in tensors):
        raise ValueError("dynamic CTMRG cell runner requires physical-dimension-2 rank-5 tensors")
    if int(iterations) < 1:
        raise ValueError("dynamic CTMRG cell runner iterations must be positive")
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError("dynamic CTMRG cell runner tolerance must be finite and positive")

    from .contracts import ConvergencePoint, ConvergenceReport, ResearchResult, TruncationReport
    from .ctmrg import (
        _double_layer,
        _environment_contraction,
        _environment_diagnostics,
        _interaction_expectation_cell,
        _term_expectation,
    )

    layers = [_double_layer(xp, tensor) for tensor in tensors]
    current = list(environments)
    convergence_points: list[Any] = []
    sweep_reports: list[dict[str, Any]] = []
    converged = False
    discarded_total = 0.0
    for iteration in range(1, int(iterations) + 1):
        before = list(current)
        current, sweep_report = run_dynamic_ctm_cell_sweep(
            xp,
            current,
            layers,
            unit_cell,
            requested_dim,
            directions=directions,
            relative_singular_floor=relative_singular_floor,
            normalize=normalize,
        )
        residual = max(
            _dynamic_environment_residual(xp, old, new)
            for old, new in zip(before, current)
        )
        discarded = sum(float(item.get("discarded_weight", 0.0)) for item in sweep_report["reports"])
        discarded_total += discarded
        sweep_report = dict(sweep_report)
        sweep_report["iteration"] = iteration
        sweep_report["residual"] = residual
        sweep_reports.append(sweep_report)
        convergence_points.append(ConvergencePoint(
            iteration=iteration,
            residual=residual,
            environment_dim=max(
                max(environment.dimensions.to_dict().values()) for environment in current
            ),
            discarded_weight=discarded,
        ))
        if residual <= float(tolerance) and iteration > 1:
            converged = True
            break

    norms = [
        _host_scalar(_environment_contraction(xp, environment, layer))
        for environment, layer in zip(current, layers)
    ]
    norm_value = sum(norms) / max(1, len(norms))
    observables: list[dict[str, Any]] = []
    energy = 0.0
    energy_complete = True
    for term in terms:
        if len(term.paulis) > 1:
            raise ValueError("dynamic cell runner onsite terms must act on at most one site")
        site = int(next(iter(term.paulis), 0))
        if site < 0 or site >= site_count:
            raise ValueError("dynamic cell runner term site exceeds the unit cell")
        value = float(_term_expectation(xp, current[site], tensors[site], term))
        coefficient = float(getattr(term, "coefficient", 1.0))
        observables.append({
            "site": site,
            "paulis": {int(index): str(pauli) for index, pauli in term.paulis.items()},
            "coefficient": coefficient,
            "value": value,
            "contribution": coefficient * value,
        })
        energy += coefficient * value

    interaction_values: list[dict[str, Any]] = []
    for interaction in interactions:
        left_site = int(interaction.left_site)
        right_site = int(interaction.right_site)
        if not 0 <= left_site < site_count or not 0 <= right_site < site_count:
            raise ValueError("dynamic cell runner interaction site exceeds the unit cell")
        value = _interaction_expectation_cell(
            xp,
            current,
            tensors,
            left_site,
            right_site,
            list(interaction.displacement),
            str(interaction.left_pauli),
            str(interaction.right_pauli),
        )
        coefficient = float(getattr(interaction, "coefficient", 1.0))
        interaction_values.append({
            "left_site": left_site,
            "right_site": right_site,
            "displacement": list(interaction.displacement),
            "left_pauli": str(interaction.left_pauli),
            "right_pauli": str(interaction.right_pauli),
            "coefficient": coefficient,
            "value": None if value is None else float(value),
            "contribution": None if value is None else coefficient * float(value),
        })
        if value is None:
            energy_complete = False
        else:
            energy += coefficient * float(value)

    diagnostics = _environment_diagnostics(xp, current)
    final_residual = float(convergence_points[-1].residual if convergence_points else math.inf)
    classification = "converged" if converged else "unconverged"
    if any(value is None for value in diagnostics["correlation_lengths_by_site"]):
        classification = "degenerate-needs-review"
    research_result = ResearchResult(
        status="needs_review",
        method="ctmrg-dynamic-covariant-v2",
        representation="ipeps-dynamic-boundary",
        metrics={
            "norm": norm_value,
            "energy": float(energy),
            "energy_complete": bool(energy_complete),
            "residual": final_residual,
            "fixed_point_classification": classification,
            "requested_environment_dim": int(requested_dim),
            "retained_dimensions": [environment.dimensions.to_dict() for environment in current],
            **diagnostics,
        },
        truncation=TruncationReport(
            discarded_weight=float(discarded_total),
            max_environment_dim=max(
                max(environment.dimensions.to_dict().values()) for environment in current
            ),
        ),
        convergence=ConvergenceReport(
            converged=converged,
            classification=classification,
            criterion=f"dynamic cell spectral residual <= {float(tolerance):.3e} with transfer-gap diagnostics",
            points=convergence_points,
            warnings=[] if converged else ["bounded dynamic cell sweep did not reach its residual tolerance"],
        ),
        warnings=[
            "dynamic covariant CTMRG is an experimental bounded periodic path",
            "production admission still requires broader unit cells and optimization gates",
        ],
        limitations=[
            "this runner is bounded to 1x1--2x2 periodic cells",
            "dynamic retained dimensions are not yet integrated into public CTMRG payload policy",
            "transfer-gap diagnostics are reported but do not prove thermodynamic-limit convergence",
        ],
        provenance={
            "sweep_schema": "quantum-circuit/ctmrg-dynamic-cell-sweep-v1",
            "relative_singular_floor": float(relative_singular_floor),
        },
        details={
            "unit_cell": [nx, ny],
            "environment_shape_manifests": [environment.shape_manifest() for environment in current],
            "sweep_reports": sweep_reports,
        },
    )
    result = {
        "schema": "quantum-circuit/research-result-v1",
        "status": "needs_review",
        "method": "ctmrg-dynamic-covariant-v2",
        "representation": "ipeps-dynamic-boundary",
        "unit_cell": [nx, ny],
        "unit_cell_sites": site_count,
        "norm": norm_value,
        "energy": float(energy),
        "energy_complete": bool(energy_complete),
        "observables": observables,
        "interactions": interaction_values,
        "converged": converged,
        "residual": final_residual,
        "fixed_point_classification": classification,
        "environment_diagnostics": diagnostics,
        "environment_shape_manifests": [environment.shape_manifest() for environment in current],
        "dynamic_cell_sweep": sweep_reports,
        "research_result": research_result.to_dict(),
    }
    return result, current
