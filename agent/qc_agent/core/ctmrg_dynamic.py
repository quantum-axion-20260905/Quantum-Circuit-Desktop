"""Rectangular boundary-state contracts for the next CTMRG environment map.

The production CTMRG path currently stores a square fixed-``chi`` environment.
This module defines the shape contract for a future dynamically retained
boundary without changing that path prematurely.  Keeping the contract in a
small backend-neutral module makes NumPy/CuPy/Torch kernels and checkpoint
code depend on the same directional vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any

from .ctmrg_admission import dynamic_ctmrg_research_gate


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


class DynamicCellCompatibilityError(ValueError):
    """Raised when neighboring rectangular environments cannot be contracted.

    A dynamic retained dimension is a shared boundary-space contract, not an
    independent per-site crop.  If one site has already reduced a row/column
    boundary while its periodic neighbor has not, continuing would either
    broadcast incorrectly or fail later inside an opaque ``einsum`` error.
    This exception keeps the rejection explicit and serializable at the HTTP
    boundary.
    """

    def __init__(
        self,
        *,
        direction: str,
        environment_dimensions: dict[str, int],
        neighbor_dimensions: dict[str, int],
        mismatches: dict[str, tuple[int, int]],
    ) -> None:
        self.direction = str(direction)
        self.environment_dimensions = dict(environment_dimensions)
        self.neighbor_dimensions = dict(neighbor_dimensions)
        self.mismatches = {
            str(name): (int(values[0]), int(values[1]))
            for name, values in mismatches.items()
        }
        mismatch_text = ", ".join(
            f"{name}={left}/{right}" for name, (left, right) in self.mismatches.items()
        )
        super().__init__(
            "dynamic CTMRG periodic boundary incompatibility for "
            f"{self.direction!r}: neighboring retained dimensions disagree ({mismatch_text}); "
            "the multi-site dynamic path requires a shared compatible row/column frame"
        )


def validate_dynamic_two_site_compatibility(
    environment: "DynamicCTMEnvironment",
    neighbor_environment: "DynamicCTMEnvironment",
    direction: str,
) -> None:
    """Validate the boundary indices required by one periodic absorption."""

    direction = str(direction).lower()
    if direction not in {"left", "right", "top", "bottom"}:
        raise ValueError("dynamic CTMRG direction must be left, right, top, or bottom")
    if environment.map_id != neighbor_environment.map_id:
        raise ValueError(
            "dynamic CTMRG neighboring environments use incompatible map identifiers"
        )
    self_dims = environment.dimensions.to_dict()
    neighbor_dims = neighbor_environment.dimensions.to_dict()
    shared_names = ("top", "bottom") if direction in {"left", "right"} else ("left", "right")
    mismatches = {
        name: (self_dims[name], neighbor_dims[name])
        for name in shared_names
        if self_dims[name] != neighbor_dims[name]
    }
    if mismatches:
        raise DynamicCellCompatibilityError(
            direction=direction,
            environment_dimensions=self_dims,
            neighbor_dimensions=neighbor_dims,
            mismatches=mismatches,
        )


def _validate_dynamic_cell_compatibility(
    environments: list["DynamicCTMEnvironment"],
    unit_cell: tuple[int, int],
    directions: tuple[str, ...],
) -> None:
    """Validate every periodic neighbor pair at a completed sweep boundary."""

    nx, ny = (int(value) for value in unit_cell)

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
    for direction in directions:
        for site in range(len(environments)):
            validate_dynamic_two_site_compatibility(
                environments[site],
                environments[neighbors(site)[positions[direction]]],
                direction,
            )


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
    validate_dynamic_two_site_compatibility(environment, neighbor_environment, direction)
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
    _allow_synchronized_retry: bool = True,
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
    try:
        for direction in directions:
            if direction not in positions:
                raise ValueError(f"unsupported dynamic cell-sweep direction {direction!r}")
            for site in range(len(working)):
                neighbor = neighbors(site)[positions[direction]]
                effective_requested_dim = min(
                    int(requested_dim),
                    min(working[site].dimensions.to_dict().values()),
                    min(working[neighbor].dimensions.to_dict().values()),
                )
                working[site], report = apply_dynamic_ctm_two_site_move(
                    xp,
                    working[site],
                    working[neighbor],
                    layers[neighbor],
                    direction,
                    effective_requested_dim,
                    relative_singular_floor=relative_singular_floor,
                    normalize=normalize,
                )
                report = dict(report)
                report["site"] = site
                report["neighbor"] = neighbor
                report["requested_dim_effective"] = effective_requested_dim
                reports.append(report)
        _validate_dynamic_cell_compatibility(working, (nx, ny), directions)
    except DynamicCellCompatibilityError:
        if not _allow_synchronized_retry:
            raise
        partial_minimum = min(
            dimension
            for environment in working
            for dimension in environment.dimensions.to_dict().values()
        )
        synchronized_dim = min(int(requested_dim), int(partial_minimum))
        if synchronized_dim >= int(requested_dim):
            raise
        synchronized, synchronized_report = run_dynamic_ctm_cell_sweep(
            xp,
            environments,
            layers,
            unit_cell,
            synchronized_dim,
            directions=directions,
            relative_singular_floor=relative_singular_floor,
            normalize=normalize,
            _allow_synchronized_retry=False,
        )
        synchronized_report = dict(synchronized_report)
        synchronized_report["synchronized_retry"] = True
        synchronized_report["original_requested_dim"] = int(requested_dim)
        synchronized_report["synchronized_requested_dim"] = int(synchronized_dim)
        synchronized_report["synchronization_policy"] = (
            "restart-from-cell-boundary-at-minimum-admissible-dimension"
        )
        return synchronized, synchronized_report
    return working, {
        "schema": "quantum-circuit/ctmrg-dynamic-cell-sweep-v1",
        "performed": True,
        "unit_cell": [nx, ny],
        "site_count": len(working),
        "directions": list(directions),
        "requested_dim": int(requested_dim),
        "synchronized_retry": False,
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
        effective_requested_dim = min(int(requested_dim), min(current.dimensions.to_dict().values()))
        current, report = apply_dynamic_ctm_move(
            xp,
            current,
            double_layer,
            direction,
            effective_requested_dim,
            relative_singular_floor=relative_singular_floor,
            normalize=normalize,
        )
        report = dict(report)
        report["requested_dim_effective"] = effective_requested_dim
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
        if getattr(xp, "__name__", "") == "torch":
            singular = xp.linalg.svdvals(matrix)
        else:
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
    start_iteration: int = 0,
    reference_validation: dict[str, Any] | None = None,
    boundary_mps_validation: dict[str, Any] | None = None,
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
    if int(start_iteration) < 0:
        raise ValueError("dynamic CTMRG cell runner start_iteration must be non-negative")
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
    reference_validation = dict(reference_validation or {
        "performed": False,
        "passed": False,
        "reason": "independent reference is attached by the payload runner",
    })
    boundary_mps_validation = dict(boundary_mps_validation or {
        "requested": False,
        "performed": False,
        "passed": True,
        "reason": "finite-cylinder boundary-MPS cross-check was not requested",
    })

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
        sweep_report["iteration"] = int(start_iteration) + iteration
        sweep_report["residual"] = residual
        sweep_reports.append(sweep_report)
        convergence_points.append(ConvergencePoint(
            iteration=int(start_iteration) + iteration,
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
    synchronized_sector_retry = any(
        bool(report.get("synchronized_retry")) for report in sweep_reports
    )
    if synchronized_sector_retry:
        classification = "synchronized-sector-needs-review"
    if any(value is None for value in diagnostics["correlation_lengths_by_site"]):
        classification = "degenerate-needs-review"
    research_gate = dynamic_ctmrg_research_gate(
        unit_cell=(nx, ny),
        converged=converged,
        residual=final_residual,
        tolerance=float(tolerance),
        energy_complete=energy_complete,
        transfer_gaps=list(diagnostics["transfer_gap_by_site"]),
        synchronized_sector_retry=synchronized_sector_retry,
        reference_validation=reference_validation,
        boundary_mps_validation=boundary_mps_validation,
    )
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
            *(
                [
                    "a cell-wide retained-sector synchronization restart was used; compare against a higher-sector run before interpreting observables"
                ]
                if synchronized_sector_retry else []
            ),
        ],
        limitations=[
            "this runner is bounded to 1x1--2x2 periodic cells",
            "dynamic retained dimensions are not yet integrated into public CTMRG payload policy",
            "transfer-gap diagnostics are reported but do not prove thermodynamic-limit convergence",
            *(
                [
                    "synchronized retained-sector fallback may reduce the requested boundary dimension and is not a production fixed-point proof"
                ]
                if synchronized_sector_retry else []
            ),
        ],
        provenance={
            "sweep_schema": "quantum-circuit/ctmrg-dynamic-cell-sweep-v1",
            "relative_singular_floor": float(relative_singular_floor),
        },
        details={
            "unit_cell": [nx, ny],
            "environment_shape_manifests": [environment.shape_manifest() for environment in current],
            "sweep_reports": sweep_reports,
            "research_gate": research_gate,
            "reference_validation": reference_validation,
            "boundary_mps_validation": boundary_mps_validation,
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
        "reference_validation": reference_validation,
        "boundary_mps_validation": boundary_mps_validation,
        "research_gate": research_gate,
        "research_result": research_result.to_dict(),
    }
    return result, current


def _dynamic_reference_validation(
    payload: Any,
    tensors: list[Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Run the smallest independent reference available for a dynamic payload."""

    from .ctmrg_reference import (
        analytic_ghz_reference,
        finite_periodic_peps_reference,
        finite_product_reference,
    )

    onsite_values = [float(item["value"]) for item in result.get("observables", [])]
    interaction_values = [item.get("value") for item in result.get("interactions", [])]
    energy = float(result.get("energy", 0.0))
    tolerance = max(float(payload.tolerance) * 10.0, 1e-6)
    reference = finite_product_reference(
        payload,
        tensors,
        onsite_values,
        interaction_values,
        energy,
        tolerance=tolerance,
    )
    if not reference.get("performed"):
        reference = analytic_ghz_reference(
            payload,
            tensors,
            onsite_values,
            interaction_values,
            energy,
            tolerance=tolerance,
        )
    if not reference.get("performed"):
        reference = finite_periodic_peps_reference(
            payload,
            tensors,
            onsite_values,
            interaction_values,
            energy,
            tolerance=tolerance,
        )
    return reference


def run_dynamic_ctmrg_payload(
    xp: Any,
    payload: Any,
    *,
    checkpoint_path: str | None = None,
    resume_from: str | None = None,
) -> tuple[dict[str, Any], list[DynamicCTMEnvironment]]:
    """Execute the explicit dynamic CTMRG backend contract for a payload.

    This is the public-facing seam for the experimental backend. It reuses the
    existing ``CTMRGPayload`` tensor construction and interaction contracts,
    but requires callers to opt into this function explicitly; the established
    square ``run_ctmrg`` route is not changed implicitly.
    """

    unit_cell = tuple(int(value) for value in payload.unit_cell)
    if len(unit_cell) != 2 or any(value < 1 or value > 2 for value in unit_cell):
        raise ValueError("dynamic CTMRG payload supports only 1x1 through 2x2 unit cells")
    if getattr(payload, "environment_sector_policy", "single") != "single":
        raise ValueError("dynamic CTMRG payload requires environment_sector_policy='single'")
    if getattr(payload, "optimization", "none") != "none":
        raise ValueError("dynamic CTMRG payload is contraction-only; optimization is not admitted")

    from .checkpoints import load_dynamic_ctm_checkpoint, save_dynamic_ctm_checkpoint
    from .contracts import CheckpointManifest
    from .ctmrg import _build_tensors, _double_layer, _initialize_environment

    tensors = _build_tensors(xp, payload)
    layers = [_double_layer(xp, tensor) for tensor in tensors]
    cell_sites = len(tensors)
    initialization_regularizer = 1e-9 if str(payload.dtype) == "complex128" else 1e-6
    initialization_sector_seed = getattr(payload, "dynamic_initialization_seed", None)
    selected_resume = resume_from or getattr(payload, "resume_from", None)
    selected_checkpoint = checkpoint_path or getattr(payload, "checkpoint_path", None)
    request_payload = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else dict(payload.__dict__)
    request_sha256 = hashlib.sha256(
        json.dumps(request_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    initial_source = "fresh"
    start_iteration = 0
    loaded_manifest: dict[str, Any] | None = None
    if selected_resume:
        loaded_manifest, loaded = load_dynamic_ctm_checkpoint(selected_resume, xp)
        if loaded_manifest.get("request_sha256") != request_sha256:
            raise ValueError("dynamic CTMRG checkpoint request digest does not match the payload")
        checkpoint_unit_cell = loaded_manifest.get("metadata", {}).get("unit_cell")
        if checkpoint_unit_cell != list(unit_cell):
            raise ValueError("dynamic CTMRG checkpoint unit cell does not match the payload")
        environments = loaded if isinstance(loaded, list) else [loaded]
        if len(environments) != cell_sites:
            raise ValueError("dynamic CTMRG checkpoint site count does not match the payload unit cell")
        start_iteration = int(loaded_manifest.get("step", 0))
        initial_source = "checkpoint"
    else:
        environments = [
            DynamicCTMEnvironment(
                *_initialize_environment(
                    xp,
                    layer,
                    int(payload.environment_bond_dim),
                    sector_seed=None if initialization_sector_seed is None else int(initialization_sector_seed),
                    regularizer=initialization_regularizer,
                ).tensors(),
                dimensions=BoundaryDimensions.uniform(int(payload.environment_bond_dim)),
            )
            for layer in layers
        ]

    requested_environment_dim = min(
        int(payload.environment_bond_dim),
        min(
            dimension
            for environment in environments
            for dimension in environment.dimensions.to_dict().values()
        ),
    )
    result, final = run_dynamic_ctmrg_cell(
        xp,
        tensors,
        environments,
        unit_cell,
        requested_dim=requested_environment_dim,
        iterations=int(payload.iterations),
        tolerance=float(payload.tolerance),
        terms=tuple(payload.terms),
        interactions=tuple(payload.interactions),
        start_iteration=start_iteration,
    )
    result = dict(result)
    reference_validation = _dynamic_reference_validation(payload, tensors, result)
    result["reference_validation"] = reference_validation
    boundary_mps_validation: dict[str, Any] = {
        "requested": bool(getattr(payload, "boundary_mps_reference", False)),
        "performed": False,
        "passed": True,
        "reason": "finite-cylinder boundary-MPS cross-check was not requested",
    }
    if boundary_mps_validation["requested"]:
        from .ctmrg_boundary_mps import run_boundary_mps_reference

        boundary_mps_validation = dict(run_boundary_mps_reference(
            tensors,
            payload,
            ctmrg_energy=float(result["energy"]),
            ctmrg_onsite=[float(item["value"]) for item in result.get("observables", [])],
            ctmrg_interactions=[item.get("value") for item in result.get("interactions", [])],
        ))
        boundary_mps_validation["requested"] = True
        boundary_mps_validation["tolerance"] = max(float(payload.tolerance) * 10.0, 1e-6)
        boundary_mps_validation["passed"] = bool(
            boundary_mps_validation.get("performed")
            and boundary_mps_validation.get("max_abs_error") is not None
            and float(boundary_mps_validation["max_abs_error"]) <= boundary_mps_validation["tolerance"]
        )
    result["boundary_mps_validation"] = boundary_mps_validation
    research_gate = dynamic_ctmrg_research_gate(
        unit_cell=unit_cell,
        converged=bool(result["converged"]),
        residual=float(result["residual"]),
        tolerance=float(payload.tolerance),
        energy_complete=bool(result["energy_complete"]),
        transfer_gaps=list(result["environment_diagnostics"]["transfer_gap_by_site"]),
        synchronized_sector_retry=any(
            bool(report.get("synchronized_retry"))
            for report in result.get("dynamic_cell_sweep", [])
        ),
        reference_validation=reference_validation,
        boundary_mps_validation=boundary_mps_validation,
    )
    result["research_gate"] = research_gate
    research_result = dict(result.get("research_result", {}))
    research_result["details"] = dict(research_result.get("details", {}))
    research_result["details"]["reference_validation"] = reference_validation
    research_result["details"]["research_gate"] = research_gate
    research_result["details"]["boundary_mps_validation"] = boundary_mps_validation
    research_result["details"]["environment_initialization_sector_seed"] = initialization_sector_seed
    research_result["metrics"] = dict(research_result.get("metrics", {}))
    research_result["metrics"]["reference_max_abs_error"] = reference_validation.get("max_abs_error")
    result["research_result"] = research_result
    result["backend"] = "tensor-network-ctmrg-dynamic"
    result["initial_environment_source"] = initial_source
    result["environment_initialization_regularizer"] = initialization_regularizer
    result["environment_initialization_sector_seed"] = initialization_sector_seed
    result["requested_environment_dim"] = requested_environment_dim

    completed_step = start_iteration + len(result["dynamic_cell_sweep"])
    if selected_checkpoint:
        manifest = CheckpointManifest(
            checkpoint_id=f"dynamic-ctmrg-{request_sha256[:12]}-iteration-{completed_step}",
            request_sha256=request_sha256,
            method="ipeps-ctmrg-contraction",
            representation="ipeps-dynamic-boundary",
            dtype=str(payload.dtype),
            device="cuda" if hasattr(xp, "cuda") else "cpu",
            step=completed_step,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={
                "completed_iterations": completed_step,
                "unit_cell": list(unit_cell),
                "requested_environment_dim": requested_environment_dim,
                "environment_initialization_regularizer": initialization_regularizer,
                "environment_initialization_sector_seed": initialization_sector_seed,
                "initial_environment_source": initial_source,
            },
        )
        saved_manifest = save_dynamic_ctm_checkpoint(selected_checkpoint, final, manifest)
        result["checkpoint"] = {
            "resumable": True,
            "path": str(selected_checkpoint),
            "manifest": saved_manifest,
            "resumed": initial_source == "checkpoint",
        }
    else:
        result["checkpoint"] = {
            "resumable": False,
            "reason": "set checkpoint_path to persist and resume the dynamic environment",
            "resumed": initial_source == "checkpoint",
        }
    return result, final


def run_dynamic_ctmrg_convergence_study(
    xp: Any,
    payload: Any,
    environment_bond_dims: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    """Compare independent dynamic contractions across bounded environment sizes.

    Each point starts from a fresh environment and has checkpoint/resume
    disabled.  The study is deliberately a comparison tool, not a composite
    solver: a small energy delta cannot override an unresolved transfer gap or
    a shared-sector synchronization fallback.
    """

    normalized_dims = [int(value) for value in environment_bond_dims]
    if not normalized_dims:
        raise ValueError("dynamic CTMRG convergence study requires at least one environment dimension")
    if len(normalized_dims) > 8:
        raise ValueError("dynamic CTMRG convergence study is limited to eight points")
    if any(value < 1 or value > 128 for value in normalized_dims):
        raise ValueError("dynamic CTMRG environment dimensions must be between 1 and 128")
    if len(set(normalized_dims)) != len(normalized_dims):
        raise ValueError("dynamic CTMRG environment dimensions must be unique")
    if getattr(payload, "optimization", "none") != "none":
        raise ValueError("dynamic CTMRG convergence studies require optimization='none'")

    points: list[dict[str, Any]] = []
    previous_energy: float | None = None
    previous_observables: list[float] | None = None
    previous_interactions: list[float | None] | None = None
    for dimension in normalized_dims:
        point_payload = payload.model_copy(update={
            "environment_bond_dim": dimension,
            "checkpoint_path": None,
            "resume_from": None,
        })
        result, _ = run_dynamic_ctmrg_payload(xp, point_payload)
        energy = float(result["energy"])
        observables = [float(item["value"]) for item in result.get("observables", [])]
        interactions = [
            None if item.get("value") is None else float(item["value"])
            for item in result.get("interactions", [])
        ]
        observable_delta = None
        if previous_observables is not None and len(previous_observables) == len(observables):
            observable_delta = max(
                (abs(current - previous) for current, previous in zip(observables, previous_observables)),
                default=0.0,
            )
        interaction_delta = None
        if previous_interactions is not None and len(previous_interactions) == len(interactions):
            interaction_delta = max(
                (
                    abs(float(current) - float(previous))
                    for current, previous in zip(interactions, previous_interactions)
                    if current is not None and previous is not None
                ),
                default=0.0,
            )
        sweeps = list(result.get("dynamic_cell_sweep", []))
        gaps = list(result["environment_diagnostics"].get("transfer_gap_by_site", []))
        points.append({
            "environment_bond_dim": dimension,
            "requested_environment_dim": int(result.get("requested_environment_dim", dimension)),
            "initialization_sector_seed": result.get("environment_initialization_sector_seed"),
            "energy": energy,
            "energy_complete": bool(result["energy_complete"]),
            "energy_delta": None if previous_energy is None else energy - previous_energy,
            "energy_abs_delta": None if previous_energy is None else abs(energy - previous_energy),
            "observable_max_abs_delta": observable_delta,
            "interaction_max_abs_delta": interaction_delta,
            "residual": float(result["residual"]),
            "converged": bool(result["converged"]),
            "fixed_point_classification": result.get("fixed_point_classification", "unconverged"),
            "transfer_gap_by_site": gaps,
            "minimum_transfer_gap": min((float(value) for value in gaps if value is not None), default=None),
            "retained_dimensions": result.get("environment_shape_manifests", []),
            "synchronized_sector_retry": any(bool(item.get("synchronized_retry")) for item in sweeps),
            "reference_validation": result.get("reference_validation"),
            "boundary_mps_validation": result.get("boundary_mps_validation"),
            "research_gate": result.get("research_gate"),
            "research_gate_status": result.get("research_gate", {}).get("status"),
            "research_gate_blocking_reasons": list(result.get("research_gate", {}).get("blocking_reasons", [])),
        })
        previous_energy = energy
        previous_observables = observables
        previous_interactions = interactions

    blocking_reasons = sorted({
        str(reason)
        for point in points
        for reason in point["research_gate_blocking_reasons"]
    })
    reference_errors = [
        float(point["reference_validation"]["max_abs_error"])
        for point in points
        if isinstance(point.get("reference_validation"), dict)
        and point["reference_validation"].get("max_abs_error") is not None
    ]
    return {
        "schema": "quantum-circuit/ctmrg-dynamic-convergence-study-v1",
        "status": "needs_review",
        "method": "ipeps-ctmrg-dynamic-convergence-study",
        "optimization": "none",
        "unit_cell": list(payload.unit_cell),
        "unit_cell_sites": math.prod(payload.unit_cell),
        "points": points,
        "reference_summary": {
            "performed_points": sum(
                1 for point in points
                if isinstance(point.get("reference_validation"), dict)
                and point["reference_validation"].get("performed")
            ),
            "passed_points": sum(
                1 for point in points
                if isinstance(point.get("reference_validation"), dict)
                and point["reference_validation"].get("passed") is True
            ),
            "maximum_reference_error": max(reference_errors, default=None),
        },
        "research_gate_summary": {
            "status": "needs_review",
            "production_ready": False,
            "points": len(points),
            "passed_points": sum(point["research_gate_status"] == "passed" for point in points),
            "review_points": sum(point["research_gate_status"] != "passed" for point in points),
            "blocking_reasons": blocking_reasons,
        },
        "energy_summary": {
            "minimum": min((point["energy"] for point in points), default=None),
            "maximum": max((point["energy"] for point in points), default=None),
            "absolute_range": (
                max(point["energy"] for point in points) - min(point["energy"] for point in points)
                if points else None
            ),
        },
        "warnings": [
            "each point is a fresh bounded dynamic contraction",
            "energy stability does not override unresolved transfer-gap or reference gates",
            "this study does not establish thermodynamic-limit convergence",
        ],
    }


def run_dynamic_ctmrg_sector_study(
    xp: Any,
    payload: Any,
    initialization_seeds: list[int | None] | tuple[int | None, ...],
) -> dict[str, Any]:
    """Compare fresh dynamic fixed-point probes across deterministic sectors."""

    normalized_seeds = [None if seed is None else int(seed) for seed in initialization_seeds]
    if not normalized_seeds:
        raise ValueError("dynamic CTMRG sector study requires at least one initialization seed")
    if len(normalized_seeds) > 8:
        raise ValueError("dynamic CTMRG sector study is limited to eight points")
    if any(seed is not None and (seed < 0 or seed > 1048575) for seed in normalized_seeds):
        raise ValueError("dynamic initialization seeds must be between 0 and 1048575")
    if len({"default" if seed is None else seed for seed in normalized_seeds}) != len(normalized_seeds):
        raise ValueError("dynamic initialization seeds must be unique")
    if getattr(payload, "environment_sector_policy", "single") != "single":
        raise ValueError("dynamic sector studies require environment_sector_policy='single'")
    if getattr(payload, "optimization", "none") != "none":
        raise ValueError("dynamic sector studies require optimization='none'")

    points: list[dict[str, Any]] = []
    for seed in normalized_seeds:
        point_payload = payload.model_copy(update={
            "dynamic_initialization_seed": seed,
            "checkpoint_path": None,
            "resume_from": None,
        })
        result, _ = run_dynamic_ctmrg_payload(xp, point_payload)
        sweeps = list(result.get("dynamic_cell_sweep", []))
        gaps = list(result["environment_diagnostics"].get("transfer_gap_by_site", []))
        reference = result.get("reference_validation") or {}
        research_gate = result.get("research_gate") or {}
        points.append({
            "initialization_sector_seed": seed,
            "environment_bond_dim": int(payload.environment_bond_dim),
            "energy": float(result["energy"]),
            "energy_complete": bool(result["energy_complete"]),
            "residual": float(result["residual"]),
            "converged": bool(result["converged"]),
            "fixed_point_classification": result.get("fixed_point_classification", "unconverged"),
            "transfer_gap_by_site": gaps,
            "minimum_transfer_gap": min((float(value) for value in gaps if value is not None), default=None),
            "synchronized_sector_retry": any(bool(item.get("synchronized_retry")) for item in sweeps),
            "retained_dimensions": result.get("environment_shape_manifests", []),
            "reference_validation": reference,
            "research_gate": research_gate,
            "research_gate_status": research_gate.get("status"),
            "research_gate_blocking_reasons": list(research_gate.get("blocking_reasons", [])),
        })

    energies = [float(point["energy"]) for point in points]
    minimum_gaps = [
        float(point["minimum_transfer_gap"])
        for point in points
        if point["minimum_transfer_gap"] is not None
    ]
    blocking_reasons = sorted({
        str(reason)
        for point in points
        for reason in point["research_gate_blocking_reasons"]
    })
    return {
        "schema": "quantum-circuit/ctmrg-dynamic-sector-study-v1",
        "status": "needs_review",
        "method": "ipeps-ctmrg-dynamic-sector-study",
        "optimization": "none",
        "unit_cell": list(payload.unit_cell),
        "unit_cell_sites": math.prod(payload.unit_cell),
        "environment_bond_dim": int(payload.environment_bond_dim),
        "points": points,
        "sector_summary": {
            "requested_seeds": normalized_seeds,
            "energy_minimum": min(energies, default=None),
            "energy_maximum": max(energies, default=None),
            "energy_absolute_range": max(energies) - min(energies) if energies else None,
            "minimum_transfer_gap": min(minimum_gaps, default=None),
            "maximum_transfer_gap": max(minimum_gaps, default=None),
            "synchronized_retry_points": sum(bool(point["synchronized_sector_retry"]) for point in points),
        },
        "research_gate_summary": {
            "status": "needs_review",
            "production_ready": False,
            "points": len(points),
            "passed_points": sum(point["research_gate_status"] == "passed" for point in points),
            "review_points": sum(point["research_gate_status"] != "passed" for point in points),
            "blocking_reasons": blocking_reasons,
        },
        "warnings": [
            "sector points are fresh deterministic initializations, not a symmetry-sector ensemble average",
            "sector sensitivity is evidence for review and does not select a physically preferred fixed point automatically",
            "this study does not establish thermodynamic-limit convergence",
        ],
    }
