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

    Corner ordering follows the existing ``CTMEnvironment`` convention:
    ``C1`` is top-left, ``C2`` top-right, ``C3`` bottom-right, and ``C4``
    bottom-left.  ``T1``/``T3`` run along the top/bottom sides and
    ``T2``/``T4`` along the right/left sides.  The fused double-layer leg is
    the middle edge axis and is intentionally not constrained here.
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
            "C1": (dims.top, dims.left),
            "C2": (dims.top, dims.right),
            "C3": (dims.bottom, dims.right),
            "C4": (dims.bottom, dims.left),
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
