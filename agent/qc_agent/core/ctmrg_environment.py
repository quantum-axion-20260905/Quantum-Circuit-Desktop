"""Versioned CTMRG environment-map contracts.

The numerical move functions remain in :mod:`ctmrg`; this module records the
ordering and gauge convention that those moves implement.  Keeping the map
identity explicit prevents a checkpoint or a comparison study from silently
mixing environments produced by incompatible projector/orientation rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


ENVIRONMENT_MAP_SCHEMA = "quantum-circuit/ctmrg-environment-map-v1"
GAUGE_CONVENTION = "paired-inverse-transpose-virtual-bonds-v1"
_PROJECTOR_POLICIES = {"half-density", "full-svd", "biorthogonal-bilinear"}


@dataclass(frozen=True)
class CTMRGEnvironmentMap:
    """Serializable identity for one CTMRG environment update convention."""

    map_id: str
    projector_policy: Literal["half-density", "full-svd", "biorthogonal-bilinear"]
    schema: str = ENVIRONMENT_MAP_SCHEMA
    virtual_leg_order: tuple[str, ...] = ("physical", "up", "down", "left", "right")
    corner_order: tuple[str, ...] = ("C1", "C2", "C3", "C4")
    edge_order: tuple[str, ...] = ("T1", "T2", "T3", "T4")
    fused_edge_order: tuple[str, ...] = ("boundary", "ket", "bra", "boundary")
    directional_update_order: tuple[str, ...] = ("left", "right", "top", "bottom")
    gauge_convention: str = GAUGE_CONVENTION

    def __post_init__(self) -> None:
        if self.schema != ENVIRONMENT_MAP_SCHEMA:
            raise ValueError("unsupported CTMRG environment-map schema")
        if self.projector_policy not in _PROJECTOR_POLICIES:
            raise ValueError(f"unsupported CTMRG projector policy {self.projector_policy!r}")
        if len(self.virtual_leg_order) != 5:
            raise ValueError("CTMRG virtual-leg order must contain physical plus four virtual legs")
        if len(self.corner_order) != 4 or len(self.edge_order) != 4:
            raise ValueError("CTMRG environment must declare four corners and four edges")
        if self.gauge_convention != GAUGE_CONVENTION:
            raise ValueError("unsupported CTMRG virtual-gauge convention")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "map_id": self.map_id,
            "projector_policy": self.projector_policy,
            "virtual_leg_order": list(self.virtual_leg_order),
            "corner_order": list(self.corner_order),
            "edge_order": list(self.edge_order),
            "fused_edge_order": list(self.fused_edge_order),
            "directional_update_order": list(self.directional_update_order),
            "gauge_convention": self.gauge_convention,
        }


def environment_map_for(projector_policy: str) -> CTMRGEnvironmentMap:
    """Return the versioned map contract for a supported projector policy."""

    policy = str(projector_policy)
    if policy == "half-density":
        return CTMRGEnvironmentMap(
            map_id="ctmrg-half-density-v1",
            projector_policy="half-density",
        )
    if policy == "full-svd":
        return CTMRGEnvironmentMap(
            map_id="ctmrg-full-svd-biorthogonal-v1",
            projector_policy="full-svd",
        )
    if policy == "biorthogonal-bilinear":
        return CTMRGEnvironmentMap(
            map_id="ctmrg-biorthogonal-bilinear-v1",
            projector_policy="biorthogonal-bilinear",
        )
    raise ValueError(f"unsupported CTMRG projector policy {policy!r}")


def validate_environment_map(value: Any, expected: CTMRGEnvironmentMap) -> None:
    """Reject a checkpoint/result map that does not match the current solver."""

    if not isinstance(value, dict):
        raise ValueError("CTMRG checkpoint environment_map metadata is missing")
    actual = value.get("map_id")
    if actual != expected.map_id:
        raise ValueError(
            f"CTMRG checkpoint environment map {actual!r} does not match the requested map {expected.map_id!r}"
        )
    if value.get("schema") != expected.schema:
        raise ValueError("CTMRG checkpoint environment-map schema does not match the request")
    if value.get("projector_policy") != expected.projector_policy:
        raise ValueError("CTMRG checkpoint environment-map projector policy does not match the request")
    for key in (
        "virtual_leg_order",
        "corner_order",
        "edge_order",
        "fused_edge_order",
        "directional_update_order",
        "gauge_convention",
    ):
        if value.get(key) != expected.to_dict()[key]:
            raise ValueError(f"CTMRG checkpoint environment-map field {key!r} does not match the request")
