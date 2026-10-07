from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class PluginInfo:
    id: str
    name: str
    version: str
    description: str
    capabilities: tuple[str, ...]
    api_version: str = "1.0"


@runtime_checkable
class DomainPlugin(Protocol):
    info: PluginInfo


class BaseDomainPlugin:
    """Convenient base class for domain plugins with optional action implementations."""

    info: PluginInfo

    def preview_lattice(self, payload: Any) -> dict[str, Any]:
        raise NotImplementedError("preview_lattice is not implemented by this plugin")

    def build_hamiltonian(self, payload: Any) -> dict[str, Any]:
        raise NotImplementedError("build_hamiltonian is not implemented by this plugin")

    def build_ctmrg(self, payload: Any) -> dict[str, Any]:
        raise NotImplementedError("build_ctmrg is not implemented by this plugin")

    def map_fermions(self, payload: Any) -> dict[str, Any]:
        raise NotImplementedError("map_fermions is not implemented by this plugin")

    def build_hubbard(self, payload: Any) -> dict[str, Any]:
        raise NotImplementedError("build_hubbard is not implemented by this plugin")


PLUGIN_ACTIONS = (
    "preview_lattice",
    "build_hamiltonian",
    "build_ctmrg",
    "map_fermions",
    "build_hubbard",
)


def plugin_actions(plugin: DomainPlugin) -> tuple[str, ...]:
    return tuple(
        action
        for action in PLUGIN_ACTIONS
        if callable(getattr(plugin, action, None))
        and getattr(type(plugin), action, None) is not getattr(BaseDomainPlugin, action, None)
    )


def validate_plugin(plugin: DomainPlugin) -> None:
    info = getattr(plugin, "info", None)
    if not isinstance(info, PluginInfo):
        raise TypeError("plugin must expose a PluginInfo instance")
    if not info.id or not info.id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("plugin id must contain only letters, numbers, hyphens, or underscores")
    if not info.version or not info.api_version:
        raise ValueError("plugin version and api_version are required")
    if len(set(info.capabilities)) != len(info.capabilities):
        raise ValueError(f"plugin {info.id} declares duplicate capabilities")
    if not plugin_actions(plugin):
        raise ValueError(f"plugin {info.id} does not expose a supported action")


def get_plugin_action(plugin: DomainPlugin, action: str) -> Any:
    if action not in PLUGIN_ACTIONS:
        raise ValueError(f"unsupported plugin action: {action}")
    handler = getattr(plugin, action, None)
    return handler if callable(handler) else None
