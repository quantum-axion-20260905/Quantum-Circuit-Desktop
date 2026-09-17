from __future__ import annotations

from dataclasses import asdict
from importlib import metadata
import threading
from typing import Any

from .base import DomainPlugin, get_plugin_action, plugin_actions, validate_plugin

_PLUGINS: dict[str, DomainPlugin] = {}
_LOCK = threading.RLock()
_ENTRY_POINTS_LOADED = False


def register(plugin: DomainPlugin) -> DomainPlugin:
    validate_plugin(plugin)
    plugin_id = plugin.info.id
    with _LOCK:
        if plugin_id in _PLUGINS:
            raise ValueError(f"domain plugin already registered: {plugin_id}")
        _PLUGINS[plugin_id] = plugin
    return plugin


def get_plugin(plugin_id: str) -> DomainPlugin | None:
    load_entry_points()
    with _LOCK:
        return _PLUGINS.get(plugin_id)


def get_action(plugin_id: str, action: str):
    plugin = get_plugin(plugin_id)
    return None if plugin is None else get_plugin_action(plugin, action)


def load_entry_points() -> None:
    """Load optional third-party plugins from ``qc_agent.plugins`` entry points."""
    global _ENTRY_POINTS_LOADED
    with _LOCK:
        if _ENTRY_POINTS_LOADED:
            return
        _ENTRY_POINTS_LOADED = True
    try:
        entry_points = metadata.entry_points()
        selected = entry_points.select(group="qc_agent.plugins") if hasattr(entry_points, "select") else entry_points.get("qc_agent.plugins", [])
        for entry_point in selected:
            try:
                loaded = entry_point.load()
                plugin = loaded() if isinstance(loaded, type) else loaded
                if plugin.info.id not in _PLUGINS:
                    register(plugin)
            except Exception:
                # One optional package must not make the built-in agent fail.
                continue
    except Exception:
        return


def catalog() -> list[dict[str, Any]]:
    load_entry_points()
    with _LOCK:
        return [
            {**asdict(plugin.info), "actions": list(plugin_actions(plugin))}
            for plugin in _PLUGINS.values()
        ]
