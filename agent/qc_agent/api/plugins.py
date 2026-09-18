from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ..plugins.base import get_plugin_action
from ..plugins.fermion import map_fermion_terms
from ..plugins.models import (
    FermionMappingPayload,
    HubbardPayload,
    CTMRGSpinModelPayload,
    LatticeHamiltonianPayload,
    LatticeSpec,
)
from ..plugins.registry import get_plugin


router = APIRouter(prefix="/plugins", tags=["domain-plugins"])


def _plugin_or_http(plugin_id: str):
    plugin = get_plugin(plugin_id)
    if plugin is None:
        raise HTTPException(status_code=404, detail=f"unknown domain plugin: {plugin_id}")
    return plugin


def _action_or_http(plugin: Any, action: str):
    handler = get_plugin_action(plugin, action)
    if handler is None:
        raise HTTPException(status_code=501, detail=f"plugin {plugin.info.id} does not implement {action}")
    return handler


@router.post("/{plugin_id}/lattice")
def preview_lattice(plugin_id: str, payload: LatticeSpec) -> dict[str, Any]:
    plugin = _plugin_or_http(plugin_id)
    return _action_or_http(plugin, "preview_lattice")(payload)


@router.post("/{plugin_id}/hamiltonian")
def build_hamiltonian(plugin_id: str, payload: LatticeHamiltonianPayload) -> dict[str, Any]:
    plugin = _plugin_or_http(plugin_id)
    return _action_or_http(plugin, "build_hamiltonian")(payload)


@router.post("/{plugin_id}/ctmrg")
def build_ctmrg(plugin_id: str, payload: CTMRGSpinModelPayload) -> dict[str, Any]:
    plugin = _plugin_or_http(plugin_id)
    return _action_or_http(plugin, "build_ctmrg")(payload)


@router.post("/{plugin_id}/fermion_mapping")
def map_fermions(plugin_id: str, payload: FermionMappingPayload) -> dict[str, Any]:
    plugin = _plugin_or_http(plugin_id)
    handler = get_plugin_action(plugin, "map_fermions")
    if handler is not None:
        return handler(payload)
    if payload.mapping == "jordan_wigner" and "jordan-wigner" in plugin.info.capabilities:
        return map_fermion_terms(payload)
    raise HTTPException(status_code=501, detail=f"plugin {plugin.info.id} does not implement fermion mapping")


@router.post("/{plugin_id}/hubbard")
def build_hubbard(plugin_id: str, payload: HubbardPayload) -> dict[str, Any]:
    plugin = _plugin_or_http(plugin_id)
    return _action_or_http(plugin, "build_hubbard")(payload)
