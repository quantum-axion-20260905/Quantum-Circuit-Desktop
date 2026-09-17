from __future__ import annotations

from typing import Any

from .base import PluginInfo
from .fermion import map_fermion_terms
from .lattice import lattice_graph
from .models import FermionAction, FermionMappingPayload, FermionOperator, FermionTerm, HubbardPayload


def _operator(mode: int, action: FermionAction) -> FermionOperator:
    return FermionOperator(mode=mode, action=action)


def build_hubbard_hamiltonian(payload: HubbardPayload) -> dict[str, Any]:
    """Build a spinful Hubbard Hamiltonian and map it with Jordan–Wigner."""
    graph = lattice_graph(payload)
    fermion_terms: list[FermionTerm] = []

    def add_number(mode: int, coefficient: float, label: str) -> None:
        fermion_terms.append(FermionTerm(
            operators=[_operator(mode, "create"), _operator(mode, "annihilate")],
            coefficient=coefficient,
            label=label,
        ))

    for edge in graph["edges"]:
        left_site, right_site = edge["source"], edge["target"]
        for spin, suffix in ((0, "up"), (1, "down")):
            left_mode = 2 * left_site + spin
            right_mode = 2 * right_site + spin
            fermion_terms.extend((
                FermionTerm(
                    operators=[_operator(left_mode, "create"), _operator(right_mode, "annihilate")],
                    coefficient=-payload.hopping,
                    label=f"hopping {suffix}",
                ),
                FermionTerm(
                    operators=[_operator(right_mode, "create"), _operator(left_mode, "annihilate")],
                    coefficient=-payload.hopping,
                    label=f"hopping {suffix}",
                ),
            ))

    for site in range(payload.n_sites):
        up, down = 2 * site, 2 * site + 1
        if payload.onsite_u:
            fermion_terms.append(FermionTerm(
                operators=[
                    _operator(up, "create"), _operator(up, "annihilate"),
                    _operator(down, "create"), _operator(down, "annihilate"),
                ],
                coefficient=payload.onsite_u,
                label="onsite U",
            ))
        if payload.chemical_potential:
            add_number(up, -payload.chemical_potential, "chemical potential up")
            add_number(down, -payload.chemical_potential, "chemical potential down")

    mapped = map_fermion_terms(FermionMappingPayload(
        n_modes=payload.n_sites * 2,
        terms=fermion_terms,
    ))
    max_locality = max((len(term["paulis"]) for term in mapped["terms"]), default=0)
    tebd_ready = max_locality <= 64
    warnings = list(mapped["warnings"])
    if max_locality > 2 and tebd_ready:
        warnings.append(
            "Jordan-Wigner parity strings are evolved with a swap-aware parity-CX network; "
            "check bond-dimension and time-step convergence"
        )
    if not tebd_ready:
        warnings.append(
            "Jordan-Wigner parity strings exceed max_term_locality=64; "
            "use expectation or exact ground-state validation until a larger-string path is added"
        )
    return {
        **graph,
        "material": "hubbard",
        "hopping": payload.hopping,
        "onsite_u": payload.onsite_u,
        "chemical_potential": payload.chemical_potential,
        "n_qubits": payload.n_sites * 2,
        "fermion_terms": [term.model_dump(mode="json") for term in fermion_terms],
        "terms": mapped["terms"],
        "complex_terms": mapped["complex_terms"],
        "expectation_ready": mapped["expectation_ready"],
        "tebd_ready": tebd_ready,
        "max_pauli_locality": max_locality,
        "max_imaginary_coefficient": mapped["max_imaginary_coefficient"],
        "warnings": warnings,
    }


class HubbardMaterialsPlugin:
    info = PluginInfo(
        id="hubbard-materials",
        name="Hubbard materials",
        version="0.1.0",
        description="Spinful Hubbard Hamiltonians mapped to sparse Pauli terms on 1D/2D/3D lattices.",
        capabilities=("hubbard", "materials", "jordan-wigner", "energy", "ground-state", "tebd"),
    )

    def preview_lattice(self, payload: HubbardPayload) -> dict[str, Any]:
        return lattice_graph(payload)

    def build_hubbard(self, payload: HubbardPayload) -> dict[str, Any]:
        return build_hubbard_hamiltonian(payload)
