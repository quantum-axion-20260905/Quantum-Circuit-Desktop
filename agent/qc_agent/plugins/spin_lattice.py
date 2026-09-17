from __future__ import annotations

from typing import Any

from .base import PluginInfo
from .lattice import build_spin_hamiltonian, lattice_graph
from .models import LatticeHamiltonianPayload, LatticeSpec


class SpinLatticePlugin:
    info = PluginInfo(
        id="spin-lattice",
        name="Spin lattice physics",
        version="0.1.0",
        description="Rectangular 1D/2D/3D spin lattices with sparse Pauli Hamiltonians.",
        capabilities=("lattice-preview", "ising", "heisenberg", "xxz", "energy", "tebd", "peps"),
    )

    def preview_lattice(self, payload: LatticeSpec) -> dict[str, Any]:
        return lattice_graph(payload)

    def build_hamiltonian(self, payload: LatticeHamiltonianPayload) -> dict[str, Any]:
        return build_spin_hamiltonian(payload)
