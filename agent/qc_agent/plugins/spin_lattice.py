from __future__ import annotations

from typing import Any

from .base import PluginInfo
from .lattice import build_ctmrg_spin_payload, build_spin_hamiltonian, lattice_graph
from .models import CTMRGSpinModelPayload, LatticeHamiltonianPayload, LatticeSpec


class SpinLatticePlugin:
    info = PluginInfo(
        id="spin-lattice",
        name="Spin lattice physics",
        version="0.1.0",
        description="Rectangular 1D/2D/3D spin lattices with sparse Pauli Hamiltonians.",
        capabilities=("lattice-preview", "ising", "heisenberg", "xxz", "energy", "tebd", "peps", "ctmrg"),
    )

    def preview_lattice(self, payload: LatticeSpec) -> dict[str, Any]:
        return lattice_graph(payload)

    def build_hamiltonian(self, payload: LatticeHamiltonianPayload) -> dict[str, Any]:
        return build_spin_hamiltonian(payload)

    def build_ctmrg(self, payload: CTMRGSpinModelPayload) -> dict[str, Any]:
        return build_ctmrg_spin_payload(
            payload,
            initial_state=payload.initial_state,
            environment_bond_dim=payload.environment_bond_dim,
            iterations=payload.iterations,
            tolerance=payload.tolerance,
            environment_damping=payload.environment_damping,
            gauge_validation=payload.gauge_validation,
            gauge_validation_tolerance=payload.gauge_validation_tolerance,
            gauge_preconditioner=payload.gauge_preconditioner,
            gauge_preconditioner_iterations=payload.gauge_preconditioner_iterations,
            environment_sector_policy=payload.environment_sector_policy,
        ).model_dump(mode="json")
