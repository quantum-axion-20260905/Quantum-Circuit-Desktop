"""Template demonstrating how to build a new domain plugin for Quantum Circuit Agent.

To add a new physics model, algorithm, or problem generator:
1. Inherit from ``BaseDomainPlugin``.
2. Define ``info = PluginInfo(...)`` with unique id and capabilities.
3. Implement required actions (e.g. ``build_hamiltonian``, ``preview_lattice``).
4. Register the plugin with ``register(MyCustomPlugin())``.
"""

from __future__ import annotations

from typing import Any

from .base import BaseDomainPlugin, PluginInfo
from .models import LatticeHamiltonianPayload, LatticeSpec


class CustomQuantumModelPlugin(BaseDomainPlugin):
    """Example plugin for custom Hamiltonians or quantum systems."""

    info = PluginInfo(
        id="custom-quantum-model",
        name="Custom Quantum Model Plugin",
        version="1.0.0",
        description="Template plugin demonstrating extensible Hamiltonian and lattice generation.",
        capabilities=("custom-hamiltonian", "lattice-preview"),
        api_version="1.0",
    )

    def preview_lattice(self, payload: LatticeSpec) -> dict[str, Any]:
        """Generate site coordinates and interaction edges."""
        sites = [{"id": i, "x": float(i), "y": 0.0, "z": 0.0} for i in range(payload.n_sites)]
        edges = [
            {"source": i, "target": i + 1, "weight": 1.0}
            for i in range(payload.n_sites - 1)
        ]
        return {
            "dimensions": payload.dimensions,
            "boundary": payload.boundary,
            "sites": sites,
            "edges": edges,
        }

    def build_hamiltonian(self, payload: LatticeHamiltonianPayload) -> dict[str, Any]:
        """Build Pauli Hamiltonian terms for simulation or DMRG."""
        terms = []
        n_qubits = payload.lattice.n_sites
        # Example 1D XX coupling
        for i in range(n_qubits - 1):
            terms.append({
                "label": f"XX_{i}_{i+1}",
                "coefficient": 1.0,
                "paulis": {i: "X", i + 1: "X"},
            })
        return {
            "model": payload.model,
            "n_qubits": n_qubits,
            "terms": terms,
        }
