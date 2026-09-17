from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Operation = Literal["samples", "selected_amplitudes", "estimate", "simulate", "expectation", "evolve", "ground_state", "dmrg", "peps"]


@dataclass(frozen=True)
class BackendDescriptor:
    name: str
    device: str
    available: bool
    operations: tuple[Operation, ...]
    performance_comparable: bool
    description: str


def catalog(*, gpu_available: bool, tensor_network_available: bool) -> list[BackendDescriptor]:
    return [
        BackendDescriptor(
            name="reference",
            device="CPU",
            available=True,
            operations=("samples", "selected_amplitudes", "expectation"),
            performance_comparable=False,
            description="Deterministic Python reference simulator for validation and small circuits.",
        ),
        BackendDescriptor(
            name="statevector",
            device="CUDA",
            available=gpu_available,
            operations=("samples", "simulate", "expectation"),
            performance_comparable=True,
            description="CuPy statevector simulator and GPU sampler.",
        ),
        BackendDescriptor(
            name="tensor-network",
            device="CUDA",
            available=gpu_available and tensor_network_available,
            operations=("samples", "selected_amplitudes", "estimate", "expectation", "evolve", "dmrg", "peps"),
            performance_comparable=True,
            description="GPU MPS simulator with bounded bond dimension; exact contraction remains available as an opt-in method.",
        ),
        BackendDescriptor(
            name="exact-diagonalization",
            device="CUDA",
            available=gpu_available,
            operations=("ground_state",),
            performance_comparable=True,
            description="Small dense GPU eigensolver for Hamiltonian validation and ground-state reference energies.",
        ),
    ]


def resolve_run_backend(
    requested: str,
    operation: Operation,
    *,
    gpu_available: bool,
    tensor_network_available: bool,
) -> str:
    """Resolve a user-facing backend choice to an executable backend name."""

    available = {item.name: item for item in catalog(
        gpu_available=gpu_available,
        tensor_network_available=tensor_network_available,
    )}

    if requested == "auto":
        if operation == "samples":
            return "statevector" if available["statevector"].available else "reference"
        if operation == "selected_amplitudes":
            return "tensor-network" if available["tensor-network"].available else "reference"
        if operation == "simulate":
            return "statevector"
        if operation == "estimate":
            return "tensor-network" if available["tensor-network"].available else "reference"
        if operation == "expectation":
            return "tensor-network" if available["tensor-network"].available else "reference"
        if operation == "evolve":
            return "tensor-network"
        if operation == "ground_state":
            return "exact-diagonalization"
        if operation == "dmrg":
            return "tensor-network"
        if operation == "peps":
            return "tensor-network"

    if requested == "reference":
        if operation not in available["reference"].operations:
            raise ValueError(f"reference backend does not support {operation}")
        return "reference"

    if requested == "tensor-network":
        if operation not in available["tensor-network"].operations:
            raise ValueError(f"tensor-network backend does not support {operation}")
        if not available["tensor-network"].available:
            raise ValueError("tensor-network backend requires CUDA")
        return "tensor-network"

    if requested in ("exact", "exact-diagonalization"):
        if operation not in available["exact-diagonalization"].operations:
            raise ValueError(f"exact-diagonalization backend does not support {operation}")
        if not available["exact-diagonalization"].available:
            raise ValueError("exact-diagonalization backend requires CUDA")
        return "exact-diagonalization"

    if requested in ("statevector", "cupy-statevector"):
        if operation not in available["statevector"].operations:
            raise ValueError(f"statevector backend does not support {operation}")
        if not available["statevector"].available:
            raise ValueError("statevector backend requires CUDA")
        return "statevector"

    raise ValueError(f"unknown backend: {requested}")
