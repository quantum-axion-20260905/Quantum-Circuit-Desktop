from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..core.contracts import CapabilityError


Operation = Literal["samples", "selected_amplitudes", "estimate", "simulate", "expectation", "evolve", "ground_state", "dmrg", "peps", "ctmrg"]
MPSMethod = Literal["dmrg", "tebd", "tdvp", "vumps", "ctmrg"]
MethodOperation = Literal["ground_state", "evolve", "ctmrg"]
MethodStatus = Literal["available", "unavailable", "planned"]


@dataclass(frozen=True)
class BackendDescriptor:
    name: str
    device: str
    available: bool
    operations: tuple[Operation, ...]
    performance_comparable: bool
    description: str


@dataclass(frozen=True)
class MethodCapability:
    """Algorithm-level capability, distinct from its execution backend."""

    id: str
    method: MPSMethod
    backend: str
    representation: str
    operation: MethodOperation
    available: bool
    status: MethodStatus
    description: str
    limitations: tuple[str, ...] = ()


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
            operations=("samples", "selected_amplitudes", "estimate", "expectation", "evolve", "dmrg", "peps", "ctmrg"),
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


def method_catalog(*, gpu_available: bool, tensor_network_available: bool) -> list[MethodCapability]:
    """Return algorithm capabilities without conflating planned and usable methods."""

    tensor_network_ready = bool(gpu_available and tensor_network_available)
    runtime_status: MethodStatus = "available" if tensor_network_ready else "unavailable"
    return [
        MethodCapability(
            id="mps-dmrg",
            method="dmrg",
            backend="tensor-network",
            representation="mps",
            operation="ground_state",
            available=tensor_network_ready,
            status=runtime_status,
            description="Finite two-site DMRG with bounded sweeps, residual and variance diagnostics.",
            limitations=("finite open 1D systems", "bounded bond dimension"),
        ),
        MethodCapability(
            id="mps-tebd",
            method="tebd",
            backend="tensor-network",
            representation="mps",
            operation="evolve",
            available=tensor_network_ready,
            status=runtime_status,
            description="Finite-MPS real/imaginary-time evolution with explicit timestep and truncation diagnostics.",
            limitations=("short controlled evolutions", "bounded bond dimension"),
        ),
        MethodCapability(
            id="mps-tdvp",
            method="tdvp",
            backend="tensor-network",
            representation="mps",
            operation="evolve",
            available=False,
            status="planned",
            description="Separate TDVP interface reserved for projector-splitting MPS evolution.",
            limitations=("solver implementation is not available", "no fallback to TEBD or DMRG"),
        ),
        MethodCapability(
            id="mps-vumps",
            method="vumps",
            backend="tensor-network",
            representation="uniform-mps",
            operation="ground_state",
            available=False,
            status="planned",
            description="Separate VUMPS interface reserved for uniform/infinite-MPS ground states.",
            limitations=("solver implementation is not available", "no fallback to DMRG"),
        ),
        MethodCapability(
            id="ipeps-ctmrg",
            method="ctmrg",
            backend="tensor-network",
            representation="ipeps",
            operation="ctmrg",
            available=tensor_network_ready,
            status=runtime_status,
            description="Bounded one-site infinite-2D iPEPS CTMRG contraction with explicit corner/edge environment convergence.",
            limitations=("one-site unit cell only", "product-state ansatz; no variational tensor optimization", "no fallback to finite boundary-MPS"),
        ),
    ]


def resolve_method_capability(
    requested: str,
    *,
    gpu_available: bool,
    tensor_network_available: bool,
) -> MethodCapability:
    """Resolve an algorithm explicitly, rejecting unavailable methods safely."""

    selected = next((item for item in method_catalog(
        gpu_available=gpu_available,
        tensor_network_available=tensor_network_available,
    ) if item.method == requested), None)
    if selected is None:
        raise CapabilityError(f"unknown MPS method capability: {requested}")
    if not selected.available:
        raise CapabilityError(
            f"{requested.upper()} capability is {selected.status}; "
            "the request is rejected and will not fall back to another solver"
        )
    return selected


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
        if operation == "ctmrg":
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
