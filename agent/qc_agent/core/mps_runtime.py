from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Iterable

from ..backends import mps as mps_backend
from ..noise import pauli_matrix
from .checkpoints import load_mps_checkpoint, save_mps_checkpoint
from .contracts import CheckpointManifest, TruncationReport


class MPSRuntime:
    """Small public adapter around the reusable GPU MPS tensor operations."""

    def __init__(self, cp: Any, payload: Any, *, progress_cb: Any = None, cancel_cb: Any = None):
        self.cp = cp
        self.payload = payload
        self.tensors, self.discarded_weight, self.bond_dim_used = mps_backend.build(
            cp, payload, progress_cb=progress_cb, cancel_cb=cancel_cb
        )
        self.labels = list(range(payload.n_qubits))

    def apply_one_site(self, qubit: int, unitary: Any) -> None:
        position = self.labels.index(qubit)
        self.tensors[position] = mps_backend.apply_one_site(self.tensors[position], unitary, self.cp)

    def apply_two_site(self, first: int, second: int, unitary: Any) -> None:
        discarded = mps_backend.apply_two_qubit(
            self.tensors,
            self.labels,
            self.cp,
            self.tensors[0].dtype,
            None,
            first,
            second,
            int(self.payload.bond_dim),
            float(self.payload.truncation_cutoff),
            unitary_override=unitary,
        )
        self.discarded_weight += discarded
        self.bond_dim_used = max((max(tensor.shape[0], tensor.shape[2]) for tensor in self.tensors), default=1)

    def apply_controlled_x(self, control: int, target: int) -> None:
        """Apply a logical CX, using the backend swap network when needed."""
        discarded = mps_backend.apply_two_qubit(
            self.tensors,
            self.labels,
            self.cp,
            self.tensors[0].dtype,
            "cx",
            control,
            target,
            int(self.payload.bond_dim),
            float(self.payload.truncation_cutoff),
        )
        self.discarded_weight += discarded
        self.bond_dim_used = max((max(tensor.shape[0], tensor.shape[2]) for tensor in self.tensors), default=1)

    def apply_pauli_string_exponential(self, paulis: dict[int, str], angle: float) -> None:
        """Apply ``exp(-i * angle * P)`` for an arbitrary Pauli string.

        A parity-CX network reduces the string to one Z rotation. This keeps
        the domain layer independent of tensor shapes while supporting the
        longer Jordan–Wigner strings produced by fermion mappings.
        """
        active = sorted((int(qubit), str(pauli).upper()) for qubit, pauli in paulis.items() if str(pauli).upper() != "I")
        if not active:
            return
        dtype = self.tensors[0].dtype
        identity = self.cp.eye(2, dtype=dtype)
        hadamard = self.cp.asarray([[1, 1], [1, -1]], dtype=dtype) / math.sqrt(2)
        s_dagger = self.cp.asarray([[1, 0], [0, -1j]], dtype=dtype)
        # Keep the actual basis matrices rather than looking the operators up
        # again during the inverse pass.  Besides being clearer, this also
        # keeps the runtime correct if a caller supplies stringified JSON keys.
        basis_changes: list[tuple[int, Any]] = []
        for qubit, pauli in active:
            if pauli == "X":
                basis = hadamard
            elif pauli == "Y":
                basis = hadamard @ s_dagger
            elif pauli == "Z":
                basis = identity
            else:
                raise ValueError(f"unknown Pauli operator: {pauli}")
            basis_changes.append((qubit, basis))
            if pauli != "Z":
                self.apply_one_site(qubit, basis)

        pivot = active[0][0]
        controls = [qubit for qubit, _ in active[1:]]
        for control in controls:
            self.apply_controlled_x(control, pivot)
        # Assign CuPy scalars into a device array instead of passing them
        # through a nested Python list (CuPy intentionally rejects implicit
        # device-to-host conversion in that case).
        phase_rotation = self.cp.zeros((2, 2), dtype=dtype)
        phase_rotation[0, 0] = self.cp.exp(-1j * angle)
        phase_rotation[1, 1] = self.cp.exp(1j * angle)
        self.apply_one_site(pivot, phase_rotation)
        for control in reversed(controls):
            self.apply_controlled_x(control, pivot)

        for qubit, basis in reversed(basis_changes):
            if not bool(self.cp.allclose(basis, identity)):
                self.apply_one_site(qubit, basis.conj().T)

    def norm2(self) -> float:
        right = mps_backend.right_environments(self.cp, self.tensors)
        return mps_backend.norm2(self.cp, right)

    def expectation(self, terms: Iterable[Any]) -> list[float]:
        from .observables import mps_expectation_from_tensors

        return mps_expectation_from_tensors(self.cp, self.tensors, terms)

    def energy_moments(self, terms: Iterable[Any]) -> tuple[float, float, float]:
        from .observables import mps_energy_moments

        return mps_energy_moments(self.cp, self.tensors, terms)

    def estimate_resources(self) -> dict[str, Any]:
        """Return a conservative memory estimate without allocating a workspace."""
        dtype = self.tensors[0].dtype
        itemsize = int(getattr(dtype, "itemsize", 16))
        tensor_values = sum(int(tensor.size) for tensor in self.tensors)
        tensor_bytes = tensor_values * itemsize
        return {
            "representation": "mps",
            "n_qubits": len(self.tensors),
            "bond_dim_used": int(self.bond_dim_used),
            "tensor_values": tensor_values,
            "tensor_bytes": tensor_bytes,
            "peak_bytes_estimate": int(math.ceil(tensor_bytes * 2.5)),
            "dtype": str(dtype),
        }

    def truncation_report(self) -> TruncationReport:
        return TruncationReport(
            discarded_weight=float(self.discarded_weight),
            cutoff=float(getattr(self.payload, "truncation_cutoff", 0.0)),
            max_bond_dim=int(getattr(self.payload, "bond_dim", self.bond_dim_used)),
        )

    def save_checkpoint(self, path: str, manifest: CheckpointManifest) -> dict[str, Any]:
        runtime_manifest = replace(
            manifest,
            metadata={
                **manifest.metadata,
                "discarded_weight": float(self.discarded_weight),
                "bond_dim_used": int(self.bond_dim_used),
            },
        )
        return save_mps_checkpoint(path, self.tensors, runtime_manifest)

    def restore_checkpoint(self, path: str) -> dict[str, Any]:
        manifest, tensors = load_mps_checkpoint(path, self.cp)
        if manifest.get("representation") != "mps":
            raise ValueError("checkpoint representation is not MPS")
        if len(tensors) != int(self.payload.n_qubits):
            raise ValueError("checkpoint qubit count does not match the requested payload")
        self.tensors = tensors
        metadata = manifest.get("metadata", {})
        self.discarded_weight = float(metadata.get("discarded_weight", 0.0))
        self.bond_dim_used = int(
            metadata.get(
                "bond_dim_used",
                max((max(tensor.shape[0], tensor.shape[2]) for tensor in tensors), default=1),
            )
        )
        self.sync()
        return manifest

    def canonicalize_left(self) -> None:
        """Move the orthogonality center to the right edge using QR sweeps."""
        for position in range(len(self.tensors) - 1):
            tensor = self.tensors[position]
            left_dim, physical_dim, right_dim = tensor.shape
            q, r = self.cp.linalg.qr(
                tensor.reshape(left_dim * physical_dim, right_dim), mode="reduced"
            )
            self.tensors[position] = q.reshape(left_dim, physical_dim, q.shape[1])
            self.tensors[position + 1] = self.cp.tensordot(
                r, self.tensors[position + 1], axes=(1, 0)
            )
        self.sync()

    def canonicalize_right(self) -> None:
        """Move the orthogonality center to the left edge using QR sweeps."""
        for position in range(len(self.tensors) - 1, 0, -1):
            tensor = self.tensors[position]
            left_dim, physical_dim, right_dim = tensor.shape
            q, r = self.cp.linalg.qr(
                tensor.reshape(left_dim, physical_dim * right_dim).T, mode="reduced"
            )
            self.tensors[position] = q.T.reshape(q.shape[1], physical_dim, right_dim)
            self.tensors[position - 1] = self.cp.tensordot(
                self.tensors[position - 1], r.T, axes=(2, 0)
            )
        self.sync()

    def sync(self) -> None:
        mps_backend.sync(self.cp)


def pauli_operator(cp: Any, dtype: Any, name: str) -> Any:
    normalized = name.lower()
    if normalized == "i":
        return cp.eye(2, dtype=dtype)
    result = pauli_matrix(cp, dtype, normalized)
    if result is None:
        return cp.eye(2, dtype=dtype)
    return result
