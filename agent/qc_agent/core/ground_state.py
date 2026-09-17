from __future__ import annotations

from typing import Any, Iterable

from ..backends.limits import MAX_EXACT_DIAGONALIZATION_QUBITS
from ..plugins.models import GroundStatePayload
from .observables import _host, _sparse_pauli_action


def dense_hamiltonian(xp: Any, n_qubits: int, terms: Iterable[Any], dtype: Any) -> Any:
    """Construct a dense Pauli Hamiltonian for the small exact reference path."""
    if n_qubits > MAX_EXACT_DIAGONALIZATION_QUBITS:
        raise ValueError("exact diagonalization is limited to 12 qubits")
    dimension = 1 << n_qubits
    matrix = xp.zeros((dimension, dimension), dtype=dtype)
    for column in range(dimension):
        for term in terms:
            row, phase = _sparse_pauli_action(column, n_qubits, term.paulis)
            matrix[row, column] += term.coefficient * phase
    return matrix


def exact_ground_state(xp: Any, payload: GroundStatePayload) -> dict[str, Any]:
    """Return the lowest eigenvalue and selected amplitudes of a Pauli Hamiltonian."""
    if payload.n_qubits > MAX_EXACT_DIAGONALIZATION_QUBITS:
        raise ValueError("exact diagonalization supports at most 12 qubits")
    dtype = xp.complex64 if payload.dtype == "complex64" else xp.complex128
    matrix = dense_hamiltonian(xp, payload.n_qubits, payload.terms, dtype)
    eigenvalues, eigenvectors = xp.linalg.eigh(matrix)
    vector = eigenvectors[:, 0]
    amplitudes = []
    for bitstring in payload.bitstrings:
        value = complex(_host(vector[int(bitstring.replace("_", ""), 2)]))
        amplitudes.append({"bitstring": bitstring, "re": value.real, "im": value.imag})
    norm2 = float(complex(_host(xp.sum(xp.abs(vector) ** 2))).real)
    return {
        "status": "done",
        "backend": "exact-diagonalization-cuda" if hasattr(xp, "cuda") else "exact-diagonalization-reference",
        "method": "dense-pauli-eigh",
        "n_qubits": payload.n_qubits,
        "hamiltonian_dimension": 1 << payload.n_qubits,
        "term_count": len(payload.terms),
        "ground_energy": float(complex(_host(eigenvalues[0])).real),
        "lowest_eigenvalues": [float(complex(_host(value)).real) for value in eigenvalues[: min(8, len(eigenvalues))]],
        "norm2": norm2,
        "amplitudes": amplitudes,
        "warnings": ["dense exact diagonalization is a validation/reference path; use MPS/DMRG for larger systems"],
    }
