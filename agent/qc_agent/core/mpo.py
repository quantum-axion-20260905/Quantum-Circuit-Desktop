"""Matrix-product-operator primitives for sparse Pauli Hamiltonians."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .mps_runtime import pauli_operator


@dataclass
class MPOOperator:
    """A finite open-boundary MPO with tensors ``(left, in, out, right)``."""

    tensors: list[Any]
    n_qubits: int
    bond_dim: int
    method: str = "pauli-prefix-mpo"

    def estimate_resources(self) -> dict[str, Any]:
        if not self.tensors:
            return {"representation": "mpo", "n_qubits": 0, "bond_dim": 0, "tensor_bytes": 0}
        itemsize = int(getattr(self.tensors[0].dtype, "itemsize", 16))
        values = sum(int(tensor.size) for tensor in self.tensors)
        tensor_bytes = values * itemsize
        return {
            "representation": "mpo",
            "n_qubits": int(self.n_qubits),
            "bond_dim": int(self.bond_dim),
            "tensor_values": values,
            "tensor_bytes": tensor_bytes,
            "dtype": str(self.tensors[0].dtype),
        }

    def expectation(self, mps_tensors: list[Any], xp: Any) -> float:
        """Contract ``<MPS|MPO|MPS>`` without materialising a statevector."""
        if len(mps_tensors) != self.n_qubits:
            raise ValueError("MPS and MPO qubit counts do not match")
        if len(self.tensors) != self.n_qubits:
            raise ValueError("MPO tensor count does not match n_qubits")
        environment = xp.ones((1, 1, 1), dtype=mps_tensors[0].dtype)
        for mps, operator in zip(mps_tensors, self.tensors):
            if int(operator.shape[0]) != int(environment.shape[1]):
                raise ValueError("MPO bond dimensions are not contiguous")
            environment = xp.einsum(
                "amb,asc,mstn,btd->cnd",
                environment,
                mps.conj(),
                operator,
                mps,
            )
        value = environment[0, 0, 0]
        try:
            value = value.get()
        except AttributeError:
            pass
        return float(complex(value).real)


def build_pauli_mpo(xp: Any, n_qubits: int, terms: Iterable[Any], dtype: Any = None) -> MPOOperator:
    """Build an exact open-boundary MPO from a sparse Pauli sum.

    The construction shares identical prefixes between terms.  Coefficients
    are placed on the terminal transition, so shared prefix edges are never
    double-counted.  The resulting bond dimension is the largest number of
    distinct prefixes at a cut, which is exposed to admission and diagnostics.
    """
    if int(n_qubits) < 1:
        raise ValueError("n_qubits must be positive")
    materialized = list(terms)
    if not materialized:
        raise ValueError("at least one Pauli term is required")
    if dtype is None:
        dtype = getattr(xp, "complex64", complex)
    sequences: dict[tuple[str, ...], float] = {}
    for term in materialized:
        sequence = tuple(str(term.paulis.get(qubit, "I")).upper() for qubit in range(int(n_qubits)))
        if any(pauli not in ("I", "X", "Y", "Z") for pauli in sequence):
            raise ValueError("MPO supports only I, X, Y, and Z Pauli operators")
        sequences[sequence] = sequences.get(sequence, 0.0) + float(term.coefficient)
    sequences = {sequence: coefficient for sequence, coefficient in sequences.items() if coefficient != 0.0}
    if not sequences:
        raise ValueError("Pauli terms cancel to the zero operator")

    prefixes: list[list[tuple[str, ...]]] = [[()]]
    for cut in range(1, int(n_qubits)):
        prefixes.append(sorted({sequence[:cut] for sequence in sequences}))
    prefixes.append(sorted(sequences))
    indices = [{prefix: index for index, prefix in enumerate(cut_prefixes)} for cut_prefixes in prefixes]

    tensors: list[Any] = []
    for site in range(int(n_qubits)):
        left_dim = 1 if site == 0 else len(prefixes[site])
        right_dim = 1 if site == n_qubits - 1 else len(prefixes[site + 1])
        tensor = xp.zeros((left_dim, 2, 2, right_dim), dtype=dtype)
        edges: set[tuple[int, int]] = set()
        for sequence, coefficient in sequences.items():
            left_prefix = sequence[:site]
            right_prefix = sequence[: site + 1]
            left = 0 if site == 0 else indices[site][left_prefix]
            right = 0 if site == n_qubits - 1 else indices[site + 1][right_prefix]
            edge = (left, right)
            if site < n_qubits - 1:
                if edge in edges:
                    continue
                edges.add(edge)
                local = pauli_operator(xp, dtype, sequence[site])
                tensor[left, :, :, right] = local
            else:
                local = pauli_operator(xp, dtype, sequence[site])
                tensor[left, :, :, right] += coefficient * local
        tensors.append(tensor)
    bond_dim = max(max(tensor.shape[0], tensor.shape[3]) for tensor in tensors)
    return MPOOperator(tensors=tensors, n_qubits=int(n_qubits), bond_dim=int(bond_dim))
