from __future__ import annotations

import cmath
import math
from typing import Any, Iterable

from ..backends.limits import MAX_REFERENCE_QUBITS, MAX_STATEVECTOR_QUBITS
from ..backends.reference import _apply_1q as reference_apply_1q
from ..backends.reference import _apply_2q as reference_apply_2q
from ..backends.reference import _matrix as reference_matrix
from ..gates import cx_matrix, cz_matrix, gate_matrix
from ..models import TNGate
from .mps_runtime import pauli_operator


def _host(value: Any) -> Any:
    try:
        return value.get()
    except AttributeError:
        return value


def mps_expectation_from_tensors(cp: Any, tensors: list[Any], terms: Iterable[Any]) -> list[float]:
    dtype = tensors[0].dtype
    output: list[float] = []
    for term in terms:
        environment = cp.ones((1, 1), dtype=dtype)
        for qubit, tensor in enumerate(tensors):
            operator = pauli_operator(cp, dtype, term.paulis.get(qubit, "I"))
            environment = cp.einsum(
                "lL,lpr,pq,LqR->rR", environment, tensor, operator, tensor.conj()
            )
        value = complex(_host(environment[0, 0]))
        output.append(float(value.real))
    return output


def _sparse_pauli_action(index: int, n_qubits: int, paulis: dict[int, str]) -> tuple[int, complex]:
    target = index
    phase = 1 + 0j
    for qubit, pauli in paulis.items():
        mask = 1 << (n_qubits - qubit - 1)
        bit = 1 if index & mask else 0
        if pauli in ("X", "Y"):
            target ^= mask
        if pauli == "Y":
            phase *= 1j if bit == 0 else -1j
        elif pauli == "Z" and bit:
            phase *= -1
    return target, phase


def statevector_expectation(state: Any, cp: Any, terms: Iterable[Any], n_qubits: int) -> list[float]:
    values: list[float] = []
    for term in terms:
        total = 0j
        for index in range(1 << n_qubits):
            target, phase = _sparse_pauli_action(index, n_qubits, term.paulis)
            total += complex(_host(state[index])).conjugate() * phase * complex(_host(state[target]))
        values.append(float(total.real))
    return values


def evolve_statevector(cp: Any, payload: Any) -> Any:
    n = int(payload.n_qubits)
    if n > MAX_STATEVECTOR_QUBITS:
        raise ValueError(f"statevector supports at most {MAX_STATEVECTOR_QUBITS} qubits")
    dtype = cp.complex64 if payload.dtype == "complex64" else cp.complex128
    state = cp.zeros((2**n,), dtype=dtype)
    state[0] = 1
    for gate in payload.gates:
        if gate.target >= n or (gate.control is not None and gate.control >= n):
            raise ValueError("gate index exceeds qubit count")
        if gate.name in ("h", "x", "rx", "ry", "rz"):
            if gate.name in ("rx", "ry", "rz") and gate.theta is None:
                raise ValueError(f"unresolved parameter for {gate.name}: {gate.parameter}")
            state = _apply_1q(cp, state, gate_matrix(cp, gate, dtype), n, gate.target)
        elif gate.name in ("cx", "cz"):
            if gate.control is None or gate.control == gate.target:
                raise ValueError("two-qubit gate requires distinct control and target")
            matrix = cx_matrix(cp, dtype) if gate.name == "cx" else cz_matrix(cp, dtype)
            state = _apply_2q(cp, state, matrix, n, gate.control, gate.target)
        else:
            raise ValueError(f"unsupported gate: {gate.name}")
    return state


def _apply_1q(cp: Any, state: Any, matrix: Any, n: int, target: int) -> Any:
    psi = state.reshape((2,) * n)
    transformed = cp.tensordot(matrix, psi, axes=[1, target])
    if target == 0:
        return transformed.reshape((2**n,))
    permutation = list(range(1, target + 1)) + [0] + list(range(target + 1, n))
    return transformed.transpose(permutation).reshape((2**n,))


def _apply_2q(cp: Any, state: Any, matrix: Any, n: int, q0: int, q1: int) -> Any:
    psi = state.reshape((2,) * n)
    permutation = [q0, q1] + [q for q in range(n) if q not in (q0, q1)]
    inverse = [0] * n
    for position, qubit in enumerate(permutation):
        inverse[qubit] = position
    front = psi.transpose(permutation).reshape(4, -1)
    return (matrix @ front).reshape((2, 2) + (2,) * (n - 2)).transpose(inverse).reshape((2**n,))


def reference_expectation(payload: Any) -> dict[str, Any]:
    n = int(payload.n_qubits)
    if n > MAX_REFERENCE_QUBITS:
        raise ValueError(f"reference-cpu supports at most {MAX_REFERENCE_QUBITS} qubits")
    state = [0j] * (1 << n)
    state[0] = 1 + 0j
    for gate in payload.gates:
        if gate.name in ("h", "x", "rx", "ry", "rz"):
            if gate.name in ("rx", "ry", "rz") and gate.theta is None:
                raise ValueError(f"unresolved parameter for {gate.name}: {gate.parameter}")
            reference_apply_1q(state, n, gate.target, reference_matrix(gate.name, gate.theta))
        elif gate.name in ("cx", "cz") and gate.control is not None:
            reference_apply_2q(state, n, gate.control, gate.target, gate.name)
        else:
            raise ValueError("invalid gate")
    values = []
    for term in payload.terms:
        total = 0j
        for index, amplitude in enumerate(state):
            target, phase = _sparse_pauli_action(index, n, term.paulis)
            total += amplitude.conjugate() * phase * state[target]
        values.append(float(total.real))
    norm2 = sum(abs(amplitude) ** 2 for amplitude in state)
    return {"backend": "reference-cpu-observable", "method": "reference-observable", "norm2": norm2, "values": values}
