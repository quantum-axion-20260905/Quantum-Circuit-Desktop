from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .models import FermionMappingPayload, FermionTerm, PauliTerm

PauliKey = tuple[tuple[int, str], ...]

_PAULI_PRODUCT: dict[tuple[str, str], tuple[str, complex]] = {
    ("X", "Y"): ("Z", 1j),
    ("Y", "X"): ("Z", -1j),
    ("Y", "Z"): ("X", 1j),
    ("Z", "Y"): ("X", -1j),
    ("Z", "X"): ("Y", 1j),
    ("X", "Z"): ("Y", -1j),
}


def _multiply_pauli(left: PauliKey, right: PauliKey) -> tuple[PauliKey, complex]:
    left_map = dict(left)
    right_map = dict(right)
    result: dict[int, str] = {}
    phase = 1 + 0j
    for mode in sorted(set(left_map) | set(right_map)):
        left_pauli = left_map.get(mode, "I")
        right_pauli = right_map.get(mode, "I")
        if left_pauli == "I":
            output = right_pauli
        elif right_pauli == "I":
            output = left_pauli
        elif left_pauli == right_pauli:
            output = "I"
        else:
            output, local_phase = _PAULI_PRODUCT[(left_pauli, right_pauli)]
            phase *= local_phase
        if output != "I":
            result[mode] = output
    return tuple(sorted(result.items())), phase


def _jw_single_operator(mode: int, action: str) -> dict[PauliKey, complex]:
    prefix = tuple((index, "Z") for index in range(mode))
    if action == "create":
        local = ((mode, "X"), 0.5), ((mode, "Y"), -0.5j)
    else:
        local = ((mode, "X"), 0.5), ((mode, "Y"), 0.5j)
    return {
        prefix + (pauli,): coefficient
        for pauli, coefficient in local
    }


def _map_term(term: FermionTerm) -> dict[PauliKey, complex]:
    result: dict[PauliKey, complex] = {tuple(): complex(term.coefficient)}
    for operator in term.operators:
        factor = _jw_single_operator(operator.mode, operator.action)
        combined: defaultdict[PauliKey, complex] = defaultdict(complex)
        for left_key, left_value in result.items():
            for right_key, right_value in factor.items():
                key, phase = _multiply_pauli(left_key, right_key)
                combined[key] += left_value * right_value * phase
        result = dict(combined)
    return result


def _label(key: PauliKey) -> str:
    return "I" if not key else " ".join(f"{pauli}{mode}" for mode, pauli in key)


def map_fermion_terms(payload: FermionMappingPayload, terms: Iterable[FermionTerm] | None = None) -> dict:
    """Map fermionic products to Jordan–Wigner Pauli strings.

    The response keeps complex coefficients losslessly and separately exposes
    real Pauli terms that can be sent directly to the observable/TEBD API.
    Hermitian inputs should have residual imaginary parts below the requested
    tolerance.
    """
    source_terms = list(payload.terms if terms is None else terms)
    accumulated: defaultdict[PauliKey, complex] = defaultdict(complex)
    for term in source_terms:
        for key, value in _map_term(term).items():
            accumulated[key] += value

    tolerance = float(payload.hermitian_tolerance)
    complex_terms = []
    real_terms: list[PauliTerm] = []
    max_imag = 0.0
    for key in sorted(accumulated, key=lambda item: (len(item), item)):
        value = accumulated[key]
        if abs(value) <= tolerance:
            continue
        max_imag = max(max_imag, abs(value.imag))
        complex_terms.append({
            "paulis": dict(key),
            "real": float(value.real),
            "imag": float(value.imag),
            "label": _label(key),
        })
        if abs(value.imag) <= tolerance:
            real_terms.append(PauliTerm(
                paulis=dict(key),
                coefficient=float(value.real),
                label=_label(key),
            ))

    expectation_ready = max_imag <= tolerance
    warnings = []
    if not expectation_ready:
        warnings.append(
            "mapped operator has non-negligible imaginary Pauli coefficients; "
            "provide a Hermitian fermionic Hamiltonian before expectation/TEBD"
        )
    return {
        "status": "done",
        "mapping": payload.mapping,
        "n_qubits": payload.n_modes,
        "source_term_count": len(source_terms),
        "terms": [term.model_dump(mode="json") for term in real_terms],
        "complex_terms": complex_terms,
        "expectation_ready": expectation_ready,
        "max_imaginary_coefficient": max_imag,
        "warnings": warnings,
    }
