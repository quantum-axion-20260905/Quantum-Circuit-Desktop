from __future__ import annotations

import math
import time
from typing import Any

from ..core.mps_runtime import MPSRuntime, pauli_operator
from ..models import TNGate, TNPayload
from .models import TEBDPayload


def _two_site_exponential(cp: Any, dtype: Any, left: str, right: str, angle: float) -> Any:
    identity = cp.eye(4, dtype=dtype)
    left_matrix = pauli_operator(cp, dtype, left)
    right_matrix = pauli_operator(cp, dtype, right)
    product = cp.kron(left_matrix, right_matrix)
    return math.cos(angle) * identity - 1j * math.sin(angle) * product


def _one_site_exponential(cp: Any, dtype: Any, pauli: str, angle: float) -> Any:
    identity = cp.eye(2, dtype=dtype)
    operator = pauli_operator(cp, dtype, pauli)
    return math.cos(angle) * identity - 1j * math.sin(angle) * operator


def _apply_term(runtime: MPSRuntime, term: Any, angle: float) -> None:
    cp = runtime.cp
    dtype = runtime.tensors[0].dtype
    active = sorted(term.paulis.items())
    if not active:
        return
    if len(active) == 1:
        runtime.apply_one_site(active[0][0], _one_site_exponential(cp, dtype, active[0][1], angle))
        return
    if len(active) > 2:
        runtime.apply_pauli_string_exponential(term.paulis, angle)
        return
    (left, left_pauli), (right, right_pauli) = active
    runtime.apply_two_site(left, right, _two_site_exponential(cp, dtype, left_pauli, right_pauli, angle))


def _energy(runtime: MPSRuntime, terms: list[Any]) -> float:
    values = runtime.expectation(terms)
    return float(sum(term.coefficient * value for term, value in zip(terms, values)))


def run_tebd(
    cp: Any,
    payload: TEBDPayload,
    *,
    progress_cb: Any = None,
    cancel_cb: Any = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    base = TNPayload(
        n_qubits=payload.n_qubits,
        gates=[TNGate(**gate) if isinstance(gate, dict) else gate for gate in payload.gates],
        dtype=payload.dtype,
        bond_dim=payload.bond_dim,
        truncation_cutoff=payload.truncation_cutoff,
    )
    runtime = MPSRuntime(cp, base, progress_cb=progress_cb, cancel_cb=cancel_cb)
    observables = payload.observables or payload.terms
    times = [0.0]
    values = [runtime.expectation(observables)]
    energies = [_energy(runtime, payload.terms)]
    for step in range(payload.steps):
        if cancel_cb and cancel_cb():
            raise RuntimeError("job canceled")
        if payload.order == 1:
            schedule = [(term, payload.dt) for term in payload.terms]
        else:
            schedule = [(term, payload.dt / 2.0) for term in payload.terms]
            schedule.extend((term, payload.dt / 2.0) for term in reversed(payload.terms))
        for term, local_dt in schedule:
            if cancel_cb and cancel_cb():
                raise RuntimeError("job canceled")
            _apply_term(runtime, term, local_dt * term.coefficient)
        runtime.sync()
        times.append((step + 1) * payload.dt)
        current_values = runtime.expectation(observables)
        values.append(current_values)
        energies.append(_energy(runtime, payload.terms))
        if progress_cb:
            progress_cb((step + 1) / max(1, payload.steps), "tebd-step")
    norm2 = runtime.norm2()
    runtime.sync()
    elapsed = round((time.perf_counter() - started) * 1000, 3)
    max_term_locality = max((len(term.paulis) for term in payload.terms), default=0)
    parity_string_terms = sum(1 for term in payload.terms if len(term.paulis) > 2)
    warnings = ["TEBD uses Suzuki-Trotter decomposition; reduce |dt| to check convergence"]
    if parity_string_terms:
        warnings.append(
            f"{parity_string_terms} term(s) use parity-CX evolution because locality exceeds two sites"
        )
    if payload.lattice is not None and len(payload.lattice.dimensions) > 1:
        warnings.append("2D/3D lattice is embedded into a snake-ordered MPS; inspect bond/truncation convergence")
    if runtime.discarded_weight > 1e-12:
        warnings.append("bond dimension truncated entanglement; inspect discarded_weight and norm2")
    return {
        "status": "done",
        "backend": "tensor-network-mps-tebd",
        "method": "tebd",
        "n_qubits": payload.n_qubits,
        "dtype": payload.dtype,
        "steps": payload.steps,
        "dt": payload.dt,
        "order": payload.order,
        "max_term_locality": max_term_locality,
        "parity_string_terms": parity_string_terms,
        "bond_dim_requested": payload.bond_dim,
        "bond_dim_used": runtime.bond_dim_used,
        "truncation_cutoff": payload.truncation_cutoff,
        "discarded_weight": runtime.discarded_weight,
        "approximate": runtime.discarded_weight > 1e-12 or payload.order == 2,
        "norm2": norm2,
        "times": times,
        "energies": energies,
        "expectations": [
            {"time": t, "values": point, "energy": energy}
            for t, point, energy in zip(times, values, energies)
        ],
        "warnings": warnings,
        "time_ms": elapsed,
    }
