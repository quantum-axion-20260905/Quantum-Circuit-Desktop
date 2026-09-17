from __future__ import annotations

import math
import time
from typing import Any

from fastapi import HTTPException

from ..gates import gate_matrix
from ..models import Gate, NoiseModel, RunPayload, TNGate, TNPayload
from ..noise import draw_paulis, noise_summary, pauli_matrix, readout_bits
from .limits import mps_memory_mb


def _sync(cp: Any) -> None:
    try:
        cp.cuda.Stream.null.synchronize()
    except AttributeError:
        pass


# Public adapter surface. Core algorithms use these names instead of reaching
# into implementation-private helpers, so the storage layout can evolve
# without forcing DMRG/TEBD/PEPS changes.
def sync(cp: Any) -> None:
    _sync(cp)


def _host(value: Any) -> Any:
    try:
        return value.get()
    except AttributeError:
        return value


def host(value: Any) -> Any:
    return _host(value)


def _dtype(cp: Any, name: str) -> Any:
    return cp.complex64 if name == "complex64" else cp.complex128


def _initial_state(cp: Any, n_qubits: int, dtype: Any) -> list[Any]:
    tensors = []
    for _ in range(n_qubits):
        tensor = cp.zeros((1, 2, 1), dtype=dtype)
        tensor[0, 0, 0] = 1
        tensors.append(tensor)
    return tensors


def _one_site(tensor: Any, unitary: Any, cp: Any) -> Any:
    return cp.einsum("ab,lbr->lar", unitary, tensor)


def apply_one_site(tensor: Any, unitary: Any, cp: Any) -> Any:
    return _one_site(tensor, unitary, cp)


def _apply_pauli(tensors: list[Any], labels: list[int], cp: Any, dtype: Any, qubit: int, name: str) -> None:
    unitary = pauli_matrix(cp, dtype, name)
    if unitary is not None:
        position = labels.index(qubit)
        tensors[position] = _one_site(tensors[position], unitary, cp)


def _adjacent_unitary(cp: Any, dtype: Any, name: str, left_label: int, right_label: int, control: int | None = None, target: int | None = None) -> Any:
    unitary = cp.zeros((4, 4), dtype=dtype)
    for left in (0, 1):
        for right in (0, 1):
            in_index = left * 2 + right
            if name == "swap":
                out_left, out_right = right, left
                phase = 1
            else:
                assert control is not None and target is not None
                control_bit = left if control == left_label else right
                target_is_left = target == left_label
                out_left, out_right = left, right
                phase = 1
                if name == "cx" and control_bit:
                    if target_is_left:
                        out_left ^= 1
                    else:
                        out_right ^= 1
                elif name == "cz" and control_bit and (left if target_is_left else right):
                    phase = -1
            unitary[out_left * 2 + out_right, in_index] = phase
    return unitary


def _apply_adjacent(tensors: list[Any], position: int, unitary: Any, cp: Any, max_bond: int, cutoff: float) -> float:
    left = tensors[position]
    right = tensors[position + 1]
    dl, _, middle = left.shape
    _, _, dr = right.shape
    merged = cp.einsum("lpm,mqr->lpqr", left, right)
    pair = merged.transpose(0, 3, 1, 2).reshape(dl * dr, 4)
    transformed = pair @ unitary.T
    merged = transformed.reshape(dl, dr, 2, 2).transpose(0, 2, 3, 1)

    matrix = merged.reshape(dl * 2, 2 * dr)
    u, singular, vh = cp.linalg.svd(matrix, full_matrices=False)
    singular_host = _host(singular)
    values = [float(abs(complex(value)) ** 2) for value in singular_host]
    total = sum(values)
    keep = min(max_bond, len(values))
    if cutoff > 0 and values:
        threshold = values[0] * cutoff * cutoff
        keep = min(keep, max(1, sum(value >= threshold for value in values)))
    discarded = sum(values[keep:]) / total if total > 0 else 0.0

    left_new = u[:, :keep].reshape(dl, 2, keep)
    right_new = (singular[:keep, None] * vh[:keep, :]).reshape(keep, 2, dr)
    tensors[position] = left_new
    tensors[position + 1] = right_new
    return discarded


def _apply_two_qubit(
    tensors: list[Any],
    labels: list[int],
    cp: Any,
    dtype: Any,
    name: str | None,
    control: int,
    target: int,
    max_bond: int,
    cutoff: float,
    unitary_override: Any = None,
) -> float:
    control_pos = labels.index(control)
    target_pos = labels.index(target)
    swaps: list[int] = []

    if control_pos < target_pos:
        while target_pos - control_pos > 1:
            swap_pos = target_pos - 1
            swap_u = _adjacent_unitary(cp, dtype, "swap", labels[swap_pos], labels[swap_pos + 1])
            discarded = _apply_adjacent(tensors, swap_pos, swap_u, cp, max_bond, cutoff)
            swaps.append(swap_pos)
            labels[swap_pos], labels[swap_pos + 1] = labels[swap_pos + 1], labels[swap_pos]
            target_pos -= 1
        left_pos = control_pos
    else:
        while control_pos - target_pos > 1:
            swap_pos = target_pos
            swap_u = _adjacent_unitary(cp, dtype, "swap", labels[swap_pos], labels[swap_pos + 1])
            discarded = _apply_adjacent(tensors, swap_pos, swap_u, cp, max_bond, cutoff)
            swaps.append(swap_pos)
            labels[swap_pos], labels[swap_pos + 1] = labels[swap_pos + 1], labels[swap_pos]
            target_pos += 1
        left_pos = target_pos

    left_label, right_label = labels[left_pos], labels[left_pos + 1]
    unitary = unitary_override if unitary_override is not None else _adjacent_unitary(cp, dtype, name, left_label, right_label, control=control, target=target)
    discarded_total = _apply_adjacent(tensors, left_pos, unitary, cp, max_bond, cutoff)

    for swap_pos in reversed(swaps):
        swap_u = _adjacent_unitary(cp, dtype, "swap", labels[swap_pos], labels[swap_pos + 1])
        discarded_total += _apply_adjacent(tensors, swap_pos, swap_u, cp, max_bond, cutoff)
        labels[swap_pos], labels[swap_pos + 1] = labels[swap_pos + 1], labels[swap_pos]
    return discarded_total


def apply_two_qubit(
    tensors: list[Any],
    labels: list[int],
    cp: Any,
    dtype: Any,
    name: str | None,
    control: int,
    target: int,
    max_bond: int,
    cutoff: float,
    unitary_override: Any = None,
) -> float:
    return _apply_two_qubit(
        tensors, labels, cp, dtype, name, control, target, max_bond, cutoff,
        unitary_override=unitary_override,
    )


def _build(
    cp: Any,
    payload: TNPayload,
    rng: Any = None,
    progress_cb: Any = None,
    cancel_cb: Any = None,
) -> tuple[list[Any], float, int]:
    n = int(payload.n_qubits)
    dtype = _dtype(cp, payload.dtype)
    tensors = _initial_state(cp, n, dtype)
    labels = list(range(n))
    discarded_weight = 0.0
    noise: NoiseModel | None = payload.noise if rng is not None else None

    total_gates = max(1, len(payload.gates))
    for gate_index, gate in enumerate(payload.gates):
        if cancel_cb and cancel_cb():
            raise RuntimeError("job canceled")
        if gate.target >= n or (gate.control is not None and gate.control >= n):
            raise HTTPException(status_code=400, detail="gate index exceeds qubit count")
        if gate.name in ("h", "x", "rx", "ry", "rz"):
            if gate.name in ("rx", "ry", "rz") and gate.theta is None:
                raise HTTPException(status_code=422, detail=f"unresolved parameter for {gate.name}: {gate.parameter}")
            position = labels.index(gate.target)
            tensors[position] = _one_site(
                tensors[position],
                gate_matrix(cp, Gate(name=gate.name, target=gate.target, theta=gate.theta), dtype),
                cp,
            )
            if noise is not None:
                error = draw_paulis(cp, rng, noise.one_qubit_depolarizing, 1)
                if error:
                    _apply_pauli(tensors, labels, cp, dtype, gate.target, error[0])
            if progress_cb:
                progress_cb((gate_index + 1) / total_gates, "mps-gate")
            continue
        if gate.name in ("cx", "cz"):
            if gate.control is None or gate.control == gate.target:
                raise HTTPException(status_code=400, detail="two-qubit gate requires distinct control and target")
            discarded_weight += _apply_two_qubit(
                tensors, labels, cp, dtype, gate.name, gate.control, gate.target,
                int(payload.bond_dim), float(payload.truncation_cutoff),
            )
            if noise is not None:
                error = draw_paulis(cp, rng, noise.two_qubit_depolarizing, 2)
                if error:
                    _apply_pauli(tensors, labels, cp, dtype, gate.control, error[0])
                    _apply_pauli(tensors, labels, cp, dtype, gate.target, error[1])
            if progress_cb:
                progress_cb((gate_index + 1) / total_gates, "mps-gate")
            continue
        raise HTTPException(status_code=400, detail=f"Unknown gate: {gate.name}")

    # The MPS remains in the original qubit order because every swap network is
    # reversed after applying a non-local gate.
    if labels != list(range(n)):
        raise RuntimeError("internal MPS qubit order was not restored")
    _sync(cp)
    max_used = max((max(tensor.shape[0], tensor.shape[2]) for tensor in tensors), default=1)
    return tensors, discarded_weight, int(max_used)


def build(
    cp: Any,
    payload: TNPayload,
    rng: Any = None,
    progress_cb: Any = None,
    cancel_cb: Any = None,
) -> tuple[list[Any], float, int]:
    return _build(cp, payload, rng, progress_cb=progress_cb, cancel_cb=cancel_cb)


def _right_environments(cp: Any, tensors: list[Any]) -> list[Any]:
    right: list[Any] = [None] * (len(tensors) + 1)
    dtype = tensors[0].dtype
    right[-1] = cp.ones((1, 1), dtype=dtype)
    for index in range(len(tensors) - 1, -1, -1):
        tensor = tensors[index]
        right[index] = cp.einsum("lpa,Lpb,ab->lL", tensor, tensor.conj(), right[index + 1])
    return right


def right_environments(cp: Any, tensors: list[Any]) -> list[Any]:
    return _right_environments(cp, tensors)


def _norm2(cp: Any, right: list[Any]) -> float:
    return float(complex(_host(right[0][0, 0])).real)


def norm2(cp: Any, right: list[Any]) -> float:
    return _norm2(cp, right)


def _parse_bitstrings(bitstrings: list[str], n_qubits: int) -> list[str]:
    if not bitstrings:
        return ["0" * n_qubits]
    output = []
    for raw in bitstrings:
        value = raw.strip().replace("_", "")
        if len(value) != n_qubits or any(char not in "01" for char in value):
            raise HTTPException(status_code=400, detail=f"invalid bitstring for {n_qubits} qubits: {raw!r}")
        output.append(value)
    return output


def _amplitude(cp: Any, tensors: list[Any], bitstring: str) -> complex:
    vector = cp.asarray([1], dtype=tensors[0].dtype)
    for tensor, bit in zip(tensors, bitstring):
        vector = cp.einsum("l,lr->r", vector, tensor[:, int(bit), :])
    return complex(_host(vector[0]))


def amplitudes(cp: Any, payload: TNPayload, progress_cb: Any = None, cancel_cb: Any = None) -> dict[str, Any]:
    if payload.noise is not None and payload.noise.active:
        raise HTTPException(status_code=422, detail="noise channels require result_type=samples; amplitudes are noiseless-only")
    started = time.perf_counter()
    tensors, discarded, max_used = _build(cp, payload, progress_cb=progress_cb, cancel_cb=cancel_cb)
    right = _right_environments(cp, tensors)
    bits = _parse_bitstrings(payload.bitstrings, payload.n_qubits)
    values = []
    for index, bit in enumerate(bits):
        if cancel_cb and cancel_cb():
            raise RuntimeError("job canceled")
        values.append(_amplitude(cp, tensors, bit))
        if progress_cb:
            progress_cb(0.85 + 0.15 * ((index + 1) / max(1, len(bits))), "mps-amplitude")
    _sync(cp)
    return {
        "status": "done",
        "backend": "tensor-network-mps",
        "method": "mps",
        "n_qubits": payload.n_qubits,
        "dtype": payload.dtype,
        "bond_dim_requested": payload.bond_dim,
        "bond_dim_used": max_used,
        "truncation_cutoff": payload.truncation_cutoff,
        "discarded_weight": discarded,
        "approximate": discarded > 1e-12,
        "warnings": (["bond dimension truncated entanglement; inspect discarded_weight and norm2"] if discarded > 1e-12 else []),
        "noise": noise_summary(payload.noise),
        "norm2": _norm2(cp, right),
        "time_ms": round((time.perf_counter() - started) * 1000, 3),
        "amplitudes": [
            {"bitstring": bit, "re": value.real, "im": value.imag}
            for bit, value in zip(bits, values)
        ],
    }


def _sample_one(cp: Any, tensors: list[Any], right: list[Any], rng: Any) -> str:
    vector = cp.asarray([1], dtype=tensors[0].dtype)
    bits: list[str] = []
    for index, tensor in enumerate(tensors):
        candidates = []
        for bit in (0, 1):
            candidate = cp.einsum("l,lr->r", vector, tensor[:, bit, :])
            probability = cp.einsum("r,rR,R->", candidate.conj(), right[index + 1], candidate)
            candidates.append(max(0.0, float(complex(_host(probability)).real)))
        total = candidates[0] + candidates[1]
        if total <= 0 or not math.isfinite(total):
            raise RuntimeError("MPS conditional probability normalization failed")
        chosen = 1 if float(_host(rng.random_sample() if hasattr(rng, "random_sample") else rng.random())) * total >= candidates[0] else 0
        bits.append(str(chosen))
        vector = cp.einsum("l,lr->r", vector, tensor[:, chosen, :])
    return "".join(bits)


def sample(cp: Any, payload: RunPayload, progress_cb: Any = None, cancel_cb: Any = None) -> dict[str, Any]:
    started = time.perf_counter()
    if payload.noise is not None and payload.noise.active:
        rng = cp.random.RandomState(payload.seed) if payload.seed is not None else cp.random
        counts: dict[str, int] = {}
        discarded_sum = 0.0
        norm_sum = 0.0
        max_used = 1
        for shot in range(payload.shots):
            if cancel_cb and cancel_cb():
                raise RuntimeError("job canceled")
            tensors, discarded, used = _build(cp, payload, rng, cancel_cb=cancel_cb)
            right = _right_environments(cp, tensors)
            bitstring = _sample_one(cp, tensors, right, rng)
            bitstring = readout_bits(rng, bitstring, payload.noise.readout_flip)
            counts[bitstring] = counts.get(bitstring, 0) + 1
            discarded_sum += discarded
            norm_sum += _norm2(cp, right)
            max_used = max(max_used, used)
            if progress_cb and (shot == 0 or (shot + 1) % max(1, payload.shots // 20) == 0):
                progress_cb((shot + 1) / payload.shots, "mps-noise-shot")
        discarded = discarded_sum / payload.shots
        norm2 = norm_sum / payload.shots
        approximate = discarded > 1e-12
        return {
            "status": "done",
            "backend": "tensor-network-mps-sample",
            "method": "mps-trajectories",
            "n_qubits": payload.n_qubits,
            "dtype": payload.dtype,
            "shots": payload.shots,
            "seed": payload.seed,
            "bond_dim_requested": payload.bond_dim,
            "bond_dim_used": max_used,
            "truncation_cutoff": payload.truncation_cutoff,
            "discarded_weight": discarded,
            "approximate": approximate,
            "noise": noise_summary(payload.noise),
            "warnings": (["noise result is estimated by shot-based trajectories"] + (["bond dimension truncated entanglement; inspect discarded_weight and norm2"] if approximate else [])),
            "norm2": norm2,
            "time_ms": round((time.perf_counter() - started) * 1000, 3),
            "counts": counts,
            "validation": {
                "normalization_error": abs(norm2 - 1.0),
                "normalization_passed": abs(norm2 - 1.0) < 1e-5,
                "counts_total": sum(counts.values()),
                "counts_total_passed": sum(counts.values()) == payload.shots,
            },
        }

    tensors, discarded, max_used = _build(cp, payload, cancel_cb=cancel_cb)
    right = _right_environments(cp, tensors)
    rng = cp.random.RandomState(payload.seed) if payload.seed is not None else cp.random
    counts: dict[str, int] = {}

    for shot in range(payload.shots):
        if cancel_cb and cancel_cb():
            raise RuntimeError("job canceled")
        bitstring = _sample_one(cp, tensors, right, rng)
        counts[bitstring] = counts.get(bitstring, 0) + 1
        if progress_cb and (shot == 0 or (shot + 1) % max(1, payload.shots // 20) == 0):
            progress_cb((shot + 1) / payload.shots, "mps-shot")

    _sync(cp)
    norm2 = _norm2(cp, right)
    return {
        "status": "done",
        "backend": "tensor-network-mps-sample",
        "method": "mps",
        "n_qubits": payload.n_qubits,
        "dtype": payload.dtype,
        "shots": payload.shots,
        "seed": payload.seed,
        "bond_dim_requested": payload.bond_dim,
        "bond_dim_used": max_used,
        "truncation_cutoff": payload.truncation_cutoff,
        "discarded_weight": discarded,
        "approximate": discarded > 1e-12,
        "warnings": (["bond dimension truncated entanglement; inspect discarded_weight and norm2"] if discarded > 1e-12 else []),
        "noise": noise_summary(payload.noise),
        "norm2": norm2,
        "time_ms": round((time.perf_counter() - started) * 1000, 3),
        "counts": counts,
        "validation": {
            "normalization_error": abs(norm2 - 1.0),
            "normalization_passed": abs(norm2 - 1.0) < 1e-5,
            "counts_total": sum(counts.values()),
            "counts_total_passed": sum(counts.values()) == payload.shots,
        },
    }


def estimate(payload: TNPayload) -> dict[str, Any]:
    return {
        "status": "done",
        "backend": "tensor-network-mps",
        "method": "mps",
        "n_qubits": payload.n_qubits,
        "dtype": payload.dtype,
        "bond_dim_requested": payload.bond_dim,
        "estimated_peak_memory_mb": round(mps_memory_mb(payload.n_qubits, payload.bond_dim, payload.dtype), 3),
        "estimated_time_ms": int(1 + max(1, len(payload.gates)) * max(1, payload.bond_dim) ** 3 / 5000),
        "truncation_cutoff": payload.truncation_cutoff,
        "warnings": ["MPS results are approximate when the requested bond dimension truncates entanglement."],
    }
