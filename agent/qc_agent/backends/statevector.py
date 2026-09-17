from __future__ import annotations

import time
import math
from typing import Any

from fastapi import HTTPException

from ..gates import cx_matrix, cz_matrix, gate_matrix
from ..noise import draw_paulis, noise_summary, pauli_matrix, readout_bits
from ..models import SamplePayload, SimulatePayload, TNGate
from .limits import MAX_STATEVECTOR_QUBITS, statevector_memory_mb


def _apply_1q(cp: Any, state, gate2, n_qubits: int, target: int):
    psi = state.reshape((2,) * n_qubits)
    tmp = cp.tensordot(gate2, psi, axes=[1, target])
    if target == 0:
        out = tmp
    else:
        perm = list(range(1, target + 1)) + [0] + list(range(target + 1, n_qubits))
        out = tmp.transpose(perm)
    return out.reshape((2**n_qubits,))


def _apply_2q(cp: Any, state, gate4, n_qubits: int, q0: int, q1: int):
    if q0 == q1:
        raise ValueError("q0 and q1 must differ")
    # Bring axes q0,q1 to front (in that order), apply 4x4, then invert permutation.
    psi = state.reshape((2,) * n_qubits)
    perm = [q0, q1] + [i for i in range(n_qubits) if i not in (q0, q1)]
    inv = [0] * n_qubits
    for i, p in enumerate(perm):
        inv[p] = i
    front = psi.transpose(perm).reshape(4, -1)
    front2 = gate4 @ front
    psi2 = front2.reshape((2, 2) + (2,) * (n_qubits - 2)).transpose(inv)
    return psi2.reshape((2**n_qubits,))


def _host(value: Any) -> Any:
    try:
        return value.get()
    except AttributeError:
        return value


def _sample_state(cp: Any, state: Any, rng: Any, n: int) -> tuple[int, float]:
    raw_norm2 = float(_host(cp.sum(cp.abs(state) ** 2)))
    if not math.isfinite(raw_norm2) or raw_norm2 <= 0:
        raise RuntimeError("statevector normalization failed")
    probs = (cp.abs(state) ** 2).astype(cp.float64, copy=False) / raw_norm2
    cdf = cp.cumsum(probs)
    random_value = rng.random_sample() if hasattr(rng, "random_sample") else rng.random()
    index = cp.searchsorted(cdf, cp.asarray([random_value], dtype=cp.float64))[0]
    return int(_host(index)), raw_norm2


def simulate(cp: Any, payload: SimulatePayload, progress_cb: Any = None, cancel_cb: Any = None) -> dict[str, Any]:
    n = int(payload.n_qubits)
    if n > MAX_STATEVECTOR_QUBITS:
        raise HTTPException(status_code=422, detail=f"statevector supports at most {MAX_STATEVECTOR_QUBITS} qubits")
    if statevector_memory_mb(n, payload.dtype) > payload.max_mem_mb:
        raise HTTPException(status_code=422, detail="statevector allocation exceeds max_mem_mb")
    dtype = cp.complex64 if payload.dtype == "complex64" else cp.complex128

    state = cp.zeros((2**n,), dtype=dtype)
    state[0] = 1
    if progress_cb:
        progress_cb(0.10, "init_state")

    t0 = time.perf_counter()
    total_gates = max(1, len(payload.gates))
    for index, g in enumerate(payload.gates):
        if cancel_cb and cancel_cb():
            raise RuntimeError("job canceled")
        if g.target >= n:
            raise HTTPException(status_code=400, detail=f"Gate target out of range: {g.target}")
        if g.name in ("h", "x", "rx", "ry", "rz"):
            if g.name in ("rx", "ry", "rz") and g.theta is None:
                raise HTTPException(status_code=422, detail=f"unresolved parameter for {g.name}: {g.parameter}")
            gate2 = gate_matrix(cp, g, dtype)
            state = _apply_1q(cp, state, gate2, n_qubits=n, target=g.target)
        elif g.name in ("cx", "cz"):
            if g.control is None:
                raise HTTPException(status_code=400, detail=f"Gate {g.name} requires control.")
            if g.control >= n:
                raise HTTPException(status_code=400, detail=f"Gate control out of range: {g.control}")
            if g.control == g.target:
                raise HTTPException(status_code=400, detail="control and target must differ.")
            gate4 = cx_matrix(cp, dtype) if g.name == "cx" else cz_matrix(cp, dtype)
            state = _apply_2q(cp, state, gate4, n_qubits=n, q0=g.control, q1=g.target)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported gate: {g.name}")
        if progress_cb:
            progress_cb(0.10 + 0.80 * ((index + 1) / total_gates), "statevector-gate")
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    if progress_cb:
        progress_cb(0.95, "statevector-readout")

    norm2 = float(cp.sum(cp.abs(state) ** 2).get())
    preview_len = min(16, 2**n)
    probs = (cp.abs(state[:preview_len]) ** 2).get().tolist()

    return {
        "status": "done",
        "backend": "cupy-statevector",
        "n_qubits": n,
        "dtype": payload.dtype,
        "time_ms": round((t1 - t0) * 1000, 3),
        "norm2": norm2,
        "validation": {"normalization_error": abs(norm2 - 1.0), "normalization_passed": abs(norm2 - 1.0) < 1e-5},
        "preview_probabilities": probs,
    }


def sample(cp: Any, payload: SamplePayload, progress_cb=None, cancel_cb=None) -> dict[str, Any]:
    """
    GPU statevector sampling for small n (<=20).
    Supports 1q gates (h/x/rx/ry/rz) and 2q gates (cx/cz).
    """
    n = int(payload.n_qubits)
    shots = int(payload.shots)
    if n > MAX_STATEVECTOR_QUBITS:
        raise HTTPException(status_code=422, detail=f"statevector supports at most {MAX_STATEVECTOR_QUBITS} qubits")
    dtype = cp.complex64 if payload.dtype == "complex64" else cp.complex128

    if payload.noise is not None and payload.noise.active:
        started = time.perf_counter()
        rng = cp.random.RandomState(payload.seed) if payload.seed is not None else cp.random
        out_counts: dict[str, int] = {}
        norm_sum = 0.0
        for shot in range(shots):
            if cancel_cb and cancel_cb():
                raise RuntimeError("job canceled")
            state = cp.zeros((2**n,), dtype=dtype)
            state[0] = 1
            for g in payload.gates:
                if g.target >= n:
                    raise HTTPException(status_code=400, detail=f"Gate target out of range: {g.target}")
                if g.name in ("h", "x", "rx", "ry", "rz"):
                    if g.name in ("rx", "ry", "rz") and g.theta is None:
                        raise HTTPException(status_code=422, detail=f"unresolved parameter for {g.name}: {g.parameter}")
                    state = _apply_1q(cp, state, gate_matrix(cp, g, dtype), n_qubits=n, target=g.target)
                    error = draw_paulis(cp, rng, payload.noise.one_qubit_depolarizing, 1)
                    if error:
                        matrix = pauli_matrix(cp, dtype, error[0])
                        if matrix is not None:
                            state = _apply_1q(cp, state, matrix, n_qubits=n, target=g.target)
                elif g.name in ("cx", "cz"):
                    if g.control is None or g.control >= n or g.control == g.target:
                        raise HTTPException(status_code=400, detail="two-qubit gate requires distinct in-range control and target")
                    gate4 = cx_matrix(cp, dtype) if g.name == "cx" else cz_matrix(cp, dtype)
                    state = _apply_2q(cp, state, gate4, n_qubits=n, q0=g.control, q1=g.target)
                    error = draw_paulis(cp, rng, payload.noise.two_qubit_depolarizing, 2)
                    if error:
                        for qubit, name in ((g.control, error[0]), (g.target, error[1])):
                            matrix = pauli_matrix(cp, dtype, name)
                            if matrix is not None:
                                state = _apply_1q(cp, state, matrix, n_qubits=n, target=qubit)
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported gate for sampling: {g.name}")
            index, norm2 = _sample_state(cp, state, rng, n)
            bitstring = readout_bits(rng, format(index, f"0{n}b"), payload.noise.readout_flip)
            out_counts[bitstring] = out_counts.get(bitstring, 0) + 1
            norm_sum += norm2
            if progress_cb and (shot == 0 or (shot + 1) % max(1, shots // 20) == 0):
                progress_cb((shot + 1) / shots, "statevector-noise-shot")
        norm2 = norm_sum / shots
        return {
            "status": "done",
            "backend": "cupy-statevector-sample",
            "method": "statevector-trajectories",
            "n_qubits": n,
            "dtype": payload.dtype,
            "shots": shots,
            "seed": payload.seed,
            "noise": noise_summary(payload.noise),
            "warnings": ["noise result is estimated by shot-based trajectories"],
            "norm2": norm2,
            "validation": {
                "normalization_error": abs(norm2 - 1.0),
                "normalization_passed": abs(norm2 - 1.0) < 1e-5,
                "counts_total": sum(out_counts.values()),
                "counts_total_passed": sum(out_counts.values()) == shots,
            },
            "time_ms": round((time.perf_counter() - started) * 1000, 3),
            "counts": out_counts,
        }

    if progress_cb:
        progress_cb(0.15, "init_state")
    state = cp.zeros((2**n,), dtype=dtype)
    state[0] = 1

    if progress_cb:
        progress_cb(0.25, "apply_gates")
    for idx, g in enumerate(payload.gates):
        if cancel_cb and cancel_cb():
            raise RuntimeError("Canceled")
        if g.target >= n:
            raise HTTPException(status_code=400, detail=f"Gate target out of range: {g.target}")
        if g.name in ("h", "x", "rx", "ry", "rz"):
            if g.name in ("rx", "ry", "rz") and g.theta is None:
                raise HTTPException(status_code=422, detail=f"unresolved parameter for {g.name}: {g.parameter}")
            gate2 = gate_matrix(cp, TNGate(name=g.name, target=g.target, theta=g.theta), dtype)
            state = _apply_1q(cp, state, gate2, n_qubits=n, target=g.target)
            continue
        if g.name in ("cx", "cz"):
            if g.control is None:
                raise HTTPException(status_code=400, detail=f"Gate {g.name} requires control.")
            if g.control >= n:
                raise HTTPException(status_code=400, detail=f"Gate control out of range: {g.control}")
            if g.control == g.target:
                raise HTTPException(status_code=400, detail="control and target must differ.")
            gate4 = cx_matrix(cp, dtype) if g.name == "cx" else cz_matrix(cp, dtype)
            state = _apply_2q(cp, state, gate4, n_qubits=n, q0=g.control, q1=g.target)
            continue
        raise HTTPException(status_code=400, detail=f"Unsupported gate for sampling: {g.name}")
    if progress_cb:
        progress_cb(0.60, "apply_gates")

    if progress_cb:
        progress_cb(0.70, "probabilities")
    raw_norm2 = float(cp.sum(cp.abs(state) ** 2).get())
    if not math.isfinite(raw_norm2) or raw_norm2 <= 0:
        raise RuntimeError("statevector normalization failed")
    probs = (cp.abs(state) ** 2).astype(cp.float64, copy=False)
    probs = probs / raw_norm2
    cdf = cp.cumsum(probs)
    if progress_cb:
        progress_cb(0.80, "sample")
    rng = cp.random.RandomState(payload.seed) if payload.seed is not None else cp.random
    rnd = rng.random_sample(shots).astype(cp.float64, copy=False)
    idx = cp.searchsorted(cdf, rnd).astype(cp.int64)
    idx = cp.minimum(idx, len(probs) - 1)
    cp.cuda.Stream.null.synchronize()

    if progress_cb:
        progress_cb(0.92, "count")
    # Count on GPU then move small dict to host
    unique, counts = cp.unique(idx, return_counts=True)
    unique_h = unique.get().tolist()
    counts_h = counts.get().tolist()

    out_counts: dict[str, int] = {}
    for u, c in zip(unique_h, counts_h):
        bit = format(int(u), f"0{n}b")
        out_counts[bit] = int(c)

    return {
        "status": "done",
        "backend": "cupy-statevector-sample",
        "n_qubits": n,
        "dtype": payload.dtype,
        "shots": shots,
        "seed": payload.seed,
        "norm2": raw_norm2,
        "validation": {
            "normalization_error": abs(raw_norm2 - 1.0),
            "normalization_passed": abs(raw_norm2 - 1.0) < 1e-5,
            "counts_total": sum(out_counts.values()),
            "counts_total_passed": sum(out_counts.values()) == shots,
        },
        "counts": out_counts,
    }
