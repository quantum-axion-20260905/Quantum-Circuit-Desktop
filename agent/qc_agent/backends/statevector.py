from __future__ import annotations

import time
from typing import Any

from fastapi import HTTPException

from ..gates import cx_matrix, cz_matrix, gate_matrix
from ..models import SamplePayload, SimulatePayload, TNGate


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


def simulate(cp: Any, payload: SimulatePayload) -> dict[str, Any]:
    n = int(payload.n_qubits)
    dtype = cp.complex64 if payload.dtype == "complex64" else cp.complex128

    state = cp.zeros((2**n,), dtype=dtype)
    state[0] = 1

    t0 = time.perf_counter()
    for g in payload.gates:
        if g.target >= n:
            raise HTTPException(status_code=400, detail=f"Gate target out of range: {g.target}")
        gate2 = gate_matrix(cp, g, dtype)
        state = _apply_1q(cp, state, gate2, n_qubits=n, target=g.target)
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()

    preview_len = min(16, 2**n)
    probs = (cp.abs(state[:preview_len]) ** 2).get().tolist()

    return {
        "status": "done",
        "backend": "cupy-statevector",
        "n_qubits": n,
        "dtype": payload.dtype,
        "time_ms": round((t1 - t0) * 1000, 3),
        "preview_probabilities": probs,
    }


def sample(cp: Any, payload: SamplePayload, progress_cb=None, cancel_cb=None) -> dict[str, Any]:
    """
    GPU statevector sampling for small n (<=20).
    Supports 1q gates (h/x/rx/ry/rz) and 2q gates (cx/cz).
    """
    n = int(payload.n_qubits)
    shots = int(payload.shots)
    dtype = cp.complex64 if payload.dtype == "complex64" else cp.complex128

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
            gate2 = gate_matrix(cp, TNGate(name=g.name, target=g.target, theta=g.theta), dtype)
            state = _apply_1q(cp, state, gate2, n_qubits=n, target=g.target)
            continue
        if g.name in ("cx", "cz"):
            if g.control is None:
                raise HTTPException(status_code=400, detail=f"Gate {g.name} requires control.")
            if g.control >= n:
                raise HTTPException(status_code=400, detail=f"Gate control out of range: {g.control}")
            gate4 = cx_matrix(cp, dtype) if g.name == "cx" else cz_matrix(cp, dtype)
            state = _apply_2q(cp, state, gate4, n_qubits=n, q0=g.control, q1=g.target)
            continue
        raise HTTPException(status_code=400, detail=f"Unsupported gate for sampling: {g.name}")
        if progress_cb and (idx % 25 == 0):
            progress_cb(0.25 + 0.35 * (idx / max(1, len(payload.gates))), "apply_gates")

    if progress_cb:
        progress_cb(0.70, "probabilities")
    probs = (cp.abs(state) ** 2).astype(cp.float64, copy=False)
    probs = probs / probs.sum()
    cdf = cp.cumsum(probs)
    if progress_cb:
        progress_cb(0.80, "sample")
    rnd = cp.random.random(shots, dtype=cp.float64)
    idx = cp.searchsorted(cdf, rnd).astype(cp.int64)
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

    return {"status": "done", "backend": "cupy-statevector-sample", "n_qubits": n, "shots": shots, "counts": out_counts}
