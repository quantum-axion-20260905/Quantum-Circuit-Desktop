from __future__ import annotations

import time
from typing import Any

from fastapi import HTTPException

from ..gates import basis_ket, cx_tensor, cz_tensor, gate_matrix
from ..models import Gate, TNPayload


def _require_tn_deps(oe: Any) -> None:
    if oe is None:
        raise HTTPException(status_code=500, detail="opt_einsum is not available.")


def _parse_bitstrings(bitstrings: list[str], n_qubits: int) -> list[list[int]]:
    if not bitstrings:
        return [[0] * n_qubits]
    out: list[list[int]] = []
    for s in bitstrings:
        ss = s.strip().replace("_", "")
        if len(ss) != n_qubits:
            raise HTTPException(status_code=400, detail=f"Bitstring length must be {n_qubits}: {s!r}")
        if any(ch not in "01" for ch in ss):
            raise HTTPException(status_code=400, detail=f"Invalid bitstring: {s!r}")
        out.append([0 if ch == "0" else 1 for ch in ss])
    return out


def _build_operands(cp: Any, payload: TNPayload):
    n = int(payload.n_qubits)
    dtype = cp.complex64 if payload.dtype == "complex64" else cp.complex128

    next_ix = 0
    wire_ix: list[int] = []
    operands: list[Any] = []

    for _q in range(n):
        ix = next_ix
        next_ix += 1
        wire_ix.append(ix)
        operands.extend([basis_ket(cp, 0, dtype), [ix]])

    for g in payload.gates:
        if g.target >= n:
            raise HTTPException(status_code=400, detail=f"Gate target out of range: {g.target}")

        if g.name in ("h", "x", "rx", "ry", "rz"):
            gate2 = gate_matrix(cp, Gate(name=g.name, target=g.target, theta=g.theta), dtype)
            in_ix = wire_ix[g.target]
            out_ix = next_ix
            next_ix += 1
            wire_ix[g.target] = out_ix
            operands.extend([gate2, [out_ix, in_ix]])
            continue

        if g.name in ("cx", "cz"):
            if g.control is None:
                raise HTTPException(status_code=400, detail=f"Gate {g.name} requires control.")
            if g.control >= n:
                raise HTTPException(status_code=400, detail=f"Gate control out of range: {g.control}")
            if g.control == g.target:
                raise HTTPException(status_code=400, detail="control and target must differ.")

            in_c = wire_ix[g.control]
            in_t = wire_ix[g.target]
            out_c = next_ix
            out_t = next_ix + 1
            next_ix += 2
            wire_ix[g.control] = out_c
            wire_ix[g.target] = out_t
            tensor4 = cx_tensor(cp, dtype) if g.name == "cx" else cz_tensor(cp, dtype)
            operands.extend([tensor4, [out_c, out_t, in_c, in_t]])
            continue

        raise HTTPException(status_code=400, detail=f"Unknown gate: {g.name}")

    return operands, wire_ix, dtype


def amplitudes(cp: Any, oe: Any, ctg: Any, payload: TNPayload) -> dict[str, Any]:
    _require_tn_deps(oe)
    bits_list = _parse_bitstrings(payload.bitstrings, payload.n_qubits)
    t0 = time.perf_counter()

    amps = []
    for bits in bits_list:
        operands, wire_ix, dtype = _build_operands(cp, payload)
        for q, b in enumerate(bits):
            ix = wire_ix[q]
            operands.extend([basis_ket(cp, b, dtype).conj(), [ix]])

        if payload.optimize == "cotengra" and ctg is not None:
            optimizer = ctg.HyperOptimizer(max_repeats=32, progbar=False)
            amp = oe.contract(*operands, [], optimize=optimizer, backend="cupy")
        else:
            amp = oe.contract(*operands, [], optimize="auto-hq", backend="cupy")
        cp.cuda.Stream.null.synchronize()
        amps.append(complex(amp.get()))

    t1 = time.perf_counter()
    return {
        "status": "done",
        "backend": "tensor-network",
        "n_qubits": payload.n_qubits,
        "dtype": payload.dtype,
        "optimize": payload.optimize,
        "time_ms": round((t1 - t0) * 1000, 3),
        "amplitudes": [
            {"bitstring": "".join(map(str, bits)), "re": a.real, "im": a.imag}
            for bits, a in zip(bits_list, amps)
        ],
    }


def estimate(cp: Any, oe: Any, ctg: Any, payload: TNPayload) -> dict[str, Any]:
    _require_tn_deps(oe)
    operands, wire_ix, dtype = _build_operands(cp, payload)
    for ix in wire_ix:
        operands.extend([basis_ket(cp, 0, dtype).conj(), [ix]])

    if payload.optimize == "cotengra" and ctg is not None:
        optimizer = ctg.HyperOptimizer(max_repeats=32, progbar=False)
        path, info = oe.contract_path(*operands, [], optimize=optimizer)
    else:
        path, info = oe.contract_path(*operands, [], optimize="auto-hq")

    head = "\n".join(str(info).splitlines()[:10])
    result: dict[str, Any] = {
        "status": "done",
        "backend": "tensor-network",
        "n_qubits": payload.n_qubits,
        "dtype": payload.dtype,
        "optimize": payload.optimize,
        "path_steps": len(path),
        "summary_head": head,
    }
    for attr in ("opt_cost", "largest_intermediate", "naive_cost", "speedup"):
        if hasattr(info, attr):
            result[attr] = getattr(info, attr)
    return result

