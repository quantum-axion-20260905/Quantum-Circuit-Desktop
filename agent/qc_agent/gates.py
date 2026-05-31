from __future__ import annotations

import cmath
import math
from typing import Any

from fastapi import HTTPException

from .models import Gate


def gate_matrix(cp: Any, gate: Gate, dtype: Any):
    if gate.name == "h":
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        return cp.asarray([[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]], dtype=dtype)
    if gate.name == "x":
        return cp.asarray([[0, 1], [1, 0]], dtype=dtype)

    if gate.theta is None:
        raise HTTPException(status_code=400, detail=f"Gate {gate.name} requires theta.")
    theta = float(gate.theta)
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)

    if gate.name == "rx":
        return cp.asarray([[c, -1j * s], [-1j * s, c]], dtype=dtype)
    if gate.name == "ry":
        return cp.asarray([[c, -s], [s, c]], dtype=dtype)
    if gate.name == "rz":
        e0 = cmath.exp(-1j * theta / 2)
        e1 = cmath.exp(1j * theta / 2)
        return cp.asarray([[e0, 0], [0, e1]], dtype=dtype)

    raise HTTPException(status_code=400, detail=f"Unknown gate: {gate.name}")


def basis_ket(cp: Any, bit: int, dtype: Any):
    if bit == 0:
        return cp.asarray([1, 0], dtype=dtype)
    if bit == 1:
        return cp.asarray([0, 1], dtype=dtype)
    raise ValueError("bit must be 0 or 1")


def cx_tensor(cp: Any, dtype: Any):
    t = cp.zeros((2, 2, 2, 2), dtype=dtype)
    for ic in (0, 1):
        for it in (0, 1):
            oc = ic
            ot = it ^ ic
            t[oc, ot, ic, it] = 1
    return t


def cz_tensor(cp: Any, dtype: Any):
    t = cp.zeros((2, 2, 2, 2), dtype=dtype)
    for ic in (0, 1):
        for it in (0, 1):
            phase = -1 if (ic == 1 and it == 1) else 1
            t[ic, it, ic, it] = phase
    return t


def cx_matrix(cp: Any, dtype: Any):
    # 4x4 CX acting on |c,t> (c is first qubit).
    return cp.asarray(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=dtype,
    )


def cz_matrix(cp: Any, dtype: Any):
    return cp.asarray(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1],
        ],
        dtype=dtype,
    )
