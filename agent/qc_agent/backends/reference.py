from __future__ import annotations

import math
import random
from typing import Any

from ..models import RunPayload, TNGate


def _matrix(name: str, theta: float | None = None) -> list[list[complex]]:
    if name == "h":
        s = 1 / math.sqrt(2)
        return [[s, s], [s, -s]]
    if name == "x":
        return [[0, 1], [1, 0]]
    if theta is None:
        raise ValueError(f"Gate {name} requires theta")
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    if name == "rx": return [[c, -1j*s], [-1j*s, c]]
    if name == "ry": return [[c, -s], [s, c]]
    if name == "rz": return [[complex(math.cos(-theta/2), math.sin(-theta/2)), 0], [0, complex(math.cos(theta/2), math.sin(theta/2))]]
    raise ValueError(f"Unknown gate {name}")


def _apply_1q(state: list[complex], n: int, q: int, m: list[list[complex]]) -> None:
    step = 1 << (n - q - 1)
    span = step * 2
    for base in range(0, len(state), span):
        for off in range(step):
            i0, i1 = base + off, base + off + step
            a, b = state[i0], state[i1]
            state[i0] = m[0][0]*a + m[0][1]*b
            state[i1] = m[1][0]*a + m[1][1]*b


def _apply_2q(state: list[complex], n: int, control: int, target: int, name: str) -> None:
    cbit, tbit = 1 << (n-control-1), 1 << (n-target-1)
    for i in range(len(state)):
        if name == "cx" and (i & cbit):
            j = i ^ tbit
            if i < j: state[i], state[j] = state[j], state[i]
        elif name == "cz" and (i & cbit) and (i & tbit):
            state[i] *= -1


def run(payload: RunPayload) -> dict[str, Any]:
    n = payload.n_qubits
    state = [0j] * (1 << n)
    state[0] = 1 + 0j
    for g in payload.gates:
        if g.target >= n or (g.control is not None and g.control >= n):
            raise ValueError("gate index exceeds qubit count")
        if g.name in ("h", "x", "rx", "ry", "rz"):
            _apply_1q(state, n, g.target, _matrix(g.name, g.theta))
        else:
            if g.control is None or g.control == g.target: raise ValueError("invalid two-qubit gate")
            _apply_2q(state, n, g.control, g.target, g.name)

    probs = [abs(x)**2 for x in state]
    bits = payload.bitstrings or [format(i, f"0{n}b") for i, p in enumerate(probs) if p > 1e-14]
    amps = [{"bitstring": b, "re": state[int(b, 2)].real, "im": state[int(b, 2)].imag} for b in bits]
    rng = random.Random(payload.seed)
    counts: dict[str, int] = {}
    for _ in range(payload.shots):
        r, acc = rng.random(), 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                b = format(i, f"0{n}b"); counts[b] = counts.get(b, 0) + 1; break
    return {"status": "done", "backend": "reference-cpu", "device": "CPU", "result_type": "samples", "n_qubits": n, "shots": payload.shots, "counts": counts, "amplitudes": amps, "norm2": sum(probs), "performance_comparable": False}
