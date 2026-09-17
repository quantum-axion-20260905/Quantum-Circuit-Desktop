from __future__ import annotations

import math
import random
from typing import Any

from ..models import RunPayload, TNGate
from ..noise import draw_paulis, noise_summary, readout_bits
from .limits import MAX_REFERENCE_QUBITS



def _matrix(name: str, theta: float | None = None) -> list[list[complex]]:
    if name == "h":
        s = 1 / math.sqrt(2)
        return [[s, s], [s, -s]]
    if name == "x":
        return [[0, 1], [1, 0]]
    if name == "y":
        return [[0, -1j], [1j, 0]]
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


def _apply_pauli(state: list[complex], n: int, q: int, name: str) -> None:
    if name == "i":
        return
    _apply_1q(state, n, q, _matrix(name))


def run(payload: RunPayload) -> dict[str, Any]:
    n = payload.n_qubits
    if n > MAX_REFERENCE_QUBITS:
        raise ValueError(f"reference-cpu supports at most {MAX_REFERENCE_QUBITS} qubits")
    if payload.noise is not None and payload.noise.active and payload.result_type == "selected_amplitudes":
        raise ValueError("noise channels require result_type=samples; amplitudes are noiseless-only")
    state = [0j] * (1 << n)
    state[0] = 1 + 0j
    rng = random.Random(payload.seed)
    for g in payload.gates:
        if g.target >= n or (g.control is not None and g.control >= n):
            raise ValueError("gate index exceeds qubit count")
        if g.name in ("h", "x", "rx", "ry", "rz"):
            if g.name in ("rx", "ry", "rz") and g.theta is None:
                raise ValueError(f"unresolved parameter for {g.name}: {g.parameter}")
            _apply_1q(state, n, g.target, _matrix(g.name, g.theta))
            if payload.noise is not None:
                error = draw_paulis(None, rng, payload.noise.one_qubit_depolarizing, 1)
                if error:
                    _apply_pauli(state, n, g.target, error[0])
        else:
            if g.control is None or g.control == g.target: raise ValueError("invalid two-qubit gate")
            _apply_2q(state, n, g.control, g.target, g.name)
            if payload.noise is not None:
                error = draw_paulis(None, rng, payload.noise.two_qubit_depolarizing, 2)
                if error:
                    _apply_pauli(state, n, g.control, error[0])
                    _apply_pauli(state, n, g.target, error[1])

    probs = [abs(x)**2 for x in state]
    if payload.bitstrings:
        bits = []
        for raw in payload.bitstrings:
            bitstring = raw.strip().replace("_", "")
            if len(bitstring) != n or any(ch not in "01" for ch in bitstring):
                raise ValueError(f"invalid bitstring for {n} qubits: {raw!r}")
            bits.append(bitstring)
    else:
        bits = [format(i, f"0{n}b") for i, p in enumerate(probs) if p > 1e-14]
    amps = [{"bitstring": b, "re": state[int(b, 2)].real, "im": state[int(b, 2)].imag} for b in bits]
    common = {"status": "done", "backend": "reference-cpu", "device": "CPU", "n_qubits": n, "amplitudes": amps, "norm2": sum(probs), "performance_comparable": False, "noise": noise_summary(payload.noise)}
    if payload.result_type == "selected_amplitudes":
        return {**common, "result_type": "selected_amplitudes"}

    counts: dict[str, int] = {}
    total = sum(probs)
    for _ in range(payload.shots):
        r, acc = rng.random() * total, 0.0
        selected = False
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                b = readout_bits(rng, format(i, f"0{n}b"), payload.noise.readout_flip if payload.noise is not None else 0.0)
                counts[b] = counts.get(b, 0) + 1; selected = True; break
        if not selected:
            b = readout_bits(rng, format(len(probs) - 1, f"0{n}b"), payload.noise.readout_flip if payload.noise is not None else 0.0)
            counts[b] = counts.get(b, 0) + 1
    return {**common, "result_type": "samples", "shots": payload.shots, "counts": counts}
