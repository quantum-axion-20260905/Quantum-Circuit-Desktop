from __future__ import annotations

from typing import Any

from .models import NoiseModel


PAULIS = ("i", "x", "y", "z")


def _host(value: Any) -> Any:
    try:
        return value.get()
    except AttributeError:
        return value


def _random_value(rng: Any) -> float:
    value = rng.random_sample() if hasattr(rng, "random_sample") else rng.random()
    return float(_host(value))


def _random_index(rng: Any, low: int, high: int) -> int:
    return int(_host(rng.randint(low, high)))


def pauli_matrix(cp: Any, dtype: Any, name: str) -> Any | None:
    if name == "i":
        return None
    if name == "x":
        return cp.asarray([[0, 1], [1, 0]], dtype=dtype)
    if name == "y":
        return cp.asarray([[0, -1j], [1j, 0]], dtype=dtype)
    if name == "z":
        return cp.asarray([[1, 0], [0, -1]], dtype=dtype)
    raise ValueError(f"unknown Pauli: {name}")


def draw_paulis(cp: Any, rng: Any, probability: float, arity: int) -> tuple[str, ...] | None:
    """Draw a standard depolarizing Pauli error for one trajectory.

    `probability` is the channel probability. Conditional on an error, the
    non-identity Pauli (or Pauli product) is selected uniformly.
    """

    if probability <= 0 or _random_value(rng) >= probability:
        return None
    if arity == 1:
        return (PAULIS[_random_index(rng, 1, 3)],)
    if arity == 2:
        index = _random_index(rng, 1, 15)
        left = PAULIS[index // 4]
        right = PAULIS[index % 4]
        return left, right
    raise ValueError(f"unsupported depolarizing arity: {arity}")


def readout_bits(rng: Any, bits: str, probability: float) -> str:
    if probability <= 0:
        return bits
    return "".join(("1" if bit == "0" else "0") if _random_value(rng) < probability else bit for bit in bits)


def noise_summary(noise: NoiseModel | None) -> dict[str, Any] | None:
    if noise is None or not noise.active:
        return None
    return {
        "model": "pauli-depolarizing-trajectories",
        "one_qubit_depolarizing": noise.one_qubit_depolarizing,
        "two_qubit_depolarizing": noise.two_qubit_depolarizing,
        "readout_flip": noise.readout_flip,
    }
