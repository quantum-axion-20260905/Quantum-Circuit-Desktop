from __future__ import annotations

import math
import sys

"""Shared execution limits and memory estimates.

These limits are intentionally conservative.  A research tool should reject an
unsafe request before allocating a large statevector instead of discovering the
problem after CUDA or the operating system starts swapping.
"""

BYTES_PER_COMPLEX = {"complex64": 8, "complex128": 16}
MAX_REFERENCE_QUBITS = 20
MAX_STATEVECTOR_QUBITS = 20
MAX_MPS_QUBITS = 4096
MAX_EXACT_DIAGONALIZATION_QUBITS = 12


def statevector_memory_mb(n_qubits: int, dtype: str) -> float:
    # Keep the conservative upper-bound diagnostic representable even when an
    # MPS request has thousands of qubits.  The actual statevector backend is
    # separately capped at MAX_STATEVECTOR_QUBITS before allocation.
    try:
        return math.ldexp(BYTES_PER_COMPLEX[dtype], int(n_qubits) - 20)
    except OverflowError:
        return sys.float_info.max


def mps_memory_mb(n_qubits: int, bond_dim: int, dtype: str) -> float:
    """Conservative MPS tensor, environment, and SVD workspace estimate.

    Sampling and amplitude readout retain one right-environment matrix per
    site in addition to the MPS tensors.  Counting that ``O(n*chi**2)`` buffer
    prevents large low-bond requests from being systematically underpriced.
    """
    d = max(1, int(bond_dim))
    values = (max(1, int(n_qubits)) * 3 * d * d) + (8 * d * d)
    return values * BYTES_PER_COMPLEX[dtype] / (1024 * 1024)
