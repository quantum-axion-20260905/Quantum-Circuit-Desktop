"""Stable interfaces for finite and infinite one-dimensional MPS methods.

The protocols in this module are deliberately small.  They define the shape
that a future TDVP or VUMPS implementation must satisfy without pretending
that either solver is executable today.  Capability admission lives in the
backend registry; numerical kernels should depend only on these contracts.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from .contracts import ResearchResult, ResourceBudget


@runtime_checkable
class TDVPSolver(Protocol):
    """Time-dependent variational principle runner over an MPS."""

    method: Literal["tdvp"]

    def run(self, problem: Any, budget: ResourceBudget) -> ResearchResult: ...


@runtime_checkable
class VUMPSSolver(Protocol):
    """Variational uniform matrix-product-state ground-state runner."""

    method: Literal["vumps"]

    def run(self, problem: Any, budget: ResourceBudget) -> ResearchResult: ...

