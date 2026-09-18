"""Run a bounded entangled-iPEPS gradient and gauge acceptance gate.

The gate is intentionally a diagnostic, not an optimizer benchmark.  It
compares selected Torch autograd components with central differences and
measures the same objective after a paired virtual gauge transformation.  A
small residual alone is not sufficient for admission: generic D=2 cells must
also pass these independent checks before the gradient path can be called
research-grade.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qc_agent.core.ctmrg_autodiff import differentiable_ctmrg_energy
from qc_agent.core.ctmrg_gauge import paired_virtual_gauge
from qc_agent.plugins.models import CTMRGPayload, IPEPSInteraction
from qc_agent.provenance import sha256_json


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def build_gradient_payload(dtype: str, *, iterations: int, tolerance: float) -> CTMRGPayload:
    """Build the fixed one-site D=2 nearest-neighbor gate problem."""

    return CTMRGPayload(
        dtype=dtype,
        virtual_bond_dim=2,
        environment_bond_dim=2,
        iterations=int(iterations),
        tolerance=float(tolerance),
        optimization="full-update",
        full_update_optimizer="autodiff-ctmrg-gradient",
        interactions=[IPEPSInteraction(
            left_site=0,
            right_site=0,
            displacement=[1, 0],
            left_pauli="X",
            right_pauli="X",
            coefficient=-0.2,
        )],
    )


def run_gradient_gate(torch: Any, *, dtype_name: str) -> dict[str, Any]:
    """Run one deterministic dtype case on the caller-selected Torch device."""

    device = torch.device("cuda")
    torch.manual_seed(20260918)
    if dtype_name == "complex64":
        dtype = torch.complex64
        real_dtype = torch.float32
        epsilon = 2e-3
        iterations = 3
        tolerance = 1e-5
    elif dtype_name == "complex128":
        dtype = torch.complex128
        real_dtype = torch.float64
        epsilon = 1e-5
        iterations = 5
        tolerance = 1e-7
    else:
        raise ValueError("gradient gate supports complex64 and complex128")

    raw = torch.randn((2, 2, 2, 2, 2), dtype=real_dtype, device=device)
    raw = raw + 1j * torch.randn((2, 2, 2, 2, 2), dtype=real_dtype, device=device)
    tensor = (raw / torch.linalg.norm(raw)).to(dtype).requires_grad_(True)
    payload = build_gradient_payload(dtype_name, iterations=iterations, tolerance=tolerance)
    energy, diagnostics = differentiable_ctmrg_energy(torch, payload, [tensor])
    gradient = torch.autograd.grad(energy, [tensor])[0].detach().reshape(-1)

    samples: list[dict[str, Any]] = []
    for index in (0, 1, 2, 3):
        for direction in (1.0, 1.0j):
            plus = tensor.detach().clone()
            minus = tensor.detach().clone()
            plus.reshape(-1)[index] += epsilon * direction
            minus.reshape(-1)[index] -= epsilon * direction
            plus_energy, _ = differentiable_ctmrg_energy(torch, payload, [plus])
            minus_energy, _ = differentiable_ctmrg_energy(torch, payload, [minus])
            finite_difference = float(((plus_energy - minus_energy) / (2 * epsilon)).detach().cpu())
            autodiff = float(
                (gradient[index].real if direction == 1.0 else gradient[index].imag).cpu()
            )
            samples.append({
                "index": int(index),
                "direction": "real" if direction == 1.0 else "imag",
                "autodiff": autodiff,
                "finite_difference": finite_difference,
                "abs_error": abs(autodiff - finite_difference),
            })

    gauged_tensor = paired_virtual_gauge(torch, [tensor.detach()])[0]
    gauged_energy, _ = differentiable_ctmrg_energy(torch, payload, [gauged_tensor])
    gauge_delta = abs(float((energy - gauged_energy).detach().cpu()))
    max_gradient_error = max(item["abs_error"] for item in samples)
    gradient_tolerance = 1e-3
    gauge_tolerance = 1e-4
    return {
        "dtype": dtype_name,
        "device": str(device),
        "energy": float(energy.detach().cpu()),
        "gauged_energy": float(gauged_energy.detach().cpu()),
        "gauge_energy_delta": gauge_delta,
        "gauge_tolerance": gauge_tolerance,
        "gradient_tolerance": gradient_tolerance,
        "max_gradient_abs_error": max_gradient_error,
        "gradient_passed": bool(max_gradient_error <= gradient_tolerance),
        "gauge_passed": bool(gauge_delta <= gauge_tolerance),
        "passed": bool(max_gradient_error <= gradient_tolerance and gauge_delta <= gauge_tolerance),
        "diagnostics": diagnostics,
        "samples": samples,
        "limitations": [
            "selected finite-difference components only, not a full tensor Jacobian",
            "the bounded objective uses a frozen truncation projector",
            "a passed small gate would still not prove thermodynamic-limit convergence",
        ],
    }


def run_gate(torch: Any) -> dict[str, Any]:
    """Run both precision cases and return a versioned acceptance artifact."""

    cases = [
        run_gradient_gate(torch, dtype_name="complex64"),
        run_gradient_gate(torch, dtype_name="complex128"),
    ]
    return {
        "schema": "quantum-circuit/ctmrg-gradient-gate-v1",
        "status": "passed" if all(case["passed"] for case in cases) else "needs_review",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": _git_revision(),
        "cases": cases,
        "limitations": [
            "this is bounded acceptance evidence for a generic one-site D=2 cell",
            "failure keeps the infinite-CTMRG gradient path out of production admission",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON artifact path")
    args = parser.parse_args()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise SystemExit(f"Torch is required for the gradient gate: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; refusing to silently run the gradient gate on CPU")
    artifact = run_gate(torch)
    artifact["result_sha256"] = sha256_json(artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "status": artifact["status"],
        "result_sha256": artifact["result_sha256"],
        "git_revision": artifact["git_revision"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
