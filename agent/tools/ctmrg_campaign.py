"""Run and record a bounded CTMRG GPU convergence campaign.

The runner is intentionally separate from the HTTP server.  It exercises the
same public CTMRG core on a real CUDA array, records enough provenance to
replay the campaign, and refuses to fall back to CPU when a GPU is requested.
The cases are small by design: this is acceptance evidence, not a stress test.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from qc_agent.core.ctmrg import run_ctmrg_convergence_study
from qc_agent.plugins.models import CTMRGPayload, IPEPSInteraction, PauliTerm
from qc_agent.provenance import sha256_json


CELL_SHAPES: tuple[tuple[int, int], ...] = ((1, 1), (2, 1), (1, 2), (2, 2))
DEFAULT_ENVIRONMENT_DIMS = (1, 2, 4)
CAMPAIGN_SEED = 0


def _cell_site(x: int, y: int, cell: tuple[int, int]) -> int:
    width, height = cell
    return (int(x) % width) + width * (int(y) % height)


def _nearest_neighbor_interactions(cell: tuple[int, int]) -> list[IPEPSInteraction]:
    """Return one positive-direction bond per cell row/column."""

    width, height = cell
    interactions: list[IPEPSInteraction] = []
    for y in range(height):
        for x in range(width):
            left_site = _cell_site(x, y, cell)
            right_site = _cell_site(x + 1, y, cell)
            interactions.append(IPEPSInteraction(
                left_site=left_site,
                right_site=right_site,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=-1.0,
                label=f"zz-x-{x}-{y}",
            ))
    for y in range(height):
        for x in range(width):
            left_site = _cell_site(x, y, cell)
            right_site = _cell_site(x, y + 1, cell)
            interactions.append(IPEPSInteraction(
                left_site=left_site,
                right_site=right_site,
                displacement=[0, 1],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=-1.0,
                label=f"zz-y-{x}-{y}",
            ))
    return interactions


def build_product_payload(cell: tuple[int, int], *, iterations: int = 4) -> CTMRGPayload:
    """Build a declared product-limit acceptance case for one cell shape."""

    return CTMRGPayload(
        unit_cell=list(cell),
        initial_state="up",
        dtype="complex128",
        virtual_bond_dim=1,
        environment_bond_dim=max(DEFAULT_ENVIRONMENT_DIMS),
        iterations=int(iterations),
        tolerance=1e-8,
        interactions=_nearest_neighbor_interactions(cell),
    )


def build_entangled_payload(*, iterations: int = 3) -> CTMRGPayload:
    """Build a deterministic generic D=2 2x2 case for failed-reference evidence."""

    tensors: list[list[float]] = []
    for site in range(4):
        tensor = np.zeros((2, 2, 2, 2, 2), dtype=np.complex128)
        tensor[0, 0, 0, 0, 0] = 1.0 + 0.05j * site
        tensor[1, 1, 1, 1, 1] = 0.2 + 0.03j * site
        tensor[0, 0, 1, 1, 0] = 0.1
        tensors.extend([[float(value.real), float(value.imag)] for value in tensor.reshape(-1)])
    return CTMRGPayload(
        unit_cell=[2, 2],
        dtype="complex128",
        virtual_bond_dim=2,
        tensor_data=tensors,
        environment_bond_dim=max(DEFAULT_ENVIRONMENT_DIMS),
        iterations=int(iterations),
        tolerance=1e-8,
        terms=[PauliTerm(paulis={3: "Z"}, coefficient=0.1, label="z3")],
        interactions=[
            IPEPSInteraction(
                left_site=0,
                right_site=1,
                displacement=[1, 0],
                left_pauli="Z",
                right_pauli="Z",
                coefficient=0.2,
                label="zz-horizontal",
            ),
            IPEPSInteraction(
                left_site=0,
                right_site=2,
                displacement=[0, 1],
                left_pauli="X",
                right_pauli="X",
                coefficient=-0.1,
                label="xx-vertical",
            ),
        ],
    )


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


def _gpu_snapshot(cp: Any) -> dict[str, Any]:
    device_id = int(cp.cuda.Device().id)
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    free_bytes, total_bytes = cp.cuda.Device().mem_info
    return {
        "device_id": device_id,
        "name": str(name),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "free_mb": round(int(free_bytes) / (1024 * 1024), 3),
        "total_mb": round(int(total_bytes) / (1024 * 1024), 3),
    }


def _run_case(cp: Any, name: str, payload: CTMRGPayload, environment_dims: tuple[int, ...]) -> dict[str, Any]:
    request = payload.model_dump(mode="json")
    request_hash = sha256_json({"problem": request, "environment_bond_dims": list(environment_dims)})
    pool = cp.get_default_memory_pool()
    before = _gpu_snapshot(cp)
    peak_pool_bytes = int(pool.used_bytes())

    def progress(_: float, __: str) -> None:
        nonlocal peak_pool_bytes
        peak_pool_bytes = max(peak_pool_bytes, int(pool.used_bytes()))

    started = time.perf_counter()
    study = run_ctmrg_convergence_study(
        cp,
        payload,
        list(environment_dims),
        progress_cb=progress,
    )
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - started
    after = _gpu_snapshot(cp)
    peak_pool_bytes = max(peak_pool_bytes, int(pool.used_bytes()))
    result_hash = sha256_json(study)
    study["provenance"] = {
        "request_sha256": request_hash,
        "result_sha256": result_hash,
        "git_revision": _git_revision(),
    }
    return {
        "name": name,
        "status": "done",
        "request": request,
        "request_sha256": request_hash,
        "result_sha256": result_hash,
        "elapsed_ms": round(elapsed * 1000.0, 3),
        "gpu_before": before,
        "gpu_after": after,
        "peak_memory_pool_used_bytes": peak_pool_bytes,
        "study": study,
    }


def run_campaign(
    cp: Any,
    *,
    environment_dims: tuple[int, ...] = DEFAULT_ENVIRONMENT_DIMS,
    product_iterations: int = 4,
    entangled_iterations: int = 3,
) -> dict[str, Any]:
    """Run all bounded Phase 4 evidence cases on the supplied CuPy module."""

    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    with cp.cuda.Device(0):
        for width, height in CELL_SHAPES:
            cases.append(_run_case(
                cp,
                f"product-{width}x{height}",
                build_product_payload((width, height), iterations=product_iterations),
                environment_dims,
            ))
        cases.append(_run_case(
            cp,
            "generic-entangled-2x2",
            build_entangled_payload(iterations=entangled_iterations),
            environment_dims[:2],
        ))
    passed_product_cases = sum(
        1
        for case in cases
        if case["name"].startswith("product-")
        and case["study"]["reference_summary"]["passed_points"] == len(case["study"]["points"])
    )
    return {
        "schema": "quantum-circuit/ctmrg-gpu-campaign-v1",
        "status": "done",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": _git_revision(),
        "seed": CAMPAIGN_SEED,
        "randomized": False,
        "environment_bond_dims": list(environment_dims),
        "cell_shapes": [list(shape) for shape in CELL_SHAPES],
        "case_count": len(cases),
        "product_cases_passed": passed_product_cases,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "cases": cases,
        "limitations": [
            "campaign is bounded acceptance evidence, not a thermodynamic-limit benchmark",
            "product cases validate the admitted product reference; the generic entangled case is expected to remain review-only until variational full-update is validated",
            "environment chi and iteration values are intentionally small to protect host RAM and GPU headroom",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON artifact path")
    args = parser.parse_args()
    try:
        import cupy as cp
    except ImportError as exc:  # pragma: no cover - exercised on non-CUDA machines
        raise SystemExit(f"CuPy is required for the GPU campaign: {exc}") from exc
    if not cp.cuda.is_available():
        raise SystemExit("CUDA is unavailable; refusing to silently run the GPU campaign on CPU")
    artifact = run_campaign(cp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "status": artifact["status"],
        "case_count": artifact["case_count"],
        "product_cases_passed": artifact["product_cases_passed"],
        "git_revision": artifact["git_revision"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
