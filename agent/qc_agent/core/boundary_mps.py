"""Finite 2D PEPS double-layer contraction with a boundary-MPS sweep.

The implementation keeps the PEPS virtual boundary as an MPS while rows are
absorbed.  It is deliberately explicit about the environment bond limit and
reports discarded singular-value weight; callers can therefore distinguish a
controlled approximation from the exact opt_einsum path.
"""

from __future__ import annotations

from typing import Any


def _host(value: Any) -> Any:
    try:
        return value.get()
    except AttributeError:
        return value


def _directions(runtime: Any) -> dict[tuple[int, str], int]:
    dimensions = runtime.payload.lattice.dimensions
    if len(dimensions) != 2 or runtime.payload.lattice.boundary != "open":
        raise ValueError("boundary-MPS currently supports open rectangular 2D lattices")
    sites = {
        int(site["index"]): tuple(int(value) for value in site["coordinate"])
        for site in runtime.graph["sites"]
    }
    result: dict[tuple[int, str], int] = {}
    for edge_id, (left, right) in enumerate(runtime.edges):
        x0, y0 = sites[left]
        x1, y1 = sites[right]
        if y0 == y1 and abs(x0 - x1) == 1:
            if x0 < x1:
                result[(left, "right")] = edge_id
                result[(right, "left")] = edge_id
            else:
                result[(right, "right")] = edge_id
                result[(left, "left")] = edge_id
        elif x0 == x1 and abs(y0 - y1) == 1:
            if y0 < y1:
                result[(left, "up")] = edge_id
                result[(right, "down")] = edge_id
            else:
                result[(right, "up")] = edge_id
                result[(left, "down")] = edge_id
        else:
            raise ValueError("boundary-MPS requires nearest-neighbor rectangular edges")
    return result


def _local_row_tensor(runtime: Any, site: int, paulis: dict[int, str], directions: dict[tuple[int, str], int]) -> Any:
    requested = str(paulis.get(site, "I")).upper()
    local = runtime._local_double_tensor(site, requested)
    axis_for_direction: dict[str, int] = {}
    for direction in ("left", "right", "down", "up"):
        edge_id = directions.get((site, direction))
        if edge_id is not None:
            axis_for_direction[direction] = runtime.edge_axes[(site, edge_id)] - 1
    present_axes = [axis_for_direction[direction] for direction in ("left", "right", "down", "up") if direction in axis_for_direction]
    if present_axes:
        local = local.transpose(present_axes)
    else:
        local = local.reshape(())
    dimensions = [
        int(runtime.tensors[site].shape[runtime.edge_axes[(site, directions[(site, direction)])]]) ** 2
        if (site, direction) in directions
        else 1
        for direction in ("left", "right", "down", "up")
    ]
    return local.reshape(tuple(dimensions))


def _row_mpo(runtime: Any, row: int, paulis: dict[int, str], directions: dict[tuple[int, str], int]) -> list[Any]:
    dimensions = runtime.payload.lattice.dimensions
    nx = int(dimensions[0])
    sites_by_coordinate = {
        tuple(int(value) for value in item["coordinate"]): int(item["index"])
        for item in runtime.graph["sites"]
    }
    tensors: list[Any] = []
    for x in range(nx):
        site = sites_by_coordinate[(x, row)]
        local = _local_row_tensor(runtime, site, paulis, directions)
        tensors.append(local)
    return tensors


def _apply_row_mpo(xp: Any, boundary: list[Any], row_mpo: list[Any]) -> list[Any]:
    if len(boundary) != len(row_mpo):
        raise ValueError("boundary-MPS width does not match the PEPS row width")
    output: list[Any] = []
    for state, operator in zip(boundary, row_mpo):
        if int(state.shape[1]) != int(operator.shape[2]):
            raise ValueError("boundary-MPS physical dimension does not match the PEPS vertical bond")
        # Keep the product bond ordering as (MPS bond, row-MPO bond) on both
        # sides.  This makes the next site's left product index contract the
        # MPS bond with MPS bond and the horizontal PEPS bond with horizontal
        # PEPS bond; swapping these two factors silently gives a wrong 2D
        # contraction while still producing shape-compatible tensors.
        product = xp.einsum("aib,xyio->axoby", state, operator)
        output.append(product.reshape(
            int(state.shape[0] * operator.shape[0]),
            int(operator.shape[3]),
            int(state.shape[2] * operator.shape[1]),
        ))
    return output


def _compress(xp: Any, tensors: list[Any], max_bond_dim: int, cutoff: float) -> tuple[list[Any], float, int]:
    compressed = list(tensors)
    discarded_total = 0.0
    max_used = 1
    for index in range(len(compressed) - 1):
        tensor = compressed[index]
        left_dim, physical_dim, right_dim = tensor.shape
        matrix = tensor.reshape(left_dim * physical_dim, right_dim)
        u, singular, vh = xp.linalg.svd(matrix, full_matrices=False)
        singular_host = _host(singular)
        weights = [float(abs(value) ** 2) for value in singular_host]
        total = sum(weights)
        keep = min(int(max_bond_dim), len(weights))
        if cutoff > 0 and weights:
            threshold = weights[0] * float(cutoff) ** 2
            keep = min(keep, max(1, sum(value >= threshold for value in weights)))
        discarded_total += sum(weights[keep:]) / total if total > 0 else 0.0
        compressed[index] = u[:, :keep].reshape(left_dim, physical_dim, keep)
        factor = singular[:keep, None] * vh[:keep, :]
        compressed[index + 1] = xp.tensordot(factor, compressed[index + 1], axes=(1, 0))
        max_used = max(max_used, int(keep))
    return compressed, discarded_total, max_used


def _sum_boundary(xp: Any, tensors: list[Any]) -> float:
    environment = xp.ones((1,), dtype=tensors[0].dtype)
    for tensor in tensors:
        if int(tensor.shape[1]) != 1:
            raise ValueError("final PEPS boundary still has open vertical indices")
        environment = xp.einsum("a,asb->b", environment, tensor)
    value = _host(environment[0])
    return float(complex(value).real)


def contract_boundary_mps(
    runtime: Any,
    paulis: dict[int, str] | None = None,
    *,
    max_bond_dim: int,
    cutoff: float = 0.0,
) -> tuple[float, dict[str, Any]]:
    """Contract one PEPS observable and return value plus sweep diagnostics."""
    if int(max_bond_dim) < 1:
        raise ValueError("boundary MPS bond dimension must be positive")
    directions = _directions(runtime)
    nx, ny = (int(value) for value in runtime.payload.lattice.dimensions)
    dtype = runtime.tensors[0].dtype
    boundary = [runtime.xp.ones((1, 1, 1), dtype=dtype) for _ in range(nx)]
    discarded_total = 0.0
    max_used = 1
    sweep: list[dict[str, Any]] = []
    requested = paulis or {}
    for row in range(ny):
        row_mpo = _row_mpo(runtime, row, requested, directions)
        boundary = _apply_row_mpo(runtime.xp, boundary, row_mpo)
        boundary, discarded, used = _compress(
            runtime.xp,
            boundary,
            int(max_bond_dim),
            float(cutoff),
        )
        discarded_total += discarded
        max_used = max(max_used, used)
        sweep.append({
            "row": row + 1,
            "bond_dim_used": used,
            "discarded_weight": discarded_total,
        })
    value = _sum_boundary(runtime.xp, boundary)
    return value, {
        "method": "boundary-mps",
        "boundary_bond_dim_requested": int(max_bond_dim),
        "boundary_bond_dim_used": int(max_used),
        "discarded_weight": float(discarded_total),
        "rows": sweep,
        "converged": discarded_total <= float(cutoff) if cutoff > 0 else discarded_total == 0.0,
    }
