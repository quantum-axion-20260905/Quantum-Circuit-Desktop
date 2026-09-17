from __future__ import annotations

from typing import Iterable

from .models import LatticeSpec, PauliTerm


def _coordinates(spec: LatticeSpec) -> list[tuple[int, ...]]:
    dims = spec.dimensions
    if len(dims) == 1:
        return [(x,) for x in range(dims[0])]
    if len(dims) == 2:
        nx, ny = dims
        return [
            (x, y)
            for y in range(ny)
            for x in (range(nx) if y % 2 == 0 else range(nx - 1, -1, -1))
        ]
    nx, ny, nz = dims
    out: list[tuple[int, ...]] = []
    for z in range(nz):
        ys: Iterable[int] = range(ny) if z % 2 == 0 else range(ny - 1, -1, -1)
        for y in ys:
            xs: Iterable[int] = range(nx) if (y + z) % 2 == 0 else range(nx - 1, -1, -1)
            out.extend((x, y, z) for x in xs)
    return out


def lattice_graph(spec: LatticeSpec) -> dict:
    coords = _coordinates(spec)
    index = {coord: site for site, coord in enumerate(coords)}
    edges: set[tuple[int, int]] = set()
    dims = spec.dimensions
    for coord in coords:
        for axis, size in enumerate(dims):
            neighbour = list(coord)
            neighbour[axis] += 1
            if neighbour[axis] >= size:
                if spec.boundary != "periodic":
                    continue
                neighbour[axis] = 0
            left, right = index[coord], index[tuple(neighbour)]
            if left == right:
                continue
            edges.add((min(left, right), max(left, right)))
    ordered_edges = sorted(edges)
    return {
        "dimensions": dims,
        "dimension": len(dims),
        "boundary": spec.boundary,
        "ordering": "snake",
        "sites": [{"index": i, "coordinate": list(coord)} for i, coord in enumerate(coords)],
        "edges": [{"source": left, "target": right} for left, right in ordered_edges],
    }


def _term(indexes: dict[int, str], coefficient: float, label: str) -> PauliTerm:
    return PauliTerm(paulis=indexes, coefficient=coefficient, label=label)


def build_spin_hamiltonian(spec: LatticeHamiltonianPayload) -> dict:
    graph = lattice_graph(spec)
    terms: list[PauliTerm] = []
    n = spec.n_sites
    if spec.model == "ising":
        for edge in graph["edges"]:
            terms.append(_term({edge["source"]: "Z", edge["target"]: "Z"}, -spec.coupling, "ZZ"))
        if spec.field:
            terms.extend(_term({site: "X"}, -spec.field, "X field") for site in range(n))
    elif spec.model == "heisenberg":
        for edge in graph["edges"]:
            for pauli in ("X", "Y", "Z"):
                terms.append(_term({edge["source"]: pauli, edge["target"]: pauli}, spec.coupling, f"{pauli}{pauli}"))
        if spec.field:
            terms.extend(_term({site: "Z"}, -spec.field, "Z field") for site in range(n))
    else:  # XXZ
        for edge in graph["edges"]:
            for pauli, factor in (("X", 1.0), ("Y", 1.0), ("Z", spec.anisotropy)):
                terms.append(_term({edge["source"]: pauli, edge["target"]: pauli}, spec.coupling * factor, f"{pauli}{pauli}"))
        if spec.field:
            terms.extend(_term({site: "Z"}, -spec.field, "Z field") for site in range(n))
    return {**graph, "model": spec.model, "coupling": spec.coupling, "field": spec.field, "anisotropy": spec.anisotropy, "n_qubits": n, "terms": [term.model_dump(mode="json") for term in terms]}
