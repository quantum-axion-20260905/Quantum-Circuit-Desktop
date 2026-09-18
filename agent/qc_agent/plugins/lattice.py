from __future__ import annotations

from typing import Iterable

from .models import CTMRGPayload, IPEPSInteraction, LatticeHamiltonianPayload, LatticeSpec, PauliTerm


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


def build_ctmrg_spin_payload(
    spec: LatticeHamiltonianPayload,
    *,
    initial_state: str = "up",
    environment_bond_dim: int = 16,
    iterations: int = 20,
    tolerance: float = 1e-8,
    gauge_validation: bool = False,
    gauge_validation_tolerance: float = 1e-4,
    gauge_preconditioner: str = "none",
    gauge_preconditioner_iterations: int = 4,
) -> CTMRGPayload:
    """Build a periodic 2D spin model for the bounded iPEPS CTMRG backend.

    The finite-lattice builder above emits explicit finite Pauli terms.  This
    builder instead emits translationally repeated +x/+y interactions for an
    infinite unit cell, keeping the domain mapping separate from CTMRG.
    Dimensions remain bounded by ``CTMRGPayload`` at construction time.
    """

    if len(spec.dimensions) != 2:
        raise ValueError("CTMRG spin models require a 2D unit-cell dimensions=[nx, ny]")
    nx, ny = (int(value) for value in spec.dimensions)
    cell_sites = nx * ny

    def site(x: int, y: int) -> int:
        return (x % nx) + nx * (y % ny)

    if spec.model == "ising":
        bond_components = [("Z", "Z", -float(spec.coupling), "ZZ")]
        field_pauli = "X"
    elif spec.model == "heisenberg":
        bond_components = [(pauli, pauli, float(spec.coupling), f"{pauli}{pauli}") for pauli in ("X", "Y", "Z")]
        field_pauli = "Z"
    else:
        bond_components = [
            ("X", "X", float(spec.coupling), "XX"),
            ("Y", "Y", float(spec.coupling), "YY"),
            ("Z", "Z", float(spec.coupling) * float(spec.anisotropy), "ZZ"),
        ]
        field_pauli = "Z"

    interactions: list[IPEPSInteraction] = []
    for y in range(ny):
        for x in range(nx):
            left_site = site(x, y)
            for right_site, displacement in (
                (site(x + 1, y), [1, 0]),
                (site(x, y + 1), [0, 1]),
            ):
                for left_pauli, right_pauli, coefficient, label in bond_components:
                    interactions.append(IPEPSInteraction(
                        left_site=left_site,
                        right_site=right_site,
                        displacement=displacement,
                        left_pauli=left_pauli,
                        right_pauli=right_pauli,
                        coefficient=coefficient,
                        label=label,
                    ))

    terms = [
        _term({site_index: field_pauli}, -float(spec.field), f"{spec.model} field")
        for site_index in range(cell_sites)
    ] if spec.field else []
    return CTMRGPayload(
        unit_cell=[nx, ny],
        terms=terms,
        interactions=interactions,
        initial_state=initial_state,
        environment_bond_dim=environment_bond_dim,
        iterations=iterations,
        tolerance=tolerance,
        gauge_validation=gauge_validation,
        gauge_validation_tolerance=gauge_validation_tolerance,
        gauge_preconditioner=gauge_preconditioner,
        gauge_preconditioner_iterations=gauge_preconditioner_iterations,
    )
