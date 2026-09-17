from __future__ import annotations

import itertools
import math
import time
from typing import Any

from ..backends import mps as mps_backend
from ..plugins.lattice import lattice_graph
from ..plugins.models import PEPSPayload
from .boundary_mps import checkpoint_path_for_operator, contract_boundary_mps
from .contracts import ConvergencePoint, ConvergenceReport, ResearchResult, TruncationReport
from .observables import statevector_expectation
from .mps_runtime import pauli_operator

try:
    import opt_einsum as oe
except Exception:  # pragma: no cover
    oe = None


def _split(theta: Any, xp: Any, bond_dim: int, cutoff: float) -> tuple[Any, Any, float, int]:
    left_size = theta.shape[0]
    right_size = theta.shape[1]
    matrix = theta.reshape(left_size, right_size)
    u, singular, vh = xp.linalg.svd(matrix, full_matrices=False)
    values = [float(abs(value) ** 2) for value in mps_backend.host(singular)]
    total = sum(values)
    keep = min(int(bond_dim), len(values))
    if cutoff > 0 and values:
        threshold = values[0] * cutoff * cutoff
        keep = min(keep, max(1, sum(value >= threshold for value in values)))
    discarded = sum(values[keep:]) / total if total > 0 else 0.0
    return u[:, :keep], singular[:keep, None] * vh[:keep, :], discarded, keep


class PEPSRuntime:
    """Open-boundary finite PEPS with local simple-update gates.

    Each virtual edge is explicit.  Contraction is deliberately bounded and
    uses a double-layer network for norms and local Pauli observables.  The
    physical statevector is retained only as a small-system compatibility
    helper; production paths never materialize it.
    """

    def __init__(self, xp: Any, payload: PEPSPayload, cancel_cb: Any = None):
        self.xp = xp
        self.payload = payload
        self.cancel_cb = cancel_cb
        self._boundary_resume_consumed = False
        self.graph = lattice_graph(payload.lattice)
        self.edges = [(edge["source"], edge["target"]) for edge in self.graph["edges"]]
        self.site_edges: list[list[int]] = [[] for _ in range(payload.n_qubits)]
        for edge_index, (left, right) in enumerate(self.edges):
            self.site_edges[left].append(edge_index)
            self.site_edges[right].append(edge_index)
        self.edge_axes: dict[tuple[int, int], int] = {}
        self.tensors: list[Any] = []
        dtype = xp.complex64 if payload.dtype == "complex64" else xp.complex128
        for site, edge_ids in enumerate(self.site_edges):
            shape = (2, *([1] * len(edge_ids)))
            tensor = xp.zeros(shape, dtype=dtype)
            tensor[(0, *([0] * len(edge_ids)))] = 1
            self.tensors.append(tensor)
            for local_axis, edge_id in enumerate(edge_ids, start=1):
                self.edge_axes[(site, edge_id)] = local_axis
        self.discarded_weight = 0.0
        self.boundary_summary: dict[str, Any] = {}

    def apply_one_site(self, site: int, unitary: Any) -> None:
        self.tensors[site] = self.xp.tensordot(unitary, self.tensors[site], axes=(1, 0))

    def _edge_id(self, left: int, right: int) -> int:
        target = frozenset((left, right))
        for edge_id, edge in enumerate(self.edges):
            if frozenset(edge) == target:
                return edge_id
        raise ValueError(f"PEPS requires a lattice edge between sites {left} and {right}")

    def apply_two_site(self, left: int, right: int, unitary: Any) -> None:
        edge_id = self._edge_id(left, right)
        tensor_left = self.tensors[left]
        tensor_right = self.tensors[right]
        axis_left = self.edge_axes[(left, edge_id)]
        axis_right = self.edge_axes[(right, edge_id)]
        left_axes = [axis for axis in range(tensor_left.ndim) if axis != axis_left]
        right_axes = [axis for axis in range(tensor_right.ndim) if axis != axis_right]
        theta = self.xp.tensordot(tensor_left, tensor_right, axes=(axis_left, axis_right))
        right_physical_position = len(left_axes)
        theta = theta.transpose(
            [0, right_physical_position]
            + list(range(1, right_physical_position))
            + list(range(right_physical_position + 1, theta.ndim))
        )
        physical_and_virtual = theta.reshape(4, -1)
        theta = (unitary @ physical_and_virtual).reshape((2, 2) + theta.shape[2:])
        left_virtual = theta.shape[2 : 2 + len(left_axes) - 1]
        right_virtual = theta.shape[2 + len(left_axes) - 1 :]
        left_size = 2 * math.prod(left_virtual or (1,))
        right_size = 2 * math.prod(right_virtual or (1,))
        matrix = theta.transpose(
            [0] + list(range(2, 2 + len(left_virtual))) + [1] + list(range(2 + len(left_virtual), theta.ndim))
        ).reshape(left_size, right_size)
        u, right_factor, discarded, keep = _split(matrix, self.xp, self.payload.bond_dim, self.payload.truncation_cutoff)
        left_temp = u.reshape((2, *left_virtual, keep))
        right_temp = right_factor.reshape((keep, 2, *right_virtual))

        # The left factor is stored as [physical, other bonds, new bond],
        # while the right factor is [new bond, physical, other bonds].
        left_order = [len(left_axes) if axis == axis_left else left_axes.index(axis) for axis in range(tensor_left.ndim)]
        right_order = [0 if axis == axis_right else 1 + right_axes.index(axis) for axis in range(tensor_right.ndim)]
        self.tensors[left] = left_temp.transpose(left_order)
        self.tensors[right] = right_temp.transpose(right_order)
        self.discarded_weight += discarded

    def apply_term(self, term: Any, angle: float) -> None:
        active = sorted((int(index), str(pauli).upper()) for index, pauli in term.paulis.items() if pauli != "I")
        if not active:
            return
        dtype = self.tensors[0].dtype
        if len(active) == 1:
            operator = pauli_operator(self.xp, dtype, active[0][1])
            unitary = math.cos(angle) * self.xp.eye(2, dtype=dtype) - 1j * math.sin(angle) * operator
            self.apply_one_site(active[0][0], unitary)
            return
        if len(active) != 2:
            raise ValueError("native PEPS currently supports one- and two-site Pauli terms")
        left, left_pauli = active[0]
        right, right_pauli = active[1]
        operator = self.xp.kron(pauli_operator(self.xp, dtype, left_pauli), pauli_operator(self.xp, dtype, right_pauli))
        unitary = math.cos(angle) * self.xp.eye(4, dtype=dtype) - 1j * math.sin(angle) * operator
        self.apply_two_site(left, right, unitary)

    def wavefunction(self) -> Any:
        if self.payload.contraction_method in ("auto", "opt_einsum") and oe is not None:
            return self._opt_einsum_wavefunction()
        if self.payload.contraction_method == "opt_einsum" and oe is None:
            raise RuntimeError("contraction_method=opt_einsum requires opt_einsum")
        return self._enumerated_wavefunction()

    def _opt_einsum_wavefunction(self) -> Any:
        # Keep physical labels open and give each virtual edge one shared
        # label. opt_einsum then chooses a contraction order instead of
        # enumerating every virtual-bond assignment in Python.
        arguments: list[Any] = []
        for site, tensor in enumerate(self.tensors):
            site_labels = [site]
            site_labels.extend(self.payload.n_qubits + edge_id for edge_id in self.site_edges[site])
            arguments.extend((tensor, site_labels))
        arguments.append(list(range(self.payload.n_qubits)))
        state_tensor = oe.contract(*arguments)
        return state_tensor.reshape((1 << self.payload.n_qubits,))

    def _enumerated_wavefunction(self) -> Any:
        if self.payload.n_qubits > 16:
            raise RuntimeError("PEPS enumeration is limited to 16 sites; use opt_einsum double-layer contraction")
        dtype = self.tensors[0].dtype
        state = self.xp.zeros((1 << self.payload.n_qubits,), dtype=dtype)
        ranges = [range(self.tensors[left].shape[self.edge_axes[(left, edge_id)]]) for edge_id, (left, _) in enumerate(self.edges)]
        for assignment in itertools.product(*ranges):
            vector = self.xp.asarray([1], dtype=dtype)
            for site, tensor in enumerate(self.tensors):
                indices = [slice(None)] + [assignment[edge_id] for edge_id in self.site_edges[site]]
                vector = self.xp.kron(vector, tensor[tuple(indices)])
            state += vector
        return state

    def _double_tensor(self, site: int) -> Any:
        """Fuse bra/ket virtual indices into a local double-layer tensor.

        The resulting axes are ``(bra_physical, ket_physical, edge_0, ...)``
        where every edge has dimension ``D_edge**2``.  Keeping the physical
        indices open until the local operator is applied avoids allocating a
        ``2**n`` statevector for 2D/3D workloads.
        """
        tensor = self.tensors[site]
        edge_count = len(self.site_edges[site])
        paired = self.xp.tensordot(self.xp.conj(tensor), tensor, axes=0)
        order = [0, 1 + edge_count]
        # Interleave bra/ket axes per edge before fusing them into D**2.
        # Keeping all bra axes followed by all ket axes would pair the wrong
        # virtual indices as soon as a site has more than one edge.
        for index in range(edge_count):
            order.extend((1 + index, 2 + edge_count + index))
        paired = paired.transpose(order)
        if not edge_count:
            return paired.reshape((2, 2))
        edge_dims = [tensor.shape[1 + index] ** 2 for index in range(edge_count)]
        return paired.reshape((2, 2, *edge_dims))

    def _local_double_tensor(self, site: int, pauli: str = "I") -> Any:
        tensor = self._double_tensor(site)
        operator = pauli_operator(self.xp, tensor.dtype, pauli)
        return self.xp.tensordot(operator, tensor, axes=([0, 1], [0, 1]))

    def _contract_double_layer(self, paulis: dict[int, str] | None = None) -> float:
        """Contract ``<psi|O|psi>`` without opening all physical indices."""
        if self.payload.contraction_method == "boundary-mps":
            resume_from = None
            if self.payload.boundary_resume_from and not self._boundary_resume_consumed:
                resume_from = self.payload.boundary_resume_from
            value, diagnostics = contract_boundary_mps(
                self,
                paulis,
                max_bond_dim=self.payload.boundary_bond_dim,
                cutoff=self.payload.truncation_cutoff,
                checkpoint_path=(
                    checkpoint_path_for_operator(self.payload.boundary_checkpoint_path, paulis)
                    if self.payload.boundary_checkpoint_path else None
                ),
                resume_from=resume_from,
                cancel_cb=self.cancel_cb,
            )
            if resume_from:
                self._boundary_resume_consumed = True
            self.boundary_summary = diagnostics
            return value
        if self.payload.contraction_method == "enumeration":
            state = self._enumerated_wavefunction()
            values = statevector_expectation(
                state,
                self.xp,
                [type("Term", (), {"paulis": paulis or {}})()],
                self.payload.n_qubits,
            )
            return float(values[0])
        if oe is None:
            if self.payload.contraction_method == "opt_einsum":
                raise RuntimeError("contraction_method=opt_einsum requires opt_einsum")
            # The fallback is intentionally bounded to the compatibility path;
            # large workloads must not silently allocate an exponential state.
            if self.payload.n_qubits > 16:
                raise RuntimeError("opt_einsum is required for PEPS contraction above 16 sites")
            state = self._enumerated_wavefunction()
            values = statevector_expectation(
                state,
                self.xp,
                [type("Term", (), {"paulis": paulis or {}})()],
                self.payload.n_qubits,
            )
            return float(values[0])

        arguments: list[Any] = []
        requested = paulis or {}
        for site in range(self.payload.n_qubits):
            tensor = self._local_double_tensor(site, requested.get(site, "I"))
            arguments.extend((tensor, list(self.site_edges[site])))
        arguments.append([])
        value = oe.contract(*arguments)
        return float(complex(mps_backend.host(value)).real)

    def expectation(self, terms: list[Any]) -> list[float]:
        values = [self._contract_double_layer(dict(term.paulis)) for term in terms]
        if self.payload.contraction_method == "boundary-mps" and values:
            self.boundary_summary = {
                **self.boundary_summary,
                "observable_count": len(values),
            }
        return values

    def norm2(self) -> float:
        return self._contract_double_layer()


def _energy(runtime: PEPSRuntime, payload: PEPSPayload) -> float:
    values = runtime.expectation(payload.terms)
    return float(sum(term.coefficient * value for term, value in zip(payload.terms, values)))


def _resource_estimate(runtime: PEPSRuntime) -> dict[str, Any]:
    dtype = runtime.tensors[0].dtype
    itemsize = int(getattr(dtype, "itemsize", 16))
    tensor_values = sum(int(tensor.size) for tensor in runtime.tensors)
    tensor_bytes = tensor_values * itemsize
    method = runtime.payload.contraction_method
    materializes_statevector = method == "enumeration" or (
        method in ("auto", "opt_einsum") and oe is None
    )
    return {
        "representation": "peps",
        "n_qubits": len(runtime.tensors),
        "tensor_values": tensor_values,
        "tensor_bytes": tensor_bytes,
        "peak_bytes_estimate": int(math.ceil(tensor_bytes * 2.5)),
        "dtype": str(dtype),
        "contraction_method": method,
        "materializes_statevector": materializes_statevector,
    }


def run_peps(
    xp: Any,
    payload: PEPSPayload,
    *,
    progress_cb: Any = None,
    cancel_cb: Any = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    runtime = PEPSRuntime(xp, payload, cancel_cb=cancel_cb)
    observables = payload.observables or payload.terms
    times = [0.0]
    values = [runtime.expectation(observables)]
    energies = [_energy(runtime, payload)]
    for step in range(payload.steps):
        if cancel_cb and cancel_cb():
            raise RuntimeError("job canceled")
        if payload.order == 1:
            schedule = [(term, payload.dt) for term in payload.terms]
        else:
            schedule = [(term, payload.dt / 2.0) for term in payload.terms]
            schedule.extend((term, payload.dt / 2.0) for term in reversed(payload.terms))
        for term, local_dt in schedule:
            if cancel_cb and cancel_cb():
                raise RuntimeError("job canceled")
            runtime.apply_term(term, local_dt * term.coefficient)
        times.append((step + 1) * payload.dt)
        values.append(runtime.expectation(observables))
        energies.append(_energy(runtime, payload))
        if progress_cb:
            progress_cb((step + 1) / max(1, payload.steps), "peps-step")
    runtime.xp.cuda.Stream.null.synchronize() if hasattr(runtime.xp, "cuda") else None
    warnings = [
        "native finite PEPS uses simple-update truncation and a bounded double-layer contraction",
        "increase bond_dim or compare against MPS/DMRG for convergence",
    ] + (["boundary-MPS contraction truncates the environment; increase boundary_bond_dim and compare convergence"] if payload.contraction_method == "boundary-mps" else [])
    boundary_diagnostics = runtime.boundary_summary if payload.contraction_method == "boundary-mps" else {}
    convergence_points = [
        ConvergencePoint(
            iteration=int(point["row"]),
            discarded_weight=float(point["discarded_weight"]),
            environment_dim=int(point["bond_dim_used"]),
        )
        for point in boundary_diagnostics.get("rows", [])
    ]
    research_result = ResearchResult(
        status="needs_review",
        method="finite-peps-simple-update",
        representation="peps",
        metrics={"norm2": runtime.norm2(), "final_energy": energies[-1]},
        truncation=TruncationReport(
            discarded_weight=float(runtime.discarded_weight + boundary_diagnostics.get("discarded_weight", 0.0)),
            cutoff=float(payload.truncation_cutoff),
            max_bond_dim=int(payload.bond_dim),
            max_environment_dim=(int(payload.boundary_bond_dim) if payload.contraction_method == "boundary-mps" else None),
        ),
        convergence=ConvergenceReport(
            converged=bool(boundary_diagnostics.get("converged", False)) if payload.contraction_method == "boundary-mps" else False,
            criterion="boundary discarded weight and environment bond dimension" if payload.contraction_method == "boundary-mps" else "simple-update PEPS requires an independent convergence study",
            points=convergence_points,
            warnings=list(warnings),
        ),
        resources=_resource_estimate(runtime),
        warnings=list(warnings),
        limitations=[
            "finite PEPS uses simple-update evolution and is not a full variational update",
            "results should be compared across physical bond dimension and, for boundary-MPS, environment bond dimension",
        ],
        details={"dimensions": list(payload.lattice.dimensions), "contraction_method": payload.contraction_method},
    ).to_dict()
    return {
        "status": "done",
        "backend": "tensor-network-peps-boundary-mps" if payload.contraction_method == "boundary-mps" else "tensor-network-peps-simple-update",
        "method": "finite-peps-simple-update",
        "native_geometry": True,
        "n_qubits": payload.n_qubits,
        "dimensions": payload.lattice.dimensions,
        "steps": payload.steps,
        "dt": payload.dt,
        "order": payload.order,
        "bond_dim_requested": payload.bond_dim,
        "bond_dim_used": max((max(tensor.shape[1:], default=1) for tensor in runtime.tensors), default=1),
        "virtual_bond_count": len(runtime.edges),
        "contraction_method": payload.contraction_method if payload.contraction_method != "auto" or oe is not None else "enumeration",
        "boundary_diagnostics": runtime.boundary_summary if payload.contraction_method == "boundary-mps" else None,
        "boundary_bond_dim_requested": (
            boundary_diagnostics.get("boundary_bond_dim_requested")
            if payload.contraction_method == "boundary-mps" else None
        ),
        "boundary_bond_dim_used": (
            boundary_diagnostics.get("boundary_bond_dim_used")
            if payload.contraction_method == "boundary-mps" else None
        ),
        "resource_estimate": _resource_estimate(runtime),
        "research_result": research_result,
        "discarded_weight": runtime.discarded_weight,
        "approximate": True,
        "norm2": runtime.norm2(),
        "times": times,
        "energies": energies,
        "expectations": [
            {"time": moment, "values": point, "energy": energy}
            for moment, point, energy in zip(times, values, energies)
        ],
        "warnings": warnings,
        "time_ms": round((time.perf_counter() - started) * 1000, 3),
    }
