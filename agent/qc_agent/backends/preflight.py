from __future__ import annotations

import math
from typing import Any

from ..models import PreflightPayload
from .limits import MAX_REFERENCE_QUBITS, MAX_STATEVECTOR_QUBITS, mps_memory_mb, statevector_memory_mb


def estimate_tebd(payload: Any, *, gpu_free_mb: float | None = None) -> dict[str, Any]:
    """Estimate a bounded-bond TEBD request before any CUDA allocation.

    Non-local lattice edges are more expensive for an MPS because the runtime
    inserts a swap network.  The estimate intentionally prices those swaps so
    a 2D/3D request cannot bypass the same resource guard used by circuit jobs.
    """
    n = int(payload.n_qubits)
    bond_dim = int(payload.bond_dim)
    terms = list(payload.terms)
    order_factor = 2 if int(payload.order) == 2 else 1
    interaction_layers = 0
    max_locality = int(getattr(payload, "max_term_locality", 64))
    for term in terms:
        active = sorted(int(index) for index in term.paulis)
        if len(active) > max_locality:
            interaction_layers += max_locality + 1
            continue
        if len(active) >= 2:
            # Each parity leg needs a forward and reverse CX. The gaps price
            # the swap networks used by the MPS runtime for non-adjacent legs.
            interaction_layers += 1 + (2 * (len(active) - 1))
            interaction_layers += sum(max(0, right - left - 1) for left, right in zip(active, active[1:]))
        else:
            interaction_layers += 1
    work = max(1, interaction_layers) * max(1, int(payload.steps)) * order_factor * max(1, bond_dim) ** 3
    peak_mb = mps_memory_mb(n, bond_dim, payload.dtype)
    estimated_ms = int(1 + work / 5000)
    warnings: list[str] = []
    if getattr(payload, "lattice", None) is not None and len(payload.lattice.dimensions) > 1:
        warnings.append("2D/3D lattice uses snake-ordered MPS; validate bond and Trotter convergence")
    if gpu_free_mb is not None and peak_mb > gpu_free_mb * 0.70:
        warnings.append(f"estimated TEBD memory {peak_mb:.1f} MB exceeds 70% of currently free GPU memory")
    if peak_mb > float(payload.max_mem_mb):
        warnings.append(f"estimated intermediate {peak_mb:.1f} MB exceeds memory budget")
    if estimated_ms > int(payload.max_time_ms):
        warnings.append(f"estimated TEBD time {estimated_ms} ms exceeds time budget")
    blocking_warnings = [warning for warning in warnings if "exceeds" in warning]
    feasible = not blocking_warnings
    return {
        "status": "ready" if feasible else "rejected",
        "feasible": feasible,
        "backend": "tensor-network",
        "n_qubits": n,
        "term_count": len(terms),
        "interaction_cost": interaction_layers,
        "estimated_peak_memory_mb": round(peak_mb, 3),
        "estimated_time_ms": estimated_ms,
        "bond_dim": bond_dim,
        "steps": int(payload.steps),
        "order": int(payload.order),
        "max_term_locality": max_locality,
        "estimate_method": "mps-tebd-swap-aware-bound",
        "blocking_warnings": blocking_warnings,
        "warnings": warnings,
    }


def estimate_ground_state(payload: Any, *, gpu_free_mb: float | None = None) -> dict[str, Any]:
    """Estimate the dense exact diagonalization reference path."""
    dimension = 1 << int(payload.n_qubits)
    bytes_per_value = 8 if payload.dtype == "complex64" else 16
    # Matrix plus eigensolver workspace and the returned eigenvector.
    peak_mb = (dimension * dimension * bytes_per_value * 2.5) / (1024 * 1024)
    warnings: list[str] = []
    if peak_mb > float(payload.max_mem_mb):
        warnings.append(f"estimated dense diagonalization memory {peak_mb:.1f} MB exceeds memory budget")
    if gpu_free_mb is not None and peak_mb > gpu_free_mb * 0.70:
        warnings.append(f"estimated dense diagonalization memory {peak_mb:.1f} MB exceeds 70% of currently free GPU memory")
    estimated_ms = int(1 + dimension**3 / 250000)
    if estimated_ms > int(payload.max_time_ms):
        warnings.append(f"estimated diagonalization time {estimated_ms} ms exceeds time budget")
    blocking_warnings = [warning for warning in warnings if "exceeds" in warning]
    return {
        "status": "ready" if not blocking_warnings else "rejected",
        "feasible": not blocking_warnings,
        "backend": "exact-diagonalization",
        "n_qubits": int(payload.n_qubits),
        "hamiltonian_dimension": dimension,
        "estimated_peak_memory_mb": round(peak_mb, 3),
        "estimated_time_ms": estimated_ms,
        "estimate_method": "dense-eigh-bound",
        "blocking_warnings": blocking_warnings,
        "warnings": warnings,
    }


def estimate_dmrg(payload: Any, *, gpu_free_mb: float | None = None) -> dict[str, Any]:
    """Estimate two-site DMRG local eigensolver memory before allocation."""
    n = int(payload.n_qubits)
    bond_dim = int(payload.bond_dim)
    local_dim = min(int(payload.max_local_dim), 4 * bond_dim * bond_dim)
    bytes_per_value = 8 if payload.dtype == "complex64" else 16
    solver = getattr(payload, "local_solver", "lanczos")
    if solver == "lanczos":
        # Matrix-free local matvec: Krylov basis plus sparse Pauli/environment
        # blocks, rather than a local_dim x local_dim eigensolver workspace.
        basis_values = local_dim * (int(getattr(payload, "lanczos_maxiter", 32)) + 4)
        environment_values = max(1, len(payload.terms)) * 2 * bond_dim * bond_dim
        mps_values = max(1, n) * 2 * bond_dim * bond_dim
        peak_mb = ((basis_values * 1.25 + environment_values * 1.5 + mps_values * 1.25) * bytes_per_value) / (1024 * 1024)
    else:
        mps_values = max(1, n) * 2 * bond_dim * bond_dim
        peak_mb = ((local_dim * local_dim * 2.5 + mps_values * 1.25) * bytes_per_value) / (1024 * 1024)
    work = max(1, int(payload.sweeps)) * max(1, n - 1) * max(1, len(payload.terms))
    solver_factor = 1.0 if solver == "lanczos" else max(1.0, local_dim / 32)
    estimated_ms = int(1 + work * local_dim * max(4, int(getattr(payload, "lanczos_maxiter", 32))) * solver_factor / 50_000)
    warnings: list[str] = []
    if 4 * bond_dim * bond_dim > int(payload.max_local_dim):
        warnings.append("requested bond_dim exceeds the configured two-site local dimension")
    if peak_mb > float(payload.max_mem_mb):
        warnings.append(f"estimated DMRG local eigensolver memory {peak_mb:.1f} MB exceeds memory budget")
    if gpu_free_mb is not None and peak_mb > gpu_free_mb * 0.70:
        warnings.append(f"estimated DMRG local eigensolver memory {peak_mb:.1f} MB exceeds 70% of currently free GPU memory")
    if estimated_ms > int(payload.max_time_ms):
        warnings.append(f"estimated DMRG time {estimated_ms} ms exceeds time budget")
    blocking_warnings = [warning for warning in warnings if "exceeds" in warning]
    return {
        "status": "ready" if not blocking_warnings else "rejected",
        "feasible": not blocking_warnings,
        "backend": "tensor-network",
        "method": f"dmrg-local-{solver}-bound",
        "n_qubits": n,
        "term_count": len(payload.terms),
        "sweeps": int(payload.sweeps),
        "bond_dim": bond_dim,
        "local_solver": solver,
        "lanczos_maxiter": int(getattr(payload, "lanczos_maxiter", 32)),
        "local_problem_dimension": local_dim,
        "estimated_peak_memory_mb": round(peak_mb, 3),
        "estimated_time_ms": estimated_ms,
        "blocking_warnings": blocking_warnings,
        "warnings": warnings,
    }


def estimate_peps(payload: Any, *, gpu_free_mb: float | None = None) -> dict[str, Any]:
    """Estimate native-PEPS double-layer contraction using a width bound.

    PEPS observables are contracted as a bra/ket network.  The relevant
    virtual dimension is therefore ``D**2``; estimating against ``2**n``
    would incorrectly reject useful 2D/3D jobs and would also hide the real
    contraction bottleneck.
    """
    from ..plugins.lattice import lattice_graph

    n = int(payload.n_qubits)
    edge_count = len(lattice_graph(payload.lattice)["edges"])
    bond_dim = int(payload.bond_dim)
    virtual_states = bond_dim ** edge_count
    double_layer_bond_dim = bond_dim ** 2
    dimensions = payload.lattice.dimensions
    if len(dimensions) == 2:
        boundary_exponent = max(1, 2 * min(dimensions) - 1)
    else:
        cross_section = min(dimensions[0] * dimensions[1], dimensions[0] * dimensions[2], dimensions[1] * dimensions[2])
        boundary_exponent = max(1, 2 * cross_section)
    boundary_states = double_layer_bond_dim ** boundary_exponent
    # Each site contributes one local double tensor.  This bound tracks the
    # width of the boundary contraction without pretending that the physical
    # Hilbert space is materialized.
    contraction_work = boundary_states * max(1, n)
    bytes_per_value = 8 if payload.dtype == "complex64" else 16
    boundary_mb = boundary_states * bytes_per_value / (1024 * 1024)
    max_degree = len(dimensions) * 2
    local_tensor_states = double_layer_bond_dim ** max_degree
    local_tensor_mb = n * local_tensor_states * bytes_per_value / (1024 * 1024)
    peak_mb = max(local_tensor_mb * 2.0, boundary_mb * 4.0)
    estimated_ms = int(1 + contraction_work / 20_000)
    warnings: list[str] = []
    method = getattr(payload, "contraction_method", "auto")
    if method == "boundary-mps" and len(dimensions) != 2:
        warnings.append("boundary-MPS unsupported for non-2D lattices; open rectangular 2D only")
    if method == "boundary-mps" and getattr(payload.lattice, "boundary", "open") != "open":
        warnings.append("boundary-MPS requires open lattice boundaries")
    if method == "boundary-mps":
        boundary_bond_dim = int(getattr(payload, "boundary_bond_dim", 32))
        boundary_width = int(dimensions[0]) if dimensions else 1
        boundary_peak_mb = (
            boundary_width * max(1, boundary_bond_dim) ** 2 * max(1, double_layer_bond_dim)
            * bytes_per_value * 2.5 / (1024 * 1024)
        )
        peak_mb = max(peak_mb, boundary_peak_mb)
        contraction_work = max(1, n) * max(1, boundary_bond_dim) ** 3 * max(1, double_layer_bond_dim)
        estimated_ms = int(1 + contraction_work / 20_000)
    if method == "enumeration" and n > 16:
        warnings.append("PEPS enumeration exceeds the 16-site statevector compatibility limit")
    if method == "enumeration" and virtual_states > int(payload.max_contraction_states):
        warnings.append("virtual-bond enumeration exceeds max_contraction_states")
    if method not in ("enumeration", "boundary-mps") and boundary_states > int(payload.max_contraction_states):
        warnings.append("double-layer boundary contraction exceeds max_contraction_states")
    if method == "boundary-mps" and int(getattr(payload, "boundary_bond_dim", 32)) > int(payload.max_contraction_states):
        warnings.append("boundary-MPS bond dimension exceeds max_contraction_states")
    if peak_mb > float(payload.max_mem_mb):
        warnings.append(f"estimated PEPS contraction memory {peak_mb:.1f} MB exceeds memory budget")
    if gpu_free_mb is not None and peak_mb > gpu_free_mb * 0.70:
        warnings.append(f"estimated PEPS contraction memory {peak_mb:.1f} MB exceeds 70% of currently free GPU memory")
    if estimated_ms > int(payload.max_time_ms):
        warnings.append(f"estimated PEPS contraction time {estimated_ms} ms exceeds time budget")
    blocking_warnings = [
        warning for warning in warnings
        if "exceeds" in warning or "unsupported" in warning or "requires open" in warning
    ]
    return {
        "status": "ready" if not blocking_warnings else "rejected",
        "feasible": not blocking_warnings,
        "backend": "tensor-network",
        "method": f"peps-{method}-bound",
        "contraction_method": method,
        "n_qubits": n,
        "edge_count": edge_count,
        "bond_dim": int(payload.bond_dim),
        "virtual_bond_states": virtual_states,
        "boundary_bond_exponent": boundary_exponent,
        "boundary_bond_states": boundary_states,
        "double_layer_bond_dim": double_layer_bond_dim,
        "boundary_bond_dim": int(getattr(payload, "boundary_bond_dim", 32)),
        "local_double_tensor_states": local_tensor_states,
        "materializes_statevector": False,
        "estimated_contraction_work": contraction_work,
        "estimated_peak_memory_mb": round(peak_mb, 3),
        "estimated_time_ms": estimated_ms,
        "blocking_warnings": blocking_warnings,
        "warnings": warnings,
    }


def estimate_ctmrg(payload: Any, *, gpu_free_mb: float | None = None) -> dict[str, Any]:
    """Estimate bounded iPEPS/CTMRG environment cost before CUDA allocation.

    CTMRG cost is controlled by the double-layer virtual dimension and the
    environment bond dimension, not by a dense ``2**N`` statevector.  The
    estimate intentionally prices both the unit-cell tensors and enlarged
    corner/edge environments so a future solver cannot bypass admission by
    only reporting the physical bond dimension.
    """
    cell_sites = math.prod(payload.unit_cell)
    physical_bond_dim = int(payload.physical_bond_dim)
    virtual_bond_dim = int(getattr(payload, "virtual_bond_dim", 1))
    environment_bond_dim = int(payload.environment_bond_dim)
    double_layer_dim = virtual_bond_dim ** 2
    bytes_per_value = 8 if payload.dtype == "complex64" else 16
    tensor_values = cell_sites * physical_bond_dim * max(1, virtual_bond_dim ** 4)
    double_layer_values = cell_sites * max(1, double_layer_dim ** 4)
    finite_reference_state_values = 16 if getattr(payload, "full_update_optimizer", None) == "finite-torus-gradient" else 0
    corner_values = 4 * environment_bond_dim * environment_bond_dim
    edge_values = 4 * environment_bond_dim * double_layer_dim * environment_bond_dim
    iteration_values = tensor_values + double_layer_values + corner_values + edge_values
    iteration_values += finite_reference_state_values
    peak_mb = iteration_values * bytes_per_value * 3.0 / (1024 * 1024)
    work = max(1, int(payload.iterations)) * max(1, cell_sites) * max(1, environment_bond_dim) ** 3 * max(1, double_layer_dim)
    if getattr(payload, "optimization", "none") != "none":
        work += max(1, int(getattr(payload, "optimization_steps", 1))) * max(1, cell_sites) * 16
    estimated_full_update_evaluations = None
    if getattr(payload, "optimization", "none") == "full-update":
        complex_parameters = cell_sites * physical_bond_dim * max(1, virtual_bond_dim ** 4) * 2
        max_parameters = int(getattr(payload, "full_update_max_parameters", 32))
        optimizer = getattr(payload, "full_update_optimizer", "coordinate")
        if optimizer == "spsa-gradient":
            # Two simultaneous perturbations per direction plus up to four
            # bounded line-search candidates, independent of tensor parameter
            # count.
            evaluations_per_parameter = 0
            directions = int(getattr(payload, "full_update_spsa_directions", 4))
            estimated_full_update_evaluations = 1 + int(getattr(payload, "optimization_steps", 1)) * (2 * directions + 4)
        elif optimizer == "finite-torus-gradient":
            # The analytic finite-reference objective shares one gradient
            # evaluation with each accepted step and tries up to four bounded
            # line-search candidates.  Its four-site reference vector is an
            # explicit, separately admitted validation mode.
            evaluations_per_parameter = 0
            estimated_full_update_evaluations = 1 + int(getattr(payload, "optimization_steps", 1)) * 5
        elif optimizer == "autodiff-ctmrg-gradient":
            # One initial objective, one gradient objective per update, and up
            # to four bounded line-search candidates.  This is deliberately
            # conservative because the optional Torch backend must obey the
            # same admission budget as the CuPy optimizers.
            evaluations_per_parameter = 0
            estimated_full_update_evaluations = 1 + int(getattr(payload, "optimization_steps", 1)) * 5
        else:
            evaluations_per_parameter = 4 if optimizer == "finite-difference-gradient" else 4
            estimated_full_update_evaluations = 1 + int(getattr(payload, "optimization_steps", 1)) * min(complex_parameters, max_parameters) * evaluations_per_parameter
            estimated_full_update_evaluations += int(getattr(payload, "optimization_steps", 1)) * (4 if optimizer == "finite-difference-gradient" else 0)
        work *= max(1, min(estimated_full_update_evaluations, int(getattr(payload, "full_update_max_evaluations", 512))))
    estimated_ms = int(1 + work / 25_000)
    warnings: list[str] = [
        "CTMRG is an experimental infinite-2D path; compare environment-dimension convergence",
    ]
    if list(payload.unit_cell) not in ([1, 1], [2, 1], [1, 2], [2, 2]):
        warnings.append("current CTMRG solver supports unit_cell dimensions no larger than 2x2")
    if physical_bond_dim != 2:
        warnings.append("current CTMRG solver supports only physical_bond_dim=2 for Pauli observables")
    if virtual_bond_dim > 1 and payload.dtype == "complex64":
        warnings.append("complex64 entangled iPEPS runs may lose transfer-sector precision; complex128 is recommended for reference-quality observables")
    if getattr(payload, "optimization", "none") == "product-coordinate-descent" and virtual_bond_dim != 1:
        warnings.append("product-coordinate-descent optimization requires virtual_bond_dim=1")
    if getattr(payload, "optimization", "none") == "full-update":
        complex_parameters = cell_sites * physical_bond_dim * max(1, virtual_bond_dim ** 4) * 2
        if complex_parameters > int(getattr(payload, "full_update_max_parameters", 32)):
            warnings.append(
                f"full-update tensor parameter count {complex_parameters} exceeds full_update_max_parameters={payload.full_update_max_parameters}"
            )
        if getattr(payload, "full_update_optimizer", "coordinate") == "finite-torus-gradient":
            warnings.append("finite-torus-gradient uses an explicitly bounded 4-site finite reference statevector; it is not an infinite-lattice CTMRG objective")
            if virtual_bond_dim > 2:
                warnings.append("finite-torus-gradient requires virtual_bond_dim<=2")
            if any(len(term.paulis) > 1 for term in payload.terms):
                warnings.append("finite-torus-gradient supports one-site terms only")
            for interaction in payload.interactions:
                if tuple(map(abs, interaction.displacement)) not in ((1, 0), (0, 1)):
                    warnings.append("finite-torus-gradient supports nearest-neighbor interactions only")
                    break
                cell_width, cell_height = (int(value) for value in payload.unit_cell)
                left_x = int(interaction.left_site) % cell_width
                left_y = int(interaction.left_site) // cell_width
                dx, dy = (int(value) for value in interaction.displacement)
                expected_right = (left_x + dx) % cell_width + cell_width * ((left_y + dy) % cell_height)
                if int(interaction.right_site) != expected_right:
                    warnings.append("finite-torus-gradient requires right_site to match left_site plus displacement")
                    break
        elif getattr(payload, "full_update_optimizer", "coordinate") == "autodiff-ctmrg-gradient":
            warnings.append("autodiff-ctmrg-gradient requires the optional PyTorch CUDA runtime")
            warnings.append("autodiff-ctmrg-gradient differentiates a bounded unrolled CTMRG environment; it is not yet an implicit fixed-point variational proof")
        if estimated_full_update_evaluations is not None and estimated_full_update_evaluations > int(getattr(payload, "full_update_max_evaluations", 512)):
            warnings.append(
                f"full-update estimated evaluations {estimated_full_update_evaluations} exceed full_update_max_evaluations={payload.full_update_max_evaluations}"
            )
    if peak_mb > float(payload.max_mem_mb):
        warnings.append(f"estimated CTMRG environment memory {peak_mb:.1f} MB exceeds memory budget")
    if gpu_free_mb is not None and peak_mb > gpu_free_mb * 0.70:
        warnings.append(f"estimated CTMRG environment memory {peak_mb:.1f} MB exceeds 70% of currently free GPU memory")
    if estimated_ms > int(payload.max_time_ms):
        warnings.append(f"estimated CTMRG time {estimated_ms} ms exceeds time budget")
    blocking_warnings = [warning for warning in warnings if "exceeds" in warning or "current CTMRG solver supports" in warning or "product-coordinate-descent optimization requires" in warning or "full-update tensor parameter count" in warning or "full-update estimated evaluations" in warning or "finite-torus-gradient requires" in warning or "finite-torus-gradient supports" in warning]
    return {
        "status": "ready" if not blocking_warnings else "rejected",
        "feasible": not blocking_warnings,
        "backend": "tensor-network",
        "method": "ctmrg-environment-bound",
        "representation": "ipeps",
        "unit_cell": list(payload.unit_cell),
        "unit_cell_sites": cell_sites,
        "physical_bond_dim": physical_bond_dim,
        "virtual_bond_dim": virtual_bond_dim,
        "double_layer_dim": double_layer_dim,
        "environment_bond_dim": environment_bond_dim,
        "iterations": int(payload.iterations),
        "tensor_values": tensor_values,
        "double_layer_values": double_layer_values,
        "finite_reference_state_values": finite_reference_state_values,
        "corner_values": corner_values,
        "edge_values": edge_values,
        "materializes_statevector": False,
        "optimization_materializes_reference_statevector": getattr(payload, "full_update_optimizer", None) == "finite-torus-gradient",
        "estimated_peak_memory_mb": round(peak_mb, 3),
        "estimated_time_ms": estimated_ms,
        "full_update_optimizer": getattr(payload, "full_update_optimizer", None),
        "estimated_full_update_evaluations": estimated_full_update_evaluations,
        "full_update_max_evaluations": getattr(payload, "full_update_max_evaluations", None),
        "blocking_warnings": blocking_warnings,
        "warnings": warnings,
    }


def estimate(
    payload: PreflightPayload,
    cotengra_available: bool = False,
    *,
    backend_name: str = "tensor-network",
    gpu_free_mb: float | None = None,
    path_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n = payload.n_qubits
    warnings: list[str] = []
    noise = getattr(payload, "noise", None)
    noise_active = bool(noise is not None and noise.active)
    if noise_active and getattr(payload, "result_type", "samples") == "selected_amplitudes":
        return {
            "feasible": False,
            "status": "rejected",
            "warnings": ["noise channels require result_type=samples; amplitudes are noiseless-only"],
        }
    if noise_active:
        warnings.append("noise is estimated with independent shot-based trajectories")
    for g in payload.gates:
        if g.target >= n or (g.control is not None and g.control >= n):
            return {"feasible": False, "status": "rejected", "warnings": ["gate index exceeds qubit count"]}
        if g.control is not None and g.control == g.target:
            return {"feasible": False, "status": "rejected", "warnings": ["control and target must differ"]}
    two_q = sum(1 for g in payload.gates if g.name in ("cx", "cz"))
    # MPS is bounded by bond dimension; exact contraction can use path stats.
    path_cost: float = float(max(1, len(payload.gates)) * max(1, 2 ** min(two_q, 20)))
    largest = 2 ** n
    peak_mb = statevector_memory_mb(n, payload.dtype)
    statevector_upper_bound_mb = peak_mb
    estimate_method = "conservative-statevector-upper-bound"
    tn_method = getattr(payload, "tn_method", "mps")
    bond_dim = int(getattr(payload, "bond_dim", 64))

    if backend_name == "tensor-network" and tn_method == "mps":
        peak_mb = mps_memory_mb(n, bond_dim, payload.dtype)
        largest = max(1, 2 * bond_dim * bond_dim)
        gate_work = max(1, len(payload.gates)) * max(1, bond_dim) ** 3
        if getattr(payload, "result_type", "selected_amplitudes") == "samples":
            shots = int(getattr(payload, "shots", 1024))
            sampling_work = max(1, n) * max(1, bond_dim) ** 2 * shots
            path_cost = float(gate_work * (shots if noise_active else 1) + sampling_work)
        else:
            path_cost = float(gate_work)
        estimate_method = "mps-bond-dimension-bound"
    elif backend_name == "tensor-network" and path_metrics:
        if path_metrics.get("largest_intermediate") is not None:
            largest = max(1, int(path_metrics["largest_intermediate"]))
            peak_mb = largest * (8 if payload.dtype == "complex64" else 16) / (1024 * 1024)
            estimate_method = "opt_einsum-contract-path"
        if path_metrics.get("path_cost") is not None:
            path_cost = float(path_metrics["path_cost"])

    if noise_active and backend_name in ("statevector", "reference"):
        # Noisy execution is a separate pure-state trajectory per shot.  The
        # memory peak stays per-trajectory, but the work scales with shots.
        path_cost *= int(getattr(payload, "shots", 1024))

    if backend_name in ("reference", "statevector"):
        hard_limit = MAX_REFERENCE_QUBITS if backend_name == "reference" else MAX_STATEVECTOR_QUBITS
        if n > hard_limit:
            warnings.append(f"{backend_name} supports at most {hard_limit} qubits")
        estimate_method = "statevector-allocation"

    optimize = getattr(payload, "optimize", "auto")
    if backend_name == "tensor-network" and optimize == "cotengra" and not cotengra_available:
        warnings.append("cotengra unavailable; optimizer will fall back to auto")

    max_mem_mb = float(getattr(payload, "max_mem_mb", 4096))
    max_time_ms = int(getattr(payload, "max_time_ms", 120000))
    if peak_mb > max_mem_mb:
        warnings.append(f"estimated intermediate {peak_mb:.1f} MB exceeds memory budget")
    if backend_name in ("statevector", "tensor-network") and gpu_free_mb is not None and peak_mb > gpu_free_mb * 0.70:
        warnings.append(f"estimated {backend_name} memory {peak_mb:.1f} MB exceeds 70% of currently free GPU memory")
    if two_q > 18: warnings.append("high two-qubit interaction count; contraction may be expensive")
    estimated_ms = int(1 + path_cost / (4 if backend_name == "tensor-network" and tn_method == "mps" else 5000))
    feasible = (
        not any("supports at most" in warning for warning in warnings)
        and peak_mb <= max_mem_mb
        and estimated_ms <= max_time_ms
        and not any("exceeds 70%" in warning for warning in warnings)
    )
    return {
        "status": "ready" if feasible else "rejected",
        "feasible": feasible,
        "backend": backend_name,
        "n_qubits": n,
        "path_steps": len(payload.gates),
        "path_cost": path_cost,
        "largest_intermediate": largest,
        "estimated_peak_memory_mb": round(peak_mb, 3),
        "statevector_upper_bound_mb": round(statevector_upper_bound_mb, 3),
        "estimated_time_ms": estimated_ms,
        "dtype": payload.dtype,
        "optimize": optimize,
        "tn_method": tn_method,
        "bond_dim": bond_dim,
        "estimate_method": estimate_method,
        "warnings": warnings,
    }
