from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from typing import Any

from ..backends import mps as mps_backend
from ..backends.limits import MAX_EXACT_DIAGONALIZATION_QUBITS
from ..models import TNGate, TNPayload
from ..plugins.models import DMRGPayload, GroundStatePayload
from .contracts import CheckpointManifest, ConvergencePoint, ConvergenceReport, ResearchResult
from .ground_state import exact_ground_state
from .mps_runtime import MPSRuntime, pauli_operator
from .observables import mps_expectation_from_tensors, structured_observables


def _left_environment(xp: Any, tensors: list[Any], site: int, term: Any) -> Any:
    dtype = tensors[0].dtype
    environment = xp.ones((1, 1), dtype=dtype)
    for qubit in range(site):
        tensor = tensors[qubit]
        operator = pauli_operator(xp, dtype, term.paulis.get(qubit, "I"))
        environment = xp.einsum(
            "ab,api,bqj,qp->ij", environment, tensor, tensor.conj(), operator
        )
    return environment


def _right_environment(xp: Any, tensors: list[Any], site: int, term: Any) -> Any:
    dtype = tensors[0].dtype
    environment = xp.ones((1, 1), dtype=dtype)
    for qubit in range(len(tensors) - 1, site - 1, -1):
        tensor = tensors[qubit]
        operator = pauli_operator(xp, dtype, term.paulis.get(qubit, "I"))
        environment = xp.einsum(
            "api,bqj,qp,ij->ab", tensor, tensor.conj(), operator, environment
        )
    return environment


def _effective_terms(xp: Any, tensors: list[Any], site: int, terms: list[Any]) -> list[tuple[float, Any, Any, Any, Any]]:
    """Compile one local effective-operator term per Pauli term.

    The environments are computed once per local optimization. Lanczos can
    then apply this list directly to a vector without materialising a dense
    ``(2*chi_left*chi_right)^2`` matrix.
    """
    compiled: list[tuple[float, Any, Any, Any, Any]] = []
    for term in terms:
        left = _left_environment(xp, tensors, site, term)
        right = _right_environment(xp, tensors, site + 2, term)
        dtype = tensors[0].dtype
        left_operator = pauli_operator(xp, dtype, term.paulis.get(site, "I"))
        right_operator = pauli_operator(xp, dtype, term.paulis.get(site + 1, "I"))
        compiled.append((float(term.coefficient), left, left_operator, right_operator, right))
    return compiled


def _effective_hamiltonian(
    xp: Any,
    tensors: list[Any],
    site: int,
    terms: list[Any],
    compiled: list[tuple[float, Any, Any, Any, Any]] | None = None,
) -> Any:
    left_dim = tensors[site].shape[0]
    right_dim = tensors[site + 1].shape[2]
    dtype = tensors[0].dtype
    dimension = left_dim * 4 * right_dim
    result = xp.zeros((dimension, dimension), dtype=dtype)
    compiled = compiled or _effective_terms(xp, tensors, site, terms)
    for coefficient, left, left_operator, right_operator, right in compiled:
        block = xp.einsum(
            "ab,qp,uv,cd->bqudapvc",
            left,
            left_operator,
            right_operator,
            right,
        )
        result += coefficient * block.reshape(dimension, dimension)
    # Small round-off asymmetries can otherwise make eigh return a complex
    # eigenvalue ordering that differs between NumPy and CuPy.
    return (result + result.conj().T) / 2


def _lanczos_ground_state(
    xp: Any,
    matvec: Any,
    dimension: int,
    dtype: Any,
    *,
    maxiter: int,
    tolerance: float,
) -> tuple[Any, Any, int, float]:
    """Return the lowest Ritz pair without a dense eigensolver workspace.

    The Pauli/environment operator is applied directly to each Krylov vector.
    Lanczos therefore avoids materialising the local dense matrix and its full
    eigensolver workspace; ``local_solver=dense`` remains available for tiny
    exact validation cases.
    """
    iterations = min(max(4, int(maxiter)), dimension)
    # A deterministic dense seed avoids the diagonal-Hamiltonian failure mode
    # of a basis-vector start (the Krylov space would then contain one basis
    # state only). The nonuniform ramp also has overlap with antisymmetric
    # ground states such as the two-spin Heisenberg singlet.
    vector = xp.arange(1, dimension + 1, dtype=xp.float64).astype(dtype)
    vector = vector / xp.linalg.norm(vector)
    previous = xp.zeros_like(vector)
    basis: list[Any] = []
    alpha: list[Any] = []
    beta: list[Any] = []
    krylov_residual = float("inf")

    for index in range(iterations):
        projected = matvec(vector)
        if index:
            projected = projected - beta[-1] * previous
        diagonal = xp.vdot(vector, projected).real
        projected = projected - diagonal * vector
        # Full re-orthogonalisation is inexpensive at the bounded local sizes
        # and prevents loss of orthogonality on nearly degenerate spectra.
        for old in basis:
            projected = projected - xp.vdot(old, projected) * old
        norm = xp.linalg.norm(projected)
        basis.append(vector)
        alpha.append(diagonal)
        residual = float(complex(mps_backend.host(norm)).real)
        krylov_residual = residual
        if residual <= tolerance or index + 1 >= iterations:
            break
        beta.append(norm)
        previous = vector
        vector = projected / norm

    small = xp.zeros((len(alpha), len(alpha)), dtype=dtype)
    for index, value in enumerate(alpha):
        small[index, index] = value
        if index < len(beta):
            small[index, index + 1] = beta[index]
            small[index + 1, index] = beta[index]
    eigenvalues, eigenvectors = xp.linalg.eigh(small)
    coefficients = eigenvectors[:, 0]
    ground = xp.zeros((dimension,), dtype=dtype)
    for coefficient, old in zip(coefficients, basis):
        ground = ground + coefficient * old
    ground = ground / xp.linalg.norm(ground)
    # Report the residual of the returned Ritz pair itself.  The norm used to
    # decide whether another Krylov vector is needed is not the same quantity
    # and can remain large even when the lowest Ritz vector is accurate.
    pair_residual = xp.linalg.norm(matvec(ground) - eigenvalues[0] * ground)
    residual_value = float(complex(mps_backend.host(pair_residual)).real)
    if not basis:
        residual_value = krylov_residual
    return eigenvalues[0], ground, len(alpha), residual_value


def _split_two_site(xp: Any, theta: Any, bond_dim: int, cutoff: float) -> tuple[Any, Any, float, int]:
    left_dim, _, _, right_dim = theta.shape
    matrix = theta.reshape(left_dim * 2, 2 * right_dim)
    u, singular, vh = xp.linalg.svd(matrix, full_matrices=False)
    singular_host = mps_backend.host(singular)
    weights = [float(abs(value) ** 2) for value in singular_host]
    total = sum(weights)
    keep = min(int(bond_dim), len(weights))
    if cutoff > 0 and weights:
        threshold = weights[0] * cutoff * cutoff
        keep = min(keep, max(1, sum(value >= threshold for value in weights)))
    discarded = sum(weights[keep:]) / total if total > 0 else 0.0
    left = u[:, :keep].reshape(left_dim, 2, keep)
    right = (singular[:keep, None] * vh[:keep, :]).reshape(keep, 2, right_dim)
    return left, right, discarded, keep


def _energy(xp: Any, tensors: list[Any], terms: list[Any]) -> float:
    values = mps_expectation_from_tensors(xp, tensors, terms)
    return float(sum(term.coefficient * value for term, value in zip(terms, values)))


def _dmrg_problem_sha256(payload: DMRGPayload) -> str:
    """Fingerprint the resumable problem, excluding run-control settings."""

    data = payload.model_dump(
        mode="json",
        exclude={
            "sweeps",
            "tolerance",
            "residual_tolerance",
            "variance_tolerance",
            "max_time_ms",
            "max_mem_mb",
            "checkpoint_path",
            "resume_from",
        },
    )
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _automatic_exact_cross_check(xp: Any, payload: DMRGPayload, energy: float) -> dict[str, Any]:
    """Run a bounded exact-energy check when the dense reference is genuinely small."""

    # Keep this stricter than the standalone 12-qubit exact endpoint.  DMRG is
    # a production path, so its automatic validation must remain cheap and
    # must not unexpectedly consume the user's entire memory budget.
    max_automatic_qubits = min(8, MAX_EXACT_DIAGONALIZATION_QUBITS)
    if payload.n_qubits > max_automatic_qubits:
        return {
            "performed": False,
            "reason": f"automatic exact cross-check is limited to {max_automatic_qubits} qubits",
        }
    dimension = 1 << payload.n_qubits
    dtype_bytes = 8 if payload.dtype == "complex64" else 16
    matrix_bytes = dimension * dimension * dtype_bytes
    conservative_workspace_mb = (8 * matrix_bytes) / (1024 * 1024)
    if conservative_workspace_mb > float(payload.max_mem_mb) * 0.5:
        return {
            "performed": False,
            "reason": "automatic exact cross-check was skipped by the declared memory budget",
            "estimated_workspace_mb": conservative_workspace_mb,
        }
    reference = exact_ground_state(
        xp,
        GroundStatePayload(
            n_qubits=payload.n_qubits,
            terms=payload.terms,
            dtype=payload.dtype,
            max_mem_mb=payload.max_mem_mb,
            max_time_ms=payload.max_time_ms,
        ),
    )
    absolute_error = abs(float(energy) - float(reference["ground_energy"]))
    tolerance = 1e-4 if payload.dtype == "complex64" else 1e-8
    return {
        "performed": True,
        "backend": reference["backend"],
        "ground_energy": float(reference["ground_energy"]),
        "dmrg_energy": float(energy),
        "absolute_error": absolute_error,
        "tolerance": tolerance,
        "passed": absolute_error <= tolerance,
        "hamiltonian_dimension": reference["hamiltonian_dimension"],
    }


def run_dmrg(
    xp: Any,
    payload: DMRGPayload,
    *,
    progress_cb: Any = None,
    cancel_cb: Any = None,
) -> dict[str, Any]:
    """Run a finite-size two-site DMRG sweep with a sparse Pauli Hamiltonian."""
    started = time.perf_counter()
    gates = [TNGate(name="x", target=index) for index, bit in enumerate(payload.initial_bitstring) if bit == "1"]
    base = TNPayload(
        n_qubits=payload.n_qubits,
        gates=gates,
        dtype=payload.dtype,
        bond_dim=payload.bond_dim,
        truncation_cutoff=payload.truncation_cutoff,
    )
    runtime = MPSRuntime(xp, base)
    problem_sha256 = _dmrg_problem_sha256(payload)
    history: list[dict[str, Any]] = []
    previous_energy: float | None = None
    converged = False
    solver_iterations_total = 0
    solver_residual_max = 0.0
    current_sweep_solver_residual_max = 0.0
    checkpoint_info: dict[str, Any] = {}

    def optimize(site: int, left_to_right: bool) -> None:
        nonlocal solver_iterations_total, solver_residual_max, current_sweep_solver_residual_max
        if cancel_cb and cancel_cb():
            raise RuntimeError("job canceled")
        compiled = _effective_terms(xp, runtime.tensors, site, payload.terms)
        left_dim = runtime.tensors[site].shape[0]
        right_dim = runtime.tensors[site + 1].shape[2]
        local_dimension = int(left_dim * 4 * right_dim)

        def matvec(vector: Any) -> Any:
            local = vector.reshape(left_dim, 2, 2, right_dim)
            output = xp.zeros_like(local)
            for coefficient, left, left_operator, right_operator, right in compiled:
                output += coefficient * xp.einsum(
                    "ab,qp,uv,cd,apvc->bqud",
                    left,
                    left_operator,
                    right_operator,
                    right,
                    local,
                )
            return output.reshape(local_dimension)

        if payload.local_solver == "dense":
            hamiltonian = _effective_hamiltonian(xp, runtime.tensors, site, payload.terms, compiled)
            eigenvalues, eigenvectors = xp.linalg.eigh(hamiltonian)
            eigenvalue = eigenvalues[0]
            local_vector = eigenvectors[:, 0]
            solver_iterations = int(hamiltonian.shape[0])
            solver_residual = 0.0
        else:
            eigenvalue, local_vector, solver_iterations, solver_residual = _lanczos_ground_state(
                xp,
                matvec,
                local_dimension,
                runtime.tensors[0].dtype,
                maxiter=payload.lanczos_maxiter,
                tolerance=payload.lanczos_tolerance,
            )
        solver_iterations_total += solver_iterations
        solver_residual_max = max(solver_residual_max, solver_residual)
        current_sweep_solver_residual_max = max(current_sweep_solver_residual_max, solver_residual)
        theta = local_vector.reshape(
            runtime.tensors[site].shape[0], 2, 2, runtime.tensors[site + 1].shape[2]
        )
        left, right, discarded, used = _split_two_site(
            xp, theta, payload.bond_dim, payload.truncation_cutoff
        )
        if left_to_right:
            runtime.tensors[site] = left
            runtime.tensors[site + 1] = right
        else:
            matrix = theta.reshape(runtime.tensors[site].shape[0] * 2, 2 * runtime.tensors[site + 1].shape[2])
            u, singular, vh = xp.linalg.svd(matrix, full_matrices=False)
            keep = min(used, len(singular))
            runtime.tensors[site] = (u[:, :keep] * singular[:keep]).reshape(
                runtime.tensors[site].shape[0], 2, keep
            )
            runtime.tensors[site + 1] = vh[:keep, :].reshape(keep, 2, runtime.tensors[site + 1].shape[2])
        runtime.discarded_weight += discarded
        runtime.bond_dim_used = max(
            (max(tensor.shape[0], tensor.shape[2]) for tensor in runtime.tensors),
            default=1,
        )

    start_sweep = 0
    if payload.resume_from:
        checkpoint_info = runtime.restore_checkpoint(
            payload.resume_from,
            expected_method="finite-two-site-dmrg",
            expected_dtype=payload.dtype,
            expected_request_sha256=problem_sha256,
        )
        metadata = checkpoint_info.get("metadata", {})
        start_sweep = int(metadata.get("completed_sweeps", checkpoint_info.get("step", 0)))
        if start_sweep > payload.sweeps:
            raise ValueError(
                f"checkpoint already contains {start_sweep} sweeps, but the requested run only allows {payload.sweeps}"
            )
        raw_history = metadata.get("history", [])
        if not isinstance(raw_history, list):
            raise ValueError("checkpoint history is invalid")
        history = [dict(point) for point in raw_history]
        previous_energy = metadata.get("previous_energy")
        if previous_energy is not None:
            previous_energy = float(previous_energy)
        solver_iterations_total = int(metadata.get("solver_iterations_total", 0))

    def save_sweep_checkpoint(sweep_number: int, energy: float) -> None:
        nonlocal checkpoint_info
        if not payload.checkpoint_path:
            return
        checkpoint_info = runtime.save_checkpoint(
            payload.checkpoint_path,
            CheckpointManifest(
                checkpoint_id=f"dmrg-{problem_sha256[:12]}-sweep-{sweep_number}",
                request_sha256=problem_sha256,
                method="finite-two-site-dmrg",
                representation="mps",
                dtype=payload.dtype,
                device="cuda" if hasattr(xp, "cuda") else "cpu",
                step=sweep_number,
                created_at=datetime.now(timezone.utc).isoformat(),
                metadata={
                    "completed_sweeps": sweep_number,
                    "previous_energy": float(energy),
                    "solver_iterations_total": solver_iterations_total,
                    "history": history,
                },
            ),
        )

    coefficient_scale = max(1.0, sum(abs(float(term.coefficient)) for term in payload.terms))
    precision_epsilon = 1.1920928955078125e-7 if payload.dtype == "complex64" else 2.220446049250313e-16
    default_variance_tolerance = max(
        float(payload.tolerance),
        8.0 * precision_epsilon * coefficient_scale * coefficient_scale,
    )
    variance_tolerance = float(payload.variance_tolerance or default_variance_tolerance)
    energy_variance: float | None = None
    energy_std: float | None = None
    last_energy_ok = False
    last_residual_ok = False
    last_variance_ok: bool | None = None

    for sweep in range(start_sweep, payload.sweeps):
        current_sweep_solver_residual_max = 0.0
        for site in range(payload.n_qubits - 1):
            optimize(site, True)
        for site in range(payload.n_qubits - 2, -1, -1):
            optimize(site, False)
        runtime.sync()
        energy = _energy(xp, runtime.tensors, payload.terms)
        delta = None if previous_energy is None else abs(energy - previous_energy)
        norm2 = runtime.norm2()
        energy_ok = delta is not None and delta <= payload.tolerance
        residual_ok = current_sweep_solver_residual_max <= payload.residual_tolerance
        sweep_variance: float | None = None
        variance_ok: bool | None = None
        if energy_ok and len(payload.terms) <= 256:
            _, _, sweep_variance = runtime.energy_moments(payload.terms)
            variance_ok = sweep_variance <= variance_tolerance
        history.append({
            "sweep": sweep + 1,
            "energy": energy,
            "delta_energy": delta,
            "bond_dim_used": runtime.bond_dim_used,
            "discarded_weight": runtime.discarded_weight,
            "norm2": norm2,
            "local_solver": payload.local_solver,
            "local_solver_iterations_total": solver_iterations_total,
            "local_solver_residual_max": current_sweep_solver_residual_max,
            "energy_variance": sweep_variance,
            "energy_std": math.sqrt(sweep_variance) if sweep_variance is not None else None,
            "energy_delta_ok": energy_ok,
            "residual_ok": residual_ok,
            "variance_ok": variance_ok,
        })
        if progress_cb:
            progress_cb((sweep + 1) / max(1, payload.sweeps), "dmrg-sweep")
        last_energy_ok = energy_ok
        last_residual_ok = residual_ok
        last_variance_ok = variance_ok
        save_sweep_checkpoint(sweep + 1, energy)
        if energy_ok and residual_ok and (variance_ok is not False):
            converged = True
            break
        previous_energy = energy

    runtime.sync()
    observables = payload.observables or payload.terms
    values = mps_expectation_from_tensors(xp, runtime.tensors, observables)
    energy = _energy(xp, runtime.tensors, payload.terms)
    if len(payload.terms) <= 256:
        _, _, energy_variance = runtime.energy_moments(payload.terms)
        energy_std = math.sqrt(energy_variance)
    else:
        energy_variance = None
        energy_std = None
    if history:
        history[-1]["energy_variance"] = energy_variance
        history[-1]["energy_std"] = energy_std
        if history[-1]["delta_energy"] is not None:
            last_energy_ok = bool(history[-1]["delta_energy"] <= payload.tolerance)
        last_residual_ok = bool(history[-1]["local_solver_residual_max"] <= payload.residual_tolerance)
        last_variance_ok = None if energy_variance is None else bool(energy_variance <= variance_tolerance)
        if not converged and last_energy_ok and last_residual_ok and (last_variance_ok is not False):
            converged = True
    warnings = [
        "two-site finite DMRG is variational within the selected MPS bond dimension",
        "increase sweeps and bond_dim until energy, residual, variance, and discarded_weight stabilize",
    ]
    if not converged:
        if not last_energy_ok:
            warnings.append("DMRG did not reach the requested energy-delta tolerance")
        if not last_residual_ok:
            warnings.append("local solver residual did not reach the requested tolerance")
        if last_variance_ok is False:
            warnings.append("energy variance did not reach the requested tolerance")
        if last_variance_ok is None:
            warnings.append("energy variance was not evaluated because the Hamiltonian has more than 256 terms")
    if runtime.discarded_weight > 1e-12:
        warnings.append("bond dimension truncated entanglement; inspect discarded_weight")
    exact_cross_check = _automatic_exact_cross_check(xp, payload, energy)
    if exact_cross_check["performed"] and not exact_cross_check["passed"]:
        warnings.append("automatic exact diagonalization cross-check exceeded the configured dtype tolerance")
    convergence = ConvergenceReport(
        converged=converged,
        criterion=(
            f"abs(delta_energy) <= {payload.tolerance} and "
            f"local_solver_residual <= {payload.residual_tolerance} and "
            f"energy_variance <= {variance_tolerance} when evaluated"
        ),
        points=[
            ConvergencePoint(
                iteration=int(point["sweep"]),
                energy=float(point["energy"]),
                residual=float(point["local_solver_residual_max"]),
                norm_drift=float(abs(point["norm2"] - 1.0)),
                discarded_weight=float(point["discarded_weight"]),
                bond_dim=int(point["bond_dim_used"]),
            )
            for point in history
        ],
        warnings=list(warnings) if not converged else [],
    )
    research_result = ResearchResult(
        status="done" if converged else "needs_review",
        method="finite-two-site-dmrg",
        representation="mps",
        metrics={"energy": energy, "energy_variance": energy_variance, "energy_std": energy_std, "norm2": runtime.norm2()},
        truncation=runtime.truncation_report(),
        convergence=convergence,
        resources=runtime.estimate_resources(),
        checkpoint=checkpoint_info,
        warnings=list(warnings),
        limitations=[
            "finite two-site DMRG is bounded by the selected MPS bond dimension",
            "results require bond-dimension and sweep convergence studies for publication-quality evidence",
        ],
        provenance={"local_solver": payload.local_solver, "lanczos_maxiter": payload.lanczos_maxiter},
        details={
            "history": history,
            "observables": len(observables),
            "exact_cross_check": exact_cross_check,
        },
    ).to_dict()
    return {
        "status": "done",
        "backend": "tensor-network-mps-dmrg",
        "method": "finite-two-site-dmrg",
        "n_qubits": payload.n_qubits,
        "dtype": payload.dtype,
        "bond_dim_requested": payload.bond_dim,
        "bond_dim_used": runtime.bond_dim_used,
        "sweeps_requested": payload.sweeps,
        "sweeps_completed": len(history),
        "tolerance": payload.tolerance,
        "local_solver": payload.local_solver,
        "lanczos_maxiter": payload.lanczos_maxiter,
        "residual_tolerance": payload.residual_tolerance,
        "variance_tolerance": variance_tolerance,
        "local_solver_iterations": solver_iterations_total,
        "local_solver_residual": history[-1]["local_solver_residual_max"] if history else solver_residual_max,
        "converged": converged,
        "ground_energy": energy,
        "energy": energy,
        "energy_variance": energy_variance,
        "energy_std": energy_std,
        "exact_cross_check": exact_cross_check,
        "observables": structured_observables(observables, values),
        "history": history,
        "discarded_weight": runtime.discarded_weight,
        "norm2": runtime.norm2(),
        "truncation_report": runtime.truncation_report().__dict__,
        "resource_estimate": runtime.estimate_resources(),
        "checkpoint": checkpoint_info,
        "research_result": research_result,
        "approximate": True,
        "warnings": warnings,
        "time_ms": round((time.perf_counter() - started) * 1000, 3),
    }
