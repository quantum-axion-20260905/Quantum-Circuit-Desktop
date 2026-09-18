"""Bounded one-site iPEPS corner-transfer-matrix contraction.

This module deliberately owns only the numerical representation.  Admission,
HTTP lifecycle, provenance, and domain builders stay outside the module so a
future two-site/2x2 unit-cell implementation can reuse the same result seam.

The first supported solver is a one-site, environment-only CTMRG contraction.
It does not optimize the iPEPS tensor and therefore reports ``needs_review``;
that distinction is important for research use and prevents a product-state
ansatz from being presented as a variational ground state.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Iterable

from ..plugins.models import CTMRGPayload, PauliTerm
from ..provenance import sha256_json
from .checkpoints import load_ctm_checkpoint, save_ctm_checkpoint
from .contracts import CheckpointManifest, ConvergencePoint, ConvergenceReport, ResearchResult, TruncationReport
from .observables import structured_observables


@dataclass(frozen=True)
class CTMEnvironment:
    """Four corners and four edge tensors around a one-site unit cell.

    Corner tensors have shape ``(chi, chi)`` and edge tensors have shape
    ``(chi, D2, chi)``, where ``D2`` is the double-layer virtual dimension.
    The ordering is ``C1/T1/C2`` at the top, ``C4/T3/C3`` at the bottom,
    with ``T4`` on the left and ``T2`` on the right.
    """

    C1: Any
    C2: Any
    C3: Any
    C4: Any
    T1: Any
    T2: Any
    T3: Any
    T4: Any

    def tensors(self) -> tuple[Any, ...]:
        return (self.C1, self.C2, self.C3, self.C4, self.T1, self.T2, self.T3, self.T4)


def _host(value: Any) -> Any:
    try:
        return value.get()
    except AttributeError:
        return value


def _real(value: Any) -> float:
    return float(complex(_host(value)).real)


def _max_abs(xp: Any, value: Any) -> float:
    return float(_host(xp.max(xp.abs(value))))


def _build_tensor(xp: Any, payload: CTMRGPayload) -> Any:
    """Create a deterministic normalized product iPEPS ansatz.

    If ``tensor_data`` is supplied it is interpreted as row-major complex
    pairs in the explicit shape ``(physical, up, down, left, right)``.
    Otherwise a deterministic product-state ansatz is built for compatibility
    and the result is marked as non-variational by the caller.
    """

    physical = int(payload.physical_bond_dim)
    if physical != 2:
        raise ValueError("the first CTMRG solver supports physical_bond_dim=2 only")
    dtype = xp.complex64 if payload.dtype == "complex64" else xp.complex128
    virtual = int(payload.virtual_bond_dim)
    tensor = xp.zeros((physical, virtual, virtual, virtual, virtual), dtype=dtype)
    if payload.tensor_data is not None:
        values = [complex(float(real), float(imaginary)) for real, imaginary in payload.tensor_data]
        return xp.asarray(values, dtype=dtype).reshape(tensor.shape)
    if payload.initial_state == "plus":
        amplitudes = [1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)]
    else:
        amplitudes = [1.0, 0.0]
        if payload.initial_state == "down":
            amplitudes = [0.0, 1.0]
        elif payload.initial_state == "neel":
            # The one-site solver uses the even sublattice.  A checkerboard
            # unit cell will receive alternating amplitudes when implemented.
            amplitudes = [1.0, 0.0]
    tensor[:, 0, 0, 0, 0] = xp.asarray(amplitudes, dtype=dtype)
    return tensor


def _pauli_matrix(xp: Any, dtype: Any, label: str) -> Any:
    if label == "I":
        return xp.eye(2, dtype=dtype)
    if label == "X":
        return xp.asarray([[0, 1], [1, 0]], dtype=dtype)
    if label == "Y":
        return xp.asarray([[0, -1j], [1j, 0]], dtype=dtype)
    if label == "Z":
        return xp.asarray([[1, 0], [0, -1]], dtype=dtype)
    raise ValueError(f"unsupported Pauli operator {label!r}")


def _double_layer(xp: Any, tensor: Any, operator: Any | None = None) -> Any:
    """Fuse ket/bra virtual legs while contracting the physical leg."""

    physical = tensor.shape[0]
    dtype = tensor.dtype
    op = operator if operator is not None else xp.eye(physical, dtype=dtype)
    raw = xp.einsum(
        "sudlr,st,tUDLR->uUdDlLrR",
        tensor,
        op,
        xp.conj(tensor),
    )
    d2 = tensor.shape[1] * tensor.shape[1]
    return raw.reshape((d2, d2, d2, d2))


def _initialize_environment(xp: Any, double_layer: Any, chi: int) -> CTMEnvironment:
    d2 = int(double_layer.shape[0])
    dtype = double_layer.dtype
    corner = xp.zeros((chi, chi), dtype=dtype)
    diagonal = min(chi, d2)
    corner[xp.arange(diagonal), xp.arange(diagonal)] = 1
    edge = xp.zeros((chi, d2, chi), dtype=dtype)
    for index in range(min(chi, d2)):
        edge[index, :, index] = 1
    return CTMEnvironment(corner, corner.copy(), corner.copy(), corner.copy(), edge, edge.copy(), edge.copy(), edge.copy())


def _ctm_move(xp: Any, left_corner: Any, right_corner: Any, grown_edge: Any, chi: int) -> tuple[Any, Any, Any, float]:
    """Truncate one enlarged boundary with a Hermitian half-system projector."""

    rho = left_corner @ xp.conj(left_corner).T + right_corner @ xp.conj(right_corner).T
    rho = 0.5 * (rho + xp.conj(rho).T)
    eigenvalues, eigenvectors = xp.linalg.eigh(rho)
    keep = min(int(chi), int(eigenvalues.shape[0]))
    projector = eigenvectors[:, -keep:]
    projector = projector[:, ::-1]
    discarded = float(_host(xp.sum(xp.clip(xp.real(eigenvalues[:-keep]), 0, None)))) if keep < eigenvalues.shape[0] else 0.0
    retained = float(_host(xp.sum(xp.clip(xp.real(eigenvalues[-keep:]), 0, None))))
    discarded_weight = discarded / max(discarded + retained, 1e-30)
    projector_dag = xp.conj(projector).T
    new_left = projector_dag @ left_corner
    new_right = projector_dag @ right_corner
    # The first factor is stored as (enlarged, kept) so einsum contracts its
    # conjugate on the enlarged index and emits the kept index.  This is the
    # same P† T P projection used for the corners, but preserves the explicit
    # edge orientation without transposing the middle leg.
    new_edge = xp.einsum("ia,idj,jb->adb", xp.conj(projector), grown_edge, projector)
    return new_left, new_right, new_edge, discarded_weight


def _left_move(xp: Any, env: CTMEnvironment, double_layer: Any, chi: int) -> tuple[CTMEnvironment, float]:
    d2 = int(double_layer.shape[0])
    c1_g = xp.einsum("ab,buc->auc", env.C1, env.T1).reshape(-1, env.T1.shape[2])
    c4_g = xp.einsum("gh,hdi->gdi", env.C4, env.T3).reshape(-1, env.T3.shape[2])
    t4_g = xp.einsum("alg,udlr->augdr", env.T4, double_layer)
    t4_g = t4_g.transpose(0, 1, 4, 2, 3).reshape(c1_g.shape[0], d2, c4_g.shape[0])
    c1, c4, t4, discarded = _ctm_move(xp, c1_g, c4_g, t4_g, chi)
    return CTMEnvironment(c1, env.C2, env.C3, c4, env.T1, env.T2, env.T3, t4), discarded


def _right_move(xp: Any, env: CTMEnvironment, double_layer: Any, chi: int) -> tuple[CTMEnvironment, float]:
    d2 = int(double_layer.shape[0])
    c2_g = xp.einsum("ce,buc->eub", env.C2, env.T1).reshape(-1, env.T1.shape[0])
    c3_g = xp.einsum("im,hdi->mdh", env.C3, env.T3).reshape(-1, env.T3.shape[0])
    t2_g = xp.einsum("erm,udlr->eumdl", env.T2, double_layer)
    t2_g = t2_g.transpose(0, 1, 4, 2, 3).reshape(c2_g.shape[0], d2, c3_g.shape[0])
    c2, c3, t2, discarded = _ctm_move(xp, c2_g, c3_g, t2_g, chi)
    return CTMEnvironment(env.C1, c2, c3, env.C4, env.T1, t2, env.T3, env.T4), discarded


def _top_move(xp: Any, env: CTMEnvironment, double_layer: Any, chi: int) -> tuple[CTMEnvironment, float]:
    d2 = int(double_layer.shape[0])
    c1_g = xp.einsum("ab,alg->blg", env.C1, env.T4).reshape(-1, env.T4.shape[2])
    c2_g = xp.einsum("ce,erm->crm", env.C2, env.T2).reshape(-1, env.T2.shape[2])
    t1_g = xp.einsum("buc,udlr->bcdlr", env.T1, double_layer)
    t1_g = t1_g.transpose(0, 3, 2, 1, 4).reshape(c1_g.shape[0], d2, c2_g.shape[0])
    c1, c2, t1, discarded = _ctm_move(xp, c1_g, c2_g, t1_g, chi)
    return CTMEnvironment(c1, c2, env.C3, env.C4, t1, env.T2, env.T3, env.T4), discarded


def _bottom_move(xp: Any, env: CTMEnvironment, double_layer: Any, chi: int) -> tuple[CTMEnvironment, float]:
    d2 = int(double_layer.shape[0])
    c4_g = xp.einsum("gh,alg->hal", env.C4, env.T4).transpose(0, 2, 1).reshape(-1, env.T4.shape[0])
    c3_g = xp.einsum("im,erm->ire", env.C3, env.T2).reshape(-1, env.T2.shape[0])
    t3_g = xp.einsum("hdi,udlr->hiulr", env.T3, double_layer)
    t3_g = t3_g.transpose(0, 3, 2, 1, 4).reshape(c4_g.shape[0], d2, c3_g.shape[0])
    c4, c3, t3, discarded = _ctm_move(xp, c4_g, c3_g, t3_g, chi)
    return CTMEnvironment(env.C1, env.C2, c3, c4, env.T1, env.T2, t3, env.T4), discarded


def _renormalize(xp: Any, env: CTMEnvironment) -> CTMEnvironment:
    def normalize(value: Any) -> Any:
        return value / (_max_abs(xp, value) + 1e-30)

    return CTMEnvironment(*(normalize(value) for value in env.tensors()))


def _environment_residual(xp: Any, before: CTMEnvironment, after: CTMEnvironment) -> float:
    residual = 0.0
    for old, new in zip(before.tensors(), after.tensors()):
        scale = max(_max_abs(xp, old), 1e-30)
        residual = max(residual, _max_abs(xp, new - old) / scale)
    return residual


def _environment_contraction(xp: Any, env: CTMEnvironment, local_tensor: Any) -> Any:
    return xp.einsum(
        "ab,buc,ce,erm,im,hdi,gh,alg,udlr->",
        env.C1,
        env.T1,
        env.C2,
        env.T2,
        env.C3,
        env.T3,
        env.C4,
        env.T4,
        local_tensor,
    )


def _term_expectation(xp: Any, env: CTMEnvironment, tensor: Any, term: PauliTerm) -> float:
    if len(term.paulis) > 1:
        raise ValueError("one-site CTMRG onsite terms must act on at most one unit-cell site")
    pauli = next(iter(term.paulis.values()), "I")
    dtype = tensor.dtype
    op = _pauli_matrix(xp, dtype, str(pauli))
    numerator = _environment_contraction(xp, env, _double_layer(xp, tensor, op))
    denominator = _environment_contraction(xp, env, _double_layer(xp, tensor))
    return _real(numerator / (denominator + 1e-30))


def _horizontal_two_site_contraction(xp: Any, env: CTMEnvironment, left: Any, right: Any) -> Any:
    """Contract two neighboring sites in the horizontal direction."""

    return xp.einsum(
        "ab,buc,cve,ef,fqm,mi,hyi,gwh,gh,azg,uwzx,vyxq->",
        env.C1,
        env.T1,
        env.T1,
        env.C2,
        env.T2,
        env.C3,
        env.T3,
        env.T3,
        env.C4,
        env.T4,
        left,
        right,
    )


def _vertical_two_site_contraction(xp: Any, env: CTMEnvironment, top: Any, bottom: Any) -> Any:
    """Contract two neighboring sites in the vertical direction."""

    return xp.einsum(
        "ab,buc,ce,erm,mqn,ni,hyi,gh,alg,gkh,uxlr,xykq->",
        env.C1,
        env.T1,
        env.C2,
        env.T2,
        env.T2,
        env.C3,
        env.T3,
        env.C4,
        env.T4,
        env.T4,
        top,
        bottom,
    )


def _interaction_expectation(
    xp: Any,
    env: CTMEnvironment,
    tensor: Any,
    displacement: list[int],
    left_pauli: str,
    right_pauli: str,
) -> float | None:
    """Evaluate a nearest-neighbor two-site Pauli expectation from the CTM.

    A two-site density matrix is only defined here for nearest horizontal or
    vertical neighbors of a one-site translational cell.  Unsupported
    displacements return ``None`` so callers cannot mistake a product of local
    averages for a connected two-site observable.
    """

    dx, dy = (int(value) for value in displacement)
    if (abs(dx), abs(dy)) not in ((1, 0), (0, 1)):
        return None
    dtype = tensor.dtype
    left_operator = _pauli_matrix(xp, dtype, left_pauli)
    right_operator = _pauli_matrix(xp, dtype, right_pauli)
    left = _double_layer(xp, tensor, left_operator)
    right = _double_layer(xp, tensor, right_operator)
    identity = _double_layer(xp, tensor)
    if abs(dx) == 1:
        numerator = _horizontal_two_site_contraction(xp, env, left, right)
        denominator = _horizontal_two_site_contraction(xp, env, identity, identity)
    else:
        numerator = _vertical_two_site_contraction(xp, env, left, right)
        denominator = _vertical_two_site_contraction(xp, env, identity, identity)
    return _real(numerator / (denominator + 1e-30))


def _resource_summary(xp: Any, env: CTMEnvironment, tensor: Any, iterations: int) -> dict[str, Any]:
    values = sum(int(item.size) for item in (*env.tensors(), tensor))
    itemsize = int(getattr(tensor.dtype, "itemsize", 8))
    return {
        "representation": "ipeps",
        "unit_cell": [1, 1],
        "physical_bond_dim": int(tensor.shape[0]),
        "virtual_bond_dim": int(tensor.shape[1]),
        "environment_bond_dim_used": int(env.C1.shape[0]),
        "double_layer_virtual_dim": int(env.T1.shape[1]),
        "tensor_values": values,
        "peak_bytes_estimate": int(math.ceil(values * itemsize * 3.0)),
        "iterations": int(iterations),
        "materializes_statevector": False,
        "device": "cuda" if hasattr(xp, "cuda") else "cpu",
    }


def _problem_sha256(payload: CTMRGPayload) -> str:
    """Fingerprint the scientific tensor problem, excluding run controls."""

    data = payload.model_dump(
        mode="json",
        exclude={"iterations", "tolerance", "max_time_ms", "max_mem_mb", "checkpoint_path", "resume_from"},
    )
    return sha256_json(data)


def _environment_from_arrays(arrays: dict[str, Any]) -> CTMEnvironment:
    return CTMEnvironment(*(arrays[name] for name in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4")))


def run_ctmrg(
    xp: Any,
    payload: CTMRGPayload,
    *,
    progress_cb: Any = None,
    cancel_cb: Any = None,
) -> dict[str, Any]:
    """Run a bounded one-site CTMRG contraction and return research evidence."""

    if list(payload.unit_cell) != [1, 1]:
        raise ValueError("the first CTMRG solver supports unit_cell=[1, 1]; multi-site cells are not silently approximated")
    started = time.perf_counter()
    tensor = _build_tensor(xp, payload)
    double_layer = _double_layer(xp, tensor)
    chi = int(payload.environment_bond_dim)
    env = _initialize_environment(xp, double_layer, chi)
    problem_sha256 = _problem_sha256(payload)
    points: list[ConvergencePoint] = []
    discarded_total = 0.0
    residual = math.inf
    start_iteration = 0
    checkpoint_info: dict[str, Any] = {}
    if payload.resume_from:
        manifest, arrays = load_ctm_checkpoint(payload.resume_from, xp)
        if manifest.get("request_sha256") != problem_sha256:
            raise ValueError("CTMRG checkpoint does not match the scientific tensor problem")
        if manifest.get("dtype") != payload.dtype:
            raise ValueError("CTMRG checkpoint dtype does not match the requested dtype")
        metadata = manifest.get("metadata", {})
        if int(metadata.get("physical_bond_dim", -1)) != int(payload.physical_bond_dim):
            raise ValueError("CTMRG checkpoint physical bond dimension does not match the request")
        if int(metadata.get("virtual_bond_dim", -1)) != int(payload.virtual_bond_dim):
            raise ValueError("CTMRG checkpoint virtual bond dimension does not match the request")
        if int(metadata.get("environment_bond_dim", -1)) != int(payload.environment_bond_dim):
            raise ValueError("CTMRG checkpoint environment bond dimension does not match the request")
        start_iteration = int(manifest.get("step", 0))
        if start_iteration > int(payload.iterations):
            raise ValueError(
                f"checkpoint already contains {start_iteration} iterations, but the requested run only allows {payload.iterations}"
            )
        raw_points = metadata.get("convergence_points", [])
        if not isinstance(raw_points, list):
            raise ValueError("CTMRG checkpoint convergence history is invalid")
        points = [ConvergencePoint(**dict(point)) for point in raw_points]
        discarded_total = float(metadata.get("discarded_weight_total", 0.0))
        residual = float(metadata.get("residual", math.inf))
        env = _environment_from_arrays(arrays)
        checkpoint_info = manifest

    def save_iteration_checkpoint(iteration: int) -> None:
        nonlocal checkpoint_info
        if not payload.checkpoint_path:
            return
        checkpoint_info = save_ctm_checkpoint(
            payload.checkpoint_path,
            env,
            CheckpointManifest(
                checkpoint_id=f"ctmrg-{problem_sha256[:12]}-iteration-{iteration}",
                request_sha256=problem_sha256,
                method="ipeps-ctmrg-contraction",
                representation="ipeps",
                dtype=payload.dtype,
                device="cuda" if hasattr(xp, "cuda") else "cpu",
                step=iteration,
                created_at=datetime.now(timezone.utc).isoformat(),
                metadata={
                    "completed_iterations": iteration,
                    "residual": float(residual),
                    "discarded_weight_total": float(discarded_total),
                    "convergence_points": [point.__dict__ for point in points],
                    "physical_bond_dim": int(payload.physical_bond_dim),
                    "virtual_bond_dim": int(payload.virtual_bond_dim),
                    "environment_bond_dim": int(payload.environment_bond_dim),
                },
            ),
        )

    for iteration in range(start_iteration + 1, int(payload.iterations) + 1):
        if cancel_cb and cancel_cb():
            raise RuntimeError("job canceled")
        before = env
        env, discarded = _left_move(xp, env, double_layer, chi)
        discarded_total += discarded
        env, discarded = _right_move(xp, env, double_layer, chi)
        discarded_total += discarded
        env, discarded = _top_move(xp, env, double_layer, chi)
        discarded_total += discarded
        env, discarded = _bottom_move(xp, env, double_layer, chi)
        discarded_total += discarded
        env = _renormalize(xp, env)
        residual = _environment_residual(xp, before, env)
        points.append(ConvergencePoint(iteration=iteration, residual=residual, environment_dim=int(env.C1.shape[0]), discarded_weight=discarded))
        save_iteration_checkpoint(iteration)
        if progress_cb:
            progress_cb(iteration / max(1, int(payload.iterations)), "ctmrg-sweep")
        if residual <= float(payload.tolerance):
            break

    norm = _real(_environment_contraction(xp, env, double_layer))
    onsite_values: list[float] = []
    for term in payload.terms:
        onsite_values.append(_term_expectation(xp, env, tensor, term))
    interaction_values: list[float | None] = []
    for interaction in payload.interactions:
        interaction_values.append(
            _interaction_expectation(
                xp,
                env,
                tensor,
                interaction.displacement,
                interaction.left_pauli,
                interaction.right_pauli,
            )
        )
    interaction_values_available = all(value is not None for value in interaction_values)
    energy = sum(float(term.coefficient) * value for term, value in zip(payload.terms, onsite_values))
    energy += sum(
        float(term.coefficient) * float(value)
        for term, value in zip(payload.interactions, interaction_values)
        if value is not None
    )
    converged = bool(residual <= float(payload.tolerance))
    warnings = [
        "CTMRG contraction is implemented for a one-site iPEPS environment; unit-cell extensions require a separate acceptance test",
        "compare environment_bond_dim and iteration convergence before using values as scientific conclusions",
    ]
    if payload.tensor_data is None:
        warnings.append("the tensor is a deterministic product-state ansatz; no variational ground-state optimization was performed")
    else:
        warnings.append("the imported tensor was contracted without variational ground-state optimization")
    if not interaction_values_available:
        warnings.append("one or more interaction displacements are outside the supported nearest-neighbor two-site CTM contraction")
    checkpoint_result = checkpoint_info or {
        "resumable": False,
        "reason": "set checkpoint_path to persist and resume the CTMRG environment",
    }
    research_result = ResearchResult(
        status="needs_review",
        method="ipeps-ctmrg-contraction",
        representation="ipeps",
        metrics={"norm": norm, "energy": float(energy), "residual": float(residual)},
        truncation=TruncationReport(
            discarded_weight=float(discarded_total),
            max_bond_dim=int(payload.virtual_bond_dim),
            max_environment_dim=int(payload.environment_bond_dim),
        ),
        convergence=ConvergenceReport(
            converged=converged,
            criterion="normalized corner/edge environment residual",
            points=points,
            warnings=list(warnings),
        ),
        resources=_resource_summary(xp, env, tensor, len(points)),
        checkpoint=checkpoint_result,
        warnings=list(warnings),
        limitations=[
            "only one-site unit cell is numerically enabled",
            "no variational tensor update or full ground-state optimization",
            "non-nearest interaction displacements are not yet supported by the one-site two-site-RDM contraction",
        ],
        details={"initial_state": payload.initial_state, "environment_shapes": [list(item.shape) for item in env.tensors()]},
    ).to_dict()
    return {
        "status": "done",
        "backend": "tensor-network-ctmrg",
        "method": "ipeps-ctmrg-contraction",
        "representation": "ipeps",
        "unit_cell": [1, 1],
        "tensor_source": "imported" if payload.tensor_data is not None else "generated-product-ansatz",
        "physical_bond_dim": int(payload.physical_bond_dim),
        "virtual_bond_dim": int(payload.virtual_bond_dim),
        "environment_bond_dim_requested": int(payload.environment_bond_dim),
        "environment_bond_dim_used": int(env.C1.shape[0]),
        "iterations": len(points),
        "converged": converged,
        "residual": float(residual),
        "norm": norm,
        "energy": float(energy),
        "observables": structured_observables(payload.terms, onsite_values),
        "interactions": [
            {
                "index": index,
                "label": interaction.label or f"{interaction.left_pauli}{interaction.left_site}-{interaction.right_pauli}{interaction.right_site}",
                "coefficient": float(interaction.coefficient),
                "value": float(value) if value is not None else None,
                "displacement": list(interaction.displacement),
            }
            for index, (interaction, value) in enumerate(zip(payload.interactions, interaction_values))
        ],
        "resource_estimate": _resource_summary(xp, env, tensor, len(points)),
        "research_result": research_result,
        "checkpoint": checkpoint_result,
        "warnings": warnings,
        "time_ms": round((time.perf_counter() - started) * 1000, 3),
    }
