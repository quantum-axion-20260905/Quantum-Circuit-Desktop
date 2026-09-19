"""Explicit, atomic checkpoint persistence for tensor-network states.

Checkpointing is intentionally opt-in.  A GPU tensor is copied to host memory
only while ``save_mps_checkpoint`` is called; solver loops never checkpoint
implicitly, which keeps the normal execution path predictable for shared RAM.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import CHECKPOINT_SCHEMA, CheckpointManifest
from .ctmrg_dynamic import (
    DYNAMIC_BOUNDARY_SCHEMA,
    BoundaryDimensions,
    DynamicCTMEnvironment,
    dynamic_boundary_manifest_digest,
)


_TENSOR_NAME = re.compile(r"tensor_(\d+)$")
_BOUNDARY_TENSOR_NAME = re.compile(r"boundary_tensor_(\d+)$")
_OPTIMIZER_TENSOR_NAME = re.compile(r"optimizer_tensor_(\d+)$")
_CTM_NAMES = ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4")


def _to_host(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    try:
        value = value.get()
    except AttributeError:
        pass
    return np.asarray(value)


def _validate_tensors(tensors: list[Any]) -> None:
    if not tensors:
        raise ValueError("an MPS checkpoint needs at least one tensor")
    for index, tensor in enumerate(tensors):
        if getattr(tensor, "ndim", None) != 3 or int(tensor.shape[1]) != 2:
            raise ValueError(f"tensor_{index} must have shape (left, 2, right)")
        if index and int(tensors[index - 1].shape[2]) != int(tensor.shape[0]):
            raise ValueError(f"MPS bond mismatch between tensor_{index - 1} and tensor_{index}")
    if int(tensors[0].shape[0]) != 1 or int(tensors[-1].shape[2]) != 1:
        raise ValueError("finite MPS checkpoints must have boundary bond dimension one")


def save_mps_checkpoint(
    path: str | os.PathLike[str],
    tensors: list[Any],
    manifest: CheckpointManifest,
) -> dict[str, Any]:
    """Write a versioned MPS checkpoint atomically and return its manifest."""
    _validate_tensors(tensors)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest_dict = manifest.to_dict()
    metadata = dict(manifest_dict.get("metadata", {}))
    metadata.setdefault("tensor_count", len(tensors))
    metadata.setdefault("discarded_weight", 0.0)
    manifest_dict["metadata"] = metadata
    arrays = {f"tensor_{index}": _to_host(tensor) for index, tensor in enumerate(tensors)}

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = handle.name
            np.savez_compressed(
                handle,
                manifest=np.asarray(json.dumps(manifest_dict, sort_keys=True)),
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    except Exception:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
        raise
    return manifest_dict


def load_mps_checkpoint(
    path: str | os.PathLike[str],
    xp: Any = np,
) -> tuple[dict[str, Any], list[Any]]:
    """Load and validate a checkpoint, converting tensors to ``xp``."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if "manifest" not in archive.files:
            raise ValueError("checkpoint is missing its manifest")
        raw_manifest = archive["manifest"].item()
        manifest = json.loads(str(raw_manifest))
        if manifest.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported checkpoint schema")
        if not manifest.get("resumable", False):
            raise ValueError("checkpoint is marked non-resumable")
        tensor_entries: list[tuple[int, str]] = []
        for name in archive.files:
            match = _TENSOR_NAME.fullmatch(name)
            if match:
                tensor_entries.append((int(match.group(1)), name))
        tensor_entries.sort()
        if not tensor_entries or [index for index, _ in tensor_entries] != list(range(len(tensor_entries))):
            raise ValueError("checkpoint tensor entries are incomplete or non-contiguous")
        host_tensors = [archive[name] for _, name in tensor_entries]
        _validate_tensors(host_tensors)
        tensors = [xp.asarray(tensor) for tensor in host_tensors]
    return manifest, tensors


def _validate_boundary_tensors(tensors: list[Any]) -> None:
    """Validate a finite boundary-MPS with an arbitrary fused physical size."""
    if not tensors:
        raise ValueError("a boundary-MPS checkpoint needs at least one tensor")
    for index, tensor in enumerate(tensors):
        if getattr(tensor, "ndim", None) != 3:
            raise ValueError(f"boundary_tensor_{index} must have shape (left, physical, right)")
        if any(int(size) <= 0 for size in tensor.shape):
            raise ValueError(f"boundary_tensor_{index} dimensions must be positive")
        if index and int(tensors[index - 1].shape[2]) != int(tensor.shape[0]):
            raise ValueError(f"boundary-MPS bond mismatch between tensors {index - 1} and {index}")
    if int(tensors[0].shape[0]) != 1 or int(tensors[-1].shape[2]) != 1:
        raise ValueError("boundary-MPS checkpoints must have open boundary bond dimension one")


def save_boundary_mps_checkpoint(
    path: str | os.PathLike[str],
    tensors: list[Any],
    manifest: CheckpointManifest,
) -> dict[str, Any]:
    """Atomically persist a boundary-MPS environment checkpoint.

    Boundary physical legs are fused PEPS virtual bra/ket indices and therefore
    are not restricted to the physical dimension two used by finite MPS
    checkpoints.  Keeping this format separate prevents a future solver from
    accidentally interpreting an environment as a physical MPS state.
    """
    _validate_boundary_tensors(tensors)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest_dict = manifest.to_dict()
    metadata = dict(manifest_dict.get("metadata", {}))
    metadata.setdefault("tensor_count", len(tensors))
    metadata.setdefault("representation", "boundary-mps")
    manifest_dict["metadata"] = metadata
    arrays = {
        f"boundary_tensor_{index}": _to_host(tensor)
        for index, tensor in enumerate(tensors)
    }

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = handle.name
            np.savez_compressed(
                handle,
                manifest=np.asarray(json.dumps(manifest_dict, sort_keys=True)),
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    except Exception:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
        raise
    return manifest_dict


def load_boundary_mps_checkpoint(
    path: str | os.PathLike[str],
    xp: Any = np,
) -> tuple[dict[str, Any], list[Any]]:
    """Load and validate a boundary-MPS environment checkpoint."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if "manifest" not in archive.files:
            raise ValueError("boundary-MPS checkpoint is missing its manifest")
        raw_manifest = archive["manifest"].item()
        manifest = json.loads(str(raw_manifest))
        if manifest.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported checkpoint schema")
        if manifest.get("method") != "finite-boundary-mps":
            raise ValueError("checkpoint method is not finite-boundary-mps")
        if manifest.get("representation") != "boundary-mps":
            raise ValueError("checkpoint representation is not boundary-mps")
        if not manifest.get("resumable", False):
            raise ValueError("checkpoint is marked non-resumable")
        tensor_entries: list[tuple[int, str]] = []
        for name in archive.files:
            match = _BOUNDARY_TENSOR_NAME.fullmatch(name)
            if match:
                tensor_entries.append((int(match.group(1)), name))
        tensor_entries.sort()
        if not tensor_entries or [index for index, _ in tensor_entries] != list(range(len(tensor_entries))):
            raise ValueError("boundary-MPS checkpoint tensor entries are incomplete or non-contiguous")
        host_tensors = [archive[name] for _, name in tensor_entries]
        _validate_boundary_tensors(host_tensors)
        tensors = [xp.asarray(tensor) for tensor in host_tensors]
    return manifest, tensors


def _validate_optimizer_tensors(tensors: list[Any]) -> None:
    """Validate optimizer tensor entries without imposing an MPS shape."""

    if not tensors:
        raise ValueError("an optimizer checkpoint needs at least one tensor")
    for index, tensor in enumerate(tensors):
        if getattr(tensor, "ndim", None) is None or int(tensor.ndim) < 1:
            raise ValueError(f"optimizer_tensor_{index} must have at least one dimension")
        if any(int(size) <= 0 for size in tensor.shape):
            raise ValueError(f"optimizer_tensor_{index} dimensions must be positive")
        host = _to_host(tensor)
        if not np.all(np.isfinite(host)):
            raise ValueError(f"optimizer_tensor_{index} contains non-finite values")


def save_optimizer_checkpoint(
    path: str | os.PathLike[str],
    tensors: list[Any],
    manifest: CheckpointManifest,
) -> dict[str, Any]:
    """Atomically persist a variational optimizer tensor state."""

    _validate_optimizer_tensors(tensors)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest_dict = manifest.to_dict()
    metadata = dict(manifest_dict.get("metadata", {}))
    metadata.setdefault("tensor_count", len(tensors))
    metadata.setdefault("tensor_shapes", [list(tensor.shape) for tensor in tensors])
    metadata.setdefault("representation", "ipeps-optimizer-state")
    manifest_dict["metadata"] = metadata
    arrays = {
        f"optimizer_tensor_{index}": _to_host(tensor)
        for index, tensor in enumerate(tensors)
    }
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = handle.name
            np.savez_compressed(
                handle,
                manifest=np.asarray(json.dumps(manifest_dict, sort_keys=True)),
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    except Exception:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
        raise
    return manifest_dict


def load_optimizer_checkpoint(
    path: str | os.PathLike[str],
    xp: Any = np,
    *,
    expected_method: str | None = None,
) -> tuple[dict[str, Any], list[Any]]:
    """Load and validate a variational optimizer tensor state.

    The file format is shared by bounded optimizer policies, while callers
    still provide the exact expected method so a state cannot be resumed by a
    different update rule accidentally.
    """

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if "manifest" not in archive.files:
            raise ValueError("optimizer checkpoint is missing its manifest")
        raw_manifest = archive["manifest"].item()
        manifest = json.loads(str(raw_manifest))
        if manifest.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported checkpoint schema")
        method = manifest.get("method")
        if not isinstance(method, str) or not method:
            raise ValueError("optimizer checkpoint method is missing")
        if expected_method is not None and method != expected_method:
            raise ValueError(f"checkpoint method is not {expected_method}")
        if manifest.get("representation") != "ipeps-optimizer-state":
            raise ValueError("checkpoint representation is not ipeps-optimizer-state")
        if not manifest.get("resumable", False):
            raise ValueError("checkpoint is marked non-resumable")
        tensor_entries: list[tuple[int, str]] = []
        for name in archive.files:
            match = _OPTIMIZER_TENSOR_NAME.fullmatch(name)
            if match:
                tensor_entries.append((int(match.group(1)), name))
        tensor_entries.sort()
        if not tensor_entries or [index for index, _ in tensor_entries] != list(range(len(tensor_entries))):
            raise ValueError("optimizer checkpoint tensor entries are incomplete or non-contiguous")
        host_tensors = [archive[name] for _, name in tensor_entries]
        _validate_optimizer_tensors(host_tensors)
        tensors = [xp.asarray(tensor) for tensor in host_tensors]
    return manifest, tensors


def _validate_ctm_environment(environment: Any) -> None:
    """Validate the shape contract before serializing a CTM environment."""

    tensors = [getattr(environment, name, None) for name in _CTM_NAMES]
    if any(tensor is None for tensor in tensors):
        raise ValueError("CTMRG checkpoint requires C1/C2/C3/C4 and T1/T2/T3/T4 tensors")
    corners = tensors[:4]
    edges = tensors[4:]
    if any(getattr(corner, "ndim", None) != 2 for corner in corners):
        raise ValueError("CTMRG corners must be rank-2 tensors")
    chi = int(corners[0].shape[0])
    if any(tuple(int(size) for size in corner.shape) != (chi, chi) for corner in corners):
        raise ValueError("CTMRG corners must all have shape (chi, chi)")
    if any(getattr(edge, "ndim", None) != 3 for edge in edges):
        raise ValueError("CTMRG edges must be rank-3 tensors")
    d2 = int(edges[0].shape[1])
    if any(tuple(int(size) for size in edge.shape) != (chi, d2, chi) for edge in edges):
        raise ValueError("CTMRG edges must all have shape (chi, double_layer_dim, chi)")


def save_ctm_checkpoint(
    path: str | os.PathLike[str],
    environment: Any,
    manifest: CheckpointManifest,
) -> dict[str, Any]:
    """Atomically persist a CTMRG environment and its resumability manifest."""

    environments = environment if isinstance(environment, list) else [environment]
    if not environments:
        raise ValueError("CTMRG checkpoint needs at least one environment")
    for item in environments:
        _validate_ctm_environment(item)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest_dict = manifest.to_dict()
    metadata = dict(manifest_dict.get("metadata", {}))
    metadata.setdefault("environment_count", len(environments))
    metadata.setdefault("environment_shapes", {
        str(index): {name: list(getattr(item, name).shape) for name in _CTM_NAMES}
        for index, item in enumerate(environments)
    })
    metadata.setdefault("representation", "ipeps")
    manifest_dict["metadata"] = metadata
    arrays: dict[str, np.ndarray] = {}
    for index, item in enumerate(environments):
        prefix = "" if len(environments) == 1 else f"site{index}_"
        arrays.update({f"{prefix}{name}": _to_host(getattr(item, name)) for name in _CTM_NAMES})

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = handle.name
            np.savez_compressed(
                handle,
                manifest=np.asarray(json.dumps(manifest_dict, sort_keys=True)),
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    except Exception:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
        raise
    return manifest_dict


def load_ctm_checkpoint(
    path: str | os.PathLike[str],
    xp: Any = np,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate a CTMRG environment checkpoint."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if "manifest" not in archive.files:
            raise ValueError("CTMRG checkpoint is missing its manifest")
        raw_manifest = archive["manifest"].item()
        manifest = json.loads(str(raw_manifest))
        if manifest.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported checkpoint schema")
        if manifest.get("method") != "ipeps-ctmrg-contraction":
            raise ValueError("checkpoint method is not ipeps-ctmrg-contraction")
        if manifest.get("representation") != "ipeps":
            raise ValueError("checkpoint representation is not ipeps")
        if not manifest.get("resumable", False):
            raise ValueError("checkpoint is marked non-resumable")
        environment_count = int(manifest.get("metadata", {}).get("environment_count", 1))
        if environment_count < 1 or environment_count > 4:
            raise ValueError("CTMRG checkpoint environment count is outside the supported range")
        arrays: dict[str, Any] = {}
        for index in range(environment_count):
            prefix = "" if environment_count == 1 else f"site{index}_"
            for name in _CTM_NAMES:
                entry = f"{prefix}{name}"
                if entry not in archive.files:
                    raise ValueError("CTMRG checkpoint is missing one or more environment tensors")
                arrays[entry] = xp.asarray(archive[entry])
    class _LoadedEnvironment:
        pass
    for index in range(environment_count):
        prefix = "" if environment_count == 1 else f"site{index}_"
        environment = _LoadedEnvironment()
        for name in _CTM_NAMES:
            setattr(environment, name, arrays[f"{prefix}{name}"])
        _validate_ctm_environment(environment)
    return manifest, arrays


def save_dynamic_ctm_checkpoint(
    path: str | os.PathLike[str],
    environment: DynamicCTMEnvironment,
    manifest: CheckpointManifest,
) -> dict[str, Any]:
    """Atomically persist a rectangular dynamic CTM environment.

    This format is deliberately separate from ``save_ctm_checkpoint``.  A
    square fixed-``chi`` checkpoint must never be resumed as a dynamic state,
    and the shape manifest/digest makes directional retained dimensions part
    of the resumability contract rather than incidental array metadata.
    """

    if not isinstance(environment, DynamicCTMEnvironment):
        raise ValueError("dynamic CTMRG checkpoint requires a DynamicCTMEnvironment")
    environment.validate_shapes()
    if manifest.representation != "ipeps-dynamic-boundary":
        raise ValueError("dynamic CTMRG checkpoint representation must be ipeps-dynamic-boundary")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest_dict = manifest.to_dict()
    metadata = dict(manifest_dict.get("metadata", {}))
    shape_manifest = environment.shape_manifest()
    metadata.setdefault("representation", "ipeps-dynamic-boundary")
    metadata["dynamic_boundary_state"] = shape_manifest
    metadata["dynamic_boundary_digest"] = dynamic_boundary_manifest_digest(shape_manifest)
    manifest_dict["metadata"] = metadata
    arrays = {
        f"dynamic_{name}": _to_host(getattr(environment, name))
        for name in _CTM_NAMES
    }

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = handle.name
            np.savez_compressed(
                handle,
                manifest=np.asarray(json.dumps(manifest_dict, sort_keys=True)),
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    except Exception:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
        raise
    return manifest_dict


def load_dynamic_ctm_checkpoint(
    path: str | os.PathLike[str],
    xp: Any = np,
) -> tuple[dict[str, Any], DynamicCTMEnvironment]:
    """Load a rectangular dynamic CTM checkpoint and verify its shape digest."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if "manifest" not in archive.files:
            raise ValueError("dynamic CTMRG checkpoint is missing its manifest")
        raw_manifest = archive["manifest"].item()
        manifest = json.loads(str(raw_manifest))
        if manifest.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported checkpoint schema")
        if manifest.get("method") != "ipeps-ctmrg-contraction":
            raise ValueError("checkpoint method is not ipeps-ctmrg-contraction")
        if manifest.get("representation") != "ipeps-dynamic-boundary":
            raise ValueError("checkpoint representation is not ipeps-dynamic-boundary")
        if not manifest.get("resumable", False):
            raise ValueError("checkpoint is marked non-resumable")
        metadata = manifest.get("metadata", {})
        shape_manifest = metadata.get("dynamic_boundary_state")
        expected_digest = metadata.get("dynamic_boundary_digest")
        if not isinstance(shape_manifest, dict) or not isinstance(expected_digest, str):
            raise ValueError("dynamic CTMRG checkpoint is missing its shape manifest or digest")
        if shape_manifest.get("schema") != DYNAMIC_BOUNDARY_SCHEMA:
            raise ValueError("unsupported dynamic CTMRG boundary schema")
        if dynamic_boundary_manifest_digest(shape_manifest) != expected_digest:
            raise ValueError("dynamic CTMRG boundary shape digest mismatch")
        arrays: dict[str, Any] = {}
        for name in _CTM_NAMES:
            entry = f"dynamic_{name}"
            if entry not in archive.files:
                raise ValueError("dynamic CTMRG checkpoint is missing one or more environment tensors")
            arrays[name] = xp.asarray(archive[entry])
        dimensions = shape_manifest.get("dimensions", {})
        try:
            boundary_dimensions = BoundaryDimensions(
                top=int(dimensions["top"]),
                left=int(dimensions["left"]),
                bottom=int(dimensions["bottom"]),
                right=int(dimensions["right"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("dynamic CTMRG checkpoint has invalid boundary dimensions") from error
        environment = DynamicCTMEnvironment(
            **{name: arrays[name] for name in _CTM_NAMES},
            dimensions=boundary_dimensions,
            map_id=str(shape_manifest.get("map_id", "")),
        )
        actual_manifest = environment.shape_manifest()
        if actual_manifest != shape_manifest:
            raise ValueError("dynamic CTMRG checkpoint tensor shapes do not match its shape manifest")
    return manifest, environment
