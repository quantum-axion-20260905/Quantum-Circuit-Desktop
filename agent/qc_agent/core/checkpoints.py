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


_TENSOR_NAME = re.compile(r"tensor_(\d+)$")


def _to_host(value: Any) -> np.ndarray:
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
