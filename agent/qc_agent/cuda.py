from __future__ import annotations

import os
import site
from pathlib import Path
from typing import Any

from fastapi import HTTPException


def add_cuda_dll_dirs() -> None:
    """
    Help Windows locate CUDA DLLs that come from pip-installed NVIDIA packages
    (e.g. nvidia-cublas-cu12). This avoids cublas/cudart import failures.
    """
    try:
        for sp in site.getsitepackages():
            nvidia_dir = Path(sp) / "nvidia"
            if not nvidia_dir.exists():
                continue
            for bin_dir in nvidia_dir.glob("*/bin"):
                try:
                    os.add_dll_directory(str(bin_dir))
                except Exception:
                    pass
    except Exception:
        pass


def require_gpu(cp: Any) -> None:
    if cp is None:
        raise HTTPException(status_code=500, detail="CuPy is not available. Install cupy-cuda12x.")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise HTTPException(status_code=500, detail="No CUDA devices detected.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CUDA init failed: {exc}") from exc


def hardware_info(cp: Any) -> dict[str, Any]:
    if cp is None:
        return {"gpu": {"available": False, "reason": "cupy import failed"}}
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        props = cp.cuda.runtime.getDeviceProperties(0) if count > 0 else None
        return {
            "gpu": {
                "available": count > 0,
                "count": count,
                "cupy_version": cp.__version__,
                "device0": {
                    "name": props["name"].decode() if props else None,
                    "total_global_mem": int(props["totalGlobalMem"]) if props else None,
                    "multi_processor_count": int(props["multiProcessorCount"]) if props else None,
                    "compute_capability": f'{props["major"]}.{props["minor"]}' if props else None,
                },
            }
        }
    except Exception as exc:
        return {"gpu": {"available": False, "reason": str(exc)}}

