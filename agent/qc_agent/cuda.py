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
        devices: dict[str, Any] = {}
        for device_id in range(count):
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            free_mem = total_mem = None
            try:
                with cp.cuda.Device(device_id):
                    free_mem, total_mem = (int(value) for value in cp.cuda.Device(device_id).mem_info)
            except Exception:
                pass
            name = props.get("name") if isinstance(props, dict) else None
            devices[f"device{device_id}"] = {
                "name": name.decode() if isinstance(name, bytes) else name,
                "total_global_mem": int(props.get("totalGlobalMem")) if isinstance(props, dict) and props.get("totalGlobalMem") is not None else None,
                "multi_processor_count": int(props.get("multiProcessorCount")) if isinstance(props, dict) and props.get("multiProcessorCount") is not None else None,
                "compute_capability": (
                    f'{props.get("major")}.{props.get("minor")}'
                    if isinstance(props, dict) and props.get("major") is not None and props.get("minor") is not None
                    else None
                ),
                "free_global_mem": free_mem,
                "total_global_mem_runtime": total_mem,
            }
        return {
            "gpu": {
                "available": count > 0,
                "count": count,
                "cupy_version": cp.__version__,
                **devices,
            }
        }
    except Exception as exc:
        return {"gpu": {"available": False, "reason": str(exc)}}

