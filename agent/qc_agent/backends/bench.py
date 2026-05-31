from __future__ import annotations

import time
from typing import Any

from ..models import BenchPayload


def bench_matmul(cp: Any, payload: BenchPayload) -> dict[str, Any]:
    n = int(payload.size)
    iters = int(payload.iters)
    dtype = cp.float16 if payload.dtype == "fp16" else cp.float32

    if payload.dtype == "fp32":
        try:
            cp.cuda.cublas.setMathMode(cp.cuda.cublas.CUBLAS_TF32_TENSOR_OP_MATH)
        except Exception:
            pass

    a = cp.random.random((n, n), dtype=cp.float32).astype(dtype, copy=False)
    b = cp.random.random((n, n), dtype=cp.float32).astype(dtype, copy=False)

    _ = a @ b
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    for _i in range(iters):
        _ = a @ b
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()

    seconds = max(t1 - t0, 1e-9)
    tflops = (2.0 * (n**3) * iters) / seconds / 1e12

    return {
        "status": "done",
        "backend": "cupy-matmul",
        "dtype": payload.dtype,
        "size": n,
        "iters": iters,
        "time_s": round(seconds, 6),
        "tflops_est": round(tflops, 3),
    }

