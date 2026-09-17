from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any


class MetricsRegistry:
    """Dependency-free process metrics for local diagnostics and dashboards."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, str, str], int] = defaultdict(int)
        self._latency: dict[tuple[str, str], list[float]] = defaultdict(list)

    def count(self, name: str, *, method: str = "", path: str = "", value: int = 1) -> None:
        with self._lock:
            self._counters[(name, method, path)] += int(value)

    def observe(self, name: str, value_ms: float, *, method: str = "", path: str = "") -> None:
        with self._lock:
            values = self._latency[(name, method, path)]
            values.append(float(value_ms))
            if len(values) > 256:
                del values[: len(values) - 256]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = [
                {"name": name, "method": method, "path": path, "value": value}
                for (name, method, path), value in sorted(self._counters.items())
            ]
            latency = []
            for (name, method, path), values in sorted(self._latency.items()):
                if not values:
                    continue
                latency.append({
                    "name": name,
                    "method": method,
                    "path": path,
                    "count": len(values),
                    "mean_ms": round(sum(values) / len(values), 3),
                    "p95_ms": round(sorted(values)[min(len(values) - 1, int(len(values) * 0.95))], 3),
                })
            return {"counters": counters, "latency": latency, "generated_at": time.time()}


metrics = MetricsRegistry()
