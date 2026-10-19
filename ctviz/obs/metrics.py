"""In-process metrics for ``GET /v1/metrics``: counters plus latency ring buffers (p50/p95/p99)."""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from typing import Any

RING_SIZE = 1000


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (``q`` in 0..100); ``None`` for no data."""
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round(q / 100 * len(ordered) + 0.5) - 1))
    return round(ordered[idx], 1)


class Metrics:
    """Thread-safe counters and per-key latency samples."""

    def __init__(self, ring_size: int = RING_SIZE):
        self._lock = threading.Lock()
        self.started = time.time()
        self.counters: Counter[str] = Counter()
        self._lat: dict[str, deque[float]] = {}
        self._ring = ring_size

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] += n

    def observe(self, key: str, ms: float) -> None:
        with self._lock:
            self._lat.setdefault(key, deque(maxlen=self._ring)).append(ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            lat = {k: list(v) for k, v in self._lat.items()}
            counters = dict(self.counters)
        return {
            "uptime_s": round(time.time() - self.started, 1),
            "counters": counters,
            "latency_ms": {
                k: {"count": len(v), "p50": percentile(v, 50), "p95": percentile(v, 95), "p99": percentile(v, 99)}
                for k, v in sorted(lat.items())
            },
        }

    def reset(self) -> None:
        with self._lock:
            self.counters.clear()
            self._lat.clear()


METRICS = Metrics()
