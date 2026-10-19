"""Async token bucket used for outbound ClinicalTrials.gov politeness (and inbound limits)."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable


class LimiterWaitTooLong(TimeoutError):
    """A token (or concurrency slot) would not be available within the caller's ``max_wait``."""

    def __init__(self, wait_s: float, max_wait_s: float):
        super().__init__(f"rate limiter needs {wait_s:.1f} s; only {max_wait_s:.1f} s may be waited")
        self.wait_s = wait_s


class AsyncTokenBucket:
    """Classic token bucket. ``acquire`` sleeps until a token is available.

    ``penalize`` imposes a global cooldown (after a 429), and ``adapt`` scales the refill rate
    (halved on 429, slowly restored by the caller).
    """

    def __init__(self, rate_per_s: float, capacity: int, clock: Callable[[], float] = time.monotonic):
        self.base_rate = rate_per_s
        self.rate = rate_per_s
        self.capacity = capacity
        self._tokens = float(capacity)
        self._clock = clock
        self._last = clock()
        self._blocked_until = 0.0
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        if now < self._last:  # still inside a penalty cooldown
            return
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    def try_acquire(self) -> tuple[bool, float]:
        """Non-blocking attempt. Returns (acquired, seconds_until_next_token)."""
        self._refill()
        now = self._clock()
        if now < self._blocked_until:
            return False, self._blocked_until - now
        if self._tokens >= 1:
            self._tokens -= 1
            return True, 0.0
        return False, (1 - self._tokens) / self.rate

    async def acquire(self, max_wait: float = math.inf) -> None:
        """Wait for a token, at most ``max_wait`` seconds in total (queueing behind other waiters included).

        Raises :class:`LimiterWaitTooLong` as soon as the wait is known to exceed ``max_wait`` (e.g. a 429
        cooldown longer than the request has left), instead of sleeping until the caller's deadline.
        """
        limit = self._clock() + max_wait
        try:
            async with asyncio.timeout(None if math.isinf(max_wait) else max(0.0, max_wait)):
                async with self._lock:
                    while True:
                        ok, wait = self.try_acquire()
                        if ok:
                            return
                        if self._clock() + wait > limit:
                            raise LimiterWaitTooLong(wait, max_wait)
                        await asyncio.sleep(wait)
        except LimiterWaitTooLong:
            raise
        except TimeoutError as exc:  # timed out queueing behind other waiters
            raise LimiterWaitTooLong(max_wait, max_wait) from exc

    def penalize(self, seconds: float) -> None:
        """Block all acquisitions for ``seconds`` and drain the bucket so no burst follows the cooldown."""
        self._refill()
        self._blocked_until = max(self._blocked_until, self._clock() + seconds)
        self._tokens = 0.0
        self._last = self._blocked_until

    def adapt(self, factor: float) -> None:
        """Scale the refill rate (0.5 after a 429; ~1.05 after successes to recover to base_rate)."""
        self.rate = max(self.base_rate * 0.1, min(self.base_rate, self.rate * factor))


class OutboundLimiter:
    """Token bucket + concurrency cap. Use as ``async with limiter:`` around each HTTP call."""

    def __init__(self, rpm: int, burst: int, concurrency: int):
        self.bucket = AsyncTokenBucket(rpm / 60.0, burst)
        self.sem = asyncio.Semaphore(concurrency)

    async def __aenter__(self) -> OutboundLimiter:
        await self.bucket.acquire()
        await self.sem.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.sem.release()

    async def enter(self, max_wait: float = math.inf) -> None:
        """Like ``async with limiter`` (pair with :meth:`release`), but waits at most ``max_wait`` s for the token
        and the concurrency slot together; raises :class:`LimiterWaitTooLong` otherwise."""
        t0 = time.monotonic()
        await self.bucket.acquire(max_wait)
        left = max_wait - (time.monotonic() - t0)
        try:
            async with asyncio.timeout(None if math.isinf(left) else max(0.0, left)):
                await self.sem.acquire()
        except TimeoutError as exc:
            raise LimiterWaitTooLong(max_wait, max_wait) from exc

    def release(self) -> None:
        self.sem.release()
