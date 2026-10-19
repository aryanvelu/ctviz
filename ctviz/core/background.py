"""Fire-and-forget background work (cache writes, study store, graph ingest) that never blocks a response.

Tasks are tracked so they are not garbage-collected mid-flight and so shutdown, the CLI and tests can
``await drain()`` before closing the database.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

log = logging.getLogger(__name__)

_TASKS: set[asyncio.Task[Any]] = set()
MAX_PENDING = 64  # beyond this, low-priority work is dropped instead of queued
dropped = 0


def spawn(
    coro: Coroutine[Any, Any, Any], name: str = "background", *, low_priority: bool = False
) -> asyncio.Task[Any] | None:
    """Schedule ``coro`` on the running loop; failures are logged, never raised to the caller.

    Low-priority work (study-store / graph writes that are pure optimizations) is dropped when
    ``MAX_PENDING`` tasks are already in flight, so a burst cannot pile up unbounded background work.
    """
    global dropped
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # no loop (sync context): drop the work rather than block
        coro.close()
        return None
    if low_priority and pending() >= MAX_PENDING:
        coro.close()
        dropped += 1
        log.info("background queue full; dropped %s", name)
        return None

    async def guarded() -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # background writes are best-effort
            log.warning("background task %s failed: %s", name, type(exc).__name__)

    task = loop.create_task(guarded(), name=f"ctviz-{name}")
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return task


def pending() -> int:
    """Number of background tasks still running."""
    return sum(1 for t in _TASKS if not t.done())


async def drain(timeout_s: float = 10.0) -> None:
    """Wait for background tasks on the current loop (bounded by ``timeout_s``)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        tasks = [t for t in _TASKS if not t.done() and t.get_loop() is loop]
        if not tasks:
            return
        remaining = deadline - loop.time()
        if remaining <= 0:
            return
        await asyncio.wait(tasks, timeout=remaining)
