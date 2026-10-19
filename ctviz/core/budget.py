"""Per-request budgets: LLM calls, agent tool calls, cost and wall-clock deadline.

A :class:`RequestBudget` lives in a contextvar (like the trace) so any agent can check or charge it.
Limits come from :class:`~ctviz.config.Settings` (``max_llm_calls``, ``request_cost_cap_usd``,
``request_deadline_s``); the tool-call cap (6 per request) is the design budget from CLAUDE.md.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import time
from collections.abc import Awaitable, Iterator
from dataclasses import dataclass, field

from ctviz.config import Settings
from ctviz.obs.trace import current_trace

MAX_TOOL_CALLS = 6


class BudgetExceeded(RuntimeError):
    """Raised when an agent tries to spend past a hard per-request limit."""


@dataclass
class RequestBudget:
    max_llm_calls: int = 4
    max_tool_calls: int = MAX_TOOL_CALLS
    cost_cap_usd: float = 0.05
    deadline_s: float = 20.0
    started: float = field(default_factory=time.monotonic)
    tool_calls: int = 0

    @classmethod
    def from_settings(cls, settings: Settings) -> RequestBudget:
        return cls(
            max_llm_calls=settings.max_llm_calls,
            cost_cap_usd=settings.request_cost_cap_usd,
            deadline_s=settings.request_deadline_s,
        )

    # LLM usage is recorded on the trace by the LLM client; read it from there.
    @property
    def llm_calls(self) -> int:
        tr = current_trace()
        return max(tr.llm_reserved, len(tr.llm_calls))  # reservations count before the call returns

    @property
    def cost_usd(self) -> float:
        return current_trace().llm_cost

    def remaining_s(self) -> float:
        return max(0.0, self.deadline_s - (time.monotonic() - self.started))

    def can_call_llm(self, n: int = 1) -> bool:
        return self.llm_calls + n <= self.max_llm_calls and self.cost_usd < self.cost_cap_usd and self.remaining_s() > 0

    def can_call_tool(self, n: int = 1) -> bool:
        return self.tool_calls + n <= self.max_tool_calls and self.remaining_s() > 0

    def charge_tool(self, n: int = 1) -> None:
        """Record ``n`` agent tool calls, raising :class:`BudgetExceeded` past the cap or deadline."""
        if not self.can_call_tool(n):
            raise BudgetExceeded(f"tool budget exhausted ({self.tool_calls}/{self.max_tool_calls} used)")
        self.tool_calls += n

    def timeout_for(self, default_s: float) -> float:
        """Per-call timeout that never runs past the request deadline (minimum 0.1 s)."""
        return max(0.1, min(default_s, self.remaining_s()))


_current: contextvars.ContextVar[RequestBudget | None] = contextvars.ContextVar("ctviz_budget", default=None)


def current_budget() -> RequestBudget:
    """The active request budget, or a default one outside requests (tests, CLI)."""
    b = _current.get()
    return b if b is not None else RequestBudget()  # fresh and NOT stored: no leakage across requests


def active_budget() -> RequestBudget | None:
    """The budget of the request in progress, or None outside a request (CLI warmers, tests)."""
    return _current.get()


@contextlib.contextmanager
def budget_scope(settings: Settings | None = None) -> Iterator[RequestBudget]:
    """Install a fresh budget for the duration of one request."""
    b = RequestBudget.from_settings(settings) if settings else RequestBudget()
    token = _current.set(b)
    try:
        yield b
    finally:
        _current.reset(token)


def _consume(task: asyncio.Future) -> None:
    if not task.cancelled():
        task.exception()  # retrieved: an abandoned task's late failure is not logged as "never retrieved"


async def await_within[T](aw: Awaitable[T], timeout: float) -> T:
    """Await ``aw`` for at most ``timeout`` seconds, then raise :class:`TimeoutError`.

    Unlike :func:`asyncio.wait_for`, this returns AT the timeout even when the awaited work is slow to honour
    cancellation (shielded connection cleanup, a gather still unwinding, a retry sleep that catches the
    cancel): the work is cancelled and left to finish on its own instead of being waited for. Use it for
    every await bounded by the request deadline.
    """
    task = asyncio.ensure_future(aw)
    try:
        done, _ = await asyncio.wait({task}, timeout=max(0.0, timeout))
    except asyncio.CancelledError:  # the caller itself is cancelled: cancel the work too
        task.cancel()
        task.add_done_callback(_consume)
        raise
    if task in done:
        return task.result()
    task.cancel()
    task.add_done_callback(_consume)
    raise TimeoutError(f"not finished within {timeout:.1f} s")
