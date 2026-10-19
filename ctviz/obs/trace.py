"""Per-request trace collected through a contextvar, so deep code can record timings and calls."""

from __future__ import annotations

import contextlib
import contextvars
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ctviz.api.schemas.response import LLMCallInfo, StepTiming, Trace


@dataclass
class RequestTrace:
    started: float = field(default_factory=time.perf_counter)
    steps: list[StepTiming] = field(default_factory=list)
    llm_calls: list[LLMCallInfo] = field(default_factory=list)
    llm_reserved: int = 0
    ctgov_calls: int = 0
    ctgov_hits: int = 0
    ctgov_misses: int = 0
    ctgov_urls: list[str] = field(default_factory=list)
    ctgov_calls_log: list[dict[str, Any]] = field(default_factory=list)  # url, cached, ms, status, cache_file

    @contextlib.contextmanager
    def step(self, name: str, **detail: Any) -> Iterator[dict[str, Any]]:
        """Time a block. The yielded dict can be mutated to add detail or set 'status'."""
        t0 = time.perf_counter()
        info: dict[str, Any] = dict(detail)
        status = "ok"
        try:
            yield info
        except Exception:
            status = "error"
            raise
        finally:
            status = info.pop("status", status)
            self.steps.append(
                StepTiming(name=name, ms=round((time.perf_counter() - t0) * 1000, 1), status=status, detail=info)
            )

    def add_llm(self, call: LLMCallInfo) -> None:
        self.llm_calls.append(call)

    @property
    def llm_cost(self) -> float:
        return sum(c.cost_usd for c in self.llm_calls)

    def to_model(self) -> Trace:
        return Trace(
            total_ms=round((time.perf_counter() - self.started) * 1000, 1),
            steps=self.steps,
            llm_calls=self.llm_calls,
            ctgov_calls=self.ctgov_calls,
            cost_usd=round(self.llm_cost, 6),
        )


_current: contextvars.ContextVar[RequestTrace | None] = contextvars.ContextVar("ctviz_trace", default=None)


def current_trace() -> RequestTrace:
    """The active request trace, or a throwaway one outside requests (tests, CLI)."""
    tr = _current.get()
    if tr is None:
        tr = RequestTrace()
        _current.set(tr)
    return tr


@contextlib.contextmanager
def trace_scope() -> Iterator[RequestTrace]:
    tr = RequestTrace()
    token = _current.set(tr)
    try:
        yield tr
    finally:
        _current.reset(token)
