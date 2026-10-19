"""LLM client contract used by every LLM-backed agent.

All agents call :meth:`LLMClient.structured`, which returns a validated Pydantic object (OpenAI
Responses API structured outputs) and records tokens/cost/latency in the request trace.
:class:`StubLLM` implements the same interface for offline mode and tests.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from ctviz.api.schemas.response import LLMCallInfo
from ctviz.config import Settings
from ctviz.obs.pricing import ALLOWED_MODELS, cost_usd
from ctviz.obs.trace import current_trace

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    """No key / offline / budget exhausted: callers must fall back to rules."""


class LLM(Protocol):
    async def structured(
        self,
        *,
        agent: str,
        model: str,
        system: str,
        user: str,
        schema: type[T],
        max_output_tokens: int = 800,
        effort: str | None = None,
        timeout_s: float | None = None,
    ) -> T: ...


class LLMClient:
    """OpenAI Responses API wrapper with structured outputs, timeouts, and accounting."""

    def __init__(self, settings: Settings):
        from openai import AsyncOpenAI

        self.settings = settings
        self._client = AsyncOpenAI(api_key=settings.require_llm_key(), max_retries=1)

    async def structured(
        self,
        *,
        agent: str,
        model: str,
        system: str,
        user: str,
        schema: type[T],
        max_output_tokens: int = 800,
        effort: str | None = None,
        timeout_s: float | None = None,
    ) -> T:
        if model not in ALLOWED_MODELS:
            raise LLMError(f"model {model!r} is not in the allowed model list")
        tr = current_trace()
        if tr.llm_reserved >= self.settings.max_llm_calls:
            raise LLMUnavailable("per-request LLM call budget exhausted")
        if tr.llm_cost >= self.settings.request_cost_cap_usd:
            raise LLMUnavailable("per-request cost cap reached")
        tr.llm_reserved += 1  # reserve before awaiting so concurrent agents can't overshoot the budget
        kwargs: dict[str, Any] = {
            "model": model,
            "input": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "text_format": schema,
            "max_output_tokens": max_output_tokens,
            "store": False,
            "prompt_cache_key": f"ctviz-{agent}",
        }
        if effort and model.startswith("gpt-5"):
            kwargs["reasoning"] = {"effort": effort}
        t0 = time.perf_counter()
        try:
            resp = await asyncio.wait_for(
                self._client.responses.parse(**kwargs), timeout=timeout_s or self.settings.llm_timeout_s
            )
        except TimeoutError as exc:
            tr.add_llm(LLMCallInfo(agent=f"{agent}:timeout", model=model, ms=(time.perf_counter() - t0) * 1000))
            raise LLMError(f"{agent}: LLM timed out") from exc
        except Exception as exc:  # openai.APIError and friends; never include request content
            tr.add_llm(LLMCallInfo(agent=f"{agent}:error", model=model, ms=(time.perf_counter() - t0) * 1000))
            raise LLMError(f"{agent}: LLM call failed ({type(exc).__name__})") from exc
        ms = (time.perf_counter() - t0) * 1000
        usage = getattr(resp, "usage", None)
        inp = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        details = getattr(usage, "input_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        cost = round(cost_usd(model, inp, cached, out), 6)
        tr.add_llm(
            LLMCallInfo(
                agent=agent,
                model=model,
                input_tokens=inp,
                cached_tokens=cached,
                output_tokens=out,
                ms=round(ms, 1),
                cost_usd=cost,
            )
        )
        parsed = getattr(resp, "output_parsed", None)
        if parsed is None:
            raise LLMError(f"{agent}: model returned no parsable output")
        return parsed


class StubLLM:
    """Offline stand-in. Register canned responders per agent; otherwise raise LLMUnavailable."""

    def __init__(self, responders: dict[str, Callable[[str, type[BaseModel]], BaseModel]] | None = None):
        self.responders = responders or {}

    async def structured(
        self,
        *,
        agent: str,
        model: str,
        system: str,
        user: str,
        schema: type[T],
        max_output_tokens: int = 800,
        effort: str | None = None,
        timeout_s: float | None = None,
    ) -> T:
        fn = self.responders.get(agent)
        if fn is None:
            raise LLMUnavailable(f"{agent}: offline mode (no LLM)")
        result = schema.model_validate(fn(user, schema).model_dump())
        tr = current_trace()
        tr.llm_reserved += 1
        tr.add_llm(LLMCallInfo(agent=agent, model="stub"))
        return result


def build_llm(settings: Settings) -> LLM:
    return LLMClient(settings) if settings.llm_enabled else StubLLM()
