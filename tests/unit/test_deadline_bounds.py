"""Regression: a request never runs (much) past ``request_deadline_s``, whatever ClinicalTrials.gov does.

Live symptom: under 429 storms requests took 546 s / 902 s before returning DEGRADED. Causes covered here:
the shared limiter queued calls through a 429 cooldown longer than the request had left, and
``asyncio.wait_for`` waited for cancelled work that was slow to unwind (e.g. shielded connection cleanup).
Everything is offline (mock transport, stub LLM, rule planner).
"""

from __future__ import annotations

import asyncio
import math
import time

import httpx
import pytest

from ctviz.agents.llm import StubLLM
from ctviz.api.schemas.request import VisualizeRequest
from ctviz.api.schemas.response import NoteCode
from ctviz.core.background import drain
from ctviz.core.budget import await_within
from ctviz.core.orchestrator import RUN_GRACE_S, Orchestrator
from ctviz.ctgov.client import CTGovClient
from ctviz.ctgov.limiter import AsyncTokenBucket, LimiterWaitTooLong, OutboundLimiter
from ctviz.store.db import Database

DEADLINE_S = 2.0
SLACK_S = 1.0
QUESTIONS = [
    "Show the trial sites (facility and city) in United States for Donanemab trials",  # geo_map, site level
    "How many patients dropped out of semaglutide trials and why?",  # sankey / dropout (results path)
    "Participant flow for NCT03548935",  # single trial fetch
]


async def _too_many(_req: httpx.Request) -> httpx.Response:
    return httpx.Response(429, headers={"Retry-After": "20"}, text="<html>Too Many Requests</html>")


async def _slow_and_stubborn(_req: httpx.Request) -> httpx.Response:
    """Never answers; on cancellation keeps cleaning up for 3 s (like a shielded connection close). Waiting for
    that cleanup (``asyncio.wait_for``) would end past DEADLINE_S + SLACK_S."""
    try:
        await asyncio.sleep(60)
    except asyncio.CancelledError:
        await asyncio.shield(asyncio.sleep(3))
        raise
    return httpx.Response(200, json={})


@pytest.fixture
async def make_orch(settings):
    made: list[tuple[CTGovClient, Database]] = []

    def make(handler) -> tuple[Orchestrator, CTGovClient]:  # noqa: ANN001
        s = settings.model_copy(update={"request_deadline_s": DEADLINE_S})
        db = Database(s.db_path).connect()
        client = CTGovClient(s, db, transport=httpx.MockTransport(handler))
        made.append((client, db))
        return Orchestrator(s, db, client, StubLLM(), runs_dir=None, use_response_cache=False), client

    yield make
    for client, _ in made:
        await client.aclose()
    await drain(1.0)
    for _, db in made:
        db.close()


async def _timed(o: Orchestrator, q: str):
    t0 = time.monotonic()
    resp = await o.run(VisualizeRequest(query=q))
    return resp, time.monotonic() - t0


@pytest.mark.parametrize("question", QUESTIONS)
async def test_429_storm_is_bounded_and_degraded(make_orch, question):
    o, _ = make_orch(_too_many)
    resp, wall = await _timed(o, question)
    assert wall <= DEADLINE_S + SLACK_S, wall
    assert any(n.code is NoteCode.DEGRADED for n in resp.meta.notes)


async def test_cooldown_from_another_request_fails_fast(make_orch):
    """A 429 cooldown (set by an earlier request) longer than the deadline: fail now, don't sleep it out."""
    o, client = make_orch(_too_many)
    client.limiter.bucket.penalize(25)
    resp, wall = await _timed(o, QUESTIONS[0])
    assert wall < DEADLINE_S / 2, wall
    msgs = [n.message for n in resp.meta.notes if n.code is NoteCode.DEGRADED]
    assert any("CTGovUnavailable" in m for m in msgs), msgs


@pytest.mark.parametrize("question", QUESTIONS)
async def test_slow_responses_that_ignore_cancellation_are_bounded(make_orch, question):
    o, _ = make_orch(_slow_and_stubborn)
    resp, wall = await _timed(o, question)
    assert wall <= DEADLINE_S + SLACK_S, wall
    assert any(n.code is NoteCode.DEGRADED for n in resp.meta.notes)


async def test_hard_cap_when_an_unbounded_step_hangs(make_orch, monkeypatch):
    """Even a step with no timeout of its own cannot hold the request past deadline + grace."""

    async def hang(*_a, **_k):
        while True:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                await asyncio.shield(asyncio.sleep(5))  # unwinds slowly
                raise

    o, _ = make_orch(_too_many)
    monkeypatch.setattr(Orchestrator, "_verify_entities", hang)
    resp, wall = await _timed(o, QUESTIONS[1])
    assert wall <= DEADLINE_S + RUN_GRACE_S + SLACK_S, wall
    msgs = [n.message for n in resp.meta.notes if n.code is NoteCode.DEGRADED]
    assert any("deadline" in m for m in msgs), msgs


# -- units -------------------------------------------------------------------------------------------
async def test_bucket_acquire_fails_fast_when_cooldown_exceeds_max_wait():
    b = AsyncTokenBucket(1.0, 1)
    b.penalize(30)
    t0 = time.monotonic()
    with pytest.raises(LimiterWaitTooLong):
        await b.acquire(max_wait=1.0)
    assert time.monotonic() - t0 < 0.2


async def test_bucket_acquire_waits_when_cooldown_fits():
    b = AsyncTokenBucket(10.0, 1)
    b.penalize(0.1)  # cooldown + one refill (0.1 s) fit in max_wait
    await b.acquire(max_wait=1.0)  # no raise
    await asyncio.wait_for(AsyncTokenBucket(1.0, 1).acquire(), 1.0)  # default: unbounded, as before


async def test_bucket_acquire_bounds_queueing_behind_other_waiters():
    b = AsyncTokenBucket(0.1, 1)  # one token now, the next in 10 s
    await b.acquire()
    first = asyncio.create_task(b.acquire())  # holds the lock while sleeping for the next token
    await asyncio.sleep(0)
    t0 = time.monotonic()
    with pytest.raises(LimiterWaitTooLong):
        await b.acquire(max_wait=0.2)
    assert time.monotonic() - t0 < 0.5
    first.cancel()


async def test_outbound_enter_bounds_semaphore_wait():
    lim = OutboundLimiter(rpm=6000, burst=10, concurrency=1)
    await lim.enter()
    with pytest.raises(LimiterWaitTooLong):
        await lim.enter(max_wait=0.1)
    lim.release()
    await lim.enter(max_wait=0.1)
    lim.release()
    await lim.enter(math.inf)
    lim.release()


async def test_await_within_does_not_wait_for_slow_cancellation():
    async def stubborn():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.shield(asyncio.sleep(5))
            raise

    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        await await_within(stubborn(), 0.1)
    assert time.monotonic() - t0 < 0.5
    assert await await_within(asyncio.sleep(0, result=7), 1.0) == 7
    with pytest.raises(ValueError):
        await await_within(_raise(), 1.0)


async def _raise():
    raise ValueError("boom")
