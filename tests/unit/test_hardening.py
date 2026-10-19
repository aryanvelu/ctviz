"""Hardening: test storage isolation, 429 caps, outage vs not-found, request validation, request ids,
citation re-verification and non-blocking reads."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
import types

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

import ctviz.ctgov.client as client_mod
from ctviz.api.schemas.plan import Intent, Persona, Scope
from ctviz.api.schemas.request import VisualizeRequest
from ctviz.api.schemas.response import Coverage, Interpretation, ResponseMeta, VisualizeResponse
from ctviz.app import create_app
from ctviz.config import PROJECT_ROOT, Settings
from ctviz.core.budget import budget_scope
from ctviz.ctgov.client import CTGovClient, CTGovError, CTGovUnavailable, parse_retry_after
from ctviz.store import db as db_mod
from ctviz.store.db import Database
from ctviz.store.response_cache import put_citations
from ctviz.viz.selector import empty_kpi
from tests.conftest import load_study_raw

BASE = "https://clinicaltrials.gov/api/v2"
NCT = "NCT03548935"


@pytest.fixture
def fast_sleep(monkeypatch):
    """Make the client's backoff sleeps instant (records requested delays)."""
    real_sleep = asyncio.sleep
    slept: list[float] = []

    async def fake(delay, *a, **k):
        slept.append(delay)
        await real_sleep(0)

    proxy = types.SimpleNamespace(**{n: getattr(asyncio, n) for n in dir(asyncio) if not n.startswith("__")})
    proxy.sleep = fake
    monkeypatch.setattr(client_mod, "asyncio", proxy)
    return slept


@pytest.fixture
async def live(tmp_path):
    """An online client over respx (the default httpx transport is mocked by respx.mock)."""
    s = Settings(offline=False, db_path=tmp_path / "h.db", ctgov_rpm=600, ctgov_burst=100, _env_file=None)  # type: ignore[call-arg]
    db = Database(s.db_path).connect()
    c = CTGovClient(s, db, transport=httpx.AsyncHTTPTransport())
    yield c
    await c.aclose()
    db.close()


# -- (1) tests never write under the project's .cache/ ----------------------------------------------
def test_settings_in_tests_never_point_at_project_storage(tmp_path):
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert not s.cache_dir.resolve().is_relative_to((PROJECT_ROOT / ".cache").resolve())
    assert s.db_path.resolve() != (PROJECT_ROOT / "ctviz.db").resolve()
    assert s.cache_dir.is_relative_to(tmp_path) and s.db_path.is_relative_to(tmp_path)


async def test_readable_copy_lands_in_tmp(live, tmp_path):
    with respx.mock() as router:
        router.get(f"{BASE}/studies/{NCT}").mock(return_value=httpx.Response(200, json=load_study_raw(NCT)))
        await live.get_study(NCT)
    assert live.settings.cache_dir.is_relative_to(tmp_path)
    assert list(live.settings.cache_dir.glob("*.json"))  # the readable copy was written, under tmp_path


# -- (2) 429: capped cooldown, fail fast past the request budget ------------------------------------
async def test_huge_retry_after_fails_fast_and_caps_cooldown(live, fast_sleep):
    with respx.mock() as router:
        route = router.get(f"{BASE}/studies/{NCT}").mock(
            return_value=httpx.Response(429, headers={"Retry-After": "600"}, text="<html>slow down</html>")
        )
        t0 = time.monotonic()
        with pytest.raises(CTGovUnavailable) as ei:
            await live.get_study(NCT)
    assert time.monotonic() - t0 < 2 and route.call_count == 1 and fast_sleep == []
    assert ei.value.status == 429
    bucket = live.limiter.bucket
    assert 0 < bucket._blocked_until - bucket._clock() <= client_mod.MAX_RETRY_WAIT_S + 0.01  # not 600 s


async def test_retry_after_beyond_request_budget_fails_fast(tmp_path, live, fast_sleep):
    s = Settings(request_deadline_s=2.0, _env_file=None)  # type: ignore[call-arg]
    with respx.mock() as router, budget_scope(s):
        route = router.get(f"{BASE}/studies/{NCT}").mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "10"}),
                httpx.Response(200, json=load_study_raw(NCT)),
            ]
        )
        with pytest.raises(CTGovUnavailable):
            await live.get_study(NCT)
    assert route.call_count == 1 and fast_sleep == []


async def test_short_retry_after_within_budget_still_retries(live, fast_sleep):
    with respx.mock() as router:
        route = router.get(f"{BASE}/studies/{NCT}").mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "2"}),
                httpx.Response(200, json=load_study_raw(NCT)),
            ]
        )
        live.limiter.bucket.penalize = lambda seconds: None  # keep the test instant
        raw = await live.get_study(NCT)
    assert route.call_count == 2 and raw["protocolSection"]["identificationModule"]["nctId"] == NCT
    assert fast_sleep and 2 <= fast_sleep[0] <= client_mod.MAX_RETRY_WAIT_S


def test_parse_retry_after():
    assert parse_retry_after("7") == 7.0
    assert parse_retry_after("1.5") == 1.5
    assert parse_retry_after(None) is None and parse_retry_after("soon") is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", now=1445412470.0) == 10.0


# -- (3) /version failures are not cached forever ---------------------------------------------------
async def test_version_failure_is_retried_after_short_ttl(live, monkeypatch):
    with respx.mock() as router:
        route = router.get(f"{BASE}/version").mock(
            side_effect=[httpx.Response(404), httpx.Response(200, json={"apiVersion": "2.0.5", "dataTimestamp": "x"})]
        )
        assert await live.version() == {}
        assert await live.version() == {} and route.call_count == 1  # brief negative cache
        monkeypatch.setattr(live, "_version_expires", 0.0)  # the failure TTL elapsed
        assert (await live.version())["apiVersion"] == "2.0.5" and route.call_count == 2
        assert (await live.version())["apiVersion"] == "2.0.5" and route.call_count == 2  # success is cached


# -- (4) outage is distinguishable from "not found" ------------------------------------------------
async def test_get_studies_outage_raises_unavailable_with_partial(live, fast_sleep):
    other = "NCT07654361"
    with respx.mock() as router:
        router.get(f"{BASE}/studies/{NCT}").mock(return_value=httpx.Response(503))
        router.get(f"{BASE}/studies/{other}").mock(return_value=httpx.Response(200, json=load_study_raw(other)))
        with pytest.raises(CTGovUnavailable) as ei:
            await live.get_studies([NCT, other])
    assert [p["protocolSection"]["identificationModule"]["nctId"] for p in ei.value.partial] == [other]
    assert isinstance(ei.value, CTGovError)  # existing `except CTGovError` handlers still catch it


async def test_get_studies_connect_errors_raise_unavailable(live, fast_sleep):
    with respx.mock() as router:
        route = router.get(f"{BASE}/studies/{NCT}").mock(side_effect=httpx.ConnectError("down"))
        with pytest.raises(CTGovUnavailable):
            await live.get_studies([NCT])
    assert route.call_count == client_mod.MAX_ATTEMPTS


async def test_get_studies_real_404_is_skipped_not_unavailable(live):
    with respx.mock() as router:
        router.get(f"{BASE}/studies/{NCT}").mock(return_value=httpx.Response(404, json={"error": "not found"}))
        assert await live.get_studies([NCT]) == []


# -- (5) request validation ----------------------------------------------------------------------
@pytest.mark.parametrize("field", ["drug_names", "conditions", "trial_phase", "status"])
def test_request_list_items_are_length_capped(field):
    with pytest.raises(ValidationError):
        VisualizeRequest(query="q", **{field: ["x" * 100_000]})


@pytest.mark.parametrize("field", ["conditions", "medications"])
def test_patient_profile_items_capped(field):
    with pytest.raises(ValidationError):
        VisualizeRequest(query="q", patient_profile={field: ["y" * 201]})
    with pytest.raises(ValidationError):
        VisualizeRequest(query="q", patient_profile={"notes": "z" * 1001})


def test_nct_ids_are_normalized_before_pattern_check():
    r = VisualizeRequest(query="q", nct_id=" nct03548935 ", nct_ids=["nct03548935", "Nct07654361"])
    assert r.nct_id == NCT and r.nct_ids == [NCT, "NCT07654361"]
    for bad in ({"nct_id": "NCT123"}, {"nct_ids": ["../version"]}):
        with pytest.raises(ValidationError):
            VisualizeRequest(query="q", **bad)


def test_patient_profile_forbids_unknown_fields():
    with pytest.raises(ValidationError, match="medication"):
        VisualizeRequest(query="q", patient_profile={"age": 40, "medication": ["metformin"]})


def test_list_items_are_stripped_and_200_chars_ok():
    r = VisualizeRequest(query="q", drug_names=["  semaglutide "], conditions=["c" * 200])
    assert r.drug_names == ["semaglutide"] and len(r.conditions[0]) == 200


# -- (6) X-Request-ID == meta.request_id; (7) citations re-verified -----------------------------------
class _Orch:
    def __init__(self, db):
        self.db, self.client, self.graph = db, None, None

    async def run(self, request: VisualizeRequest) -> VisualizeResponse:
        interp = Interpretation(
            intent=Intent.PHASE_DISTRIBUTION,
            intent_confidence=1,
            persona=Persona.RESEARCHER,
            scope=Scope.COHORT,
            parser="rules",
        )
        return VisualizeResponse(
            request_id="f" * 32,
            created_at="2026-10-18T00:00:00+00:00",
            visualization=empty_kpi("t", "m"),
            meta=ResponseMeta(interpretation=interp, coverage=Coverage(studies_matched_total=0, studies_analyzed=0)),
        )


@pytest.fixture
def api(settings, tmp_path):
    db = Database(tmp_path / "api.db").connect()
    s = settings.model_copy(update={"openai_api_key": SecretStr("sk-test-" + "x" * 24)})
    with TestClient(create_app(settings=s, orchestrator=_Orch(db))) as tc:
        yield tc, db
    db.close()


def test_request_id_header_matches_body(api):
    tc, _ = api
    r = tc.post("/v1/visualize", json={"query": "phases"}, headers={"X-Request-ID": "client-supplied"})
    assert r.status_code == 200
    assert r.headers["x-request-id"] == r.json()["meta"]["request_id"] == "f" * 32
    assert tc.get("/v1/health", headers={"X-Request-ID": "probe-1"}).headers["x-request-id"] == "probe-1"


def test_citations_endpoint_reverifies_against_stored_record(api):
    tc, db = api
    raw = load_study_raw(NCT)
    tc.portal.call(db.put_studies, [(NCT, "full", None, raw)])
    url = f"https://clinicaltrials.gov/study/{NCT}"
    good = {"nct_id": NCT, "excerpt": NCT, "field_path": "protocolSection.identificationModule.nctId", "url": url}
    tampered = {**good, "excerpt": "NCT99999999"}
    missing = {**good, "nct_id": "NCT00000001", "excerpt": "NCT00000001"}
    derived = {
        "nct_id": "",
        "excerpt": "42",
        "field_path": "$.totalCount",
        "kind": "derived_count",
        "url": "https://clinicaltrials.gov/search?cond=x",
    }
    malformed = {"nct_id": NCT}
    rid = "e" * 32
    tc.portal.call(put_citations, db, rid, {"d1": [good, tampered, missing, derived, malformed]}, 60)
    body = tc.get(f"/v1/responses/{rid}/datums/d1/citations").json()
    assert [c["verified"] for c in body["citations"]] == [True, False, False, True, False]
    assert body["total"] == 5 and body["verified"] == 2 and body["unverified"] == 3


# -- (8) reads are not blocked by a long write -------------------------------------------------------
async def test_read_not_blocked_by_long_write(tmp_path):
    db = Database(tmp_path / "rw.db").connect()
    await db.cache_set("http", "k", {"v": 1}, 60)
    holding, release = threading.Event(), threading.Event()

    def long_write() -> None:
        with db.lock:  # simulate a multi-second ingest holding the writer connection mid-transaction
            db.conn.execute("INSERT OR REPLACE INTO cache VALUES ('http','k2',x'00',0,0)")
            holding.set()
            release.wait(5)
            db.conn.commit()

    t = threading.Thread(target=long_write)
    t.start()
    holding.wait(5)
    try:
        t0 = time.perf_counter()
        assert await asyncio.wait_for(db.cache_get("http", "k"), 2) == {"v": 1}
        assert time.perf_counter() - t0 < 1.0
    finally:
        release.set()
        t.join()
    db.close()


async def test_reader_connection_is_query_only(tmp_path):
    db = Database(tmp_path / "ro.db").connect()

    def sneaky_write() -> None:
        db.conn.execute("DELETE FROM cache")

    with pytest.raises(sqlite3.OperationalError):
        await db.run(sneaky_write)
    db.close()


async def test_memory_db_falls_back_to_writer(tmp_path):
    db = Database(":memory:").connect()
    await db.cache_set("plan", "a", [1], 60)
    assert await db.cache_get("plan", "a") == [1]
    db.close()


async def test_put_studies_commits_in_chunks(tmp_path, monkeypatch):
    db = Database(tmp_path / "chunks.db").connect()
    monkeypatch.setattr(db_mod, "PUT_STUDIES_CHUNK", 2)
    calls: list[int] = []
    real = db.put_studies_sync
    monkeypatch.setattr(db, "put_studies_sync", lambda recs: (calls.append(len(recs)), real(recs))[1])
    await db.put_studies([(f"NCT{i:08d}", "projected", None, {"i": i}) for i in range(5)])
    assert calls == [2, 2, 1]
    assert len(await db.get_studies([f"NCT{i:08d}" for i in range(5)])) == 5
    db.close()
