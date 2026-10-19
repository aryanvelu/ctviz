"""HTTP layer: routes, validation errors, inbound rate limiting, secrets never exposed."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ctviz.api.schemas.plan import Intent, Persona, Scope
from ctviz.api.schemas.request import VisualizeRequest
from ctviz.api.schemas.response import Coverage, Interpretation, ResponseMeta, VisualizeResponse
from ctviz.app import create_app
from ctviz.obs.metrics import METRICS
from ctviz.store.db import Database
from ctviz.store.response_cache import put_citations
from ctviz.viz.selector import empty_kpi

FAKE_KEY = "sk-test-" + "x" * 24


class FakeOrchestrator:
    def __init__(self, db):
        self.db = db
        self.client = None
        self.graph = None
        self.requests: list[VisualizeRequest] = []

    async def run(self, request: VisualizeRequest) -> VisualizeResponse:
        self.requests.append(request)
        interp = Interpretation(
            intent=Intent.PHASE_DISTRIBUTION,
            intent_confidence=1,
            persona=Persona.RESEARCHER,
            scope=Scope.COHORT,
            parser="rules",
        )
        return VisualizeResponse(
            request_id="a" * 32,
            created_at="2026-10-18T00:00:00+00:00",
            visualization=empty_kpi("t", "m"),
            meta=ResponseMeta(interpretation=interp, coverage=Coverage(studies_matched_total=0, studies_analyzed=0)),
        )


@pytest.fixture
def make_client(settings, tmp_path):
    clients = []

    def make(**overrides) -> tuple[TestClient, FakeOrchestrator]:
        s = settings.model_copy(update={"openai_api_key": SecretStr(FAKE_KEY), **overrides})
        db = Database(tmp_path / "api.db").connect()
        orch = FakeOrchestrator(db)
        tc = TestClient(create_app(settings=s, orchestrator=orch))
        tc.__enter__()
        clients.append((tc, db))
        return tc, orch

    yield make
    for tc, db in clients:
        tc.__exit__(None, None, None)
        db.close()


def test_health_never_reveals_key(make_client):
    tc, _ = make_client()
    r = tc.get("/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["offline"] is True and "llm_configured" in body
    assert FAKE_KEY not in r.text and "sk-" not in r.text
    assert r.headers["x-request-id"]


def test_schema_and_intents(make_client):
    tc, _ = make_client()
    s = tc.get("/v1/schema").json()
    assert "query" in s["request"]["properties"] and "visualization" in s["response"]["properties"]
    intents = tc.get("/v1/intents").json()["intents"]
    names = {i["intent"]: i for i in intents}
    assert len(names) == len(Intent)
    assert names["phase_distribution"]["registered"] is True
    assert names["phase_distribution"]["primary_visualization"] == "bar_chart"


def test_visualize_ok_exclude_none(make_client):
    tc, orch = make_client()
    r = tc.post(
        "/v1/visualize",
        json={"query": "phases of semaglutide trials", "drug_name": "semaglutide"},
        headers={"X-Request-ID": "abc-123"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # the header is the response's request_id (the id its citations are stored under), not the client's echo
    assert r.headers["x-request-id"] == body["meta"]["request_id"] == "a" * 32
    assert body["visualization"]["type"] == "kpi_cards"
    assert "answer_summary" not in body and "trace" not in body["meta"]  # None fields are omitted
    assert orch.requests[0].drug_name == "semaglutide"


def test_validation_error_422(make_client):
    tc, _ = make_client()
    r = tc.post("/v1/visualize", json={"query": "x", "trial_phase": ["PHASE9"], "bogus": 1})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_error" and body["detail"]
    assert tc.post("/v1/visualize", json={"query": ""}).status_code == 422


def test_inbound_rate_limit_429(make_client):
    METRICS.reset()
    tc, _ = make_client(inbound_rpm=1, inbound_burst=2, api_keys=[SecretStr("k1"), SecretStr("k2")])
    body = {"query": "phases"}
    codes = [tc.post("/v1/visualize", json=body, headers={"X-API-Key": "k1"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = tc.post("/v1/visualize", json=body, headers={"X-API-Key": "k1"})
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert r.json()["error"] == "rate_limited" and r.json()["retry_after_s"] > 0
    assert tc.post("/v1/visualize", json=body, headers={"X-API-Key": "k2"}).status_code == 200  # separate bucket
    assert tc.get("/v1/health").status_code == 200  # exempt
    snap = tc.get("/v1/metrics").json()
    assert snap["counters"]["rate_limited"] >= 2


def test_item6_unconfigured_api_key_shares_ip_bucket(make_client):
    tc, _ = make_client(inbound_rpm=1, inbound_burst=1, api_keys=[SecretStr("good")])
    assert tc.post("/v1/visualize", json={"query": "q"}, headers={"X-API-Key": "made-up-1"}).status_code == 200
    # A fresh, unknown key must not mint a new bucket: same client IP -> limited.
    assert tc.post("/v1/visualize", json={"query": "q"}, headers={"X-API-Key": "made-up-2"}).status_code == 429
    assert tc.post("/v1/visualize", json={"query": "q"}, headers={"X-API-Key": "good"}).status_code == 200


def test_item6_caller_key_uses_rightmost_forwarded_hop():
    from starlette.requests import Request

    from ctviz.api.middleware import caller_key

    def req(headers: dict[str, str]) -> Request:
        raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
        return Request({"type": "http", "headers": raw, "client": ("10.0.0.9", 1234), "method": "GET", "path": "/"})

    spoof = {"X-Forwarded-For": "1.2.3.4, 203.0.113.7"}
    assert caller_key(req(spoof), trust_proxy=True) == "ip:203.0.113.7"
    assert caller_key(req(spoof), trust_proxy=False) == "ip:10.0.0.9"
    assert caller_key(req({"X-API-Key": "k"}), api_keys=frozenset({"k"})).startswith("key:")
    assert caller_key(req({"X-API-Key": "k"}), api_keys=frozenset()) == "ip:10.0.0.9"


def test_inflight_cap_503(make_client):
    tc, _ = make_client(max_inflight=1)
    # Simulate one request already in flight by bumping the guard's counter.
    guard = tc.app.middleware_stack
    while guard is not None and type(guard).__name__ != "GuardMiddleware":
        guard = getattr(guard, "app", None)
    guard._inflight = 1
    r = tc.post("/v1/visualize", json={"query": "phases"})
    guard._inflight = 0
    assert r.status_code == 503 and r.json()["error"] == "overloaded"


def test_citations_endpoint(make_client):
    tc, orch = make_client()
    rid = "b" * 32
    tc.portal.call(put_citations, orch.db, rid, {"d1": [{"nct_id": "NCT1", "excerpt": "x"}]}, 60)
    r = tc.get(f"/v1/responses/{rid}/datums/d1/citations")
    assert r.status_code == 200 and r.json()["total"] == 1
    assert tc.get(f"/v1/responses/{rid}/datums/d9/citations").status_code == 404
    assert tc.get(f"/v1/responses/{'c' * 32}/datums/d1/citations").status_code == 404


def test_graph_unavailable_503(make_client):
    tc, _ = make_client()
    assert tc.get("/v1/graph/neighbors", params={"name": "semaglutide"}).status_code == 503
