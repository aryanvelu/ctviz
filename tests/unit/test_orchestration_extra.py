"""Extra orchestration guarantees: privacy, cache isolation, summary numbers, degradation over HTTP,
partition completeness, inbound limiting and secret hygiene. Everything is mocked (no network, no OpenAI).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import types
import uuid
import zlib
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ctviz.agents.llm import StubLLM
from ctviz.api.public import PublicResponse
from ctviz.api.schemas.plan import CohortFilter, Intent, Persona, QueryPlan, Scope
from ctviz.api.schemas.request import VisualizeRequest
from ctviz.api.schemas.response import NoteCode, VisualizeResponse
from ctviz.api.schemas.viz import Channel, Encoding, GraphData, VizType
from ctviz.app import create_app
from ctviz.config import Settings
from ctviz.core.background import drain
from ctviz.core.orchestrator import Orchestrator
from ctviz.core.provenance import Dataset
from ctviz.ctgov import client as client_mod
from ctviz.ctgov import limiter as limiter_mod
from ctviz.ctgov.client import CTGovClient
from ctviz.ctgov.fields import FieldSet
from ctviz.ctgov.retrieval import MISSING_START, fetch_cohort
from ctviz.obs.logging import JsonFormatter
from ctviz.store import response_cache as rc
from ctviz.store.db import Database
from ctviz.viz import base as viz_base
from ctviz.viz.base import HandlerResult, make_viz
from tests.conftest import load_cohort_raw
from tests.unit.test_api import FakeOrchestrator
from tests.unit.test_orchestrator import FakeClient, comparison_plan, fixed_planner

BASE = "https://clinicaltrials.gov/api/v2"
TRIAL = "NCT03548935"


# -- shared helpers --------------------------------------------------------------------------------
class MelanomaClient(FakeClient):
    """FakeClient that also serves the melanoma fixture (every record has every trial-table field)."""

    def _studies_for(self, spec) -> list[dict]:
        if "melanoma" in (spec.cond or "").lower():
            return load_cohort_raw("melanoma_ph2_3")["studies"]
        return super()._studies_for(spec)


def eligibility_plan() -> QueryPlan:
    return QueryPlan(
        intent=Intent.ELIGIBILITY_CHECK,
        intent_confidence=0.9,
        persona=Persona.PATIENT,
        scope=Scope.SINGLE_TRIAL,
        nct_ids=[TRIAL],
    )


async def profile_aware_eligibility(ctx):
    """Stand-in eligibility handler whose answer depends on the patient profile (but never echoes it)."""
    prof = ctx.request.patient_profile
    trial = ctx.trials[0]
    ok = prof is not None and prof.age is not None and prof.age >= 18
    ds = Dataset(prefix="e")
    ds.emit(
        {"label": "Adult age criterion", "value": "met" if ok else "not met", "kind": "text"}, [trial.ref("nct_id")]
    )
    viz = make_viz(
        VizType.KPI_CARDS,
        "Eligibility",
        Encoding(
            label=Channel(field="label", type="nominal", title="Criterion"),
            y=Channel(field="value", type="nominal", title="Result"),
        ),
        ds.rows,
    )
    return HandlerResult(visualization=viz, studies_analyzed=1, full_refs=ds.full_refs)


@pytest.fixture
async def oenv(settings, tmp_path):
    db = Database(tmp_path / "x.db").connect()
    client = MelanomaClient(db)
    yield settings, db, client
    await drain()
    db.close()


def orch(env, plan: QueryPlan | None = None, **kw) -> Orchestrator:
    settings, db, client = env
    kw.setdefault("runs_dir", None)
    return Orchestrator(settings, db, client, StubLLM(), planner=fixed_planner(plan) if plan else None, **kw)


def _cache_rows(db: Database) -> list[tuple[str, str, str]]:
    rows = db.conn.execute("SELECT ns, key, value FROM cache").fetchall()
    return [(ns, key, zlib.decompress(v).decode()) for ns, key, v in rows]


def _format_lines(records: list[logging.LogRecord]) -> list[str]:
    jf, plain = JsonFormatter(), logging.Formatter("%(levelname)s %(name)s %(message)s")
    out = []
    for r in records:
        out.append(jf.format(r))
        out.append(plain.format(r))  # includes any traceback text
    return out


# -- (1) response cache never crosses patient profiles -----------------------------------------------
async def test_response_cache_never_crosses_patient_profiles(oenv, monkeypatch):
    """Patient (profile) answers are never written to the response cache; researcher answers are."""
    monkeypatch.setitem(viz_base._HANDLERS, Intent.ELIGIBILITY_CHECK, profile_aware_eligibility)
    o = orch(oenv, eligibility_plan())
    adult = VisualizeRequest(query="Am I eligible?", nct_id=TRIAL, patient_profile={"age": 51, "sex": "female"})
    child = adult.model_copy(update={"patient_profile": adult.patient_profile.model_copy(update={"age": 12})})

    # caching is active for non-private requests, so the checks below are meaningful
    o_res = orch(
        oenv,
        QueryPlan(
            intent=Intent.SINGLE_TRIAL_OVERVIEW, intent_confidence=0.9, scope=Scope.SINGLE_TRIAL, nct_ids=[TRIAL]
        ),
    )
    researcher = VisualizeRequest(query="What is this trial?", nct_id=TRIAL, persona="researcher")
    await o_res.run(researcher)
    await drain()
    assert (await o_res.run(researcher)).meta.cache.response_hit
    await drain()

    first = await o.run(adult)
    await drain()
    assert first.visualization.data[0].model_extra["value"] == "met"
    again = await o.run(adult)
    assert not again.meta.cache.response_hit  # private: recomputed, never served from cache
    assert again.visualization.data[0].model_extra["value"] == "met"
    await drain()

    other = await o.run(child)
    assert not other.meta.cache.response_hit
    assert other.visualization.data[0].model_extra["value"] == "not met"
    await drain()
    no_profile = await o.run(adult.model_copy(update={"patient_profile": None}))
    assert not no_profile.meta.cache.response_hit
    assert (
        rc.response_key(adult)
        != rc.response_key(child)
        != rc.response_key(adult.model_copy(update={"patient_profile": None}))
    )


# -- (2) patient_profile never leaks into artifacts, logs or cache ---------------------------------
SENTINELS = ("zqxcondsentinel", "zqxmedsentinel", "zqxnotesentinel")


def _sentinel_request(**kw) -> VisualizeRequest:
    return VisualizeRequest(
        query="Am I eligible for this trial?",
        nct_id=TRIAL,
        persona="patient",
        patient_profile={
            "age": 51,
            "sex": "female",
            "conditions": [SENTINELS[0]],
            "medications": [SENTINELS[1]],
            "notes": SENTINELS[2],
        },
        **kw,
    )


@pytest.mark.parametrize("planner", ["fixed", "default"])
async def test_patient_profile_never_in_artifacts_logs_or_cache(oenv, tmp_path, monkeypatch, caplog, planner):
    monkeypatch.setitem(viz_base._HANDLERS, Intent.ELIGIBILITY_CHECK, profile_aware_eligibility)
    caplog.set_level(logging.DEBUG)
    runs = tmp_path / "runs"
    o = orch(oenv, eligibility_plan() if planner == "fixed" else None, runs_dir=runs)
    req = _sentinel_request()
    resp = await o.run(req)
    await drain()
    await o.run(req)  # cache-hit path too
    await drain()

    files = [p for p in runs.rglob("*") if p.is_file()]
    assert any(p.name == "request.json" for p in files) and (runs / resp.request_id / "response.json").exists()
    for p in files:
        text = p.read_text()
        for s in SENTINELS:
            assert s not in text, f"{s} leaked into {p.relative_to(runs)}"

    lines = _format_lines(caplog.records)
    assert lines, "expected step log lines"
    for line in lines:
        for s in SENTINELS:
            assert s not in line, f"{s} leaked into a log line: {line[:200]}"

    rows = _cache_rows(oenv[1])
    assert not any(ns == "response" for ns, _, _ in rows)  # private answers are never response-cached
    for ns, key, value in rows:
        for s in SENTINELS:
            assert s not in key, f"{s} in cache key ({ns})"
            assert s not in value, f"{s} stored in cache namespace {ns}"


# -- (3) every number in the deterministic summary is a chart/fact number --------------------------
_NCT = re.compile(r"NCT\d{8}", re.I)
_NUMTOK = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.,])\d+(?:\.\d+)?")


def _num_tokens(text: str) -> list[float]:
    return [float(t.replace(",", "")) for t in _NUMTOK.findall(_NCT.sub(" ", text))]


def _numbers_from(obj) -> set[float]:
    out: set[float] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("citations", "id", "citations_total"):
                continue
            out |= _numbers_from(k) | _numbers_from(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            out |= _numbers_from(v)
    elif isinstance(obj, bool) or obj is None:
        pass
    elif isinstance(obj, int | float):
        out.add(float(obj))
    elif isinstance(obj, str):
        out |= set(_num_tokens(obj))
    return out


def _sema_plan(intent: Intent, **kw) -> QueryPlan:
    return QueryPlan(
        intent=intent, intent_confidence=0.9, cohorts=[CohortFilter(label="Semaglutide", drugs=["semaglutide"], **kw)]
    )


SUMMARY_CASES = {
    "comparison": (comparison_plan(), VisualizeRequest(query="Compare phases for semaglutide vs tirzepatide")),
    "single_cohort": (
        QueryPlan(
            intent=Intent.PHASE_DISTRIBUTION,
            intent_confidence=0.9,
            cohorts=[CohortFilter(label="Melanoma", conditions=["melanoma"])],
        ),
        VisualizeRequest(query="melanoma trials by phase"),
    ),
    "trial_required": (
        QueryPlan(
            intent=Intent.ELIGIBILITY_CHECK,
            intent_confidence=0.8,
            persona=Persona.PATIENT,
            scope=Scope.SINGLE_TRIAL,
            cohorts=[CohortFilter(label="Melanoma", conditions=["melanoma"])],
        ),
        VisualizeRequest(query="Am I eligible for a melanoma trial?"),
    ),
    "zero_results": (
        QueryPlan(
            intent=Intent.PHASE_DISTRIBUTION,
            intent_confidence=0.9,
            cohorts=[CohortFilter(label="N", drugs=["nonexistium"], countries=["Atlantis"])],
        ),
        VisualizeRequest(query="phases of nonexistium trials in Atlantis"),
    ),
    "relaxed": (
        _sema_plan(Intent.PHASE_DISTRIBUTION, countries=["Atlantis"]),
        VisualizeRequest(query="semaglutide trials in Atlantis by phase"),
    ),
    "single_trial": (
        QueryPlan(
            intent=Intent.SINGLE_TRIAL_OVERVIEW, intent_confidence=0.9, scope=Scope.SINGLE_TRIAL, nct_ids=[TRIAL]
        ),
        VisualizeRequest(query="What is this trial?", nct_id=TRIAL),
    ),
    "missing_handler": (_sema_plan(Intent.SPONSOR_LANDSCAPE), VisualizeRequest(query="who sponsors these")),
    "default_planner": (
        None,
        VisualizeRequest(
            query="Compare phases for semaglutide vs tirzepatide", drug_names=["semaglutide", "tirzepatide"]
        ),
    ),
}


@pytest.mark.parametrize("case", list(SUMMARY_CASES))
async def test_summary_numbers_come_from_data_or_facts(oenv, case):
    plan, req = SUMMARY_CASES[case]
    resp, rec = await orch(oenv, plan).run_recorded(req)
    VisualizeResponse.model_validate(resp.model_dump(mode="json"))
    viz = resp.visualization
    if viz is None:  # clarification answers carry no chart and no numeric summary
        assert resp.status != "ok" and not resp.answer_summary
        return
    rows = viz.data.nodes + viz.data.edges if isinstance(viz.data, GraphData) else viz.data
    allowed = _numbers_from([r.model_dump(mode="json") for r in rows])
    allowed |= _numbers_from(rec.artifacts.get("handler_result", {}).get("facts", {}))
    tokens = _num_tokens(resp.answer_summary or "")
    unsupported = [t for t in tokens if t not in allowed]
    assert not unsupported, f"{case}: summary {resp.answer_summary!r} has numbers {unsupported} not in data/facts"


async def test_summary_number_check_is_not_vacuous(oenv):
    """At least one of the exercised intents produces a summary containing numbers."""
    plan, req = SUMMARY_CASES["comparison"]
    resp = await orch(oenv, plan).run(req)
    assert resp.answer_summary and _num_tokens(resp.answer_summary)
    facts_numbers = {
        float(v) for k, v in (await orch(oenv, plan).run_recorded(req))[1].artifacts["handler_result"]["facts"].items()
    }
    assert facts_numbers & set(_num_tokens(resp.answer_summary))


# -- (4) handler errors, handler timeouts and 429 storms over HTTP ---------------------------------
def _http(settings: Settings, client, plan: QueryPlan) -> TestClient:
    db = client.db
    o = Orchestrator(settings, db, client, StubLLM(), planner=fixed_planner(plan), runs_dir=None)
    tc = TestClient(create_app(settings=settings, orchestrator=o))
    return tc


def _status_plan(cond_or_drug: dict) -> QueryPlan:
    return QueryPlan(
        intent=Intent.STATUS_DISTRIBUTION, intent_confidence=0.9, cohorts=[CohortFilter(label="X", **cond_or_drug)]
    )


def _post_ok(tc: TestClient, query: str) -> PublicResponse:
    r = tc.post("/v1/visualize", json={"query": query})
    assert r.status_code == 200, r.text
    return PublicResponse.model_validate(r.json())


async def _boom(ctx):
    raise RuntimeError("bad data")


def test_http_handler_raising_returns_200_degraded_table(settings, tmp_path, monkeypatch):
    monkeypatch.setitem(viz_base._HANDLERS, Intent.STATUS_DISTRIBUTION, _boom)
    db = Database(tmp_path / "h.db").connect()
    with _http(settings, MelanomaClient(db), _status_plan({"conditions": ["melanoma"]})) as tc:
        resp = _post_ok(tc, "status of melanoma trials")
    db.close()
    assert resp.visualization.type is VizType.TABLE
    msgs = [n.message for n in resp.meta.notes if n.code is NoteCode.DEGRADED]
    assert any("could not be built" in m and "RuntimeError" in m for m in msgs), msgs


def test_http_handler_raising_fallback_table_with_missing_fields(settings, tmp_path, monkeypatch):
    monkeypatch.setitem(viz_base._HANDLERS, Intent.STATUS_DISTRIBUTION, _boom)
    db = Database(tmp_path / "h2.db").connect()
    with _http(settings, MelanomaClient(db), _status_plan({"drugs": ["tirzepatide"]})) as tc:
        resp = _post_ok(tc, "status of tirzepatide trials")
    db.close()
    assert resp.visualization.type is VizType.TABLE


def test_http_handler_timeout_returns_200_degraded(settings, tmp_path, monkeypatch):
    async def slow(ctx):
        await asyncio.sleep(30)

    monkeypatch.setitem(viz_base._HANDLERS, Intent.STATUS_DISTRIBUTION, slow)
    s = settings.model_copy(update={"request_deadline_s": 1.5})
    db = Database(tmp_path / "t.db").connect()
    t0 = time.perf_counter()
    with _http(s, MelanomaClient(db), _status_plan({"conditions": ["melanoma"]})) as tc:
        resp = _post_ok(tc, "status of melanoma trials")
    db.close()
    assert time.perf_counter() - t0 < 6
    assert resp.visualization.type is VizType.TABLE
    msgs = [n.message for n in resp.meta.notes if n.code is NoteCode.DEGRADED]
    assert any("time budget" in m for m in msgs), msgs


def test_http_ctgov_429_storm_returns_200_degraded(tmp_path, monkeypatch):
    real_sleep = asyncio.sleep

    async def fast_sleep(_delay, *a, **k):
        await real_sleep(0)

    for mod in (client_mod, limiter_mod):
        proxy = types.SimpleNamespace(**{n: getattr(asyncio, n) for n in dir(asyncio) if not n.startswith("__")})
        proxy.sleep = fast_sleep
        monkeypatch.setattr(mod, "asyncio", proxy)
    # keep the storm fast: no Retry-After lockout / rate halving (retry count is still bounded and checked)
    monkeypatch.setattr(limiter_mod.AsyncTokenBucket, "penalize", lambda self, seconds: None)
    monkeypatch.setattr(limiter_mod.AsyncTokenBucket, "adapt", lambda self, factor: None)
    s = Settings(
        offline=False,
        db_path=tmp_path / "s.db",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        ctgov_rpm=600,
        ctgov_burst=100,
        _env_file=None,
    )  # type: ignore[call-arg]
    db = Database(s.db_path).connect()
    plan = _sema_plan(Intent.PHASE_DISTRIBUTION)
    with respx.mock(assert_all_called=False) as router:
        route = router.get(url__startswith=BASE).mock(
            return_value=httpx.Response(
                429, headers={"Retry-After": "1"}, text="<html><body>Too Many Requests</body></html>"
            )
        )
        client = CTGovClient(s, db, transport=httpx.AsyncHTTPTransport())
        with _http(s, client, plan) as tc:
            resp = _post_ok(tc, "semaglutide trials by phase")
            tc.portal.call(client.aclose)
    db.close()
    assert route.call_count >= client_mod.MAX_ATTEMPTS
    assert route.call_count <= 3 * client_mod.MAX_ATTEMPTS  # bounded retries (search + version)
    assert resp.visualization.type is VizType.KPI_CARDS
    msgs = [n.message for n in resp.meta.notes if n.code is NoteCode.DEGRADED]
    assert any("CTGovUnavailable" in m and "Semaglutide" in m for m in msgs), msgs
    assert resp.meta.coverage.studies_matched_total == 0


# -- (5) StartDate partitions: boundaries, missing dates, no dups, no gaps --------------------------
_DRANGE = re.compile(r"AREA\[StartDate\]RANGE\[(MIN|\d{4}-\d{2}-\d{2}),(MAX|\d{4}-\d{2}-\d{2})\]")


def _norm_date(d: str | None) -> str | None:
    if not d:
        return None
    parts = d.split("-")
    return "-".join([parts[0], *(parts[1:] + ["01", "01"])[:2]])


def _start(raw: dict) -> str | None:
    return _norm_date(raw["protocolSection"]["statusModule"].get("startDateStruct", {}).get("date"))


def _rec(i: int, date: str | None) -> dict:
    status: dict = {"overallStatus": "COMPLETED"}
    if date:
        status["startDateStruct"] = {"date": date}
    return {"protocolSection": {"identificationModule": {"nctId": f"NCT{i:08d}"}, "statusModule": status}}


def boundary_universe(n: int, missing: int, *, missing_first: bool = False) -> list[dict]:
    """Every dated study sits on a year boundary (Jan 1 / Dec 31), or is a year-only / year-month date."""
    forms = ["{y}-01-01", "{y}-12-31", "{y}-12", "{y}", "{y}-01"]
    dated = [_rec(i, forms[i % len(forms)].format(y=2000 + (i // 5) % 24)) for i in range(n)]
    undated = [_rec(n + j, None) for j in range(missing)]
    outliers = [_rec(n + missing + k, d) for k, d in enumerate(["1985-01-01", "1999-12-31", "2031-01-01", "2040"])]
    return (undated + dated + outliers) if missing_first else (dated + undated + outliers)


def date_handler(universe: list[dict], calls: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        calls.append(params)
        adv = params.get("filter.advanced", "")
        rows = universe
        if MISSING_START in adv:
            rows = [r for r in rows if _start(r) is None]
        for lo, hi in _DRANGE.findall(adv):
            rows = [
                r
                for r in rows
                if _start(r) is not None and (lo == "MIN" or _start(r) >= lo) and (hi == "MAX" or _start(r) <= hi)
            ]
        size, start = int(params.get("pageSize", 10)), int(params.get("pageToken", 0))
        body: dict = {"studies": rows[start : start + size]}
        if params.get("countTotal") == "true":
            body["totalCount"] = len(rows)
        if start + size < len(rows):
            body["nextPageToken"] = str(start + size)
        return httpx.Response(200, json=body)

    return handler


@pytest.fixture
async def live_client(tmp_path):
    s = Settings(
        offline=False,
        db_path=tmp_path / "p.db",
        cache_dir=tmp_path / "cache",
        ctgov_rpm=600,
        ctgov_burst=100,
        _env_file=None,
    )  # type: ignore[call-arg]
    db = Database(s.db_path).connect()
    c = CTGovClient(s, db, transport=httpx.AsyncHTTPTransport())
    yield c
    await drain()
    await c.aclose()
    db.close()


@pytest.mark.parametrize("missing_first", [False, True])
async def test_partitions_on_boundaries_no_dups_no_gaps(live_client, missing_first):
    universe = boundary_universe(2400, missing=1100 if missing_first else 37, missing_first=missing_first)
    calls: list[dict] = []
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{BASE}/studies").mock(side_effect=date_handler(universe, calls))
        data = await fetch_cohort(
            live_client, CohortFilter(label="b", conditions=["c"]), field_set=FieldSet.LANDSCAPE, max_studies=5000
        )
    ids = [s.nct_id for s in data.studies]
    expected = {r["protocolSection"]["identificationModule"]["nctId"] for r in universe}
    assert data.total_count == len(universe)
    assert len(ids) == len(set(ids)), "duplicate studies across partitions"
    assert set(ids) == expected, f"gap: {len(expected - set(ids))} studies missing"
    assert not data.sampled
    advs = [c.get("filter.advanced", "") for c in calls]
    assert any(MISSING_START in a for a in advs)
    # partitions are contiguous year ranges: each 'lo' is the previous 'hi' + 1 day across a year boundary
    ranges = sorted({m for a in advs for m in _DRANGE.findall(a)}, key=lambda r: (r[0] != "MIN", r[0]))
    assert ranges[0][0] == "MIN" and ranges[-1][1] == "MAX"
    for (_, hi), (lo, _) in zip(ranges, ranges[1:], strict=False):
        assert hi.endswith("-12-31") and lo == f"{int(hi[:4]) + 1}-01-01"


# -- (6) inbound limiter ----------------------------------------------------------------------------
@pytest.fixture
def limited_app(settings, tmp_path):
    # only configured API keys get their own bucket (arbitrary header values must not mint buckets)
    keys = [SecretStr(k) for k in ("alpha", "beta", "gamma")]
    s = settings.model_copy(update={"inbound_rpm": 1, "inbound_burst": 5, "api_keys": keys})
    db = Database(tmp_path / "l.db").connect()
    with TestClient(create_app(settings=s, orchestrator=FakeOrchestrator(db))) as tc:
        yield tc, s.inbound_burst
    db.close()


def test_inbound_burst_plus_one_gives_single_429(limited_app):
    tc, burst = limited_app
    rs = [tc.post("/v1/visualize", json={"query": "phases"}, headers={"X-API-Key": "alpha"}) for _ in range(burst + 1)]
    codes = [r.status_code for r in rs]
    assert codes == [200] * burst + [429]
    limited = rs[-1]
    assert int(limited.headers["Retry-After"]) >= 1
    assert limited.json()["error"] == "rate_limited"
    assert limited.headers["X-Request-ID"]


def test_inbound_buckets_are_per_api_key(limited_app):
    tc, burst = limited_app
    for key in ("alpha", "beta", "gamma"):
        codes = [
            tc.post("/v1/visualize", json={"query": "q"}, headers={"X-API-Key": key}).status_code
            for _ in range(burst + 1)
        ]
        assert codes == [200] * burst + [429], key
    # the anonymous (IP) bucket is separate from all key buckets
    assert tc.post("/v1/visualize", json={"query": "q"}).status_code == 200
    # an unconfigured key shares the IP bucket instead of getting a fresh one
    codes = [
        tc.post("/v1/visualize", json={"query": "q"}, headers={"X-API-Key": f"made-up-{i}"}).status_code
        for i in range(burst)
    ]
    assert codes[-1] == 429 and codes.count(200) == burst - 1


# -- (7) the OpenAI key never appears in /v1/health or any log line --------------------------------
def test_openai_key_never_in_health_or_logs(tmp_path, monkeypatch, capsys):
    sentinel = "sk-proj-SENTINEL" + uuid.uuid4().hex
    monkeypatch.setenv("OPENAI_API_KEY", sentinel)
    monkeypatch.setenv("CTVIZ_RUNS_DIR", str(tmp_path / "runs"))
    s = Settings(
        offline=True,
        db_path=tmp_path / "k.db",
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        log_level="DEBUG",
        _env_file=None,
    )  # type: ignore[call-arg]
    assert s.openai_api_key is not None and s.openai_api_key.get_secret_value() == sentinel
    records: list[logging.LogRecord] = []

    class Grab(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    grab = Grab(level=logging.DEBUG)
    bodies: list[str] = []
    try:
        with TestClient(create_app(settings=s), raise_server_exceptions=False) as tc:
            root.addHandler(grab)
            root.setLevel(logging.DEBUG)
            for r in (
                tc.get("/v1/health"),
                tc.post("/v1/visualize", json={"query": "phases of semaglutide trials", "drug_name": "semaglutide"}),
                tc.post(
                    "/v1/visualize", json={"query": "Am I eligible?", "nct_id": TRIAL, "patient_profile": {"age": 40}}
                ),
                tc.post("/v1/visualize", json={"query": ""}),
                tc.get("/v1/metrics"),
                tc.get("/v1/trials/NCT00000000"),
            ):
                bodies.append(r.text)
            health = tc.get("/v1/health")
            assert health.status_code == 200 and health.json()["status"] == "ok"
            bodies.append(health.text)
            bodies.append(repr(s))
        drain_records = list(records)
    finally:
        root.handlers, _ = saved
        root.setLevel(saved[1])
    assert sentinel not in health.text and "SENTINEL" not in health.text
    assert any(r.name == "ctviz.request" for r in drain_records)
    for b in bodies:
        assert sentinel not in b
    for line in _format_lines(drain_records):
        assert sentinel not in line, line[:200]
    captured = capsys.readouterr()
    assert sentinel not in captured.err and sentinel not in captured.out
    for p in Path(tmp_path).rglob("*"):
        if p.is_file() and p.suffix in (".json", ".jsonl"):
            assert sentinel not in p.read_text(errors="ignore"), p
