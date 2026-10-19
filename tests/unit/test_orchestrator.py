"""End-to-end orchestrator runs, in-process, with a fake CT.gov client over recorded fixtures."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ctviz.agents.llm import StubLLM
from ctviz.api.schemas.plan import CohortFilter, Intent, Persona, QueryPlan, Scope
from ctviz.api.schemas.request import VisualizeRequest
from ctviz.api.schemas.response import NoteCode, VisualizeResponse
from ctviz.api.schemas.viz import VizType
from ctviz.core import orchestrator as orch_mod
from ctviz.core.background import drain
from ctviz.core.orchestrator import Orchestrator, minimal_plan
from ctviz.ctgov.client import SearchPage
from ctviz.store import response_cache as rc
from ctviz.store.db import Database
from ctviz.viz import base as viz_base
from tests.conftest import FIXTURES, assert_citations_verified, load_cohort_raw, load_study_raw

COHORT_FOR = {"semaglutide": "semaglutide", "tirzepatide": "tirzepatide", "pembrolizumab": "pembrolizumab_since2015"}


class FakeClient:
    """Implements the subset of CTGovClient the retrieval layer uses, backed by fixture files."""

    def __init__(self, db: Database):
        self.db = db
        self.base = "https://clinicaltrials.gov/api/v2"
        self.search_calls: list = []
        self.study_calls: list[str] = []

    def _studies_for(self, spec) -> list[dict]:
        if spec.locn and "atlantis" in spec.locn.lower():
            return []
        key = (spec.intr or spec.term or "").lower().strip('"')
        name = COHORT_FOR.get(key)
        return load_cohort_raw(name)["studies"] if name else []

    async def search_page(
        self, spec, *, page_size=1000, page_token=None, count_total=True, refresh=False, fields_override=None
    ) -> SearchPage:
        self.search_calls.append(spec)
        rows = self._studies_for(spec)
        return SearchPage(
            studies=rows[:page_size], total_count=len(rows), next_page_token=None, url=f"{self.base}/studies?fake=1"
        )

    async def count(self, spec, *, refresh=False) -> int:
        return (await self.search_page(spec, page_size=1)).total_count or 0

    async def get_studies(self, nct_ids, *, refresh=False) -> list[dict]:
        self.study_calls += nct_ids
        return [load_study_raw(n) for n in nct_ids if (FIXTURES / "studies" / f"{n}.json").exists()]

    async def version(self) -> dict:
        return {"apiVersion": "2.0.test", "dataTimestamp": "2026-10-17T09:00:00"}


def fixed_planner(plan: QueryPlan):
    async def planner(request, *, settings, llm, db):
        return plan, []

    return planner


@pytest.fixture
async def env(settings, tmp_path):
    db = Database(tmp_path / "o.db").connect()
    client = FakeClient(db)
    yield settings, db, client
    await drain()
    db.close()


def make(env, plan: QueryPlan | None = None, **kw) -> Orchestrator:
    settings, db, client = env
    kw.setdefault("runs_dir", None)
    return Orchestrator(settings, db, client, StubLLM(), planner=fixed_planner(plan) if plan else None, **kw)


def comparison_plan() -> QueryPlan:
    return QueryPlan(
        intent=Intent.PHASE_DISTRIBUTION,
        intent_confidence=0.9,
        scope=Scope.COMPARISON,
        cohorts=[
            CohortFilter(label="Semaglutide", drugs=["semaglutide"]),
            CohortFilter(label="Tirzepatide", drugs=["tirzepatide"]),
        ],
    )


def raw_index() -> dict[str, dict]:
    out = {}
    for name in ("semaglutide", "tirzepatide", "pembrolizumab_since2015"):
        for s in load_cohort_raw(name)["studies"]:
            out[s["protocolSection"]["identificationModule"]["nctId"]] = s
    return out


async def test_comparison_end_to_end(env):
    o = make(env, comparison_plan())
    resp = await o.run(VisualizeRequest(query="Compare phases for semaglutide vs tirzepatide", include_trace=True))
    VisualizeResponse.model_validate(resp.model_dump(mode="json"))
    assert resp.visualization.type is VizType.BAR_CHART
    assert resp.meta.coverage.studies_matched_total == 400 + 300
    assert len(resp.meta.coverage.queries) == 2
    assert set(resp.meta.filters_applied["cohorts"]) == {"Semaglutide", "Tirzepatide"}
    assert resp.ctgov_api_version == "2.0.test" and resp.data_timestamp
    assert resp.answer_summary and "Semaglutide" in resp.answer_summary
    assert not [n for n in resp.meta.notes if n.code in (NoteCode.DEGRADED, NoteCode.SUMMARY_TRIMMED)]
    assert resp.meta.trace is not None and any(s.name == "handler" for s in resp.meta.trace.steps)
    assert assert_citations_verified(resp.visualization, raw_index()) > 0
    await drain()
    cites = await rc.get_citations(env[1], resp.request_id)
    d0 = resp.visualization.data[0]
    assert len(cites[d0.id]) == (d0.citations_total or len(d0.citations))


async def test_response_cache_hit(env):
    o = make(env, comparison_plan())
    req = VisualizeRequest(query="Compare phases for semaglutide vs tirzepatide")
    first = await o.run(req)
    await drain()
    calls = len(env[2].search_calls)
    second = await o.run(req.model_copy(update={"include_trace": False}))
    assert second.meta.cache.response_hit and second.meta.interpretation.parser == "cache"
    assert second.request_id != first.request_id and second.meta.trace is None
    assert len(env[2].search_calls) == calls
    assert second.visualization == first.visualization
    third = await o.run(req.model_copy(update={"refresh": True}))
    assert not third.meta.cache.response_hit and len(env[2].search_calls) > calls
    await drain()
    assert await rc.get_citations(env[1], second.request_id) is not None  # aliased for the citations endpoint


async def test_trial_required_lists_candidates(env):
    plan = QueryPlan(
        intent=Intent.ELIGIBILITY_CHECK,
        intent_confidence=0.8,
        persona=Persona.PATIENT,
        scope=Scope.SINGLE_TRIAL,
        cohorts=[CohortFilter(label="Semaglutide", drugs=["semaglutide"])],
    )
    resp = await make(env, plan).run(VisualizeRequest(query="Am I eligible for a semaglutide trial?"))
    assert resp.status == "needs_clarification" and resp.visualization is None
    assert resp.clarification.reason == "trial_required"
    assert 0 < len(resp.clarification.options) <= 10
    assert all(o["nct_id"].startswith("NCT") for o in resp.clarification.options)
    assert resp.meta.coverage.studies_matched_total == 400
    assert env[2].study_calls == []  # no full-record fetch, no handler call


async def test_zero_results_after_relax(env):
    plan = QueryPlan(
        intent=Intent.PHASE_DISTRIBUTION,
        intent_confidence=0.9,
        cohorts=[CohortFilter(label="Nonexistium", drugs=["nonexistium"], countries=["Atlantis"])],
    )
    resp = await make(env, plan).run(VisualizeRequest(query="phases of nonexistium trials in Atlantis"))
    assert resp.visualization.type is VizType.KPI_CARDS
    assert any(n.code is NoteCode.ZERO_RESULTS for n in resp.meta.notes)
    assert resp.meta.coverage.studies_matched_total == 0


async def test_zero_results_suggest_relaxing_but_never_widen(env):
    plan = QueryPlan(
        intent=Intent.PHASE_DISTRIBUTION,
        intent_confidence=0.9,
        cohorts=[CohortFilter(label="Semaglutide", drugs=["semaglutide"], countries=["Atlantis"])],
    )
    resp = await make(env, plan).run(VisualizeRequest(query="semaglutide trials in Atlantis by phase"))
    assert resp.visualization.type is VizType.KPI_CARDS
    assert resp.meta.interpretation.cohorts[0].countries == ["Atlantis"]  # the question's filter is kept
    zero = [n.message for n in resp.meta.notes if n.code is NoteCode.ZERO_RESULTS]
    assert zero and "country" in zero[0]  # tells the user which filter to drop
    assert not any(n.code is NoteCode.FILTERS_RELAXED for n in resp.meta.notes)


async def test_single_trial_never_fails(env):
    plan = QueryPlan(
        intent=Intent.SINGLE_TRIAL_OVERVIEW, intent_confidence=0.9, scope=Scope.SINGLE_TRIAL, nct_ids=["NCT03548935"]
    )
    resp = await make(env, plan).run(VisualizeRequest(query="What is this trial?", nct_id="NCT03548935"))
    VisualizeResponse.model_validate(resp.model_dump(mode="json"))
    assert resp.meta.coverage.queries == ["https://clinicaltrials.gov/study/NCT03548935"]
    assert assert_citations_verified(resp.visualization, {"NCT03548935": load_study_raw("NCT03548935")}) >= 0


async def test_unknown_trial_zero_results(env):
    plan = QueryPlan(
        intent=Intent.SINGLE_TRIAL_OVERVIEW, intent_confidence=0.9, scope=Scope.SINGLE_TRIAL, nct_ids=["NCT99999999"]
    )
    resp = await make(env, plan).run(VisualizeRequest(query="What is this trial?", nct_id="NCT99999999"))
    assert any(n.code is NoteCode.ZERO_RESULTS for n in resp.meta.notes)


async def test_failing_handler_degrades_to_table(env, monkeypatch):
    async def boom(ctx):
        raise RuntimeError("bad data")

    monkeypatch.setitem(viz_base._HANDLERS, Intent.STATUS_DISTRIBUTION, boom)
    plan = QueryPlan(
        intent=Intent.STATUS_DISTRIBUTION,
        intent_confidence=0.9,
        cohorts=[CohortFilter(label="Tirzepatide", drugs=["tirzepatide"])],
    )
    o = make(env, plan)
    resp = await o.run(VisualizeRequest(query="status of tirzepatide trials"))
    assert resp.visualization.type is VizType.TABLE
    assert any(n.code is NoteCode.DEGRADED for n in resp.meta.notes)
    await drain()
    assert (
        await rc.get_response(env[1], rc.response_key(VisualizeRequest(query="status of tirzepatide trials"))) is None
    )  # degraded answers are not cached


async def test_missing_handler_gives_kpi(env, monkeypatch):
    monkeypatch.delitem(viz_base._HANDLERS, Intent.SPONSOR_LANDSCAPE, raising=False)
    plan = QueryPlan(
        intent=Intent.SPONSOR_LANDSCAPE,
        intent_confidence=0.9,
        cohorts=[CohortFilter(label="Tirzepatide", drugs=["tirzepatide"])],
    )
    resp = await make(env, plan).run(VisualizeRequest(query="who sponsors tirzepatide trials"))
    assert resp.visualization.type is VizType.KPI_CARDS
    assert resp.visualization.data[0].citations[0].kind.value == "derived_count"


async def test_preference_ignored_note(env):
    plan = comparison_plan().model_copy(update={"preferred_visualization": VizType.SURVIVAL_CURVE})
    resp = await make(env, plan).run(VisualizeRequest(query="phases semaglutide vs tirzepatide"))
    assert any(n.code is NoteCode.VIZ_PREFERENCE_IGNORED for n in resp.meta.notes)
    assert resp.meta.alternatives


async def test_default_planner_path(env):
    resp = await make(env).run(
        VisualizeRequest(
            query="Compare phases for semaglutide vs tirzepatide", drug_names=["semaglutide", "tirzepatide"]
        )
    )
    VisualizeResponse.model_validate(resp.model_dump(mode="json"))
    assert resp.meta.interpretation.parser in ("rules", "llm")


async def test_planner_exception_falls_back(env):
    async def broken(request, **_):
        raise ValueError("nope")

    settings, db, client = env
    o = Orchestrator(settings, db, client, StubLLM(), planner=broken, runs_dir=None)
    resp = await o.run(VisualizeRequest(query="phases", drug_name="semaglutide"))
    assert resp.visualization.type is VizType.BAR_CHART


async def test_run_artifacts_written_and_redacted(env, tmp_path):
    o = make(env, comparison_plan(), runs_dir=tmp_path / "runs")
    req = VisualizeRequest(query="phases semaglutide vs tirzepatide", patient_profile={"age": 51, "sex": "female"})
    resp = await o.run(req)
    await drain()
    d: Path = tmp_path / "runs" / resp.request_id
    for name in (
        "request.json",
        "plan.json",
        "ctgov_calls.json",
        "handler_result.json",
        "steps.jsonl",
        "response.json",
    ):
        assert (d / name).exists(), name
    assert json.loads((d / "request.json").read_text())["patient_profile"] == "[redacted]"
    steps = [json.loads(line) for line in (d / "steps.jsonl").read_text().splitlines()]
    assert [s["tool"] for s in steps][:3] == ["response_cache", "plan", "fetch_cohorts"]
    assert all("ms" in s and "summary" in s for s in steps)


# -- review fixes ---------------------------------------------------------------------------------
def test_item1_trial_table_fills_every_column():
    from ctviz.core.orchestrator import MISSING_VALUE, trial_table
    from ctviz.core.provenance import Study

    bare = Study({"protocolSection": {"identificationModule": {"nctId": "NCT00000001", "briefTitle": "T"}}})
    viz, _ = trial_table([bare, Study(load_study_raw("NCT03548935"))], "Candidates")
    cols = [c.field for c in viz.meta.columns]
    for row in viz.data:
        assert all((row.model_extra or {}).get(f) is not None for f in cols)
    first = viz.data[0].model_extra
    assert first["phase"] == MISSING_VALUE and "phase" in first["cite_note"]
    assert next(c for c in viz.meta.columns if c.field == "enrollment").type == "nominal"


class CancellingClient(FakeClient):
    async def search_page(self, spec, **kw):
        if spec.intr == "cancelme":
            raise asyncio.CancelledError
        return await super().search_page(spec, **kw)


async def test_item2_cancelled_cohort_task_degrades(env):
    settings, db, _ = env
    plan = QueryPlan(
        intent=Intent.PHASE_DISTRIBUTION,
        intent_confidence=0.9,
        scope=Scope.COMPARISON,
        cohorts=[CohortFilter(label="C", drugs=["cancelme"]), CohortFilter(label="Tirzepatide", drugs=["tirzepatide"])],
    )
    o = Orchestrator(settings, db, CancellingClient(db), StubLLM(), planner=fixed_planner(plan), runs_dir=None)
    resp = await o.run(VisualizeRequest(query="phases"))
    assert resp.visualization.type is VizType.BAR_CHART
    assert any(n.code is NoteCode.DEGRADED and "cancelled" in n.message for n in resp.meta.notes)


async def test_item2_bounded_child_vs_own_cancellation(env):
    from ctviz.core.orchestrator import _Run
    from ctviz.viz.base import HandlerContext

    o = make(env)
    req = VisualizeRequest(query="x")
    plan = minimal_plan(req)
    st = _Run(
        request=req,
        plan=plan,
        entities=[],
        ctx=HandlerContext(request=req, plan=plan, settings=env[0], client=None, db=None, llm=None),
    )

    async def child_cancelled():
        raise asyncio.CancelledError

    assert await o._bounded(child_cancelled(), 1.0, st, "probe") is None and st.degraded
    task = asyncio.create_task(o._bounded(asyncio.sleep(10), 30.0, st, "sleep"))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


class LoggingClient(FakeClient):
    async def search_page(self, spec, **kw):
        from ctviz.obs.trace import current_trace

        current_trace().ctgov_calls_log.append(
            {
                "url": f"https://x/studies?intr={spec.intr}",
                "cached": False,
                "ms": 1.0,
                "status": 200,
                "cache_file": "/cache/abc.json",
            }
        )
        return await super().search_page(spec, **kw)


async def test_item3_runs_use_trace_call_log_and_settings(env, tmp_path):
    settings, db, _ = env
    assert make(env, comparison_plan(), runs_dir="default").runs_dir is None  # server default: off
    on = settings.model_copy(update={"runs_enabled": True, "runs_dir": tmp_path / "r"})
    o = Orchestrator(on, db, LoggingClient(db), StubLLM(), planner=fixed_planner(comparison_plan()))
    assert o.runs_dir == tmp_path / "r"
    resp, rec = await o.run_recorded(VisualizeRequest(query="phases"))
    await drain()
    calls = json.loads((tmp_path / "r" / resp.request_id / "ctgov_calls.json").read_text())
    assert len(calls) == 2 and all(c["cache_file"] == "/cache/abc.json" for c in calls)
    assert not (tmp_path / "r" / resp.request_id / "ctgov").exists()  # no second copy of raw data


async def test_item7_patient_requests_private(env, tmp_path):
    o = make(env, comparison_plan(), runs_dir=tmp_path / "runs")
    req = VisualizeRequest(query="Can I join a semaglutide trial near Boston?", persona="patient")
    resp, rec = await o.run_recorded(req)
    await drain()
    assert await rc.get_response(env[1], rc.response_key(req)) is None
    plan_step = next(s for s in rec.steps if s["tool"] == "plan")
    assert plan_step["inputs"]["query"].startswith("sha256:")
    saved = json.loads((tmp_path / "runs" / resp.request_id / "request.json").read_text())
    assert saved["query"].startswith("sha256:") and "Boston" not in json.dumps(saved)
    assert "Boston" not in (tmp_path / "runs" / resp.request_id / "steps.jsonl").read_text()


async def test_item9_slow_cache_read_is_a_miss(env, monkeypatch):
    async def slow(db, key):
        await asyncio.sleep(1.0)
        return {"response": {}}

    monkeypatch.setattr(rc, "get_response", slow)
    monkeypatch.setattr(orch_mod, "CACHE_READ_TIMEOUT_S", 0.05)
    resp = await make(env, comparison_plan()).run(VisualizeRequest(query="phases"))
    assert resp.visualization.type is VizType.BAR_CHART and not resp.meta.cache.response_hit


async def test_item9_background_drops_low_priority_when_full(monkeypatch):
    from ctviz.core import background

    monkeypatch.setattr(background, "MAX_PENDING", 0)
    ran: list[str] = []

    async def work(tag):
        ran.append(tag)

    before = background.dropped
    assert background.spawn(work("low"), "low", low_priority=True) is None
    assert background.spawn(work("high"), "high") is not None
    await drain()
    assert ran == ["high"] and background.dropped == before + 1


async def test_item10_emergency_trace_respects_settings(env, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(Orchestrator, "_run", boom)
    settings, db, client = env
    quiet = Orchestrator(settings.model_copy(update={"include_trace": False}), db, client, StubLLM(), runs_dir=None)
    resp = await quiet.run(VisualizeRequest(query="x", include_trace=True))
    assert resp.meta.trace is None and resp.meta.notes[0].code is NoteCode.DEGRADED
    loud = Orchestrator(settings, db, client, StubLLM(), runs_dir=None)
    assert (await loud.run(VisualizeRequest(query="x", include_trace=True))).meta.trace is not None


def test_minimal_plan_shapes():
    p = minimal_plan(VisualizeRequest(query="NCT03548935 vs NCT02578680 results"))
    assert p.scope is Scope.COMPARISON and p.nct_ids == ["NCT03548935", "NCT02578680"]
    p = minimal_plan(VisualizeRequest(query="trials per year", drug_name="x"))
    assert p.intent is Intent.TREND_OVER_TIME and p.cohorts[0].drugs == ["x"]
    assert orch_mod._trial_only(orch_mod.get_intent_spec(Intent.ELIGIBILITY_CHECK))
