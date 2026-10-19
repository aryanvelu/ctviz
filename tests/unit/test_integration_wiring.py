"""Integration wiring: retrieval agent on ambiguity / zero results, background graph + FTS growth,
graph neighbours route, and the CLI request loaders. Offline (fixture-backed fake client, stub LLM)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from ctviz.agents.llm import StubLLM
from ctviz.api.schemas.plan import CohortFilter, Intent, QueryPlan
from ctviz.api.schemas.request import VisualizeRequest
from ctviz.api.schemas.response import NoteCode, VisualizeResponse
from ctviz.app import create_app
from ctviz.cli import EXAMPLES_FILE, WARM_FILE, _load_named_requests
from ctviz.core.background import drain
from ctviz.core.orchestrator import Orchestrator
from ctviz.store.db import Database
from ctviz.store.fts import TextIndex
from ctviz.store.graph import GraphStore
from tests.conftest import load_cohort_raw
from tests.unit.test_orchestrator import FakeClient, fixed_planner


class SponsorClient(FakeClient):
    """Serves the melanoma fixture for the given sponsor names; everything else as FakeClient."""

    def __init__(self, db, sponsors_with_trials: set[str]):
        super().__init__(db)
        self.sponsors_with_trials = sponsors_with_trials

    def _studies_for(self, spec) -> list[dict]:
        if spec.spons:
            name = spec.spons.strip('"')
            return load_cohort_raw("melanoma_ph2_3")["studies"] if name in self.sponsors_with_trials else []
        return super()._studies_for(spec)


def _orch(settings, db, client, plan: QueryPlan | None = None, **kw) -> Orchestrator:
    kw.setdefault("runs_dir", None)
    return Orchestrator(settings, db, client, StubLLM(), planner=fixed_planner(plan) if plan else None, **kw)


async def test_ambiguous_sponsor_with_one_meaning_is_resolved_and_noted(settings, tmp_path):
    db = Database(tmp_path / "a.db").connect()
    client = SponsorClient(db, {"Merck Sharp & Dohme"})
    resp = await _orch(settings, db, client).run(VisualizeRequest(query="Merck trials by phase", include_trace=True))
    await drain()
    db.close()
    VisualizeResponse.model_validate(resp.model_dump(mode="json"))
    assert resp.meta.interpretation.cohorts[0].sponsors == ["Merck Sharp & Dohme"]
    msgs = [n.message for n in resp.meta.notes if n.code is NoteCode.APPROXIMATED]
    assert any("interpreted as 'Merck Sharp & Dohme'" in m for m in msgs), msgs
    assert resp.meta.coverage.studies_matched_total > 0
    assert any(s.name == "retrieval_agent" for s in (resp.meta.trace.steps if resp.meta.trace else []))


async def test_ambiguous_sponsor_with_two_meanings_asks(settings, tmp_path):
    db = Database(tmp_path / "b.db").connect()
    client = SponsorClient(db, {"Merck Sharp & Dohme", "Merck KGaA"})
    resp = await _orch(settings, db, client).run(VisualizeRequest(query="Merck trials by phase"))
    await drain()
    db.close()
    assert resp.status == "needs_clarification" and resp.visualization is None  # ask, never pick or merge
    clar = resp.clarification
    assert clar.reason == "ambiguous_entity" and "Which did you mean" in clar.message and "Merck KGaA" in clar.message
    assert any("Merck KGaA" in x for x in clar.suggestions)
    assert any(n.code is NoteCode.ENTITY_AMBIGUOUS for n in resp.meta.notes)


async def test_zero_results_are_answered_not_rewritten(settings, tmp_path):
    """Zero matches is reported as 0 (cited by its search URL); the question is never silently rewritten."""
    db = Database(tmp_path / "z.db").connect()
    plan = QueryPlan(
        intent=Intent.PHASE_DISTRIBUTION,
        intent_confidence=0.9,
        cohorts=[CohortFilter(label="Keytruda", drugs=["keytruda"])],
    )
    resp = await _orch(settings, db, FakeClient(db), plan).run(VisualizeRequest(query="keytruda trials by phase"))
    await drain()
    db.close()
    assert resp.meta.interpretation.cohorts[0].drugs == ["keytruda"]
    assert resp.meta.coverage.studies_matched_total == 0
    assert any(n.code is NoteCode.ZERO_RESULTS for n in resp.meta.notes)


async def test_background_writes_grow_graph_and_text_index(settings, tmp_path):
    db = Database(tmp_path / "g.db").connect()
    graph, fts = GraphStore(db), TextIndex(db)
    plan = QueryPlan(
        intent=Intent.PHASE_DISTRIBUTION,
        intent_confidence=0.9,
        cohorts=[CohortFilter(label="Semaglutide", drugs=["semaglutide"])],
    )
    o = Orchestrator(
        settings, db, FakeClient(db), StubLLM(), graph, text_index=fts, planner=fixed_planner(plan), runs_dir=None
    )
    await o.run(VisualizeRequest(query="semaglutide trials by phase"))
    await drain()
    stats = graph.stats_sync()
    assert stats["nodes"].get("trial", 0) > 0 and stats["edges"] > 0
    assert fts.count_sync() > 0
    db.close()


def test_graph_neighbors_route_reports_match_kind(settings, tmp_path):
    db = Database(tmp_path / "n.db").connect()
    graph = GraphStore(db)
    from ctviz.core.provenance import Study

    graph.ingest_sync([Study(r) for r in load_cohort_raw("semaglutide")["studies"]])
    o = Orchestrator(settings, db, FakeClient(db), StubLLM(), graph, runs_dir=None)
    with TestClient(create_app(settings=settings, orchestrator=o)) as tc:
        r = tc.get("/v1/graph/neighbors", params={"name": "semaglutide", "node_type": "drug"})
        assert r.status_code == 200
        body = r.json()
        assert body["match"] in ("exact", "alias", "fuzzy") and body["center"]["type"] == "drug"
        assert body["edges"] and all(e["evidence"] for e in body["edges"])
        miss = tc.get("/v1/graph/neighbors", params={"name": "zzzunknownzzz", "node_type": "drug"}).json()
        assert miss["match"] is None and miss["center"] is None
        bad = tc.get("/v1/graph/neighbors", params={"name": "x", "node_type": "planet"})
        assert bad.status_code == 422
    db.close()


async def test_real_eligibility_handler_keeps_profile_out_of_runs_and_logs(settings, tmp_path, caplog):
    """The checklist echoes the profile to the caller (meta.diagnostics.profile_used) but never to disk/logs."""
    import logging

    from ctviz.obs.logging import JsonFormatter

    caplog.set_level(logging.DEBUG)
    db = Database(tmp_path / "p.db").connect()
    runs = tmp_path / "runs"
    o = Orchestrator(settings, db, FakeClient(db), StubLLM(), runs_dir=runs)
    req = VisualizeRequest(
        query="Am I eligible for this trial?",
        nct_id="NCT03548935",
        persona="patient",
        patient_profile={
            "age": 45,
            "sex": "female",
            "conditions": ["zqxcondsentinel"],
            "medications": ["zqxmedsentinel"],
            "notes": "zqxnotesentinel",
        },
    )
    resp = await o.run(req)
    await drain()
    db.close()
    assert resp.visualization.type.value == "eligibility_checklist"
    assert resp.visualization.meta.diagnostics["profile_used"]["medications"] == ["zqxmedsentinel"]
    files = [p for p in (runs / resp.request_id).iterdir() if p.is_file()]
    assert {p.name for p in files} >= {
        "request.json",
        "plan.json",
        "handler_result.json",
        "response.json",
        "steps.jsonl",
    }
    jf = JsonFormatter()
    for text in [p.read_text() for p in files] + [jf.format(r) for r in caplog.records]:
        assert "zqx" not in text


async def test_trend_summary_survives_verification(settings, tmp_path):
    """Row-based trend sentence uses the channel's time_format ('2019', not '2019-01-01'), so it is kept."""
    db = Database(tmp_path / "t.db").connect()
    plan = QueryPlan(
        intent=Intent.TREND_OVER_TIME,
        intent_confidence=0.9,
        time_granularity="year",
        cohorts=[CohortFilter(label="Pembrolizumab", drugs=["pembrolizumab"], start_year=2015)],
    )
    resp = await _orch(settings, db, FakeClient(db), plan).run(VisualizeRequest(query="pembrolizumab trials per year"))
    await drain()
    db.close()
    assert resp.visualization.encoding.x.time_format == "%Y" and resp.visualization.encoding.x.format is None
    assert not any(n.code is NoteCode.SUMMARY_TRIMMED for n in resp.meta.notes)
    assert resp.answer_summary and "has the most trials" in resp.answer_summary
    assert "-01-01" not in resp.answer_summary


def test_temporal_channel_with_number_format_is_rejected():
    import pytest

    from ctviz.api.schemas.viz import Channel, Datum, Encoding, VizType
    from ctviz.viz.base import EncodingError, make_viz

    enc = Encoding(
        x=Channel(field="d", type="temporal", title="Date", format="%Y"),
        y=Channel(field="n", type="quantitative", title="N"),
    )
    with pytest.raises(EncodingError, match="time_format"):
        make_viz(VizType.TIME_SERIES, "t", enc, [Datum(id="a", d="2020-01-01", n=1)])


def test_warm_queries_and_examples_load_as_named_requests():
    warm = _load_named_requests(WARM_FILE)
    assert len(warm) >= 10 and all(isinstance(b, dict) and b.get("query") for _, b in warm)
    assert len({n for n, _ in warm}) == len(warm)
    for _, body in warm:
        VisualizeRequest.model_validate(body)
    examples = _load_named_requests(EXAMPLES_FILE)
    assert len(examples) >= 12
    for _, body in examples:
        VisualizeRequest.model_validate(body)
    personas = {b.get("persona") for _, b in examples}
    assert {"patient", "researcher"} <= personas
    assert json.loads(EXAMPLES_FILE.read_text())  # valid JSON
