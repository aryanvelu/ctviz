"""Gap tests for schema 2.2 (offline, fixture-backed):

* an end-to-end orchestrator run of a cohort survival question: every public datum's citations verify verbatim
  against the recorded raw JSON (``path_get``), and every ``derived_from`` id names a reported datum of the same
  public response, in ``sample``, ``ids_only`` and ``full`` citation detail;
* the citations endpoint, called over HTTP after a response-cache hit, returns the complete ``nct_ids`` list;
* a survival question about a trial without posted results never draws a curve;
* edge cases of the deterministic refusal regex.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ctviz.agents.llm import StubLLM
from ctviz.api.public import MAX_NCT_IDS_PER_DATUM, PublicResponse, to_public
from ctviz.api.schemas.plan import CohortFilter, Intent, QueryPlan, Scope
from ctviz.api.schemas.request import VisualizeRequest
from ctviz.api.schemas.response import NoteCode, VisualizeResponse
from ctviz.api.schemas.viz import VizType
from ctviz.app import create_app
from ctviz.core.answerability import is_subjective
from ctviz.core.background import drain
from ctviz.core.orchestrator import Orchestrator
from ctviz.ctgov.paths import path_get
from ctviz.store import response_cache as rc
from ctviz.store.db import Database
from tests.conftest import load_study_raw
from tests.unit.test_orchestrator import FakeClient, comparison_plan, fixed_planner, raw_index

SURV = ["NCT02578680", "NCT02142738", "NCT01844505"]  # KEYNOTE-189, KEYNOTE-024, CheckMate 067 (results posted)
NO_RESULTS = "NCT07654361"  # recorded record without resultsSection
QUERY = "Kaplan-Meier overall survival curves across pembrolizumab trials"


class SurvivalClient(FakeClient):
    """A 'pembrolizumab' search returns the three survival fixtures plus one trial without results."""

    def _studies_for(self, spec) -> list[dict]:
        key = (spec.intr or spec.term or "").lower().strip('"')
        if key == "pembrolizumab":
            return [load_study_raw(n) for n in [*SURV, NO_RESULTS]]
        return super()._studies_for(spec)


@pytest.fixture
async def surv_env(settings, tmp_path):
    db = Database(tmp_path / "s.db").connect()
    yield settings, db, SurvivalClient(db)
    await drain()
    db.close()


def _orch(env, plan: QueryPlan) -> Orchestrator:
    settings, db, client = env
    return Orchestrator(settings, db, client, StubLLM(), planner=fixed_planner(plan), runs_dir=None)


def _survival_plan() -> QueryPlan:
    return QueryPlan(
        intent=Intent.SURVIVAL_SUMMARY,
        intent_confidence=0.9,
        scope=Scope.COHORT,
        cohorts=[CohortFilter(label="Pembrolizumab", drugs=["pembrolizumab"])],
    )


def _raws() -> dict[str, dict]:
    return {n: load_study_raw(n) for n in [*SURV, NO_RESULTS]}


def _public_rows(pub: dict) -> list[dict]:
    data = pub["visualization"]["data"]
    return data["nodes"] + data["edges"] if isinstance(data, dict) else data


def _verify_public_citations(pub: dict, raws: dict[str, dict]) -> int:
    """Citation invariant on the PUBLIC JSON: resolve each excerpt (with the datum's hoisted field_path)."""
    checked = 0
    for d in _public_rows(pub):
        for c in d.get("citations", []):
            kind = c.get("kind", "field_value")
            if kind == "derived_count":
                assert c["excerpt"].isdigit()
                continue
            path = c.get("field_path", d.get("field_path"))
            assert path, f"{d['id']}: citation without a field_path"
            assert c["nct_id"] in raws, f"{d['id']}: citation to a trial that was not retrieved ({c['nct_id']})"
            value = path_get(raws[c["nct_id"]], path)
            assert value is not None, f"{d['id']}: {path} does not resolve in {c['nct_id']}"
            if kind == "text_excerpt":
                assert value[c["char_start"] : c["char_end"]] == c["excerpt"], f"{d['id']}: {path}"
            else:
                leaf = value if isinstance(value, str) else str(value)
                assert c["excerpt"] == leaf or (len(c["excerpt"]) == 300 and leaf.startswith(c["excerpt"])), (
                    f"{d['id']}: {path}: {c['excerpt']!r} != {leaf!r}"
                )
            checked += 1
    return checked


def _check_derived_from(pub: dict) -> int:
    rows = {d["id"]: d for d in _public_rows(pub)}
    linked = 0
    for d in rows.values():
        if d.get("source") == "interpolated":
            assert len(d.get("derived_from", [])) == 2, d["id"]
        for ref in d.get("derived_from", []):
            assert ref in rows, f"{d['id']}: derived_from {ref} is not in this response"
            anchor = rows[ref]
            assert anchor["source"] == "reported" and anchor["series"] == d["series"]
            assert "derived_from" not in anchor
            linked += 1
    return linked


async def _survival_response(env, **req) -> VisualizeResponse:
    resp = await _orch(env, _survival_plan()).run(VisualizeRequest(query=QUERY, **req))
    VisualizeResponse.model_validate(resp.model_dump(mode="json"))
    return resp


# -- end to end: cohort survival curve ------------------------------------------------------------------
async def test_cohort_survival_public_citations_verify_and_derived_from_resolves(surv_env):
    resp = await _survival_response(surv_env)
    assert resp.status == "ok" and resp.visualization.type is VizType.SURVIVAL_CURVE
    assert NO_RESULTS not in surv_env[2].study_calls  # only result-bearing trials are fetched in full
    pub = to_public(resp)
    PublicResponse.model_validate(pub)
    assert _verify_public_citations(pub, _raws()) > 0
    assert _check_derived_from(pub) > 0
    rows = _public_rows(pub)
    assert len({r["nct_id"] for r in rows}) >= 2  # really a cross-trial chart
    assert {r["endpoint"] for r in rows} == {"OS"}
    assert pub["meta"]["validation"]["passed"] is True
    assert pub["meta"]["approximation"]["method"]
    assert not [n for n in resp.meta.notes if n.code in (NoteCode.DEGRADED, NoteCode.SUMMARY_TRIMMED)]
    assert resp.answer_summary and resp.answer_summary.startswith("Approximate overall survival curves")


async def test_cohort_survival_ids_only_keeps_derived_from(surv_env):
    resp = await _survival_response(surv_env, citation_detail="ids_only")
    pub = to_public(resp, citation_detail="ids_only")
    PublicResponse.model_validate(pub)
    rows = _public_rows(pub)
    assert all("citations" not in r and "field_path" not in r for r in rows)
    assert all(r["nct_ids"] and r["nct_ids_total"] == len(r["nct_ids"]) == 1 for r in rows)
    assert _check_derived_from(pub) > 0


async def test_cohort_survival_full_detail_verifies_every_citation(surv_env):
    resp = await _survival_response(surv_env, citation_detail="full")
    pub = to_public(resp, citation_detail="full")
    PublicResponse.model_validate(pub)
    assert all(not r["excerpts_truncated"] for r in _public_rows(pub))
    assert _verify_public_citations(pub, _raws()) >= _verify_public_citations(to_public(resp), _raws())
    assert _check_derived_from(pub) > 0


async def test_cohort_survival_citations_endpoint_store_matches_public(surv_env):
    resp = await _survival_response(surv_env)
    await drain()
    got = await rc.get_datum_sources(surv_env[1], resp.request_id)
    assert got is not None
    cites, ncts = got
    for d in _public_rows(to_public(resp)):
        assert ncts[d["id"]] == d["nct_ids"]
        assert len(cites[d["id"]]) == d["citations_total"]


# -- survival question about a trial without posted results ---------------------------------------------
async def test_survival_for_trial_without_results_draws_no_curve(surv_env):
    plan = QueryPlan(
        intent=Intent.SURVIVAL_SUMMARY, intent_confidence=0.9, scope=Scope.SINGLE_TRIAL, nct_ids=[NO_RESULTS]
    )
    resp = await _orch(surv_env, plan).run(VisualizeRequest(query="Show the survival curve", nct_id=NO_RESULTS))
    VisualizeResponse.model_validate(resp.model_dump(mode="json"))
    pub = to_public(resp)
    PublicResponse.model_validate(pub)
    assert resp.visualization is None or resp.visualization.type is not VizType.SURVIVAL_CURVE
    codes = {n.code for n in resp.meta.notes}
    assert NoteCode.NO_RESULTS_POSTED in codes or resp.status != "ok"
    if resp.visualization is not None:
        for d in _public_rows(pub):  # nothing numeric was invented about survival
            assert "survival" not in d and "derived_from" not in d
        _verify_public_citations(pub, {NO_RESULTS: load_study_raw(NO_RESULTS)})
        assert resp.meta.validation.passed, resp.meta.validation


async def test_survival_cohort_where_no_trial_has_results(surv_env, monkeypatch):
    monkeypatch.setattr(SurvivalClient, "_studies_for", lambda self, spec: [load_study_raw(NO_RESULTS)], raising=True)
    resp = await _survival_response(surv_env)
    assert resp.visualization is None or resp.visualization.type is not VizType.SURVIVAL_CURVE
    assert surv_env[2].study_calls == []  # nothing result-bearing to fetch
    assert NoteCode.NOT_IN_REGISTRY in {n.code for n in resp.meta.notes}
    if resp.visualization is not None:
        assert not [d for d in _public_rows(to_public(resp)) if "survival" in d]
        # An honest "0 trials with usable values" card is a valid answer and must pass the final validation
        # (otherwise it is flagged VALIDATION_FAILED and never response-cached).
        assert resp.meta.validation.passed, [c for c in resp.meta.validation.checks if not c.passed]


# -- citations endpoint over HTTP after a response-cache hit ----------------------------------------------
@pytest.fixture
def http(settings, tmp_path):
    db = Database(tmp_path / "h.db").connect()
    orch = Orchestrator(
        settings, db, FakeClient(db), StubLLM(), planner=fixed_planner(comparison_plan()), runs_dir=None
    )
    s = settings.model_copy(update={"openai_api_key": SecretStr("sk-test-" + "x" * 24), "inbound_burst": 100})
    tc = TestClient(create_app(settings=s, orchestrator=orch))
    tc.__enter__()
    yield tc
    tc.__exit__(None, None, None)
    db.close()


def test_citations_endpoint_after_response_cache_hit_returns_complete_nct_ids(http):
    body = {"query": "Compare phases for semaglutide vs tirzepatide"}
    first = http.post("/v1/visualize", json=body).json()
    http.portal.call(drain)
    second_r = http.post("/v1/visualize", json=body)
    second = second_r.json()
    http.portal.call(drain)
    rid1, rid2 = first["meta"]["request_id"], second["meta"]["request_id"]
    assert rid1 != rid2 and second_r.headers["x-request-id"] == rid2
    assert second["visualization"]["data"] == first["visualization"]["data"]
    big = [d for d in second["visualization"]["data"] if d.get("nct_ids_total", 0) > MAX_NCT_IDS_PER_DATUM]
    assert big
    raws = raw_index()
    for d in second["visualization"]["data"]:
        if "nct_ids" not in d:
            continue
        ev = http.get(f"/v1/responses/{rid2}/datums/{d['id']}/citations")
        assert ev.status_code == 200
        ev = ev.json()
        assert ev["nct_ids_total"] == d["nct_ids_total"] == len(ev["nct_ids"]) == d["trial_count"]
        assert ev["nct_ids"][: len(d["nct_ids"])] == d["nct_ids"] and ev["nct_ids"] == sorted(ev["nct_ids"])
        assert set(ev["nct_ids"]) <= set(raws)
        assert ev["unverified"] == 0 and ev["total"] == d["citations_total"]
        assert ev == {**http.get(f"/v1/responses/{rid1}/datums/{d['id']}/citations").json(), "request_id": rid2}


# -- deterministic refusal regex: edge cases ------------------------------------------------------------
REFUSE = [
    "Which drug has the best overall survival?",  # "overall survival" is not an excluded endpoint name
    "Which drug is best?",
    "Is pembrolizumab safe and effective?",
    "Is NCT03548935 safe?",
    "Is it effective?",
    "Should I enroll in NCT03548935?",
    "Should i enrol in this trial",
    "Is semaglutide any good?",
    "Are there trials of drugs safer than metformin?",
    "best available therapy for myelofibrosis",  # documented: "best available ..." stays a verdict
    "How safe is semaglutide?",  # same verdict as "Is semaglutide safe?"
]
ANSWER = [
    "BEST SUPPORTIVE CARE comparator trials in AML",  # case-insensitive exclusion
    "Best Overall Response by arm in KEYNOTE-024",
    "best response rates in NCT02142738",
    "Best Corrected Visual Acuity change in aflibercept trials",
    "best-known GLP-1 trials",
    "bestatin trials",  # word boundary
    "Is this trial recruiting?",
    "What is the safety profile of semaglutide?",
    "non-inferiority trials of semaglutide",
    "superiority trials for tirzepatide",
    "Which arm had better outcomes in NCT02142738?",
    "Is a lower dose better tolerated in semaglutide trials?",
    "Trials of drug X versus best supportive-care in AML",  # hyphenated form of an excluded comparator
    "How is worst-case imputation used in semaglutide trials?",  # statistical method, not a verdict
]


@pytest.mark.parametrize("q", REFUSE)
def test_refusal_regex_refuses_verdicts(q):
    assert is_subjective(q)


@pytest.mark.parametrize("q", ANSWER)
def test_refusal_regex_answers_endpoint_and_design_terms(q):
    assert not is_subjective(q)
