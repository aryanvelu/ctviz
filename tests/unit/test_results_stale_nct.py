"""A named trial without posted results + a question naming a drug/condition -> the cohort answer (offline).

Bug: "Give me a Kaplan Meier for nivolumab in melanoma" sent with an NCT id left over in the UI form answered
with that trial's registered outcomes. Now results-only intents answer for the named drug/condition (trials
with posted results) and say why in a NO_RESULTS_POSTED note; with no drug/condition the single-trial answer
stays and its title says no results are posted.
"""

from __future__ import annotations

import pytest

from ctviz.agents.llm import StubLLM
from ctviz.api.schemas.plan import Intent, QueryPlan, Scope
from ctviz.api.schemas.request import VisualizeRequest
from ctviz.api.schemas.response import NoteCode, ResolvedEntity
from ctviz.api.schemas.viz import VizType
from ctviz.core.background import drain
from ctviz.core.intents import get_intent_spec
from ctviz.core.orchestrator import Orchestrator, cohort_instead
from ctviz.core.provenance import Study
from ctviz.store.db import Database
from tests.conftest import load_study_raw
from tests.unit.test_orchestrator import FakeClient

SURV = ["NCT02578680", "NCT02142738", "NCT01844505"]  # results posted
NO_RESULTS = "NCT07654361"  # recorded record without resultsSection
QUERY = "Give me a Kaplan Meier for pembrolizumab in melanoma"


class PembroClient(FakeClient):
    """A pembrolizumab search returns the three survival fixtures."""

    def _studies_for(self, spec) -> list[dict]:
        if (spec.intr or "").lower().strip('"') == "pembrolizumab":
            return [load_study_raw(n) for n in SURV]
        return super()._studies_for(spec)


@pytest.fixture
async def env(settings, tmp_path):
    db = Database(tmp_path / "s.db").connect()
    yield settings, db, PembroClient(db)
    await drain()
    db.close()


def _orch(env, planner=None) -> Orchestrator:
    settings, db, client = env
    return Orchestrator(settings, db, client, StubLLM(), planner=planner, runs_dir=None, use_response_cache=False)


def _entity(kind: str, name: str, terms: list[str] | None = None) -> ResolvedEntity:
    return ResolvedEntity(kind=kind, input=name, resolved=name, method="exact", search_terms=terms or [name])  # type: ignore[arg-type]


def _plan(intent: Intent = Intent.SURVIVAL_SUMMARY, ncts: list[str] | None = None) -> QueryPlan:
    return QueryPlan(intent=intent, intent_confidence=0.9, scope=Scope.SINGLE_TRIAL, nct_ids=ncts or [NO_RESULTS])


# -- end to end (rules planner, offline) ------------------------------------------------------------


async def test_stale_nct_with_drug_and_condition_answers_the_cohort(env):
    resp = await _orch(env).run(VisualizeRequest(query=QUERY, nct_id=NO_RESULTS))
    viz = resp.visualization
    assert viz is not None and viz.type in (VizType.SURVIVAL_CURVE, VizType.FOREST_PLOT)
    cited = {c.nct_id for d in viz.data for c in d.citations}
    assert cited and cited <= set(SURV)  # the stale trial contributes nothing
    interp = resp.meta.interpretation
    assert interp.scope is Scope.COHORT and not interp.nct_ids
    msg = next(n.message for n in resp.meta.notes if n.code is NoteCode.NO_RESULTS_POSTED and NO_RESULTS in n.message)
    assert msg == (
        f"{NO_RESULTS} has no results posted on ClinicalTrials.gov (hasResults: false), so it cannot answer this; "
        "showing trials of pembrolizumab in melanoma that do."
    )
    assert any(NO_RESULTS in a for a in interp.assumptions)


async def test_stale_nct_without_entities_keeps_the_trial_and_says_no_results(env):
    resp = await _orch(env).run(VisualizeRequest(query="Show me the Kaplan Meier curve", nct_id=NO_RESULTS))
    viz = resp.visualization
    assert viz is not None and viz.title.startswith("No results posted for")
    assert {c.nct_id for d in viz.data for c in d.citations} == {NO_RESULTS}
    assert resp.meta.interpretation.nct_ids == [NO_RESULTS]
    assert any(n.code is NoteCode.NO_RESULTS_POSTED for n in resp.meta.notes)


async def test_trial_with_results_is_never_swapped(env):
    resp = await _orch(env).run(VisualizeRequest(query=QUERY, nct_id="NCT02578680"))
    assert resp.meta.interpretation.nct_ids == ["NCT02578680"]
    assert resp.visualization.type is VizType.SURVIVAL_CURVE
    assert {c.nct_id for d in resp.visualization.data for c in d.citations} == {"NCT02578680"}


# -- the pure decision --------------------------------------------------------------------------------


def _decide(intent=Intent.SURVIVAL_SUMMARY, trials=(NO_RESULTS,), entities=None):
    plan = _plan(intent, list(trials))
    studies = [Study(load_study_raw(n)) for n in trials]
    ents = entities if entities is not None else [_entity("drug", "nivolumab", ["nivolumab", "opdivo"])]
    return cohort_instead(plan, get_intent_spec(intent), studies, ents, VisualizeRequest(query="q"))


@pytest.mark.parametrize("intent", [Intent.SURVIVAL_SUMMARY, Intent.RESULTS_OUTCOMES, Intent.DROPOUT_FLOW])
def test_results_only_intents_swap(intent):
    swap = _decide(intent)
    assert swap is not None
    plan, n = swap
    assert plan.nct_ids == [] and plan.scope is Scope.COHORT and plan.intent is intent
    assert plan.cohorts[0].drugs == ["nivolumab"] and n.code is NoteCode.NO_RESULTS_POSTED


@pytest.mark.parametrize("intent", [Intent.RISKS_ADVERSE_EVENTS, Intent.DEMOGRAPHICS, Intent.SINGLE_TRIAL_OVERVIEW])
def test_intents_with_a_protocol_answer_keep_the_trial(intent):
    assert _decide(intent) is None  # risk statements / eligibility still describe the named trial


def test_no_swap_without_drug_or_condition_or_when_any_trial_has_results():
    assert _decide(entities=[]) is None
    assert _decide(entities=[_entity("sponsor", "Merck")]) is None
    assert _decide(trials=(NO_RESULTS, "NCT02578680")) is None


def test_drug_class_searches_its_members_and_condition_label():
    pd1 = ResolvedEntity(
        kind="drug", input="PD-1 inhibitors", resolved="PD-1 inhibitors", method="alias",
        search_terms=["pembrolizumab", "nivolumab"],
    )  # fmt: skip
    plan, n = _decide(entities=[pd1, _entity("condition", "melanoma")])
    c = plan.cohorts[0]
    assert c.drugs == ["pembrolizumab", "nivolumab"] and c.conditions == ["melanoma"]
    assert c.label == "PD-1 inhibitors in melanoma"
    assert n.message.endswith("showing trials of PD-1 inhibitors in melanoma that do.")
