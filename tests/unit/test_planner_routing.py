"""Routing fixes from user testing round 2 (rules planner + LLM plan validator, offline / stub LLM).

H3 chart words and preferred_visualization steer the intent · M7 broad landscape questions · H2 gibberish and
non-English input never become "All trials" · M1 heatmap rows · M2 geography · M3 networks · M8 sponsor
timelines · M4 stop-words in entity capture · CAR-T aliases · H1 a slow / failing LLM falls back to the rule plan.
"""

from __future__ import annotations

import asyncio

import pytest

from ctviz.agents import planner as planner_mod
from ctviz.agents.llm import StubLLM
from ctviz.agents.planner import LLMCohort, LLMPlan, LLMSpan, plan_query, planner_deadline_s
from ctviz.agents.rules_planner import (
    chart_intents,
    detect_chart,
    extract,
    heatmap_dims,
    is_non_english,
    rule_plan,
    steer_by_chart,
)
from ctviz.api.schemas import Intent, Persona, Scope, VisualizeRequest
from ctviz.api.schemas.viz import VizType
from ctviz.core.background import drain
from ctviz.core.budget import budget_scope
from ctviz.core.intents import load_intents
from ctviz.core.orchestrator import Orchestrator
from ctviz.obs.trace import trace_scope
from ctviz.store.db import Database
from tests.unit.test_orchestrator import FakeClient

IN, S, V = Intent, Scope, VizType
NCT = "NCT03548935"


def _plan(q: str, **kw):
    return rule_plan(VisualizeRequest(query=q, **kw))


def cohort(**kw) -> LLMCohort:
    base = dict(
        label="X",
        drugs=[],
        conditions=[],
        sponsors=[],
        countries=[],
        phases=[],
        statuses=[],
        start_year=None,
        end_year=None,
    )
    return LLMCohort(**{**base, **kw})


def llm_plan(**kw) -> LLMPlan:
    base = dict(
        intent=IN.TREND_OVER_TIME,
        intent_confidence=0.9,
        persona=Persona.RESEARCHER,
        scope=S.COHORT,
        nct_ids=[],
        cohorts=[],
        group_by=None,
        series_by=None,
        network_kind=None,
        time_granularity=None,
        top_n=None,
        preferred_visualization=None,
        focus_terms=[],
        ambiguous_entities=[],
        assumptions=[],
        clarification=None,
    )
    return LLMPlan(**{**base, **kw})


def stub(result) -> StubLLM:
    def responder(user, schema):
        if isinstance(result, Exception):
            raise result
        return result

    return StubLLM({"planner": responder})


async def run(req: VisualizeRequest, llm, settings):
    with trace_scope() as tr:
        plan, ents = await plan_query(req, settings=settings, llm=llm)
    return plan, ents, tr


# ------------------------------------------------------------------------------------------- H3 chart words
@pytest.mark.parametrize(
    ("text", "viz"),
    [
        ("show me a sankey of dropouts", V.SANKEY),
        ("network of obesity trials", V.NETWORK_GRAPH),
        ("map of semaglutide trials", V.GEO_MAP),
        ("heat map of phases", V.HEATMAP),
        ("heatmap", V.HEATMAP),
        ("timeline of Novo Nordisk trials", V.TIMELINE),
        ("kaplan meier for X", V.SURVIVAL_CURVE),
        ("survival curve", V.SURVIVAL_CURVE),
        ("forest plot", V.FOREST_PLOT),
        ("histogram of enrollment", V.HISTOGRAM),
        ("start date scatter", V.SCATTER_PLOT),
        ("pie chart of phases", V.PIE_CHART),
        ("phases per year", None),
        ("roadmap of trials", None),
        ("trials that map to MeSH", None),
    ],
)
def test_detect_chart(text, viz):
    assert detect_chart(text) is viz


def test_chart_intents_come_from_intents_yaml_primary_first():
    specs = load_intents()
    for viz in VizType:
        got = chart_intents(viz)
        assert set(got) == {i for i, s in specs.items() if viz in s.compatible_viz}
        primaries = [i for i in got if specs[i].primary is viz]
        assert got[: len(primaries)] == primaries
    assert chart_intents(V.SANKEY)[0] is IN.DROPOUT_FLOW
    assert chart_intents(V.NETWORK_GRAPH) == [IN.ENTITY_NETWORK]


@pytest.mark.parametrize(
    ("question", "intent", "scope"),
    [
        ("show me a sankey of dropouts in NCT03548935", IN.DROPOUT_FLOW, S.SINGLE_TRIAL),
        ("show me a sankey of NCT03548935", IN.DROPOUT_FLOW, S.SINGLE_TRIAL),
        ("forest plot for NCT03548935", IN.RESULTS_OUTCOMES, S.SINGLE_TRIAL),
        ("kaplan meier curve for NCT02142738", IN.SURVIVAL_SUMMARY, S.SINGLE_TRIAL),
        ("timeline of NCT03548935", IN.TRIAL_JOURNEY_TIMELINE, S.SINGLE_TRIAL),
        ("map of semaglutide trials", IN.GEOGRAPHIC_DISTRIBUTION, S.COHORT),
        ("histogram of enrollment for obesity trials", IN.ENROLLMENT_DISTRIBUTION, S.COHORT),
        ("enrollment vs start date scatter for obesity trials", IN.ENROLLMENT_DISTRIBUTION, S.COHORT),
        ("network of semaglutide trials", IN.ENTITY_NETWORK, S.COHORT),
        ("pie chart of phases for semaglutide trials", IN.PHASE_DISTRIBUTION, S.COHORT),
        ("timeline of Novo Nordisk trials", IN.SPONSOR_ACTIVITY, S.COHORT),  # M8
        ("gantt chart of Eli Lilly obesity trials", IN.SPONSOR_ACTIVITY, S.COHORT),
    ],
)
def test_chart_words_steer_intent(question, intent, scope):
    plan = _plan(question)
    assert (plan.intent, plan.scope) == (intent, scope)
    assert plan.intent_confidence >= 0.6 and not plan.clarification
    viz = detect_chart(question)
    assert viz in load_intents()[plan.intent].compatible_viz
    assert plan.preferred_visualization is viz


@pytest.mark.parametrize(
    ("viz", "fields", "intent"),
    [
        ("network_graph", {"drug_name": "semaglutide"}, IN.ENTITY_NETWORK),
        ("geo_map", {"condition": "obesity"}, IN.GEOGRAPHIC_DISTRIBUTION),
        ("sankey", {"nct_id": NCT}, IN.DROPOUT_FLOW),
        ("survival_curve", {"nct_id": NCT}, IN.SURVIVAL_SUMMARY),
        ("heatmap", {"condition": "obesity"}, IN.PHASE_STATUS_HEATMAP),
        ("histogram", {"condition": "obesity"}, IN.ENROLLMENT_DISTRIBUTION),
        ("timeline", {"sponsor": "Novo Nordisk"}, IN.SPONSOR_ACTIVITY),
        ("power_curve", {"nct_id": NCT}, IN.ENROLLMENT_VS_POWER),
    ],
)
def test_preferred_visualization_steers_intent(viz, fields, intent):
    plan = _plan("trials", preferred_visualization=viz, **fields)
    assert plan.intent is intent
    assert plan.preferred_visualization == viz
    assert VizType(viz) in load_intents()[intent].compatible_viz


def test_structured_preference_beats_chart_word():
    plan = _plan("map of obesity trials", preferred_visualization="network_graph")
    assert plan.intent is IN.ENTITY_NETWORK and plan.preferred_visualization == "network_graph"


def test_generic_chart_words_never_change_the_intent():
    plan = _plan("bar chart of phases for obesity trials")
    assert plan.intent is IN.PHASE_DISTRIBUTION and plan.preferred_visualization == "bar_chart"
    plan = _plan("Trials per year for obesity", preferred_visualization="table")
    assert plan.intent is IN.TREND_OVER_TIME


def test_chart_that_no_scope_compatible_intent_draws_is_kept_for_the_ignored_note():
    plan = _plan("network for NCT03548935")  # networks are cohort-only: keep a trial intent, say it was not drawn
    assert plan.scope is S.SINGLE_TRIAL and plan.intent is not IN.ENTITY_NETWORK
    assert plan.preferred_visualization == "network_graph"
    plan = _plan("timeline of trials for obesity")  # 'timeline of trials' is the trend's own wording
    assert plan.intent is IN.TREND_OVER_TIME and plan.preferred_visualization is None


def test_steer_requires_sponsor_or_drug_for_timelines():
    kw = dict(trial_ctx=False, cohort_ctx=True)
    assert (
        steer_by_chart(IN.TREND_OVER_TIME, V.TIMELINE, "x", has_sponsor=True, has_drug=False, **kw)
        is IN.SPONSOR_ACTIVITY
    )
    assert (
        steer_by_chart(IN.TREND_OVER_TIME, V.TIMELINE, "x", has_sponsor=False, has_drug=True, **kw)
        is IN.DEVELOPMENT_PROGRESSION
    )
    assert steer_by_chart(IN.TREND_OVER_TIME, V.TIMELINE, "x", has_sponsor=False, has_drug=False, **kw) is None
    assert steer_by_chart(IN.PHASE_DISTRIBUTION, V.PIE_CHART, "x", has_sponsor=False, has_drug=False, **kw) is None


# ------------------------------------------------------------------------------------------- M1 heatmap rows
@pytest.mark.parametrize(
    ("question", "group_by", "series_by"),
    [
        ("heatmap of sponsors by phase for obesity", "sponsor", "phase"),
        ("drugs by status heatmap for melanoma", "intervention", "status"),
        ("heatmap of phase by country for semaglutide trials", "country", "phase"),
        ("matrix of conditions vs start year for semaglutide", "condition", "year"),
        ("heatmap of sponsor class by phase for obesity", "sponsor_class", "phase"),
    ],
)
def test_heatmap_with_rows_is_gap_heatmap_with_dims(question, group_by, series_by):
    plan = _plan(question)
    assert plan.intent is IN.LANDSCAPE_GAP_HEATMAP
    assert (plan.group_by, plan.series_by) == (group_by, series_by)
    assert plan.cohorts[0].phases == []  # never filter the charted dimension


def test_phase_by_status_heatmap_unchanged():
    plan = _plan("Heatmap of phase by recruitment status across all GLP-1 trials")
    assert plan.intent is IN.PHASE_STATUS_HEATMAP and plan.group_by is None
    assert heatmap_dims("heatmap of phase by status") == (None, None)


# ------------------------------------------------------------------------------------------- M7 landscapes
@pytest.mark.parametrize(
    ("question", "intent", "kind"),
    [
        ("lung cancer trials", IN.ENTITY_NETWORK, "sponsor_drug"),
        ("diabetes drug landscape", IN.ENTITY_NETWORK, "sponsor_drug"),
        ("overview of obesity research", IN.ENTITY_NETWORK, "sponsor_drug"),
        ("tell me about lung cancer trials", IN.ENTITY_NETWORK, "sponsor_drug"),
        ("tell me about semaglutide", IN.ENTITY_NETWORK, "drug_condition"),
        ("Pfizer trials", IN.SPONSOR_ACTIVITY, None),
    ],
)
def test_broad_questions_get_a_landscape(question, intent, kind):
    plan = _plan(question)
    assert plan.intent is intent and plan.scope is S.COHORT and plan.network_kind == kind
    assert plan.intent_confidence >= 0.6 and not plan.clarification
    assert any(a.startswith("Broad question") for a in plan.assumptions)


@pytest.mark.parametrize(
    "question",
    [
        "Tell me about NCT03548935",  # one trial
        "how many trials for lung cancer",  # a count
        "Tell me about this trial for obesity",  # a specific trial
        "I have lung cancer, what trials are there?",  # patients ask about joining, not landscapes
    ],
)
def test_not_landscapes(question):
    plan = _plan(question)
    assert plan.intent not in (IN.ENTITY_NETWORK, IN.SPONSOR_ACTIVITY)
    assert not any(a.startswith("Broad question") for a in plan.assumptions)


# ------------------------------------------------------------------------------------------- H2 gibberish
@pytest.mark.parametrize(
    "question",
    [
        "asdf qwerty",
        "?????",
        "hello",
        "¿Cuántos ensayos de semaglutida hay?",
        "Combien d'essais cliniques sur le diabète?",
        "Wie viele Studien gibt es?",
    ],
)
def test_unrecognizable_questions_get_no_cohort(question):
    plan = _plan(question)
    assert plan.cohorts == [] and not plan.nct_ids
    assert plan.clarification and "drug, condition, sponsor or trial" in plan.clarification
    assert not any("all registered trials" in a for a in plan.assumptions)


@pytest.mark.parametrize(
    "question",
    [
        "how many trials started each year since 2020",
        "Which drugs frequently co-occur in combination studies?",
        "recruiting sites in Texas",
    ],
)
def test_real_all_trials_questions_still_answered(question):
    plan = _plan(question)
    assert [c.label for c in plan.cohorts] == ["All trials"]


def test_structured_fields_or_entities_make_a_question_intelligible():
    assert _plan("asdf", condition="obesity").cohorts[0].conditions == ["obesity"]
    assert _plan("asdf", start_year=2020).cohorts[0].start_year == 2020
    assert _plan("¿Cuántos ensayos de semaglutide hay?").cohorts[0].drugs == ["semaglutide"]


def test_is_non_english():
    assert is_non_english("¿Cuántos ensayos de semaglutida hay?")
    assert is_non_english("Combien d'essais cliniques sur le diabète?")
    assert not is_non_english("How many trials for Sjögren's syndrome started per year?")
    assert not is_non_english("trials in the US")


@pytest.fixture
async def env(settings, tmp_path):
    db = Database(tmp_path / "o.db").connect()
    yield settings, db, FakeClient(db)
    await drain()
    db.close()


@pytest.mark.parametrize("question", ["asdf qwerty", "¿Cuántos ensayos de semaglutida hay?"])
async def test_gibberish_end_to_end_asks_instead_of_all_trials(env, question):
    settings, db, client = env
    o = Orchestrator(settings, db, client, StubLLM(), runs_dir=None, use_response_cache=False)  # rules planner
    resp = await o.run(VisualizeRequest(query=question))
    assert resp.status == "needs_clarification" and resp.visualization is None
    assert resp.clarification.reason == "no_entity"
    assert client.search_calls == []  # nothing was searched, least of all every trial


# ------------------------------------------------------------------------------------------- M2 geography
@pytest.mark.parametrize(
    ("question", "statuses"),
    [
        ("where are obesity trials recruiting", ["RECRUITING"]),
        ("Where are semaglutide trials being run?", []),
        ("recruiting sites in Texas for obesity", ["RECRUITING"]),
        ("obesity trials near Boston", []),
    ],
)
def test_where_and_near_questions_map_sites(question, statuses):
    plan = _plan(question)
    assert plan.intent is IN.GEOGRAPHIC_DISTRIBUTION and plan.scope is S.COHORT
    assert plan.cohorts[0].statuses == statuses
    assert plan.cohorts[0].countries == []  # a city/state is never passed as a country


def test_city_is_reported_not_filtered():
    plan = _plan("obesity trials near Boston")
    assert any("'Boston' is not a country" in a for a in plan.assumptions)
    plan = _plan("Which countries have the most recruiting trials for type 2 diabetes in Germany?")
    assert not any("is not a country" in a for a in plan.assumptions)


def test_patient_where_questions_unchanged():
    plan = _plan("Where are the sites for NCT03548935 being run?")
    assert plan.scope is S.SINGLE_TRIAL and plan.intent is IN.SITES_AND_CONTACTS
    assert _plan("Is there a site near me?", nct_id=NCT).intent is IN.SITES_AND_CONTACTS


# ------------------------------------------------------------------------------------------- M3 networks
@pytest.mark.parametrize(
    ("question", "kind"),
    [
        ("drugs combined with pembrolizumab", "drug_drug"),
        ("what is pembrolizumab combined with", "drug_drug"),
        ("what is nivolumab paired with in melanoma trials", "drug_drug"),
        ("how do sponsors and drugs connect in obesity", "sponsor_drug"),
        ("relationship between sponsors and drugs for obesity", "sponsor_drug"),
    ],
)
def test_combination_and_relationship_questions_are_networks(question, kind):
    plan = _plan(question)
    assert plan.intent is IN.ENTITY_NETWORK and plan.network_kind == kind and plan.intent_confidence >= 0.8


# ------------------------------------------------------------------------------------------- M4 stop-words
@pytest.mark.parametrize(
    "question",
    [
        "which conditions are studied the most with semaglutide",
        "enrollment vs start date scatter for obesity trials",
        "start date scatter of obesity studies",
        "what diseases are tested in tirzepatide studies",
    ],
)
def test_verbs_and_chart_words_are_not_entities(question):
    ex = extract(VisualizeRequest(query=question))
    names = {f.entity.resolved.lower() for f in ex.found}
    assert not {n for n in names if any(w in n for w in ("studied", "studies", "scatter", "date", "tested"))}


# ------------------------------------------------------------------------------------------- CAR-T aliases
@pytest.mark.parametrize(
    "surface", ["CAR-T", "CAR T cell therapy", "chimeric antigen receptor T cell therapy", "car-t cells"]
)
def test_car_t_is_a_drug_class(surface):
    plan = _plan(f"How many {surface} trials started per year?")
    c = plan.cohorts[0]
    assert c.label == "CAR-T cell therapies" and c.conditions == []
    assert {
        "tisagenlecleucel",
        "axicabtagene ciloleucel",
        "lisocabtagene maraleucel",
        "idecabtagene vicleucel",
        "ciltacabtagene autoleucel",
        "brexucabtagene autoleucel",
    } == set(c.drugs)


def test_car_t_brand_names():
    assert _plan("Kymriah trials per year").cohorts[0].drugs == ["tisagenlecleucel"]


# ------------------------------------------------------------------------------------------- validator guards
async def test_validator_keeps_the_requested_chart(settings):
    req = VisualizeRequest(query="show me a sankey of dropouts in NCT03548935")
    plan, _, tr = await run(
        req, stub(llm_plan(intent=IN.SINGLE_TRIAL_OVERVIEW, scope=S.SINGLE_TRIAL, nct_ids=[NCT])), settings
    )
    assert plan.parser == "llm" and plan.intent is IN.DROPOUT_FLOW
    assert plan.preferred_visualization is V.SANKEY


async def test_validator_heatmap_rows_use_gap_heatmap(settings):
    req = VisualizeRequest(query="heatmap of sponsors by phase for obesity")
    plan, _, _ = await run(
        req,
        stub(llm_plan(intent=IN.PHASE_STATUS_HEATMAP, cohorts=[cohort(label="Obesity", conditions=["obesity"])])),
        settings,
    )
    assert plan.intent is IN.LANDSCAPE_GAP_HEATMAP
    assert (plan.group_by, plan.series_by) == ("sponsor", "phase")


async def test_validator_broad_question_is_not_a_single_trial(settings):
    req = VisualizeRequest(query="tell me about lung cancer trials")
    plan, _, _ = await run(
        req,
        stub(
            llm_plan(
                intent=IN.SINGLE_TRIAL_OVERVIEW,
                scope=S.SINGLE_TRIAL,
                cohorts=[cohort(label="Lung cancer", conditions=["lung cancer"])],
            )
        ),
        settings,
    )
    assert plan.intent is IN.ENTITY_NETWORK and plan.scope is S.COHORT and not plan.clarification
    assert any(a.startswith("Broad question") for a in plan.assumptions)  # the overview is never silent


async def test_validator_never_invents_all_trials_for_gibberish(settings):
    req = VisualizeRequest(query="asdf qwerty")
    for cohorts in ([], [cohort(label="All trials")]):
        plan, _, _ = await run(req, stub(llm_plan(cohorts=cohorts)), settings)
        assert plan.parser == "llm" and plan.cohorts == [] and plan.clarification
        assert not any("all registered trials" in a for a in plan.assumptions)


async def test_validator_keeps_a_translated_entity_with_a_verbatim_span(settings):
    req = VisualizeRequest(query="¿Cuántos ensayos de semaglutida hay?")
    c = cohort(
        label="Semaglutide",
        drugs=["semaglutide"],
        entity_spans=[LLMSpan(kind="drug", name="semaglutide", source_text="semaglutida")],
    )
    plan, _, _ = await run(req, stub(llm_plan(cohorts=[c])), settings)
    assert plan.cohorts[0].drugs == ["semaglutide"]
    assert any("semaglutida" in a for a in plan.assumptions)  # the reading is shown, never silent


# ------------------------------------------------------------------------------------------- H1 fallback
class SlowLLM(StubLLM):
    """Never answers in time."""

    async def structured(self, **kw):  # noqa: ANN003
        await asyncio.sleep(30)
        raise AssertionError("unreachable")


async def test_slow_llm_falls_back_to_the_rule_plan_inside_the_orchestrator_timeout(settings):
    s = settings.model_copy(update={"request_deadline_s": 4.6, "llm_timeout_s": 30.0})
    req = VisualizeRequest(query="where are obesity trials recruiting")
    with budget_scope(s) as budget, trace_scope() as tr:
        deadline = planner_deadline_s(s)
        orchestrator_timeout = max(0.5, min(budget.remaining_s() - 3.0, s.llm_timeout_s + 4.0))
        assert deadline < orchestrator_timeout
        plan, _ = await asyncio.wait_for(plan_query(req, settings=s, llm=SlowLLM()), timeout=orchestrator_timeout)
    assert plan.parser == "rules" and plan.intent is IN.GEOGRAPHIC_DISTRIBUTION
    assert plan.cohorts[0].conditions == ["obesity"]
    step = next(st for st in tr.steps if st.name == "plan")
    assert step.detail.get("fallback_reason") == "timeout"


async def test_llm_error_and_repair_bug_fall_back_to_the_rule_plan(settings, monkeypatch):
    from ctviz.agents.llm import LLMError

    req = VisualizeRequest(query="timeline of Novo Nordisk trials")
    plan, _, _ = await run(req, stub(LLMError("planner: LLM timed out")), settings)
    assert plan.parser == "rules" and plan.intent is IN.SPONSOR_ACTIVITY

    def boom(self, p):  # noqa: ANN001
        raise KeyError("bug")

    monkeypatch.setattr(planner_mod.PlanValidator, "repair", boom)
    plan, _, _ = await run(req, stub(llm_plan(cohorts=[cohort(sponsors=["Novo Nordisk"])])), settings)
    assert plan.parser == "rules" and plan.intent is IN.SPONSOR_ACTIVITY


async def test_escalation_skipped_when_no_time_is_left(settings, monkeypatch):
    calls: list[str] = []

    def responder(user, schema):
        calls.append("x")
        return llm_plan(intent_confidence=0.2, cohorts=[cohort(conditions=["obesity"])])

    monkeypatch.setattr(planner_mod, "ESCALATION_MIN_S", 10_000.0)
    req = VisualizeRequest(query="obesity trials per year")
    plan, _, tr = await run(req, StubLLM({"planner": responder}), settings)
    assert calls == ["x"] and plan.parser == "llm"
    assert next(st for st in tr.steps if st.name == "plan").detail.get("escalation_skipped")


async def test_validator_carries_the_city_note(settings):
    req = VisualizeRequest(query="obesity trials near Boston")
    plan, _, _ = await run(
        req,
        stub(
            llm_plan(
                intent=IN.GEOGRAPHIC_DISTRIBUTION,
                cohorts=[cohort(label="Obesity", conditions=["obesity"], countries=["Boston"])],
            )
        ),
        settings,
    )
    assert plan.intent is IN.GEOGRAPHIC_DISTRIBUTION and plan.cohorts[0].countries == []
    assert plan.cohorts[0].conditions == ["obesity"]
    assert any("'Boston' is not a country" in a for a in plan.assumptions)
