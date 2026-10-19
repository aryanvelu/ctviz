"""LLM planner path with StubLLM: success, escalation, fallbacks, validator repairs, plan cache."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

from ctviz.agents.llm import LLMError, LLMUnavailable, StubLLM
from ctviz.agents.planner import LLMCohort, LLMPlan, plan_cache_key, plan_query, system_prompt
from ctviz.api.schemas import Intent, Persona, Scope, VisualizeRequest
from ctviz.core.intents import load_intents
from ctviz.obs.trace import trace_scope
from ctviz.store.db import Database


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
        intent=Intent.TREND_OVER_TIME,
        intent_confidence=0.9,
        persona=Persona.RESEARCHER,
        scope=Scope.COHORT,
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


class Recorder:
    """StubLLM responder returning queued results (or raising queued exceptions); records calls."""

    def __init__(self, *items):
        self.items = list(items)
        self.users: list[dict] = []

    def __call__(self, user: str, schema: type[BaseModel]):
        self.users.append(json.loads(user))
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, schema)
        return item


class ModelSpy(StubLLM):
    def __init__(self, responders):
        super().__init__(responders)
        self.models: list[str] = []

    async def structured(self, *, model, **kw):  # noqa: ANN001
        self.models.append(model)
        return await super().structured(model=model, **kw)


async def run(req: VisualizeRequest, llm, settings, db=None):
    with trace_scope() as tr:
        plan, ents = await plan_query(req, settings=settings, llm=llm, db=db)
    return plan, ents, tr


def test_prompt_lists_every_intent_and_is_static():
    p = system_prompt()
    for intent, spec in load_intents().items():
        assert f"`{intent.value}`" in p
        assert spec.description in p
    assert "{intents}" not in p and "{{" not in p  # fully rendered template
    assert system_prompt() is p


async def test_llm_plan_used_and_validated(settings):
    q = "How has the number of trials for pembrolizumab changed per year since 2015?"
    rec = Recorder(
        llm_plan(
            cohorts=[cohort(label="Pembrolizumab", drugs=["pembrolizumab"], start_year=2015)], time_granularity="year"
        )
    )
    plan, ents, tr = await run(VisualizeRequest(query=q), StubLLM({"planner": rec}), settings)
    assert plan.parser == "llm" and plan.intent is Intent.TREND_OVER_TIME and plan.scope is Scope.COHORT
    assert plan.cohorts[0].drugs == ["pembrolizumab"] and plan.cohorts[0].start_year == 2015
    assert any(e.resolved == "pembrolizumab" for e in ents)
    # user message carries grounded hints, not the system prompt
    assert rec.users[0]["entity_hints"][0]["resolved"] == "pembrolizumab"
    step = next(s for s in tr.steps if s.name == "plan")
    assert step.detail["parser"] == "llm"


async def test_unavailable_falls_back_to_rules(settings):
    plan, _, tr = await run(
        VisualizeRequest(query="How are breast cancer trials distributed across phases?"), StubLLM(), settings
    )
    assert plan.parser == "rules" and plan.intent is Intent.PHASE_DISTRIBUTION
    assert next(s for s in tr.steps if s.name == "plan").detail["fallback_reason"] == "LLMUnavailable"


async def test_transport_error_falls_back_without_escalation(settings):
    llm = ModelSpy({"planner": Recorder(LLMError("planner: LLM timed out"))})
    plan, _, _ = await run(VisualizeRequest(query="Trials per year for obesity"), llm, settings)
    assert plan.parser == "rules" and llm.models == [settings.model_planner]


async def test_low_confidence_escalates_once(settings):
    low = llm_plan(
        intent=Intent.STATUS_DISTRIBUTION,
        intent_confidence=0.3,
        cohorts=[cohort(label="Obesity", conditions=["obesity"])],
    )
    high = llm_plan(
        intent=Intent.TREND_OVER_TIME, intent_confidence=0.85, cohorts=[cohort(label="Obesity", conditions=["obesity"])]
    )
    llm = ModelSpy({"planner": Recorder(low, high)})
    plan, _, tr = await run(VisualizeRequest(query="obesity trials over time"), llm, settings)
    assert llm.models == [settings.model_planner, settings.model_escalation]
    assert plan.intent is Intent.TREND_OVER_TIME and plan.parser == "llm"
    assert next(s for s in tr.steps if s.name == "plan").detail["escalation_reason"] == "low_confidence"


async def test_schema_failure_escalates_then_rules_if_both_fail(settings):
    try:
        LLMPlan.model_validate({"intent": "nope"})
    except ValidationError as ve:
        err = LLMError("planner: LLM call failed (ValidationError)")
        err.__cause__ = ve
    llm = ModelSpy({"planner": Recorder(err, LLMError("planner: model returned no parsable output"))})
    plan, _, _ = await run(
        VisualizeRequest(query="How are breast cancer trials distributed across phases?"), llm, settings
    )
    assert len(llm.models) == 2 and plan.parser == "rules"


async def test_schema_failure_then_escalation_success(settings):
    good = llm_plan(intent=Intent.PHASE_DISTRIBUTION, cohorts=[cohort(label="BC", conditions=["breast cancer"])])
    llm = ModelSpy({"planner": Recorder(LLMError("planner: model returned no parsable output"), good)})
    plan, _, _ = await run(
        VisualizeRequest(query="How are breast cancer trials distributed across phases?"), llm, settings
    )
    assert plan.parser == "llm" and plan.intent is Intent.PHASE_DISTRIBUTION


async def test_validator_drops_hallucinated_entities_and_ncts(settings):
    bad = llm_plan(
        intent=Intent.PHASE_DISTRIBUTION,
        nct_ids=["NCT99999999"],
        cohorts=[
            cohort(
                label="BC",
                conditions=["breast cancer", "ovarian cancer"],
                drugs=["trastuzumab"],
                phases=["PHASE3"],
                start_year=1900,
                end_year=2999,
            )
        ],
    )
    plan, _, tr = await run(
        VisualizeRequest(query="How are breast cancer trials distributed across phases?"),
        StubLLM({"planner": Recorder(bad)}),
        settings,
    )
    (c,) = plan.cohorts
    assert c.conditions == ["breast cancer"] and c.drugs == []
    assert plan.nct_ids == []
    assert c.start_year is None and c.end_year is None  # years not in the request are dropped
    repairs = next(s for s in tr.steps if s.name == "plan").detail["repairs"]
    assert any("dropped ungrounded condition" in r for r in repairs) and any("NCT ids" in r for r in repairs)
    assert "ovarian" not in json.dumps(repairs)  # LLM text is never echoed into the response/trace
    assert plan.intent_confidence < 0.9


async def test_validator_comparison_needs_two_cohorts(settings):
    p = llm_plan(
        intent=Intent.PHASE_DISTRIBUTION, scope=Scope.COMPARISON, cohorts=[cohort(label="Sema", drugs=["semaglutide"])]
    )
    plan, _, _ = await run(
        VisualizeRequest(query="phases of semaglutide trials"), StubLLM({"planner": Recorder(p)}), settings
    )
    assert plan.scope is Scope.COHORT


async def test_validator_scope_from_nct(settings):
    p = llm_plan(
        intent=Intent.ELIGIBILITY_CHECK,
        scope=Scope.COHORT,
        persona=Persona.PATIENT,
        cohorts=[cohort(label="Obesity", conditions=["obesity"])],
    )
    plan, _, _ = await run(
        VisualizeRequest(query="Am I eligible?", nct_id="NCT03548935"), StubLLM({"planner": Recorder(p)}), settings
    )
    assert plan.scope is Scope.SINGLE_TRIAL and plan.nct_ids == ["NCT03548935"] and plan.cohorts == []


async def test_validator_replaces_cohort_only_intent_for_trial(settings):
    p = llm_plan(intent=Intent.PHASE_DISTRIBUTION, scope=Scope.SINGLE_TRIAL, nct_ids=["NCT03548935"])
    plan, _, _ = await run(
        VisualizeRequest(query="Which phase is this trial and what endpoints are measured?", nct_id="NCT03548935"),
        StubLLM({"planner": Recorder(p)}),
        settings,
    )
    assert plan.intent is Intent.ENDPOINTS_CATALOG and plan.scope is Scope.SINGLE_TRIAL


async def test_validator_merges_for_non_comparison_intent(settings):
    p = llm_plan(
        intent=Intent.ENTITY_NETWORK,
        scope=Scope.COMPARISON,
        network_kind=None,
        cohorts=[cohort(label="A", drugs=["semaglutide"]), cohort(label="B", drugs=["tirzepatide"])],
    )
    plan, _, _ = await run(
        VisualizeRequest(query="network of sponsors for semaglutide vs tirzepatide"),
        StubLLM({"planner": Recorder(p)}),
        settings,
    )
    assert plan.scope is Scope.COHORT and len(plan.cohorts) == 1
    assert plan.network_kind == "sponsor_drug" and plan.cohorts[0].interventional_only


async def test_validator_keeps_class_expansion_and_structured_override(settings):
    p = llm_plan(intent=Intent.PHASE_STATUS_HEATMAP, cohorts=[cohort(label="GLP-1", drugs=["GLP-1"])])
    plan, _, _ = await run(
        VisualizeRequest(
            query="Phase vs recruitment status heatmap across GLP-1 trials", trial_phase=["2", "3"], persona="patient"
        ),
        StubLLM({"planner": Recorder(p)}),
        settings,
    )
    (c,) = plan.cohorts
    assert "semaglutide" in c.drugs and "tirzepatide" in c.drugs
    assert c.phases == ["PHASE2", "PHASE3"] and plan.persona is Persona.PATIENT


async def test_structured_drug_names_override_llm(settings):
    p = llm_plan(intent=Intent.TREND_OVER_TIME, cohorts=[cohort(label="X", drugs=["insulin"])])
    plan, _, _ = await run(
        VisualizeRequest(query="trials per year", drug_names=["Ozempic", "Mounjaro"]),
        StubLLM({"planner": Recorder(p)}),
        settings,
    )
    assert plan.scope is Scope.COMPARISON
    assert [c.drugs for c in plan.cohorts] == [["semaglutide"], ["tirzepatide"]]


async def test_plan_cache_roundtrip(settings, tmp_path):
    db = Database(tmp_path / "c.db").connect()
    req = VisualizeRequest(query="Trials per year for obesity")
    rec = Recorder(llm_plan(cohorts=[cohort(label="Obesity", conditions=["obesity"])]))
    p1, _, _ = await run(req, StubLLM({"planner": rec}), settings, db)
    p2, e2, tr = await run(VisualizeRequest(query="  Trials per year   for obesity "), StubLLM(), settings, db)
    assert p1.parser == "llm" and p2.parser == "cache" and p2.intent is p1.intent
    assert any(e.resolved == "obesity" for e in e2)
    p3, _, _ = await run(VisualizeRequest(query="Trials per year for obesity", refresh=True), StubLLM(), settings, db)
    assert p3.parser == "rules"
    db.close()


def test_cache_key_ignores_presentation_flags():
    a = plan_cache_key(VisualizeRequest(query="X  y", narrative=False))
    b = plan_cache_key(VisualizeRequest(query=" X y ", include_trace=False))
    assert a == b != plan_cache_key(VisualizeRequest(query="X y", drug_name="z"))
    assert a != plan_cache_key(VisualizeRequest(query="x y"))  # case is kept (MS != ms)
    assert a != plan_cache_key(VisualizeRequest(query="X y"), model="gpt-5.4-mini")


@pytest.mark.parametrize("exc", [LLMUnavailable("off"), LLMError("boom")])
async def test_never_raises(settings, exc):
    plan, _, _ = await run(VisualizeRequest(query="?"), StubLLM({"planner": Recorder(exc)}), settings)
    assert plan.parser == "rules"


# -- answerability: only the deterministic check may refuse -----------------------------------------
async def test_llm_only_subjective_flag_is_overruled(settings):
    q = "What is the overall survival across pembrolizumab trials in NSCLC?"
    p = llm_plan(
        intent=Intent.SURVIVAL_SUMMARY,
        answerability="subjective_judgment",
        cohorts=[cohort(label="Pembrolizumab", drugs=["pembrolizumab"], conditions=["non-small cell lung cancer"])],
    )
    plan, _, tr = await run(VisualizeRequest(query=q), StubLLM({"planner": Recorder(p)}), settings)
    assert plan.parser == "llm" and plan.answerability == "answerable"
    repairs = next(s for s in tr.steps if s.name == "plan").detail["repairs"]
    assert any(r.startswith("planner flagged this as a judgment question") for r in repairs)


async def test_regex_subjective_question_still_refused(settings):
    q = "Which drug is the safest between semaglutide and canagliflozin for diabetes?"
    p = llm_plan(  # even when the LLM says answerable, the deterministic check refuses
        intent=Intent.RISKS_ADVERSE_EVENTS,
        answerability="answerable",
        scope=Scope.COMPARISON,
        cohorts=[
            cohort(label="Semaglutide", drugs=["semaglutide"]),
            cohort(label="Canagliflozin", drugs=["canagliflozin"]),
        ],
    )
    plan, _, _ = await run(VisualizeRequest(query=q), StubLLM({"planner": Recorder(p)}), settings)
    assert plan.answerability == "subjective_judgment"
