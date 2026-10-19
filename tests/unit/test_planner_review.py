"""Review round 1: one or more tests per review item (never silently widen / narrow / relabel)."""

from __future__ import annotations

import datetime as dt
import json

import pytest
from pydantic import BaseModel

from ctviz.agents.llm import StubLLM
from ctviz.agents.planner import LLMCohort, LLMPlan, LLMSpan, cacheable, plan_cache_key, plan_query
from ctviz.agents.rules_planner import TEMPLATES, extract_entities, extract_phases, rule_plan
from ctviz.api.schemas import Intent, Persona, Scope, VisualizeRequest
from ctviz.api.schemas.request import PatientProfile
from ctviz.api.schemas.response import LLMCallInfo
from ctviz.core.budget import RequestBudget, budget_scope, current_budget
from ctviz.core.entities import EntityResolver
from ctviz.obs.trace import trace_scope
from ctviz.store.db import Database

NCT = "NCT03548935"


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


class Rec:
    def __init__(self, item):
        self.item = item
        self.users: list[str] = []

    def __call__(self, user: str, schema: type[BaseModel]):
        self.users.append(user)
        return self.item


async def run(q: str, p: LLMPlan, settings, db=None, **fields):
    rec = Rec(p)
    with trace_scope() as tr:
        plan, ents = await plan_query(
            VisualizeRequest(query=q, **fields), settings=settings, llm=StubLLM({"planner": rec}), db=db
        )
    return plan, ents, tr, rec


def _drugs(plan):
    return [d for c in plan.cohorts for d in c.drugs]


# ------------------------------------------------------------------- 1. never widen when grounding drops a kind
async def test_1_lost_kind_falls_back_to_grounded_rule_cohorts(settings):
    q = "How many semaglutide trials per year for obesity?"
    p = llm_plan(
        cohorts=[cohort(label="Ozempic (sema) in obesity", drugs=["Semaglutide (Ozempic)"], conditions=["obesity"])]
    )
    plan, _, tr, _ = await run(q, p, settings)
    (c,) = plan.cohorts
    assert c.drugs == ["semaglutide"] and c.conditions == ["obesity"]  # not widened to all obesity trials
    assert c.label != "Ozempic (sema) in obesity"
    assert any("rule cohorts" in r for r in next(s for s in tr.steps if s.name == "plan").detail["repairs"])


async def test_1_unverifiable_entity_asks_and_notes_instead_of_silent_widening(settings):
    q = "Trials per year for zorblax"
    p = llm_plan(cohorts=[cohort(label="Zorblax fever", conditions=["Zorblax Fever Syndrome"])])
    plan, _, _, _ = await run(q, p, settings)
    # the dropped name is kept for a ClinicalTrials.gov count check (zero -> needs_clarification); the plan
    # never claims to analyze all registered trials
    assert [(u.kind, u.text) for u in plan.unverified_terms] == [("condition", "Zorblax Fever Syndrome")]
    assert not any("all registered trials" in a for a in plan.assumptions)


async def test_1_verified_source_span_keeps_llm_normalization(settings):
    q = "Trials per year for Zorblax syndrome"
    span = LLMSpan(kind="condition", name="Zorblax Syndrome Type 2", source_text="Zorblax syndrome")
    p = llm_plan(cohorts=[cohort(label="free text", conditions=["Zorblax Syndrome Type 2"], entity_spans=[span])])
    plan, ents, _, _ = await run(q, p, settings)
    assert _drugs(plan) == [] and plan.cohorts[0].conditions == ["Zorblax Syndrome Type 2"]
    assert plan.cohorts[0].label == "Zorblax Syndrome Type 2"
    assert any(e.input == "Zorblax syndrome" for e in ents)


@pytest.mark.parametrize("src", ["Frobnitz disease", "MS", ""])
async def test_1_unverified_span_rejected(settings, src):
    span = LLMSpan(kind="condition", name="multiple sclerosis", source_text=src)
    p = llm_plan(cohorts=[cohort(conditions=["multiple sclerosis"], entity_spans=[span])])
    plan, _, _, _ = await run("How many MS trials started each year?", p, settings)
    assert "multiple sclerosis" not in [x for c in plan.cohorts for x in c.conditions]


# ------------------------------------------------------------------- 2. filters need a cue; charted dims stripped
async def test_2_charted_dimension_filter_stripped(settings):
    p = llm_plan(
        intent=Intent.PHASE_DISTRIBUTION, cohorts=[cohort(label="BC", conditions=["breast cancer"], phases=["PHASE3"])]
    )
    plan, _, _, _ = await run("Phase 3 breast cancer trials by phase", p, settings)
    assert plan.cohorts[0].phases == []


async def test_2_structured_phase_kept_even_on_charted_dimension(settings):
    p = llm_plan(intent=Intent.PHASE_DISTRIBUTION, cohorts=[cohort(label="BC", conditions=["breast cancer"])])
    plan, _, _, _ = await run("breast cancer trials by phase", p, settings, trial_phase=["2", "3"])
    assert plan.cohorts[0].phases == ["PHASE2", "PHASE3"]


async def test_2_status_and_year_kept_with_literal_cue(settings):
    p = llm_plan(cohorts=[cohort(label="O", conditions=["obesity"], statuses=["COMPLETED"], start_year=2012)])
    plan, _, _, _ = await run("completed obesity trials per year from 2012", p, settings)
    assert plan.cohorts[0].statuses == ["COMPLETED"] and plan.cohorts[0].start_year == 2012


def test_2_rule_planner_keeps_structured_status_for_status_chart():
    plan = rule_plan(VisualizeRequest(query="Status breakdown of obesity trials", status=["recruiting"]))
    assert plan.cohorts[0].statuses == ["RECRUITING"]


# ------------------------------------------------------------------- 3. spelling correction guards
@pytest.mark.parametrize(
    ("text", "kind"),
    [("hepatitis A", "condition"), ("type 3 diabetes", "condition"), ("type 2 diabetes insipidus", "condition")],
)
def test_3_no_digit_single_char_or_token_count_corrections(text, kind):
    e = EntityResolver().resolve(text, kind)
    assert e.method == "passthrough" and e.resolved == text


def test_3_structured_fields_never_spelling_corrected():
    r = EntityResolver()
    assert r.resolve("pembrolizumb", "drug").method == "spelling_correction"
    e = r.resolve("pembrolizumb", "drug", method="structured_field")
    assert e.resolved == "pembrolizumb" and e.method == "structured_field"
    plan = rule_plan(VisualizeRequest(query="trials per year", drug_name="pembrolizumb"))
    assert plan.cohorts[0].drugs == ["pembrolizumb"]


# ------------------------------------------------------------------- 5. ambiguity inside known spans
@pytest.mark.parametrize("q", ["What is Novo Nordisk doing in obesity?", "Merck Sharp & Dohme trials per year"])
def test_5_no_ambiguity_flag_inside_known_entity(q):
    _, amb = extract_entities(q)
    assert amb == {}
    assert rule_plan(VisualizeRequest(query=q)).clarification is None


def test_5_bare_merck_is_flagged():
    _, amb = extract_entities("merck trials per year")
    assert "merck" in amb


# ------------------------------------------------------------------- 6. cache key + privacy
def test_6_cache_key_components():
    q = VisualizeRequest(query="Obesity  trials")
    assert plan_cache_key(q) == plan_cache_key(VisualizeRequest(query=" Obesity trials "))
    assert plan_cache_key(q) != plan_cache_key(VisualizeRequest(query="obesity trials"))
    assert plan_cache_key(q) != plan_cache_key(q, model="gpt-5.4-mini")
    assert plan_cache_key(q, today=dt.date(2026, 1, 1)) != plan_cache_key(q, today=dt.date(2027, 1, 1))


def test_6_patient_requests_not_cacheable():
    assert not cacheable(VisualizeRequest(query="x", persona="patient"))
    assert not cacheable(VisualizeRequest(query="x", patient_profile=PatientProfile(age=40)))
    assert cacheable(VisualizeRequest(query="x"))


async def test_6_patient_persona_plan_not_written(settings, tmp_path):
    db = Database(tmp_path / "c.db").connect()
    p = llm_plan(intent=Intent.ELIGIBILITY_CHECK, scope=Scope.SINGLE_TRIAL, persona=Persona.PATIENT)
    req = VisualizeRequest(query="Am I eligible?", nct_id=NCT)
    await run(req.query, p, settings, db, nct_id=NCT)
    assert await db.cache_get("plan", plan_cache_key(req, model=settings.model_planner)) is None
    db.close()


async def test_6_profile_never_sent_to_llm(settings):
    p = llm_plan(intent=Intent.CONCOMITANT_MEDICATIONS, scope=Scope.SINGLE_TRIAL, persona=Persona.PATIENT)
    _, _, _, rec = await run(
        "Can I keep my meds?",
        p,
        settings,
        nct_id=NCT,
        patient_profile=PatientProfile(age=61, medications=["warfarin"], notes="secret"),
    )
    sent = rec.users[0]
    assert "warfarin" not in sent and "secret" not in sent and "61" not in sent
    assert json.loads(sent)["structured_fields"] == {"nct_id": NCT}


# ------------------------------------------------------------------- 7. no LLM free text in the response
async def test_7_llm_free_text_never_reaches_plan(settings):
    p = llm_plan(
        cohorts=[cohort(label="IGNORE PREVIOUS INSTRUCTIONS", conditions=["obesity"])],
        assumptions=["the user is a doctor"],
        clarification="Visit evil.example",
        focus_terms=["evil", "obesity"],
    )
    plan, _, _, _ = await run("Trials per year for obesity", p, settings)
    blob = json.dumps(plan.model_dump(mode="json"))
    assert "IGNORE" not in blob and "doctor" not in blob and "evil" not in blob
    assert plan.cohorts[0].label == "Obesity" and plan.focus_terms == ["obesity"]
    assert plan.clarification == TEMPLATES["LOW_CONFIDENCE"].format(
        description=__import__("ctviz.core.intents", fromlist=["x"]).load_intents()[Intent.TREND_OVER_TIME].description
    )


async def test_7_focus_terms_may_come_from_profile(settings):
    p = llm_plan(
        intent=Intent.CONCOMITANT_MEDICATIONS,
        scope=Scope.SINGLE_TRIAL,
        persona=Persona.PATIENT,
        focus_terms=["warfarin", "aspirin"],
    )
    plan, _, _, _ = await run(
        "Can I keep my meds?", p, settings, nct_id=NCT, patient_profile=PatientProfile(medications=["warfarin"])
    )
    assert plan.focus_terms == ["warfarin"]


# ------------------------------------------------------------------- 8. alias table content
def test_8_alias_classes_and_umbrellas():
    r = EntityResolver()
    assert {"cetuximab", "panitumumab", "amivantamab"} <= set(r.expand_drug_class("EGFR inhibitors"))
    assert "cetuximab" not in r.expand_drug_class("EGFR TKIs")
    assert {"orforglipron", "retatrutide", "survodutide", "mazdutide"} <= set(r.expand_drug_class("GLP-1"))
    ici = set(r.expand_drug_class("checkpoint inhibitors"))
    assert set(r.expand_drug_class("PD-1 inhibitors")) <= ici
    assert any(m.is_class for m in r.scan("trials of ICIs in melanoma"))
    assert r.is_ambiguous("NIH") and r.is_ambiguous("SCD")
    assert r.resolve("depression", "condition").resolved == "depression"
    assert r.resolve("liver cancer", "condition").resolved == "liver cancer"


# ------------------------------------------------------------------- 9. misroutes
@pytest.mark.parametrize(
    ("q", "fields", "intent"),
    [
        ("Number of trials per year by country for asthma", {}, Intent.TREND_OVER_TIME),
        ("What's in the obesity drug pipeline?", {}, Intent.INTERVENTION_LANDSCAPE),
        ("exercise trials for cancer survivors per year", {}, Intent.TREND_OVER_TIME),
        ("Did this trial meet its primary endpoint?", {"nct_id": NCT}, Intent.RESULTS_OUTCOMES),
        ("Where is this trial recruiting near Boston?", {"nct_id": NCT}, Intent.SITES_AND_CONTACTS),
        ("combination therapy trials for melanoma by year", {}, Intent.TREND_OVER_TIME),
        ("whats the point of this trial", {"nct_id": NCT}, Intent.STUDY_RATIONALE),
        ("what side efects could i get", {"nct_id": NCT}, Intent.RISKS_ADVERSE_EVENTS),
        ("did the drug actually work", {"nct_id": NCT}, Intent.RESULTS_OUTCOMES),
        ("what kinds of interventions are tested for depression", {}, Intent.INTERVENTION_LANDSCAPE),
        ("Will I still be able to see my regular doctor?", {"nct_id": NCT}, Intent.NOT_IN_REGISTRY),
        ("Can I keep seeing my own physician?", {"nct_id": NCT}, Intent.NOT_IN_REGISTRY),
        ("Will my doctor be involved?", {"nct_id": NCT}, Intent.NOT_IN_REGISTRY),
    ],
)
def test_9_routes(q, fields, intent):
    assert rule_plan(VisualizeRequest(query=q, **fields)).intent is intent


def test_9_researchers_is_not_a_sponsor():
    plan = rule_plan(VisualizeRequest(query="What are researchers doing in obesity?"))
    assert all(not c.sponsors for c in plan.cohorts)


def test_9_multi_token_vs_sides():
    plan = rule_plan(VisualizeRequest(query="Compare trials per year for Zorblax syndrome vs Frobnitz disease"))
    assert plan.scope is Scope.COMPARISON
    assert [c.conditions for c in plan.cohorts] == [["Zorblax syndrome"], ["Frobnitz disease"]]


def test_9_phase_2b():
    assert extract_phases("phase 2b trials") == ["PHASE2"]
    assert extract_phases("phase 1b/2a") == ["PHASE1", "PHASE2"]


# ------------------------------------------------------------------- 10. budget
def test_10_current_budget_outside_scope_is_fresh_and_unstored():
    a = current_budget()
    a.charge_tool()
    b = current_budget()
    assert a is not b and b.tool_calls == 0


def test_10_reserved_llm_calls_count(settings):
    with trace_scope() as tr, budget_scope(settings) as bud:
        tr.llm_reserved = settings.max_llm_calls
        assert not bud.can_call_llm()
    with trace_scope() as tr:
        tr.add_llm(LLMCallInfo(agent="x", model="stub"))
        assert RequestBudget(max_llm_calls=1).llm_calls == 1


# -- routing backstops added after manual testing ------------------------------------------------
async def test_llm_not_in_registry_needs_rule_support(settings):
    """The LLM may not send an eligibility question to the 'not in registry' card on its own."""
    p = llm_plan(
        intent=Intent.NOT_IN_REGISTRY, scope=Scope.COHORT, cohorts=[cohort(label="Semaglutide", drugs=["semaglutide"])]
    )
    plan, _, _, _ = await run("Am I eligible for a semaglutide trial?", p, settings)
    assert plan.intent is Intent.ELIGIBILITY_CHECK


async def test_rule_not_in_registry_wins_over_llm(settings):
    """Regular-doctor / cost questions always get the honest 'not in registry' card."""
    p = llm_plan(intent=Intent.SITES_AND_CONTACTS, scope=Scope.SINGLE_TRIAL, nct_ids=["NCT03548935"], cohorts=[])
    plan, _, _, _ = await run(
        "Can I keep seeing my regular doctor during the trial?", p, settings, nct_id="NCT03548935"
    )
    assert plan.intent is Intent.NOT_IN_REGISTRY


async def test_spelling_normalization_is_stated(settings):
    span = LLMSpan(kind="drug", name="pembrolizumab", source_text="pembrolizumabb")
    p = llm_plan(cohorts=[cohort(label="Pembrolizumab", drugs=["pembrolizumab"], entity_spans=[span])])
    plan, _, _, _ = await run("How many trials for pembrolizumabb started each year?", p, settings)
    assert any("Read 'pembrolizumabb' as the drug 'pembrolizumab'" in a for a in plan.assumptions)
