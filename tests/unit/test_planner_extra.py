"""Extra planner hardening tests: paraphrase routing, LLM injection, structured overrides, plan cache,
retrieval-agent guards and alias-table sanity. Everything is mocked (StubLLM, fake tools, tmp SQLite)."""

from __future__ import annotations

import asyncio
import json
import re

import pytest
from pydantic import BaseModel

from ctviz.agents import retrieval_agent as ra
from ctviz.agents.llm import StubLLM
from ctviz.agents.planner import LLMCohort, LLMPlan, plan_cache_key, plan_query
from ctviz.agents.retrieval_agent import (
    AgentStep,
    RetrievalTools,
    StepCohort,
    resolve_retrieval,
)
from ctviz.agents.rules_planner import rule_plan
from ctviz.api.schemas import Intent, Persona, QueryPlan, Scope, VisualizeRequest
from ctviz.api.schemas.plan import CohortFilter
from ctviz.api.schemas.request import NCT_PATTERN, PHASES, STATUSES, PatientProfile
from ctviz.core.budget import budget_scope
from ctviz.core.entities import ALIASES_FILE, EntityResolver
from ctviz.core.entities import norm as _norm
from ctviz.obs.trace import trace_scope
from ctviz.store.db import Database

NCT = "NCT03548935"
NCT2 = "NCT02142738"
IN, S = Intent, Scope


# =========================================================================== (1) paraphrase routing
def _xf(reason: str):
    return pytest.mark.xfail(reason=f"rule planner misroute: {reason}", strict=True)


# (question, structured fields, expected intent, expected scope[, marks])
PARAPHRASES = [
    # ---- patient-style, single trial
    ("whats the point of this trial", {"nct_id": NCT}, IN.STUDY_RATIONALE, S.SINGLE_TRIAL),
    ("can i join this study? im 52 with type 2 diabetes", {"nct_id": NCT}, IN.ELIGIBILITY_CHECK, S.SINGLE_TRIAL),
    (
        "am i allowed to keep taking metformin while in the study",
        {"nct_id": NCT},
        IN.CONCOMITANT_MEDICATIONS,
        S.SINGLE_TRIAL,
    ),
    ("how long would i be in the trial for", {"nct_id": NCT}, IN.TRIAL_JOURNEY_TIMELINE, S.SINGLE_TRIAL),
    ("how many times do i have to come to the clinic", {"nct_id": NCT}, IN.VISIT_BURDEN, S.SINGLE_TRIAL),
    ("what side efects could i get", {"nct_id": NCT}, IN.RISKS_ADVERSE_EVENTS, S.SINGLE_TRIAL),
    ("what if i get sicker during the trial", {"nct_id": NCT}, IN.SAFETY_PROVISIONS, S.SINGLE_TRIAL),
    ("will they pay for my travel", {"nct_id": NCT}, IN.NOT_IN_REGISTRY, S.SINGLE_TRIAL),
    ("did the drug actually work", {"nct_id": NCT}, IN.RESULTS_OUTCOMES, S.SINGLE_TRIAL),
    ("where is the nearest site and who do i contact", {"nct_id": NCT}, IN.SITES_AND_CONTACTS, S.SINGLE_TRIAL),
    ("Is there a site near me I can go to?", {"nct_id": NCT}, IN.SITES_AND_CONTACTS, S.SINGLE_TRIAL),
    # ---- researcher-style, single trial
    ("Was this study randomized and double-blind?", {"nct_id": NCT}, IN.DESIGN_METHODOLOGY, S.SINGLE_TRIAL),
    ("what are the primary and secondary endpoints", {"nct_id": NCT}, IN.ENDPOINTS_CATALOG, S.SINGLE_TRIAL),
    (
        "Was enrollment enough to detect the treatment effect (power)?",
        {"nct_id": NCT},
        IN.ENROLLMENT_VS_POWER,
        S.SINGLE_TRIAL,
    ),
    ("what PK parameters like Cmax and AUC were measured", {"nct_id": NCT}, IN.BIOMARKERS_PKPD, S.SINGLE_TRIAL),
    ("How many participants discontinued and why?", {"nct_id": NCT}, IN.DROPOUT_FLOW, S.SINGLE_TRIAL),
    ("age and sex breakdown of participants", {"nct_id": NCT}, IN.DEMOGRAPHICS, S.SINGLE_TRIAL),
    ("What limitations or caveats does this study have?", {"nct_id": NCT}, IN.STUDY_LIMITATIONS, S.SINGLE_TRIAL),
    ("give me an overview of NCT03548935", {}, IN.SINGLE_TRIAL_OVERVIEW, S.SINGLE_TRIAL),
    ("show the kaplan-meier overall survival for NCT03548935", {}, IN.SURVIVAL_SUMMARY, S.SINGLE_TRIAL),
    ("side by side endpoints of NCT03548935 and NCT02142738", {}, IN.ENDPOINTS_CATALOG, S.COMPARISON),
    # ---- cohort / appendix-style (brand names, typos, lowercase, extra words)
    ("keytruda trials per year since 2016", {}, IN.TREND_OVER_TIME, S.COHORT),
    ("how many ozempic studies started every year", {}, IN.TREND_OVER_TIME, S.COHORT),
    ("Number of Alzheimers trials over time", {}, IN.TREND_OVER_TIME, S.COHORT),
    ("phase breakdown of NSCLC trials", {}, IN.PHASE_DISTRIBUTION, S.COHORT),
    ("what phases are the lung cancer trials in", {}, IN.PHASE_DISTRIBUTION, S.COHORT),
    ("How many obesity trials are currently recruiting vs completed?", {}, IN.STATUS_DISTRIBUTION, S.COHORT),
    ("status breakdown for crohn's disease trials", {}, IN.STATUS_DISTRIBUTION, S.COHORT),
    ("heatmap of phase by status for mounjaro trials", {}, IN.PHASE_STATUS_HEATMAP, S.COHORT),
    ("Where are the gaps in Parkinson's research?", {}, IN.LANDSCAPE_GAP_HEATMAP, S.COHORT),
    ("who sponsors the most melanoma trials", {}, IN.SPONSOR_LANDSCAPE, S.COHORT),
    ("industry vs academic sponsors for asthma studies", {}, IN.SPONSOR_LANDSCAPE, S.COHORT),
    ("what is pfizer doing in oncology", {}, IN.SPONSOR_ACTIVITY, S.COHORT),
    ("which countries run the most semaglutide trials", {}, IN.GEOGRAPHIC_DISTRIBUTION, S.COHORT),
    ("map of where hiv trials are located", {}, IN.GEOGRAPHIC_DISTRIBUTION, S.COHORT),
    ("network of drugs and conditions for nivolumab", {}, IN.ENTITY_NETWORK, S.COHORT),
    ("distribution of enrollment sizes for type 1 diabetes trials", {}, IN.ENROLLMENT_DISTRIBUTION, S.COHORT),
    ("what kinds of interventions are tested for depression", {}, IN.INTERVENTION_LANDSCAPE, S.COHORT),
    ("how did tirzepatide progress from phase 1 to phase 3", {}, IN.DEVELOPMENT_PROGRESSION, S.COHORT),
    ("compare trial counts per year for wegovy versus zepbound", {}, IN.TREND_OVER_TIME, S.COMPARISON),
]
# Observed misroutes of the current rule planner (question -> what it returned). Kept as strict xfails.
# Fixed in review round 1: "point of this trial", "side efects", "did the drug actually work", "kinds of interventions".
MISROUTES: dict[str, str] = {}


def _params():
    out = []
    for q, fields, intent, scope in PARAPHRASES:
        marks = [_xf(f"got {MISROUTES[q]}, expected {intent.value}/{scope.value}")] if q in MISROUTES else []
        out.append(pytest.param(q, fields, intent, scope, marks=marks, id=q[:50]))
    return out


def test_paraphrase_set_is_new_and_sized():
    from tests.unit import test_planner_rules as base

    known = {q for q, *_ in base.PATIENT + base.RESEARCHER_TRIAL + base.COHORT}
    assert len(PARAPHRASES) == 40
    assert not known & {q for q, *_ in PARAPHRASES}


@pytest.mark.parametrize(("question", "fields", "intent", "scope"), _params())
def test_paraphrase_routes(question, fields, intent, scope):
    plan = rule_plan(VisualizeRequest(query=question, **fields))
    assert (plan.intent, plan.scope) == (intent, scope)


def test_paraphrase_misroute_rate_is_bounded():
    """Report-only guard: the documented misroute rate (xfails above) stays at or below 25%."""
    bad = 0
    for q, fields, intent, scope in PARAPHRASES:
        p = rule_plan(VisualizeRequest(query=q, **fields))
        bad += (p.intent, p.scope) != (intent, scope)
    assert bad == len(MISROUTES)
    assert bad / len(PARAPHRASES) <= 0.25


# =========================================================================== LLM helpers
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


class Raw:
    """A responder payload that bypasses LLMPlan construction (simulates off-schema model output)."""

    def __init__(self, data: dict):
        self.data = data

    def model_dump(self):
        return self.data


class Queue:
    def __init__(self, *items):
        self.items = list(items)
        self.calls = 0

    def __call__(self, user: str, schema: type[BaseModel]):
        self.calls += 1
        item = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if isinstance(item, Exception):
            raise item
        return item


async def run(req: VisualizeRequest, llm, settings, db=None):
    with trace_scope() as tr:
        plan, ents = await plan_query(req, settings=settings, llm=llm, db=db)
    return plan, ents, tr


def _all(plan: QueryPlan, attr: str) -> list:
    return [x for c in plan.cohorts for x in getattr(c, attr)]


# =========================================================================== (2) LLM injection
OBESITY = "Trials per year for obesity"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("drugs", ["nivolumab"]),
        ("drugs", ["PD-1"]),  # an injected drug class must not expand into members
        ("conditions", ["ovarian cancer"]),
        ("countries", ["Germany"]),
        ("sponsors", ["Pfizer"]),
    ],
)
async def test_injected_entity_is_dropped(settings, field, value):
    kw = {"conditions": ["obesity"], **{field: value}}
    if field == "conditions":
        kw["conditions"] = ["obesity", *value]
    p = llm_plan(cohorts=[cohort(label="Obesity", **kw)])
    plan, ents, tr = await run(VisualizeRequest(query=OBESITY), StubLLM({"planner": Queue(p)}), settings)
    assert plan.parser == "llm"
    injected = {v.lower() for v in value} | {"nivolumab", "pembrolizumab"}
    present = {x.lower() for x in _all(plan, field)}
    assert not present & injected, present
    if field != "conditions":
        assert _all(plan, "conditions") == ["obesity"]
    assert not any(e.resolved.lower() in injected for e in ents)
    repairs = next(s for s in tr.steps if s.name == "plan").detail["repairs"]
    assert any("dropped ungrounded" in r for r in repairs)


@pytest.mark.parametrize(("field", "value"), [("phases", ["PHASE3"]), ("statuses", ["RECRUITING"])])
async def test_injected_phase_or_status_filter_is_dropped(settings, field, value):
    p = llm_plan(cohorts=[cohort(label="Obesity", conditions=["obesity"], **{field: value})])
    plan, _, _ = await run(VisualizeRequest(query=OBESITY), StubLLM({"planner": Queue(p)}), settings)
    assert _all(plan, field) == []


async def test_injected_year_filter_is_dropped(settings):
    p = llm_plan(cohorts=[cohort(label="Obesity", conditions=["obesity"], start_year=2015, end_year=2018)])
    plan, _, _ = await run(VisualizeRequest(query=OBESITY), StubLLM({"planner": Queue(p)}), settings)
    assert all(c.start_year is None and c.end_year is None for c in plan.cohorts)


async def test_grounded_phase_filter_is_kept(settings):
    p = llm_plan(cohorts=[cohort(label="Obesity", conditions=["obesity"], phases=["PHASE3"])])
    plan, _, _ = await run(
        VisualizeRequest(query="Phase 3 trials per year for obesity"), StubLLM({"planner": Queue(p)}), settings
    )
    assert _all(plan, "phases") == ["PHASE3"]


async def test_intent_outside_enum_falls_back_to_rules(settings):
    bad = llm_plan().model_dump(mode="json") | {"intent": "delete_all_trials"}
    q = Queue(Raw(bad))
    plan, _, tr = await run(
        VisualizeRequest(query="How are breast cancer trials distributed across phases?"),
        StubLLM({"planner": q}),
        settings,
    )
    assert plan.parser == "rules" and plan.intent is IN.PHASE_DISTRIBUTION
    assert q.calls == 2  # one escalation on schema failure, then rules
    assert next(s for s in tr.steps if s.name == "plan").detail["escalation_reason"] == "schema_failure"


@pytest.mark.parametrize("conf", [2.0, -1.0, 1e9])
async def test_confidence_out_of_range_is_clamped(settings, conf):
    p = llm_plan(intent_confidence=conf, cohorts=[cohort(label="Obesity", conditions=["obesity"])])
    plan, _, _ = await run(VisualizeRequest(query=OBESITY), StubLLM({"planner": Queue(p)}), settings)
    assert plan.parser == "llm" and 0.0 <= plan.intent_confidence <= 1.0
    QueryPlan.model_validate(plan.model_dump())


async def test_seven_cohorts_are_capped(settings):
    drugs = ["semaglutide", "tirzepatide", "liraglutide", "dulaglutide", "exenatide", "lixisenatide", "retatrutide"]
    q = "Compare trials per year for " + ", ".join(drugs)
    p = llm_plan(scope=Scope.COMPARISON, cohorts=[cohort(label=d, drugs=[d]) for d in drugs])
    plan, _, _ = await run(VisualizeRequest(query=q), StubLLM({"planner": Queue(p)}), settings)
    assert plan.parser == "llm" and 2 <= len(plan.cohorts) <= 5 and plan.scope is S.COMPARISON
    assert set(_all(plan, "drugs")) <= set(drugs)


async def test_llm_nct_ids_not_in_request_are_dropped(settings):
    p = llm_plan(
        intent=IN.SINGLE_TRIAL_OVERVIEW,
        scope=Scope.SINGLE_TRIAL,
        nct_ids=[NCT, "NCT99999999", "garbage", "nct12345678"],
    )
    plan, _, _ = await run(VisualizeRequest(query=f"Tell me about {NCT}"), StubLLM({"planner": Queue(p)}), settings)
    assert plan.nct_ids == [NCT] and plan.scope is S.SINGLE_TRIAL


async def test_llm_nct_ids_without_any_in_request(settings):
    p = llm_plan(intent=IN.SINGLE_TRIAL_OVERVIEW, scope=Scope.SINGLE_TRIAL, nct_ids=["NCT99999999"])
    plan, _, _ = await run(
        VisualizeRequest(query="tell me about the keytruda trial"), StubLLM({"planner": Queue(p)}), settings
    )
    assert plan.nct_ids == []
    assert plan.scope is not S.SINGLE_TRIAL or plan.clarification


# =========================================================================== (3) structured fields win
# (request fields, conflicting LLM plan kwargs, cohort kwargs, check(plan))
_C = dict(label="X", conditions=["obesity"])
OVERRIDES = [
    (
        "drug_name",
        {"drug_name": "Keytruda"},
        {},
        {"drugs": ["nivolumab"]},
        lambda p: _all(p, "drugs") == ["pembrolizumab"],
    ),
    ("condition", {"condition": "asthma"}, {}, {}, lambda p: _all(p, "conditions") == ["asthma"]),
    ("sponsor", {"sponsor": "Lilly"}, {}, {"sponsors": ["Pfizer"]}, lambda p: _all(p, "sponsors") == ["Eli Lilly"]),
    (
        "country",
        {"country": "USA"},
        {},
        {"countries": ["Germany"]},
        lambda p: _all(p, "countries") == ["United States"],
    ),
    ("trial_phase", {"trial_phase": ["II"]}, {}, {"phases": ["PHASE3"]}, lambda p: _all(p, "phases") == ["PHASE2"]),
    (
        "status",
        {"status": ["completed"]},
        {},
        {"statuses": ["RECRUITING"]},
        lambda p: _all(p, "statuses") == ["COMPLETED"],
    ),
    ("start_year", {"start_year": 2012}, {}, {"start_year": 2020}, lambda p: p.cohorts[0].start_year == 2012),
    ("end_year", {"end_year": 2019}, {}, {"end_year": 2024}, lambda p: p.cohorts[0].end_year == 2019),
    ("persona", {"persona": "researcher"}, {"persona": Persona.PATIENT}, {}, lambda p: p.persona is Persona.RESEARCHER),
    ("group_by", {"group_by": "country"}, {"group_by": "phase"}, {}, lambda p: p.group_by == "country"),
    ("series_by", {"series_by": "status"}, {"series_by": "phase"}, {}, lambda p: p.series_by == "status"),
    (
        "time_granularity",
        {"time_granularity": "month"},
        {"time_granularity": "year"},
        {},
        lambda p: p.time_granularity == "month",
    ),
    ("top_n", {"top_n": 7}, {"top_n": 50}, {}, lambda p: p.top_n == 7),
    (
        "preferred_visualization",
        {"preferred_visualization": "bar_chart"},
        {"preferred_visualization": "time_series"},
        {},
        lambda p: p.preferred_visualization == "bar_chart",
    ),
    (
        "nct_id",
        {"nct_id": NCT},
        {"intent": IN.SINGLE_TRIAL_OVERVIEW, "scope": Scope.SINGLE_TRIAL, "nct_ids": [NCT2]},
        {},
        lambda p: p.nct_ids == [NCT] and p.cohorts == [],
    ),
    (
        "nct_ids",
        {"nct_ids": [NCT, NCT2]},
        {"intent": IN.ENROLLMENT_DISTRIBUTION, "scope": Scope.SINGLE_TRIAL, "nct_ids": [NCT]},
        {},
        lambda p: p.nct_ids == [NCT, NCT2] and p.scope is S.COMPARISON,
    ),
    (
        "drug_names",
        {"drug_names": ["Ozempic", "Mounjaro"]},
        {},
        {"drugs": ["insulin"]},
        lambda p: [c.drugs for c in p.cohorts] == [["semaglutide"], ["tirzepatide"]],
    ),
    (
        "conditions",
        {"conditions": ["asthma", "COPD"]},
        {},
        {},
        lambda p: [c.conditions for c in p.cohorts] == [["asthma"], ["chronic obstructive pulmonary disease"]],
    ),
]


@pytest.mark.parametrize(("name", "fields", "plan_kw", "cohort_kw", "check"), OVERRIDES, ids=[o[0] for o in OVERRIDES])
async def test_structured_field_overrides_llm(settings, name, fields, plan_kw, cohort_kw, check):
    p = llm_plan(**plan_kw, cohorts=[cohort(**{**_C, **cohort_kw})])
    plan, _, _ = await run(VisualizeRequest(query=OBESITY, **fields), StubLLM({"planner": Queue(p)}), settings)
    assert plan.parser == "llm"
    assert check(plan), plan.model_dump(exclude_defaults=True)


# =========================================================================== (4) plan cache
def test_cache_key_differs_by_drug_name_and_profile():
    keys = {
        plan_cache_key(VisualizeRequest(query=OBESITY, **kw))
        for kw in (
            {},
            {"drug_name": "Keytruda"},
            {"drug_name": "Ozempic"},
            {"patient_profile": PatientProfile(age=40, medications=["warfarin"])},
            {"patient_profile": PatientProfile(age=40, medications=["insulin"])},
        )
    }
    assert len(keys) == 5


async def test_cache_separates_drug_names(settings, tmp_path):
    db = Database(tmp_path / "c.db").connect()
    q = "How has the number of trials for this drug changed over time?"
    p = llm_plan(cohorts=[cohort(label="Pembro", drugs=["pembrolizumab"])])
    a, _, _ = await run(VisualizeRequest(query=q, drug_name="Keytruda"), StubLLM({"planner": Queue(p)}), settings, db)
    b, _, _ = await run(VisualizeRequest(query=q, drug_name="Ozempic"), StubLLM(), settings, db)
    c, _, _ = await run(VisualizeRequest(query=q, drug_name="Keytruda"), StubLLM(), settings, db)
    db.close()
    assert a.parser == "llm" and b.parser == "rules" and c.parser == "cache"
    assert _all(b, "drugs") == ["semaglutide"] and _all(c, "drugs") == ["pembrolizumab"]


async def test_cache_no_patient_profile_leakage(settings, tmp_path):
    db = Database(tmp_path / "c.db").connect()
    q = "Can I keep taking my medications?"
    p = llm_plan(
        intent=IN.CONCOMITANT_MEDICATIONS, scope=Scope.SINGLE_TRIAL, persona=Persona.PATIENT, focus_terms=["warfarin"]
    )
    prof_a = PatientProfile(age=60, medications=["warfarin"])
    prof_b = PatientProfile(age=30, medications=["insulin"])
    a, _, _ = await run(
        VisualizeRequest(query=q, nct_id=NCT, patient_profile=prof_a), StubLLM({"planner": Queue(p)}), settings, db
    )
    b, _, _ = await run(VisualizeRequest(query=q, nct_id=NCT, patient_profile=prof_b), StubLLM(), settings, db)
    db.close()
    assert a.parser == "llm" and b.parser == "rules"
    assert "warfarin" not in json.dumps(b.model_dump(mode="json"))
    assert "insulin" in b.focus_terms


async def test_patient_profile_not_persisted_in_plan_cache(settings, tmp_path):
    db = Database(tmp_path / "c.db").connect()
    req = VisualizeRequest(
        query="Can I keep taking my medications?",
        nct_id=NCT,
        patient_profile=PatientProfile(age=60, medications=["warfarin"]),
    )
    p = llm_plan(
        intent=IN.CONCOMITANT_MEDICATIONS, scope=Scope.SINGLE_TRIAL, persona=Persona.PATIENT, focus_terms=["warfarin"]
    )
    await run(req, StubLLM({"planner": Queue(p)}), settings, db)
    hit = await db.cache_get("plan", plan_cache_key(req))
    db.close()
    assert hit is None or "warfarin" not in json.dumps(hit)


# =========================================================================== (5) retrieval agent
RPLAN = QueryPlan(intent=Intent.TREND_OVER_TIME, intent_confidence=0.5)
ZERO = CohortFilter(label="Pembro", drugs=["pembrolizumb"], phases=["PHASE3"])


def sc(**kw) -> StepCohort:
    base = dict(
        drugs=[],
        conditions=[],
        sponsors=[],
        countries=[],
        phases=[],
        statuses=[],
        start_year=None,
        end_year=None,
        term=None,
    )
    return StepCohort(**{**base, **kw})


def step(action, cohort=None, nct_id=None, page_size=None) -> AgentStep:
    return AgentStep(action=action, reason="r", cohort=cohort, page_size=page_size, nct_id=nct_id)


class Steps:
    def __init__(self, *steps, repeat_last=False):
        self.steps = list(steps)
        self.repeat_last = repeat_last
        self.calls = 0

    def __call__(self, user, schema):
        self.calls += 1
        if self.repeat_last and len(self.steps) == 1:
            return self.steps[0]
        return self.steps.pop(0)


class SpyTools:
    """Tool spies: record every argument that actually reaches a tool callable."""

    def __init__(self, counts=None, hang_first=0):
        self.counts = counts if counts is not None else {"pembrolizumab": 120}
        self.executed: list[tuple] = []
        self.hang_first = hang_first

    async def probe(self, cf: CohortFilter) -> int:
        self.executed.append(("probe_count", cf))
        if self.hang_first:
            self.hang_first -= 1
            await asyncio.Event().wait()  # never completes -> wait_for timeout
        return sum(self.counts.get(d, 0) for d in cf.drugs) or (5 if cf.term and self.counts else 0)

    async def search(self, cf: CohortFilter, n: int):
        self.executed.append(("search_studies", cf, n))
        return [{"nct_id": "NCT01295827", "title": "KEYNOTE-001"}]

    async def get(self, nct: str):
        self.executed.append(("get_study", nct))
        return {"protocolSection": {"identificationModule": {"nctId": nct, "briefTitle": "A study"}}}

    def tools(self) -> RetrievalTools:
        return RetrievalTools(probe_count=self.probe, search_studies=self.search, get_study=self.get)


def _assert_valid_cohort(cf: CohortFilter):
    assert set(cf.phases) <= set(PHASES) and set(cf.statuses) <= set(STATUSES)
    for y in (cf.start_year, cf.end_year):
        assert y is None or 1990 <= y <= ra._max_year()
    assert cf.drugs or cf.conditions or cf.sponsors or cf.countries or cf.term
    assert all(x.strip() and len(x) <= 200 for x in cf.drugs + cf.conditions + cf.sponsors + cf.countries)
    assert len(cf.drugs) <= 10 and (cf.term is None or len(cf.term) <= 200)


INVALID_STEPS = [
    step("probe_count", sc(drugs=["pembrolizumab"], phases=["PHASE9"])),
    step("probe_count", sc(drugs=["pembrolizumab"], statuses=["OPEN"])),
    step("probe_count", sc(drugs=["pembrolizumab"], start_year=1950)),
    step("probe_count", sc(drugs=["pembrolizumab"], start_year=2020, end_year=2010)),
    step("probe_count", sc()),
    step("probe_count", sc(drugs=["   "])),
    step("probe_count", sc(drugs=[f"d{i}" for i in range(11)])),
    step("probe_count", None),
    step("search_studies", sc(drugs=["pembrolizumab"]), page_size=5000),
    step("search_studies", sc(drugs=["pembrolizumab"]), page_size=0),
    step("search_studies", sc(term="x" * 500)),
    step("get_study", nct_id="NCT123"),
    step("get_study", nct_id="NCT03548935; DROP TABLE studies"),
    step("get_study", nct_id=None),
]


@pytest.mark.parametrize("bad", INVALID_STEPS, ids=lambda s: f"{s.action}")
async def test_invalid_llm_args_never_reach_tools(settings, bad):
    spy = SpyTools(counts={})
    steps = Steps(bad, bad)
    llm = StubLLM({"retrieval": steps})
    with trace_scope(), budget_scope(settings):
        d = await resolve_retrieval(
            RPLAN, "pembro trials per year", ZERO, llm=llm, settings=settings, tools=spy.tools(), backoff_s=0.0
        )
    # retrieval makes at most ONE LLM call (CLAUDE.md budget); an invalid proposal -> rules fallback
    assert len(d.rejected) >= 1 and steps.calls == 1
    for call in spy.executed:  # only rule-fallback probes may run, and they are valid
        if call[0] == "get_study":
            assert re.match(NCT_PATTERN, call[1])
        else:
            _assert_valid_cohort(call[1])
            if call[0] == "search_studies":
                assert 1 <= call[2] <= 1000
    assert all(c[0] == "probe_count" for c in spy.executed)


async def test_interleaved_invalid_args_never_reach_tools(settings):
    good = sc(drugs=["nothing"])
    seq = [
        INVALID_STEPS[0],
        step("probe_count", good),
        INVALID_STEPS[11],
        step("probe_count", good),
        INVALID_STEPS[8],
        step("probe_count", good),
    ]
    spy = SpyTools(counts={})
    with trace_scope(), budget_scope(settings):
        d = await resolve_retrieval(
            RPLAN,
            "q",
            ZERO,
            llm=StubLLM({"retrieval": (steps := Steps(*seq))}),
            settings=settings,
            tools=spy.tools(),
            backoff_s=0.0,
        )
    # one LLM call (invalid) -> 1 rejection, nothing from the model executed; then valid rule-fallback probes
    assert steps.calls == 1 and len(d.rejected) == 1 and len(spy.executed) <= 3
    assert all(call[1].drugs in (["pembrolizumab"], []) for call in spy.executed)  # rule candidates only
    assert not any(call[1].drugs == ["nothing"] for call in spy.executed)
    for call in spy.executed:
        assert call[0] == "probe_count"
        _assert_valid_cohort(call[1])


async def test_max_three_tool_calls_when_llm_keeps_asking(settings):
    mixed = [
        step("search_studies", sc(drugs=["nothing"]), page_size=5),
        step("get_study", nct_id=NCT),
        step("probe_count", sc(drugs=["nothing"])),
    ]
    steps = Steps(*mixed, step("probe_count", sc(drugs=["more"])), repeat_last=True)
    spy = SpyTools(counts={})
    with trace_scope() as tr, budget_scope(settings):
        d = await resolve_retrieval(
            RPLAN,
            "q",
            ZERO,
            llm=StubLLM({"retrieval": steps}),
            settings=settings,
            tools=spy.tools(),
            backoff_s=0.0,
            max_steps=10,
        )
    assert len(spy.executed) <= 3 and d.tool_calls <= 3 and steps.calls <= 3
    assert sum(1 for s in tr.steps if s.name.startswith("tool:") and not s.detail.get("rejected")) <= 3


async def test_tool_timeout_backs_off_then_falls_back(settings, monkeypatch):
    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(s, *a, **k):
        delays.append(s)
        await real_sleep(0)

    monkeypatch.setattr(ra.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(ra.random, "random", lambda: 0.0)
    spy = SpyTools(hang_first=3)  # first probe and both retries hang; later probes answer
    steps = Steps(step("probe_count", sc(drugs=["pembrolizumab"])), step("finish", None))
    with trace_scope() as tr, budget_scope(settings):
        d = await resolve_retrieval(
            RPLAN,
            "pembro trials per year",
            ZERO,
            llm=StubLLM({"retrieval": steps}),
            settings=settings,
            tools=spy.tools(),
            tool_timeout_s=0.01,
            tool_retries=2,
            backoff_s=0.2,
        )
    first = d.calls[0]
    assert not first.ok and "TimeoutError" in first.error
    assert delays[:2] == pytest.approx([0.2, 0.4])  # exponential backoff between the timed-out attempts
    assert tr.steps[0].detail["retries"] == 2
    assert d.cohort is not None and d.count == 120 and "rule rewrite" in d.rationale
    assert d.tool_calls <= 3


# =========================================================================== (6) alias sanity
TABLE = json.loads(ALIASES_FILE.read_text())


@pytest.fixture(scope="module")
def r() -> EntityResolver:
    return EntityResolver()


CLASS_MEMBERS = sorted({m for c in TABLE["drug_classes"].values() for m in c["members"]})


@pytest.mark.parametrize("member", CLASS_MEMBERS)
def test_class_member_resolves_to_itself(r, member):
    e = r.resolve(member, "drug")
    assert e.resolved == member and e.method == "exact"
    assert not r.is_drug_class(member)
    assert r.known(member) == "drug"


def _pairs_hook(pairs):
    keys = [k for k, _ in pairs]
    dupes = {k for k in keys if keys.count(k) > 1}
    assert not dupes, f"duplicate raw keys in aliases.json: {dupes}"
    return dict(pairs)


def test_no_duplicate_raw_keys():
    json.loads(ALIASES_FILE.read_text(), object_pairs_hook=_pairs_hook)


@pytest.mark.parametrize("kind", ["drugs", "conditions", "sponsors", "countries"])
def test_no_alias_maps_to_two_canonicals(kind):
    seen: dict[str, str] = {}
    for alias, canon in TABLE[kind].items():
        k = _norm(alias)
        assert seen.setdefault(k, canon) == canon, f"{alias!r} -> {seen[k]!r} and {canon!r}"


@pytest.mark.parametrize(("kind", "canon_key"), [("drugs", "generics"), ("conditions", "known_conditions")])
def test_alias_is_not_another_canonical(kind, canon_key):
    canon = {_norm(c) for c in TABLE[canon_key]} | {_norm(v) for v in TABLE[kind].values()}
    clash = {a: v for a, v in TABLE[kind].items() if _norm(a) in canon and _norm(a) != _norm(v)}
    assert not clash


def test_class_aliases_unique_and_disjoint_from_drug_aliases():
    seen: dict[str, str] = {}
    for name, spec in TABLE["drug_classes"].items():
        for a in [name, *spec["aliases"]]:
            assert seen.setdefault(_norm(a), name) == name, f"class alias {a!r} in two classes"
    drug_keys = {_norm(k) for k in TABLE["drugs"]} | {_norm(g) for g in TABLE["generics"]}
    assert not set(seen) & drug_keys


AMBIG = sorted(TABLE["ambiguous"].items())


def test_ambiguous_keys_not_in_any_alias_table():
    kinds = ("drugs", "conditions", "sponsors", "countries")
    keys = {_norm(k) for kind in kinds for k in TABLE[kind]}
    keys |= {_norm(a) for c in TABLE["drug_classes"].values() for a in c["aliases"]}
    assert not {_norm(k) for k in TABLE["ambiguous"]} & keys


@pytest.mark.parametrize(("abbr", "meanings"), AMBIG, ids=[a for a, _ in AMBIG])
def test_ambiguous_abbreviation_never_auto_resolves(r, abbr, meanings):
    surface = abbr.upper() if len(abbr) <= 4 else abbr.title()
    assert all(m.surface.lower() != abbr for m in r.scan(f"How many {surface} trials started each year?"))
    plan = rule_plan(VisualizeRequest(query=f"How many {surface} trials started each year?"))
    resolved = {x.lower() for c in plan.cohorts for x in c.drugs + c.conditions + c.sponsors + c.countries}
    assert not resolved & {m.lower() for m in meanings}
    assert plan.clarification and surface in plan.clarification


async def test_llm_cannot_pick_a_meaning_for_ambiguous_abbreviation(settings):
    p = llm_plan(cohorts=[cohort(label="MS", conditions=["multiple sclerosis"])])
    plan, _, _ = await run(
        VisualizeRequest(query="How many MS trials started each year?"), StubLLM({"planner": Queue(p)}), settings
    )
    assert "multiple sclerosis" not in _all(plan, "conditions")
    assert plan.clarification and "MS" in plan.clarification
