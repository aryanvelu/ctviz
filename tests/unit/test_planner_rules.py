"""Rule-based planner: every design-doc / intents.yaml question maps to the right intent + scope."""

from __future__ import annotations

import datetime as dt

import pytest
import yaml

from ctviz.agents.rules_planner import (
    extract_phases,
    extract_statuses,
    extract_years,
    rule_plan,
)
from ctviz.api.schemas import Intent, Persona, Scope, VisualizeRequest
from ctviz.core.intents import INTENTS_FILE, load_intents

NCT = "NCT03548935"
IN, S = Intent, Scope

PATIENT = [
    ("What is the main goal of this study?", IN.STUDY_RATIONALE),
    ("Why do researchers think this treatment will work?", IN.STUDY_RATIONALE),
    ("Do I meet the inclusion and exclusion criteria to join?", IN.ELIGIBILITY_CHECK),
    ("Can I continue taking my current medications?", IN.CONCOMITANT_MEDICATIONS),
    ("How long will the trial last?", IN.TRIAL_JOURNEY_TIMELINE),
    ("How many clinic visits or hospital stays are required?", IN.VISIT_BURDEN),
    ("Will I still be able to see my regular doctor?", IN.NOT_IN_REGISTRY),  # user decision
    ("What are the known or potential risks and side effects?", IN.RISKS_ADVERSE_EVENTS),
    ("What happens if my condition gets worse during the study?", IN.SAFETY_PROVISIONS),
    ("Will the trial cost me anything out of pocket, or are travel and parking reimbursed?", IN.NOT_IN_REGISTRY),
    ("What were the final results?", IN.RESULTS_OUTCOMES),
]

RESEARCHER_TRIAL = [
    ("What is the study design and methodology?", IN.DESIGN_METHODOLOGY),
    ("Which phase is this trial and what endpoints are measured: safety, dosage or efficacy?", IN.ENDPOINTS_CATALOG),
    ("How are randomization and blinding structured?", IN.DESIGN_METHODOLOGY),
    ("What is the statistical power and sample size?", IN.ENROLLMENT_VS_POWER),
    ("What biomarkers and PK/PD are tracked?", IN.BIOMARKERS_PKPD),
    ("How many patients dropped out?", IN.DROPOUT_FLOW),
    ("What was the patient demographic?", IN.DEMOGRAPHICS),
    ("How were protocol deviations handled?", IN.STUDY_LIMITATIONS),
]

# (question, structured fields, intent, scope)
COHORT = [
    (
        "How did this drug advance from first-in-human to later phases?",
        {"drug_name": "semaglutide"},
        IN.DEVELOPMENT_PROGRESSION,
        S.COHORT,
    ),
    ("Phase vs recruitment status heatmap across GLP-1 trials", {}, IN.PHASE_STATUS_HEATMAP, S.COHORT),
    ("What is Novo Nordisk doing in obesity?", {}, IN.SPONSOR_ACTIVITY, S.COHORT),
    ("How has the number of trials for pembrolizumab changed per year since 2015?", {}, IN.TREND_OVER_TIME, S.COHORT),
    ("How many trials started each year for Alzheimer's disease?", {}, IN.TREND_OVER_TIME, S.COHORT),
    ("How are breast cancer trials distributed across phases?", {}, IN.PHASE_DISTRIBUTION, S.COHORT),
    ("What are the most common intervention types for insomnia trials?", {}, IN.INTERVENTION_LANDSCAPE, S.COHORT),
    ("Compare phases for trials involving semaglutide vs tirzepatide", {}, IN.PHASE_DISTRIBUTION, S.COMPARISON),
    ("Compare sponsor categories across breast cancer and lung cancer", {}, IN.SPONSOR_LANDSCAPE, S.COMPARISON),
    ("Which countries have the most recruiting trials for type 2 diabetes?", {}, IN.GEOGRAPHIC_DISTRIBUTION, S.COHORT),
    ("Show a network of sponsors and drugs for obesity trials", {}, IN.ENTITY_NETWORK, S.COHORT),
    ("Which drugs frequently co-occur in combination studies?", {}, IN.ENTITY_NETWORK, S.COHORT),
]


def _yaml_examples() -> list[tuple[str, Intent]]:
    raw = yaml.safe_load(INTENTS_FILE.read_text())
    return [(q, Intent(k)) for k, v in raw.items() for q in v.get("examples", [])]


def _plan(q: str, **kw):
    return rule_plan(VisualizeRequest(query=q, **kw))


@pytest.mark.parametrize(("question", "intent"), PATIENT + RESEARCHER_TRIAL)
def test_trial_questions_with_nct(question, intent):
    plan = _plan(question, nct_id=NCT)
    assert plan.intent is intent
    assert plan.scope is S.SINGLE_TRIAL
    assert plan.nct_ids == [NCT]
    assert plan.cohorts == []
    assert plan.parser == "rules"


@pytest.mark.parametrize(("question", "fields", "intent", "scope"), COHORT)
def test_cohort_questions(question, fields, intent, scope):
    plan = _plan(question, **fields)
    assert plan.intent is intent
    assert plan.scope is scope
    assert plan.cohorts, "cohort scopes always carry at least one cohort"


@pytest.mark.parametrize(("question", "intent"), _yaml_examples())
def test_every_intents_yaml_example(question, intent):
    plan = _plan(question)
    assert plan.intent is intent
    spec = load_intents()[intent]
    assert plan.scope in spec.scopes


def test_every_intent_has_an_example_covered():
    assert {i for _, i in _yaml_examples()} == set(Intent)


# --------------------------------------------------------------------------- entity + filter details
def test_pembrolizumab_trend_entities():
    plan = _plan("How has the number of trials for pembrolizumab changed per year since 2015?")
    (c,) = plan.cohorts
    assert c.drugs == ["pembrolizumab"] and c.start_year == 2015
    assert plan.time_granularity == "year"


def test_alzheimers_condition():
    (c,) = _plan("How many trials started each year for Alzheimer's disease?").cohorts
    assert c.conditions == ["Alzheimer's disease"]


def test_semaglutide_vs_tirzepatide_two_cohorts():
    plan = _plan("Compare phases for trials involving semaglutide vs tirzepatide")
    assert [c.drugs for c in plan.cohorts] == [["semaglutide"], ["tirzepatide"]]
    assert [c.label for c in plan.cohorts] == ["Semaglutide", "Tirzepatide"]
    assert plan.group_by == "phase"


def test_sponsor_categories_comparison():
    plan = _plan("Compare sponsor categories across breast cancer and lung cancer")
    assert plan.group_by == "sponsor_class"
    assert [c.conditions for c in plan.cohorts] == [["breast cancer"], ["lung cancer"]]


def test_recruiting_becomes_status_filter():
    (c,) = _plan("Which countries have the most recruiting trials for type 2 diabetes?").cohorts
    assert c.statuses == ["RECRUITING"]
    assert c.conditions == ["type 2 diabetes"]


def test_status_distribution_does_not_filter_its_own_dimension():
    (c,) = _plan("How many semaglutide trials are recruiting?").cohorts
    assert c.statuses == []


def test_network_kinds():
    assert _plan("Show a network of sponsors and drugs for obesity trials").network_kind == "sponsor_drug"
    assert _plan("Which drugs frequently co-occur in combination studies?").network_kind == "drug_drug"


def test_heatmap_vs_is_not_a_comparison_and_expands_glp1():
    plan = _plan("Phase vs recruitment status heatmap across GLP-1 trials")
    (c,) = plan.cohorts
    assert {"semaglutide", "liraglutide", "dulaglutide", "exenatide", "tirzepatide", "lixisenatide"} <= set(c.drugs)
    assert c.statuses == [] and c.phases == []


def test_sponsor_activity_entities():
    (c,) = _plan("What is Novo Nordisk doing in obesity?").cohorts
    assert c.sponsors == ["Novo Nordisk"] and c.conditions == ["obesity"]


def test_structured_fields_override_text():
    plan = _plan(
        "How has the number of trials for this drug changed over time?",
        drug_name="Keytruda",
        trial_phase=["3"],
        start_year=2018,
    )
    (c,) = plan.cohorts
    assert c.drugs == ["pembrolizumab"] and c.phases == ["PHASE3"] and c.start_year == 2018


def test_structured_drug_names_make_comparison():
    plan = _plan("How many trials per year?", drug_names=["Ozempic", "Mounjaro"])
    assert plan.scope is S.COMPARISON
    assert [c.drugs for c in plan.cohorts] == [["semaglutide"], ["tirzepatide"]]


def test_two_nct_ids_is_comparison():
    plan = _plan("Compare the designs of NCT03548935 and NCT02142738")
    assert plan.scope is S.COMPARISON and plan.nct_ids == ["NCT03548935", "NCT02142738"]


def test_nct_in_text_is_single_trial():
    plan = _plan("What were the final results of nct03548935?")
    assert plan.scope is S.SINGLE_TRIAL and plan.nct_ids == [NCT]


def test_single_trial_intent_without_nct_asks_for_trial():
    plan = _plan("How long will the trial last?")
    assert plan.scope is S.SINGLE_TRIAL and plan.nct_ids == [] and plan.clarification


def test_persona_detection():
    assert _plan("Do I meet the criteria?", nct_id=NCT).persona is Persona.PATIENT
    assert _plan("Tell me about NCT03548935").persona is Persona.RESEARCHER
    assert _plan("How many trials per year for obesity?", persona="patient").persona is Persona.PATIENT
    assert _plan("What is the study design?", nct_id=NCT).persona is Persona.RESEARCHER


def test_ambiguous_abbreviation_sets_clarification():
    plan = _plan("How many MS trials started each year?")
    assert plan.clarification and "MS" in plan.clarification


def test_no_entities_assumption():
    plan = _plan("Which drugs frequently co-occur in combination studies?")
    assert plan.cohorts[0].label == "All trials"
    assert plan.assumptions


def test_comparison_merged_for_intent_without_comparison_scope():
    plan = _plan("Network of sponsors for semaglutide vs tirzepatide")
    assert plan.intent is IN.ENTITY_NETWORK and plan.scope is S.COHORT
    assert set(plan.cohorts[0].drugs) == {"semaglutide", "tirzepatide"}


def test_request_overrides_viz_prefs():
    plan = _plan("Trials per year for obesity", group_by="phase", top_n=7, preferred_visualization="bar_chart")
    assert plan.group_by == "phase" and plan.top_n == 7 and plan.preferred_visualization == "bar_chart"


# --------------------------------------------------------------------------- extractors
TODAY = dt.date(2026, 10, 18)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("trials since 2015", (2015, None)),
        ("from 2015-2020", (2015, 2020)),
        ("between 2012 and 2018", (2012, 2018)),
        ("before 2020", (None, 2019)),
        ("after 2018", (2019, None)),
        ("in 2019", (2019, 2019)),
        ("over the last 5 years", (2021, None)),
        ("since 1950", (1990, None)),
        ("2020 to 2015", (2015, 2020)),
        ("NCT03548935 started", (None, None)),
    ],
)
def test_extract_years(text, expected):
    assert extract_years(text, TODAY) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("phase 3 trials", ["PHASE3"]),
        ("Phase III", ["PHASE3"]),
        ("phase 2/3 studies", ["PHASE2", "PHASE3"]),
        ("phase 1 and phase 2", ["PHASE1", "PHASE2"]),
        ("early phase 1", ["EARLY_PHASE1"]),
        ("all phases", []),
    ],
)
def test_extract_phases(text, expected):
    assert extract_phases(text) == expected


def test_extract_statuses():
    assert extract_statuses("not yet recruiting trials") == ["NOT_YET_RECRUITING"]
    assert extract_statuses("recruiting trials") == ["RECRUITING"]
    assert extract_statuses("recruitment status") == []
    assert "COMPLETED" in extract_statuses("completed trials of x")
