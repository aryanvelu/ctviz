"""Single-trial text handlers on recorded fixtures: shapes, notes, honesty, and the citation invariant."""

from __future__ import annotations

import pytest

import ctviz.handlers.trial_text  # noqa: F401  (registers handlers)
from ctviz.api.schemas import Intent, VisualizeRequest
from ctviz.api.schemas.request import PatientProfile
from ctviz.api.schemas.viz import VizType
from ctviz.viz.base import get_handler
from tests.conftest import assert_citations_verified, load_study_raw

FIXTURE_IDS = ["NCT03548935", "NCT07654361", "NCT02578680", "NCT02142738", "NCT01844505", "NCT01284335"]
TRIAL_INTENTS = [
    Intent.SINGLE_TRIAL_OVERVIEW,
    Intent.STUDY_RATIONALE,
    Intent.ELIGIBILITY_CHECK,
    Intent.CONCOMITANT_MEDICATIONS,
    Intent.TRIAL_JOURNEY_TIMELINE,
    Intent.VISIT_BURDEN,
    Intent.SITES_AND_CONTACTS,
    Intent.SAFETY_PROVISIONS,
    Intent.STUDY_LIMITATIONS,
    Intent.ENDPOINTS_CATALOG,
    Intent.BIOMARKERS_PKPD,
    Intent.NOT_IN_REGISTRY,
]
STEP1, NO_RESULTS = "NCT03548935", "NCT07654361"


def codes(res):
    return [n.code.value for n in res.notes]


def rows(res):
    return [d.model_extra or {} for d in res.visualization.data]


@pytest.fixture
def run(make_ctx, study):
    async def go(intent, nct, query="test", profile=None, **plan):
        ctx = make_ctx(intent, trials=[study(nct)], query=query, persona="patient", **plan)
        if profile is not None:
            ctx.request = VisualizeRequest(query=query, patient_profile=profile)
        return await get_handler(intent)(ctx)

    return go


def test_all_trial_text_intents_registered():
    for intent in TRIAL_INTENTS:
        assert get_handler(intent) is not None
    handler = get_handler(Intent.DESIGN_METHODOLOGY)  # owned by the cohort builder, not this module
    assert handler is None or handler.__module__ != "ctviz.handlers.trial_text"


@pytest.mark.parametrize("nct", FIXTURE_IDS)
@pytest.mark.parametrize("intent", TRIAL_INTENTS, ids=lambda i: i.value)
async def test_citation_invariant(run, intent, nct):
    res = await run(
        intent,
        nct,
        query="Is parking reimbursed?",
        profile=PatientProfile(age=45, sex="female", conditions=["obesity"]),
    )
    viz = res.visualization
    assert viz.data, "handlers never return an empty viz"
    checked = assert_citations_verified(viz, {nct: load_study_raw(nct)})
    assert checked > 0
    for d in viz.data:
        for c in d.citations:
            assert len(c.excerpt) <= 300 and c.nct_id == nct
    assert set(res.full_refs) == {d.id for d in viz.data}


@pytest.mark.parametrize("intent", TRIAL_INTENTS, ids=lambda i: i.value)
async def test_no_trial_returns_trial_required(make_ctx, intent):
    res = await get_handler(intent)(make_ctx(intent))
    assert res.visualization.type is VizType.KPI_CARDS
    assert "TRIAL_REQUIRED" in codes(res)


async def test_overview_kpis(run):
    res = await run(Intent.SINGLE_TRIAL_OVERVIEW, STEP1)
    by = {r["label"]: r for r in rows(res)}
    assert by["Status"]["value"] == "COMPLETED" and by["Phase"]["value"] == "Phase 3"
    assert by["Enrollment"]["kind"] == "number" and by["Enrollment"]["detail"] == "ACTUAL"
    assert by["Sites listed"]["value"] == 128 and by["Results posted"]["value"] == "yes"
    assert all(r["kind"] in ("number", "text", "date", "enum", "list") for r in rows(res))
    enc = res.visualization.encoding
    assert enc.label.field == "label" and enc.y.field == "value"


async def test_overview_no_results_trial(run):
    res = await run(Intent.SINGLE_TRIAL_OVERVIEW, NO_RESULTS)
    by = {r["label"]: r for r in rows(res)}
    assert by["Results posted"]["value"] == "no" and by["Study completion"]["detail"] == "ESTIMATED"
    assert "NO_RESULTS_POSTED" in codes(res)


async def test_rationale_hypothesis_and_absence(run):
    res = await run(Intent.STUDY_RATIONALE, "NCT02142738")
    assert any("hypothesis" in str(r["value"]) for r in rows(res))
    absent = await run(Intent.STUDY_RATIONALE, STEP1)
    assert "NOT_IN_REGISTRY" in codes(absent)
    assert any(r["value"] == "Not stated in the ClinicalTrials.gov record" for r in rows(absent))


async def test_eligibility_rules_only(run):
    res = await run(Intent.ELIGIBILITY_CHECK, NO_RESULTS, profile=PatientProfile(age=82, sex="male"))
    viz = res.visualization
    assert viz.type is VizType.ELIGIBILITY_CHECKLIST
    rs = rows(res)
    age = next(r for r in rs if r["section"] == "structured" and r["text"].startswith("Age"))
    assert age["status"] == "unmet" and age["evaluated_by"] == "rule"
    free = [r for r in rs if r["section"] != "structured"]
    assert free and all(r["status"] == "unknown" and r["evaluated_by"] == "none" for r in free)
    assert viz.meta.summary["total"] == len(rs) and viz.meta.summary["by_status"]["unmet"] == 1
    assert viz.meta.diagnostics["profile_used"] == {"age": 82, "sex": "male"}
    raw = load_study_raw(NO_RESULTS)["protocolSection"]["eligibilityModule"]["eligibilityCriteria"]
    for d in viz.data:
        if d.model_extra["section"] != "structured":
            c = d.citations[0]
            assert raw[c.char_start : c.char_end] == c.excerpt and d.model_extra["text"].startswith(c.excerpt)


async def test_eligibility_without_profile_notes_it(run):
    res = await run(Intent.ELIGIBILITY_CHECK, STEP1)
    assert any("No patient profile" in n.message for n in res.notes)
    assert res.visualization.meta.diagnostics["profile_used"] == {}


async def test_concomitant_medications_rows_are_medication_related(run):
    res = await run(Intent.CONCOMITANT_MEDICATIONS, "NCT02578680")
    assert res.visualization.type is VizType.ELIGIBILITY_CHECKLIST
    assert any("steroid" in r["text"] for r in rows(res))
    assert "NOT_IN_REGISTRY" in codes(res)


async def test_journey_timeline_lanes(run):
    res = await run(Intent.TRIAL_JOURNEY_TIMELINE, STEP1)
    viz = res.visualization
    assert viz.type is VizType.TIMELINE
    lanes = {r["lane"] for r in rows(res)}
    assert lanes == {"registry_milestones", "participant_journey"}
    reg = [r for r in rows(res) if r["lane"] == "registry_milestones"]
    assert {"Study start", "Primary completion", "Study completion"} <= {r["event"] for r in reg}
    assert all(r["date_type"] == "ACTUAL" for r in reg)
    part = [r for r in rows(res) if r["lane"] == "participant_journey"]
    assert any(r["day"] == 476 for r in part)  # Week 68
    assert any(r["event"].startswith("Treatment period") and r["day_end"] == 476 for r in part)
    # no mixed types in one field: registry rows carry calendar `date`, participant rows integer `day`
    assert all("date" in r and "day" not in r for r in reg)
    assert all(isinstance(r["day"], int) and "date" not in r for r in part)
    lanes_meta = {ln.name: ln for ln in viz.meta.lanes}
    assert set(lanes_meta) == {"registry_milestones", "participant_journey"}
    assert viz.encoding.series.field == "lane" and viz.meta.time_axis is None
    assert lanes_meta["registry_milestones"].axis.kind == "calendar"
    assert lanes_meta["participant_journey"].axis.kind == "relative"
    assert lanes_meta["participant_journey"].axis.unit == "days" and lanes_meta["participant_journey"].approximated
    assert viz.meta.approximation.is_approximation
    approx = next(n for n in res.notes if n.code.value == "APPROXIMATED")
    part_ids = {d.id for d in viz.data if d.model_extra["lane"] == "participant_journey"}
    assert set(approx.affects) == part_ids


async def test_journey_estimated_and_partial_dates(run):
    res = await run(Intent.TRIAL_JOURNEY_TIMELINE, NO_RESULTS)
    reg = {r["event"]: r for r in rows(res) if r["lane"] == "registry_milestones"}
    assert reg["Study completion"]["date_type"] == "ESTIMATED"
    assert "PARTIAL_DATES" in codes(res)  # 2028-11 has no day
    assert any(r["day"] == 532 for r in rows(res) if r["lane"] == "participant_journey")  # Week 76


async def test_visit_burden_lower_bound(run):
    res = await run(Intent.VISIT_BURDEN, STEP1)
    rs = rows(res)
    lb = next(r for r in rs if r["kind"] == "lower_bound")
    tps = [r for r in rs if r["kind"] == "assessment_timepoint"]
    assert lb["value"] == len(tps) >= 2
    assert any("clinic visits" in r["event"] for r in rs if r["kind"] == "visit_mention")
    assert {"APPROXIMATED", "NOT_IN_REGISTRY"} <= set(codes(res))
    assert res.visualization.meta.columns


async def test_sites_geo_map(run):
    res = await run(Intent.SITES_AND_CONTACTS, NO_RESULTS)
    viz = res.visualization
    assert viz.type is VizType.GEO_MAP
    assert viz.meta.geo.level == "site" and viz.meta.geo.lat_field == "lat"
    sites = [r for r in rows(res) if r["row_type"] == "site"]
    assert len(sites) == 71 and all({"facility", "city", "country", "lat", "lon", "status"} <= set(r) for r in sites)
    assert any(r["row_type"] == "central_contact" for r in rows(res))
    assert any("regular doctor" in n.message and n.code.value == "NOT_IN_REGISTRY" for n in res.notes)


async def test_sites_absent_falls_back_to_contacts(run):
    res = await run(Intent.SITES_AND_CONTACTS, "NCT02142738")
    assert res.visualization.type is VizType.KPI_CARDS
    labels = [r["label"] for r in rows(res)]
    assert "Study sites" in labels and ("Study official" in labels or "Results point of contact" in labels)
    assert any("regular doctor" in n.message for n in res.notes)


async def test_safety_provisions(run):
    res = await run(Intent.SAFETY_PROVISIONS, "NCT01284335")
    rs = rows(res)
    assert res.visualization.type is VizType.TABLE
    assert any(r["topic"] == "Independent monitoring" for r in rs)
    assert any("unacceptable toxicity" in r["text"] for r in rs)
    assert "NOT_IN_REGISTRY" in codes(res)


async def test_study_limitations(run):
    res = await run(Intent.STUDY_LIMITATIONS, "NCT01284335")
    topics = {r["topic"] for r in rows(res)}
    assert {"Overall status", "Why the study stopped", "Limitations and caveats", "Pre-assignment details"} <= topics
    assert any("deviation" in n.message for n in res.notes)
    none = await run(Intent.STUDY_LIMITATIONS, NO_RESULTS)
    assert none.visualization.type is VizType.KPI_CARDS and "NO_RESULTS_POSTED" in codes(none)


async def test_endpoints_catalog(run):
    res = await run(Intent.ENDPOINTS_CATALOG, "NCT01284335")
    cats = {r["category_key"] for r in rows(res)}
    assert {"dose", "pk", "efficacy"} <= cats
    assert all(r["outcome_type"] in ("Primary", "Secondary", "Other") for r in rows(res))
    assert res.visualization.meta.columns[0].field == "endpoint"


async def test_endpoints_cohort_bar_chart(make_ctx, cohort):
    c = cohort("semaglutide", label="Semaglutide")
    res = await get_handler(Intent.ENDPOINTS_CATALOG)(make_ctx(Intent.ENDPOINTS_CATALOG, cohorts=[c]))
    raw = {s.nct_id: s.raw for s in c.studies}
    assert res.visualization.type in (VizType.BAR_CHART, VizType.KPI_CARDS)
    assert_citations_verified(res.visualization, raw)


async def test_biomarkers_pkpd(run):
    res = await run(Intent.BIOMARKERS_PKPD, STEP1)
    cats = {r["category_key"] for r in rows(res)}
    assert cats <= {"pk", "pd", "biomarker"} and "biomarker" in cats
    assert any("HbA1C" in r["endpoint"] for r in rows(res))
    pk = await run(Intent.BIOMARKERS_PKPD, "NCT01284335")
    assert any(r["category_key"] == "pk" for r in rows(pk))


async def test_not_in_registry_costs(run):
    res = await run(Intent.NOT_IN_REGISTRY, NO_RESULTS, query="Will it cost me anything? Is parking reimbursed?")
    rs = rows(res)
    assert res.visualization.type is VizType.KPI_CARDS
    assert any(r["value"] == "Not stated in the ClinicalTrials.gov record" for r in rs)
    assert any(r["label"] == "Central contact" and "1-650-457-1959" in r["value"] for r in rs)
    assert "NOT_IN_REGISTRY" in codes(res)
    assert {r["label"] for r in rs} >= {"Costs to participants", "Travel and parking reimbursement"}
