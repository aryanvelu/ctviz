"""One regression test (or more) per code-review item on the trial-text branch.

Guiding rule: a wrong "met" is the worst possible bug; when in doubt the answer is needs_review.
"""

from __future__ import annotations

import json

import pytest

import ctviz.handlers.trial_text  # noqa: F401  (registers handlers)
from ctviz.agents.extractors import LLMVerdict, extract_criteria_detailed
from ctviz.agents.llm import StubLLM
from ctviz.api.schemas import Intent
from ctviz.api.schemas.request import PatientProfile
from ctviz.compute.eligibility import heading_section, split_criteria, structured_rules
from ctviz.compute.timeframes import mentions_cycles, parse_durations, parse_timeframe
from ctviz.core.provenance import Study, verify_citation
from ctviz.viz.base import check_encoding, get_handler
from tests.conftest import assert_citations_verified

NCT = "NCT00000002"


def make_study(
    criteria: str | None = None,
    *,
    min_age: str | None = None,
    max_age: str | None = None,
    outcomes: list[str] | None = None,
    healthy: bool | None = None,
) -> Study:
    elig: dict = {}
    if criteria is not None:
        elig["eligibilityCriteria"] = criteria
    if min_age:
        elig["minimumAge"] = min_age
    if max_age:
        elig["maximumAge"] = max_age
    if healthy is not None:
        elig["healthyVolunteers"] = healthy
    ps: dict = {
        "identificationModule": {"nctId": NCT, "briefTitle": "Synthetic"},
        "eligibilityModule": elig,
        "statusModule": {"overallStatus": "RECRUITING", "startDateStruct": {"date": "2024-01-15", "type": "ACTUAL"}},
    }
    if outcomes:
        ps["outcomesModule"] = {
            "primaryOutcomes": [{"measure": f"Outcome {i}", "timeFrame": tf} for i, tf in enumerate(outcomes)]
        }
    return Study({"protocolSection": ps})


def stub(build):
    calls: list[str] = []

    def responder(user, schema):
        calls.append(user)
        return schema(verdicts=build(json.loads(user)["chunks"]))

    return StubLLM({"extraction": responder}), calls


def v(chunk: dict, applies: str, quote: str, confidence: str = "high") -> LLMVerdict:
    return LLMVerdict(
        chunk_id=chunk["chunk_id"], criterion=chunk["criterion"], applies=applies, confidence=confidence, quote=quote
    )


def find(chunks, needle):
    return next(c for c in chunks if needle in c["text"])


def row(out, needle):
    return next(r for r in out.verdicts if needle in r.text)


CRIT = (
    "Inclusion Criteria:\n* Age 18 years or older\n* Diagnosis of type 2 diabetes\n\n"
    "Exclusion Criteria:\n* Pregnancy or breastfeeding\n* Current use of insulin therapy\n"
)


async def run(settings, study, build, profile, focus=None, mode="eligibility"):
    llm, calls = stub(build)
    out = await extract_criteria_detailed(
        study, llm=llm, settings=settings, profile=profile, focus_terms=focus or [], mode=mode
    )
    return out, calls


# -- 1. absence-based verdicts are never final ------------------------------------------------------


async def test_item1_favourable_absence_verdict_is_needs_review(settings):
    s = make_study(CRIT, min_age="18 Years")
    prof = PatientProfile(age=40, conditions=["type 2 diabetes"])
    out, _ = await run(
        settings,
        s,
        lambda ch: [
            v(find(ch, "Pregnancy"), "no", "Pregnancy or breastfeeding"),
            v(find(ch, "insulin"), "no", "Current use of insulin therapy"),
        ],
        prof,
    )
    assert row(out, "Pregnancy").status == "needs_review"
    assert row(out, "insulin").status == "needs_review"


async def test_item1_grounded_and_age_backed_verdicts_are_final(settings):
    s = make_study(CRIT, min_age="18 Years")
    prof = PatientProfile(age=40, conditions=["type 2 diabetes"], medications=["insulin"])
    out, _ = await run(
        settings,
        s,
        lambda ch: [
            v(find(ch, "Diagnosis"), "yes", "Diagnosis of type 2 diabetes"),  # names a profile condition
            v(find(ch, "insulin"), "yes", "Current use of insulin therapy"),  # names a profile medication
            v(find(ch, "Age 18"), "yes", "Age 18 years or older"),  # backed by the age rule
        ],
        prof,
    )
    assert row(out, "Diagnosis").status == "met"
    assert row(out, "insulin").status == "unmet"
    assert row(out, "Age 18").status == "met"


async def test_item1_age_verdict_contradicting_rule_is_not_final(settings):
    s = make_study(CRIT, min_age="18 Years")
    out, _ = await run(
        settings, s, lambda ch: [v(find(ch, "Age 18"), "no", "Age 18 years or older")], PatientProfile(age=40)
    )
    assert row(out, "Age 18").status == "needs_review"


async def test_item1_blocking_verdict_without_grounding_is_needs_review(settings):
    s = make_study(CRIT)
    out, _ = await run(
        settings,
        s,
        lambda ch: [v(find(ch, "Pregnancy"), "yes", "Pregnancy or breastfeeding")],
        PatientProfile(age=30, conditions=["asthma"]),
    )
    assert row(out, "Pregnancy").status == "needs_review"


# -- 2. heading variants and unknown sections -------------------------------------------------------


@pytest.mark.parametrize(
    "line,section",
    [
        ("Exclusion:", "exclusion"),
        ("Exclusions", "exclusion"),
        ("Key Exclusion", "exclusion"),
        ("Inclusion:", "inclusion"),
        ("KEY INCLUSION CRITERIA:", "inclusion"),
        ("Exclusion criteria include:", "exclusion"),
        ("Inclusion/Exclusion Criteria:", "unknown"),
        ("Eligibility Criteria:", "unknown"),
        ("Exclusion of patients with cancer", None),
        ("* Inclusion in another trial", None),
    ],
)
def test_item2_heading_variants(line, section):
    assert heading_section(line) == section


def test_item2_bare_headings_split_sections():
    items = split_criteria("Inclusion:\n- Adults\nExclusions\n- Pregnant\n")
    assert [(c.section, c.text) for c in items] == [("inclusion", "Adults"), ("exclusion", "Pregnant")]


async def test_item2_unknown_section_clamps_llm_verdicts(settings):
    s = make_study("- Diagnosis of type 2 diabetes\n- Current use of insulin therapy")
    out, _ = await run(
        settings,
        s,
        lambda ch: [v(find(ch, "insulin"), "yes", "Current use of insulin therapy")],
        PatientProfile(medications=["insulin"]),
    )
    r = row(out, "insulin")
    assert r.section == "unknown" and r.status == "needs_review" and r.evaluated_by == "llm"


async def test_item2_unknown_section_handler_note(make_ctx, settings):
    s = make_study("- Adults\n- Signed consent")
    res = await get_handler(Intent.ELIGIBILITY_CHECK)(make_ctx(Intent.ELIGIBILITY_CHECK, trials=[s]))
    assert any("no clear Inclusion/Exclusion headings" in n.message for n in res.notes)
    assert {(d.model_extra or {})["section"] for d in res.visualization.data} == {"unknown"}


# -- 3. nested bullets keep parent context ----------------------------------------------------------

NESTED = (
    "Exclusion Criteria:\n"
    "* Must not have any of the following:\n"
    "  * Current use of insulin therapy\n"
    "  * Prior bariatric surgery\n"
    "* Pregnancy or breastfeeding\n"
)


def test_item3_nested_items_record_parent():
    items = split_criteria(NESTED)
    child = next(c for c in items if "insulin" in c.text)
    assert child.parent_text == "Must not have any of the following:" and child.parent_index == 1
    assert next(c for c in items if "Pregnancy" in c.text).parent_text is None


def test_item3_colon_parent_same_indent_children():
    items = split_criteria("Exclusion Criteria:\n* Any of the following:\n* Heart failure\n* Stroke\n\n")
    assert [c.parent_text for c in items] == [None, "Any of the following:", "Any of the following:"]


async def test_item3_child_chunks_carry_parent_and_clamp(settings):
    s = make_study(NESTED)
    out, calls = await run(
        settings,
        s,
        lambda ch: [v(find(ch, "insulin"), "yes", "Current use of insulin therapy")],
        PatientProfile(medications=["insulin"]),
    )
    sent = find(json.loads(calls[0])["chunks"], "insulin")
    assert sent["parent_text"] == "Must not have any of the following:"
    assert row(out, "insulin").status == "needs_review"  # grounded, but meaning depends on the parent


# -- 4. sub-year age limits -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lo,hi,age,expected",
    [
        ("18 Months", None, 1, "needs_review"),  # 1 year = 12-23 months vs 18 months: ambiguous
        ("18 Months", None, 2, "met"),
        ("18 Months", None, 0, "unmet"),
        (None, "18 Months", 1, "needs_review"),
        (None, "18 Months", 2, "unmet"),
        ("18 Years", "65 Years", 18, "met"),  # whole-year limits stay inclusive
        ("18 Years", "65 Years", 65, "met"),
    ],
)
def test_item4_sub_year_limits(lo, hi, age, expected):
    r = structured_rules(min_age=lo, max_age=hi, sex=None, healthy_volunteers=None, profile_age=age)[0]
    assert r.status == expected


# -- 5. cycles, hour lists, negative days -----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Day 1 of each 21-day cycle",
        "Cycle 1 Day 1 and Cycle 3 Day 1",
        "Each cycle is 28 days",
        "Baseline to Cycle 1 (Up to Day 28)",
    ],
)
def test_item5_cycle_time_frames_are_not_parsed(text):
    assert mentions_cycles(text) and parse_timeframe(text) == []


def test_item5_cycle_lengths_are_not_durations():
    assert parse_durations("200 mg on Day 1 of each 21-day cycle; the cycle is 28 days") == []


def test_item5_hour_lists_and_negative_days():
    text = "Predose and 0.5, 1, 2, 4 hours postdose"
    pts = parse_timeframe(text)
    assert [p.day for p in pts] == pytest.approx([0.5 / 24, 1 / 24, 2 / 24, 4 / 24], abs=0.01)
    assert [p.day for p in parse_timeframe("Day -7, Day 1 and Week 4")] == [-7.0, 1.0, 28.0]


async def test_item5_handler_notes_cycles_and_negative_days(make_ctx):
    s = make_study("Inclusion Criteria:\n* Adults", outcomes=["Day 1 of each 21-day cycle", "Day -7, Day 1 and Week 4"])
    res = await get_handler(Intent.TRIAL_JOURNEY_TIMELINE)(make_ctx(Intent.TRIAL_JOURNEY_TIMELINE, trials=[s]))
    msgs = " ".join(n.message for n in res.notes)
    assert "treatment cycles" in msgs and "negative days are not shown" in msgs
    days = [d.model_extra["day"] for d in res.visualization.data if d.model_extra["lane"] == "participant_journey"]
    assert days and min(days) >= 0 and 28 in days


# -- 6. uniform tables ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "intent",
    [
        Intent.VISIT_BURDEN,
        Intent.ENDPOINTS_CATALOG,
        Intent.BIOMARKERS_PKPD,
        Intent.SAFETY_PROVISIONS,
        Intent.STUDY_LIMITATIONS,
    ],
)
@pytest.mark.parametrize("nct", ["NCT03548935", "NCT02578680", "NCT01284335"])
async def test_item6_every_declared_column_in_every_row(make_ctx, study, intent, nct):
    res = await get_handler(intent)(make_ctx(intent, trials=[study(nct)]))
    viz = res.visualization
    check_encoding(viz)
    if viz.meta.columns:
        for d in viz.data:
            assert all((d.model_extra or {}).get(c.field) is not None for c in viz.meta.columns)


# -- 7. quote length and grounding ------------------------------------------------------------------


@pytest.mark.parametrize("quote", ["Pregnancy", "insulin", "No HIV", "  a b  "])
async def test_item7_short_quotes_rejected(settings, quote):
    s = make_study(CRIT + "* No HIV\n")
    out, _ = await run(settings, s, lambda ch: [v(ch[2], "yes", quote)], PatientProfile(medications=["insulin"]))
    assert out.rejected == [(json.loads(json.dumps(out.rejected[0][0])), "quote_too_short")]
    assert all(r.evaluated_by != "llm" for r in out.verdicts)


async def test_item7_final_requires_term_in_quote_not_just_chunk(settings):
    s = make_study(CRIT)
    # the chunk names insulin, but the quote does not -> not final
    out, _ = await run(
        settings,
        s,
        lambda ch: [v(find(ch, "insulin"), "yes", "Current use of")],
        PatientProfile(medications=["insulin"]),
    )
    assert row(out, "insulin").status == "needs_review"


# -- 8. healthy-volunteer rule ----------------------------------------------------------------------


def test_item8_healthy_volunteer_rule_never_met_from_a_condition():
    r = structured_rules(
        min_age=None, max_age=None, sex=None, healthy_volunteers=False, profile_conditions=["obesity"]
    )[0]
    assert r.status == "needs_review"


# -- 9. row text equals its citations ---------------------------------------------------------------


async def test_item9_long_criterion_fully_cited(make_ctx):
    long_item = "Has a documented history of " + ", ".join(f"condition number {i}" for i in range(60))
    assert len(long_item) > 900
    s = make_study(f"Inclusion Criteria:\n* {long_item}\n")
    res = await get_handler(Intent.ELIGIBILITY_CHECK)(make_ctx(Intent.ELIGIBILITY_CHECK, trials=[s]))
    d = next(d for d in res.visualization.data if d.model_extra["section"] == "inclusion")
    texts = [c for c in d.citations if c.field_path.endswith("eligibilityCriteria")]
    assert "".join(c.excerpt for c in texts) == d.model_extra["text"] == long_item
    assert all(len(c.excerpt) <= 300 and verify_citation(c, s.raw) for c in texts)


# -- 10. timeline fields are not mixed --------------------------------------------------------------


async def test_item10_timeline_date_and_day_fields(make_ctx):
    s = make_study("Inclusion Criteria:\n* Adults", outcomes=["Baseline and Week 12"])
    res = await get_handler(Intent.TRIAL_JOURNEY_TIMELINE)(make_ctx(Intent.TRIAL_JOURNEY_TIMELINE, trials=[s]))
    viz = res.visualization
    check_encoding(viz)
    lanes = {ln.name: ln for ln in viz.meta.lanes}
    assert viz.encoding.x is None and lanes["registry_milestones"].start_field == "date"
    assert lanes["participant_journey"].start_field == "day" and lanes["participant_journey"].end_field == "day_end"
    for d in viz.data:
        e = d.model_extra
        if e["lane"] == "registry_milestones":
            assert isinstance(e["date"], str) and "day" not in e
        else:
            assert isinstance(e["day"], int) and isinstance(e["day_end"], int) and "date" not in e
    assert_citations_verified(viz, {NCT: s.raw})


# -- addendum B: one bad enum does not discard valid verdicts ---------------------------------------


async def test_addendum_b_bad_enum_rejected_alone(settings):
    s = make_study(CRIT)
    out, _ = await run(
        settings,
        s,
        lambda ch: [
            v(find(ch, "insulin"), "definitely", "Current use of insulin therapy"),
            v(find(ch, "Diagnosis"), "yes", "Diagnosis of type 2 diabetes", confidence="sure"),
            v(find(ch, "Pregnancy"), "unclear", "Pregnancy or breastfeeding"),
        ],
        PatientProfile(conditions=["diabetes"]),
    )
    assert out.llm_status == "ok" and out.accepted == 1
    assert [r for _, r in out.rejected] == ["invalid_enum", "invalid_enum"]
    assert row(out, "Pregnancy").status == "needs_review" and row(out, "Pregnancy").evaluated_by == "llm"


# -- addendum C: single baseline point --------------------------------------------------------------


def test_addendum_c_single_baseline_point():
    pts = parse_timeframe("Baseline (Day 0) through Week 76")
    assert [(p.day, p.kind) for p in pts if p.day == 0] == [(0.0, "baseline")]
    assert {p.day for p in pts} == {0.0, 532.0}
