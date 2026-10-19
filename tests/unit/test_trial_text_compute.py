"""Pure compute helpers for trial-text intents: eligibility parsing, timeframes, text scan, endpoints."""

from __future__ import annotations

import pytest

from ctviz.compute.eligibility import mentions_any, parse_age, split_criteria, structured_rules
from ctviz.compute.endpoints import classify_endpoint
from ctviz.compute.text_scan import scan_text, split_sentences
from ctviz.compute.timeframes import day_to_week_label, parse_durations, parse_timeframe
from tests.conftest import load_study_raw

FIXTURE_IDS = ["NCT03548935", "NCT07654361", "NCT02578680", "NCT02142738", "NCT01844505", "NCT01284335"]


# -- eligibility parser ----------------------------------------------------------------------------


@pytest.mark.parametrize("nct", FIXTURE_IDS)
def test_split_criteria_spans_are_exact_on_fixtures(nct):
    text = load_study_raw(nct)["protocolSection"]["eligibilityModule"]["eligibilityCriteria"]
    items = split_criteria(text)
    assert items
    for c in items:
        assert text[c.start : c.end] == c.text
        assert c.text == c.text.strip() and c.text
        assert not c.text.startswith(("* ", "- "))
    for sec in ("inclusion", "exclusion"):
        idx = [c.index for c in items if c.section == sec]
        assert idx == list(range(1, len(idx) + 1))


def test_split_criteria_step1_subheadings():
    text = load_study_raw("NCT03548935")["protocolSection"]["eligibilityModule"]["eligibilityCriteria"]
    items = split_criteria(text)
    inc = [c for c in items if c.section == "inclusion"]
    exc = [c for c in items if c.section == "exclusion"]
    assert len(inc) == 5 and len(exc) == 4
    assert inc[0].subheading == "Main phase" and inc[3].subheading == "Extension phase"
    assert exc[0].text.startswith("Glycated haemoglobin (HbA1C)")
    assert all("Main phase" not in c.text and "Criteria" not in c.text for c in items)


def test_split_criteria_skips_preamble_and_empty_bullets():
    bms = load_study_raw("NCT01844505")["protocolSection"]["eligibilityModule"]["eligibilityCriteria"]
    assert all("BMSStudyConnect" not in c.text for c in split_criteria(bms))
    assert all(c.text.strip() for c in split_criteria("Inclusion Criteria:\n\n* a thing\n*\n"))


def test_split_criteria_variants():
    text = (
        "Key Inclusion Criteria\n"
        "1. Adults aged 18 or older\n"
        "2) Diagnosed with type 2\n   diabetes for at least 6 months\n"
        "• HbA1c 7-10%\n"
        "\n"
        "Key Exclusion Criteria:\n"
        "Cohort A:\n"
        "- Prior insulin use\n"
        "  - Including basal insulin\n"
        "Pregnancy\n"
    )
    items = split_criteria(text)
    for c in items:
        assert text[c.start : c.end] == c.text
    inc = [c.text for c in items if c.section == "inclusion"]
    exc = [c for c in items if c.section == "exclusion"]
    assert inc == ["Adults aged 18 or older", "Diagnosed with type 2\n   diabetes for at least 6 months", "HbA1c 7-10%"]
    assert [c.text for c in exc] == ["Prior insulin use", "Including basal insulin", "Pregnancy"]
    assert exc[0].subheading == "Cohort A"


def test_split_criteria_without_headings():
    items = split_criteria("* Adults\n* Signed consent")
    assert [c.text for c in items] == ["Adults", "Signed consent"]
    assert all(c.section == "unknown" and not c.headed for c in items)
    assert split_criteria("") == [] and split_criteria(None) == []


@pytest.mark.parametrize(
    "raw,years",
    [
        ("18 Years", 18),
        ("6 Months", 0.5),
        ("2 Weeks", 14 / 365.25),
        ("N/A", None),
        (None, None),
        ("30 Days", 30 / 365.25),
    ],
)
def test_parse_age(raw, years):
    got = parse_age(raw)
    assert got == pytest.approx(years) if years is not None else got is None


def test_structured_rules_against_profile():
    rules = {
        r.key: r
        for r in structured_rules(
            min_age="18 Years",
            max_age="79 Years",
            sex="ALL",
            healthy_volunteers=False,
            profile_age=45,
            profile_sex="female",
            profile_conditions=["obesity"],
        )
    }
    assert rules["age"].status == "met" and rules["age"].fields == ("min_age", "max_age")
    assert rules["sex"].status == "met" and rules["healthy_volunteers"].status == "needs_review"
    assert (
        structured_rules(min_age="18 Years", max_age=None, sex=None, healthy_volunteers=None, profile_age=16)[0].status
        == "unmet"
    )
    assert (
        structured_rules(min_age="18 Years", max_age="65 Years", sex=None, healthy_volunteers=None, profile_age=70)[
            0
        ].status
        == "unmet"
    )
    assert structured_rules(min_age="18 Years", max_age=None, sex=None, healthy_volunteers=None)[0].status == "unknown"
    female_only = structured_rules(
        min_age=None, max_age=None, sex="FEMALE", healthy_volunteers=None, profile_sex="male"
    )
    assert female_only[0].status == "unmet"


def test_mentions_any_word_boundaries():
    assert mentions_any("Treated with metformin daily", ["Metformin", "in"]) == ["Metformin"]
    assert mentions_any("Glycated haemoglobin", ["insulin"]) == []


# -- timeframes ------------------------------------------------------------------------------------


def _days(text):
    return [(round(p.day), p.kind) for p in parse_timeframe(text)]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Baseline (week 0) to week 68", [(0, "baseline"), (476, "point")]),
        ("Week 0 to Week 68", [(0, "range_start"), (476, "range_end")]),
        ("Day 1 to Day 28", [(1, "range_start"), (28, "range_end")]),
        ("Month 6", [(183, "point")]),
        ("24 months", [(730, "point")]),
        ("1 Year", [(365, "point")]),
        ("Up to approximately 21 months", [(639, "up_to")]),
        ("Up to ~1.6 years", [(584, "up_to")]),
        ("Baseline to Cycle 1 (Up to Day 28)", []),  # cycle-based: left unconverted
        ("Baseline and Week 5, 7 and 11", [(0, "baseline"), (35, "point"), (49, "point"), (77, "point")]),
    ],
)
def test_parse_timeframe(text, expected):
    assert _days(text) == expected


def test_timeframe_spans_and_ignored_offsets():
    text = "Up to approximately 21 months (Serious AEs: Up to 90 days after last dose of study treatment)"
    pts = parse_timeframe(text)
    assert [text[p.start : p.end] for p in pts] == ["21 months"]  # 'after last dose' offset is ignored
    cal = parse_timeframe("Assessed up to September 2016, approximately 39 months")
    assert [round(p.day) for p in cal] == [1187]  # calendar year 2016 is not a duration
    for t in ("Baseline (week 0) to week 68", "Week 5, 7 and 157"):
        for p in parse_timeframe(t):
            assert t[p.start : p.end]
    assert parse_timeframe(None) == [] and parse_timeframe("At the end of the study") == []


def test_parse_durations():
    text = (
        "Participants will receive semaglutide for 68 weeks. The trial included an initial 16-week "
        "dose-escalation period and a 52-week dose maintenance period."
    )
    durs = parse_durations(text)
    assert [round(d.days) for d in durs] == [476, 112, 364]
    assert [text[d.start : d.end] for d in durs][0] == "for 68 weeks"
    assert day_to_week_label(476) == "Week 68" and day_to_week_label(3) == "Day 3"


# -- text scan / endpoints -------------------------------------------------------------------------


def test_split_sentences_offsets():
    text = "First sentence. Second one!\n* Bullet item\n\n" + "word " * 100
    spans = split_sentences(text)
    assert [text[a:b] for a, b in spans[:3]] == ["First sentence.", "Second one!", "Bullet item"]
    assert all(b - a <= 300 for a, b in spans)
    assert all(text[a:b] == text[a:b].strip() for a, b in spans)


def test_scan_text_categories():
    text = "Travel costs will be reimbursed. Participants visit the clinic 15 times. Nothing else."
    hits = scan_text(text, "x", ["costs", "visits"])
    assert [(h.category, text[h.start : h.end]) for h in hits] == [
        ("costs", "Travel costs will be reimbursed."),
        ("visits", "Participants visit the clinic 15 times."),
    ]


@pytest.mark.parametrize(
    "measure,cat",
    [
        ("Number of Participants With Dose-Limiting Toxicities Cycle 1", "dose"),
        ("Pharmacokinetic (PK): Concentration Maximum (Cmax)", "pk"),
        ("Overall Survival (OS)", "efficacy"),
        ("Progression-Free Survival Based on PD-L1 Expression Level", "efficacy"),
        ("Change in HbA1C (%)", "biomarker"),
        ("Change in Short Form 36 (SF-36)", "qol"),
        ("Number of Serious Adverse Events (SAEs)", "safety"),
        ("Change in Body Weight (%)", "efficacy"),
        ("Receptor occupancy at 4 hours", "pd"),
        ("Something unrelated", "other"),
    ],
)
def test_classify_endpoint(measure, cat):
    res = classify_endpoint(measure)
    assert res.category == cat
    if res.start is not None:
        assert measure[res.start : res.end] == res.term
