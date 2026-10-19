"""Pure compute tests for results/statistics: parsing, AE rates, flow, survival, power."""

from __future__ import annotations

import math

import pytest

from ctviz.compute import adverse_events as ae
from ctviz.compute import flow as fl
from ctviz.compute import power as pw
from ctviz.compute import survival as sv
from ctviz.compute.results import (
    analyses_for,
    effect_kind,
    forest_estimates,
    measure_catalog,
    parse_int,
    parse_num,
    parse_qualified,
)
from ctviz.core.provenance import verify_citation
from tests.conftest import load_study_raw

# -- parsing ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1306", 1306.0),
        ("-12.44", -12.44),
        ("NA", None),
        ("", None),
        (None, None),
        ("<0.001", 0.001),
        ("0.62 (0.52-0.74)", 0.62),
        ("1,234", 1234.0),
        (5, 5.0),
        ("Not reached", None),
        ("abc", None),
        (float("nan"), None),
        (True, None),
    ],
)
def test_parse_num_defensive(raw, expected):
    assert parse_num(raw) == expected


def test_parse_qualified_and_int():
    assert parse_qualified("<0.00001") == (0.00001, "<")
    assert parse_qualified("0.03") == (0.03, None)
    assert parse_int("12") == 12 and parse_int("1.5") is None and parse_int("NA") is None and parse_int("-1") is None


@pytest.mark.parametrize(
    "ptype,kind",
    [
        ("Hazard Ratio (HR)", "ratio"),
        ("Odds Ratio (OR)", "ratio"),
        ("Treatment difference", "difference"),
        ("Mean Difference (Final Values)", "difference"),
        ("Median", None),
    ],
)
def test_effect_kind(ptype, kind):
    assert effect_kind(ptype) == kind


def test_forest_estimates_keynote189_hr(study):
    s = study("NCT02578680")
    ests = analyses_for(s, 1)
    assert len(ests) == 1
    e = ests[0]
    assert (e.estimate, e.ci_low, e.ci_high, e.kind, e.p_qualifier) == (0.49, 0.38, 0.64, "ratio", "<")
    raw = load_study_raw("NCT02578680")
    assert e.refs[0].field_path.endswith("analyses[0].paramValue")
    assert all(verify_citation(r.to_citation(), raw) for r in e.refs)


def test_forest_estimates_difference_keeps_one_unit(study):
    s = study("NCT03548935")
    ests = forest_estimates(s, [0, 1], prefer=None)
    assert {e.kind for e in ests} == {"difference"} and {e.measure_index for e in ests} == {0}
    assert {e.kind for e in forest_estimates(s, [0, 1])} == {"ratio"}  # default prefers ratios


# -- adverse events --------------------------------------------------------------------------------


def test_ae_rates_and_threshold(study):
    s = study("NCT03548935")
    rates = ae.ae_rates(s)
    nausea = [r for r in rates if r.term == "Nausea" and r.seriousness == "other"]
    assert {(r.n_affected, r.n_at_risk) for r in nausea} == {(576, 1306), (114, 655)}
    any_sae = [r for r in rates if r.seriousness == "any_serious"]
    assert len(any_sae) == 2 and all(0 < r.rate < 1 for r in any_sae)
    thr, ref = ae.frequency_threshold(s)
    assert thr == pytest.approx(0.05) and ref.field_path.endswith("frequencyThreshold")
    keys, omitted = ae.top_terms(rates, 15)
    assert len(keys) == 15 and omitted > 0 and keys[0] == ("Nausea", "other")


def test_top_terms_ignores_tiny_groups(study):
    s = study("NCT02578680")  # has a 2-participant crossover event group
    keys, _ = ae.top_terms(ae.ae_rates(s), 5)
    assert keys and all(isinstance(k[0], str) for k in keys)


def test_dose_escalation_detection(study):
    assert ae.is_dose_escalation(study("NCT01284335"))
    assert not ae.is_dose_escalation(study("NCT03548935"))
    assert ae.parse_dose("Drug X 200 mg Q3W") == 200.0 and ae.parse_dose("Placebo") is None
    assert ae.dose_order(["B 10 mg", "Placebo", "A 2.5 mg"]) == [2, 0, 1]


def test_risk_statements_are_verbatim(study):
    s = study("NCT07654361")
    raw = load_study_raw("NCT07654361")
    stmts = ae.risk_statements(s)
    assert any(st.source == "Exclusion criterion" for st in stmts)
    assert all(verify_citation(st.ref.to_citation(), raw) for st in stmts)


# -- participant flow ------------------------------------------------------------------------------


@pytest.mark.parametrize("nct", ["NCT03548935", "NCT02578680", "NCT01844505", "NCT02142738", "NCT01284335"])
def test_flow_conservation(study, nct):
    for arm in fl.flow_table(study(nct)):
        for p in arm.periods:
            assert p.started == p.completed + sum(r.count for r in p.reasons) + p.unexplained or p.inconsistent


def test_flow_step1_values(study):
    arms = fl.flow_table(study("NCT03548935"))
    sema = arms[0]
    assert (sema.started, sema.completed) == (1306, 1240)
    assert {r.label: r.count for r in sema.periods[0].reasons} == {
        "Withdrawal by Subject": 26,
        "Lost to Follow-up": 39,
        "Death": 1,
    }
    assert sema.periods[0].unexplained == 0 and sema.dropout_rate == pytest.approx(66 / 1306)


def test_flow_unexplained_remainder_checkmate(study):
    arms = fl.flow_table(study("NCT01844505"))
    pre = arms[0].periods[0]
    assert pre.started - pre.completed == 3 and pre.reasons == [] and pre.unexplained == 3


# -- survival --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,months",
    [
        ("12 months", 12.0),
        ("Month 6", 6.0),
        ("Rate of OS at 24 months", 24.0),
        ("2-year", 24.0),
        ("52 weeks", 52 * 7 / 30.4375),
        ("none", None),
    ],
)
def test_parse_time_months(text, months):
    got = sv.parse_time_months(text)
    assert (got is None and months is None) or got == pytest.approx(months)


def test_keynote189_anchors(study):
    s = study("NCT02578680")
    ext = sv.extract_anchors(s, [0, 1])
    medians = {k: [(a.time, a.survival) for a in v] for k, v in ext.anchors.items()}
    assert medians["OS · Control"] == [(11.3, 0.5)]
    assert any(k.startswith("PFS · Pembrolizumab") for k in medians) and "PFS · Control" in medians
    assert any(k.startswith("OS · Pembrolizumab") for k in ext.not_reached)  # median OS = "NA"


def test_reconstruct_exponential_and_monotone():
    a = sv.Anchor("S", "OS", "g", 10.0, 0.5, "median")
    pts = sv.reconstruct_survival({"S": [a]}, {"S": []}, step_months=1.0)
    assert pts[0].time == 0 and pts[0].survival == 1.0 and pts[-1].time == 10.0 and pts[-1].survival == 0.5
    lam = math.log(2) / 10
    for p in pts:
        if p.source == "interpolated":
            assert p.survival == pytest.approx(math.exp(-lam * p.time), abs=1e-4)
    surv = [p.survival for p in pts]
    assert surv == sorted(surv, reverse=True)
    assert max(p.time for p in pts) == 10.0  # no extrapolation


def test_checkmate_percent_unit_quirk(study):
    # 'Rate of Progression-Free Survival' says percent but reports proportions (0.52): skipped, not rescaled
    ext = sv.extract_anchors(study("NCT01844505"), [0, 3])
    pfs = ext.anchors["PFS · Nivolumab"]
    assert all(0.3 < a.survival <= 1 for a in pfs) and not ext.dropped


# -- power -----------------------------------------------------------------------------------------


def test_phi_and_inverse():
    assert pw.phi(0) == pytest.approx(0.5)
    assert pw.inv_phi(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert pw.inv_phi(0.8) == pytest.approx(0.841621, abs=1e-5)


def test_power_at_zero_is_one_tail_alpha():
    assert pw.power(0.0, 100) == pytest.approx(0.025, abs=1e-6)


def test_power_monotone_in_d_and_n():
    curve = [p for _, p in pw.power_curve(200)]
    assert curve == sorted(curve) and curve[-1] > 0.99
    assert pw.power(0.3, 50) < pw.power(0.3, 100) < pw.power(0.3, 200)


def test_mde_matches_textbook():
    # classic: n = 64 per arm detects d = 0.5 with ~80% power (alpha 0.05 two-sided)
    assert pw.mde(64) == pytest.approx(0.495, abs=0.005)
    assert pw.power(pw.mde(150), 150) == pytest.approx(0.8, abs=1e-6)
    assert pw.mde(0) == math.inf


def test_harmonic_mean():
    assert pw.harmonic_mean([1306, 655]) == pytest.approx(2 / (1 / 1306 + 1 / 655))


def test_catalog_has_no_numbers(study):
    entry = measure_catalog(study("NCT02578680"))[1].catalog_entry()
    assert set(entry) == {
        "index",
        "title",
        "type",
        "paramType",
        "unit",
        "timeFrame",
        "groups",
        "group_ids",
        "has_analyses",
        "classes",
    }
