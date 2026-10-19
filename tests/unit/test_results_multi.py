"""Results intents across several trials (cohort / comparison scope, or several NCT ids) + survival curve validity."""

from __future__ import annotations

from collections import defaultdict

import pytest

from ctviz.api.schemas import Intent, Scope
from ctviz.api.schemas.response import NoteCode
from ctviz.api.schemas.viz import VizType
from ctviz.compute import survival as sv
from ctviz.handlers import results as R
from ctviz.viz.base import CohortData
from tests.conftest import assert_citations_verified, load_study_raw

SURV = ["NCT02578680", "NCT02142738", "NCT01844505"]  # KEYNOTE-189, KEYNOTE-024, CheckMate 067


def _raw(*ncts):
    return {n: load_study_raw(n) for n in ncts}


def _rows(viz):
    return [d.model_extra or {} for d in viz.data]


def _codes(res):
    return {n.code for n in res.notes}


def _by_series(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["series"]].append((r["time"], r["survival"]))
    return out


def _assert_valid_curves(rows):
    """Every series starts at S(0) = 1 and never rises."""
    series = _by_series(rows)
    assert series
    for name, pts in series.items():
        assert pts[0] == (0.0, 1.0), name
        times = [t for t, _ in pts]
        surv = [s for _, s in pts]
        assert times == sorted(times), name
        assert all(b <= a for a, b in zip(surv, surv[1:], strict=False)), name
        assert all(0.0 <= s <= 1.0 for s in surv), name


# -- survival curves are never upside down -----------------------------------------------------------


@pytest.mark.parametrize("nct", SURV)
async def test_single_trial_curves_start_at_one_and_decrease(make_ctx, study, nct):
    ctx = make_ctx(Intent.SURVIVAL_SUMMARY, trials=[study(nct)], query="Show overall survival")
    res = await R.survival_summary(ctx)
    assert res.visualization.type is VizType.SURVIVAL_CURVE
    assert res.visualization.encoding.y.field == "survival"
    assert res.visualization.encoding.y.scale.domain == [0, 1]
    _assert_valid_curves(_rows(res.visualization))


def _pt(series, t, s, kind="landmark"):
    return sv.CurvePoint(series, "OS", "A", t, s, "reported", [], kind)


def test_curve_problems_flags_rising_and_bad_start():
    good = [_pt("a", 0, 1.0, "start"), _pt("a", 6, 0.8), _pt("a", 12, 0.5, "median")]
    rising = [_pt("b", 0, 1.0, "start"), _pt("b", 6, 0.4), _pt("b", 12, 0.6)]
    bad_start = [_pt("c", 0, 0.0, "start"), _pt("c", 6, 0.3)]
    probs = sv.curve_problems(good + rising + bad_start)
    assert set(probs) == {"b", "c"}
    assert "rises" in probs["b"] and "S(0) = 1" in probs["c"]


async def test_invalid_series_is_dropped_with_note(make_ctx, study, monkeypatch):
    def rising(anchors, start_refs, step_months=None):
        return [_pt("OS · X", 0, 1.0, "start"), _pt("OS · X", 6, 0.4), _pt("OS · X", 12, 0.7)]

    monkeypatch.setattr(sv, "reconstruct_survival", rising)
    ctx = make_ctx(Intent.SURVIVAL_SUMMARY, trials=[study("NCT02578680")], query="overall survival")
    res = await R.survival_summary(ctx)
    assert res.visualization.type is not VizType.SURVIVAL_CURVE  # nothing valid left to draw
    assert any("OS · X: not drawn" in n.message for n in res.notes)


# -- M5: several NCT ids are all analysed ------------------------------------------------------------


async def test_survival_two_named_trials_series_per_trial(make_ctx, study):
    ctx = make_ctx(
        Intent.SURVIVAL_SUMMARY,
        trials=[study("NCT02578680"), study("NCT02142738")],
        scope=Scope.COMPARISON,
        query="Compare survival in NCT02578680 and NCT02142738",
    )
    res = await R.survival_summary(ctx)
    viz = res.visualization
    assert viz.type is VizType.SURVIVAL_CURVE
    assert viz.meta.approximation.is_approximation
    rows = _rows(viz)
    assert {r["nct_id"] for r in rows} == {"NCT02578680", "NCT02142738"}
    assert all(r["series"].startswith(("NCT02578680", "NCT02142738", "KEYNOTE")) for r in rows)
    _assert_valid_curves(rows)
    assert all(d.nct_ids == [(d.model_extra or {})["nct_id"]] for d in viz.data)
    assert_citations_verified(viz, _raw("NCT02578680", "NCT02142738"))
    assert NoteCode.APPROXIMATED in _codes(res)


async def test_survival_named_trial_without_results_is_listed(make_ctx, study):
    ctx = make_ctx(
        Intent.SURVIVAL_SUMMARY, trials=[study("NCT02578680"), study("NCT07654361")], query="survival curves"
    )
    res = await R.survival_summary(ctx)
    assert {r["nct_id"] for r in _rows(res.visualization)} == {"NCT02578680"}
    msg = next(n.message for n in res.notes if n.code is NoteCode.NO_RESULTS_POSTED)
    assert "NCT07654361" in msg


async def test_results_two_named_trials_forest_rows_per_trial(make_ctx, study):
    ctx = make_ctx(Intent.RESULTS_OUTCOMES, trials=[study("NCT02578680"), study("NCT02142738")], query="results")
    res = await R.results_outcomes(ctx)
    viz = res.visualization
    assert viz.type is VizType.FOREST_PLOT
    assert {r["nct_id"] for r in _rows(viz)} == {"NCT02578680", "NCT02142738"}
    assert_citations_verified(viz, _raw("NCT02578680", "NCT02142738"))
    for d in viz.data:
        assert any(c.field_path and ".analyses[" in c.field_path for c in d.citations)


async def test_demographics_several_trials_notes_the_unanalysed(make_ctx, study):
    ctx = make_ctx(Intent.DEMOGRAPHICS, trials=[study("NCT03548935"), study("NCT02578680")])
    res = await R.demographics(ctx)
    assert any("NCT02578680" in n.message and n.code is NoteCode.TRUNCATED for n in res.notes)


# -- C1: cohort / comparison scope -------------------------------------------------------------------


async def test_survival_cohort_hazard_ratio_forest(make_ctx, study):
    trials = [study(n) for n in SURV]
    ctx = make_ctx(Intent.SURVIVAL_SUMMARY, trials=trials, scope=Scope.COHORT, query="hazard ratios for PFS")
    ctx.plan = ctx.plan.model_copy(update={"nct_ids": []})  # cohort search, not named trials
    res = await R.survival_summary(ctx)
    viz = res.visualization
    assert viz.type is VizType.FOREST_PLOT
    rows = _rows(viz)
    assert len({r["nct_id"] for r in rows}) >= 2
    assert all("hazard" in r["param_type"].lower() for r in rows)
    assert all(r["ci_low"] <= r["estimate"] <= r["ci_high"] for r in rows)
    assert viz.meta.reference_line.value == 1.0
    assert_citations_verified(viz, _raw(*SURV))
    # rows are ordered by enrollment, largest trial first
    enrol = {t.nct_id: R._enrollment(t) for t in trials}
    firsts = list(dict.fromkeys(r["nct_id"] for r in rows))
    assert [enrol[n] for n in firsts] == sorted((enrol[n] for n in firsts), reverse=True)


async def test_survival_cohort_kaplan_meier_curves(make_ctx, study):
    ctx = make_ctx(
        Intent.SURVIVAL_SUMMARY,
        trials=[study(n) for n in SURV],
        scope=Scope.COHORT,
        query="kaplan meier for pembrolizumab",
    )
    ctx.plan = ctx.plan.model_copy(update={"nct_ids": []})
    res = await R.survival_summary(ctx)
    viz = res.visualization
    assert viz.type is VizType.SURVIVAL_CURVE and viz.meta.approximation.is_approximation
    rows = _rows(viz)
    assert len({r["nct_id"] for r in rows}) == 3
    assert len({r["endpoint"] for r in rows if r["nct_id"] == "NCT02578680"}) == 1  # one endpoint per trial
    _assert_valid_curves(rows)
    assert_citations_verified(viz, _raw(*SURV))


async def test_survival_comparison_rows_carry_cohort(make_ctx, study):
    a = CohortData(
        label="Pembrolizumab", spec=None, studies=[study("NCT02578680"), study("NCT02142738")], total_count=2
    )
    b = CohortData(label="Nivolumab", spec=None, studies=[study("NCT01844505")], total_count=1)
    ctx = make_ctx(Intent.SURVIVAL_SUMMARY, cohorts=[a, b], query="compare survival hazard ratio pembro vs nivo")
    ctx.trials = [study(n) for n in SURV]  # what the orchestrator's result-record fetch leaves in ctx.trials
    res = await R.survival_summary(ctx)
    viz = res.visualization
    assert viz.type is VizType.FOREST_PLOT
    assert viz.encoding.series.field == "cohort"
    by_trial = {r["nct_id"]: r["cohort"] for r in _rows(viz)}
    assert by_trial["NCT01844505"] == "Nivolumab" and by_trial["NCT02578680"] == "Pembrolizumab"


async def test_survival_cohort_without_values_says_so(make_ctx, study):
    ctx = make_ctx(Intent.SURVIVAL_SUMMARY, trials=[study("NCT07654361")], scope=Scope.COHORT, query="survival")
    ctx.plan = ctx.plan.model_copy(update={"nct_ids": []})
    res = await R.survival_summary(ctx)
    assert res.visualization.type is VizType.KPI_CARDS
    assert NoteCode.NOT_IN_REGISTRY in _codes(res)


async def test_power_cohort_scatter(make_ctx, cohort):
    c = cohort("melanoma_ph2_3", label="Melanoma")
    ctx = make_ctx(Intent.ENROLLMENT_VS_POWER, cohorts=[c], query="are melanoma trials adequately powered")
    res = await R.enrollment_vs_power(ctx)
    viz = res.visualization
    assert viz.type is VizType.SCATTER_PLOT and viz.meta.approximation.is_approximation
    rows = _rows(viz)
    assert rows and all(r["mde_at_80"] > 0 for r in rows)
    enrol = [r["enrollment"] for r in rows]
    assert enrol == sorted(enrol, reverse=True)
    big, small = rows[0], rows[-1]
    if big["arms"] == small["arms"]:
        assert big["mde_at_80"] <= small["mde_at_80"]  # more participants -> smaller detectable effect
    assert_citations_verified(viz, {s.nct_id: s.raw for s in c.studies})
    assert NoteCode.APPROXIMATED in _codes(res)


async def test_comparison_cohort_without_rows_is_named(make_ctx, study):
    a = CohortData(label="Pembrolizumab", spec=None, studies=[study("NCT02578680")], total_count=1)
    b = CohortData(label="Nivolumab", spec=None, studies=[study("NCT01844505")], total_count=1)
    ctx = make_ctx(Intent.SURVIVAL_SUMMARY, cohorts=[a, b], query="compare hazard ratios pembro vs nivo")
    ctx.trials = [study("NCT02578680")]  # e.g. the result-record fetch filled its slots from the first cohort
    res = await R.survival_summary(ctx)
    assert {r["cohort"] for r in _rows(res.visualization)} == {"Pembrolizumab"}
    assert any(n.code is NoteCode.NOT_IN_REGISTRY and "Nivolumab is missing" in n.message for n in res.notes)


async def test_forest_labels_use_group_titles(make_ctx, study):
    ctx = make_ctx(Intent.SURVIVAL_SUMMARY, trials=[study(n) for n in SURV], scope=Scope.COHORT, query="hazard ratio")
    ctx.plan = ctx.plan.model_copy(update={"nct_ids": []})
    res = await R.survival_summary(ctx)
    labels = [r["label"] for r in _rows(res.visualization)]
    assert len(labels) == len(set(labels))
    assert any("SOC Chemotherapy" in lab for lab in labels)  # KEYNOTE-024 arms, not the analysis description
