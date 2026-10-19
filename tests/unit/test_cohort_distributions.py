"""status_distribution, sponsor_landscape, intervention_landscape, design_methodology."""

from __future__ import annotations

from ctviz.api.schemas import Intent, VizType
from ctviz.compute.cohort import normalize_name
from tests.unit.cohort_helpers import assert_nonzero_cites, assert_valid, data_numbers, rows, run


async def test_status_distribution_counts_add_up(make_ctx, cohort):
    c = cohort("semaglutide", "Semaglutide")
    res = await run(Intent.STATUS_DISTRIBUTION, make_ctx(Intent.STATUS_DISTRIBUTION, cohorts=[c]))
    viz = res.visualization
    assert viz.type is VizType.BAR_CHART and viz.encoding.x.field == "status" and viz.encoding.series is None
    assert sum(r["trial_count"] for r in rows(viz)) == len(c.studies)
    assert_valid(viz, c)
    assert_nonzero_cites(viz)
    for d in viz.data:  # each status bucket cites overallStatus
        assert all(cit.field_path.endswith("overallStatus") for cit in d.citations)
    assert res.facts["top_category_count"] == max(r["trial_count"] for r in rows(viz))
    assert res.facts["top_category_count"] in data_numbers(viz)


async def test_status_distribution_comparison(make_ctx, cohort):
    a, b = cohort("semaglutide", "Semaglutide"), cohort("tirzepatide", "Tirzepatide")
    res = await run(Intent.STATUS_DISTRIBUTION, make_ctx(Intent.STATUS_DISTRIBUTION, cohorts=[a, b]))
    viz = res.visualization
    assert viz.encoding.series.field == "cohort"
    for c in (a, b):
        assert sum(r["trial_count"] for r in rows(viz) if r["cohort"] == c.label) == len(c.studies)
        assert sum(res.facts["by_cohort"][c.label].values()) == len(c.studies)
    assert_valid(viz, a, b)


async def test_status_pie_when_preferred(make_ctx, cohort):
    c = cohort("t2d_recruiting")
    ctx = make_ctx(Intent.STATUS_DISTRIBUTION, cohorts=[c], preferred_visualization=VizType.PIE_CHART)
    viz = (await run(Intent.STATUS_DISTRIBUTION, ctx)).visualization
    assert viz.type is VizType.PIE_CHART and viz.encoding.label.field == "status"
    assert_valid(viz, c)


async def test_sponsor_landscape_top_n_and_other(make_ctx, cohort):
    c = cohort("obesity_since2020")
    res = await run(Intent.SPONSOR_LANDSCAPE, make_ctx(Intent.SPONSOR_LANDSCAPE, cohorts=[c], top_n=10))
    viz = res.visualization
    rs = rows(viz)
    assert len(rs) == 11 and rs[-1]["is_other"] is True
    # Lead sponsor is single-valued, so named buckets + Other cover every trial with a sponsor.
    assert sum(r["trial_count"] for r in rs) == len(c.studies)
    assert viz.meta.truncation.top_n == 10 and viz.meta.layout.orientation == "horizontal"
    assert any(n.code.value == "TRUNCATED" for n in res.notes)
    assert_valid(viz, c)
    for d in viz.data:
        assert all(cit.field_path.endswith("leadSponsor.name") for cit in d.citations)
        if not (d.model_extra or {}).get("is_other"):
            keys = {normalize_name(cit.excerpt) for cit in d.citations}
            assert len(keys) == 1  # one normalized sponsor per bucket


async def test_sponsor_class_comparison(make_ctx, cohort):
    a, b = cohort("semaglutide", "Semaglutide"), cohort("tirzepatide", "Tirzepatide")
    ctx = make_ctx(Intent.SPONSOR_LANDSCAPE, cohorts=[a, b], group_by="sponsor_class")
    res = await run(Intent.SPONSOR_LANDSCAPE, ctx)
    viz = res.visualization
    assert viz.encoding.x.field == "sponsor_class" and viz.encoding.series.field == "cohort"
    for c in (a, b):
        assert sum(r["trial_count"] for r in rows(viz) if r["cohort"] == c.label) == len(c.studies)
    assert_valid(viz, a, b)


async def test_sponsor_stacked_by_phase(make_ctx, cohort):
    c = cohort("pembrolizumab_since2015")
    ctx = make_ctx(Intent.SPONSOR_LANDSCAPE, cohorts=[c], series_by="phase", top_n=5)
    res = await run(Intent.SPONSOR_LANDSCAPE, ctx)
    viz = res.visualization
    assert viz.meta.layout.mode == "stacked" and viz.encoding.series.field == "phase"
    for sponsor, total in res.facts["sponsor_totals_derived"].items():
        assert sum(r["trial_count"] for r in rows(viz) if r["sponsor"] == sponsor) == total
    assert_valid(viz, c)


async def test_intervention_landscape_names(make_ctx, cohort):
    c = cohort("melanoma_ph2_3")
    res = await run(Intent.INTERVENTION_LANDSCAPE, make_ctx(Intent.INTERVENTION_LANDSCAPE, cohorts=[c]))
    viz = res.visualization
    rs = rows(viz)
    assert viz.encoding.x.field == "intervention"
    assert not any("placebo" in r["intervention"].lower() for r in rs)
    assert all(r["trial_count"] <= len(c.studies) for r in rs)
    named = [d for d in viz.data if not (d.model_extra or {}).get("is_other")]
    for d in named:  # each trial cited once, via interventions[i].name
        assert all(".interventions[" in cit.field_path and cit.field_path.endswith(".name") for cit in d.citations)
        assert (d.citations_total or len(d.citations)) == (d.model_extra or {})["trial_count"]
    assert_valid(viz, c)


async def test_intervention_types_comparison(make_ctx, cohort):
    a, b = cohort("semaglutide", "Semaglutide"), cohort("tirzepatide", "Tirzepatide")
    ctx = make_ctx(Intent.INTERVENTION_LANDSCAPE, cohorts=[a, b], group_by="intervention_type")
    res = await run(Intent.INTERVENTION_LANDSCAPE, ctx)
    viz = res.visualization
    assert viz.encoding.x.field == "intervention_type" and viz.encoding.series.field == "cohort"
    drug = [r for r in rows(viz) if r["intervention_type"] == "Drug"]
    assert {r["cohort"] for r in drug} == {"Semaglutide", "Tirzepatide"}
    assert_valid(viz, a, b)


async def test_design_mix_cohort(make_ctx, cohort):
    c = cohort("obesity_since2020")
    res = await run(Intent.DESIGN_METHODOLOGY, make_ctx(Intent.DESIGN_METHODOLOGY, cohorts=[c]))
    viz = res.visualization
    assert viz.type is VizType.BAR_CHART and viz.encoding.series.field == "dimension"
    for dim in ("Allocation", "Masking", "Intervention model"):
        assert sum(r["trial_count"] for r in rows(viz) if r["dimension"] == dim) == len(c.studies)
    assert sum(res.facts["masking"].values()) == len(c.studies)
    assert_valid(viz, c)


async def test_design_cards_single_trial(make_ctx, study):
    s = study("NCT03548935")
    res = await run(Intent.DESIGN_METHODOLOGY, make_ctx(Intent.DESIGN_METHODOLOGY, trials=[s]))
    viz = res.visualization
    assert viz.type is VizType.KPI_CARDS
    labels = {r["label"] for r in rows(viz)}
    assert {"Allocation", "Masking", "Enrollment"} <= labels
    from tests.conftest import assert_citations_verified

    assert assert_citations_verified(viz, {s.nct_id: s.raw}) >= len(viz.data)
    assert res.facts["Allocation"] == "RANDOMIZED"


async def test_handler_records_trace_step(make_ctx, cohort):
    from ctviz.obs.trace import trace_scope

    c = cohort("semaglutide")
    with trace_scope() as tr:
        await run(Intent.STATUS_DISTRIBUTION, make_ctx(Intent.STATUS_DISTRIBUTION, cohorts=[c]))
    step = next(s for s in tr.steps if s.name == "handler.status_distribution")
    assert step.status == "ok" and step.detail["rows"] > 0 and step.detail["viz"] == "bar_chart"
