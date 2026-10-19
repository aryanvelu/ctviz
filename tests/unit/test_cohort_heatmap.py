"""phase_status_heatmap and landscape_gap_heatmap."""

from __future__ import annotations

from ctviz.api.schemas import Intent, VizType
from tests.unit.cohort_helpers import assert_valid, rows, run


async def test_phase_status_heatmap(make_ctx, cohort):
    c = cohort("obesity_since2020")
    res = await run(Intent.PHASE_STATUS_HEATMAP, make_ctx(Intent.PHASE_STATUS_HEATMAP, cohorts=[c]))
    viz = res.visualization
    assert viz.type is VizType.HEATMAP
    enc = viz.encoding
    assert (enc.x.field, enc.y.field, enc.series.field) == ("phase", "status", "trial_count")
    assert enc.x.type == "ordinal" and enc.series.type == "quantitative"
    assert viz.meta.highlight_zero is False  # empty phase × status cells are often structural, not gaps
    rs = rows(viz)
    assert len(rs) == res.facts["rows_derived"] * res.facts["columns_derived"]  # complete matrix
    assert sum(r["trial_count"] for r in rs) == len(c.studies)
    for d in viz.data:
        r = d.model_extra
        assert r["is_gap"] == (r["trial_count"] == 0)
        if r["is_gap"]:
            assert d.citations == []
        else:  # each trial cites its phases (or nctId when absent) and its overallStatus
            tails = {cit.field_path.rsplit(".", 1)[-1] for cit in d.citations}
            assert "overallStatus" in tails and tails & {"phases", "nctId"}
    assert_valid(viz, c)
    assert res.facts["max_cell_count"] == max(r["trial_count"] for r in rs)


async def test_landscape_gap_heatmap_conditions_by_phase(make_ctx, cohort):
    c = cohort("semaglutide", "Semaglutide")
    ctx = make_ctx(Intent.LANDSCAPE_GAP_HEATMAP, cohorts=[c], group_by="condition", top_n=8)
    res = await run(Intent.LANDSCAPE_GAP_HEATMAP, ctx)
    viz = res.visualization
    assert viz.encoding.x.field == "phase" and viz.encoding.y.field == "condition"
    rs = rows(viz)
    phases = viz.encoding.x.sort.values
    assert phases[:5] == ["Early Phase 1", "Phase 1", "Phase 2", "Phase 3", "Phase 4"]
    assert len(rs) == 8 * len(phases)
    gaps = [d for d in viz.data if d.model_extra["is_gap"]]
    assert gaps and all(d.citations == [] and d.model_extra["trial_count"] == 0 for d in gaps)
    assert res.facts["gap_cells_derived"] == len(gaps)
    assert viz.meta.highlight_zero is True  # the gap heatmap is the one view that asks for gap styling
    for d in viz.data:
        if d.citations:
            tails = {cit.field_path.split(".")[-1] for cit in d.citations}
            assert any(t.startswith("conditions[") for t in tails)
    assert_valid(viz, c)


async def test_landscape_gap_default_rows_and_year_columns(make_ctx, cohort):
    c = cohort("melanoma_ph2_3")
    ctx = make_ctx(Intent.LANDSCAPE_GAP_HEATMAP, cohorts=[c], series_by="year", top_n=5)
    res = await run(Intent.LANDSCAPE_GAP_HEATMAP, ctx)
    viz = res.visualization
    assert res.facts["row_dimension"] == "intervention" and viz.encoding.x.field == "year"
    years = [int(y) for y in viz.encoding.x.sort.values]
    assert years == list(range(years[0], years[-1] + 1))
    assert_valid(viz, c)
