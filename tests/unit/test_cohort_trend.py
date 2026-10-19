"""trend_over_time."""

from __future__ import annotations

from ctviz.api.schemas import Intent, VizType
from ctviz.compute.cohort import parse_partial_date
from ctviz.core.provenance import Study
from ctviz.viz.base import CohortData
from tests.unit.cohort_helpers import assert_nonzero_cites, assert_valid, data_numbers, rows, run


async def test_trend_per_year(make_ctx, cohort):
    c = cohort("pembrolizumab_since2015", "Pembrolizumab")
    res = await run(Intent.TREND_OVER_TIME, make_ctx(Intent.TREND_OVER_TIME, cohorts=[c]))
    viz = res.visualization
    assert viz.type is VizType.TIME_SERIES and viz.meta.time_granularity == "year"
    assert viz.encoding.x.field == "period" and viz.encoding.x.type == "temporal"
    rs = rows(viz)
    assert rs[0]["period"].endswith("-01-01") and isinstance(rs[0]["year"], int)
    dated = sum(1 for s in c.studies if parse_partial_date(s.get("start_date")))
    assert sum(r["trial_count"] for r in rs) == dated
    years = [r["year"] for r in rs]
    assert years == list(range(years[0], years[-1] + 1))  # zero-filled, contiguous
    assert_valid(viz, c)
    assert_nonzero_cites(viz)
    for d in viz.data:  # time buckets cite the start date, whose year matches the bucket
        for cit in d.citations:
            assert cit.field_path.endswith("startDateStruct.date")
            assert cit.excerpt[:4] == str((d.model_extra or {})["year"])
    assert res.facts["peak_count"] in data_numbers(viz)
    assert res.facts["trials_plotted_derived"] == dated


async def test_trend_comparison_series(make_ctx, cohort):
    a, b = cohort("semaglutide", "Semaglutide"), cohort("tirzepatide", "Tirzepatide")
    res = await run(Intent.TREND_OVER_TIME, make_ctx(Intent.TREND_OVER_TIME, cohorts=[a, b]))
    viz = res.visualization
    assert viz.encoding.series.field == "cohort"
    rs = rows(viz)
    for c in (a, b):
        assert sum(r["trial_count"] for r in rs if r["cohort"] == c.label) == len(c.studies)
        assert res.facts["series_totals_derived"][c.label] == len(c.studies)
    # both series cover the same periods
    periods = {lbl: [r["period"] for r in rs if r["cohort"] == lbl] for lbl in ("Semaglutide", "Tirzepatide")}
    assert periods["Semaglutide"] == periods["Tirzepatide"]
    assert_valid(viz, a, b)


async def test_trend_series_by_phase_cites_phase(make_ctx, cohort):
    c = cohort("obesity_since2020")
    res = await run(Intent.TREND_OVER_TIME, make_ctx(Intent.TREND_OVER_TIME, cohorts=[c], series_by="phase"))
    viz = res.visualization
    assert viz.encoding.series.field == "phase"
    for d in viz.data:
        paths = {cit.field_path.rsplit(".", 1)[-1] for cit in d.citations}
        if d.citations:
            assert "date" in paths and paths & {"phases", "nctId"}
    assert_valid(viz, c)


async def test_trend_quarter_excludes_year_only_dates(make_ctx, cohort):
    c = cohort("semaglutide")
    raw = dict(c.studies[0].raw)
    raw["protocolSection"] = {
        **raw["protocolSection"],
        "statusModule": {**raw["protocolSection"]["statusModule"], "startDateStruct": {"date": "2019"}},
    }
    nodate = {"protocolSection": {"identificationModule": {"nctId": "NCT99999999"}}}
    studies = [Study(raw), Study(nodate)] + c.studies[1:50]
    cd = CohortData(label="x", spec=c.spec, studies=studies, total_count=len(studies))
    res = await run(Intent.TREND_OVER_TIME, make_ctx(Intent.TREND_OVER_TIME, cohorts=[cd], time_granularity="quarter"))
    viz = res.visualization
    assert viz.meta.time_granularity == "quarter"
    assert all("-Q" in r["label"] for r in rows(viz))
    assert res.facts["trials_without_start_date_derived"] == 1 and res.facts["trials_with_imprecise_start_derived"] == 1
    assert sum(r["trial_count"] for r in rows(viz)) == len(studies) - 2
    assert [n.code.value for n in res.notes].count("PARTIAL_DATES") >= 2
    assert_valid(viz, cd)
