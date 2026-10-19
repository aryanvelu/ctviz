"""geographic_distribution and enrollment_distribution."""

from __future__ import annotations

import re

from ctviz.api.schemas import Intent, VizType
from tests.unit.cohort_helpers import assert_nonzero_cites, assert_valid, rows, run

_LOC = re.compile(r"\.locations\[(\d+)\]\.country$")


async def test_geographic_distribution(make_ctx, cohort):
    c = cohort("t2d_recruiting")
    res = await run(Intent.GEOGRAPHIC_DISTRIBUTION, make_ctx(Intent.GEOGRAPHIC_DISTRIBUTION, cohorts=[c]))
    viz = res.visualization
    assert viz.type is VizType.GEO_MAP
    assert viz.meta.geo.level == "country"
    assert viz.meta.geo.value_field == "trial_count"
    by_id = {s.nct_id: s for s in c.studies}
    expected: dict[str, int] = {}
    for s in c.studies:
        for country in {loc.get("country") for _, loc, _ in s.items("locations") if loc.get("country")}:
            expected[country] = expected.get(country, 0) + 1
    rs = rows(viz)
    assert {r["country"]: r["trial_count"] for r in rs} == expected
    assert all(r["site_count"] >= r["trial_count"] for r in rs)
    for d in viz.data:
        country = d.model_extra["country"]
        refs = res.full_refs[d.id]
        # one ref per trial, each the first matching locations[i] index
        first = refs
        assert len(refs) == d.model_extra["trial_count"] == len(d.nct_ids)
        assert len({r.nct_id for r in first}) == len(first)
        for r in first:
            assert r.excerpt == country
            idx = int(_LOC.search(r.field_path).group(1))
            locs = by_id[r.nct_id].get("locations")
            assert idx == next(i for i, loc in enumerate(locs) if loc.get("country") == country)
        assert d.model_extra["site_count"] >= len(refs)
        assert all(r.excerpt == country and _LOC.search(r.field_path) for r in refs)
    assert_valid(viz, c)
    assert res.facts["top_country_trials"] == max(r["trial_count"] for r in rs)


async def test_geographic_comparison(make_ctx, cohort):
    a, b = cohort("semaglutide", "Semaglutide"), cohort("tirzepatide", "Tirzepatide")
    res = await run(Intent.GEOGRAPHIC_DISTRIBUTION, make_ctx(Intent.GEOGRAPHIC_DISTRIBUTION, cohorts=[a, b]))
    viz = res.visualization
    assert viz.encoding.series.field == "cohort"
    assert {r["cohort"] for r in rows(viz)} == {"Semaglutide", "Tirzepatide"}
    assert_valid(viz, a, b)


async def test_enrollment_histogram(make_ctx, cohort):
    c = cohort("obesity_since2020")
    res = await run(Intent.ENROLLMENT_DISTRIBUTION, make_ctx(Intent.ENROLLMENT_DISTRIBUTION, cohorts=[c]))
    viz = res.visualization
    assert viz.type is VizType.HISTOGRAM
    assert (viz.encoding.x.field, viz.encoding.x2.field, viz.encoding.y.field) == (
        "bin_start",
        "bin_end",
        "trial_count",
    )
    rs = rows(viz)
    with_n = [s for s in c.studies if isinstance(s.get("enrollment_count"), int)]
    assert sum(r["trial_count"] for r in rs) == len(with_n) == res.facts["trials_with_enrollment_derived"]
    for d in viz.data:
        r = d.model_extra
        for cit in d.citations:
            assert cit.field_path.endswith("enrollmentInfo.count")
            assert r["bin_start"] <= int(cit.excerpt) and (r["bin_end"] is None or int(cit.excerpt) < r["bin_end"])
    assert_nonzero_cites(viz)
    st = viz.meta.summary
    assert st["actual"]["n"] + st["estimated"]["n"] == len(with_n)
    assert_valid(viz, c)


async def test_enrollment_comparison_and_scatter(make_ctx, cohort):
    a, b = cohort("semaglutide", "Semaglutide"), cohort("tirzepatide", "Tirzepatide")
    res = await run(Intent.ENROLLMENT_DISTRIBUTION, make_ctx(Intent.ENROLLMENT_DISTRIBUTION, cohorts=[a, b]))
    viz = res.visualization
    assert viz.encoding.series.field == "cohort"
    for c in (a, b):
        st = res.facts["stats_derived"][c.label]
        assert sum(r["trial_count"] for r in rows(viz) if r["cohort"] == c.label) == (
            st["actual"]["n"] + st["estimated"]["n"]
        )
    assert_valid(viz, a, b)

    ctx = make_ctx(Intent.ENROLLMENT_DISTRIBUTION, cohorts=[a], preferred_visualization=VizType.SCATTER_PLOT)
    res = await run(Intent.ENROLLMENT_DISTRIBUTION, ctx)
    sc = res.visualization
    assert sc.type is VizType.SCATTER_PLOT and sc.encoding.y.field == "enrollment"
    assert len(sc.data) + res.facts["trials_zero_enrollment_derived"] == len(a.studies)
    assert all(r["enrollment"] > 0 for r in rows(sc))
    assert_valid(sc, a)
