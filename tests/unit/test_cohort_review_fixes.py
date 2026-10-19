"""Regression tests for the code-review findings on the cohort handlers (one test per item)."""

from __future__ import annotations

import copy
import json

from ctviz.api.schemas import Intent, VizType
from ctviz.compute.network import CooccurrenceGraph
from ctviz.core.provenance import Study
from ctviz.viz.base import CohortData
from tests.unit.cohort_helpers import assert_valid, rows, run


def _with(study: Study, **status_or_design) -> Study:
    """Copy of a study with protocolSection sub-module fields overridden: key 'module.path' -> value."""
    raw = copy.deepcopy(study.raw)
    for dotted, value in status_or_design.items():
        cur = raw["protocolSection"]
        *parents, leaf = dotted.split("__")
        for p in parents:
            cur = cur.setdefault(p, {})
        if value is None:
            cur.pop(leaf, None)
        else:
            cur[leaf] = value
    return Study(raw)


def _cohort(label: str, studies: list[Study]) -> CohortData:
    from ctviz.ctgov.requests import SearchSpec

    return CohortData(label=label, spec=SearchSpec(), studies=studies, total_count=len(studies))


# 1 -----------------------------------------------------------------------------------------------
async def test_duplicate_studies_counted_once(make_ctx, cohort):
    c = cohort("semaglutide")
    nodate = _with(c.studies[0], statusModule__startDateStruct=None)
    dup = _cohort("dup", [nodate, nodate, *c.studies[1:30], *c.studies[1:30]])
    res = await run(Intent.TREND_OVER_TIME, make_ctx(Intent.TREND_OVER_TIME, cohorts=[dup]))
    assert res.facts["trials_without_start_date_derived"] == 1
    assert res.facts["trials_analyzed_derived"] == 30 == res.studies_analyzed
    sc = await run(
        Intent.ENROLLMENT_DISTRIBUTION,
        make_ctx(Intent.ENROLLMENT_DISTRIBUTION, cohorts=[dup], preferred_visualization=VizType.SCATTER_PLOT),
    )
    ids = [r["nct_id"] for r in rows(sc.visualization)]
    assert len(ids) == len(set(ids))
    st = await run(Intent.ENROLLMENT_DISTRIBUTION, make_ctx(Intent.ENROLLMENT_DISTRIBUTION, cohorts=[dup]))
    assert st.facts["trials_with_enrollment_derived"] == 30


# 2 -----------------------------------------------------------------------------------------------
async def test_trend_peaks_per_series_and_distinct_plotted(make_ctx, cohort):
    a = cohort("semaglutide", "A")
    b = cohort("semaglutide", "B")  # same trials in both cohorts
    res = await run(Intent.TREND_OVER_TIME, make_ctx(Intent.TREND_OVER_TIME, cohorts=[a, b]))
    f = res.facts
    assert "peak_count" not in f and "trials_plotted_derived" not in f
    assert set(f["by_series_peak"]) == {"A", "B"}
    for label in ("A", "B"):
        series = [r for r in rows(res.visualization) if r["cohort"] == label]
        peak = max(series, key=lambda r: (r["trial_count"], r["label"]))
        assert f["by_series_peak"][label] == {"period": peak["label"], "count": peak["trial_count"]}
    assert f["trials_plotted_distinct_derived"] == len(a.studies)  # not 2x


# 3 -----------------------------------------------------------------------------------------------
async def test_site_count_backed_by_evidence_and_blank_countries_ignored(make_ctx, cohort):
    c = cohort("t2d_recruiting")
    s = c.studies[0]
    locs = copy.deepcopy(s.get("locations"))
    country = locs[0]["country"]
    locs += [{"facility": "X", "country": country}, {"facility": "Y", "country": "   "}]
    s2 = _with(s, contactsLocationsModule__locations=locs)
    cd = _cohort("x", [s2])
    res = await run(Intent.GEOGRAPHIC_DISTRIBUTION, make_ctx(Intent.GEOGRAPHIC_DISTRIBUTION, cohorts=[cd]))
    viz = res.visualization
    assert "   " not in {r["country"] for r in rows(viz)}
    for d in viz.data:  # counts are trials (one citation each); sites are a separate, larger-or-equal figure
        assert len(res.full_refs[d.id]) == d.model_extra["trial_count"] == len(d.nct_ids)
        assert d.model_extra["site_count"] >= d.model_extra["trial_count"]
    row = next(r for r in rows(viz) if r["country"] == country)
    assert row["site_count"] == sum(1 for loc in locs if loc.get("country") == country)
    assert_valid(viz, cd)


# 4 -----------------------------------------------------------------------------------------------
async def test_histogram_linear_scale_and_scatter_excludes_zero(make_ctx, cohort):
    c = cohort("semaglutide")
    zero = _with(c.studies[0], designModule__enrollmentInfo={"count": 0, "type": "ACTUAL"})
    cd = _cohort("x", [zero, *c.studies[1:20]])
    h = await run(Intent.ENROLLMENT_DISTRIBUTION, make_ctx(Intent.ENROLLMENT_DISTRIBUTION, cohorts=[cd]))
    assert h.visualization.encoding.x.scale.type == "linear"
    assert rows(h.visualization)[0]["bin_start"] == 0
    ctx = make_ctx(Intent.ENROLLMENT_DISTRIBUTION, cohorts=[cd], preferred_visualization=VizType.SCATTER_PLOT)
    sc = await run(Intent.ENROLLMENT_DISTRIBUTION, ctx)
    assert zero.nct_id not in {r["nct_id"] for r in rows(sc.visualization)}
    assert sc.facts["trials_zero_enrollment_derived"] == sum(1 for s in cd.studies if s.get("enrollment_count") == 0)
    assert any("enrollment of 0" in n.message for n in sc.notes)


# 5 -----------------------------------------------------------------------------------------------
async def test_enrollment_stats_split_actual_vs_estimated(make_ctx, cohort):
    c = cohort("obesity_since2020")
    res = await run(Intent.ENROLLMENT_DISTRIBUTION, make_ctx(Intent.ENROLLMENT_DISTRIBUTION, cohorts=[c]))
    st = res.facts["stats_derived"]
    actual = [
        s.get("enrollment_count")
        for s in c.studies
        if s.get("enrollment_type") != "ESTIMATED" and isinstance(s.get("enrollment_count"), int)
    ]
    est = [
        s.get("enrollment_count")
        for s in c.studies
        if s.get("enrollment_type") == "ESTIMATED" and isinstance(s.get("enrollment_count"), int)
    ]
    assert st["actual"]["n"] == len(actual) and st["estimated"]["n"] == len(est)
    assert st["actual"]["total_participants_derived"] == sum(actual)
    assert "total_participants" not in st["estimated"] and "total_participants" not in st["actual"]


# 6 -----------------------------------------------------------------------------------------------
def test_prune_keeps_pinned_nodes_and_edges():
    g = CooccurrenceGraph(directed=True)
    for i in range(60):
        for j in range(60):
            for k in range(1 + (i * j) % 4):
                t = f"T{i}-{j}-{k}"
                g.add_node(f"s{i}", f"S{i}", "sponsor", t)
                g.add_node(f"d{j}", f"D{j}", "drug", t)
                g.add_edge(f"s{i}", f"d{j}", "tests", t)
    g.add_node("rare", "Rare", "drug", "TR")
    g.add_node("s1", "S1", "sponsor", "TR")
    g.add_edge("s1", "rare", "tests", "TR")  # weight 1, would be cut by the raised threshold
    pruned = g.prune(max_nodes=40, max_edges=300, pinned={"rare"})
    keys = {n.key for n in pruned.nodes}
    assert "rare" in keys and pruned.min_edge_weight > 1
    assert any(e.target == "rare" and e.weight == 1 for e in pruned.edges)
    assert len(pruned.nodes) <= 40 and len(pruned.edges) <= 300


async def test_network_pins_plan_entities(make_ctx, cohort):
    c = cohort("obesity_since2020")
    ctx = make_ctx(Intent.ENTITY_NETWORK, cohorts=[c], top_n=10)
    ctx.plan.cohorts[0].drugs = ["Retatrutide"]
    res = await run(Intent.ENTITY_NETWORK, ctx)
    viz = res.visualization
    pinned = [n for n in viz.data.nodes if n.model_extra.get("pinned")]
    assert [n.model_extra["label"].lower() for n in pinned] == ["retatrutide"]
    assert viz.meta.pinned == [pinned[0].id]
    assert any(pinned[0].id in (e.model_extra["source"], e.model_extra["target"]) for e in viz.data.edges)
    assert_valid(viz, c)


# 7 -----------------------------------------------------------------------------------------------
async def test_partial_dates_kept_raw_with_precision(make_ctx, cohort):
    c = cohort("semaglutide")
    early = _with(c.studies[0], statusModule__startDateStruct={"date": "2001-05"}, designModule__phases=["PHASE1"])
    cd = _cohort("x", [early, *c.studies[1:40]])
    res = await run(Intent.DEVELOPMENT_PROGRESSION, make_ctx(Intent.DEVELOPMENT_PROGRESSION, cohorts=[cd]))
    r = next(r for r in rows(res.visualization) if r["task"] == "Phase 1")
    assert (r["start"], r["start_raw"], r["start_precision"]) == ("2001-05-01", "2001-05", "month")
    assert res.facts["phases"]["Phase 1"]["first_start_raw"] == "2001-05"
    assert any(n.code.value == "PARTIAL_DATES" and "month- or" in n.message for n in res.notes)
    sa = await run(Intent.SPONSOR_ACTIVITY, make_ctx(Intent.SPONSOR_ACTIVITY, cohorts=[cd]))
    assert all({"start_raw", "start_precision", "end_raw", "end_precision"} <= set(r) for r in rows(sa.visualization))
    sc = await run(
        Intent.ENROLLMENT_DISTRIBUTION,
        make_ctx(Intent.ENROLLMENT_DISTRIBUTION, cohorts=[cd], preferred_visualization=VizType.SCATTER_PLOT),
    )
    pt = next(r for r in rows(sc.visualization) if r["nct_id"] == early.nct_id)
    assert pt["start_raw"] == "2001-05" and pt["start_precision"] == "month"
    assert any(n.code.value == "PARTIAL_DATES" for n in sc.notes)


# 8 -----------------------------------------------------------------------------------------------
async def test_gap_heatmap_year_columns(make_ctx, cohort):
    c = cohort("semaglutide")
    s0 = c.studies[0]
    absurd = _with(c.studies[1], statusModule__startDateStruct={"date": "1900-01-01"})
    nodate = _with(c.studies[2], statusModule__startDateStruct=None)
    cd = _cohort("x", [s0, absurd, nodate, *c.studies[3:60]])
    ctx = make_ctx(Intent.LANDSCAPE_GAP_HEATMAP, cohorts=[cd], group_by="sponsor", series_by="year", top_n=3)
    res = await run(Intent.LANDSCAPE_GAP_HEATMAP, ctx)
    viz = res.visualization
    years = [int(y) for y in viz.encoding.x.sort.values]
    assert min(years) >= 1980
    msgs = " ".join(n.message for n in res.notes)
    assert "no start date" in msgs
    shown = set(viz.encoding.y.sort.values)
    used = {int(r["year"]) for r in rows(viz) if r["trial_count"] > 0}
    assert years[0] == min(used) and years[-1] == max(used)
    assert all(r["sponsor"] in shown for r in rows(viz))


# 9 -----------------------------------------------------------------------------------------------
async def test_site_investigator_caps_sites_per_trial(make_ctx, cohort):
    c = cohort("melanoma_ph2_3")
    s = next(s for s in c.studies if s.get("overall_officials"))
    locs = [{"facility": f"Site {i}", "city": "City", "country": "US"} for i in range(70)]
    big = _with(s, contactsLocationsModule__locations=locs)
    cd = _cohort("x", [big])
    ctx = make_ctx(Intent.ENTITY_NETWORK, cohorts=[cd], network_kind="site_investigator")
    res = await run(Intent.ENTITY_NETWORK, ctx)
    assert res.facts["nodes_total_derived"] <= 50 + len(s.get("overall_officials"))
    assert any("more than 50 sites" in n.message for n in res.notes)
    assert_valid(res.visualization, cd)


# 10 ----------------------------------------------------------------------------------------------
async def test_open_ended_bin_and_comparison_country_facts(make_ctx, cohort):
    c = cohort("semaglutide")
    huge = _with(c.studies[0], designModule__enrollmentInfo={"count": 20000, "type": "ACTUAL"})
    cd = _cohort("x", [huge, *c.studies[1:20]])
    viz = (
        await run(Intent.ENROLLMENT_DISTRIBUTION, make_ctx(Intent.ENROLLMENT_DISTRIBUTION, cohorts=[cd]))
    ).visualization
    last = viz.data[-1]
    assert last.model_extra["is_open_ended"] is True and last.model_extra["bin_end"] is None
    wire = json.loads(viz.model_dump_json(exclude_none=True))
    assert "bin_end" not in wire["data"][-1] and "bin_end" in wire["data"][0]

    a, b = cohort("semaglutide", "Semaglutide"), cohort("tirzepatide", "Tirzepatide")
    res = await run(Intent.GEOGRAPHIC_DISTRIBUTION, make_ctx(Intent.GEOGRAPHIC_DISTRIBUTION, cohorts=[a, b]))
    assert "top_country_trials" not in res.facts
    for label in ("Semaglutide", "Tirzepatide"):
        top = res.facts["by_cohort_top"][label]
        best = max((r for r in rows(res.visualization) if r["cohort"] == label), key=lambda r: r["trial_count"])
        assert top["trials"] == best["trial_count"]


# follow-up: comparison-scope "trials_without_*" counts are distinct trials ---------------------------
async def test_comparison_missing_counts_are_distinct(make_ctx, cohort):
    c = cohort("semaglutide")
    bare = _with(
        c.studies[0],
        statusModule__startDateStruct=None,
        designModule__enrollmentInfo=None,
        contactsLocationsModule__locations=None,
        armsInterventionsModule__interventions=None,
    )
    a, b = _cohort("A", [bare, *c.studies[1:20]]), _cohort("B", [bare, *c.studies[20:40]])
    trend = await run(Intent.TREND_OVER_TIME, make_ctx(Intent.TREND_OVER_TIME, cohorts=[a, b]))
    assert trend.facts["trials_without_start_date_derived"] == 1
    geo = await run(Intent.GEOGRAPHIC_DISTRIBUTION, make_ctx(Intent.GEOGRAPHIC_DISTRIBUTION, cohorts=[a, b]))
    missing_geo = {s.nct_id for s in [*a.studies, *b.studies] if not s.get("locations")}
    assert geo.facts["trials_without_locations_derived"] == len(missing_geo)
    enr = await run(Intent.ENROLLMENT_DISTRIBUTION, make_ctx(Intent.ENROLLMENT_DISTRIBUTION, cohorts=[a, b]))
    assert enr.facts["trials_without_enrollment_derived"] == 1
    iv = await run(Intent.INTERVENTION_LANDSCAPE, make_ctx(Intent.INTERVENTION_LANDSCAPE, cohorts=[a, b]))
    no_iv = {
        s.nct_id
        for s in [*a.studies, *b.studies]
        if not any(i.get("name") and "placebo" not in i["name"].lower() for i in s.get("interventions") or [])
    }
    assert iv.facts.get("trials_without_value_derived", 0) == len(no_iv)


# follow-up: country grouping is whitespace/case-insensitive ---------------------------------------
def test_country_key():
    from ctviz.compute.cohort import country_key

    assert country_key("United States ") == country_key("united  states") == "united states"
    assert country_key("   ") == "" and country_key(None) == ""
