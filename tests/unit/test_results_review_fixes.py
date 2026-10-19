"""Regression tests for code-review findings (one or more per item). Synthetic records are mutated
copies of recorded fixtures, so every citation is still checked against the record it came from."""

from __future__ import annotations

import copy

import pytest

from ctviz.api.schemas import Intent
from ctviz.api.schemas.response import NoteCode
from ctviz.api.schemas.viz import GraphData, VizType
from ctviz.compute import survival as sv
from ctviz.compute.results import analyses_for, effect_kind, parse_exact, parse_num, parse_qualified
from ctviz.core.provenance import Study
from ctviz.handlers import results as R
from tests.conftest import assert_citations_verified, load_study_raw

OM = "outcomeMeasures"


def mutated(nct: str, fn) -> tuple[Study, dict]:
    raw = copy.deepcopy(load_study_raw(nct))
    fn(raw)
    return Study(raw), raw


def oms(raw: dict) -> list[dict]:
    return raw["resultsSection"]["outcomeMeasuresModule"][OM]


def _om(title: str, ptype: str, unit: str, values: dict[str, str], *, time_frame: str = "", classes=None) -> dict:
    groups = [{"id": g, "title": f"Arm {g[-1]}"} for g in values]
    meas = [{"groupId": g, "value": v} for g, v in values.items()]
    return {
        "type": "PRIMARY",
        "title": title,
        "paramType": ptype,
        "unitOfMeasure": unit,
        "timeFrame": time_frame,
        "groups": groups,
        "classes": classes or [{"categories": [{"measurements": meas}]}],
    }


async def run(make_ctx, handler, intent, s: Study, raw: dict, query: str = "test"):
    ctx = make_ctx(intent, trials=[s], query=query)
    res = await handler(ctx)
    assert_citations_verified(res.visualization, {s.nct_id: raw})
    return res


# -- item 1: sankey with STARTED = 0 / missing intermediate nodes ------------------------------------------


def _zero_first_period(raw):
    per = raw["resultsSection"]["participantFlowModule"]["periods"][0]  # CheckMate pre-treatment period
    for m in per["milestones"]:
        for a in m["achievements"]:
            if a["groupId"] == "FG000":
                a["numSubjects"] = "0"


async def test_sankey_period1_started_zero_never_raises(make_ctx):
    s, raw = mutated("NCT01844505", _zero_first_period)
    res = await run(make_ctx, R.dropout_flow, Intent.DROPOUT_FLOW, s, raw)
    g: GraphData = res.visualization.data
    ids = {n.id for n in g.nodes}
    assert all(e.model_extra["source"] in ids and e.model_extra["target"] in ids for e in g.edges)
    nivo = [e.model_extra for e in g.edges if e.model_extra["group"] == "Nivolumab"]
    assert nivo and all(e["share_of_started"] <= 1 for e in nivo)  # divides by the first non-zero start
    assert any("0 started" in n.message for n in res.notes)
    out: dict[str, int] = {}
    for e in g.edges:
        out[e.model_extra["source"]] = out.get(e.model_extra["source"], 0) + e.model_extra["weight"]
    assert all(next(n for n in g.nodes if n.id == k).model_extra["value"] == v for k, v in out.items())


def _all_zero(raw):
    for per in raw["resultsSection"]["participantFlowModule"]["periods"]:
        for m in per["milestones"]:
            for a in m["achievements"]:
                a["numSubjects"] = "0"


async def test_sankey_all_zero_falls_back(make_ctx):
    s, raw = mutated("NCT03548935", _all_zero)
    res = await run(make_ctx, R.dropout_flow, Intent.DROPOUT_FLOW, s, raw)
    assert res.visualization.type is VizType.KPI_CARDS


def _mismatch_periods(raw):
    per = raw["resultsSection"]["participantFlowModule"]["periods"][1]  # treatment period STARTED != completed
    st = next(m for m in per["milestones"] if m["type"] == "STARTED")
    st["achievements"][0]["numSubjects"] = "300"


async def test_sankey_period_mismatch_gets_own_start_node(make_ctx):
    s, raw = mutated("NCT01844505", _mismatch_periods)
    res = await run(make_ctx, R.dropout_flow, Intent.DROPOUT_FLOW, s, raw)
    ids = {n.id for n in res.visualization.data.nodes}
    assert "FG000:p1:start" in ids and any(n.code is NoteCode.APPROXIMATED for n in res.notes)


# -- item 2: medians only from survival measures ------------------------------------------------------------


def _median_followup(raw):
    oms(raw).insert(
        0,
        _om(
            "Median Duration of Follow-up for Overall Survival", "MEDIAN", "Months", {"OG000": "20.1", "OG001": "19.8"}
        ),
    )


def test_median_follow_up_is_not_a_survival_anchor():
    s, _ = mutated("NCT02578680", _median_followup)
    assert not sv.is_survival_measure("Median Duration of Follow-up for Overall Survival")
    assert not sv.is_survival_measure("Duration of Treatment Exposure")
    ext = sv.extract_anchors(s, [0])
    assert ext.anchors == {} and any("non-survival" in x for x in ext.skipped)


async def test_followup_not_offered_to_selector(settings):
    from ctviz.agents.analysis_agent import build_catalog

    s, _ = mutated("NCT02578680", _median_followup)
    titles = [m["title"] for m in build_catalog(s, "survival")["measures"]]
    assert "Median Duration of Follow-up for Overall Survival" not in titles


# -- item 3: landmark rates need a survival/event-rate meaning and a known scale ---------------------------


def _count_landmark(raw):
    oms(raw).insert(
        0, _om("Overall Survival Rate at 12 Months", "NUMBER", "Participants", {"OG000": "120", "OG001": "80"})
    )


def test_landmark_participant_counts_dropped():
    s, _ = mutated("NCT02142738", _count_landmark)
    ext = sv.extract_anchors(s, [0])
    assert ext.anchors == {} and any("neither percent nor probability" in x for x in ext.skipped)


def _event_rate(raw):
    oms(raw).insert(
        0,
        _om(
            "Percentage of Participants Who Died at 12 Months",
            "NUMBER",
            "Percentage of participants",
            {"OG000": "30", "OG001": "45"},
        ),
    )


async def test_event_rate_converted_with_note(make_ctx):
    s, raw = mutated("NCT02142738", _event_rate)
    ext = sv.extract_anchors(s, [0])
    assert [round(a.survival, 4) for a in ext.anchors["Percentage of Participants Who Died at 12 Months · Arm 0"]] == [
        0.7
    ]
    assert ext.converted


def _ambiguous_rate(raw):
    oms(raw).insert(
        0,
        _om(
            "Objective Response Rate",
            "NUMBER",
            "Percentage of participants",
            {"OG000": "40", "OG001": "20"},
            time_frame="12 months",
        ),
    )


def test_non_survival_rate_dropped():
    s, _ = mutated("NCT02142738", _ambiguous_rate)
    assert sv.extract_anchors(s, [0]).anchors == {}


def test_percent_unit_with_proportion_values_dropped(study):
    ext = sv.extract_anchors(study("NCT01844505"), [0, 3])  # measure 3: '%' unit but values like 0.52
    assert all(a.kind == "median" for a in ext.anchors["PFS · Nivolumab"])
    assert any("do not match the stated unit" in x for x in ext.skipped)


# -- item 4: time from exactly one expression, cited -------------------------------------------------------


def _two_times(raw):
    oms(raw).insert(
        0,
        _om(
            "Overall Survival Rate",
            "NUMBER",
            "Percentage of participants",
            {"OG000": "70", "OG001": "55"},
            time_frame="Assessed at 12 months and 24 months",
        ),
    )


def test_timeframe_with_two_times_dropped():
    s, _ = mutated("NCT02142738", _two_times)
    ext = sv.extract_anchors(s, [0])
    assert ext.anchors == {} and any("unambiguous time" in x for x in ext.skipped)


def test_time_source_field_is_cited(study):
    ext = sv.extract_anchors(study("NCT02142738"), [1])  # 'Overall Survival (OS) Rate', timeFrame '12 months'
    a = ext.anchors["OS · Pembrolizumab"][0]
    assert a.time == 12.0
    assert any(r.field_path.endswith("outcomeMeasures[1].timeFrame") and r.label == "time point" for r in a.refs)
    ext = sv.extract_anchors(study("NCT01844505"), [2])  # class titles carry the time
    assert any(".classes[" in r.field_path and r.label == "time point" for r in ext.anchors["OS · Nivolumab"][0].refs)


def test_open_ended_timeframe_rejected():
    s, _ = mutated("NCT02142738", lambda raw: oms(raw)[1].update(timeFrame="Up to approximately 21 months"))
    assert sv.extract_anchors(s, [1]).anchors == {}


# -- item 5: subgroup cells without a parseable time are skipped -------------------------------------------


def _subgroup_no_time(raw):
    classes = [{"title": "PD-L1 >= 50%", "categories": [{"measurements": [{"groupId": "OG000", "value": "70"}]}]}]
    oms(raw).insert(
        0,
        _om(
            "Overall Survival Rate",
            "NUMBER",
            "Percentage of participants",
            {"OG000": "70"},
            time_frame="12 months",
            classes=classes,
        ),
    )


def test_subgroup_cell_without_time_skipped():
    s, _ = mutated("NCT02142738", _subgroup_no_time)
    ext = sv.extract_anchors(s, [0])
    assert ext.anchors == {}  # the measure's timeFrame is NOT borrowed for a subgroup cell


# -- item 6: one median per series --------------------------------------------------------------------------


def test_one_median_per_series(study):
    s = study("NCT01844505")  # measures 1 and 5 are both OS medians
    ext = sv.extract_anchors(s, [1, 5])
    for pts in ext.anchors.values():
        assert sum(a.kind == "median" for a in pts) <= 1
    assert any("additional median" in x for x in ext.skipped)


# -- item 7: DLT value semantics ----------------------------------------------------------------------------


def _dlt_percent(raw):
    om = oms(raw)[0]
    om["unitOfMeasure"] = "Percentage of participants"
    om["paramType"] = "NUMBER"
    om["classes"][0]["categories"][0]["measurements"][0]["value"] = "16.0"


async def test_dlt_percent_unit(make_ctx):
    s, raw = mutated("NCT01284335", _dlt_percent)
    res = await run(make_ctx, R.risks_adverse_events, Intent.RISKS_ADVERSE_EVENTS, s, raw)
    row = next(
        d.model_extra
        for d in res.visualization.data
        if d.model_extra["group"] == "Arm A Tasisulam + Gemcitabine Dose Escalation"
    )
    assert row["rate"] == pytest.approx(0.16) and "n_events" not in row


def _dlt_fraction_count(raw):
    oms(raw)[0]["classes"][0]["categories"][0]["measurements"][0]["value"] = "2.5"


async def test_dlt_non_integer_count_dropped(make_ctx):
    s, raw = mutated("NCT01284335", _dlt_fraction_count)
    res = await run(make_ctx, R.risks_adverse_events, Intent.RISKS_ADVERSE_EVENTS, s, raw)
    groups = [d.model_extra["group"] for d in res.visualization.data]
    assert "Arm A Tasisulam + Gemcitabine Dose Escalation" not in groups
    assert any("not whole counts" in n.message for n in res.notes)


def _dlt_weird_unit(raw):
    oms(raw)[0]["unitOfMeasure"] = "mg/m2"
    oms(raw)[0]["paramType"] = "MEAN"


async def test_dlt_unknown_unit_not_plotted(make_ctx):
    s, raw = mutated("NCT01284335", _dlt_weird_unit)
    res = await run(make_ctx, R.risks_adverse_events, Intent.RISKS_ADVERSE_EVENTS, s, raw)
    assert res.visualization.encoding.y.title == "Serious adverse event rate"
    assert any("neither a participant count nor a percentage" in n.message for n in res.notes)


# -- item 9: log estimates are differences; neutral comparison labels --------------------------------------


def test_log_param_types_are_differences():
    assert effect_kind("Log Hazard Ratio") == "difference"
    assert effect_kind("Log odds ratio") == "difference"
    assert effect_kind("Hazard Ratio (HR)") == "ratio"


def test_comparison_labels_neutral(study):
    e = analyses_for(study("NCT02578680"), 1)[0]
    assert "vs" not in e.comparison and e.comparison.startswith("groups: ")
    s, _ = mutated("NCT02578680", lambda raw: oms(raw)[1]["analyses"][0].update(groupDescription="Pembro vs Control"))
    e = analyses_for(s, 1)[0]
    assert e.comparison == "Pembro vs Control" and e.description is None


async def test_log_estimate_forest_linear(make_ctx):
    def fn(raw):
        an = oms(raw)[1]["analyses"][0]
        an.update(paramType="Log Hazard Ratio", paramValue="-0.71", ciLowerLimit="-0.97", ciUpperLimit="-0.45")

    s, raw = mutated("NCT02578680", fn)
    res = await run(make_ctx, R.results_outcomes, Intent.RESULTS_OUTCOMES, s, raw, query="overall survival")
    viz = res.visualization
    assert viz.type is VizType.FOREST_PLOT and viz.encoding.x.scale.type == "linear"
    assert viz.meta.reference_line is not None and viz.meta.reference_line.value == 0.0


# -- item 10: unicode minus and inequality-qualified values -------------------------------------------------


def test_unicode_minus_and_qualifiers():
    assert parse_num("−12.44") == -12.44
    assert parse_num("– 3.1") == -3.1
    assert parse_qualified("−1.5") == (-1.5, None)
    assert parse_exact(">30.2") is None and parse_exact("30.2") == 30.2


def test_qualified_median_is_not_an_anchor():
    s, _ = mutated(
        "NCT02578680", lambda raw: oms(raw)[1]["classes"][0]["categories"][0]["measurements"][1].update(value=">11.3")
    )
    ext = sv.extract_anchors(s, [1])
    assert "OS · Control" not in ext.anchors and "OS · Control" in ext.not_reached


def test_qualified_estimate_not_plotted():
    s, _ = mutated("NCT02578680", lambda raw: oms(raw)[1]["analyses"][0].update(paramValue="<0.5"))
    assert analyses_for(s, 1) == []


# -- output hygiene: no None in encoded fields, unique datum ids --------------------------------------------


@pytest.mark.parametrize("nct", ["NCT03548935", "NCT02578680", "NCT01844505", "NCT01284335"])
@pytest.mark.parametrize(
    "handler,intent",
    [
        (R.risks_adverse_events, Intent.RISKS_ADVERSE_EVENTS),
        (R.results_outcomes, Intent.RESULTS_OUTCOMES),
        (R.survival_summary, Intent.SURVIVAL_SUMMARY),
        (R.dropout_flow, Intent.DROPOUT_FLOW),
        (R.demographics, Intent.DEMOGRAPHICS),
        (R.enrollment_vs_power, Intent.ENROLLMENT_VS_POWER),
    ],
)
async def test_no_none_in_encoded_fields_and_unique_ids(make_ctx, study, handler, intent, nct):
    res = await handler(make_ctx(intent, trials=[study(nct)]))
    viz = res.visualization
    if isinstance(viz.data, GraphData):
        items = viz.data.nodes + viz.data.edges
        assert len({n.id for n in viz.data.nodes}) == len(viz.data.nodes)
        assert len({e.id for e in viz.data.edges}) == len(viz.data.edges)
    else:
        items = viz.data
        assert len({d.id for d in items}) == len(items)
        enc = viz.encoding
        fields = [
            c.field
            for c in (enc.x, enc.x2, enc.y, enc.y2, enc.series, enc.size, enc.label, enc.error_low, enc.error_high)
            if c is not None
        ]
        for d in items:
            assert all((d.model_extra or {}).get(f) is not None for f in fields), d.id
    for d in items:
        assert all(v is not None for v in (d.model_extra or {}).values())
