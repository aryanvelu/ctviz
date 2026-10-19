"""Handler tests for the results/statistical intents on recorded fixtures (offline, llm=None)."""

from __future__ import annotations

import pytest

from ctviz.api.schemas import Intent, Scope
from ctviz.api.schemas.response import NoteCode
from ctviz.api.schemas.viz import GraphData, VizType
from ctviz.handlers import results as R
from tests.conftest import assert_citations_verified, load_study_raw

ALL = ["NCT03548935", "NCT02578680", "NCT02142738", "NCT01844505", "NCT01284335", "NCT07654361"]


def _raw(*ncts):
    return {n: load_study_raw(n) for n in ncts}


def _rows(viz):
    return [d.model_extra or {} for d in viz.data]


def _codes(res):
    return {n.code for n in res.notes}


async def _run(handler, make_ctx, study, nct, intent, query="test", **kw):
    ctx = make_ctx(intent, trials=[study(nct)], query=query, **kw)
    res = await handler(ctx)
    assert_citations_verified(res.visualization, _raw(nct))
    return res


# -- citation invariant across every handler and fixture ---------------------------------------------

HANDLERS = [
    (R.risks_adverse_events, Intent.RISKS_ADVERSE_EVENTS),
    (R.results_outcomes, Intent.RESULTS_OUTCOMES),
    (R.survival_summary, Intent.SURVIVAL_SUMMARY),
    (R.dropout_flow, Intent.DROPOUT_FLOW),
    (R.demographics, Intent.DEMOGRAPHICS),
    (R.enrollment_vs_power, Intent.ENROLLMENT_VS_POWER),
]


@pytest.mark.parametrize("nct", ALL)
@pytest.mark.parametrize("handler,intent", HANDLERS, ids=[h[1].value for h in HANDLERS])
async def test_citation_invariant(make_ctx, study, handler, intent, nct):
    res = await _run(handler, make_ctx, study, nct, intent)
    data = res.visualization.data
    items = data.nodes + data.edges if isinstance(data, GraphData) else data
    assert items, "handler produced no data"
    # every numeric datum except explicitly uncited derived rows carries citations
    uncited = [d.id for d in items if not d.citations]
    noted = {i for n in res.notes for i in n.affects}
    assert all(i in noted or "*" in noted for i in uncited)


def test_handlers_registered():
    from ctviz.viz.base import get_handler

    for _, intent in HANDLERS:
        assert get_handler(intent) is not None


# -- adverse events ----------------------------------------------------------------------------------


async def test_ae_step1_bar(make_ctx, study):
    res = await _run(R.risks_adverse_events, make_ctx, study, "NCT03548935", Intent.RISKS_ADVERSE_EVENTS)
    viz = res.visualization
    assert viz.type is VizType.BAR_CHART
    rows = _rows(viz)
    nausea = next(r for r in rows if r["term"] == "Nausea" and r["n_at_risk"] == 1306)
    assert nausea["n_affected"] == 576 and nausea["rate"] == pytest.approx(576 / 1306, abs=1e-4)
    assert {r["seriousness"] for r in rows} >= {"any_serious", "other"}
    assert rows[0]["term"] == "Any serious adverse event"
    assert len({r["term"] for r in rows}) == 16  # any-serious + top 15
    thr = viz.meta.thresholds[0]
    assert thr.source == "registry" and thr.value == pytest.approx(0.05)
    cites = viz.meta.thresholds[0].citations
    assert cites[0].field_path == "resultsSection.adverseEventsModule.frequencyThreshold"
    assert NoteCode.TRUNCATED in _codes(res) and viz.meta.truncation.top_n == 15


async def test_ae_dose_escalation_threshold_line(make_ctx, study):
    res = await _run(R.risks_adverse_events, make_ctx, study, "NCT01284335", Intent.RISKS_ADVERSE_EVENTS)
    viz = res.visualization
    assert viz.type is VizType.THRESHOLD_LINE
    t = viz.meta.thresholds[0]
    assert t.value == pytest.approx(0.33) and t.source == "convention" and "not from the registry" in t.label
    rows = _rows(viz)
    assert [r["dose_order"] for r in rows] == list(range(1, len(rows) + 1))
    first = rows[0]
    assert first["rate"] == pytest.approx(first["n_events"] / first["n_evaluable"], abs=1e-4)
    assert viz.encoding.x.sort.values == [r["group"] for r in rows]


async def test_ae_no_results_risk_table(make_ctx, study):
    res = await _run(R.risks_adverse_events, make_ctx, study, "NCT07654361", Intent.RISKS_ADVERSE_EVENTS)
    assert res.visualization.type is VizType.TABLE
    assert NoteCode.NO_RESULTS_POSTED in _codes(res)
    assert all(d.citations[0].kind.value == "text_excerpt" for d in res.visualization.data)


async def test_ae_cohort_scope(make_ctx, study):
    trials = [study(n) for n in ("NCT03548935", "NCT02578680", "NCT07654361")]
    ctx = make_ctx(Intent.RISKS_ADVERSE_EVENTS, trials=trials, scope=Scope.COHORT)
    res = await R.risks_adverse_events(ctx)
    viz = res.visualization
    assert viz.type is VizType.BAR_CHART and len(viz.data) == 2  # the no-results trial is skipped
    assert_citations_verified(viz, _raw("NCT03548935", "NCT02578680", "NCT07654361"))
    rates = [r["serious_rate"] for r in _rows(viz)]
    assert rates == sorted(rates, reverse=True)


# -- results ------------------------------------------------------------------------------------------


async def test_results_forest_step1(make_ctx, study):
    res = await _run(
        R.results_outcomes,
        make_ctx,
        study,
        "NCT03548935",
        Intent.RESULTS_OUTCOMES,
        query="What were the final results?",
    )
    viz = res.visualization
    assert viz.type is VizType.FOREST_PLOT
    row = _rows(viz)[0]
    assert set(row) >= {"label", "estimate", "ci_low", "ci_high", "ci_pct", "p_value", "param_type", "null_value"}
    assert row["estimate"] == -12.44 and row["null_value"] == 0.0 and row["p_value"] == "<.0001"
    c = viz.data[0].citations[0]
    assert c.field_path.endswith(".analyses[0].paramValue") and c.excerpt == "-12.44"


async def test_results_forest_hr_keynote189(make_ctx, study):
    res = await _run(
        R.results_outcomes,
        make_ctx,
        study,
        "NCT02578680",
        Intent.RESULTS_OUTCOMES,
        query="hazard ratio for overall survival",
    )
    viz = res.visualization
    assert viz.type is VizType.FOREST_PLOT and viz.encoding.x.scale.type == "log"
    assert any(r["estimate"] == 0.49 and r["ci_low"] == 0.38 for r in _rows(viz))


async def test_results_bars_when_no_estimates(make_ctx, study):
    res = await _run(
        R.results_outcomes,
        make_ctx,
        study,
        "NCT03548935",
        Intent.RESULTS_OUTCOMES,
        query="change in waist circumference",
    )
    viz = res.visualization
    assert viz.type is VizType.BAR_CHART and viz.encoding.error_low is not None
    vals = {r["group"]: r["value"] for r in _rows(viz)}
    assert vals == {"Semaglutide 2.4 mg": -14.1, "Placebo": -4.4}


async def test_results_no_results_registered_outcomes_table(make_ctx, study):
    res = await _run(R.results_outcomes, make_ctx, study, "NCT07654361", Intent.RESULTS_OUTCOMES)
    viz = res.visualization
    assert viz.type is VizType.TABLE and NoteCode.NO_RESULTS_POSTED in _codes(res)
    assert [c.field for c in viz.meta.columns] == ["type", "measure", "time_frame"]
    raw = load_study_raw("NCT07654361")["protocolSection"]["outcomesModule"]
    prim, sec = raw.get("primaryOutcomes", []), raw.get("secondaryOutcomes", [])
    rows = _rows(viz)
    # registry order, primary first; every row cites its own measure field
    assert [r["measure"] for r in rows] == [o["measure"] for o in prim + sec][: len(rows)]
    assert [r["type"] for r in rows] == ["Primary"] * len(prim) + ["Secondary"] * (len(rows) - len(prim))
    for d, r in zip(viz.data, rows, strict=True):
        kind = "primaryOutcomes" if r["type"] == "Primary" else "secondaryOutcomes"
        assert any(
            c.field_path.startswith(f"protocolSection.outcomesModule.{kind}[") and c.field_path.endswith(".measure")
            for c in d.citations
        )
    assert "status" in (viz.subtitle or "") and "no results posted" in viz.subtitle
    assert res.facts["primary_outcomes_derived"] == len(prim)
    assert res.facts["secondary_outcomes_derived"] == len(sec)


def test_registered_outcomes_caps_secondary_and_cuts_long_descriptions():
    from ctviz.core.provenance import Study, verify_citation

    long_desc = ("First sentence of a long description that keeps going for a while to pass the cut. " * 6).strip()
    raw = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001"},
            "statusModule": {"overallStatus": "RECRUITING"},
            "outcomesModule": {
                "primaryOutcomes": [{"measure": "DLT rate", "timeFrame": "28 days", "description": long_desc}],
                "secondaryOutcomes": [{"measure": f"Secondary {i}"} for i in range(R.MAX_SECONDARY_ROWS + 3)],
            },
        }
    }
    res = R._registered_outcomes(Study(raw), "Registered outcomes", "outcome results")
    rows = _rows(res.visualization)
    assert len(rows) == 1 + R.MAX_SECONDARY_ROWS
    assert NoteCode.TRUNCATED in _codes(res) and res.visualization.meta.truncation.omitted_count == 3
    assert res.facts["secondary_outcomes_derived"] == R.MAX_SECONDARY_ROWS + 3
    assert rows[1]["time_frame"] == "Not stated"
    desc = rows[0]["description"]
    assert desc.endswith(" …") and long_desc.startswith(desc[:-2]) and len(desc) <= 302
    for d in res.visualization.data:
        assert d.citations and all(verify_citation(c, raw) for c in d.citations)


# -- survival -----------------------------------------------------------------------------------------


async def test_survival_keynote189(make_ctx, study):
    res = await _run(
        R.survival_summary, make_ctx, study, "NCT02578680", Intent.SURVIVAL_SUMMARY, query="Show the survival curve"
    )
    viz = res.visualization
    assert viz.type is VizType.SURVIVAL_CURVE
    assert viz.meta.approximation.is_approximation and viz.meta.approximation.assumptions
    rows = _rows(viz)
    assert {r["source"] for r in rows} == {"reported", "interpolated"}
    os_ctrl = [r for r in rows if r["series"] == "OS · Control"]
    assert os_ctrl[0]["time"] == 0 and os_ctrl[0]["survival"] == 1.0
    med = next(r for r in os_ctrl if r["source"] == "reported" and r["time"] > 0)
    assert (med["time"], med["survival"]) == (11.3, 0.5)
    assert all(r["time"] <= 11.3 for r in os_ctrl)
    reported_med = next(d for d in viz.data if (d.model_extra or {}).get("time") == 11.3)
    assert reported_med.citations[0].excerpt == "11.3"
    interp = [d for d in viz.data if (d.model_extra or {})["source"] == "interpolated"]
    assert interp and all(d.citations for d in interp) and "Interpolated" in interp[0].model_extra["label"]
    hr = viz.meta.stats["analyses"]
    assert any(a["estimate"] == 0.49 and a["param_type"] == "Hazard Ratio (HR)" for a in hr)
    assert NoteCode.APPROXIMATED in _codes(res)
    assert any("not reached" in n.message for n in res.notes)


async def test_survival_landmarks_checkmate(make_ctx, study):
    res = await _run(
        R.survival_summary, make_ctx, study, "NCT01844505", Intent.SURVIVAL_SUMMARY, query="overall survival"
    )
    rows = _rows(res.visualization)
    nivo = [(r["time"], r["survival"]) for r in rows if r["series"] == "OS · Nivolumab" and r["source"] == "reported"]
    assert (12.0, 0.74) in nivo and (24.0, 0.59) in nivo
    assert all(r["series"].startswith("OS") for r in rows)


async def test_survival_fallback_forest_when_no_anchors(make_ctx, study, monkeypatch):
    from ctviz.compute import survival as sv

    monkeypatch.setattr(sv, "extract_anchors", lambda s, idx: sv.SurvivalExtraction({}, [], []))
    res = await _run(R.survival_summary, make_ctx, study, "NCT02578680", Intent.SURVIVAL_SUMMARY)
    assert res.visualization.type is VizType.FOREST_PLOT
    assert any(n.code is NoteCode.NOT_IN_REGISTRY and R.NOT_IN_REGISTRY_TTE in n.message for n in res.notes)


async def test_survival_fallback_table_without_hr(make_ctx, study, monkeypatch):
    from ctviz.compute import survival as sv

    monkeypatch.setattr(sv, "extract_anchors", lambda s, idx: sv.SurvivalExtraction({}, [], []))
    res = await _run(R.survival_summary, make_ctx, study, "NCT01284335", Intent.SURVIVAL_SUMMARY)
    assert res.visualization.type is VizType.TABLE and NoteCode.NOT_IN_REGISTRY in _codes(res)


# -- dropout flow -------------------------------------------------------------------------------------


def _check_conservation(graph: GraphData):
    nodes = {n.id: n.model_extra for n in graph.nodes}
    out: dict[str, int] = {}
    for e in graph.edges:
        ex = e.model_extra
        out[ex["source"]] = out.get(ex["source"], 0) + ex["weight"]
    for nid, total in out.items():
        assert nodes[nid]["value"] == total, f"{nid}: {nodes[nid]['value']} != {total}"
    return nodes, out


@pytest.mark.parametrize("nct", ["NCT03548935", "NCT02578680", "NCT01844505", "NCT02142738"])
async def test_sankey_conservation(make_ctx, study, nct):
    res = await _run(R.dropout_flow, make_ctx, study, nct, Intent.DROPOUT_FLOW)
    viz = res.visualization
    assert viz.type is VizType.SANKEY
    nodes, out = _check_conservation(viz.data)
    starts = [n for n in nodes if n.endswith(":start")]
    assert starts and all(s in out for s in starts)
    assert viz.meta.stages[0] == "Started"
    assert set(viz.meta.stats["dropout_rate"]) == {n["group"] for n in nodes.values()}


async def test_sankey_step1_values(make_ctx, study):
    res = await _run(R.dropout_flow, make_ctx, study, "NCT03548935", Intent.DROPOUT_FLOW)
    viz = res.visualization
    edges = [e.model_extra for e in viz.data.edges]
    sema = [e for e in edges if e["group"] == "Semaglutide 2.4 mg"]
    assert {e["type"] for e in sema} == {"completed", "dropout"}
    assert next(e for e in sema if e["type"] == "completed")["weight"] == 1240
    assert sum(e["weight"] for e in sema) == 1306
    assert viz.meta.stats["dropout_rate"]["Semaglutide 2.4 mg"] == pytest.approx(66 / 1306, abs=1e-4)
    assert res.facts["Semaglutide 2.4 mg|dropout_rate_pct"] == 5.1


async def test_sankey_unexplained_is_uncited_and_noted(make_ctx, study):
    res = await _run(R.dropout_flow, make_ctx, study, "NCT01844505", Intent.DROPOUT_FLOW)
    g = res.visualization.data
    unexplained = [e for e in g.edges if e.model_extra["type"] == "unexplained"]
    assert unexplained and all(not e.citations for e in unexplained)
    note = next(n for n in res.notes if n.code is NoteCode.NOT_IN_REGISTRY)
    assert all(e.id in note.affects for e in unexplained)
    assert sum(e.model_extra["weight"] for e in unexplained) == 3 + 1 + 4


async def test_dropout_no_results(make_ctx, study):
    res = await _run(R.dropout_flow, make_ctx, study, "NCT07654361", Intent.DROPOUT_FLOW)
    assert res.visualization.type is VizType.KPI_CARDS and NoteCode.NO_RESULTS_POSTED in _codes(res)


# -- demographics -------------------------------------------------------------------------------------


async def test_demographics_step1(make_ctx, study):
    res = await _run(R.demographics, make_ctx, study, "NCT03548935", Intent.DEMOGRAPHICS)
    viz = res.visualization
    assert viz.type is VizType.BAR_CHART and viz.meta.layout.mode == "stacked"
    rows = _rows(viz)
    assert all(r["group"] != "Total" for r in rows)
    female = next(r for r in rows if r["bar"] == "Sex: Placebo" and r["category"] == "Female")
    assert female["n"] == 498
    sex_shares = sum(r["share"] for r in rows if r["bar"] == "Sex: Placebo")
    assert sex_shares == pytest.approx(1.0, abs=1e-3)
    ages = viz.meta.stats["age_summary"]
    assert {a["value"] for a in ages} == {46.0, 47.0} and ages[0]["citations"]


async def test_demographics_age_only_kpi(make_ctx, study):
    res = await _run(
        R.demographics, make_ctx, study, "NCT03548935", Intent.DEMOGRAPHICS, query="what was the age breakdown"
    )
    assert res.visualization.type is VizType.KPI_CARDS
    assert {r["value"] for r in _rows(res.visualization)} == {46.0, 47.0}


async def test_demographics_cohort(make_ctx, cohort):
    c = cohort("semaglutide", label="Semaglutide")
    ctx = make_ctx(Intent.DEMOGRAPHICS, cohorts=[c])
    res = await R.demographics(ctx)
    raw = {s.nct_id: s.raw for s in c.studies}
    assert_citations_verified(res.visualization, raw)
    rows = _rows(res.visualization)
    assert {r["dimension"] for r in rows} == {"Eligible age group", "Eligible sex"}
    adult = next(r for r in rows if r["category"].startswith("Adult"))
    assert adult["trial_count"] == sum(1 for s in c.studies if "ADULT" in (s.get("std_ages") or []))


# -- power --------------------------------------------------------------------------------------------


async def test_power_step1_actual_counts(make_ctx, study):
    res = await _run(R.enrollment_vs_power, make_ctx, study, "NCT03548935", Intent.ENROLLMENT_VS_POWER)
    viz = res.visualization
    assert viz.type is VizType.POWER_CURVE
    rows = _rows(viz)
    assert len({r["series"] for r in rows}) == 3
    ns = sorted({r["n_per_arm"] for r in rows})
    assert ns[1] == pytest.approx(2 / (1 / 1306 + 1 / 655), abs=0.1)
    assert ns[0] == pytest.approx(ns[1] / 2, abs=0.2) and ns[2] == pytest.approx(ns[1] * 2, abs=0.2)
    zero = [r["power"] for r in rows if r["effect_size"] == 0]
    assert all(p == pytest.approx(0.025, abs=1e-3) for p in zero)
    for series in {r["series"] for r in rows}:
        curve = [r["power"] for r in rows if r["series"] == series]
        assert curve == sorted(curve)
    assert any(t.value == 0.8 for t in viz.meta.thresholds)
    mde = viz.meta.stats["mde_at_80"]
    vals = sorted(mde.values(), reverse=True)
    assert vals[0] > vals[1] > vals[2]
    assert viz.meta.approximation.is_approximation and NoteCode.APPROXIMATED in _codes(res)
    assert "started" in viz.data[0].citations[0].label


async def test_power_from_enrollment_no_results(make_ctx, study):
    res = await _run(R.enrollment_vs_power, make_ctx, study, "NCT07654361", Intent.ENROLLMENT_VS_POWER)
    rows = _rows(res.visualization)
    assert {r["n_per_arm"] for r in rows} == {450.0, 900.0, 1800.0}  # 3600 / 4 arms
    assert res.visualization.data[0].citations[0].field_path.endswith("enrollmentInfo.count")
    assert any("estimate" in a for a in res.visualization.meta.approximation.assumptions)
