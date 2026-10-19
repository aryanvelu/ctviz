"""Slim cross-trial survival curves + cohort-level summaries (offline fixtures).

* interpolated points cite one excerpt (the measure title) and link their anchors via ``derived_from``
* one endpoint per multi-trial chart, with a note naming what was left out
* bounded trials / arms / points, with a truncation note giving the omitted count
* the public payload of a large cohort stays well under 60 KB
* cohort summaries describe ALL rows and survive the summary-number verification
"""

from __future__ import annotations

import copy
import json
import math
from collections import defaultdict

import pytest

from ctviz.agents.synthesis import _cohort_curve_sentences, _cohort_effect_sentences, template_summary
from ctviz.agents.verification import Evidence, verify_response, verify_summary
from ctviz.api.public import to_public
from ctviz.api.schemas import Intent, Scope
from ctviz.api.schemas.response import (
    Coverage,
    Interpretation,
    NoteCode,
    ResponseMeta,
    VisualizeResponse,
)
from ctviz.compute import survival as sv
from ctviz.core.provenance import FieldRef, Study
from ctviz.handlers import results as R
from ctviz.viz.base import CohortData
from tests.conftest import assert_citations_verified, load_study_raw

SURV = ["NCT02578680", "NCT02142738", "NCT01844505"]  # KEYNOTE-189, KEYNOTE-024, CheckMate 067
_TITLE_PATH = "outcomeMeasures["


def _clones(n: int, mixed: bool = True) -> tuple[list[Study], dict[str, dict]]:
    """``n`` copies of the survival fixtures under fake NCT ids (a large cohort without network access).

    ``mixed``: enrollment decreases with the index, so the three source trials alternate in the ranking.
    """
    studies, raws = [], {}
    for i in range(n):
        raw = copy.deepcopy(load_study_raw(SURV[i % 3]))
        nct = f"NCT9{i:07d}"
        raw["protocolSection"]["identificationModule"]["nctId"] = nct
        if mixed:
            raw["protocolSection"]["designModule"]["enrollmentInfo"]["count"] = 5000 - i
        studies.append(Study(raw))
        raws[nct] = raw
    return studies, raws


def _cohort_ctx(make_ctx, studies, query, intent=Intent.SURVIVAL_SUMMARY):
    ctx = make_ctx(intent, cohorts=[CohortData(label="X", spec=None, studies=studies, total_count=2033)], query=query)
    ctx.trials = studies  # what the orchestrator's result-record fetch leaves in ctx.trials
    assert ctx.plan.scope is Scope.COHORT and not ctx.plan.nct_ids
    return ctx


def _rows(viz):
    return [d.model_extra or {} for d in viz.data]


def _response(ctx, res) -> VisualizeResponse:
    summary = template_summary(ctx.plan, res, res.visualization)
    return VisualizeResponse(
        request_id="a" * 32,
        created_at="2026-10-19T00:00:00+00:00",
        visualization=res.visualization,
        answer_summary=summary,
        meta=ResponseMeta(
            interpretation=Interpretation(
                intent=ctx.plan.intent,
                intent_confidence=1,
                persona=ctx.plan.persona,
                scope=ctx.plan.scope,
                parser="rules",
            ),
            coverage=Coverage(studies_matched_total=2033, studies_analyzed=res.studies_analyzed),
            notes=res.notes,
        ),
    )


# -- provenance of interpolated points --------------------------------------------------------------


@pytest.mark.parametrize("cohort", [False, True])
async def test_interpolated_points_link_existing_anchors(make_ctx, study, cohort):
    if cohort:
        studies, raws = _clones(12)
        ctx = _cohort_ctx(make_ctx, studies, "kaplan meier curves")
    else:
        raws = {"NCT02578680": load_study_raw("NCT02578680")}
        ctx = make_ctx(Intent.SURVIVAL_SUMMARY, trials=[study("NCT02578680")], query="Show the survival curve")
    res = await R.survival_summary(ctx)
    viz = res.visualization
    by_id = {d.id: d for d in viz.data}
    interp = [d for d in viz.data if d.model_extra["source"] == "interpolated"]
    assert interp
    for d in interp:
        extra = d.model_extra
        assert len(d.citations) == 1 and d.citations_total is None  # exactly one excerpt: the measure title
        assert d.citations[0].field_path.endswith(".title") and _TITLE_PATH in d.citations[0].field_path
        assert ".groups[" not in d.citations[0].field_path
        assert len(extra["derived_from"]) == 2
        lo, hi = (by_id[i].model_extra for i in extra["derived_from"])  # ids exist in the chart
        assert lo["source"] == hi["source"] == "reported"
        assert lo["series"] == hi["series"] == extra["series"]
        assert lo["time"] < extra["time"] < hi["time"]
        assert hi["survival"] <= extra["survival"] <= lo["survival"]
    for d in viz.data:  # reported anchors keep their full citations and never carry derived_from
        if d.model_extra["source"] == "reported":
            assert "derived_from" not in d.model_extra and d.citations
            if d.model_extra["time"] > 0:
                assert any(c.field_path.endswith(".value") for c in d.citations)
    assert_citations_verified(viz, raws)
    approx = next(n for n in res.notes if n.code is NoteCode.APPROXIMATED and "interpolated" in n.message)
    assert set(approx.affects) == {d.id for d in interp}
    assert viz.meta.approximation.is_approximation


def test_reconstruct_density_is_bounded():
    a = sv.Anchor("S", "OS", "g", 60.0, 0.2, "median")
    b = sv.Anchor("S", "OS", "g", 6.0, 0.9, "landmark")
    pts = sv.reconstruct_survival({"S": [b, a]}, {"S": []}, max_interpolated=4)
    interp = [p for p in pts if p.source == "interpolated"]
    assert 0 < len(interp) <= 4
    times = [p.time for p in pts]
    assert times == sorted(times) and times[0] == 0 and times[-1] == 60.0
    assert all(p.between in ((0, 1), (1, 2)) for p in interp)
    assert [p.anchor_pos for p in pts if p.source == "reported"] == [0, 1, 2]
    short = sv.reconstruct_survival({"S": [sv.Anchor("S", "OS", "g", 3.0, 0.5, "median")]}, {"S": []})
    gaps = [b.time - a.time for a, b in zip(short, short[1:], strict=False)]
    assert all(g >= sv.MIN_STEP_MONTHS - 1e-9 for g in gaps)  # never denser than one point per month


# -- one endpoint per chart, bounded size -------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "endpoint", "left_out"),
    # an unspecific question draws OS and says PFS was left out; a PFS question only selects PFS measures
    [("kaplan meier for pembrolizumab", "OS", "PFS"), ("kaplan meier progression-free survival", "PFS", None)],
)
async def test_one_endpoint_per_cohort_chart(make_ctx, query, endpoint, left_out):
    studies, _ = _clones(6)
    res = await R.survival_summary(_cohort_ctx(make_ctx, studies, query))
    rows = _rows(res.visualization)
    assert {r["endpoint"] for r in rows} == {endpoint}
    assert all(f" · {endpoint} · " in r["series"] for r in rows)
    if left_out:
        note = next(n for n in res.notes if "One endpoint per chart" in n.message)
        assert f"only {endpoint} curves are drawn" in note.message and left_out in note.message
    assert all(
        a["label"].lower().find("progression" if endpoint == "OS" else "overall") < 0
        for a in res.visualization.meta.stats.get("analyses", [])
    )


@pytest.mark.parametrize("mixed", [True, False])
async def test_large_cohort_payload_is_bounded(make_ctx, mixed):
    studies, raws = _clones(25, mixed=mixed)
    ctx = _cohort_ctx(make_ctx, studies, "kaplan meier for nivolumab")
    res = await R.survival_summary(ctx)
    rows = _rows(res.visualization)
    trials = list(dict.fromkeys(r["nct_id"] for r in rows))
    series = {r["series"] for r in rows}
    assert 1 <= len(trials) <= R.MAX_CURVE_TRIALS
    if len(trials) > 1:  # the first trial is always drawn; later trials only if they fit the chart budget
        cost = sum(1 + len(d.citations) for d in res.visualization.data)
        assert len(series) <= R.MAX_CURVE_SERIES and cost <= R.MAX_CURVE_COST
    per_series = defaultdict(int)
    for r in rows:
        per_series[r["series"]] += r["source"] == "interpolated"
    assert max(per_series.values()) <= R.MAX_COHORT_INTERPOLATED
    trunc = next(n for n in res.notes if n.message.startswith("Curves drawn for"))
    omitted = 25 - len(trials)
    assert f"{omitted} omitted" in trunc.message
    resp, vnotes = verify_response(_response(ctx, res), raws, facts=res.facts)
    assert not [n for n in vnotes if n.code in (NoteCode.DEGRADED, NoteCode.SUMMARY_TRIMMED)]
    size = len(json.dumps(to_public(resp), separators=(",", ":")))
    assert size < 45_000, size  # before: ~430 KB of chart data alone for the same cohort
    assert len(json.dumps(to_public(resp), indent=2)) < 65_000  # pretty-printed, as ``ctviz ask --out`` writes it


# -- cohort-level summaries ---------------------------------------------------------------------------


async def test_cohort_curve_summary_covers_every_drawn_trial(make_ctx):
    studies, raws = _clones(12)
    ctx = _cohort_ctx(make_ctx, studies, "kaplan meier curves")
    res = await R.survival_summary(ctx)
    rows = _rows(res.visualization)
    resp, vnotes = verify_response(_response(ctx, res), raws, facts=res.facts)
    summary = resp.answer_summary
    assert summary and not [n for n in vnotes if n.code is NoteCode.SUMMARY_TRIMMED]
    drawn = len({r["nct_id"] for r in rows})
    assert summary.startswith(f"Approximate overall survival curves are drawn for {drawn} of the 12 analysed trials")
    medians = sorted(r["time"] for r in rows if r.get("label", "").startswith("Reported median"))
    assert f"from {medians[0]:g} to {round(medians[-1], 1):g} months ({len(medians)} arms)" in summary
    assert "Hazard Ratio" not in summary and "progression" not in summary.lower()


async def test_cohort_forest_summary_counts_all_rows(make_ctx):
    studies, raws = _clones(9)
    ctx = _cohort_ctx(make_ctx, studies, "hazard ratios for overall survival")
    res = await R.survival_summary(ctx)
    rows = _rows(res.visualization)
    assert res.visualization.type.value == "forest_plot" and len(rows) > 3
    resp, vnotes = verify_response(_response(ctx, res), raws, facts=res.facts)
    summary = resp.answer_summary
    assert summary and not [n for n in vnotes if n.code is NoteCode.SUMMARY_TRIMMED]
    trials = len({r["nct_id"] for r in rows})
    assert f"reported by {trials} trials" in summary and f"of {len(rows)} estimates" in summary
    c = res.facts["cohort_effects_derived"]
    below = sum(r["ci_high"] < 1 for r in rows)
    above = sum(r["ci_low"] > 1 for r in rows)
    assert (c["ci_below_null"], c["ci_above_null"], c["ci_crosses_null"]) == (below, above, len(rows) - below - above)
    est = [r["estimate"] for r in rows]
    assert f"from {min(est):g} to {max(est):g}" in summary


async def test_results_across_trials_summary_is_cohort_level(make_ctx):
    studies, raws = _clones(6)
    ctx = _cohort_ctx(make_ctx, studies, "compare results across trials", intent=Intent.RESULTS_OUTCOMES)
    res = await R.results_outcomes(ctx)
    rows = _rows(res.visualization)
    resp, _ = verify_response(_response(ctx, res), raws, facts=res.facts)
    trials = len({r["nct_id"] for r in rows})
    assert f"reported by {trials} trials" in resp.answer_summary
    assert "and 2 more" in resp.answer_summary  # 6 trials: 4 named + 2 more, none silently dropped


def test_effect_sentence_mixed_positions_and_differences():
    facts = {
        "what": "Hazard ratios",
        "endpoint_name": "overall survival",
        "estimates": 10,
        "trials": 10,
        "trial_names": ["CheckMate 067", "KEYNOTE-006", "COMBI-v", "coBRIM"],
        "names_more": 6,
        "null_value": 1,
        "ci_below_null": 6,
        "ci_above_null": 1,
        "ci_crosses_null": 3,
        "estimate_low": 0.55,
        "estimate_high": 1.25,
    }
    out = _cohort_effect_sentences(facts)
    assert out[0] == (
        "Hazard ratios for overall survival reported by 10 trials (CheckMate 067, KEYNOTE-006, COMBI-v, coBRIM and 6 "
        "more): the confidence interval lies entirely below 1 for 6, lies entirely above 1 for 1 and crosses 1 for 3."
    )
    assert out[1] == "Estimates range from 0.55 to 1.25."
    diff = {**facts, "what": "Mean Difference estimates", "null_value": 0, "unit": "percent", "estimate_low": -1.2}
    diff.pop("endpoint_name")
    assert "crosses 0 for 3" in _cohort_effect_sentences(diff)[0]
    assert _cohort_effect_sentences(diff)[1] == "Estimates range from -1.2 to 1.25 percent."


async def test_single_trial_summary_states_medians(make_ctx, study):
    ctx = make_ctx(Intent.SURVIVAL_SUMMARY, trials=[study("NCT02578680")], query="Show the survival curve")
    res = await R.survival_summary(ctx)
    summary = template_summary(ctx.plan, res, res.visualization)
    assert "Reported medians:" in summary and "OS · Control 11.3 months" in summary
    assert not math.isnan(res.facts["medians_months_derived"]["OS · Control"])


# -- review fixes ---------------------------------------------------------------------------------------


def _set_acronym(studies: list[Study], acronym: str) -> list[Study]:
    for st in studies:
        st.raw["protocolSection"]["identificationModule"]["acronym"] = acronym
    return studies


def _all_os_not_reached(st: Study) -> Study:
    """KEYNOTE-189 clone whose OS medians are all 'NA' (median not reached in every arm)."""
    om = st.raw["resultsSection"]["outcomeMeasuresModule"]["outcomeMeasures"][1]
    assert om["title"] == "Overall Survival (OS)"
    for cl in om.get("classes", []):
        for cat in cl.get("categories", []):
            for m in cat.get("measurements", []):
                m["value"] = "NA"
    return st


async def test_all_not_reached_trial_is_named_as_such_and_counted(make_ctx):
    studies, raws = _clones(4)
    nr = _all_os_not_reached(studies[0])  # KEYNOTE-189 clone, ranked first
    ctx = _cohort_ctx(make_ctx, studies, "kaplan meier for overall survival")
    res = await R.survival_summary(ctx)
    msgs = [n.message for n in res.notes]
    nr_note = next(m for m in msgs if "median not reached in every reported arm" in m)
    assert nr.nct_id in nr_note
    assert not any(nr.nct_id in m for m in msgs if "no reported median or landmark" in m)
    assert nr.nct_id not in {r["nct_id"] for r in _rows(res.visualization)}
    c = res.facts["cohort_curves_derived"]
    nr_arms = sum(1 for s in sv.extract_anchors(nr, [1]).not_reached if s.startswith("OS "))
    assert nr_arms >= 2 and c["arms_median_not_reached"] >= nr_arms


async def test_shared_acronym_keeps_trials_apart(make_ctx):
    studies, _ = _clones(4)
    _set_acronym(studies, "SAME")
    res = await R.survival_summary(_cohort_ctx(make_ctx, studies, "kaplan meier curves"))
    rows = _rows(res.visualization)
    by_series = defaultdict(set)
    for r in rows:
        by_series[r["series"]].add(r["nct_id"])
        assert r["series"].startswith(f"SAME ({r['nct_id']}) · ")
    assert all(len(v) == 1 for v in by_series.values())  # no series mixes two trials
    names = res.facts["cohort_curves_derived"]["trial_names"]
    assert len(names) == len(set(names)) == min(4, len({r["nct_id"] for r in rows}))


async def test_derived_from_ids_resolve_in_public_json(make_ctx):
    studies, raws = _clones(6)
    ctx = _cohort_ctx(make_ctx, studies, "kaplan meier curves")
    res = await R.survival_summary(ctx)
    resp, _ = verify_response(_response(ctx, res), raws, facts=res.facts)
    for detail in ("sample", "ids_only", "full"):
        rows = to_public(resp, citation_detail=detail)["visualization"]["data"]
        by_id = {r["id"]: r for r in rows}
        derived = [r for r in rows if r["source"] == "interpolated"]
        assert derived
        for r in derived:
            assert all(
                by_id[i]["source"] == "reported" and by_id[i]["series"] == r["series"] for i in r["derived_from"]
            )


async def test_arms_per_trial_truncation_note(make_ctx, monkeypatch):
    monkeypatch.setattr(R, "MAX_ARMS_PER_TRIAL", 2)
    studies, _ = _clones(3)  # the CheckMate 067 clone reports OS for 3 arms
    res = await R.survival_summary(_cohort_ctx(make_ctx, studies, "kaplan meier curves"))
    assert any("2 of 3 arms drawn" in n.message and "omitted" in n.message for n in res.notes)
    per_trial = defaultdict(set)
    for r in _rows(res.visualization):
        per_trial[r["nct_id"]].add(r["series"])
    assert max(len(v) for v in per_trial.values()) <= 2


async def test_first_trial_is_cut_to_the_budget(make_ctx, monkeypatch):
    monkeypatch.setattr(R, "MAX_CURVE_COST", 60)
    studies, _ = _clones(3, mixed=False)  # CheckMate 067 (largest, many landmarks) ranks first
    res = await R.survival_summary(_cohort_ctx(make_ctx, studies, "kaplan meier curves"))
    assert sum(1 + len(d.citations) for d in res.visualization.data) <= 60
    msgs = [n.message for n in res.notes]
    assert any("to keep the chart within its point budget" in m for m in msgs)
    reported = defaultdict(int)
    for r in _rows(res.visualization):
        reported[r["series"]] += r["source"] == "reported"
    assert max(reported.values()) <= R.MAX_COHORT_ANCHORS_PER_ARM + 1  # + S(0)


def test_derived_points_fall_back_to_the_value_citation():
    ref = FieldRef("NCT00000001", "resultsSection.x.value", "11.3")
    a = sv.Anchor("S", "OS", "g", 10.0, 0.5, "median", [ref])
    pts = sv.reconstruct_survival({"S": [a]}, {"S": [sv.single_ref(a)]}, max_interpolated=3)
    assert all(p.refs == [ref] for p in pts)  # no measure title among the refs: the value is cited instead


def test_landmark_rate_of_100_percent_survives_the_summary_check():
    c = R._curve_facts("OS", ["A", "B"], 2, 2, [], {12.0: [1.0, 0.8]}, 0)
    summary = " ".join(_cohort_curve_sentences(c))
    assert "from 80.0% to 100.0%" in summary
    ev = Evidence()
    ev.add_values(c)
    for name in c["trial_names"]:
        ev.add_text(name)
    kept, notes = verify_summary(summary, ev)
    assert kept == summary and not notes


@pytest.mark.parametrize(
    ("query", "endpoint"),
    [("compare overall survival results", "OS"), ("compare progression-free survival results", "PFS")],
)
async def test_cohort_forest_follows_the_named_endpoint(make_ctx, query, endpoint):
    studies, _ = _clones(6)
    res = await R.results_outcomes(_cohort_ctx(make_ctx, studies, query, intent=Intent.RESULTS_OUTCOMES))
    rows = _rows(res.visualization)
    assert rows and {r["endpoint"] for r in rows} == {endpoint}
    assert res.facts["cohort_effects_derived"]["endpoint_name"] in ("overall survival", "progression-free survival")


def test_identical_forest_rows_are_deduplicated():
    from types import SimpleNamespace

    st = Study(load_study_raw("NCT02578680"))
    e = SimpleNamespace(measure_title="OS", group_names=("A", "B"), comparison="", param_type="HR", estimate=0.7,
                        ci_low=0.6, ci_high=0.8)  # fmt: skip
    other = SimpleNamespace(**{**vars(e), "estimate": 0.75})
    kept, dups = R._dedupe_estimates([(st, e), (st, SimpleNamespace(**vars(e))), (st, other)])
    assert dups == 1 and [x.estimate for _, x in kept] == [0.7, 0.75]


async def test_reported_points_per_arm_are_capped_earliest_first(make_ctx, monkeypatch):
    monkeypatch.setattr(R, "MAX_COHORT_ANCHORS_PER_ARM", 2)
    studies, _ = _clones(3, mixed=False)
    res = await R.survival_summary(_cohort_ctx(make_ctx, studies, "kaplan meier curves"))
    assert any("only the first 2 reported OS values per arm are drawn" in n.message for n in res.notes)
    times = defaultdict(list)
    for r in _rows(res.visualization):
        if r["source"] == "reported" and r["time"] > 0:
            times[r["series"]].append(r["time"])
    assert times and all(len(v) <= 2 for v in times.values())
    assert all(v == sorted(v) for v in times.values())
