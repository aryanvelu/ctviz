"""Verification agent: citation re-checks and summary-number trimming; template synthesis."""

from __future__ import annotations

from ctviz.agents.synthesis import fmt_number, template_summary
from ctviz.agents.verification import (
    Evidence,
    as_table,
    numbers_in,
    verify_citations,
    verify_response,
    verify_summary,
)
from ctviz.api.schemas.citations import Citation
from ctviz.api.schemas.plan import Intent, Persona, QueryPlan
from ctviz.api.schemas.response import (
    Coverage,
    Interpretation,
    NoteCode,
    ResponseMeta,
    VisualizeResponse,
)
from ctviz.api.schemas.viz import Channel, VizType
from ctviz.core.provenance import Dataset
from ctviz.viz.base import HandlerResult
from ctviz.viz.builders.bar_chart import bar_chart
from tests.conftest import load_study_raw

NCT = "NCT03548935"


def _viz(study, bad: bool = False):
    ds = Dataset()
    refs = [study.ref("phases"), study.ref("enrollment_count")]
    row = ds.emit({"phase": "Phase 3", "trial_count": 1306, "share": 0.442}, refs)
    if bad:
        forged = Citation(
            nct_id=NCT,
            excerpt="PHASE2",
            field_path="protocolSection.designModule.phases",
            url="https://clinicaltrials.gov/study/" + NCT,
        )
        ghost = Citation(
            nct_id="NCT00000000",
            excerpt="x",
            field_path="protocolSection.identificationModule.nctId",
            url="https://clinicaltrials.gov/study/NCT00000000",
        )
        ds.rows[0] = row.model_copy(update={"citations": [*row.citations, forged, ghost]})
    return bar_chart(
        ds.rows,
        title="Trials by phase",
        x=Channel(field="phase", type="nominal", title="Phase"),
        y=Channel(field="trial_count", type="quantitative", title="Trials", unit="trials"),
    )


def _resp(viz, summary: str) -> VisualizeResponse:
    interp = Interpretation(
        intent=Intent.PHASE_DISTRIBUTION,
        intent_confidence=1,
        persona=Persona.RESEARCHER,
        scope="cohort",
        parser="rules",
    )
    return VisualizeResponse(
        request_id="r",
        created_at="now",
        visualization=viz,
        answer_summary=summary,
        meta=ResponseMeta(interpretation=interp, coverage=Coverage(studies_matched_total=2916, studies_analyzed=400)),
    )


def test_bad_citations_dropped(study):
    s = study(NCT)
    viz, notes = verify_citations(_viz(s, bad=True), {NCT: load_study_raw(NCT)})
    assert len(viz.data[0].citations) == 2
    assert notes and notes[0].affects == [viz.data[0].id]


def test_good_citations_untouched(study):
    s = study(NCT)
    viz = _viz(s)
    out, notes = verify_citations(viz, {NCT: s.raw})
    assert out is viz and notes == []


def _ev(*values) -> Evidence:
    ev = Evidence()
    for v in values:
        ev.add_values(v)
    return ev


def test_number_formats_tolerated():
    ev = _ev(1306, 0.442, 2916)
    text = "There are 1,306 trials. That is 44.2% of the total. Share 0.442 overall. Out of 2,916 matched."
    assert verify_summary(text, ev) == (text, [])


def test_summary_sentence_with_unknown_number_trimmed(study):
    s = study(NCT)
    resp = _resp(
        _viz(s), f"Phase 3 has 1,306 trials [{NCT}]. Approval is expected in 2031. Of 400 analyzed, 44% are Phase 3."
    )
    out, notes = verify_response(resp, {NCT: s.raw})
    assert out.answer_summary == f"Phase 3 has 1,306 trials [{NCT}]. Of 400 analyzed, 44% are Phase 3."
    assert [n.code for n in notes] == [NoteCode.SUMMARY_TRIMMED]
    assert any(n.code is NoteCode.SUMMARY_TRIMMED for n in out.meta.notes)


def test_all_sentences_trimmed_gives_none(study):
    s = study(NCT)
    out, _ = verify_response(_resp(_viz(s), "Roughly 9,999 people joined."), {NCT: s.raw})
    assert out.answer_summary is None


def test_nct_ids_are_not_numbers():
    assert numbers_in("See [NCT03548935] and NCT02578680 for 12 trials") == ["12"]


def test_template_summary_passes_verification(study):
    s = study(NCT)
    viz = _viz(s)
    result = HandlerResult(
        visualization=viz, facts={"A|Phase 3": 1306, "A|Phase 2": 12, "B|Phase 3": 7}, studies_analyzed=400
    )
    plan = QueryPlan(intent=Intent.PHASE_DISTRIBUTION, intent_confidence=1)
    summary = template_summary(plan, result)
    # studies_analyzed (400) is coverage metadata, not a data/fact value, so the summary does not state it
    assert "1,306" in summary and "For A" in summary and "400" not in summary
    resp = _resp(viz, summary)
    out, notes = verify_response(resp, {NCT: s.raw}, facts=result.facts)
    assert out.answer_summary == summary and notes == []
    # when the handler reports the analyzed count as a fact, the summary may cite it
    counted = HandlerResult(
        visualization=viz, facts={**result.facts, "trials_analyzed_derived": 400}, studies_analyzed=400
    )
    assert "Based on 400 trials" in template_summary(plan, counted)
    patient = template_summary(plan.model_copy(update={"persona": Persona.PATIENT}), result)
    assert "largest group" in patient


def test_template_summary_skips_only_the_remainder_bucket(study):
    """'Other (N more)' pools categories and is never the top one; a real 'Other (...)' category can be."""
    viz = _viz(study(NCT))
    plan = QueryPlan(intent=Intent.SPONSOR_LANDSCAPE, intent_confidence=1)
    facts = {"Lung|Other (academic, hospital, non-profit)": 8187, "Lung|Industry": 4321, "Lung|Other (5 more)": 9000}
    summary = template_summary(plan, HandlerResult(visualization=viz, facts=facts))
    assert "Other (academic, hospital, non-profit) has the most" in summary
    assert "Other (5 more)" not in summary


def test_template_summary_from_rows_without_facts(study):
    viz = _viz(study(NCT))
    summary = template_summary(
        QueryPlan(intent=Intent.PHASE_DISTRIBUTION, intent_confidence=1), HandlerResult(visualization=viz)
    )
    assert "Phase 3" in summary and "1,306" in summary


def test_template_summary_prefers_rows_over_bare_scalars(study):
    """Scalar-only facts (e.g. a trend's analyzed count) do not hide the chart's distribution sentence."""
    viz = _viz(study(NCT))
    plan = QueryPlan(intent=Intent.PHASE_DISTRIBUTION, intent_confidence=1)
    res = HandlerResult(visualization=viz, facts={"trials_analyzed_derived": 1306}, studies_analyzed=1306)
    summary = template_summary(plan, res)
    assert summary.startswith("Phase 3 has the most trials (1,306)")
    assert "derived" not in summary and summary.count("1,306") == 2  # bucket + "Based on 1,306 trials"


def test_template_summary_no_bucket_sentence_for_non_distributions(study):
    """Survival-style 'series|kind|time' facts are not a distribution: no 'largest bucket' sentence."""
    viz = _viz(study(NCT)).model_copy(update={"type": VizType.SURVIVAL_CURVE})
    res = HandlerResult(visualization=viz, facts={"OS · Control|median|11.3": 0.5}, studies_analyzed=1)
    summary = template_summary(QueryPlan(intent=Intent.SURVIVAL_SUMMARY, intent_confidence=1), res) or ""
    assert "highest value" not in summary and "11.3" not in summary


def test_fmt_number():
    assert fmt_number(1306) == "1,306" and fmt_number(44.0) == "44" and fmt_number(0.4421) == "0.44"


# -- review fixes ---------------------------------------------------------------------------------
def _broken_bar():
    from ctviz.api.schemas.viz import Datum, Encoding, Visualization, VizType

    rows = [Datum(id="d1", phase="Phase 2"), Datum(id="d2", phase="Phase 3", trial_count=3)]
    return Visualization(
        type=VizType.BAR_CHART,
        title="Broken",
        data=rows,
        encoding=Encoding(
            x=Channel(field="phase", type="nominal", title="Phase"),
            y=Channel(field="trial_count", type="quantitative", title="n"),
        ),
    )


def test_item1_encoding_error_degrades_to_valid_table():
    resp = _resp(_broken_bar(), None)
    out, notes = verify_response(resp, {})
    assert out.visualization.type.value == "table"
    assert [n.code for n in notes] == [NoteCode.VIZ_DOWNGRADED]
    d1 = out.visualization.data[0].model_extra
    assert d1["trial_count"] == "—" and "trial_count" in d1["cite_note"]
    assert {c.field for c in out.visualization.meta.columns} == {"phase", "trial_count"}


def test_item1_as_table_empty_is_kpi():
    from ctviz.api.schemas.viz import Encoding, Visualization, VizType

    viz = Visualization(type=VizType.BAR_CHART, title="x", data=[], encoding=Encoding())
    assert as_table(viz).type.value == "kpi_cards"


def test_item4_typed_scalar_facts_not_compared(study):
    from ctviz.core.orchestrator import trial_table

    viz, _ = trial_table([study(NCT)], "Candidates")
    result = HandlerResult(
        visualization=viz, facts={"candidate_trials": 10, "matching_trials": 400}, studies_analyzed=10
    )
    text = template_summary(QueryPlan(intent=Intent.ELIGIBILITY_CHECK, intent_confidence=1), result)
    assert "400 trials on ClinicalTrials.gov match" in text and "10 most recently updated" in text
    assert "highest" not in text and "then" not in text


def test_item4_no_summary_for_empty_or_zero_states(study):
    from ctviz.viz.selector import empty_kpi

    plan = QueryPlan(intent=Intent.PHASE_DISTRIBUTION, intent_confidence=1)
    empty = HandlerResult(visualization=empty_kpi("t", "No matching data"), facts={"A|Phase 2": 5}, studies_analyzed=5)
    assert template_summary(plan, empty) is None
    zero = HandlerResult(visualization=_viz(study(NCT)), facts={"A|Phase 2": 0, "A|Phase 3": 0})
    assert template_summary(plan, zero) is None


def test_item4_uses_final_viz(study):
    from ctviz.viz.selector import empty_kpi

    plan = QueryPlan(intent=Intent.PHASE_DISTRIBUTION, intent_confidence=1)
    result = HandlerResult(visualization=_viz(study(NCT)), studies_analyzed=1)
    assert template_summary(plan, result) is not None
    assert template_summary(plan, result, viz=empty_kpi("t", "nothing")) is None


def test_item5_strict_number_evidence():
    ev = _ev([{"share": 0.442, "trial_count": 44.2, "start": "2019-03-01", "phase": "Phase 2"}])
    assert verify_summary("Share is 44.2%.", ev)[1] == []  # fraction x100 for % tokens
    assert verify_summary("Value 44.2.", ev)[1] == []  # exact
    assert verify_summary("Share 0.442 overall.", _ev([{"v": 44.2}]))[1]  # no /100 for plain tokens
    assert verify_summary("Value 44.2%.", _ev([{"v": 44}]))[1]  # token precision required
    assert verify_summary("Started in 2019.", ev)[1] == []  # year from a date string
    assert verify_summary("Phase 2 dominates.", ev)[1] == []  # digits inside a label are masked
    assert verify_summary("About 3 started.", ev)[1]  # '03' in a date is not evidence
    keys_only = Evidence()
    keys_only.add_values({"2015": "x"})
    assert verify_summary("Since 2015 things grew.", keys_only)[1]  # dict keys are not evidence


def test_item5_query_and_filters_are_not_evidence(study):
    s = study(NCT)
    resp = _resp(_viz(s), "There were 777 trials.")
    resp = resp.model_copy(update={"meta": resp.meta.model_copy(update={"filters_applied": {"pageSize": "777"}})})
    out, notes = verify_response(resp, {NCT: s.raw})
    assert out.answer_summary is None and notes[0].code is NoteCode.SUMMARY_TRIMMED
