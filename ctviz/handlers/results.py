"""Results / statistical intent handlers.

risks_adverse_events, results_outcomes, survival_summary, dropout_flow, demographics,
enrollment_vs_power. Pattern (see ``cohort.phase_distribution``): select (Analysis agent, ids
only) -> compute (``ctviz.compute.*``, deterministic) -> emit cited datums -> build viz -> notes +
facts. Every number in ``visualization.data`` cites the exact JSON leaf it came from; derived values
(rates, interpolated survival, power) cite their inputs and are labelled as such.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from ctviz.agents.analysis_agent import AnalysisSelection, rule_select, select_measures, wanted_endpoints
from ctviz.agents.extractors import span_refs
from ctviz.api.schemas.citations import MAX_EXCERPT_CHARS
from ctviz.api.schemas.plan import Intent, Scope
from ctviz.api.schemas.response import Note, NoteCode
from ctviz.api.schemas.viz import (
    Approximation,
    Channel,
    Scale,
    TableColumn,
    Threshold,
    Truncation,
    VizMeta,
    VizType,
)
from ctviz.compute import adverse_events as ae
from ctviz.compute import flow as fl
from ctviz.compute import power as pw
from ctviz.compute import survival as sv
from ctviz.compute.results import (
    OM_PATH,
    analyses_for,
    forest_estimates,
    has_results_section,
    measure_catalog,
    measure_cells,
    parse_int,
    parse_num,
    short_endpoint,
)
from ctviz.core.provenance import Dataset, FieldRef, GraphDataset, Study
from ctviz.obs.trace import current_trace
from ctviz.viz.base import HandlerContext, HandlerResult, note, register
from ctviz.viz.builders.bar_chart import bar_chart, ordered
from ctviz.viz.builders.forest_plot import forest_plot
from ctviz.viz.builders.kpi_cards import kpi_cards
from ctviz.viz.builders.power_curve import power_curve
from ctviz.viz.builders.sankey import sankey
from ctviz.viz.builders.scatter_plot import scatter_plot
from ctviz.viz.builders.survival_curve import survival_curve
from ctviz.viz.builders.table import table
from ctviz.viz.builders.threshold_line import threshold_line

log = logging.getLogger(__name__)
TOP_AE_TERMS = 15
MAX_COHORT_TRIALS = 10
BASELINE_PATH = "resultsSection.baselineCharacteristicsModule"
NOT_IN_REGISTRY_TTE = "patient-level time-to-event data are not in the registry"


# -- small shared helpers --------------------------------------------------------------------------


def _trial(ctx: HandlerContext) -> Study | None:
    return ctx.trials[0] if ctx.trials else None


def _name(s: Study) -> str:
    return s.get("acronym") or s.nct_id


def _clean(values: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None}


def _refs(*refs: FieldRef | None) -> list[FieldRef]:
    return [r for r in refs if r is not None]


def _pct(x: float) -> float:
    return round(100.0 * x, 1)


def _no_trial(title: str) -> HandlerResult:
    ds = Dataset()
    ds.emit({"label": "Trial", "value": "No trial record was available for this question"}, [])
    return HandlerResult(
        visualization=kpi_cards(ds.rows, title=title),
        studies_analyzed=0,
        notes=[note(NoteCode.TRIAL_REQUIRED, "This question needs a specific trial (NCT id).", "warning")],
    )


def _no_results(s: Study, title: str, intent_label: str) -> HandlerResult:
    """KPI cards from the registered protocol + NO_RESULTS_POSTED."""
    ds = Dataset()
    ds.emit({"label": "Overall status", "value": s.get("status") or "Unknown"}, _refs(s.ref("status")))
    for i, o, p in list(s.items("primary_outcomes"))[:3]:
        if isinstance(o, dict) and o.get("measure"):
            ds.emit(
                _clean({"label": f"Primary outcome {i + 1}", "value": o["measure"], "detail": o.get("timeFrame")}),
                _refs(s.ref(f"{p}.measure"), s.ref(f"{p}.timeFrame")),
            )
    enr = s.get("enrollment_count")
    if enr is not None:
        ds.emit(
            {"label": "Enrollment", "value": str(enr), "unit": "participants", "detail": s.get("enrollment_type")},
            _refs(s.ref("enrollment_count"), s.ref("enrollment_type")),
        )
    viz = kpi_cards(ds.rows, title=title, subtitle=f"{_name(s)}: no results posted")
    return HandlerResult(
        visualization=viz,
        full_refs=ds.full_refs,
        studies_analyzed=1,
        studies_with_results=0,
        notes=[
            note(
                NoteCode.NO_RESULTS_POSTED,
                f"{s.nct_id} has no results posted on ClinicalTrials.gov, so {intent_label} "
                "cannot be shown; registered protocol details are shown instead.",
                "warning",
            )
        ],
        facts={"nct_id": s.nct_id, "has_results": False},
    )


MAX_SECONDARY_ROWS = 15
OUTCOME_ROW_KEYS = (("primary_outcomes", "Primary"), ("secondary_outcomes", "Secondary"))
_OUTCOME_COLUMNS = [
    TableColumn(field="type", title="Type"),
    TableColumn(field="measure", title="Outcome measure"),
    TableColumn(field="time_frame", title="Time frame"),
]


def _whole_refs(s: Study, path: str, label: str | None = None) -> list[FieldRef]:
    """Refs that together cover the whole field value (several <=300-char spans for long strings)."""
    value = s.get(path)
    if isinstance(value, str) and len(value) > MAX_EXCERPT_CHARS:
        return span_refs(s, path, 0, len(value), label=label)
    ref = s.ref(path, label=label)
    return [ref] if ref else []


def _short_description(s: Study, path: str) -> tuple[str, FieldRef] | None:
    """The description if <=300 chars, else its verbatim prefix cut at a sentence/word end (marked "…")."""
    value = s.get(path)
    if not isinstance(value, str) or not value.strip():
        return None
    if len(value) <= MAX_EXCERPT_CHARS:
        ref = s.ref(path, label="description")
        return (value, ref) if ref else None
    head = value[:MAX_EXCERPT_CHARS]
    sentence, word = head.rfind(". ") + 1, head.rfind(" ")
    cut = sentence if sentence >= 100 else word if word >= 100 else MAX_EXCERPT_CHARS
    ref = s.text_ref(path, 0, cut, label="description")
    return (value[:cut].rstrip() + " …", ref) if ref else None


def _registered_outcomes(s: Study, title: str, intent_label: str) -> HandlerResult:
    """Table of the registered primary/secondary outcomes (registry order) + NO_RESULTS_POSTED.

    One row per outcome (secondary capped at ``MAX_SECONDARY_ROWS``, TRUNCATED note), each cell cited to its
    exact ``outcomesModule`` field. Status and enrollment are display context in the subtitle, not rows.
    """
    ds = Dataset()
    counts = {"Primary": 0, "Secondary": 0}
    omitted = 0
    for key, otype in OUTCOME_ROW_KEYS:
        for _, o, p in s.items(key):
            if not isinstance(o, dict) or not o.get("measure"):
                continue
            if otype == "Secondary" and counts[otype] >= MAX_SECONDARY_ROWS:
                omitted += 1
                continue
            refs: list[FieldRef | None] = [*_whole_refs(s, f"{p}.measure", label=f"{otype} outcome")]
            refs += _whole_refs(s, f"{p}.timeFrame", label="time frame")
            row: dict[str, Any] = {
                "type": otype,
                "measure": o["measure"],
                "time_frame": o.get("timeFrame") or "Not stated",
            }
            desc = _short_description(s, f"{p}.description")
            if desc is not None:
                row["description"] = desc[0]
                refs.append(desc[1])
            ds.emit(row, refs, keep_order=True)
            counts[otype] += 1
    if not ds.rows:
        return _no_results(s, title, intent_label)
    notes = [
        note(
            NoteCode.NO_RESULTS_POSTED,
            f"{s.nct_id} has no results posted on ClinicalTrials.gov, so {intent_label} "
            "cannot be shown; the registered outcome measures are listed instead.",
            "warning",
        )
    ]
    meta = VizMeta()
    if omitted:
        meta.truncation = Truncation(top_n=MAX_SECONDARY_ROWS, omitted_count=omitted)
        notes.append(
            note(
                NoteCode.TRUNCATED,
                f"Only the first {MAX_SECONDARY_ROWS} of {MAX_SECONDARY_ROWS + omitted} secondary outcomes are listed.",
            )
        )
    status, enr = s.get("status"), s.get("enrollment_count")
    sub = [
        f"{_name(s)}: no results posted",
        f"status {status.replace('_', ' ').capitalize()}" if status else None,
        f"enrollment {enr} ({(s.get('enrollment_type') or '').lower() or 'type not stated'})"
        if enr is not None
        else None,
    ]
    viz = table(
        ds.rows,
        title=title,
        columns=_OUTCOME_COLUMNS,
        subtitle=" · ".join(x for x in sub if x),
        meta=meta,
        tooltip=["description"] if any("description" in (r.model_extra or {}) for r in ds.rows) else None,
    )
    return HandlerResult(
        visualization=viz,
        full_refs=ds.full_refs,
        studies_analyzed=1,
        studies_with_results=0,
        notes=notes,
        facts={
            "nct_id": s.nct_id,
            "has_results": False,
            "primary_outcomes_derived": counts["Primary"],
            "secondary_outcomes_derived": counts["Secondary"] + omitted,
        },
    )


async def _select(ctx: HandlerContext, s: Study, purpose: str) -> AnalysisSelection:
    return await select_measures(s, ctx.request.query, purpose, llm=ctx.llm, settings=ctx.settings)  # type: ignore[arg-type]


def _selection_meta(sel: AnalysisSelection) -> dict[str, Any]:
    out: dict[str, Any] = {"method": sel.method, "measure_indices": sel.measure_indices}
    if sel.group_ids:
        out["group_ids"] = sel.group_ids
    if sel.rejected:
        out["llm_rejected"] = sel.rejected
    return out


def _diagnostics(sel: AnalysisSelection | None) -> dict[str, Any]:
    """``meta.diagnostics``: which measures / groups the Analysis agent (or its rule fallback) selected."""
    return {"analysis": _selection_meta(sel)} if sel else {}


def _cohort_mode(ctx: HandlerContext) -> bool:
    return ctx.plan.scope is not Scope.SINGLE_TRIAL or len(ctx.trials) > 1


async def _cohort_trials_with_results(ctx: HandlerContext, limit: int = MAX_COHORT_TRIALS) -> list[Study]:
    """Full records with results for cohort scope: ctx.trials, or fetched via ctx.client when available."""
    full = [t for t in ctx.trials if has_results_section(t)]
    if full or ctx.client is None:
        return full[:limit]
    ids = [s.nct_id for s in ctx.studies if s.raw.get("hasResults")][:limit]
    if not ids:
        return []
    try:
        from ctviz.ctgov.retrieval import fetch_trials  # built by another builder
    except ImportError:  # pragma: no cover - retrieval not available yet
        return []
    with current_trace().step("results.fetch_trials", n=len(ids)):
        fetched = await fetch_trials(ctx.client, ids)
    return [t for t in fetched if has_results_section(t)]


# == risks_adverse_events ==========================================================================


@register(Intent.RISKS_ADVERSE_EVENTS)
async def risks_adverse_events(ctx: HandlerContext) -> HandlerResult:
    """AE rates per arm (bar), dose-escalation DLT rates (threshold_line), or risk statements."""
    if _cohort_mode(ctx):
        return await _ae_cohort(ctx)
    s = _trial(ctx)
    if s is None:
        return _no_trial("Adverse events")
    if not has_results_section(s) or not s.get("adverse_events"):
        return _risk_statements(s)
    if ae.is_dose_escalation(s):
        return _dose_escalation(s)
    groups = ae.event_groups(s)
    sel: AnalysisSelection | None = None
    if len(groups) > 4 and ctx.llm is not None:
        sel = await _select(ctx, s, "ae")
    keep = set(sel.group_ids) if sel and sel.group_ids else {g.id for g in groups}
    with current_trace().step("analysis.ae_rates", tool="ae_rates", nct_id=s.nct_id) as info:
        rates = [r for r in ae.ae_rates(s) if r.group_id in keep]
        keys, omitted = ae.top_terms(rates, TOP_AE_TERMS, keep)
        info.update(rows=len(rates), terms=len(keys), omitted=omitted)
    by_key: dict[tuple[str, str], list[ae.AERate]] = defaultdict(list)
    for r in rates:
        by_key[r.key].append(r)
    order = (
        [("Any serious adverse event", "any_serious")]
        if by_key.get(("Any serious adverse event", "any_serious"))
        else []
    )
    order += keys
    ds = Dataset()
    labels: list[str] = []
    arm_order = [g.title for g in groups if g.id in keep]
    for key in order:
        label = key[0] if key[1] != "serious" else f"{key[0]} (serious)"
        labels.append(label)
        for r in sorted(
            by_key[key], key=lambda r: arm_order.index(r.group_title) if r.group_title in arm_order else 99
        ):
            ds.emit(
                {
                    "term": label,
                    "arm": r.group_title,
                    "rate": round(r.rate, 4),
                    "n_affected": r.n_affected,
                    "n_at_risk": r.n_at_risk,
                    "organ_system": r.organ_system,
                    "seriousness": r.seriousness,
                },
                r.refs,
            )
    thr, thr_ref = ae.frequency_threshold(s)
    meta = VizMeta(
        grouping={"by": ["term", "arm"], "rate": "numAffected / numAtRisk"},
        diagnostics=_diagnostics(sel),
    )
    notes: list[Note] = []
    if thr is not None and thr_ref is not None:
        meta.thresholds.append(
            Threshold(
                axis="y",
                value=thr,
                source="registry",
                label=f"Registry reporting threshold for non-serious AEs ({thr_ref.excerpt}%)",
                citations=[thr_ref.to_citation()],
            )
        )
        notes.append(
            note(
                NoteCode.NOT_IN_REGISTRY,
                f"Non-serious adverse events below the {thr_ref.excerpt}% frequency threshold were not "
                "required to be reported, so rarer events may be missing.",
                "info",
            )
        )
    if omitted:
        meta.truncation = Truncation(top_n=TOP_AE_TERMS, omitted_count=omitted)
        notes.append(
            note(
                NoteCode.TRUNCATED,
                f"Showing the {len(keys)} most frequent event terms "
                f"(by highest arm rate); {omitted} more terms are not shown.",
            )
        )
    x = Channel(field="term", type="nominal", title="Adverse event", sort=ordered(labels))
    y = Channel(field="rate", type="quantitative", title="Participants affected", unit="%", format=".1%")
    series = Channel(field="arm", type="nominal", title="Arm")
    viz = bar_chart(
        ds.rows,
        title=f"Adverse events by arm: {_name(s)}",
        x=x,
        y=y,
        series=series,
        orientation="horizontal",
        meta=meta,
        tooltip=["n_affected", "n_at_risk", "organ_system", "seriousness"],
    )
    facts: dict[str, Any] = {"nct_id": s.nct_id}
    for d in ds.rows[: 3 * max(1, len(arm_order))]:
        e = d.model_extra or {}
        facts[f"{e['term']}|{e['arm']}"] = f"{e['n_affected']}/{e['n_at_risk']} ({_pct(e['rate'])}%)"
    if thr_ref is not None:
        facts["frequency_threshold_pct"] = thr_ref.excerpt
    return HandlerResult(
        visualization=viz, notes=notes, facts=facts, full_refs=ds.full_refs, studies_analyzed=1, studies_with_results=1
    )


def _dose_escalation(s: Study) -> HandlerResult:
    """DLT rate (or serious-AE rate) per dose group, ordered by dose, with the 33% convention line."""
    ds = Dataset()
    notes: list[Note] = []
    dlt = next((m for m in measure_catalog(s) if ae.DLT_TITLE.search(m.title)), None)
    rows: list[tuple[str, float, int | None, int, list[FieldRef]]] = []
    metric = "DLT rate"
    dlt_mode = None
    if dlt is not None:
        unit_l = dlt.unit.lower()
        if "percent" in unit_l or "%" in unit_l:
            dlt_mode = "percent"
        elif dlt.param_type.upper() == "COUNT_OF_PARTICIPANTS" or re.search(r"participant|patient|subject", unit_l):
            dlt_mode = "count"
        else:
            notes.append(
                note(
                    NoteCode.NOT_IN_REGISTRY,
                    "The dose-limiting toxicity outcome is reported in a unit that is neither a participant "
                    "count nor a percentage, so it is not plotted.",
                )
            )
    if dlt is not None and dlt_mode is not None:
        base = f"{OM_PATH}[{dlt.index}]"
        dropped_cells: list[str] = []
        denoms: dict[str, tuple[int, str]] = {}
        for ci, c in enumerate(s.get(f"{base}.denoms[0].counts") or []):
            n = parse_int(c.get("value")) if isinstance(c, dict) else None
            if n:
                denoms[c.get("groupId")] = (n, f"{base}.denoms[0].counts[{ci}].value")
        for cell in measure_cells(s, dlt.index):
            if cell.value is None or cell.group_id not in denoms or cell.class_title or cell.category_title:
                continue
            n, npath = denoms[cell.group_id]
            if dlt_mode == "count":
                if cell.qualified or abs(cell.value - round(cell.value)) > 1e-9 or not 0 <= cell.value <= n:
                    dropped_cells.append(cell.group_title)
                    continue
                k: int | None = int(round(cell.value))
                rate = k / n
            else:
                if cell.qualified or not 0 <= cell.value <= 100:
                    dropped_cells.append(cell.group_title)
                    continue
                k, rate = None, cell.value / 100.0
            rows.append(
                (
                    cell.group_title,
                    rate,
                    k,
                    n,
                    _refs(
                        s.ref(
                            cell.value_path,
                            label=f"{cell.group_title}: "
                            + ("participants with DLT" if dlt_mode == "count" else "% with DLT"),
                        ),
                        s.ref(npath, label=f"{cell.group_title}: participants analyzed"),
                        s.ref(f"{base}.title"),
                        s.ref(cell.group_path) if cell.group_path else None,
                    ),
                )
            )
        if dropped_cells:
            notes.append(
                note(
                    NoteCode.NOT_IN_REGISTRY,
                    f"{len(dropped_cells)} dose group value(s) were not whole counts / valid percentages "
                    "and are not plotted.",
                )
            )
    if not rows:
        metric = "Serious adverse event rate"
        for r in ae.ae_rates_any_serious(s):
            rows.append((r.group_title, r.rate, r.n_affected, r.n_at_risk, r.refs))
        notes.append(
            note(
                NoteCode.NOT_IN_REGISTRY,
                "No dose-limiting toxicity outcome is reported; serious "
                "adverse event rates per dose group are shown instead.",
            )
        )
    order = ae.dose_order([r[0] for r in rows])
    labels = []
    for rank, i in enumerate(order):
        title, rate, k, n, refs = rows[i]
        labels.append(title)
        ds.emit(
            _clean(
                {
                    "group": title,
                    "dose_order": rank + 1,
                    "rate": round(rate, 4),
                    "n_events": k,
                    "n_evaluable": n,
                    "dose": ae.parse_dose(title),
                }
            ),
            refs,
        )
    thr = Threshold(
        axis="y",
        value=ae.DLT_CONVENTION,
        source="convention",
        label="33% DLT rate (3+3 design convention; not from the registry)",
    )
    x = Channel(field="group", type="ordinal", title="Dose group", sort=ordered(labels))
    y = Channel(field="rate", type="quantitative", title=metric, unit="%", format=".0%", scale=Scale(domain=[0, 1]))
    meta = VizMeta(
        grouping={
            "order": "dose parsed from group title, then registry order",
            "rate": "participants with event / participants analyzed",
        }
    )
    viz = threshold_line(
        ds.rows,
        title=f"{metric} by dose group: {_name(s)}",
        x=x,
        y=y,
        thresholds=[thr],
        meta=meta,
        tooltip=["n_events", "n_evaluable"],
    )
    if not any(ae.parse_dose(t) is not None for t in labels):
        notes.append(
            note(
                NoteCode.APPROXIMATED,
                "Group titles carry no numeric dose; groups are shown in registry order rather than by dose.",
            )
        )
    facts = {
        str((d.model_extra or {})["group"]): (
            f"{(d.model_extra or {})['n_events']}/{(d.model_extra or {})['n_evaluable']}"
            if "n_events" in (d.model_extra or {})
            else f"{_pct((d.model_extra or {})['rate'])}%"
        )
        for d in ds.rows
    }
    return HandlerResult(
        visualization=viz,
        notes=notes,
        facts={"nct_id": s.nct_id, **facts},
        full_refs=ds.full_refs,
        studies_analyzed=1,
        studies_with_results=1,
    )


def _risk_statements(s: Study) -> HandlerResult:
    ds = Dataset()
    for st in ae.risk_statements(s):
        ds.emit({"source": st.source, "statement": st.text}, [st.ref])
    cols = [TableColumn(field="statement", title="Statement (verbatim)"), TableColumn(field="source", title="Source")]
    viz = table(
        ds.rows,
        title=f"Known risks (protocol text): {_name(s)}",
        columns=cols,
        subtitle="No adverse-event results are posted; statements are quoted from the registry record",
    )
    notes = [
        note(
            NoteCode.NO_RESULTS_POSTED,
            f"{s.nct_id} has no adverse-event results posted; risk-related "
            "statements from the description and exclusion criteria are shown.",
            "warning",
        )
    ]
    return HandlerResult(
        visualization=viz,
        notes=notes,
        full_refs=ds.full_refs,
        studies_analyzed=1,
        studies_with_results=0,
        facts={"nct_id": s.nct_id, "statements": len(ds.rows)},
    )


async def _ae_cohort(ctx: HandlerContext) -> HandlerResult:
    trials = await _cohort_trials_with_results(ctx)
    ds = Dataset()
    pooled = []
    for t in trials:
        r = ae.serious_rate_pooled(t)
        if r is not None:
            pooled.append((t, r))
    pooled.sort(key=lambda tr: -tr[1].rate)
    for t, r in pooled:
        ds.emit(
            {
                "trial": f"{_name(t)} ({t.nct_id})" if _name(t) != t.nct_id else t.nct_id,
                "nct_id": t.nct_id,
                "serious_rate": round(r.rate, 4),
                "n_affected": r.n_affected,
                "n_at_risk": r.n_at_risk,
            },
            r.refs,
        )
    notes = [
        note(
            NoteCode.APPROXIMATED,
            "Serious adverse event rates are pooled across all reported event groups "
            "of each trial (participants in overlapping groups may be counted twice).",
        )
    ]
    if not ds.rows:
        ds.emit({"label": "Trials with posted adverse-event results", "value": 0}, [])
        notes = [
            note(
                NoteCode.NO_RESULTS_POSTED,
                "None of the analysed trials has adverse-event results available.",
                "warning",
            )
        ]
        return HandlerResult(
            visualization=kpi_cards(ds.rows, title="Serious adverse events across trials"),
            notes=notes,
            studies_analyzed=len(ctx.studies),
            studies_with_results=0,
        )
    if len(ctx.studies) > len(trials):
        notes.append(
            note(
                NoteCode.SAMPLED,
                f"Rates shown for {len(ds.rows)} trials with posted results (of {len(ctx.studies)} analysed).",
            )
        )
    x = Channel(
        field="trial", type="nominal", title="Trial", sort=ordered([(d.model_extra or {})["trial"] for d in ds.rows])
    )
    y = Channel(
        field="serious_rate", type="quantitative", title="Participants with a serious AE", unit="%", format=".1%"
    )
    viz = bar_chart(
        ds.rows,
        title="Serious adverse event rate per trial",
        x=x,
        y=y,
        meta=VizMeta(grouping={"by": ["trial"], "rate": "sum seriousNumAffected / sum seriousNumAtRisk"}),
        tooltip=["n_affected", "n_at_risk"],
    )
    facts = {(d.model_extra or {})["nct_id"]: f"{_pct((d.model_extra or {})['serious_rate'])}%" for d in ds.rows}
    return HandlerResult(
        visualization=viz,
        notes=notes,
        facts=facts,
        full_refs=ds.full_refs,
        studies_analyzed=len(ctx.studies) or len(trials),
        studies_with_results=len(trials),
    )


# == results_outcomes ==============================================================================


@register(Intent.RESULTS_OUTCOMES)
async def results_outcomes(ctx: HandlerContext) -> HandlerResult:
    """Forest plot of effect estimates, else grouped bars of the selected measures (several trials: forest)."""
    if _multi_trial(ctx):
        return await _results_multi(ctx)
    s = _trial(ctx)
    if s is None:
        return _no_trial("Results")
    if not has_results_section(s) or not s.get("outcome_measures"):
        return _registered_outcomes(s, f"No results posted for {_name(s)}: registered outcomes", "outcome results")
    sel = await _select(ctx, s, "results")
    with current_trace().step("analysis.forest_rows", tool="forest_rows", measures=sel.measure_indices) as info:
        # The first selected measure decides the chart: forest plot only if it has effect estimates.
        lead = sel.measure_indices[:1]
        ests = forest_estimates(s, sel.measure_indices, prefer=None) if lead and analyses_for(s, lead[0]) else []
        info["rows"] = len(ests)
    if ests:
        return _forest_result(s, ests, sel, title=f"Effect estimates: {_name(s)}")
    return _measure_bars(s, sel)


def _forest_rows(s: Study, ests: list) -> Dataset:
    ds = Dataset()
    seen: set[str] = set()
    for e in ests:
        label = e.label
        if label in seen:
            label = f"{label} [analysis {e.analysis_index + 1}]"
        seen.add(label)
        ds.emit(
            _clean(
                {
                    "label": label,
                    "estimate": e.estimate,
                    "ci_low": e.ci_low,
                    "ci_high": e.ci_high,
                    "ci_pct": e.ci_pct,
                    "p_value": e.p_raw,
                    "param_type": e.param_type,
                    "null_value": e.null_value,
                    "method": e.method,
                    "measure": e.measure_title,
                }
            ),
            e.refs,
        )
    return ds


def _forest_result(
    s: Study, ests: list, sel: AnalysisSelection | None, title: str, notes: list[Note] | None = None
) -> HandlerResult:
    ds = _forest_rows(s, ests)
    kind = ests[0].kind
    est_title = (
        ests[0].param_type
        if len({e.param_type for e in ests}) == 1
        else ("Ratio estimate" if kind == "ratio" else "Difference estimate")
    )
    meta = VizMeta(diagnostics=_diagnostics(sel))  # forest_plot sets meta.reference_line from ``kind``
    viz = forest_plot(ds.rows, title=title, kind=kind, estimate_title=est_title, meta=meta)
    facts: dict[str, Any] = {"nct_id": s.nct_id}
    for e in ests[:8]:
        facts[e.label] = f"{e.param_type} {e.estimate:g} ({e.ci_low:g} to {e.ci_high:g}), p {e.p_raw or 'NR'}"
    return HandlerResult(
        visualization=viz,
        notes=notes or [],
        facts=facts,
        full_refs=ds.full_refs,
        studies_analyzed=1,
        studies_with_results=1,
    )


def _measure_bars(s: Study, sel: AnalysisSelection) -> HandlerResult:
    cat = {m.index: m for m in measure_catalog(s)}
    chosen = [i for i in sel.measure_indices if i in cat]
    if not chosen:
        return _registered_outcomes(s, f"No results posted for {_name(s)}: registered outcomes", "outcome results")
    unit = cat[chosen[0]].unit
    used = [i for i in chosen if cat[i].unit == unit]
    ds = Dataset()
    rows: list[tuple[dict[str, Any], list[FieldRef]]] = []
    for i in used:
        m = cat[i]
        disp = m.dispersion.lower()
        for c in measure_cells(s, i):
            if c.value is None:
                continue
            lo = hi = None
            refs = _refs(s.ref(c.value_path, label=f"{c.group_title}: value"))
            if "confidence" in disp or "range" in disp or "inter-quartile" in disp:
                lo, hi = c.lower, c.upper
                refs += _refs(
                    s.ref(c.lower_path) if c.lower_path else None, s.ref(c.upper_path) if c.upper_path else None
                )
            elif c.spread is not None and ("deviation" in disp or "error" in disp):
                lo, hi = c.value - c.spread, c.value + c.spread
                refs += _refs(s.ref(c.spread_path) if c.spread_path else None)
            refs += _refs(
                s.ref(f"{m.path}.title"),
                s.ref(c.group_path) if c.group_path else None,
                s.ref(c.class_path) if c.class_path else None,
                s.ref(c.category_path) if c.category_path else None,
            )
            category = m.title if len(used) > 1 or not c.label else c.label
            if len(used) > 1 and c.label:
                category = f"{m.title} · {c.label}"
            rows.append(
                (
                    _clean(
                        {
                            "category": category,
                            "group": c.group_title,
                            "value": c.value,
                            "error_low": round(lo, 4) if lo is not None else None,
                            "error_high": round(hi, 4) if hi is not None else None,
                            "measure": m.title,
                            "unit": unit,
                            "dispersion": m.dispersion or None,
                        }
                    ),
                    refs,
                )
            )
    with_err = bool(rows) and all("error_low" in r and "error_high" in r for r, _ in rows)
    for values, refs in rows:
        if not with_err:
            values.pop("error_low", None)
            values.pop("error_high", None)
        ds.emit(values, refs)
    notes: list[Note] = []
    if len(used) < len(chosen):
        notes.append(
            note(
                NoteCode.TRUNCATED,
                f"Only measures sharing the first selected measure's unit ({unit}) are shown together.",
            )
        )
    cats = list(dict.fromkeys((d.model_extra or {})["category"] for d in ds.rows))
    x = Channel(field="category", type="nominal", title="Outcome", sort=ordered(cats))
    y = Channel(field="value", type="quantitative", title=unit or "Value", unit=unit)
    series = Channel(field="group", type="nominal", title="Arm")
    meta = VizMeta(diagnostics=_diagnostics(sel))
    viz = bar_chart(
        ds.rows,
        title=f"Outcome results: {_name(s)}",
        x=x,
        y=y,
        series=series,
        meta=meta,
        tooltip=["measure", "dispersion"],
    )
    if with_err:
        viz.encoding.error_low = Channel(field="error_low", type="quantitative", title="Lower")
        viz.encoding.error_high = Channel(field="error_high", type="quantitative", title="Upper")
    facts = {
        f"{(d.model_extra or {})['category']}|{(d.model_extra or {})['group']}": (d.model_extra or {})["value"]
        for d in ds.rows[:12]
    }
    return HandlerResult(
        visualization=viz,
        notes=notes,
        facts={"nct_id": s.nct_id, **facts},
        full_refs=ds.full_refs,
        studies_analyzed=1,
        studies_with_results=1,
    )


# == survival_summary ==============================================================================


@register(Intent.SURVIVAL_SUMMARY)
async def survival_summary(ctx: HandlerContext) -> HandlerResult:
    """Approximate survival curves from reported medians / landmark rates (+ HRs in meta)."""
    if _multi_trial(ctx):
        return await _survival_multi(ctx)
    s = _trial(ctx)
    if s is None:
        return _no_trial("Survival")
    if not has_results_section(s) or not s.get("outcome_measures"):
        return _registered_outcomes(s, f"No results posted for {_name(s)}: registered outcomes", "survival results")
    sel = await _select(ctx, s, "survival")
    with current_trace().step(
        "analysis.reconstruct_survival", tool="reconstruct_survival", measures=sel.measure_indices
    ) as info:
        ext = sv.extract_anchors(s, sel.measure_indices)
        start_refs = {
            series: [r for r in pts[0].refs if r.field_path.endswith(".title")] for series, pts in ext.anchors.items()
        }
        points = sv.reconstruct_survival(ext.anchors, start_refs)
        bad = sv.curve_problems(points)  # never draw a curve that rises or does not start at S(0) = 1
        points = [p for p in points if p.series not in bad]
        info.update(series=len(ext.anchors), points=len(points), invalid=list(bad))
    hr = [e for e in forest_estimates(s, sel.measure_indices) if e.kind == "ratio"]
    if not hr:
        surv_idx = [m.index for m in measure_catalog(s) if sv.is_survival_measure(m.title)]
        hr = [e for e in forest_estimates(s, surv_idx) if e.kind == "ratio" and not e.description][:6]
    notes: list[Note] = [
        note(NoteCode.APPROXIMATED, f"{series}: not drawn, the reported values {why}.", "warning")
        for series, why in bad.items()
    ]
    if ext.skipped:
        notes.append(
            note(
                NoteCode.NOT_IN_REGISTRY,
                "Some reported values were not used because their meaning as a survival point was unclear: "
                f"{'; '.join(ext.skipped[:6])}.",
            )
        )
    if ext.converted:
        notes.append(
            note(
                NoteCode.APPROXIMATED,
                "Event rates were converted to survival as S = 1 - rate: " + ", ".join(ext.converted[:6]) + ".",
            )
        )
    if not points:
        notes.append(
            note(
                NoteCode.NOT_IN_REGISTRY,
                f"No reported medians or landmark rates to anchor a curve; {NOT_IN_REGISTRY_TTE}.",
                "warning",
            )
        )
        if hr:
            return _forest_result(s, hr, sel, title=f"Survival hazard ratios: {_name(s)}", notes=notes)
        return _survival_table(s, notes)
    ds = Dataset()
    emit_curve_points(
        ds,
        points,
        lambda p: {
            "series": p.series,
            "time": p.time,
            "survival": p.survival,
            "source": p.source,
            "endpoint": p.endpoint,
            "group": p.group,
            "ci_low": _r(p.ci_low),
            "ci_high": _r(p.ci_high),
            "time_ci_low": _r(p.time_ci_low),
            "time_ci_high": _r(p.time_ci_high),
            "label": _point_label(p),
        },
    )
    meta = VizMeta(
        approximation=Approximation(is_approximation=True, method=sv.METHOD, assumptions=sv.ASSUMPTIONS),
        stats={"analyses": [e.as_meta() for e in hr]} if hr else {},
        diagnostics=_diagnostics(sel),
    )
    viz = survival_curve(
        ds.rows,
        title=f"Approximate survival: {_name(s)}",
        subtitle="Reconstructed from reported medians/landmark rates, not patient-level data",
        meta=meta,
    )
    notes.append(
        note(
            NoteCode.APPROXIMATED,
            "Curves are approximations through reported medians and landmark "
            f"rates; {NOT_IN_REGISTRY_TTE}. Interpolated points are marked "
            "source=interpolated.",
            "warning",
            affects=[d.id for d in ds.rows if (d.model_extra or {}).get("source") == "interpolated"] or ["*"],
        )
    )
    for series in ext.not_reached:
        notes.append(
            note(
                NoteCode.NOT_IN_REGISTRY,
                f"{series}: median not reached at the time of analysis; no curve can be anchored for this arm.",
            )
        )
    if ext.dropped:
        notes.append(
            note(
                NoteCode.APPROXIMATED,
                "Some reported points were inconsistent with a non-increasing "
                f"curve and were not used: {', '.join(ext.dropped[:5])}.",
            )
        )
    facts: dict[str, Any] = {"nct_id": s.nct_id}
    drawn = {p.series for p in points}
    medians: dict[str, float] = {}
    for series, pts in ext.anchors.items():
        if series not in drawn:
            continue
        for a in pts:
            facts[f"{series}|{a.kind}|{a.time:g}"] = round(a.survival, 4)
            if a.kind == "median":
                medians[series] = round(a.time, 2)
    if medians:
        facts["medians_months_derived"] = medians  # read by the survival summary template
    for e in hr:
        facts[e.label] = f"{e.param_type} {e.estimate:g} ({e.ci_low:g} to {e.ci_high:g})"
    return HandlerResult(
        visualization=viz, notes=notes, facts=facts, full_refs=ds.full_refs, studies_analyzed=1, studies_with_results=1
    )


def _r(x: float | None) -> float | None:
    return round(x, 4) if x is not None else None


def emit_curve_points(
    ds: Dataset, points: list[sv.CurvePoint], values_of: Callable[[sv.CurvePoint], dict[str, Any]]
) -> list[str]:
    """Emit curve points in order; interpolated points get ``derived_from`` = their two anchors' datum ids.

    Reported points (S(0) = 1, medians, landmark rates) carry the anchor's citations. Interpolated
    points carry the single citation ``reconstruct_survival`` gave them (the measure title) plus
    ``derived_from``, so a reader follows the link to the reported values instead of seeing them
    repeated on every point. Returns the ids of the interpolated datums.
    """
    ids = [ds.next_id() for _ in points]
    anchor_id: dict[tuple[str, int], str] = {
        (p.series, p.anchor_pos): did for p, did in zip(points, ids, strict=True) if p.anchor_pos is not None
    }
    interpolated: list[str] = []
    for p, did in zip(points, ids, strict=True):
        values = _clean(values_of(p))
        if p.source == "interpolated" and p.between is not None:
            values["derived_from"] = [anchor_id[(p.series, pos)] for pos in p.between if (p.series, pos) in anchor_id]
            interpolated.append(did)
        ds.emit(values, p.refs, datum_id=did)
    return interpolated


def _survival_table(s: Study, notes: list[Note]) -> HandlerResult:
    ds = Dataset()
    for m in measure_catalog(s):
        if sv.is_survival_measure(m.title):
            ds.emit(
                {"measure": m.title, "param_type": m.param_type, "unit": m.unit, "type": m.type},
                _refs(s.ref(f"{m.path}.title"), s.ref(f"{m.path}.paramType")),
            )
    if not ds.rows:
        ds.emit({"measure": "No survival outcome measures reported", "param_type": "", "unit": "", "type": ""}, [])
    cols = [
        TableColumn(field="measure", title="Outcome measure"),
        TableColumn(field="param_type", title="Parameter"),
        TableColumn(field="unit", title="Unit"),
        TableColumn(field="type", title="Type"),
    ]
    viz = table(ds.rows, title=f"Survival outcomes reported: {_name(s)}", columns=cols)
    return HandlerResult(
        visualization=viz,
        notes=notes,
        full_refs=ds.full_refs,
        studies_analyzed=1,
        studies_with_results=1,
        facts={"nct_id": s.nct_id},
    )


# == dropout_flow ==================================================================================


@register(Intent.DROPOUT_FLOW)
async def dropout_flow(ctx: HandlerContext) -> HandlerResult:
    """Sankey: per arm Started -> Completed / Not completed: <reason> (+ explicit unexplained remainder)."""
    if _cohort_mode(ctx):
        return await _dropout_cohort(ctx)
    s = _trial(ctx)
    if s is None:
        return _no_trial("Participant flow")
    arms = fl.flow_table(s) if has_results_section(s) else []
    if not arms:
        title = f"Participant flow: {_name(s)}" if has_results_section(s) else f"No results posted for {_name(s)}"
        return _no_results(s, title, "participant flow")
    sel: AnalysisSelection | None = None
    if len(arms) > 4 and ctx.llm is not None:
        sel = await _select(ctx, s, "flow")
        if sel.group_ids:
            arms = [a for a in arms if a.group_id in set(sel.group_ids)] or arms
    periods = fl.period_titles(arms)
    multi = len(periods) > 1
    stages = ["Started"] + ([f"End of {p}" for p in periods] if multi else ["End of study"])
    g = GraphDataset()
    notes: list[Note] = []
    unexplained_ids: list[str] = []
    dropout_rate: dict[str, float] = {}
    facts: dict[str, Any] = {"nct_id": s.nct_id}
    for arm in arms:
        base_started = next((p.started for p in arm.periods if p.started > 0), 0)
        if base_started <= 0:
            notes.append(
                note(NoteCode.NOT_IN_REGISTRY, f"{arm.title}: no participants reported as started; arm skipped.")
            )
            continue
        source: str | None = None  # node holding the participants who enter the next period
        prev_completed: int | None = None
        for pi, per in enumerate(arm.periods):
            stage = periods.index(per.period) + 1
            last = pi == len(arm.periods) - 1
            if per.started <= 0:
                notes.append(
                    note(NoteCode.NOT_IN_REGISTRY, f"{arm.title}: {per.period} reports 0 started; period skipped.")
                )
                source, prev_completed = None, None
                continue
            if source is not None and prev_completed != per.started:
                notes.append(
                    note(
                        NoteCode.APPROXIMATED,
                        f"{arm.title}: {per.period} started with {per.started} but the previous period "
                        f"completed {prev_completed}; the period is drawn from its own start count.",
                    )
                )
                source = None
            if source is None:
                sid = f"{arm.group_id}:start" if pi == 0 else f"{arm.group_id}:p{pi}:start"
                g.add_node(
                    sid,
                    {
                        "label": "Started" if pi == 0 else f"Started {per.period}",
                        "stage": stage - 1,
                        "group": arm.title,
                        "value": per.started,
                    },
                    [*per.started_refs, *_refs(arm.title_ref)],
                )
                source = sid

            def edge(
                target: str,
                weight: int,
                etype: str,
                refs: list[FieldRef],
                src: str = source,
                title: str = arm.title,
                denom: int = base_started,
            ) -> str:
                d = g.add_edge(
                    {
                        "source": src,
                        "target": target,
                        "weight": weight,
                        "type": etype,
                        "group": title,
                        "share_of_started": round(weight / denom, 4) if denom > 0 else 0.0,
                    },
                    refs,
                )
                return d.id

            cid: str | None = None
            if per.completed > 0:
                cid = f"{arm.group_id}:p{pi}:completed"
                g.add_node(
                    cid,
                    {
                        "label": "Completed" if last else f"Completed {per.period}",
                        "stage": stage,
                        "group": arm.title,
                        "value": per.completed,
                    },
                    per.completed_refs,
                )
                edge(cid, per.completed, "completed", per.completed_refs)
            for ri, r in enumerate(per.reasons):
                rid = f"{arm.group_id}:p{pi}:r{ri}"
                g.add_node(
                    rid,
                    {"label": f"Not completed: {r.label}", "stage": stage, "group": arm.title, "value": r.count},
                    r.refs,
                )
                edge(rid, r.count, "dropout", r.refs)
            if per.unexplained > 0:
                uid = f"{arm.group_id}:p{pi}:unexplained"
                g.add_node(
                    uid,
                    {
                        "label": "Not completed: reason not reported",
                        "stage": stage,
                        "group": arm.title,
                        "value": per.unexplained,
                    },
                    [],
                )
                eid = edge(uid, per.unexplained, "unexplained", [])
                unexplained_ids += [uid, eid]
            if per.inconsistent:
                notes.append(
                    note(
                        NoteCode.APPROXIMATED,
                        f"{arm.title}: reported withdrawal reasons exceed the number not completing {per.period}.",
                    )
                )
            source, prev_completed = cid, per.completed  # cid is None when nobody completed the period
        rate = arm.dropout_rate
        if rate is not None:
            dropout_rate[arm.title] = round(rate, 4)
            facts[f"{arm.title}|dropout_rate_pct"] = _pct(rate)
            facts[f"{arm.title}|started"] = arm.started
            facts[f"{arm.title}|completed"] = arm.completed
    if not g.nodes.rows:
        res = _no_results(s, f"Participant flow: {_name(s)}", "participant flow")
        res.notes = [*notes, *res.notes]
        return res
    if unexplained_ids:
        notes.append(
            note(
                NoteCode.NOT_IN_REGISTRY,
                "Some participants who did not complete have no reason listed "
                "in the registry; they are shown as 'reason not reported' "
                "(derived as not completed minus listed reasons, so uncited).",
                affects=unexplained_ids,
            )
        )
    meta = VizMeta(stats={"dropout_rate": dropout_rate, "periods": periods}, diagnostics=_diagnostics(sel))
    viz = sankey(g.to_graph(), title=f"Participant flow: {_name(s)}", stages=stages, meta=meta)
    return HandlerResult(
        visualization=viz, notes=notes, facts=facts, full_refs=g.full_refs, studies_analyzed=1, studies_with_results=1
    )


async def _dropout_cohort(ctx: HandlerContext) -> HandlerResult:
    trials = await _cohort_trials_with_results(ctx)
    ds = Dataset()
    for t in trials:
        arms = fl.flow_table(t)
        started = sum(a.started for a in arms)
        if not started:
            continue
        completed = sum(a.completed for a in arms)
        refs = [r for a in arms for r in (*a.periods[0].started_refs, *a.periods[-1].completed_refs)]
        ds.emit(
            {
                "trial": t.nct_id if _name(t) == t.nct_id else f"{_name(t)} ({t.nct_id})",
                "nct_id": t.nct_id,
                "dropout_rate": round((started - completed) / started, 4),
                "started": started,
                "completed": completed,
            },
            refs,
        )
    if not ds.rows:
        ds.emit({"label": "Trials with posted participant flow", "value": 0}, [])
        return HandlerResult(
            visualization=kpi_cards(ds.rows, title="Dropout across trials"),
            studies_with_results=0,
            studies_analyzed=len(ctx.studies),
            notes=[note(NoteCode.NO_RESULTS_POSTED, "No analysed trial has participant-flow results.", "warning")],
        )
    ds.rows.sort(key=lambda d: -(d.model_extra or {})["dropout_rate"])
    x = Channel(
        field="trial", type="nominal", title="Trial", sort=ordered([(d.model_extra or {})["trial"] for d in ds.rows])
    )
    y = Channel(field="dropout_rate", type="quantitative", title="Did not complete", unit="%", format=".1%")
    viz = bar_chart(
        ds.rows,
        title="Share of participants not completing, per trial",
        x=x,
        y=y,
        meta=VizMeta(grouping={"by": ["trial"], "rate": "(started - completed) / started, all arms"}),
        tooltip=["started", "completed"],
    )
    facts = {(d.model_extra or {})["nct_id"]: f"{_pct((d.model_extra or {})['dropout_rate'])}%" for d in ds.rows}
    return HandlerResult(
        visualization=viz,
        facts=facts,
        full_refs=ds.full_refs,
        studies_with_results=len(trials),
        studies_analyzed=len(ctx.studies) or len(trials),
    )


# == demographics ==================================================================================

_CHARACTERISTICS = [
    ("Age group", re.compile(r"\bage\b", re.I)),
    ("Sex", re.compile(r"\bsex\b|gender", re.I)),
    ("Race", re.compile(r"\brace\b", re.I)),
    ("Ethnicity", re.compile(r"ethnic", re.I)),
]
_STD_AGE_LABELS = {"CHILD": "Child (<18)", "ADULT": "Adult (18-64)", "OLDER_ADULT": "Older adult (65+)"}
_SEX_LABELS = {"ALL": "All sexes", "FEMALE": "Female only", "MALE": "Male only"}


def _wanted_characteristics(question: str) -> list[str]:
    q = question.lower()
    want = [
        name
        for name, rx in (
            ("Age group", r"\bage"),
            ("Sex", r"\bsex|gender|women|men\b|female|male"),
            ("Race", r"\brace|racial"),
            ("Ethnicity", r"ethnic"),
        )
        if re.search(rx, q)
    ]
    return want or ["Sex", "Age group", "Race"]


@register(Intent.DEMOGRAPHICS)
async def demographics(ctx: HandlerContext) -> HandlerResult:
    """Baseline demographics per arm (stacked bars); cohort scope: eligible age groups / sex."""
    if not ctx.trials:
        return _demographics_cohort(ctx)
    res = await _demographics_trial(ctx, ctx.trials[0])
    if len(ctx.trials) > 1:  # per-arm baseline tables of different trials do not share one chart
        others = ", ".join(t.nct_id for t in ctx.trials[1:])
        res.notes.append(
            note(
                NoteCode.TRUNCATED,
                f"Baseline demographics are shown for {ctx.trials[0].nct_id} only; not analysed: {others}. "
                "Ask about each trial separately to see its arms.",
                "warning",
            )
        )
    return res


async def _demographics_trial(ctx: HandlerContext, s: Study) -> HandlerResult:
    """Per-arm baseline characteristics of one trial."""
    if not has_results_section(s) or not s.get(BASELINE_PATH):
        return _no_results(s, f"Eligibility: {_name(s)}", "baseline demographics")
    want = _wanted_characteristics(ctx.request.query)
    ds = Dataset()
    age_kpis = Dataset(prefix="k")
    groups: dict[str, tuple[str, str]] = {}
    for _gi, gr, gp in s.items(f"{BASELINE_PATH}.groups"):
        if isinstance(gr, dict) and gr.get("id") and str(gr.get("title", "")).strip().lower() != "total":
            groups[gr["id"]] = (str(gr.get("title", gr["id"])), f"{gp}.title")
    bars: list[str] = []
    for _mi, m, mp in s.items(f"{BASELINE_PATH}.measures"):
        if not isinstance(m, dict):
            continue
        title = str(m.get("title", ""))
        char = next((name for name, rx in _CHARACTERISTICS if rx.search(title)), None)
        if char is None:
            continue
        ptype = str(m.get("paramType", "")).upper()
        for ci, cls in enumerate(m.get("classes", []) or []):
            for ki, catg in enumerate(cls.get("categories", []) or []):
                for xi, meas in enumerate(catg.get("measurements", []) or []):
                    gid = meas.get("groupId")
                    if gid not in groups:
                        continue
                    vpath = f"{mp}.classes[{ci}].categories[{ki}].measurements[{xi}]"
                    gtitle, gpath = groups[gid]
                    if ptype.startswith("COUNT"):
                        if char not in want:
                            continue
                        n = parse_int(meas.get("value"))
                        category = catg.get("title") or cls.get("title")
                        if n is None or not category:
                            continue
                        label_path = (
                            f"{mp}.classes[{ci}].categories[{ki}].title"
                            if catg.get("title")
                            else f"{mp}.classes[{ci}].title"
                        )
                        bar = f"{char}: {gtitle}"
                        if bar not in bars:
                            bars.append(bar)
                        ds.emit(
                            {"bar": bar, "characteristic": char, "group": gtitle, "category": category, "n": n},
                            _refs(
                                s.ref(f"{vpath}.value", label=f"{gtitle}: {category}"),
                                s.ref(label_path),
                                s.ref(gpath),
                                s.ref(f"{mp}.title"),
                            ),
                        )
                    elif char == "Age group" and ptype in ("MEAN", "MEDIAN"):
                        val = parse_num(meas.get("value"))
                        if val is None:
                            continue
                        spread = parse_num(meas.get("spread"))
                        age_kpis.emit(
                            _clean(
                                {
                                    "label": f"{ptype.title()} age · {gtitle}",
                                    "value": val,
                                    "unit": str(m.get("unitOfMeasure", "")) or None,
                                    "detail": f"{m.get('dispersionType', '')} {meas.get('spread')}".strip()
                                    if spread is not None
                                    else None,
                                }
                            ),
                            _refs(
                                s.ref(f"{vpath}.value"),
                                s.ref(f"{vpath}.spread") if spread is not None else None,
                                s.ref(gpath),
                                s.ref(f"{mp}.title"),
                                s.ref(f"{mp}.paramType"),
                            ),
                        )
    age_summary = [
        {**(d.model_extra or {}), "citations": [c.model_dump(exclude_none=True) for c in d.citations]}
        for d in age_kpis.rows
    ]
    facts: dict[str, Any] = {"nct_id": s.nct_id}
    for d in age_kpis.rows:
        facts[(d.model_extra or {})["label"]] = (d.model_extra or {})["value"]
    if not ds.rows:
        if age_kpis.rows:
            viz = kpi_cards(age_kpis.rows, title=f"Baseline age: {_name(s)}")
            return HandlerResult(
                visualization=viz, facts=facts, full_refs=age_kpis.full_refs, studies_analyzed=1, studies_with_results=1
            )
        return _no_results(s, f"Eligibility: {_name(s)}", "baseline demographics")
    totals: dict[str, int] = defaultdict(int)
    for d in ds.rows:
        totals[(d.model_extra or {})["bar"]] += (d.model_extra or {})["n"]
    for d in ds.rows:
        e = d.model_extra or {}
        total = totals[e["bar"]]
        d.__pydantic_extra__["share"] = round(e["n"] / total, 4) if total else 0.0  # type: ignore[index]
        facts[f"{e['bar']}|{e['category']}"] = e["n"]
    x = Channel(field="bar", type="nominal", title="Characteristic · arm", sort=ordered(bars))
    y = Channel(field="n", type="quantitative", title="Participants", unit="participants", format=",d")
    series = Channel(field="category", type="nominal", title="Category")
    meta = VizMeta(
        grouping={"by": ["characteristic", "group", "category"], "stack": "category within bar"},
        stats={"age_summary": age_summary} if age_summary else {},
    )
    viz = bar_chart(
        ds.rows,
        title=f"Baseline demographics: {_name(s)}",
        x=x,
        y=y,
        series=series,
        mode="stacked",
        meta=meta,
        tooltip=["share", "characteristic", "group"],
    )
    return HandlerResult(
        visualization=viz,
        facts=facts,
        full_refs={**ds.full_refs, **age_kpis.full_refs},
        studies_analyzed=1,
        studies_with_results=1,
    )


def _demographics_cohort(ctx: HandlerContext) -> HandlerResult:
    ds = Dataset()
    buckets: dict[tuple[str, str], list[FieldRef]] = defaultdict(list)
    studies = ctx.studies
    for s in studies:
        for a in s.get("std_ages") or []:
            buckets[("Eligible age group", _STD_AGE_LABELS.get(a, a))].append(s.ref("std_ages"))
        sex = s.get("sex")
        if sex:
            buckets[("Eligible sex", _SEX_LABELS.get(sex, sex))].append(s.ref("sex"))
    order = [("Eligible age group", v) for v in _STD_AGE_LABELS.values()] + [
        ("Eligible sex", v) for v in _SEX_LABELS.values()
    ]
    order += sorted(k for k in buckets if k not in order)
    for key in order:
        if key in buckets:
            ds.emit({"category": key[1], "dimension": key[0], "trial_count": len(buckets[key])}, buckets[key])
    notes = [
        note(
            NoteCode.NOT_IN_REGISTRY,
            "Cohort-level demographics use each trial's eligibility (who may "
            "enroll), not the enrolled participants' characteristics.",
        )
    ]
    if any(c.sampled for c in ctx.cohorts):
        notes.append(note(NoteCode.SAMPLED, "Counts reflect a sample of the matching trials; see meta.coverage."))
    x = Channel(
        field="category",
        type="nominal",
        title="Eligibility category",
        sort=ordered([(d.model_extra or {})["category"] for d in ds.rows]),
    )
    y = Channel(field="trial_count", type="quantitative", title="Trials", unit="trials", format=",d")
    series = Channel(field="dimension", type="nominal", title="Dimension")
    viz = bar_chart(
        ds.rows,
        title="Who can enroll: eligible age groups and sex",
        x=x,
        y=y,
        series=series,
        meta=VizMeta(
            grouping={
                "by": ["dimension", "category"],
                "aggregate": "count",
                "note": "a trial can accept several age groups",
            }
        ),
    )
    facts = {
        f"{(d.model_extra or {})['dimension']}|{(d.model_extra or {})['category']}": (d.model_extra or {})[
            "trial_count"
        ]
        for d in ds.rows
    }
    return HandlerResult(
        visualization=viz, notes=notes, facts=facts, full_refs=ds.full_refs, studies_analyzed=len(studies)
    )


# == enrollment_vs_power ===========================================================================


@register(Intent.ENROLLMENT_VS_POWER)
async def enrollment_vs_power(ctx: HandlerContext) -> HandlerResult:
    """Power vs effect size at n/2, n, 2n per arm (two-sample z approximation); cohorts: enrollment vs MDE."""
    if _multi_trial(ctx):
        return _power_multi(ctx)
    s = _trial(ctx)
    if s is None:
        return _no_trial("Statistical power")
    arms = fl.flow_table(s) if has_results_section(s) else []
    arm_groups = [(i, a, p) for i, a, p in s.items("arm_groups") if isinstance(a, dict)]
    refs: list[FieldRef]
    if len(arms) >= 1:
        sizes = [float(a.started) for a in arms if a.started]
        n_source = "actual participants started per arm (participant flow)"
        refs = [r for a in arms for r in a.periods[0].started_refs]
        n_arms = len(sizes)
        n = pw.harmonic_mean(sizes) if n_arms > 1 else (sizes[0] if sizes else 0.0)
    else:
        enrollment = parse_num(s.get("enrollment_count"))
        n_arms = max(1, len(arm_groups))
        n = (enrollment or 0.0) / n_arms
        n_source = f"{s.get('enrollment_type') or 'registered'} enrollment / number of arms ({n_arms})"
        refs = _refs(
            s.ref("enrollment_count"), s.ref("enrollment_type"), *[s.ref(f"{p}.label") for _, _, p in arm_groups[:8]]
        )
    if n <= 1:
        ds = Dataset()
        ds.emit({"label": "Enrollment", "value": "Not reported"}, [])
        return HandlerResult(
            visualization=kpi_cards(ds.rows, title=f"Statistical power: {_name(s)}"),
            studies_analyzed=1,
            notes=[note(NoteCode.NOT_IN_REGISTRY, "No usable enrollment count to estimate power.", "warning")],
        )
    one_sample = n_arms == 1
    scenarios = [("n/2", n / 2), ("n", n), ("2n", 2 * n)]
    ds = Dataset()
    mde_at_80: dict[str, float] = {}
    grid = pw.effect_grid()
    with current_trace().step("analysis.power_curve", tool="power_curve", n_per_arm=round(n, 1), arms=n_arms) as info:
        for name, nn in scenarios:
            series = f"{name}: {nn:,.0f} per arm" if not one_sample else f"{name}: {nn:,.0f} participants"
            mde_at_80[series] = round(pw.mde(nn, one_sample=one_sample), 4)
            for d, p in pw.power_curve(nn, grid, one_sample=one_sample):
                ds.emit({"series": series, "effect_size": d, "power": p, "n_per_arm": round(nn, 1)}, refs)
        info["rows"] = len(ds.rows)
    assumptions = list(pw.ASSUMPTIONS)
    assumptions.append(f"n per arm = {n:,.1f} from {n_source}.")
    if one_sample:
        assumptions.append("Single-arm design: one-sample z-test, power = Phi(d*sqrt(n) - 1.96).")
    if s.get("enrollment_type") == "ESTIMATED" and not arms:
        assumptions.append("Enrollment is the sponsor's estimate, not the actual enrollment.")
    meta = VizMeta(
        approximation=Approximation(
            is_approximation=True,
            method="two-sample z-test normal approximation, power(d) = Phi(d*sqrt(n/2) - z_{0.975})",
            assumptions=assumptions,
        ),
        stats={
            "mde_at_80": mde_at_80,
            "power_inputs": {"n_per_arm": round(n, 1), "arms": n_arms, "alpha": pw.ALPHA, "n_source": n_source},
        },
    )
    viz = power_curve(
        ds.rows,
        title=f"Power vs effect size: {_name(s)}",
        meta=meta,
        subtitle="Approximation from the sample size; not the trial's registered power calculation",
    )
    notes = [
        note(
            NoteCode.APPROXIMATED,
            "Power is approximated from the sample size with a two-sample z-test "
            "(alpha 0.05, two-sided); the registry does not report the trial's own "
            "power calculation.",
            "warning",
        )
    ]
    facts: dict[str, Any] = {"nct_id": s.nct_id, "n_per_arm": round(n, 1), "arms": n_arms}
    for k, v in mde_at_80.items():
        facts[f"mde_at_80|{k}"] = v
    return HandlerResult(
        visualization=viz,
        notes=notes,
        facts=facts,
        full_refs=ds.full_refs,
        studies_analyzed=1,
        studies_with_results=1 if arms else 0,
    )


# == several trials: cohort / comparison scope, or several NCT ids ==================================
#
# Survival, effect estimates and power across trials. Measure selection is the deterministic rule
# selector (one LLM call per trial would break the per-request LLM budget). Every row names its trial
# (``nct_id``) and, in comparisons, its cohort; trials that cannot contribute are counted in notes.

MAX_RESULT_TRIALS = 25  # full records analysed per request (= the orchestrator's max_full_records default)
MAX_CURVE_TRIALS = 8  # trials whose curves are drawn (more lines are unreadable)
MAX_COHORT_ANCHORS_PER_ARM = 6  # reported points per arm in multi-trial charts (earliest kept)
MAX_COHORT_INTERPOLATED = 3  # interpolated points per arm in multi-trial charts (payload stays small)
MAX_CURVE_SERIES = 16  # arms (lines) per multi-trial curve chart: 8 two-arm trials
# Payload budget per multi-trial curve chart: each point costs 1 plus its number of excerpts (a reported median
# or landmark carries ~5, S(0) and interpolated points 1). An arm drawn as S(0) + median + 3 interpolated points
# costs 14, so 12 such arms fit; trials reporting many landmark rates fill it sooner. Measured on fixture
# cohorts: public JSON <= ~40 KB compact (~60 KB pretty-printed); before this budget it was ~300-430 KB.
MAX_CURVE_COST = 170
MAX_ESTIMATES_PER_TRIAL = 3
MAX_POWER_ROWS = 200
MAX_ARMS_PER_TRIAL = 4
_AGGREGATE_GROUP = re.compile(r"\s*(total|overall|all (participants|patients|subjects))\b", re.I)
_CURVE_WORDS = re.compile(r"kaplan|meier|\bkm\b|curves?\b", re.I)
_HAZARD_WORDS = re.compile(r"hazard|\bhrs?\b|forest", re.I)
_HAZARD_PARAM = re.compile(r"hazard", re.I)
_ASKED_ENDPOINT = re.compile(r"progression|\bpfs\b|overall survival|\bos\b|death|\bdie\b|live longer|mortality", re.I)
_ENDPOINT_NAMES = {
    "OS": "overall survival",
    "PFS": "progression-free survival",
    "EFS": "event-free survival",
    "DFS": "disease-free survival",
}


def _multi_trial(ctx: HandlerContext) -> bool:
    """Several trials to analyse: more than one full record, or a cohort/comparison without NCT ids."""
    return len(ctx.trials) > 1 or (ctx.plan.scope is not Scope.SINGLE_TRIAL and not ctx.plan.nct_ids)


def _display(s: Study) -> str:
    name = _name(s)
    return s.nct_id if name == s.nct_id else f"{name} ({s.nct_id})"


def _enrollment(s: Study) -> float:
    return parse_num(s.get("enrollment_count")) or 0.0


def _cohort_labels(ctx: HandlerContext) -> dict[str, str]:
    """nct_id -> compared cohort label(s); empty unless the plan compares several cohorts."""
    if len(ctx.cohorts) < 2:
        return {}
    out: dict[str, list[str]] = {}
    for c in ctx.cohorts:
        for st in c.studies:
            labels = out.setdefault(st.nct_id, [])
            if c.label not in labels:
                labels.append(c.label)
    return {k: " + ".join(v) for k, v in out.items()}


async def _result_trials(ctx: HandlerContext) -> tuple[list[Study], list[str]]:
    """(full records with posted results to analyse, NCT ids skipped because they have no results).

    Named trials keep the user's order. Cohort trials are ordered by enrollment, largest first: the
    largest trials are usually the pivotal ones and report the most precise estimates.
    """
    named = bool(ctx.plan.nct_ids) and bool(ctx.trials)
    pool = ctx.trials if named else await _cohort_trials_with_results(ctx, MAX_RESULT_TRIALS)
    with_results = [t for t in pool if has_results_section(t)]
    without = [t.nct_id for t in pool if not has_results_section(t)]
    if not named:
        with_results.sort(key=lambda t: -_enrollment(t))
    return with_results[:MAX_RESULT_TRIALS], without


def _scope_notes(ctx: HandlerContext, analysed: int, without: list[str], what: str) -> list[Note]:
    """NO_RESULTS_POSTED for named trials without results; SAMPLED when a cohort was not fully analysed."""
    notes: list[Note] = []
    if without:
        notes.append(
            note(
                NoteCode.NO_RESULTS_POSTED,
                f"No results posted, so not analysed: {', '.join(without[:10])}"
                + (f" and {len(without) - 10} more" if len(without) > 10 else "")
                + ".",
                "warning",
            )
        )
    if not ctx.plan.nct_ids and ctx.cohorts:
        total = sum(c.total_count for c in ctx.cohorts)
        if total > analysed:
            notes.append(
                note(
                    NoteCode.SAMPLED,
                    f"{what} come from {_plural_trials(analysed)} with posted results (the largest by "
                    f"enrollment, at most {MAX_RESULT_TRIALS}) out of {total:,} matching trials.",
                )
            )
    return notes


def _plural_trials(n: int) -> str:
    return f"{n} trial" if n == 1 else f"{n} trials"


def _skipped_note(skipped: list[str], reason: str) -> list[Note]:
    if not skipped:
        return []
    listed = ", ".join(skipped[:8]) + (f" and {len(skipped) - 8} more" if len(skipped) > 8 else "")
    return [note(NoteCode.NOT_IN_REGISTRY, f"{_plural_trials(len(skipped))} skipped ({reason}): {listed}.")]


def _arms(e: Any) -> str:
    """The compared groups' titles (the registry does not say which is the numerator), shortened for a label."""
    text = ", ".join(e.group_names) if e.group_names else e.comparison.removeprefix("groups: ")
    return text if len(text) <= 70 else text[:67] + "..."


def _survival_hrs(s: Study, indices: list[int]) -> list:
    """Reported hazard ratios (with CI) of the selected survival measures, else of any survival measure."""
    ests = [e for e in forest_estimates(s, indices) if _HAZARD_PARAM.search(e.param_type)]
    if not ests:
        surv = [m.index for m in measure_catalog(s) if sv.is_survival_measure(m.title)]
        ests = [e for e in forest_estimates(s, surv) if _HAZARD_PARAM.search(e.param_type) and not e.description]
    return ests[:MAX_ESTIMATES_PER_TRIAL]


def asked_endpoints(question: str) -> list[str]:
    """Survival endpoints the question names (OS / PFS, in the order ``wanted_endpoints`` gives); [] if none."""
    return wanted_endpoints(question) if _ASKED_ENDPOINT.search(question or "") else []


def _primary_effects(s: Study, hazard_only: bool, endpoints: list[str] | None = None) -> list:
    """The first primary outcome (registry order) that reports an effect estimate with CI.

    With ``endpoints`` (the question names OS / PFS) only measures of those endpoints count, primary first;
    a trial that does not report one is skipped rather than answered with another endpoint.
    """
    cat = measure_catalog(s)
    order = [m for m in cat if m.type == "PRIMARY"] + [m for m in cat if m.type != "PRIMARY"]
    if endpoints:
        for ep in endpoints:
            for m in order:
                if short_endpoint(m.title) != ep:
                    continue
                ests = [e for e in analyses_for(s, m.index) if not hazard_only or _HAZARD_PARAM.search(e.param_type)]
                if ests:
                    return ests[:MAX_ESTIMATES_PER_TRIAL]
        return []
    for m in order:
        ests = [e for e in analyses_for(s, m.index) if not hazard_only or _HAZARD_PARAM.search(e.param_type)]
        if ests:
            return ests[:MAX_ESTIMATES_PER_TRIAL]
        if m.type != "PRIMARY" and not hazard_only:
            break  # only primary outcomes for a generic effect question
    return []


def _multi_forest(
    ctx: HandlerContext,
    entries: list[tuple[Study, Any]],
    *,
    title: str,
    notes: list[Note],
    analysed: int,
) -> HandlerResult:
    """One forest row per trial x reported comparison, each cited to its analysis fields."""
    labels = _cohort_labels(ctx)
    ratio = [en for en in entries if en[1].kind == "ratio"]
    if ratio:
        kept = ratio
    else:  # differences only share an axis when they are in the same unit
        unit_of = {id(e): str(st.get(f"{OM_PATH}[{e.measure_index}].unitOfMeasure", "")) for st, e in entries}
        common = max(set(unit_of.values()), key=list(unit_of.values()).count)
        kept = [en for en in entries if unit_of[id(en[1])] == common]
    dropped = sorted({st.nct_id for st, _ in entries} - {st.nct_id for st, _ in kept})
    kept, duplicates = _dedupe_estimates(kept)
    if duplicates:
        notes.append(
            note(
                NoteCode.TRUNCATED,
                f"{duplicates} duplicate estimate(s) (same trial, measure, comparison and values, reported in "
                "more than one analysis) are shown once.",
            )
        )
    if dropped:
        notes.append(
            note(
                NoteCode.TRUNCATED,
                "Estimates on a different scale (ratio vs difference, or another "
                f"unit) cannot share one axis and are not shown for: {', '.join(dropped[:8])}.",
            )
        )
    ds = Dataset()
    seen: set[str] = set()
    for st, e in kept:
        label = f"{_name(st)} · {short_endpoint(e.measure_title)} · {_arms(e)}"
        k = 2
        base = label
        while label in seen:
            label, k = f"{base} [{k}]", k + 1
        seen.add(label)
        ds.emit(
            _clean(
                {
                    "label": label,
                    "trial": _display(st),
                    "nct_id": st.nct_id,
                    "cohort": labels.get(st.nct_id),
                    "endpoint": short_endpoint(e.measure_title),
                    "comparison": e.comparison,
                    "estimate": e.estimate,
                    "ci_low": e.ci_low,
                    "ci_high": e.ci_high,
                    "ci_pct": e.ci_pct,
                    "p_value": e.p_raw,
                    "param_type": e.param_type,
                    "method": e.method,
                    "measure": e.measure_title,
                }
            ),
            e.refs,
            keep_order=True,
        )
    notes += _missing_cohort_notes(ctx, {st.nct_id for st, _ in kept})
    kind = kept[0][1].kind
    ptypes = {e.param_type for _, e in kept}
    est_title = next(iter(ptypes)) if len(ptypes) == 1 else ("Ratio estimate" if kind == "ratio" else "Difference")
    series = Channel(field="cohort", type="nominal", title="Cohort") if labels else None
    viz = forest_plot(
        ds.rows,
        title=title,
        kind=kind,
        estimate_title=est_title,
        subtitle="As reported by each trial; different populations and comparators, not a head-to-head comparison",
        series=series,
        tooltip=["trial", "comparison", "p_value", "ci_pct", "param_type", "method"],
    )
    facts: dict[str, Any] = {}
    for st, e in kept[:6]:
        facts.setdefault(f"{_display(st)} {short_endpoint(e.measure_title)}", _effect_text(e))
    unit = None
    if kind != "ratio":
        units = {str(st.get(f"{OM_PATH}[{e.measure_index}].unitOfMeasure", "")) for st, e in kept}
        unit = units.pop() if len(units) == 1 else None
    facts["cohort_effects_derived"] = _effect_facts(kept, kind, analysed, unit)
    return HandlerResult(
        visualization=viz,
        notes=notes,
        facts=facts,
        full_refs=ds.full_refs,
        studies_analyzed=analysed,
        studies_with_results=analysed,
    )


def _effect_facts(kept: list[tuple[Study, Any]], kind: str, analysed: int, unit: str | None) -> dict[str, Any]:
    """Facts over EVERY forest row (read by the cross-trial summary template; each number is also evidence).

    Counts estimates whose whole confidence interval lies below / above the no-effect value (1 for
    ratios, 0 for differences) or crosses it; the template states them without interpreting direction.
    """
    null = 1 if kind == "ratio" else 0
    below = sum(1 for _, e in kept if e.ci_high < null)
    above = sum(1 for _, e in kept if e.ci_low > null)
    ptypes = {e.param_type for _, e in kept}
    endpoints = list(dict.fromkeys(short_endpoint(e.measure_title) for _, e in kept))
    if all(_HAZARD_PARAM.search(pt) for pt in ptypes):
        what = "Hazard ratios"
    elif len(ptypes) == 1:
        what = f"{next(iter(ptypes))} estimates"
    else:
        what = "Ratio estimates" if kind == "ratio" else "Difference estimates"
    trial_labels = _trial_labels([st for st, _ in kept])
    names, more = _names([trial_labels[st.nct_id] for st, _ in kept])
    out: dict[str, Any] = {
        "kind": kind,
        "what": what,
        "estimates": len(kept),
        "trials": len({st.nct_id for st, _ in kept}),
        "trials_with_results": analysed,
        "trial_names": names,
        "names_more": more,
        "null_value": null,
        "ci_below_null": below,
        "ci_above_null": above,
        "ci_crosses_null": len(kept) - below - above,
        "estimate_low": min(e.estimate for _, e in kept),
        "estimate_high": max(e.estimate for _, e in kept),
    }
    if len(endpoints) == 1 and endpoints[0] in _ENDPOINT_NAMES:
        out["endpoint_name"] = _ENDPOINT_NAMES[endpoints[0]]
    if unit:
        out["unit"] = unit
    return out


def _dedupe_estimates(entries: list[tuple[Study, Any]]) -> tuple[list[tuple[Study, Any]], int]:
    """Drop rows identical to an earlier one (same trial, measure title, comparison, parameter and values)."""
    seen: set[tuple] = set()
    out: list[tuple[Study, Any]] = []
    for st, e in entries:
        key = (
            st.nct_id,
            e.measure_title,
            tuple(e.group_names) or e.comparison,
            e.param_type,
            e.estimate,
            e.ci_low,
            e.ci_high,
        )
        if key not in seen:
            seen.add(key)
            out.append((st, e))
    return out, len(entries) - len(out)


def _effect_text(e: Any) -> str:
    return f"{e.param_type} {e.estimate:g} ({e.ci_low:g} to {e.ci_high:g})"


def _missing_cohort_notes(ctx: HandlerContext, shown: set[str]) -> list[Note]:
    """Comparisons: say so when a compared cohort contributed no row (never imply it was compared)."""
    out = []
    for c in ctx.cohorts if len(ctx.cohorts) > 1 else []:
        if not {st.nct_id for st in c.studies} & shown:
            out.append(
                note(
                    NoteCode.NOT_IN_REGISTRY,
                    f"No {c.label} trial among those analysed reports the values needed, so {c.label} "
                    "is missing from this comparison.",
                    "warning",
                )
            )
    return out


def _multi_empty(title: str, message: str, notes: list[Note], analysed: int) -> HandlerResult:
    ds = Dataset()
    ds.emit({"label": "Trials with usable reported values", "value": 0}, [])
    viz = kpi_cards(ds.rows, title=title)
    if viz.encoding.y is not None:  # the card's value axis says what it counts (final validation checks this)
        viz.encoding.y.title, viz.encoding.y.unit = "Trials with usable reported values", "trials"
    return HandlerResult(
        visualization=viz,
        notes=[*notes, note(NoteCode.NOT_IN_REGISTRY, message, "warning")],
        studies_analyzed=analysed,
        studies_with_results=analysed,
    )


def _wants(ctx: HandlerContext, words: re.Pattern[str], viz_type: VizType) -> bool:
    return ctx.request.preferred_visualization is viz_type or bool(words.search(ctx.request.query or ""))


async def _survival_multi(ctx: HandlerContext) -> HandlerResult:
    """Several trials: hazard-ratio forest (default for cohorts) or approximate curves (asked / named trials)."""
    trials, without = await _result_trials(ctx)
    notes = _scope_notes(ctx, len(trials), without, "Survival values")
    if not trials:
        return _multi_empty("Survival across trials", "None of the trials has posted results.", notes, 0)
    question = ctx.request.query or ""
    asked = asked_endpoints(question)
    per_trial = []
    with current_trace().step("analysis.survival_multi", tool="reconstruct_survival", trials=len(trials)) as info:
        for t in trials:
            sel = rule_select(t, question, "survival")
            hrs = _survival_hrs(t, sel.measure_indices)
            if asked:  # the question names an endpoint: never fall back to another endpoint's hazard ratio
                hrs = [e for e in hrs if short_endpoint(e.measure_title) in asked]
            per_trial.append((t, sv.extract_anchors(t, sel.measure_indices), hrs))
        info.update(
            with_anchors=sum(1 for _, ext, _ in per_trial if ext.anchors), with_hr=sum(1 for *_, hr in per_trial if hr)
        )
    has_curves = any(ext.anchors for _, ext, _ in per_trial)
    has_hr = any(hr for *_, hr in per_trial)
    curves_first = _wants(ctx, _CURVE_WORDS, VizType.SURVIVAL_CURVE) or (
        bool(ctx.plan.nct_ids) and not _wants(ctx, _HAZARD_WORDS, VizType.FOREST_PLOT)
    )
    if has_curves and (curves_first or not has_hr):
        return _multi_curves(ctx, per_trial, notes, question)
    if has_hr:
        entries = [(t, e) for t, _, hrs in per_trial for e in hrs]
        what = f"no reported {' or '.join(asked)} hazard ratio with CI" if asked else "no reported hazard ratio with CI"
        notes += _skipped_note([t.nct_id for t, _, hrs in per_trial if not hrs], what)
        return _multi_forest(
            ctx, entries, title="Reported survival hazard ratios by trial", notes=notes, analysed=len(trials)
        )
    return _multi_empty(
        "Survival across trials",
        f"No analysed trial reports a survival median, landmark rate or hazard ratio with CI; {NOT_IN_REGISTRY_TTE}.",
        notes,
        len(trials),
    )


def _chart_endpoint(question: str, by_trial: list[tuple[Study, dict[str, dict[str, list]]]]) -> str | None:
    """The one endpoint a multi-trial curve chart draws.

    Order: endpoints the question names (``wanted_endpoints``: PFS if asked, OS for "overall survival" /
    death; OS then PFS when nothing specific is asked), then any other endpoint, most trials first.
    The first one that at least one trial reports wins.
    """
    counts: dict[str, int] = defaultdict(int)
    for _, eps in by_trial:
        for ep in eps:
            counts[ep] += 1
    order = [*wanted_endpoints(question), *sorted(counts, key=lambda ep: -counts[ep])]
    return next((ep for ep in order if counts.get(ep)), None)


def _endpoint_name(endpoint: str) -> str:
    return _ENDPOINT_NAMES.get(endpoint, endpoint)


def _multi_curves(ctx: HandlerContext, per_trial: list, notes: list[Note], question: str) -> HandlerResult:
    """Approximate curves of ONE endpoint for up to ``MAX_CURVE_TRIALS`` trials; series = trial · endpoint · arm.

    Trials are taken in the user's order (named NCT ids) or, for cohorts, randomized trials first and
    then by enrollment (largest first), while they fit ``MAX_CURVE_SERIES`` arms and the ``MAX_CURVE_COST``
    payload budget (the first trial is always drawn; a trial that does not fit is skipped and a smaller later
    one may still fit). A trial whose arms all fail the curve validity check gives its
    slot to the next one. Interpolation is sparser than for one trial (``MAX_COHORT_INTERPOLATED``
    points per arm) and interpolated points cite only the measure title plus ``derived_from`` anchor ids.
    """
    labels = _cohort_labels(ctx)
    usable = [(t, ext) for t, ext, _ in per_trial if ext.anchors]
    if not ctx.plan.nct_ids:  # cohorts: randomized trials first (their arms are comparable), then by enrollment
        usable.sort(key=lambda te: (te[0].get("allocation") != "RANDOMIZED", -_enrollment(te[0])))
    nr_only = [(t, ext) for t, ext, _ in per_trial if not ext.anchors and ext.not_reached]
    notes += _skipped_note(
        [t.nct_id for t, ext, _ in per_trial if not ext.anchors and not ext.not_reached],
        "no reported median or landmark survival value",
    )
    notes += _skipped_note(
        [t.nct_id for t, _ in nr_only], "median not reached in every reported arm, so no curve can be anchored"
    )
    prefix = _trial_labels([t for t, _ in usable])  # "CheckMate 067", or "Name (NCT…)" when names repeat
    by_trial: list[tuple[Study, dict[str, dict[str, list]]]] = []  # trial -> endpoint -> series -> anchors
    for t, ext in usable:
        eps: dict[str, dict[str, list]] = {}
        for k, pts in ext.anchors.items():
            if not _AGGREGATE_GROUP.match(pts[0].group):
                eps.setdefault(pts[0].endpoint, {})[f"{prefix[t.nct_id]} · {k}"] = pts
        by_trial.append((t, eps))
    endpoint = _chart_endpoint(question, by_trial)
    if endpoint is None:
        return _multi_empty(
            "Survival across trials", "No reported survival values formed a valid curve.", notes, len(per_trial)
        )
    having = [(t, eps[endpoint]) for t, eps in by_trial if endpoint in eps]
    other_eps: dict[str, int] = defaultdict(int)
    for _, eps in by_trial:
        for ep in eps:
            if ep != endpoint:
                other_eps[ep] += 1
    if other_eps:
        listed = ", ".join(f"{ep} ({_plural_trials(n)})" for ep, n in sorted(other_eps.items(), key=lambda x: -x[1]))
        only_other = [t.nct_id for t, eps in by_trial if endpoint not in eps]
        notes.append(
            note(
                NoteCode.TRUNCATED,
                f"One endpoint per chart: only {endpoint} curves are drawn. Also reported but not drawn: {listed}"
                + (f"; trials with no {endpoint} values: {', '.join(only_other[:8])}" if only_other else "")
                + (f" and {len(only_other) - 8} more" if len(only_other) > 8 else "")
                + ". Ask for that endpoint by name to see it.",
            )
        )
    ds = Dataset()
    facts: dict[str, Any] = {}
    stats: list[dict[str, Any]] = []
    interpolated: list[str] = []
    drawn: list[Study] = []
    medians: list[float] = []
    landmarks: dict[float, list[float]] = defaultdict(list)
    not_reached = 0
    over_budget: list[str] = []
    series_drawn: set[str] = set()
    spent = 0
    for t, anchors in having:
        if len(drawn) >= MAX_CURVE_TRIALS:
            break
        name = prefix[t.nct_id]
        if len(anchors) > MAX_ARMS_PER_TRIAL:
            notes.append(
                note(
                    NoteCode.TRUNCATED,
                    f"{name}: {MAX_ARMS_PER_TRIAL} of {len(anchors)} arms drawn "
                    f"({', '.join(pts[0].group for pts in list(anchors.values())[MAX_ARMS_PER_TRIAL:])} omitted).",
                )
            )
            anchors = dict(list(anchors.items())[:MAX_ARMS_PER_TRIAL])
        long_arms = [pts[0].group for pts in anchors.values() if len(pts) > MAX_COHORT_ANCHORS_PER_ARM]
        if long_arms:  # earliest reported points kept: the curve ends earlier, it is never extrapolated
            anchors = {k: pts[:MAX_COHORT_ANCHORS_PER_ARM] for k, pts in anchors.items()}
            notes.append(
                note(
                    NoteCode.TRUNCATED,
                    f"{name}: only the first {MAX_COHORT_ANCHORS_PER_ARM} reported {endpoint} values per arm are "
                    f"drawn; later time points omitted for {', '.join(long_arms)}.",
                )
            )
        for k, pts in anchors.items():
            for a in pts:
                a.series = k
        # S(0) = 1 is definitional: one citation, the measure title (else the arm's first reported value)
        start_refs = {k: [r for r in [sv.single_ref(pts[0])] if r] for k, pts in anchors.items()}
        points = sv.reconstruct_survival(anchors, start_refs, max_interpolated=MAX_COHORT_INTERPOLATED)
        bad = sv.curve_problems(points)
        for series, why in bad.items():
            notes.append(note(NoteCode.APPROXIMATED, f"{series}: not drawn, the reported values {why}.", "warning"))
        points = [p for p in points if p.series not in bad]
        if not points:
            continue
        cost = _curve_cost(points)
        arms = list(dict.fromkeys(p.series for p in points))
        cut: list[str] = []
        while not drawn and cost > MAX_CURVE_COST and len(arms) > 1:  # the first trial alone must fit the budget
            cut.insert(0, arms.pop())
            points = [p for p in points if p.series in arms]
            cost = _curve_cost(points)
        if cut:
            notes.append(
                note(
                    NoteCode.TRUNCATED,
                    f"{name}: {len(arms)} of {len(arms) + len(cut)} arms drawn to keep the chart within its point "
                    f"budget ({', '.join(cut)} omitted).",
                )
            )
        n_series = len(arms)
        if drawn and (len(series_drawn) + n_series > MAX_CURVE_SERIES or spent + cost > MAX_CURVE_COST):
            over_budget.append(t.nct_id)  # does not fit; a smaller later trial may still fit
            continue
        drawn.append(t)
        spent += cost
        series_drawn.update(p.series for p in points)
        interpolated += emit_curve_points(
            ds,
            points,
            lambda p, t=t: {
                "series": p.series,
                "time": p.time,
                "survival": p.survival,
                "source": p.source,
                "nct_id": t.nct_id,  # ``series`` = trial name · endpoint · arm
                "cohort": labels.get(t.nct_id),
                "endpoint": p.endpoint,
                "ci_low": _r(p.ci_low),
                "ci_high": _r(p.ci_high),
                # reported points say what they are; interpolated ones are identified by source + derived_from
                "label": _point_label(p) if p.source == "reported" else None,
            },
        )
        for p in points:
            if p.source == "reported" and p.kind != "start":
                facts[f"{p.series}|{p.kind}|{p.time:g}"] = round(p.survival, 4)
                if p.kind == "median":
                    medians.append(p.time)
                else:
                    landmarks[p.time].append(p.survival)
        ext = next(ext for st, ext in usable if st is t)
        reached_not = [s for s in ext.not_reached if s.startswith(f"{endpoint} ")]
        not_reached += len(reached_not)
        notes += [
            note(NoteCode.NOT_IN_REGISTRY, f"{name} · {s}: median not reached; no curve for this arm.")
            for s in reached_not
        ]
        for e in _survival_hrs(t, []):
            if short_endpoint(e.measure_title) == endpoint:  # same endpoint as the curves, nothing else
                stats.append({**e.as_meta(), "nct_id": t.nct_id})
    # trials whose every arm had not reached the median count toward "arms had not reached the median"
    not_reached += sum(1 for _, ext in nr_only for s in ext.not_reached if s.startswith(f"{endpoint} "))
    omitted = [t.nct_id for t, _ in having if t not in drawn]
    if omitted:
        order = "in the order named" if ctx.plan.nct_ids else "randomized trials first, then largest enrollment"
        budget = (
            f"; {len(over_budget)} skipped to keep the chart within {MAX_CURVE_SERIES} arms and its point budget"
            if over_budget
            else ""
        )
        notes.append(
            note(
                NoteCode.TRUNCATED,
                f"Curves drawn for {_plural_trials(len(drawn))} of the {len(having)} with reported {endpoint} "
                f"values ({order}; at most {MAX_CURVE_TRIALS} trials{budget}); {len(omitted)} omitted: "
                f"{', '.join(omitted[:8])}" + (f" and {len(omitted) - 8} more" if len(omitted) > 8 else "") + ".",
            )
        )
    notes += _missing_cohort_notes(ctx, {(d.model_extra or {})["nct_id"] for d in ds.rows})
    if not ds.rows:
        return _multi_empty(
            "Survival across trials", "No reported survival values formed a valid curve.", notes, len(per_trial)
        )
    facts["cohort_curves_derived"] = _curve_facts(
        endpoint, [prefix[t.nct_id] for t in drawn], len(having), len(per_trial), medians, landmarks, not_reached
    )
    meta = VizMeta(
        approximation=Approximation(is_approximation=True, method=sv.METHOD, assumptions=sv.ASSUMPTIONS),
        stats={"analyses": stats} if stats else {},
    )
    viz = survival_curve(
        ds.rows,
        title=f"Approximate {_endpoint_name(endpoint)} by trial",
        subtitle="Reconstructed from each trial's reported medians/landmark rates; not a head-to-head comparison",
        meta=meta,
    )
    viz.encoding.series = Channel(field="series", type="nominal", title="Trial · endpoint · arm")
    notes.append(
        note(
            NoteCode.APPROXIMATED,
            f"Curves are approximations through reported medians and landmark rates; {NOT_IN_REGISTRY_TTE}. "
            "Interpolated points are marked source=interpolated and list the reported points they were "
            "derived from (derived_from).",
            "warning",
            affects=interpolated or ["*"],
        )
    )
    return HandlerResult(
        visualization=viz,
        notes=notes,
        facts=facts,
        full_refs=ds.full_refs,
        studies_analyzed=len(per_trial),
        studies_with_results=len(per_trial),
    )


def _curve_cost(points: list[sv.CurvePoint]) -> int:
    """Payload cost of a trial's curve points: 1 per point plus its excerpts (see ``MAX_CURVE_COST``)."""
    return sum(1 + len(p.refs) for p in points)


def _trial_labels(studies: list[Study]) -> dict[str, str]:
    """nct_id -> display name; a name shared by several trials (same acronym) gets its NCT id appended."""
    counts = defaultdict(set)
    for st in studies:
        counts[_name(st)].add(st.nct_id)
    return {st.nct_id: _name(st) if len(counts[_name(st)]) == 1 else f"{_name(st)} ({st.nct_id})" for st in studies}


def _names(labels: list[str], limit: int = 4) -> tuple[list[str], int]:
    """(the first ``limit`` distinct trial labels, how many more) for summary sentences."""
    names = list(dict.fromkeys(labels))
    return names[:limit], max(0, len(names) - limit)


def _curve_facts(
    endpoint: str,
    drawn: list[str],
    with_endpoint: int,
    analysed: int,
    medians: list[float],
    landmarks: dict[float, list[float]],
    not_reached: int,
) -> dict[str, Any]:
    """Facts over every drawn arm (read by the survival summary template; each number is also evidence)."""
    names, more = _names(drawn)
    out: dict[str, Any] = {
        "endpoint": endpoint,
        "endpoint_name": _endpoint_name(endpoint),
        "trials_drawn": len(drawn),
        "trials_with_endpoint": with_endpoint,
        "trials_with_results": analysed,
        "trial_names": names,
        "names_more": more,
        "arms_median_not_reached": not_reached,
    }
    if medians:
        out.update(
            medians_reported=len(medians),
            median_low_months=round(min(medians), 1),
            median_high_months=round(max(medians), 1),
        )
    if landmarks:  # the landmark time reported by the most arms (earliest on ties)
        lt = min(landmarks, key=lambda x: (-len(landmarks[x]), x))
        vals = landmarks[lt]
        out.update(
            landmark_month=round(lt, 1),
            landmark_arms=len(vals),
            landmark_low=round(min(vals), 3),
            landmark_high=round(max(vals), 3),
            # percent forms are evidence for "100.0%" (a rate of 1.0 is not a fraction for the summary checker)
            landmark_low_pct=round(min(vals) * 100, 1),
            landmark_high_pct=round(max(vals) * 100, 1),
        )
    return out


def _point_label(p: sv.CurvePoint) -> str:
    if p.source == "interpolated":
        return "Interpolated (piecewise exponential)"
    if p.kind == "start":
        return "Start of follow-up (S=1 by definition)"
    return "Reported median (S=0.5)" if p.kind == "median" else "Reported landmark rate"


async def _results_multi(ctx: HandlerContext) -> HandlerResult:
    """Several trials: forest of each trial's reported primary-outcome effect (hazard ratios when asked)."""
    trials, without = await _result_trials(ctx)
    notes = _scope_notes(ctx, len(trials), without, "Effect estimates")
    if not trials:
        return _multi_empty("Reported effects across trials", "None of the trials has posted results.", notes, 0)
    query = ctx.request.query or ""
    hazard = bool(_HAZARD_WORDS.search(query) or sv.is_survival_measure(query))  # survival -> hazard ratios
    with current_trace().step("analysis.forest_multi", tool="forest_rows", trials=len(trials)) as info:
        asked = asked_endpoints(query)
        found = [(t, _primary_effects(t, hazard, asked)) for t in trials]
        info["rows"] = sum(len(e) for _, e in found)
    entries = [(t, e) for t, ests in found for e in ests]
    what = "hazard ratio with CI" if hazard else "effect estimate with CI for a primary outcome"
    if asked:
        what = f"{' or '.join(asked)} {'hazard ratio' if hazard else 'effect estimate'} with CI"
    notes += _skipped_note([t.nct_id for t, ests in found if not ests], f"no reported {what}")
    if not entries:
        return _multi_empty(
            "Reported effects across trials", f"No analysed trial reports a {what}.", notes, len(trials)
        )
    title = "Reported hazard ratios by trial" if hazard else "Reported primary-outcome effects by trial"
    return _multi_forest(ctx, entries, title=title, notes=notes, analysed=len(trials))


def _power_multi(ctx: HandlerContext) -> HandlerResult:
    """Cohort: enrollment vs the smallest standardized effect detectable with 80% power (derived, approximate)."""
    labels = _cohort_labels(ctx)
    studies = ctx.trials if ctx.trials else ctx.studies
    ds = Dataset()
    skipped: list[str] = []
    observational: list[str] = []
    rows = []
    for st in studies:
        if st.get("study_type") not in (None, "INTERVENTIONAL"):
            observational.append(st.nct_id)  # no randomized comparison to power
            continue
        n = parse_num(st.get("enrollment_count"))
        arm_groups = [(i, a, p) for i, a, p in st.items("arm_groups") if isinstance(a, dict)]
        if not n or n <= 1 or not arm_groups:
            skipped.append(st.nct_id)
            continue
        n_arms = len(arm_groups)
        per_arm = n / n_arms
        rows.append((n, st, n_arms, per_arm))
    rows.sort(key=lambda r: -r[0])
    for n, st, n_arms, per_arm in rows[:MAX_POWER_ROWS]:
        d = pw.mde(per_arm, one_sample=n_arms == 1)
        refs = _refs(
            st.ref("enrollment_count"),
            st.ref("enrollment_type"),
            *[st.ref(f"{p}.label") for _, _, p in list(st.items("arm_groups"))[:8]],
        )
        ds.emit(
            _clean(
                {
                    "trial": _display(st),
                    "nct_id": st.nct_id,
                    "cohort": labels.get(st.nct_id),
                    "enrollment": int(n) if float(n).is_integer() else n,
                    "enrollment_type": st.get("enrollment_type"),
                    "arms": n_arms,
                    "n_per_arm": round(per_arm, 1),
                    "mde_at_80": round(d, 4),
                }
            ),
            refs,
        )
    notes = [
        note(
            NoteCode.APPROXIMATED,
            "The detectable effect is derived from enrollment / number of arms with a two-sample z-test "
            "(alpha 0.05, two-sided, 80% power; one-sample for single-arm trials). The registry does not "
            "report each trial's own power calculation.",
            "warning",
        )
    ]
    notes += _skipped_note(skipped, "no enrollment count or arm list")
    notes += _skipped_note(observational, "not an interventional study")
    if len(rows) > MAX_POWER_ROWS:
        notes.append(note(NoteCode.TRUNCATED, f"The {MAX_POWER_ROWS} largest of {len(rows)} trials are shown."))
    total = sum(c.total_count for c in ctx.cohorts)
    if total > len(studies):
        notes.append(note(NoteCode.SAMPLED, f"{len(studies):,} of {total:,} matching trials were retrieved."))
    if not ds.rows:
        return _multi_empty(
            "Sample size and detectable effect", "No trial reports an enrollment count and arms.", notes[1:], 0
        )
    meta = VizMeta(
        approximation=Approximation(
            is_approximation=True,
            method="minimum detectable standardized effect d at 80% power, d = (z_0.975 + z_0.8) / sqrt(n_per_arm / 2)",
            assumptions=[
                *pw.ASSUMPTIONS,
                "n per arm = enrollment / number of registered arm groups.",
                "Estimated (planned) enrollment is used when actual enrollment is not reported.",
            ],
        )
    )
    viz = scatter_plot(
        ds.rows,
        title="Enrollment vs smallest effect detectable with 80% power",
        x=Channel(
            field="enrollment", type="quantitative", title="Enrollment", unit="participants", scale=Scale(type="log")
        ),
        y=Channel(
            field="mde_at_80",
            type="quantitative",
            title="Detectable effect at 80% power (Cohen's d)",
            unit="Cohen's d",
            format=".2f",
        ),
        series=Channel(field="cohort", type="nominal", title="Cohort") if labels else None,
        meta=meta,
        subtitle="Derived from registered enrollment; not each trial's registered power calculation",
        tooltip=["trial", "arms", "n_per_arm", "enrollment_type"],
    )
    ys = sorted((d.model_extra or {})["mde_at_80"] for d in ds.rows)
    facts = {"trials_shown_derived": len(ds.rows), "median_mde_at_80": ys[len(ys) // 2]}
    return HandlerResult(
        visualization=viz,
        notes=notes,
        facts=facts,
        full_refs=ds.full_refs,
        studies_analyzed=len(studies),
        studies_with_results=None,
    )
