"""Synthesis "agent": a deterministic, template-based ``answer_summary`` (no LLM).

Sentences come from typed templates over ``HandlerResult.facts`` and the FINAL (post-downgrade) chart:

* ``"<group>|<bucket>"`` facts are distributions: one sentence per group naming its largest buckets.
  Buckets are only ever compared with buckets of the same group.
* Known scalar facts (``matching_trials``, ``candidate_trials`` …) have their own sentence template;
  unknown scalars are never stated ("Nodes: 15" is not an answer).
* Networks answer in words: the named entity's most frequent partners, or the most connected entities
  and the strongest link. Eligibility checklists state how many criteria are met / block / need review.
* A value is always named by what it measures (the chart's value-axis title), never "the highest value".
* With no usable facts, the chart's own label/value rows are summarized the same way.
* Empty or zero states (no data, "nothing found" cards, all-zero buckets) get no summary.

Every number is therefore a value the chart or facts already contain; the Verification agent re-checks.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

from ctviz.api.schemas.plan import Intent, Persona, QueryPlan
from ctviz.api.schemas.viz import GraphData, Visualization, VizType
from ctviz.ctgov.fields import status_phrase
from ctviz.viz.base import HandlerResult

MAX_SENTENCES = 4
MAX_GROUPS = 3
DISTRIBUTION_TYPES = frozenset(
    {VizType.BAR_CHART, VizType.PIE_CHART, VizType.GEO_MAP, VizType.HISTOGRAM, VizType.TIME_SERIES, VizType.HEATMAP}
)
SCALAR_TEMPLATES: dict[str, str] = {
    "matching_trials": "{v} {trials} on ClinicalTrials.gov match the question.",
    "candidate_trials": "The {v} most recently updated {trials} are listed; pick one to ask about it.",
    "studies_with_results": "{v} of these {trials} have posted results.",
}


def fmt_number(v: float | int) -> str:
    """``1306`` → ``'1,306'``; ``0.4421`` → ``'0.44'``; ``44.0`` → ``'44'``."""
    if isinstance(v, int) or float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _is_empty_state(viz: Visualization) -> bool:
    rows = viz.data.nodes if isinstance(viz.data, GraphData) else viz.data
    if not rows:
        return True
    # "Nothing found" / clarification cards carry no evidence at all.
    return viz.type is VizType.KPI_CARDS and all(not r.citations for r in rows)


def _row_groups(viz: Visualization) -> dict[str, list[tuple[str, float]]]:
    """``{series: [(label, value)]}`` from the chart rows via its quantitative channel."""
    if isinstance(viz.data, GraphData):
        return {}
    enc = viz.encoding
    value_ch = next((c for c in (enc.y, enc.x, enc.size) if c is not None and c.type == "quantitative"), None)
    label_ch = next(
        (c for c in (enc.x, enc.label, enc.y) if c is not None and c is not value_ch and c.type != "quantitative"), None
    )
    if value_ch is None or label_ch is None:
        return {}
    series_field = enc.series.field if enc.series is not None and enc.series.type != "quantitative" else None
    groups: dict[str, list[tuple[str, float]]] = {}
    for d in viz.data:
        extra = d.model_extra or {}
        v, lab = extra.get(value_ch.field), extra.get(label_ch.field)
        if _is_number(v) and lab is not None:
            text = _display_date(str(lab), label_ch.time_format) if label_ch.type == "temporal" else str(lab)
            groups.setdefault(str(extra.get(series_field, "")) if series_field else "", []).append((text, v))
    return groups


def _display_date(value: str, time_format: str | None) -> str:
    """ISO bucket start → the channel's display form (``'2026-01-01'`` with ``'%Y'`` → ``'2026'``)."""
    try:
        return dt.date.fromisoformat(value[:10]).strftime(time_format or "%Y-%m-%d")
    except ValueError:
        return value


def _value_channel(viz: Visualization):  # noqa: ANN202
    enc = viz.encoding
    return next((c for c in (enc.y, enc.x, enc.size) if c is not None and c.type == "quantitative"), None)


def _fmt_value(v: float, viz: Visualization) -> str:
    ch = _value_channel(viz)
    if ch is not None and ch.format and "%" in ch.format and isinstance(v, float) and 0 <= v <= 1:
        return f"{v * 100:.1f}%"
    return fmt_number(v)


def _unit(viz: Visualization) -> str:
    ch = _value_channel(viz)
    return ch.unit if ch is not None and ch.unit and ch.unit != "%" else ""


_REMAINDER = re.compile(r"^Other \(\d[\d,]* more\)$")


def _distribution_sentences(
    groups: dict[str, list[tuple[str, float]]], unit: str, patient: bool, viz: Visualization | None = None
) -> list[str]:
    out = []
    ch = _value_channel(viz) if viz is not None else None
    measure = ch.title.lower() if ch is not None and ch.title else "value"

    def f(v: float) -> str:
        return _fmt_value(v, viz) if viz is not None else fmt_number(v)

    for group, items in list(groups.items())[:MAX_GROUPS]:
        # the "Other (N more)" remainder is many categories pooled, never "the most common" one; a real
        # category such as "Other (academic, hospital, non-profit)" still counts
        positive = sorted(
            ((lab, v) for lab, v in items if v > 0 and not _REMAINDER.match(str(lab))), key=lambda kv: -kv[1]
        )
        if not positive:
            continue
        (top_label, top_value), others = positive[0], positive[1:3]
        prefix = f"For {group}, " if group else ""
        if patient:
            out.append(f"{prefix}the largest group is {top_label} ({f(top_value)}{' ' + unit if unit else ''}).")
        else:
            lead = f"has the most {unit}" if unit else f"has the highest {measure}"
            tail = "".join(f", then {lab} ({f(v)})" for lab, v in others)
            out.append(f"{prefix}{top_label} {lead} ({f(top_value)}){tail}.")
    return out


def _scalar_sentences(scalars: dict[str, float]) -> list[str]:
    out = []
    for key, v in scalars.items():
        tmpl = SCALAR_TEMPLATES.get(key)
        if tmpl is not None:
            out.append(tmpl.format(v=fmt_number(v), trials="trial" if v == 1 else "trials"))
    return out


def _plural(n: int, word: str) -> str:
    return f"{fmt_number(n)} {word}{'' if n == 1 else 's'}"


def _network_sentences(facts: dict[str, Any]) -> list[str]:
    """Plain-language answer for a co-occurrence network (the chart shows the structure)."""
    kind = facts.get("network_kind")
    out: list[str] = []
    for pin in facts.get("pinned_partners") or []:
        parts = pin.get("partners") or []
        if not parts:
            out.append(
                f"{pin['label']} is not combined with another drug in the same arm in these trials."
                if kind == "drug_drug"
                else f"{pin['label']} has no links in these trials."
            )
            continue
        listing = ", ".join(f"{p['label']} ({_plural(p['trials'], 'trial')})" for p in parts)
        if kind == "drug_drug":
            out.append(
                f"The drugs most often given in the same arm as {pin['label']} are {listing}; "
                f"{_plural(pin['partners_total_derived'], 'different drug')} appear with it in total."
            )
        else:
            out.append(f"{pin['label']} is most often linked to {listing}.")
    if not out:
        edges = (facts.get("top_edges") or [])[:3]
        joiner = " + " if kind == "drug_drug" else " – "
        pairs = ", ".join(f"{e['source']}{joiner}{e['target']} ({_plural(e['weight'], 'trial')})" for e in edges)
        lead = {
            "drug_drug": "The drugs most often given together in the same arm are",
            "sponsor_drug": "The sponsor–drug pairs with the most trials are",
            "drug_condition": "The drug–condition pairs with the most trials are",
            "sponsor_collaborator": "The most frequent sponsor–collaborator pairs are",
            "site_investigator": "The most frequent investigator–site pairs are",
        }.get(kind, "The strongest links are")
        if pairs:
            out.append(f"{lead} {pairs}.")
        first_type = next(iter(facts.get("type_labels") or {}), None)
        top = [n for n in facts.get("top_nodes") or [] if n["type"] == first_type][:3]
        if top and kind in ("sponsor_drug", "sponsor_collaborator"):
            listing = ", ".join(f"{n['label']} ({_plural(n['size'], 'trial')})" for n in top)
            out.append(f"The sponsors with the most trials here are {listing}.")
    return out


def _checklist_sentences(summary: dict[str, Any]) -> list[str]:
    by = summary.get("by_status") or {}
    total = summary.get("total")
    if not total:
        return []
    return [
        f"Of {fmt_number(total)} {'criterion' if total == 1 else 'criteria'} checked, "
        f"{fmt_number(by.get('met', 0))} {'is' if by.get('met', 0) == 1 else 'are'} met, "
        f"{fmt_number(by.get('unmet', 0))} would exclude you, "
        f"{fmt_number(by.get('needs_review', 0))} need review with the study team and "
        f"{fmt_number(by.get('unknown', 0))} cannot be checked from the information given."
    ]


# Fact keys whose value is the number of trials the chart was computed from.
_ANALYZED_FACT_KEYS = ("trials_analyzed_derived", "trials_total_derived", "trials_analyzed", "studies_analyzed")


def _count_is_fact(facts: dict[str, Any], n: int) -> bool:
    """True when ``n`` is reported by the handler as its analyzed-trial count (so the summary may state it)."""
    return any(_is_number(facts.get(k)) and facts[k] == n for k in _ANALYZED_FACT_KEYS)


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _heatmap_sentences(f: dict[str, Any]) -> list[str]:
    out = []
    if f.get("max_cell_count"):
        out.append(
            f"The largest cell is {f['max_cell_row']} × {f['max_cell_column']} "
            f"({_plural(f['max_cell_count'], 'trial')})."
        )
    gaps, cells = f.get("gap_cells_derived"), f.get("cells_derived")
    if gaps and cells:
        examples = "; ".join((f.get("gaps") or [])[:3])
        out.append(
            f"{fmt_number(gaps)} of {fmt_number(cells)} combinations have no trials"
            + (f" (e.g. {examples})." if examples else ".")
        )
    return out


def _enrollment_sentences(f: dict[str, Any]) -> list[str]:
    stats = f.get("stats_derived") or {}
    per = stats if "actual" not in stats else {"": stats}
    out = []
    for label, st in list(per.items())[:MAX_GROUPS]:
        a, e = st.get("actual") or {}, st.get("estimated") or {}
        prefix = f"{label}: " if label else ""
        if a.get("n"):
            out.append(
                f"{prefix}median actual enrollment is {fmt_number(a['median'])} participants across "
                f"{_plural(a['n'], 'trial')} that report actual enrollment"
                + (
                    f"; trials still planning enrollment target a median of {fmt_number(e['median'])} "
                    f"({_plural(e['n'], 'trial')})."
                    if e.get("n")
                    else "."
                )
            )
        elif e.get("n"):
            out.append(
                f"{prefix}planned enrollment has a median of {fmt_number(e['median'])} participants "
                f"({_plural(e['n'], 'trial')})."
            )
    return [o[0].upper() + o[1:] for o in out]


def _activity_sentences(f: dict[str, Any]) -> list[str]:
    total, shown = f.get("trials_total_derived"), f.get("trials_shown_derived")
    if not total:
        return []
    out = [
        f"The timeline shows the {_plural(shown, 'most recently started trial')} of {fmt_number(total)}."
        if shown and shown < total
        else f"The timeline shows all {_plural(total, 'trial')}."
    ]
    statuses = sorted((f.get("trials_by_status_derived") or {}).items(), key=lambda kv: -kv[1])[:3]
    if statuses:
        out.append("Of those shown: " + ", ".join(f"{fmt_number(n)} {k.lower()}" for k, n in statuses) + ".")
    return out


def _flow_sentences(f: dict[str, Any]) -> list[str]:
    arms: dict[str, dict[str, Any]] = {}
    for k, v in f.items():
        if "|" in k:
            arm, _, what = k.rpartition("|")
            arms.setdefault(arm, {})[what] = v
    out = []
    for arm, a in list(arms.items())[:MAX_GROUPS]:
        if a.get("started") and a.get("completed") is not None:
            rate = f" ({fmt_number(a['dropout_rate_pct'])}% did not complete)" if "dropout_rate_pct" in a else ""
            out.append(
                f"In {arm}, {fmt_number(a['completed'])} of {fmt_number(a['started'])} participants completed{rate}."
            )
    return out


def _names_text(c: dict[str, Any]) -> str:
    names = ", ".join(c.get("trial_names") or [])
    more = c.get("names_more") or 0
    return f"{names} and {fmt_number(more)} more" if more else names


def _cohort_effect_sentences(c: dict[str, Any]) -> list[str]:
    """Cross-trial forest: one sentence over ALL rows (CI position vs no effect), one on the estimate range."""
    n, trials, null = c.get("estimates"), c.get("trials"), c.get("null_value")
    if not n or not trials or null is None:
        return []
    ep = f" for {c['endpoint_name']}" if c.get("endpoint_name") else ""
    of_n = "" if n == trials else f"of {_plural(n, 'estimate')}, "
    nv = fmt_number(null)
    parts = [
        (c.get("ci_below_null") or 0, f"lies entirely below {nv}"),
        (c.get("ci_above_null") or 0, f"lies entirely above {nv}"),
        (c.get("ci_crosses_null") or 0, f"crosses {nv}"),
    ]
    nonzero = [(k, where) for k, where in parts if k]
    if len(nonzero) == 1:
        position = f"the confidence interval {nonzero[0][1]} for all of them"
    else:
        listed = [f"{where} for {fmt_number(k)}" for k, where in nonzero]
        position = "the confidence interval " + ", ".join(listed[:-1]) + " and " + listed[-1]
    out = [f"{c['what']}{ep} reported by {_plural(trials, 'trial')} ({_names_text(c)}): {of_n}{position}."]
    lo, hi = c.get("estimate_low"), c.get("estimate_high")
    if _is_number(lo) and _is_number(hi) and lo != hi:
        unit = f" {c['unit']}" if c.get("unit") else ""
        out.append(f"Estimates range from {fmt_number(lo)} to {fmt_number(hi)}{unit}.")
    return out


def _cohort_curve_sentences(c: dict[str, Any]) -> list[str]:
    """Cross-trial survival curves: coverage over all drawn trials, then the range of reported medians."""
    drawn, total = c.get("trials_drawn"), c.get("trials_with_results")
    if not drawn:
        return []
    ep = c.get("endpoint_name") or c.get("endpoint") or "survival"
    total = total or drawn
    share = (
        f"all {_plural(total, 'analysed trial')}"
        if drawn == total
        else f"{fmt_number(drawn)} of the {_plural(total, 'analysed trial')}"
    )
    out = [f"Approximate {ep} curves are drawn for {share} with posted results ({_names_text(c)})."]
    m = c.get("medians_reported") or 0
    lo, hi = c.get("median_low_months"), c.get("median_high_months")
    if m == 1:
        out.append(f"The one reported median {ep} is {fmt_number(lo)} months.")
    elif m > 1 and lo == hi:
        out.append(f"Every reported median {ep} is {fmt_number(lo)} months ({_plural(m, 'arm')}).")
    elif m > 1:
        out.append(
            f"Reported median {ep} ranges from {fmt_number(lo)} to {fmt_number(hi)} months ({_plural(m, 'arm')})."
        )
    if c.get("landmark_arms"):
        k, llo, lhi = c["landmark_arms"], c["landmark_low"], c["landmark_high"]

        def pct(frac: float, key: str) -> str:  # the percent fact (if any) is what the summary checker matches
            return f"{c[key]:.1f}%" if _is_number(c.get(key)) else _pct(frac)

        lo_txt, hi_txt = pct(llo, "landmark_low_pct"), pct(lhi, "landmark_high_pct")
        rate = f"is {lo_txt}" if k == 1 or llo == lhi else f"ranges from {lo_txt} to {hi_txt}"
        out.append(f"At {fmt_number(c['landmark_month'])} months, the reported {ep} rate {rate} ({_plural(k, 'arm')}).")
    elif c.get("arms_median_not_reached"):
        out.append(
            f"{_plural(c['arms_median_not_reached'], 'arm')} had not reached the median {ep} when results were posted."
        )
    return out


def _survival_sentences(f: dict[str, Any]) -> list[str]:
    if isinstance(f.get("cohort_curves_derived"), dict):
        return _cohort_curve_sentences(f["cohort_curves_derived"])
    if isinstance(f.get("cohort_effects_derived"), dict):
        return _cohort_effect_sentences(f["cohort_effects_derived"])
    out: list[str] = []
    medians = f.get("medians_months_derived") or {}
    if medians:
        listed = ", ".join(f"{series} {fmt_number(v)} months" for series, v in list(medians.items())[:4])
        out.append(f"Reported medians: {listed}.")
    land: dict[tuple[str, str], list[str]] = {}
    for k, v in f.items():
        parts = k.split("|")
        if len(parts) == 3 and parts[1] == "landmark" and _is_number(v):
            endpoint, _, arm = parts[0].partition(" · ")
            land.setdefault((endpoint, parts[2]), []).append(f"{arm} {_pct(v)}")
    out += [f"{ep} at month {t}: {', '.join(vals)}." for (ep, t), vals in list(land.items())[:2]]
    out += [f"{k}: {v}." for k, v in f.items() if isinstance(v, str) and "Hazard Ratio" in v][:2]
    return out


def _progression_sentences(f: dict[str, Any]) -> list[str]:
    phases = f.get("phases") or {}
    out = []
    for label, p in list(phases.items())[:3]:
        out.append(
            f"{label}: first registered trial started {p['first_start_raw']} ({p['first_nct']}), "
            f"{_plural(p['trial_count'], 'trial')} in total."
        )
    return out


def _visit_sentences(f: dict[str, Any]) -> list[str]:
    n = f.get("assessment_timepoints")
    if not n:
        return []
    return [
        f"The registry lists {_plural(n, 'distinct assessment time point')} in the outcome time frames, a lower "
        "bound on the number of visits (the full visit schedule is in the protocol, not the registry)."
    ]


def _effect_sentences(f: dict[str, Any]) -> list[str]:
    if isinstance(f.get("cohort_effects_derived"), dict):
        return _cohort_effect_sentences(f["cohort_effects_derived"])
    if f.get("has_results") is False and _is_number(f.get("primary_outcomes_derived")):
        p, q = f["primary_outcomes_derived"], f.get("secondary_outcomes_derived") or 0
        return [
            f"{f.get('nct_id', 'This trial')} has no results posted yet; the registry lists "
            f"{_plural(p, 'primary outcome measure')} and {_plural(q, 'secondary outcome measure')}."
        ]
    return [f"{k}: {v}." for k, v in f.items() if k != "nct_id" and isinstance(v, str)][:3]


_INTENT_SENTENCES = {
    Intent.PHASE_STATUS_HEATMAP: _heatmap_sentences,
    Intent.LANDSCAPE_GAP_HEATMAP: _heatmap_sentences,
    Intent.ENROLLMENT_DISTRIBUTION: _enrollment_sentences,
    Intent.SPONSOR_ACTIVITY: _activity_sentences,
    Intent.DROPOUT_FLOW: _flow_sentences,
    Intent.SURVIVAL_SUMMARY: _survival_sentences,
    Intent.RESULTS_OUTCOMES: _effect_sentences,
    Intent.DEVELOPMENT_PROGRESSION: _progression_sentences,
    Intent.VISIT_BURDEN: _visit_sentences,
}


def _closing(result: HandlerResult, counting: str, tail: str, plan: QueryPlan | None = None) -> str | None:
    phrases = {status_phrase(cf.statuses) for cf in (plan.cohorts if plan else [])}
    phrase = phrases.pop() if len(phrases) == 1 else None
    if phrase:  # "status today", so nobody reads a per-start-year chart as "recruiting in that year"
        tail = f" ({phrase})" + tail
    n = result.studies_analyzed
    nct = result.facts.get("nct_id")
    if isinstance(nct, str):
        what = "registered protocol" if result.facts.get("has_results") is False else "results posted"
        return f"Source: the {what} for {nct} on ClinicalTrials.gov" + tail
    if n and _count_is_fact(result.facts, n) and counting == "registry_counts":
        return f"Exact registry counts over all {fmt_number(n)} matching trials on ClinicalTrials.gov" + tail
    if n and _count_is_fact(result.facts, n) and counting == "sample":
        return f"Based on a sample of {fmt_number(n)} matching trials, not all of them" + tail
    if n and _count_is_fact(result.facts, n):
        return f"Based on {fmt_number(n)} {'trial' if n == 1 else 'trials'} from ClinicalTrials.gov" + tail
    if n:  # the count is coverage metadata, not a chart/fact value: say it without a number
        return "Based on the matching trials retrieved from ClinicalTrials.gov" + tail
    return None


def template_summary(
    plan: QueryPlan, result: HandlerResult, viz: Visualization | None = None, counting: str = "all_records"
) -> str | None:
    """Build 1–4 sentences from typed facts or the final chart rows; ``None`` for empty/zero states."""
    viz = viz or result.visualization
    if _is_empty_state(viz):
        return None
    if viz.type is VizType.NETWORK_GRAPH and result.facts.get("network_kind"):
        sentences = _network_sentences(result.facts)
        n = result.studies_analyzed
        if sentences and n and _count_is_fact(result.facts, n):
            sentences.append(
                f"Based on {_plural(n, 'trial')} from ClinicalTrials.gov"
                + (" (a sample of the matching trials)." if counting == "sample" else ".")
            )
        return " ".join(sentences[:MAX_SENTENCES]) or None
    if viz.type is VizType.ELIGIBILITY_CHECKLIST and viz.meta.summary:
        return " ".join(_checklist_sentences(viz.meta.summary)) or None
    patient = plan.persona is Persona.PATIENT
    tail = "; every value links to its source record." if patient else "."
    special = _INTENT_SENTENCES.get(plan.intent)
    if special is not None and (sentences := special(result.facts)):
        closing = _closing(result, counting, tail, plan)
        return " ".join([*sentences[: MAX_SENTENCES - 1], *([closing] if closing else [])])
    unit = _unit(viz)
    buckets: dict[str, list[tuple[str, float]]] = {}
    scalars: dict[str, float] = {}
    for key, value in result.facts.items():
        if not _is_number(value):
            continue
        if "|" in str(key):
            group, _, label = str(key).rpartition("|")
            buckets.setdefault(group, []).append((label, value))
        else:
            scalars[str(key)] = value
    # bucket facts first, else the chart rows; the analyzed-trial count is said once, in the closing sentence.
    # "Largest bucket" sentences only make sense for count distributions (not survival points or flow counts).
    sentences = (
        _distribution_sentences(buckets or _row_groups(viz), unit, patient, viz)
        if viz.type in DISTRIBUTION_TYPES
        else []
    )
    sentences += _scalar_sentences({k: v for k, v in scalars.items() if k not in _ANALYZED_FACT_KEYS})
    if not sentences:
        return None  # nothing non-zero to say
    closing = _closing(result, counting, tail, plan)
    return " ".join([*sentences[: MAX_SENTENCES - 1], *([closing] if closing else [])])


async def synthesize(
    plan: QueryPlan,
    result: HandlerResult,
    *,
    llm: Any = None,
    settings: Any = None,
    viz: Visualization | None = None,
    counting: str = "all_records",
) -> str | None:
    """Deterministic answer summary for the final ``viz`` (``llm``/``settings`` kept for interface stability)."""
    return template_summary(plan, result, viz, counting)
