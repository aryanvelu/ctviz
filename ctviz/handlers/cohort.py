"""Cohort / landscape intent handlers.

``phase_distribution`` is the reference implementation every other handler follows:
1. iterate studies, 2. bucket with provenance refs, 3. emit cited datums, 4. build the viz,
5. return notes + compact facts for the Synthesis agent.

Conventions shared by all handlers below:

* A datum cites exactly the fields that put a trial into its bucket (e.g. the start date for a
  time bucket, ``phases`` + ``overallStatus`` for a heatmap cell, the first matching
  ``locations[i].country`` for a country). When a field is absent the trial's ``nctId`` is cited.
* A trial counts at most once per bucket (:class:`Bucket`).
* Comparison scope (2+ cohorts) adds a ``cohort`` field and an ``encoding.series`` channel.
* ``HandlerResult.facts`` holds the headline numbers for the deterministic summary template. Rule:
  every number in facts equals a value in ``visualization.data`` *unless* some key on its path ends
  in ``_derived`` — those are documented aggregates (sums, distinct counts, medians, excluded-trial
  counts, matrix sizes) computed here from the same studies. ``tests/unit/test_cohort_facts.py``
  enforces this for every handler.
"""

from __future__ import annotations

import functools
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ctviz.api.schemas.plan import Intent, Scope
from ctviz.api.schemas.response import Note, NoteCode
from ctviz.api.schemas.viz import (
    Channel,
    Encoding,
    GeoSpec,
    GraphData,
    Layout,
    Scale,
    Sort,
    TimeAxis,
    Truncation,
    VizMeta,
    VizType,
)
from ctviz.compute.cohort import (
    ENROLLMENT_BINS,
    Granularity,
    LabelPicker,
    PartialDate,
    bucket_label,
    bucket_range,
    bucket_start,
    country_key,
    enrollment_bin,
    enrollment_bin_label,
    is_placebo,
    median,
    normalize_name,
    parse_partial_date,
    split_arm_intervention,
    supports_granularity,
    top_keys,
)
from ctviz.compute.network import CooccurrenceGraph, partners
from ctviz.core.entities import get_resolver
from ctviz.core.provenance import Dataset, FieldRef, GraphDataset, Study
from ctviz.ctgov.fields import (
    PHASE_LABELS,
    PHASE_ORDER,
    STATUS_GROUP_LABELS,
    STATUS_GROUPS,
    STATUS_ORDER,
    P,
    status_group,
    status_phrase,
)
from ctviz.handlers import registry_counts
from ctviz.obs.trace import current_trace
from ctviz.viz.base import HandlerContext, HandlerResult, make_viz, note, register
from ctviz.viz.builders.bar_chart import bar_chart, ordered
from ctviz.viz.builders.geo_map import geo_map
from ctviz.viz.builders.heatmap import heatmap
from ctviz.viz.builders.histogram import histogram
from ctviz.viz.builders.network_graph import network_graph
from ctviz.viz.builders.pie_chart import pie_chart
from ctviz.viz.builders.scatter_plot import scatter_plot
from ctviz.viz.builders.time_series import time_series
from ctviz.viz.builders.timeline import timeline


def phase_bucket(study: Study) -> tuple[str, FieldRef | None]:
    """Joined phase label (e.g. 'PHASE1/PHASE2'); 'NA' when absent. Ref cites designModule.phases."""
    phases = study.get("phases") or []
    return ("/".join(phases) if phases else "NA"), study.ref("phases")


@register(Intent.PHASE_DISTRIBUTION)
async def phase_distribution(ctx: HandlerContext) -> HandlerResult:
    if registry_counts.active(ctx):
        return await registry_counts.phases(ctx)
    ds = Dataset()
    comparison = ctx.plan.scope is Scope.COMPARISON and len(ctx.cohorts) > 1
    counts: dict[tuple[str, str], list[FieldRef]] = defaultdict(list)
    for cohort in ctx.cohorts:
        for s in {st.nct_id: st for st in cohort.studies}.values():  # dedupe within a cohort
            phase, ref = phase_bucket(s)
            # A study with no phases field still counts (as NA); cite its NCT id field instead.
            counts[(cohort.label, phase)].append(ref or s.ref("nct_id"))
    present = [p for p in PHASE_ORDER if any(k[1] == p for k in counts)]
    present += sorted({k[1] for k in counts} - set(present))
    for cohort in ctx.cohorts:
        for phase in present:
            refs = counts.get((cohort.label, phase), [])
            if not refs and not comparison:
                continue
            row = {"phase": PHASE_LABELS.get(phase, phase), "trial_count": len(refs)}
            if comparison:
                row["cohort"] = cohort.label
            ds.emit(row, refs)
    x = Channel(field="phase", type="ordinal", title="Phase", sort=ordered([PHASE_LABELS.get(p, p) for p in present]))
    y = Channel(field="trial_count", type="quantitative", title="Trials", unit="trials", format=",d")
    series = Channel(field="cohort", type="nominal", title="Cohort") if comparison else None
    labels = " vs ".join(c.label for c in ctx.cohorts)
    viz = bar_chart(
        ds.rows,
        title=f"Trials by phase: {labels}",
        x=x,
        y=y,
        series=series,
        meta=VizMeta(
            grouping={
                "by": ["phase"],
                "aggregate": "count_distinct_trials",
                "phase_combination": "joined",
                "each_trial_counted_once": True,
            }
        ),
    )
    notes = []
    if any(c.sampled for c in ctx.cohorts):
        notes.append(note(NoteCode.SAMPLED, "Counts reflect a sample of the matching trials; see meta.coverage."))
    facts: dict[str, Any] = {
        f"{c.label}|{PHASE_LABELS.get(p, p)}": len(counts.get((c.label, p), [])) for c in ctx.cohorts for p in present
    }
    facts["trials_analyzed_derived"] = sum(len(c.studies) for c in ctx.cohorts)
    return HandlerResult(
        visualization=viz,
        notes=notes,
        facts=facts,
        full_refs=ds.full_refs,
        studies_analyzed=sum(len(c.studies) for c in ctx.cohorts),
    )


# ================================================================================================
# Shared machinery
# ================================================================================================

STATUS_LABELS = {
    "NOT_YET_RECRUITING": "Not yet recruiting",
    "RECRUITING": "Recruiting",
    "ENROLLING_BY_INVITATION": "Enrolling by invitation",
    "ACTIVE_NOT_RECRUITING": "Active, not recruiting",
    "SUSPENDED": "Suspended",
    "COMPLETED": "Completed",
    "TERMINATED": "Terminated",
    "WITHDRAWN": "Withdrawn",
    "UNKNOWN": "Unknown status",
    "AVAILABLE": "Available",
    "NO_LONGER_AVAILABLE": "No longer available",
    "TEMPORARILY_NOT_AVAILABLE": "Temporarily not available",
    "APPROVED_FOR_MARKETING": "Approved for marketing",
    "WITHHELD": "Withheld",
}
SPONSOR_CLASS_LABELS = {
    "INDUSTRY": "Industry",
    "NIH": "NIH",
    "FED": "Other U.S. federal",
    "OTHER_GOV": "Other government",
    "OTHER": "Other (academic, hospital, non-profit)",
    "NETWORK": "Network",
    "INDIV": "Individual",
    "AMBIG": "Ambiguous",
    "UNKNOWN": "Unknown",
}
INTERVENTION_TYPE_LABELS = {
    "DRUG": "Drug",
    "BIOLOGICAL": "Biological",
    "DEVICE": "Device",
    "PROCEDURE": "Procedure",
    "BEHAVIORAL": "Behavioral",
    "DIETARY_SUPPLEMENT": "Dietary supplement",
    "RADIATION": "Radiation",
    "GENETIC": "Genetic",
    "DIAGNOSTIC_TEST": "Diagnostic test",
    "COMBINATION_PRODUCT": "Combination product",
    "OTHER": "Other",
}
DRUG_TYPES = {"DRUG", "BIOLOGICAL"}
MIN_YEAR = 1980  # earlier start years in a gap matrix are treated as registry errors
SPLIT_PHASES = ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"]
NOT_REPORTED = "Not reported"

COUNT = Channel(field="trial_count", type="quantitative", title="Trials", unit="trials", format=",d")
COHORT = Channel(field="cohort", type="nominal", title="Cohort")


@dataclass
class Bucket:
    """Distinct trials in one bucket plus the evidence that placed each of them there."""

    trials: set[str] = field(default_factory=set)
    refs: list[FieldRef] = field(default_factory=list)

    def add(self, nct_id: str, refs: list[FieldRef | None]) -> None:
        if nct_id in self.trials:
            return
        self.trials.add(nct_id)
        self.refs.extend(r for r in refs if r is not None)

    @property
    def count(self) -> int:
        return len(self.trials)


# (key, raw display label, evidence) — one entry per distinct key in the study.
Item = tuple[str, str, list[FieldRef | None]]
Extractor = Callable[[Study], list[Item]]


def _traced(name: str):
    """Record the handler as a trace step with row/study counts."""

    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(ctx: HandlerContext) -> HandlerResult:
            with current_trace().step(f"handler.{name}", studies=len(ctx.studies)) as info:
                result = await fn(ctx)
                data = result.visualization.data
                if isinstance(data, GraphData):
                    info.update(nodes=len(data.nodes), edges=len(data.edges))
                else:
                    info["rows"] = len(data)
                info["viz"] = result.visualization.type.value
                return result

        return wrapper

    return deco


def _comparison(ctx: HandlerContext) -> bool:
    return ctx.plan.scope is Scope.COMPARISON and len(ctx.cohorts) > 1


def _groups(ctx: HandlerContext) -> list[tuple[str, list[Study]]]:
    """Per-cohort study lists in comparison scope; otherwise one merged group. Each group is de-duplicated
    by nct_id (a cohort page can repeat a study), so every per-trial count below counts a trial once."""
    if _comparison(ctx):
        return [(c.label, _dedupe(c.studies)) for c in ctx.cohorts]
    label = " + ".join(c.label for c in ctx.cohorts) or "Selected trials"
    return [(label, _dedupe(ctx.studies))]


def _dedupe(studies: list[Study]) -> list[Study]:
    seen: dict[str, Study] = {}
    for s in studies:
        seen.setdefault(s.nct_id, s)
    return list(seen.values())


def _title_label(ctx: HandlerContext) -> str:
    label = " vs ".join(g for g, _ in _groups(ctx))
    return _with_status(ctx, label)


def _with_status(ctx: HandlerContext, label: str) -> str:
    """Append the status filter as "(status today: ...)": a 2019 trial that is recruiting today may not have
    been recruiting in 2019, so titles never say "recruiting trials per year"."""
    phrases = {status_phrase(cf.statuses) for cf in ctx.plan.cohorts}
    phrase = phrases.pop() if len(phrases) == 1 else None
    return f"{label} ({phrase})" if phrase else label


def _base_notes(ctx: HandlerContext) -> list[Note]:
    notes: list[Note] = []
    if any(c.sampled for c in ctx.cohorts):
        notes.append(note(NoteCode.SAMPLED, "Counts reflect a sample of the matching trials; see meta.coverage."))
    return notes


def _analyzed(ctx: HandlerContext) -> int:
    return sum(len(s) for _, s in _groups(ctx))


def _peak(counts: dict[str, int]) -> dict[str, Any]:
    """Largest bucket of one series (latest period wins ties)."""
    period, count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return {"period": period, "count": count}


def _top_named(counts: dict[str, int], other_label: str | None) -> dict[str, Any] | None:
    """Largest named (non-'Other') category; first in display order wins ties."""
    named = [(k, v) for k, v in counts.items() if k != other_label and v > 0]
    if not named:
        return None
    k, v = max(named, key=lambda kv: kv[1])
    return {"category": k, "count": v}


def _top_n(ctx: HandlerContext, default: int) -> int:
    return ctx.plan.top_n or ctx.request.top_n or default


def _tally(
    groups: list[tuple[str, list[Study]]], extract: Extractor, labels: LabelPicker
) -> tuple[dict[str, dict[str, Bucket]], set[str]]:
    """Bucket every study of every group; returns (group -> key -> Bucket, nct ids without a value).

    The second value is a set, so a trial that sits in two compared cohorts is reported once."""
    out: dict[str, dict[str, Bucket]] = {}
    missing: set[str] = set()
    for glabel, studies in groups:
        buckets: dict[str, Bucket] = defaultdict(Bucket)
        for s in studies:
            items = extract(s)
            if not items:
                missing.add(s.nct_id)
            for key, raw, refs in items:
                labels.add(key, raw)
                buckets[key].add(s.nct_id, refs)
        out[glabel] = buckets
    return out, missing


# -- extractors ------------------------------------------------------------------------------------


def _ex_status(s: Study) -> list[Item]:
    status = s.get("status")
    if not isinstance(status, str):
        return [(NOT_REPORTED, NOT_REPORTED, [s.ref("nct_id")])]
    return [(status, STATUS_LABELS.get(status, status), [s.ref("status")])]


def _ex_status_group(s: Study) -> list[Item]:
    status = s.get("status")
    group = status_group(status if isinstance(status, str) else None)
    return [(group, STATUS_GROUP_LABELS[group], [s.ref("status") or s.ref("nct_id")])]


def _ex_phase_joined(s: Study) -> list[Item]:
    phase, ref = phase_bucket(s)
    return [(phase, PHASE_LABELS.get(phase, phase), [ref or s.ref("nct_id")])]


def _ex_phase_split(s: Study) -> list[Item]:
    phases = s.get("phases") or []
    ref = s.ref("phases")
    if not phases:
        return [("NA", PHASE_LABELS["NA"], [s.ref("nct_id")])]
    return [(p, PHASE_LABELS.get(p, p), [ref]) for p in dict.fromkeys(phases)]


def _ex_sponsor(s: Study) -> list[Item]:
    name = s.get("lead_sponsor")
    key = normalize_name(name)
    return [(key, name, [s.ref("lead_sponsor")])] if key else []


def _ex_sponsor_class(s: Study) -> list[Item]:
    cls = s.get("lead_sponsor_class")
    if not isinstance(cls, str):
        return []
    return [(cls, SPONSOR_CLASS_LABELS.get(cls, cls), [s.ref("lead_sponsor_class")])]


def _drug_items(s: Study, types: set[str] | None = None, *, with_type: bool = False) -> list[Item]:
    """Intervention names (placebo excluded), first index per normalized name."""
    out: dict[str, Item] = {}
    for _, item, path in s.items("interventions"):
        if not isinstance(item, dict):
            continue
        name, itype = item.get("name"), item.get("type")
        if types is not None and itype not in types:
            continue
        if not isinstance(name, str) or is_placebo(name):
            continue
        key = normalize_name(canonical_drug(name) or name)  # variants of one drug count as one
        if not key or key in out:
            continue
        refs = [s.ref(f"{path}.name")] + ([s.ref(f"{path}.type")] if with_type else [])
        out[key] = (key, drug_label(name), refs)
    return list(out.values())


def _ex_intervention(s: Study) -> list[Item]:
    return _drug_items(s)


def _ex_intervention_type(s: Study) -> list[Item]:
    out: dict[str, Item] = {}
    for _, item, path in s.items("interventions"):
        itype = item.get("type") if isinstance(item, dict) else None
        if isinstance(itype, str) and itype not in out:
            out[itype] = (itype, INTERVENTION_TYPE_LABELS.get(itype, itype), [s.ref(f"{path}.type")])
    return list(out.values())


def _ex_condition(s: Study) -> list[Item]:
    out: dict[str, Item] = {}
    for _, cond, path in s.items("conditions"):
        key = normalize_name(cond)
        if key and key not in out:
            out[key] = (key, cond, [s.ref(path)])
    return list(out.values())


def _ex_country(s: Study) -> list[Item]:
    """Countries of a study's sites, grouped by :func:`country_key`; the citation is the first location
    index per country (its excerpt is the raw registry value)."""
    out: dict[str, Item] = {}
    for _, loc, path in s.items("locations"):
        country = loc.get("country") if isinstance(loc, dict) else None
        key = country_key(country)
        if key and key not in out:
            out[key] = (key, " ".join(country.split()), [s.ref(f"{path}.country")])
    return list(out.values())


def _ex_start_year(s: Study) -> list[Item]:
    pd = parse_partial_date(s.get("start_date"))
    return [(str(pd.value.year), str(pd.value.year), [s.ref("start_date")])] if pd else []


EXTRACTORS: dict[str, tuple[Extractor, str]] = {
    "status": (_ex_status, "Status"),
    "phase": (_ex_phase_split, "Phase"),
    "sponsor": (_ex_sponsor, "Lead sponsor"),
    "sponsor_class": (_ex_sponsor_class, "Sponsor class"),
    "intervention": (_ex_intervention, "Intervention"),
    "intervention_type": (_ex_intervention_type, "Intervention type"),
    "condition": (_ex_condition, "Condition"),
    "country": (_ex_country, "Country"),
    "year": (_ex_start_year, "Start year"),
}


# -- categorical bar / pie -------------------------------------------------------------------------


def _categorical(
    ctx: HandlerContext,
    *,
    extract: Extractor,
    field_name: str,
    axis_title: str,
    title: str,
    order: list[str] | None = None,
    top_n: int | None = None,
    horizontal: bool = False,
    allow_pie: bool = True,
    grouping: dict[str, Any] | None = None,
    extra_notes: list[Note] | None = None,
) -> HandlerResult:
    """Count trials per category (optionally per cohort) and build a bar (or preferred pie) chart.

    ``order`` fixes the category order (e.g. statuses); otherwise categories are ranked by total
    count and, with ``top_n``, the remainder is folded into an 'Other' bucket of distinct trials.
    """
    comparison = _comparison(ctx)
    groups = _groups(ctx)
    labels = LabelPicker()
    tallies, missing = _tally(groups, extract, labels)
    totals: Counter[str] = Counter()
    for buckets in tallies.values():
        for k, b in buckets.items():
            totals[k] += b.count
    if order is not None:
        keys = [k for k in order if totals.get(k)] + sorted(set(totals) - set(order))
        rest: list[str] = []
    else:
        ranked = top_keys(dict(totals), len(totals))
        keys = ranked[:top_n] if top_n else ranked
        rest = ranked[len(keys) :]
    ds = Dataset()
    other_label = f"Other ({len(rest)} more)" if rest else None
    facts: dict[str, Any] = {"trials_analyzed_derived": _analyzed(ctx), "categories_total_derived": len(totals)}
    per_group_facts: dict[str, dict[str, int]] = {}
    for glabel, buckets in tallies.items():
        gfacts: dict[str, int] = {}
        for k in keys:
            b = buckets.get(k)
            if (b is None or not b.count) and not comparison:
                continue
            row = {field_name: labels.label(k), "trial_count": b.count if b else 0}
            if comparison:
                row["cohort"] = glabel
            ds.emit(row, b.refs if b else [])
            gfacts[row[field_name]] = row["trial_count"]
        if rest:
            trials: set[str] = set()
            refs: list[FieldRef] = []
            for k in rest:
                if k in buckets:
                    trials |= buckets[k].trials
                    refs.extend(buckets[k].refs)
            if trials or comparison:
                row = {field_name: other_label, "trial_count": len(trials), "is_other": True}
                if comparison:
                    row["cohort"] = glabel
                ds.emit(row, refs)
                gfacts[other_label] = len(trials)
        per_group_facts[glabel] = gfacts
    shown = [labels.label(k) for k in keys] + ([other_label] if other_label else [])
    if comparison:
        facts["by_cohort"] = per_group_facts
        facts["trials_by_cohort_derived"] = {g: len(s) for g, s in groups}
        facts["by_cohort_top"] = {g: _top_named(v, other_label) for g, v in per_group_facts.items()}
    else:
        only = next(iter(per_group_facts.values()), {})
        facts["counts"] = only
        top = _top_named(only, other_label)
        if top:
            facts["top_category"], facts["top_category_count"] = top["category"], top["count"]
    missing_total = len(missing)
    if missing_total:
        facts["trials_without_value_derived"] = missing_total
    notes = _base_notes(ctx) + list(extra_notes or [])
    if missing_total:
        notes.append(
            note(
                NoteCode.NOT_IN_REGISTRY,
                f"{missing_total} trial(s) do not report a {axis_title.lower()} and are not counted.",
            )
        )
    grouping = {"by": [field_name], "aggregate": "count_distinct_trials", **(grouping or {})}
    if not grouping.get("multi_valued"):
        grouping["each_trial_counted_once"] = True
    meta = VizMeta(grouping=grouping)
    if rest:
        meta.truncation = Truncation(top_n=len(keys), other_bucket=True, omitted_count=len(rest))
        notes.append(
            note(
                NoteCode.TRUNCATED,
                f"Showing the top {len(keys)} of {len(totals)} categories; the rest are grouped as 'Other'.",
            )
        )
    if allow_pie and not comparison and ctx.plan.preferred_visualization is VizType.PIE_CHART and ds.rows:
        viz = pie_chart(
            ds.rows,
            title=title,
            label=Channel(field=field_name, type="nominal", title=axis_title),
            value=COUNT,
            meta=meta,
        )
    else:
        x = Channel(
            field=field_name, type="ordinal" if order is not None else "nominal", title=axis_title, sort=ordered(shown)
        )
        viz = bar_chart(
            ds.rows,
            title=title,
            x=x,
            y=COUNT,
            series=COHORT if comparison else None,
            meta=meta,
            orientation="horizontal" if horizontal else "vertical",
        )
    return HandlerResult(
        visualization=viz, notes=notes, facts=facts, full_refs=ds.full_refs, studies_analyzed=_analyzed(ctx)
    )


# ================================================================================================
# trend_over_time
# ================================================================================================


@register(Intent.TREND_OVER_TIME)
@_traced("trend_over_time")
async def trend_over_time(ctx: HandlerContext) -> HandlerResult:
    """Trials started per year/quarter/month; split by cohort (comparison) or plan.series_by."""
    if registry_counts.active(ctx):
        return await registry_counts.trend(ctx)
    gran: Granularity = ctx.plan.time_granularity or ctx.request.time_granularity or "year"
    comparison = _comparison(ctx)
    series_by = (
        None
        if comparison
        else (ctx.plan.series_by if ctx.plan.series_by in ("phase", "status", "status_group") else None)
    )
    splitter: Extractor | None = {
        "phase": _ex_phase_joined,
        "status": _ex_status,
        "status_group": _ex_status_group,
    }.get(series_by or "")
    buckets: dict[tuple[str, Any], Bucket] = defaultdict(Bucket)
    series_labels: dict[str, str] = {}
    no_date_ids: set[str] = set()
    imprecise_ids: set[str] = set()
    estimated_ids: set[str] = set()
    estimated_in: dict[tuple[str, Any], set[str]] = defaultdict(set)  # bucket -> trials with a planned start
    for glabel, studies in _groups(ctx):
        series_labels.setdefault(glabel, glabel)
        for s in studies:
            pd = parse_partial_date(s.get("start_date"))
            if pd is None:
                no_date_ids.add(s.nct_id)
                continue
            if not supports_granularity(pd, gran):
                imprecise_ids.add(s.nct_id)
                continue
            refs: list[FieldRef | None] = [s.ref("start_date")]
            skey = glabel
            if splitter is not None:
                key, raw, srefs = splitter(s)[0]
                skey = key
                series_labels[key] = raw
                refs += srefs
            bkey = (skey, bucket_start(pd.value, gran))
            buckets[bkey].add(s.nct_id, refs)
            if s.get("start_date_type") == "ESTIMATED":
                estimated_ids.add(s.nct_id)
                estimated_in[bkey].add(s.nct_id)
    ds = Dataset()
    facts: dict[str, Any] = {"granularity": gran, "trials_analyzed_derived": _analyzed(ctx)}
    if buckets:
        starts = [b for _, b in buckets]
        periods = bucket_range(min(starts), max(starts), gran)
        if splitter is not None:
            order = {"phase": PHASE_ORDER, "status": STATUS_ORDER}.get(series_by or "", list(STATUS_GROUPS))
            present = {k for k, _ in buckets}
            skeys = [k for k in order if k in present] + sorted(present - set(order))
        else:
            skeys = [g for g, _ in _groups(ctx)]
        series_field = "cohort" if comparison else series_by
        per_series: dict[str, dict[str, int]] = {}
        for skey in skeys:
            label = series_labels.get(skey, skey)
            counts: dict[str, int] = {}
            for p in periods:
                b = buckets.get((skey, p))
                row: dict[str, Any] = {
                    "period": p.isoformat(),
                    "year": p.year,
                    "label": bucket_label(p, gran),
                    "trial_count": b.count if b else 0,
                    "estimated_count": len(estimated_in.get((skey, p), ())),
                }
                if series_field:
                    row[series_field] = label
                ds.emit(row, b.refs if b else [])
                counts[row["label"]] = row["trial_count"]
            per_series[label] = counts
        facts.update(first_period=bucket_label(periods[0], gran), last_period=bucket_label(periods[-1], gran))
        peaks = {label: _peak(counts) for label, counts in per_series.items()}
        if series_field:
            facts["by_series"] = per_series
            facts["by_series_peak"] = peaks
            facts["series_totals_derived"] = {k: sum(v.values()) for k, v in per_series.items()}
            plotted: set[str] = set()
            for b in buckets.values():
                plotted |= b.trials
            facts["trials_plotted_distinct_derived"] = len(plotted)  # a trial in two cohorts counts once
        else:
            only = next(iter(per_series.values()))
            facts["counts"] = only
            facts.update(
                peak_period=peaks[next(iter(peaks))]["period"],
                peak_count=peaks[next(iter(peaks))]["count"],
                trials_plotted_derived=sum(only.values()),
            )
    no_date, imprecise, estimated = len(no_date_ids), len(imprecise_ids), len(estimated_ids)
    facts.update(trials_without_start_date_derived=no_date, trials_with_estimated_start_derived=estimated)
    notes = _base_notes(ctx)
    if no_date:
        notes.append(note(NoteCode.PARTIAL_DATES, f"{no_date} trial(s) have no start date and are excluded."))
    if imprecise:
        facts["trials_with_imprecise_start_derived"] = imprecise
        notes.append(
            note(
                NoteCode.PARTIAL_DATES,
                f"{imprecise} trial(s) give only a start year, too coarse for {gran} buckets; they are excluded.",
            )
        )
    if estimated:
        notes.append(
            note(
                NoteCode.PARTIAL_DATES,
                f"{estimated} trial(s) have an estimated (anticipated) start "
                "date; future buckets reflect plans, not actual starts.",
            )
        )
    series = None
    if comparison:
        series = COHORT
    elif series_by:
        series = Channel(
            field=series_by,
            type="nominal",
            title={"phase": "Phase", "status": "Status today"}.get(series_by, "Status today (group)"),
        )
    fmt = {"year": "%Y", "quarter": "%Y-%m", "month": "%Y-%m"}[gran]
    viz = time_series(
        ds.rows,
        title=f"Trials started per {gran}: {_title_label(ctx)}",
        x=Channel(field="period", type="temporal", title=f"Start {gran}", time_format=fmt),
        y=COUNT,
        granularity=gran,
        series=series,
        tooltip=["label", "estimated_count"],
        meta=VizMeta(
            grouping={
                "by": ["start_date"],
                "aggregate": "count_distinct_trials",
                "zero_filled": True,
                "each_trial_counted_once": True,
            },
            units={"trial_count": "trials", "estimated_count": "trials"},
        ),
    )
    return HandlerResult(
        visualization=viz, notes=notes, facts=facts, full_refs=ds.full_refs, studies_analyzed=_analyzed(ctx)
    )


# ================================================================================================
# status / sponsor / intervention distributions
# ================================================================================================


@register(Intent.STATUS_DISTRIBUTION)
@_traced("status_distribution")
async def status_distribution(ctx: HandlerContext) -> HandlerResult:
    """Trials per overall recruitment status."""
    if ctx.plan.group_by == "status_group" or ctx.plan.series_by == "status_group":
        if registry_counts.active(ctx):  # too many to download: exact counts per group from the registry
            return await registry_counts.status_groups(ctx)
        return _categorical(
            ctx,
            extract=_ex_status_group,
            field_name="status_group",
            axis_title="Status today",
            title=f"Trials by status today (active / done / other): {_title_label(ctx)}",
            order=list(STATUS_GROUPS),
        )
    if registry_counts.active(ctx):
        return await registry_counts.statuses(ctx, STATUS_LABELS)
    return _categorical(
        ctx,
        extract=_ex_status,
        field_name="status",
        axis_title="Status",
        title=f"Trials by recruitment status: {_title_label(ctx)}",
        order=STATUS_ORDER + [NOT_REPORTED],
    )


@register(Intent.SPONSOR_LANDSCAPE)
@_traced("sponsor_landscape")
async def sponsor_landscape(ctx: HandlerContext) -> HandlerResult:
    """Top lead sponsors (or sponsor classes) by trial count; stacked by phase when series_by=phase."""
    by_class = ctx.plan.group_by == "sponsor_class"
    if by_class and registry_counts.active(ctx):
        return await registry_counts.sponsor_classes(ctx, SPONSOR_CLASS_LABELS)
    if not by_class and ctx.plan.series_by == "phase" and not _comparison(ctx):
        return _sponsor_by_phase(ctx)
    if by_class:
        return _categorical(
            ctx,
            extract=_ex_sponsor_class,
            field_name="sponsor_class",
            axis_title="Sponsor class",
            title=f"Trials by lead sponsor class: {_title_label(ctx)}",
        )
    return _categorical(
        ctx,
        extract=_ex_sponsor,
        field_name="sponsor",
        axis_title="Lead sponsor",
        title=f"Top lead sponsors: {_title_label(ctx)}",
        top_n=_top_n(ctx, 15),
        horizontal=True,
        grouping={"name_normalization": "casefold, punctuation and ®/™ removed; no fuzzy merge"},
    )


def _sponsor_by_phase(ctx: HandlerContext) -> HandlerResult:
    labels = LabelPicker()
    sponsor_totals: Counter[str] = Counter()
    cells: dict[tuple[str, str], Bucket] = defaultdict(Bucket)
    studies = ctx.studies
    for s in studies:
        sp = _ex_sponsor(s)
        if not sp:
            continue
        key, raw, srefs = sp[0]
        labels.add(key, raw)
        phase, _, prefs = _ex_phase_joined(s)[0]
        cells[(key, phase)].add(s.nct_id, srefs + prefs)
        sponsor_totals[key] += 1
    n = _top_n(ctx, 15)
    keys = top_keys(dict(sponsor_totals), n)
    phases = [p for p in PHASE_ORDER if any(k[1] == p for k in cells)]
    phases += sorted({k[1] for k in cells} - set(phases))
    ds = Dataset()
    for k in keys:
        for p in phases:
            b = cells.get((k, p))
            if b and b.count:
                ds.emit({"sponsor": labels.label(k), "phase": PHASE_LABELS.get(p, p), "trial_count": b.count}, b.refs)
    meta = VizMeta(grouping={"by": ["sponsor", "phase"], "aggregate": "count_distinct_trials"})
    notes = _base_notes(ctx)
    if len(sponsor_totals) > len(keys):
        meta.truncation = Truncation(top_n=len(keys), omitted_count=len(sponsor_totals) - len(keys))
        notes.append(note(NoteCode.TRUNCATED, f"Showing the top {len(keys)} of {len(sponsor_totals)} sponsors."))
    viz = bar_chart(
        ds.rows,
        title=f"Top lead sponsors by phase: {_title_label(ctx)}",
        x=Channel(field="sponsor", type="nominal", title="Lead sponsor", sort=ordered([labels.label(k) for k in keys])),
        y=COUNT,
        series=Channel(
            field="phase", type="ordinal", title="Phase", sort=ordered([PHASE_LABELS.get(p, p) for p in phases])
        ),
        mode="stacked",
        orientation="horizontal",
        meta=meta,
    )
    facts: dict[str, Any] = {
        "trials_analyzed_derived": len(studies),
        "sponsors_total_derived": len(sponsor_totals),
        "sponsor_totals_derived": {labels.label(k): sponsor_totals[k] for k in keys},  # sum over phase rows
        "by_sponsor_phase": {
            labels.label(k): {PHASE_LABELS.get(p, p): cells[(k, p)].count for p in phases if (k, p) in cells}
            for k in keys
        },
    }
    if keys:
        facts["top_sponsor"] = labels.label(keys[0])
    return HandlerResult(
        visualization=viz, notes=notes, facts=facts, full_refs=ds.full_refs, studies_analyzed=len(studies)
    )


@register(Intent.INTERVENTION_LANDSCAPE)
@_traced("intervention_landscape")
async def intervention_landscape(ctx: HandlerContext) -> HandlerResult:
    """Most common intervention names (placebo excluded) or intervention types."""
    by_type = registry_counts.by_intervention_type(ctx.plan, ctx.request.query)
    if by_type and registry_counts.active(ctx):
        return await registry_counts.intervention_types(ctx, INTERVENTION_TYPE_LABELS)
    if by_type:
        return _categorical(
            ctx,
            extract=_ex_intervention_type,
            field_name="intervention_type",
            axis_title="Intervention type",
            title=f"Trials by intervention type: {_title_label(ctx)}",
            grouping={"multi_valued": True},
        )
    return _categorical(
        ctx,
        extract=_ex_intervention,
        field_name="intervention",
        axis_title="Intervention",
        title=f"Most common interventions: {_title_label(ctx)}",
        top_n=_top_n(ctx, 15),
        horizontal=True,
        allow_pie=False,
        grouping={"multi_valued": True, "excluded": ["placebo", "sham", "vehicle"]},
        extra_notes=[
            note(
                NoteCode.APPROXIMATED,
                "A trial testing several interventions counts once for each; "
                "placebo/sham comparators are excluded; names are grouped "
                "case- and punctuation-insensitively without fuzzy merging.",
            )
        ],
    )


# ================================================================================================
# heatmaps
# ================================================================================================


def _heatmap_result(
    ctx: HandlerContext,
    *,
    cells: dict[tuple[str, str], Bucket],
    xkeys: list[str],
    ykeys: list[str],
    xlab: Callable[[str], str],
    ylab: Callable[[str], str],
    xfield: str,
    yfield: str,
    xtitle: str,
    ytitle: str,
    title: str,
    notes: list[Note],
    grouping: dict[str, Any],
    facts: dict[str, Any],
    highlight_zero: bool = True,
) -> HandlerResult:
    ds = Dataset()
    gaps = 0
    best: tuple[int, str, str] | None = None
    for yk in ykeys:
        for xk in xkeys:
            b = cells.get((yk, xk))
            n = b.count if b else 0
            ds.emit({xfield: xlab(xk), yfield: ylab(yk), "trial_count": n, "is_gap": n == 0}, b.refs if b else [])
            gaps += n == 0
            if n and (best is None or n > best[0]):
                best = (n, ylab(yk), xlab(xk))
    viz = heatmap(
        ds.rows,
        title=title,
        x=Channel(field=xfield, type="ordinal", title=xtitle, sort=ordered([xlab(k) for k in xkeys])),
        y=Channel(field=yfield, type="ordinal", title=ytitle, sort=ordered([ylab(k) for k in ykeys])),
        value=COUNT,
        meta=VizMeta(grouping=grouping),
        highlight_zero=highlight_zero,
        tooltip=["is_gap"],
    )
    facts.update(
        cells_derived=len(ds.rows), gap_cells_derived=gaps, rows_derived=len(ykeys), columns_derived=len(xkeys)
    )
    if best:
        facts.update(max_cell_count=best[0], max_cell_row=best[1], max_cell_column=best[2])
    facts["gaps"] = [f"{r[yfield]} × {r[xfield]}" for r in (d.model_extra or {} for d in ds.rows) if r["is_gap"]][:25]
    return HandlerResult(
        visualization=viz, notes=notes, facts=facts, full_refs=ds.full_refs, studies_analyzed=len(ctx.studies)
    )


def _cohort_only_notes(ctx: HandlerContext) -> list[Note]:
    notes = _base_notes(ctx)
    if len(ctx.cohorts) > 1:
        notes.append(note(NoteCode.DEGRADED, "Cohorts are merged (de-duplicated) for this matrix view."))
    return notes


@register(Intent.PHASE_STATUS_HEATMAP)
@_traced("phase_status_heatmap")
async def phase_status_heatmap(ctx: HandlerContext) -> HandlerResult:
    """Phase (joined combinations) × overall status matrix.

    Empty cells are not styled as gaps here: many are structural (expanded-access statuses never carry a
    phase), so only the landscape gap heatmap asks renderers to emphasize them.
    """
    cells: dict[tuple[str, str], Bucket] = defaultdict(Bucket)
    for s in ctx.studies:
        phase, _, prefs = _ex_phase_joined(s)[0]
        status, _, srefs = _ex_status(s)[0]
        cells[(status, phase)].add(s.nct_id, prefs + srefs)
    phases = [p for p in PHASE_ORDER if any(k[1] == p for k in cells)]
    phases += sorted({k[1] for k in cells} - set(phases))
    order = STATUS_ORDER + [NOT_REPORTED]
    statuses = [st for st in order if any(k[0] == st for k in cells)]
    statuses += sorted({k[0] for k in cells} - set(statuses))
    facts: dict[str, Any] = {
        "trials_analyzed_derived": len(ctx.studies),
        "trials_by_phase_derived": {
            PHASE_LABELS.get(p, p): sum(b.count for k, b in cells.items() if k[1] == p) for p in phases
        },
        "trials_by_status_derived": {
            STATUS_LABELS.get(st, st): sum(b.count for k, b in cells.items() if k[0] == st) for st in statuses
        },
    }
    return _heatmap_result(
        ctx,
        cells=cells,
        xkeys=phases,
        ykeys=statuses,
        xlab=lambda p: PHASE_LABELS.get(p, p),
        ylab=lambda st: STATUS_LABELS.get(st, st),
        xfield="phase",
        yfield="status",
        xtitle="Phase",
        ytitle="Status",
        title=f"Phase × status: {_title_label(ctx)}",
        notes=_cohort_only_notes(ctx),
        grouping={
            "by": ["phase", "status"],
            "aggregate": "count_distinct_trials",
            "phase_combination": "joined",
            "each_trial_counted_once": True,
        },
        facts=facts,
        highlight_zero=False,
    )


@register(Intent.LANDSCAPE_GAP_HEATMAP)
@_traced("landscape_gap_heatmap")
async def landscape_gap_heatmap(ctx: HandlerContext) -> HandlerResult:
    """Entities (rows) × phase/status/year (columns), every cell present so empty cells show gaps."""
    row_dim = (
        ctx.plan.group_by
        if ctx.plan.group_by
        in ("condition", "intervention", "sponsor", "sponsor_class", "country", "intervention_type")
        else None
    )
    if row_dim is None:
        row_dim = "condition" if any(c.drugs for c in ctx.plan.cohorts) else "intervention"
    col_dim = ctx.plan.series_by if ctx.plan.series_by in ("phase", "status", "year") else "phase"
    row_ex, row_title = EXTRACTORS[row_dim]
    col_ex, col_title = EXTRACTORS[col_dim]
    labels, col_labels = LabelPicker(), LabelPicker()
    cells: dict[tuple[str, str], Bucket] = defaultdict(Bucket)
    row_trials: dict[str, set[str]] = defaultdict(set)
    no_column = 0
    for s in ctx.studies:
        cols = col_ex(s)
        no_column += not cols
        for rk, rraw, rrefs in row_ex(s):
            labels.add(rk, rraw)
            row_trials[rk].add(s.nct_id)
            for ck, craw, crefs in cols:
                col_labels.add(ck, craw)
                cells[(rk, ck)].add(s.nct_id, rrefs + crefs)
    n = _top_n(ctx, 15)
    ykeys = top_keys({k: len(v) for k, v in row_trials.items()}, n)
    present_cols = {k[1] for k in cells}
    absurd: set[int] = set()
    if col_dim == "phase":
        xkeys = SPLIT_PHASES + (["NA"] if "NA" in present_cols else [])
        col_labels.add("NA", PHASE_LABELS["NA"])
        for p in SPLIT_PHASES:
            col_labels.add(p, PHASE_LABELS[p])
    elif col_dim == "status":
        xkeys = [st for st in STATUS_ORDER if st in present_cols] + sorted(present_cols - set(STATUS_ORDER))
    else:
        # Year range from the rows actually shown; registry typos (< MIN_YEAR) are dropped with a note.
        shown_years = {int(ck) for rk, ck in cells if rk in set(ykeys)}
        absurd = {y for y in shown_years if y < MIN_YEAR}
        years = sorted(shown_years - absurd)
        xkeys = [str(y) for y in range(years[0], years[-1] + 1)] if years else []
    notes = _cohort_only_notes(ctx)
    if col_dim == "year" and no_column:
        notes.append(
            note(
                NoteCode.PARTIAL_DATES,
                f"{no_column} trial(s) have no start date and are not placed in any year column.",
            )
        )
    if col_dim == "year" and absurd:
        n_absurd = len({t for (rk, ck), b in cells.items() if rk in set(ykeys) and int(ck) in absurd for t in b.trials})
        notes.append(
            note(
                NoteCode.PARTIAL_DATES,
                f"{n_absurd} trial(s) report a start year before {MIN_YEAR} "
                f"({', '.join(map(str, sorted(absurd)))}); treated as registry "
                "errors and left out of the columns.",
            )
        )
    if len(row_trials) > len(ykeys):
        notes.append(
            note(
                NoteCode.TRUNCATED,
                f"Showing the {len(ykeys)} {row_title.lower()}s with the most trials (of {len(row_trials)}).",
            )
        )
    if col_dim == "phase":
        notes.append(
            note(
                NoteCode.APPROXIMATED,
                "Combined phases (e.g. Phase 1/2) count toward each phase they "
                "include, so a row's cells can sum to more than its trials.",
            )
        )
    facts: dict[str, Any] = {
        "trials_analyzed_derived": len(ctx.studies),
        "row_dimension": row_dim,
        "column_dimension": col_dim,
        "row_totals_derived": {labels.label(k): len(row_trials[k]) for k in ykeys},
    }
    return _heatmap_result(
        ctx,
        cells=cells,
        xkeys=xkeys,
        ykeys=ykeys,
        xlab=col_labels.label,
        ylab=labels.label,
        xfield=col_dim,
        yfield=row_dim,
        xtitle=col_title,
        ytitle=row_title,
        title=f"Landscape gaps: {row_title} × {col_title}",
        notes=notes,
        facts=facts,
        grouping={
            "by": [row_dim, col_dim],
            "aggregate": "count_distinct_trials",
            "phase_combination": "split" if col_dim == "phase" else None,
        },
    )


# ================================================================================================
# geography
# ================================================================================================


@register(Intent.GEOGRAPHIC_DISTRIBUTION)
@_traced("geographic_distribution")
async def geographic_distribution(ctx: HandlerContext) -> HandlerResult:
    """Trials per country (each trial once per country) with site counts.

    Countries are grouped by :func:`country_key` (whitespace/case-insensitive); rows show the most
    common spelling and citations keep the raw registry value. In comparison scope the shared country
    ranking (and ``top_country``) uses distinct trials across all cohorts, so a trial in two cohorts is
    not counted twice; rows and ``by_cohort`` stay per cohort.
    """
    if registry_counts.site_level(ctx.plan, ctx.request.query):
        return _site_map(ctx)
    if registry_counts.active(ctx):
        return await registry_counts.countries(ctx)
    comparison = _comparison(ctx)
    groups = _groups(ctx)
    labels = LabelPicker()
    tallies, missing = _tally(groups, _ex_country, labels)
    # Every counted site is evidence: (group, country key) -> refs to each locations[i].country.
    sites: dict[tuple[str, str], list[FieldRef | None]] = defaultdict(list)
    for glabel, studies in groups:
        for s in studies:
            for _, loc, path in s.items("locations"):
                key = country_key(loc.get("country") if isinstance(loc, dict) else None)
                if key:  # same rule as _ex_country
                    sites[(glabel, key)].append(s.ref(f"{path}.country"))
    distinct: dict[str, set[str]] = defaultdict(set)
    for buckets in tallies.values():
        for k, b in buckets.items():
            distinct[k] |= b.trials
    totals = {k: len(v) for k, v in distinct.items()}
    keys = top_keys(totals, len(totals))
    limit = ctx.plan.top_n or ctx.request.top_n
    if limit:
        keys = keys[:limit]
    ds = Dataset()
    per_group: dict[str, dict[str, int]] = {}
    for glabel, buckets in tallies.items():
        g: dict[str, int] = {}
        for k in keys:
            b = buckets.get(k)
            if b is None or not b.count:
                continue
            site_refs = sites[(glabel, k)]
            row = {"country": labels.label(k), "trial_count": b.count, "site_count": len(site_refs)}
            if comparison:
                row["cohort"] = glabel
            # One citation per trial (its first site in this country): the count is distinct trials, never
            # sites, so a trial with 20 U.S. sites adds 1 to the United States. site_count is shown separately.
            ds.emit(row, b.refs)
            g[row["country"]] = b.count
        per_group[glabel] = g
    notes = _base_notes(ctx)
    no_loc = len(missing)
    if no_loc:
        notes.append(note(NoteCode.NOT_IN_REGISTRY, f"{no_loc} trial(s) list no site locations and are not mapped."))
    if limit and len(totals) > len(keys):
        notes.append(note(NoteCode.TRUNCATED, f"Showing the top {len(keys)} of {len(totals)} countries."))
    facts: dict[str, Any] = {
        "trials_analyzed_derived": _analyzed(ctx),
        "countries_total_derived": len(totals),
        "trials_without_locations_derived": no_loc,
    }
    if comparison:
        facts["by_cohort"] = per_group
        facts["by_cohort_top"] = {
            g: {"country": max(v, key=lambda c: v[c]), "trials": max(v.values())} for g, v in per_group.items() if v
        }
        if keys:  # ranked by distinct trials across cohorts (not a single row value, hence _derived)
            facts.update(top_country=labels.label(keys[0]), top_country_trials_distinct_derived=totals[keys[0]])
    else:
        counts = next(iter(per_group.values()), {})
        facts["counts"] = counts
        facts["site_counts"] = {
            labels.label(k): len(sites[(groups[0][0], k)]) for k in keys if labels.label(k) in counts
        }
        if counts:
            top = max(counts, key=lambda c: counts[c])
            facts.update(top_country=top, top_country_trials=counts[top])
    viz = geo_map(
        ds.rows,
        title=f"Trials by country: {_title_label(ctx)}",
        region=Channel(field="country", type="nominal", title="Country"),
        value=Channel(field="trial_count", type="quantitative", title="Trials", unit="trials", format=",d"),
        series=COHORT if comparison else None,
        tooltip=["site_count"],
        meta=VizMeta(
            units={"site_count": "sites"},
            sort=Sort(by="trial_count", order="desc"),
            grouping={
                "by": ["country"],
                "aggregate": "count_distinct_trials",
                "per_trial": "once per country",
                "country_grouping": "whitespace- and case-insensitive",
            },
        ),
    )
    return HandlerResult(
        visualization=viz, notes=notes, facts=facts, full_refs=ds.full_refs, studies_analyzed=_analyzed(ctx)
    )


SITE_ROWS_CAP = 200


def _site_map(ctx: HandlerContext) -> HandlerResult:
    """One point per site (facility + city + country), sized by the number of trials it runs.

    When the question names countries, only sites in those countries are drawn (the cohort search returns
    every trial with at least one site there, and those trials may have sites elsewhere too).
    """
    wanted = {country_key(c) for cf in ctx.plan.cohorts for c in cf.countries}
    sites: dict[tuple[str, str, str], dict[str, Any]] = {}
    for s in ctx.studies:
        seen: set[tuple[str, str, str]] = set()
        for _, loc, path in s.items("locations"):
            if not isinstance(loc, dict) or not isinstance(loc.get("facility"), str):
                continue
            ckey = country_key(loc.get("country"))
            if wanted and ckey not in wanted:
                continue
            key = (normalize_name(loc["facility"]), normalize_name(loc.get("city")), ckey)
            if key in seen:
                continue
            seen.add(key)
            site = sites.setdefault(
                key,
                {
                    "site": " ".join(loc["facility"].split()),
                    "city": loc.get("city"),
                    "state": loc.get("state"),
                    "country": loc.get("country"),
                    "lat": None,
                    "lon": None,
                    "refs": [],
                    "trials": set(),
                },
            )
            geo = loc.get("geoPoint")
            if (
                site["lat"] is None
                and isinstance(geo, dict)
                and all(isinstance(geo.get(k), int | float) for k in ("lat", "lon"))
            ):
                site["lat"], site["lon"] = geo["lat"], geo["lon"]
            site["trials"].add(s.nct_id)
            site["refs"].append(s.ref(f"{path}.facility"))
    ranked = sorted(sites.values(), key=lambda x: (-len(x["trials"]), x["site"]))
    shown = ranked[:SITE_ROWS_CAP]
    ds = Dataset()
    for x in shown:
        row = {k: x[k] for k in ("site", "city", "state", "country", "lat", "lon") if x[k] is not None}
        row["trial_count"] = len(x["trials"])
        ds.emit(row, x["refs"])
    notes = _base_notes(ctx)
    if len(ranked) > len(shown):
        notes.append(note(NoteCode.TRUNCATED, f"Showing the {len(shown)} sites with the most trials of {len(ranked)}."))
    no_geo = sum(1 for x in shown if x["lat"] is None)
    if no_geo:
        notes.append(
            note(
                NoteCode.NOT_IN_REGISTRY,
                f"{no_geo} site(s) have no coordinates in the registry; they are listed but not drawn.",
            )
        )
    label = _title_label(ctx)
    where = ", ".join(sorted({cf_c for cf in ctx.plan.cohorts for cf_c in cf.countries}))
    facts: dict[str, Any] = {
        "trials_analyzed_derived": len(ctx.studies),
        "site_level": True,
        "sites_total_derived": len(ranked),
    }
    if shown:
        facts["counts"] = {
            f"{x['site']} ({x['city']})" if x["city"] else x["site"]: len(x["trials"]) for x in shown[:5]
        }
    meta = VizMeta(
        sort=Sort(by="trial_count", order="desc"),
        grouping={"by": ["site", "city", "country"], "aggregate": "count_distinct_trials"},
    )
    meta.geo = GeoSpec(level="site", lat_field="lat", lon_field="lon", value_field="trial_count")
    enc = Encoding(
        label=Channel(field="site", type="nominal", title="Site"),
        size=Channel(field="trial_count", type="quantitative", title="Trials at this site", unit="trials", format=",d"),
        tooltip=["city", "state", "country"],
    )
    viz = make_viz(
        VizType.GEO_MAP, f"Trial sites: {label}", enc, ds.rows, meta, f"Sites in {where} only" if where else None
    )
    return HandlerResult(
        visualization=viz, notes=notes, facts=facts, full_refs=ds.full_refs, studies_analyzed=len(ctx.studies)
    )


# ================================================================================================
# enrollment
# ================================================================================================


@register(Intent.ENROLLMENT_DISTRIBUTION)
@_traced("enrollment_distribution")
async def enrollment_distribution(ctx: HandlerContext) -> HandlerResult:
    """Histogram of enrollment (log-like bins), or scatter vs start date when preferred."""
    if ctx.plan.preferred_visualization is VizType.SCATTER_PLOT:
        return _enrollment_scatter(ctx)
    comparison = _comparison(ctx)
    bins: dict[tuple[str, int], Bucket] = defaultdict(Bucket)
    values: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"actual": [], "estimated": []})
    missing_ids: set[str] = set()
    estimated_ids: set[str] = set()
    for glabel, studies in _groups(ctx):
        for s in studies:
            n = s.get("enrollment_count")
            if not isinstance(n, int) or isinstance(n, bool):
                missing_ids.add(s.nct_id)
                continue
            is_est = s.get("enrollment_type") == "ESTIMATED"
            if is_est:
                estimated_ids.add(s.nct_id)
            values[glabel]["estimated" if is_est else "actual"].append(n)
            bins[(glabel, enrollment_bin(n))].add(s.nct_id, [s.ref("enrollment_count")])
    ds = Dataset()
    missing, estimated = len(missing_ids), len(estimated_ids)
    n_with = len({nct for (_, _), b in bins.items() for nct in b.trials})  # distinct across cohorts
    facts: dict[str, Any] = {
        "trials_analyzed_derived": _analyzed(ctx),
        "trials_with_enrollment_derived": n_with,
        "trials_without_enrollment_derived": missing,
        "trials_with_estimated_enrollment_derived": estimated,
    }
    if bins:
        last = max(i for _, i in bins)
        per_group: dict[str, dict[str, int]] = {}
        for glabel, _ in _groups(ctx):
            g: dict[str, int] = {}
            for i in range(last + 1):
                lo, hi = ENROLLMENT_BINS[i]
                b = bins.get((glabel, i))
                # Open-ended last bin: bin_end is None (omitted on the wire) and is_open_ended is true.
                row: dict[str, Any] = {
                    "bin_start": lo,
                    "bin_end": hi,
                    "bin_label": enrollment_bin_label(i),
                    "trial_count": b.count if b else 0,
                    "is_open_ended": hi is None,
                }
                if comparison:
                    row["cohort"] = glabel
                ds.emit(row, b.refs if b else [])
                g[row["bin_label"]] = row["trial_count"]
            per_group[glabel] = g
        stats = {g: _enroll_stats(v) for g, v in values.items()}
        facts["stats_derived"] = stats if comparison else next(iter(stats.values()))
        facts["by_cohort" if comparison else "counts"] = per_group if comparison else next(iter(per_group.values()))
    notes = _base_notes(ctx)
    if missing:
        notes.append(note(NoteCode.NOT_IN_REGISTRY, f"{missing} trial(s) report no enrollment count."))
    if estimated:
        notes.append(
            note(
                NoteCode.APPROXIMATED,
                f"{estimated} trial(s) report an estimated (planned) enrollment rather than actual participants.",
            )
        )
    summary = facts.get("stats_derived")
    # Linear scale: bins start at 0 (log is undefined there); bins are unequal-width, so renderers should
    # draw [bin_start, bin_end) or fall back to the ordinal bin_label axis.
    viz = histogram(
        ds.rows,
        title=f"Enrollment per trial: {_title_label(ctx)}",
        x=Channel(
            field="bin_start",
            type="quantitative",
            title="Participants",
            unit="participants",
            format=",d",
            scale=Scale(type="linear", zero=True),
        ),
        x2=Channel(field="bin_end", type="quantitative", title="Bin end (exclusive)", unit="participants", format=",d"),
        y=COUNT,
        series=COHORT if comparison else None,
        label=Channel(field="bin_label", type="ordinal", title="Participants"),
        summary=summary,
        meta=VizMeta(
            grouping={"bins": "fixed, log-like widths", "bin_interval": "[start, end)", "open_ended_last_bin": True}
        ),
    )
    return HandlerResult(
        visualization=viz, notes=notes, facts=facts, full_refs=ds.full_refs, studies_analyzed=_analyzed(ctx)
    )


def _describe(vals: list[int]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "median": median(vals), "min": min(vals), "max": max(vals)}


def _enroll_stats(kinds: dict[str, list[int]]) -> dict[str, Any]:
    """Actual (enrolled) and estimated (planned) enrollment described separately; the participant total
    is only summed over ACTUAL counts, since planned targets are not people."""
    return {
        "actual": {**_describe(kinds["actual"]), "total_participants_derived": sum(kinds["actual"])},
        "estimated": _describe(kinds["estimated"]),
    }


def _enrollment_scatter(ctx: HandlerContext) -> HandlerResult:
    comparison = _comparison(ctx)
    ds = Dataset()
    skipped_ids: set[str] = set()
    zero_ids: set[str] = set()
    partial_ids: set[str] = set()
    kinds: dict[str, list[int]] = {"actual": [], "estimated": []}
    for glabel, studies in _groups(ctx):
        for s in studies:
            raw_start = s.get("start_date")
            n, pd = s.get("enrollment_count"), parse_partial_date(raw_start)
            if not isinstance(n, int) or isinstance(n, bool) or pd is None:
                skipped_ids.add(s.nct_id)
                continue
            if n <= 0:  # log y axis cannot show 0
                zero_ids.add(s.nct_id)
                continue
            if pd.precision != "day":
                partial_ids.add(s.nct_id)
            _, plabel, prefs = _ex_phase_joined(s)[0]
            row: dict[str, Any] = {
                "nct_id": s.nct_id,
                "start": pd.iso,
                "start_raw": raw_start,
                "start_precision": pd.precision,
                "enrollment": n,
                "phase": plabel,
            }
            if comparison:
                row["cohort"] = glabel
            ds.emit(row, [s.ref("enrollment_count"), s.ref("start_date"), *prefs])
            kinds["estimated" if s.get("enrollment_type") == "ESTIMATED" else "actual"].append(n)
    notes = _base_notes(ctx)
    skipped, zero, partial = len(skipped_ids), len(zero_ids), len(partial_ids)
    if skipped:
        notes.append(note(NoteCode.NOT_IN_REGISTRY, f"{skipped} trial(s) lack a start date or enrollment count."))
    if zero:
        notes.append(
            note(NoteCode.TRUNCATED, f"{zero} trial(s) report an enrollment of 0 and are not plotted on the log scale.")
        )
    if partial:
        notes.append(
            note(
                NoteCode.PARTIAL_DATES,
                f"{partial} trial(s) give only a start month or year; they are "
                "placed on the first day of that period (see start_raw).",
            )
        )
    facts: dict[str, Any] = {
        "trials_plotted_derived": len(ds.rows),
        "trials_skipped_derived": skipped,
        "trials_zero_enrollment_derived": zero,
        "stats_derived": _enroll_stats(kinds),
    }
    viz = scatter_plot(
        ds.rows,
        title=f"Enrollment vs start date: {_title_label(ctx)}",
        x=Channel(field="start", type="temporal", title="Start date", time_format="%Y"),
        y=Channel(
            field="enrollment",
            type="quantitative",
            title="Participants",
            unit="participants",
            format=",d",
            scale=Scale(type="log"),
        ),
        series=COHORT if comparison else Channel(field="phase", type="nominal", title="Phase"),
        label=Channel(field="nct_id", type="nominal", title="Trial"),
    )
    return HandlerResult(
        visualization=viz, notes=notes, facts=facts, full_refs=ds.full_refs, studies_analyzed=_analyzed(ctx)
    )


# ================================================================================================
# entity_network
# ================================================================================================

NETWORKS: dict[str, dict[str, Any]] = {
    "sponsor_drug": {
        "title": "Sponsors and the drugs they test",
        "types": [("sponsor", "Lead sponsor"), ("drug", "Drug / biological")],
        "edge": "sponsor_tests_drug",
        "directed": True,
        "semantics": "Lead sponsor runs at least one trial testing the drug; weight = distinct trials.",
    },
    "drug_drug": {
        "title": "Drugs given together in the same arm",
        "types": [("drug", "Drug"), ("biological", "Biological")],
        "edge": "combined_in_arm",
        "directed": False,
        "semantics": "Both drugs are given together in the same arm (armGroups[].interventionNames, "
        ">= 2 drug/biological, placebo excluded); weight = distinct trials.",
    },
    "drug_condition": {
        "title": "Drugs and the conditions they are tested for",
        "types": [("drug", "Drug / biological"), ("condition", "Condition")],
        "edge": "drug_studied_for",
        "directed": True,
        "semantics": "Drug is tested in a trial listing the condition; weight = distinct trials.",
    },
    "sponsor_collaborator": {
        "title": "Lead sponsors and their collaborators",
        "types": [("lead_sponsor", "Lead sponsor"), ("collaborator", "Collaborator only")],
        "edge": "collaborates_with",
        "directed": True,
        "semantics": "Lead sponsor lists the organisation as collaborator; weight = distinct trials.",
    },
    "site_investigator": {
        "title": "Investigators and their trial sites",
        "types": [("investigator", "Overall official"), ("site", "Site")],
        "edge": "official_of_trial_at_site",
        "directed": True,
        "semantics": "Overall official of a trial that lists the site; weight = distinct trials.",
    },
}


def _node_key(kind: str, name: str) -> str:
    if kind == "drug":
        name = canonical_drug(name) or name
    return f"{kind}:{normalize_name(name).replace(' ', '-')}"


_DOSE_NOISE = re.compile(
    r"\b(\d+(\.\d+)?\s*(mg|mcg|µg|g|ml|mg/kg|mg/m2|iu|units?)|auc\s*\d+(\.\d+)?|injections?|infusions?|intravenous|i\.?v|oral|"
    r"tablets?|capsules?|subcutaneous|s\.?c|solution|doses?|monotherapy|alone|therapy|treatment|q\d+w|weekly|"
    r"daily|arm|group|cohort|part [a-z0-9])\b",
    re.IGNORECASE,
)


@functools.lru_cache(maxsize=20000)
def canonical_drug(name: str) -> str | None:
    """The vocabulary drug a registry intervention name denotes, when it is that one drug plus noise.

    "Pembrolizumab (MK-3475)", "pembrolizumab [KEYTRUDA]", "Pembrolizumab IV 200 mg" -> "pembrolizumab", so
    network nodes and intervention bars do not split one drug into variants. Names that combine drugs or that
    the vocabulary does not know stay as written (no fuzzy matching: a wrong merge is worse than a split).
    """
    resolver = get_resolver()
    bare = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", name)  # bracketed code / brand names
    for candidate in (name, bare, _DOSE_NOISE.sub(" ", bare)):
        text = " ".join(candidate.split()).strip(" ,;:-")
        if text:
            r = resolver.resolve(text, "drug")
            if r.method in ("exact", "alias"):
                return r.resolved
    mentions = [m for m in resolver.scan(name) if m.kind == "drug" and not m.is_class]
    if len({m.entity.resolved for m in mentions}) != 1:
        return None
    rest = name
    for m in mentions:
        rest = rest.replace(m.surface, " ")
    rest = _DOSE_NOISE.sub(" ", re.sub(r"[\[\]()+,/;:\-]", " ", rest))
    return None if re.search(r"[A-Za-z]{3,}", rest) else mentions[0].entity.resolved


def drug_label(name: str) -> str:
    c = canonical_drug(name)
    return c[:1].upper() + c[1:] if c else name


SITES_PER_TRIAL_CAP = 50
MIN_REPEATED_LINKS = 10  # links with >= 2 shared trials needed before single-trial links are dropped
MAX_EDGES_SHOWN = 120  # beyond this a force layout is a hairball

# Lazy evidence handle: (study, field path). Resolved to a FieldRef only for surviving nodes/edges.
Ev = tuple[Study, str]


def _resolve(evidence: list[Any]) -> list[FieldRef | None]:
    return [s.ref(path) for s, path in evidence]


@dataclass
class NetworkBuild:
    graph: CooccurrenceGraph
    trials_with_capped_sites: int = 0


def _build_network(kind: str, studies: list[Study]) -> NetworkBuild:
    g = CooccurrenceGraph(directed=NETWORKS[kind]["directed"])
    out = NetworkBuild(graph=g)
    etype = NETWORKS[kind]["edge"]
    for s in studies:
        nct = s.nct_id
        if kind == "drug_drug":
            for _, arm, apath in s.items("arm_groups"):
                names = arm.get("interventionNames") if isinstance(arm, dict) else None
                drugs: dict[str, Ev] = {}
                for j, entry in enumerate(names or []):
                    parsed = split_arm_intervention(entry)
                    if not parsed or parsed[0].upper() not in DRUG_TYPES or is_placebo(parsed[1]):
                        continue
                    key = _node_key("drug", parsed[1])
                    if key in drugs:
                        continue
                    ev = (s, f"{apath}.interventionNames[{j}]")
                    drugs[key] = ev
                    g.add_node(key, drug_label(parsed[1]), parsed[0].lower(), nct, ev)
                keys = list(drugs)
                for a in range(len(keys)):
                    for b in range(a + 1, len(keys)):
                        g.add_edge(keys[a], keys[b], etype, nct, [drugs[keys[a]], drugs[keys[b]]])
        elif kind in ("sponsor_drug", "drug_condition"):
            drugs_items = [(key, name, refs[0].field_path) for key, name, refs in _drug_items(s, DRUG_TYPES) if refs[0]]
            others: list[tuple[str, str, Ev]] = []
            if kind == "sponsor_drug":
                name = s.get("lead_sponsor")
                if isinstance(name, str) and normalize_name(name):
                    others = [(_node_key("sponsor", name), name, (s, P["lead_sponsor"]))]
                otype = "sponsor"
            else:
                others = [
                    (_node_key("condition", raw), raw, (s, refs[0].field_path))
                    for _, raw, refs in _ex_condition(s)
                    if refs[0]
                ]
                otype = "condition"
            for okey, oname, oev in others:
                g.add_node(okey, oname, otype, nct, oev)
            for key, dname, dpath in drugs_items:
                dkey, dev = f"drug:{key.replace(' ', '-')}", (s, dpath)  # key already merges name variants
                g.add_node(dkey, dname, "drug", nct, dev)
                for okey, _, oev in others:
                    src, dst = (okey, dkey) if kind == "sponsor_drug" else (dkey, okey)
                    g.add_edge(src, dst, etype, nct, [oev, dev])
        elif kind == "sponsor_collaborator":
            lead = s.get("lead_sponsor")
            if not (isinstance(lead, str) and normalize_name(lead)):
                continue
            lkey, lev = _node_key("org", lead), (s, P["lead_sponsor"])
            collabs = []
            for _, c, cpath in s.items("collaborators"):
                cname = c.get("name") if isinstance(c, dict) else None
                if isinstance(cname, str) and normalize_name(cname):
                    collabs.append((_node_key("org", cname), cname, (s, f"{cpath}.name")))
            if not collabs:
                continue
            g.add_node(lkey, lead, "lead_sponsor", nct, lev)
            for ckey, cname, cev in collabs:
                g.add_node(ckey, cname, "collaborator", nct, cev)
                g.add_edge(lkey, ckey, etype, nct, [lev, cev])
        elif kind == "site_investigator":
            officials = []
            for _, o, opath in s.items("overall_officials"):
                oname = o.get("name") if isinstance(o, dict) else None
                if isinstance(oname, str) and normalize_name(oname):
                    officials.append((_node_key("investigator", oname), oname, (s, f"{opath}.name")))
            if not officials:
                continue
            sites: list[tuple[str, str, list[Ev]]] = []
            for _, loc, lpath in s.items("locations"):
                fac = loc.get("facility") if isinstance(loc, dict) else None
                if not (isinstance(fac, str) and normalize_name(fac)):
                    continue
                if len(sites) == SITES_PER_TRIAL_CAP:
                    out.trials_with_capped_sites += 1
                    break
                city = loc.get("city") if isinstance(loc.get("city"), str) else ""
                evs = [(s, f"{lpath}.facility")] + ([(s, f"{lpath}.city")] if city else [])
                sites.append((_node_key("site", f"{fac} {city}"), f"{fac}, {city}" if city else fac, evs))
            for okey, oname, oev in officials:
                g.add_node(okey, oname, "investigator", nct, oev)
            for skey, label, sevs in sites:
                g.add_node(skey, label, "site", nct, sevs[0])
                for okey, _, oev in officials:
                    g.add_edge(okey, skey, etype, nct, [oev, *sevs])
    if kind == "sponsor_collaborator":  # an organisation that leads anywhere is a lead sponsor
        for s in studies:
            lead = s.get("lead_sponsor")
            if isinstance(lead, str) and _node_key("org", lead) in g.nodes:
                g.nodes[_node_key("org", lead)].type = "lead_sponsor"
    return out


def _pinned_keys(ctx: HandlerContext, kind: str) -> set[str]:
    """Node keys for the entities named in the plan (drugs / sponsors / conditions of each cohort)."""
    pins: set[str] = set()
    for c in ctx.plan.cohorts:
        if kind in ("sponsor_drug", "drug_drug", "drug_condition"):
            pins |= {_node_key("drug", d) for d in c.drugs}
        if kind == "sponsor_drug":
            pins |= {_node_key("sponsor", sp) for sp in c.sponsors}
        if kind == "sponsor_collaborator":
            pins |= {_node_key("org", sp) for sp in c.sponsors}
        if kind == "drug_condition":
            pins |= {_node_key("condition", cond) for cond in c.conditions}
    return pins


@register(Intent.ENTITY_NETWORK)
@_traced("entity_network")
async def entity_network(ctx: HandlerContext) -> HandlerResult:
    """Co-occurrence network of the requested kind; pruned to the most connected entities."""
    kind = ctx.plan.network_kind or "sponsor_drug"
    spec = NETWORKS[kind]
    studies = ctx.studies
    built = _build_network(kind, studies)
    g = built.graph
    pins = {k for k in _pinned_keys(ctx, kind) if k in g.nodes}
    max_nodes = min(_top_n(ctx, 30), 40)
    # "Which drugs are combined WITH X": one named drug in a combination network -> X's ego network.
    ego = kind == "drug_drug" and len(pins) == 1
    # A link seen in a single trial is noise once the cohort has real repeated pairs: require >= 2 then.
    repeated = sum(1 for e in g.edges.values() if e.weight >= 2)
    floor = 2 if repeated >= MIN_REPEATED_LINKS else 1
    pruned = g.prune(
        max_nodes=max_nodes if not ego else min(max_nodes, 21),
        max_edges=MAX_EDGES_SHOWN,
        min_edge_weight=floor,
        pinned=pins,
        ego=ego,
    )
    gds = GraphDataset()
    for n in pruned.nodes:
        values = {"label": n.label, "type": n.type, "size": len(n.trials), "degree": n.degree}
        if n.key in pins:
            values["pinned"] = True
        gds.add_node(n.key, values, _resolve(n.evidence))
    for e in pruned.edges:
        gds.add_edge({"source": e.source, "target": e.target, "weight": e.weight, "type": e.type}, _resolve(e.evidence))
    type_counts = Counter(n.type for n in pruned.nodes)
    node_types = [{"type": t, "label": lbl, "count": type_counts.get(t, 0)} for t, lbl in spec["types"]]
    meta = VizMeta(
        grouping={"weight": "distinct_trials"},
        network_kind=kind,
        directed=spec["directed"],
        min_edge_weight=pruned.min_edge_weight,
        pinned=sorted(pins),
    )
    notes = _base_notes(ctx)
    if pruned.total_nodes > len(pruned.nodes) or pruned.total_edges > len(pruned.edges):
        meta.truncation = Truncation(top_n=len(pruned.nodes), omitted_count=pruned.total_nodes - len(pruned.nodes))
        notes.append(
            note(
                NoteCode.TRUNCATED,
                f"Showing {len(pruned.nodes)} of {pruned.total_nodes} entities (most connected"
                + (", plus the ones you asked about" if pins else "")
                + f") and "
                f"{len(pruned.edges)} of {pruned.total_edges} links"
                + (f"; other links need weight >= {pruned.min_edge_weight}" if pruned.min_edge_weight > 1 else "")
                + ".",
            )
        )
    if pruned.tie_truncated:
        notes.append(
            note(
                NoteCode.TRUNCATED,
                f"More links share weight {pruned.min_edge_weight} than fit; kept "
                f"{len(pruned.edges)} links, preferring better-connected entities "
                "(ties broken by name).",
            )
        )
    if built.trials_with_capped_sites:
        notes.append(
            note(
                NoteCode.TRUNCATED,
                f"{built.trials_with_capped_sites} trial(s) list more than "
                f"{SITES_PER_TRIAL_CAP} sites; only their first "
                f"{SITES_PER_TRIAL_CAP} sites are linked.",
            )
        )
    if kind == "drug_drug" and not pruned.edges:
        notes.append(note(NoteCode.ZERO_RESULTS, "No arms combine two or more drugs/biologicals in these trials."))
    labels = {n.key: n.label for n in pruned.nodes}
    facts: dict[str, Any] = {
        "network_kind": kind,
        "trials_analyzed_derived": len(studies),
        "nodes_derived": len(pruned.nodes),
        "edges_derived": len(pruned.edges),
        "nodes_total_derived": pruned.total_nodes,
        "edges_total_derived": pruned.total_edges,
        "min_edge_weight_derived": pruned.min_edge_weight,
        "node_type_counts_derived": dict(type_counts),
        "pinned": [labels[k] for k in pins if k in labels],
        "top_nodes": [
            {"label": n.label, "type": n.type, "degree": n.degree, "size": len(n.trials)}
            for n in sorted(pruned.nodes, key=lambda n: (-len(n.trials), n.label))[:5]
        ],
        "top_edges": [
            {"source": labels[e.source], "target": labels[e.target], "weight": e.weight} for e in pruned.edges[:5]
        ],
        "type_labels": {t: lbl for t, lbl in spec["types"]},
        "cohort_label": _title_label(ctx),
    }
    for k in pins:  # the named entity's own links, for the plain-language answer
        node = g.nodes[k]
        facts.setdefault("pinned_partners", []).append(
            {
                "label": node.label,
                "trials": len(node.trials),
                "partners_total_derived": len(partners(g, k, 10**6)),
                "partners": [{"label": lab, "trials": w} for lab, w in partners(g, k, 5)],
            }
        )
    if ego:
        meta.network_layout = "ego"  # renderers place the pinned node at the center
    viz = network_graph(
        gds.to_graph(),
        title=f"{spec['title']}: {_title_label(ctx)}",
        node_types=node_types,
        edge_semantics=spec["semantics"],
        meta=meta,
    )
    return HandlerResult(
        visualization=viz, notes=notes, facts=facts, full_refs=gds.full_refs, studies_analyzed=len(studies)
    )


# ================================================================================================
# timelines
# ================================================================================================

SPONSOR_ACTIVITY_CAP = 40


def _end_date(s: Study) -> tuple[PartialDate | None, str | None, str | None]:
    """(date, date field key, type field key) using completion date, else primary completion."""
    for key, tkey in (("completion_date", "completion_type"), ("primary_completion_date", "primary_completion_type")):
        pd = parse_partial_date(s.get(key))
        if pd:
            return pd, key, tkey
    return None, None, None


@register(Intent.SPONSOR_ACTIVITY)
@_traced("sponsor_activity")
async def sponsor_activity(ctx: HandlerContext) -> HandlerResult:
    """Gantt of the most recent trials (start → completion), laned by phase, colored by status."""
    dated: list[tuple[PartialDate, Study]] = []
    no_start = 0
    for s in ctx.studies:
        pd = parse_partial_date(s.get("start_date"))
        if pd is None:
            no_start += 1
        else:
            dated.append((pd, s))
    dated.sort(key=lambda t: (t[0].value, t[1].nct_id), reverse=True)
    shown = dated[:SPONSOR_ACTIVITY_CAP]
    lane_order = {p: i for i, p in enumerate(PHASE_ORDER)}
    shown.sort(key=lambda t: (lane_order.get(phase_bucket(t[1])[0], 99), t[0].value, t[1].nct_id))
    ds = Dataset()
    partial = estimated = 0
    status_counts: Counter[str] = Counter()
    lane_counts: Counter[str] = Counter()
    for start, s in shown:
        end, end_key, end_type = _end_date(s)
        phase, phase_ref = phase_bucket(s)
        status = s.get("status") if isinstance(s.get("status"), str) else NOT_REPORTED
        is_est = s.get("start_date_type") == "ESTIMATED" or bool(end_type and s.get(end_type) == "ESTIMATED")
        partial += start.precision != "day" or bool(end and end.precision != "day")
        estimated += is_est
        acronym = s.get("acronym")
        lane = PHASE_LABELS.get(phase, phase)
        row = {
            "task": f"{acronym} · {s.nct_id}" if isinstance(acronym, str) and acronym else s.nct_id,
            "nct_id": s.nct_id,
            "title": s.get("brief_title"),
            "start": start.iso,
            "start_raw": s.get("start_date"),
            "start_precision": start.precision,
            "end": end.iso if end else None,
            "end_raw": s.get(end_key) if end_key else None,
            "end_precision": end.precision if end else None,
            "lane": lane,
            "category": STATUS_LABELS.get(status, status),
            "status": status,
            "is_estimated": is_est,
            "point": end is None,
        }
        refs = [
            s.ref("start_date"),
            s.ref(end_key) if end_key else None,
            phase_ref or s.ref("nct_id"),
            s.ref("status"),
            s.ref("brief_title"),
            s.ref("acronym"),
        ]
        if is_est:
            refs += [s.ref("start_date_type"), s.ref(end_type) if end_type else None]
        ds.emit(row, refs)
        status_counts[row["category"]] += 1
        lane_counts[lane] += 1
    notes = _base_notes(ctx)
    if len(dated) > len(shown):
        notes.append(
            note(NoteCode.TRUNCATED, f"Showing the {len(shown)} most recently started of {len(dated)} trials.")
        )
    if no_start:
        notes.append(note(NoteCode.PARTIAL_DATES, f"{no_start} trial(s) have no start date and are not shown."))
    if partial:
        notes.append(
            note(
                NoteCode.PARTIAL_DATES,
                f"{partial} trial(s) give month- or year-only dates; they are drawn from the first day of that period.",
            )
        )
    if estimated:
        notes.append(
            note(NoteCode.APPROXIMATED, f"{estimated} trial(s) use estimated (planned) dates (is_estimated: true).")
        )
    lane_order = [PHASE_LABELS.get(p, p) for p in PHASE_ORDER if PHASE_LABELS.get(p, p) in lane_counts]
    meta = VizMeta(
        time_axis=TimeAxis(kind="calendar"), grouping={"lane": "phase", "lane_order": lane_order, "color": "status"}
    )
    facts: dict[str, Any] = {
        "trials_total_derived": len(ctx.studies),
        "trials_shown_derived": len(shown),
        "trials_without_start_date_derived": no_start,
        "trials_by_status_derived": dict(status_counts),
        "trials_by_phase_derived": dict(lane_counts),
        "trials_with_estimated_dates_derived": estimated,
        "trials_with_partial_dates_derived": partial,
    }
    if shown:
        first = min(shown, key=lambda t: t[0].value)
        last = max(shown, key=lambda t: t[0].value)
        facts.update(
            earliest_start_shown=first[0].iso,
            earliest_start_shown_raw=first[1].get("start_date"),
            earliest_start_shown_precision=first[0].precision,
            latest_start_shown=last[0].iso,
            latest_start_shown_raw=last[1].get("start_date"),
            latest_start_shown_precision=last[0].precision,
        )
    viz = timeline(
        ds.rows,
        title=f"Trial activity: {_title_label(ctx)}",
        category=Channel(field="category", type="nominal", title="Status"),
        tooltip=["title", "lane", "status", "is_estimated", "point"],
        meta=meta,
    )
    return HandlerResult(
        visualization=viz, notes=notes, facts=facts, full_refs=ds.full_refs, studies_analyzed=len(ctx.studies)
    )


@register(Intent.DEVELOPMENT_PROGRESSION)
@_traced("development_progression")
async def development_progression(ctx: HandlerContext) -> HandlerResult:
    """Per phase: first and latest registered trial start (registered human studies only)."""
    comparison = _comparison(ctx)
    ds = Dataset()
    no_start_ids: set[str] = set()
    facts: dict[str, Any] = {"trials_analyzed_derived": _analyzed(ctx)}
    phase_facts: dict[str, dict[str, Any]] = {}
    partial_ncts: set[str] = set()
    for glabel, studies in _groups(ctx):
        per_phase: dict[str, list[tuple[PartialDate, Study]]] = defaultdict(list)
        all_dated: list[tuple[PartialDate, Study]] = []
        for s in studies:
            pd = parse_partial_date(s.get("start_date"))
            if pd is None:
                no_start_ids.add(s.nct_id)
                continue
            all_dated.append((pd, s))
            for p in dict.fromkeys(s.get("phases") or []):
                if p in SPLIT_PHASES:
                    per_phase[p].append((pd, s))
        if all_dated:
            first_d, first_s = min(all_dated, key=lambda t: (t[0].value, t[1].nct_id))
            if first_d.precision != "day":
                partial_ncts.add(first_s.nct_id)
            ds.emit(
                {
                    "task": f"{glabel}: first registered trial" if comparison else "First registered trial",
                    "start": first_d.iso,
                    "start_raw": first_s.get("start_date"),
                    "start_precision": first_d.precision,
                    "end": None,
                    "end_raw": None,
                    "end_precision": None,
                    "lane": glabel,
                    "category": "milestone",
                    "status": first_s.get("status"),
                    "is_estimated": first_s.get("start_date_type") == "ESTIMATED",
                    "point": True,
                    "trial_count": 1,
                    "first_nct": first_s.nct_id,
                    "last_nct": first_s.nct_id,
                },
                [first_s.ref("start_date"), first_s.ref("status")],
            )
        gfacts: dict[str, Any] = {}
        for p in SPLIT_PHASES:
            items = sorted(per_phase.get(p, []), key=lambda t: (t[0].value, t[1].nct_id))
            if not items:
                continue
            (fd, fs), (ld, ls) = items[0], items[-1]
            single = fs.nct_id == ls.nct_id or fd.value == ld.value
            label = PHASE_LABELS[p]
            refs = [fs.ref("start_date"), fs.ref("phases"), ls.ref("start_date"), ls.ref("phases"), ls.ref("status")]
            refs += [r for _, s in items[1:-1] for r in (s.ref("start_date"), s.ref("phases"))]
            is_est = any(x.get("start_date_type") == "ESTIMATED" for x in (fs, ls))
            partial_ncts.update(x.nct_id for d, x in ((fd, fs), (ld, ls)) if d.precision != "day")
            ds.emit(
                {
                    "task": f"{glabel}: {label}" if comparison else label,
                    "start": fd.iso,
                    "start_raw": fs.get("start_date"),
                    "start_precision": fd.precision,
                    "end": None if single else ld.iso,
                    "end_raw": None if single else ls.get("start_date"),
                    "end_precision": None if single else ld.precision,
                    "lane": glabel,
                    "category": label,
                    "status": ls.get("status"),
                    "is_estimated": is_est,
                    "point": single,
                    "trial_count": len(items),
                    "first_nct": fs.nct_id,
                    "last_nct": ls.nct_id,
                },
                refs,
                keep_order=True,
            )  # first and latest trial cited first
            gfacts[label] = {
                "first_start": fd.iso,
                "first_start_raw": fs.get("start_date"),
                "first_start_precision": fd.precision,
                "latest_start": ld.iso,
                "latest_start_raw": ls.get("start_date"),
                "latest_start_precision": ld.precision,
                "trial_count": len(items),
                "first_nct": fs.nct_id,
            }
        phase_facts[glabel] = gfacts
    facts["by_cohort" if comparison else "phases"] = phase_facts if comparison else next(iter(phase_facts.values()), {})
    no_start = len(no_start_ids)
    facts["trials_without_start_date_derived"] = no_start
    notes = _base_notes(ctx)
    notes.append(
        note(
            NoteCode.NOT_IN_REGISTRY,
            "Preclinical development is not recorded in ClinicalTrials.gov; "
            "the timeline starts at the first registered human study.",
        )
    )
    notes.append(
        note(
            NoteCode.APPROXIMATED,
            "Combined phases (e.g. Phase 1/2) count toward each phase; bars span "
            "the first to the latest trial start in that phase.",
        )
    )
    if no_start:
        notes.append(note(NoteCode.PARTIAL_DATES, f"{no_start} trial(s) have no start date and are excluded."))
    if partial_ncts:
        notes.append(
            note(
                NoteCode.PARTIAL_DATES,
                f"{len(partial_ncts)} plotted date(s) come from month- or "
                "year-only registry dates; they are drawn from the first day of "
                "that period (see start_raw / start_precision).",
            )
        )
    viz = timeline(
        ds.rows,
        title=f"Development progression: {_title_label(ctx)}",
        category=Channel(
            field="category",
            type="nominal",
            title="Phase",
            sort=ordered(["milestone"] + [PHASE_LABELS[p] for p in SPLIT_PHASES]),
        ),
        tooltip=["trial_count", "first_nct", "last_nct", "status", "is_estimated"],
        meta=VizMeta(
            time_axis=TimeAxis(kind="calendar"), grouping={"by": ["phase"], "span": "first to latest trial start"}
        ),
    )
    return HandlerResult(
        visualization=viz, notes=notes, facts=facts, full_refs=ds.full_refs, studies_analyzed=_analyzed(ctx)
    )


# ================================================================================================
# design_methodology (cohort design mix; single-trial design cards)
# ================================================================================================

DESIGN_DIMS: dict[str, tuple[str, str]] = {  # dimension -> (field key, title)
    "allocation": ("allocation", "Allocation"),
    "masking": ("masking", "Masking"),
    "intervention_model": ("intervention_model", "Intervention model"),
    "study_type": ("study_type", "Study type"),
    "primary_purpose": ("primary_purpose", "Primary purpose"),
}
DESIGN_CARDS = [
    ("study_type", "Study type"),
    ("phases", "Phase"),
    ("allocation", "Allocation"),
    ("intervention_model", "Intervention model"),
    ("primary_purpose", "Primary purpose"),
    ("masking", "Masking"),
    ("who_masked", "Who is masked"),
    ("enrollment_count", "Enrollment"),
    ("enrollment_type", "Enrollment type"),
]


@register(Intent.DESIGN_METHODOLOGY)
@_traced("design_methodology")
async def design_methodology(ctx: HandlerContext) -> HandlerResult:
    """Single trial(s): cited design cards. Cohort: design mix (see :func:`design_mix`)."""
    if ctx.trials and not ctx.cohorts:
        return _design_cards(ctx)
    return design_mix(ctx)


def _design_cards(ctx: HandlerContext) -> HandlerResult:
    ds = Dataset()
    multi = len(ctx.trials) > 1
    facts: dict[str, Any] = {}
    for s in ctx.trials:
        tf: dict[str, Any] = {}
        for key, label in DESIGN_CARDS:
            value = s.get(key)
            if value is None or value == [] or value == "":
                continue
            text = ", ".join(map(str, value)) if isinstance(value, list) else value
            row = {"label": label, "value": text, "kind": "number" if isinstance(value, int) else "text"}
            if multi:
                row["trial"] = s.nct_id
            ds.emit(row, [s.ref(key)])
            tf[label] = text
        arms = [(i, a, p) for i, a, p in s.items("arm_groups") if isinstance(a, dict) and a.get("label")]
        for _, arm, apath in arms:
            row = {"label": f"Arm ({arm.get('type', 'arm')})", "value": arm["label"], "kind": "text"}
            if multi:
                row["trial"] = s.nct_id
            ds.emit(row, [s.ref(f"{apath}.label"), s.ref(f"{apath}.type")])
        tf["arms_derived"] = len(arms)
        facts[s.nct_id] = tf
    enc = Encoding(
        label=Channel(field="label", type="nominal", title="Design element"),
        y=Channel(field="value", type="nominal", title="Value"),
        series=Channel(field="trial", type="nominal", title="Trial") if multi else None,
    )
    viz = make_viz(
        VizType.KPI_CARDS,
        f"Study design: {', '.join(s.nct_id for s in ctx.trials)}",
        enc,
        ds.rows,
        VizMeta(layout=Layout(columns=3)),
    )
    return HandlerResult(
        visualization=viz,
        facts=facts if multi else next(iter(facts.values()), {}),
        full_refs=ds.full_refs,
        studies_analyzed=len(ctx.trials),
    )


def design_mix(ctx: HandlerContext) -> HandlerResult:
    """Cohort design mix: trials per allocation / masking / intervention model (or one ``group_by`` dimension)."""
    comparison = _comparison(ctx)
    chosen = ctx.plan.group_by if ctx.plan.group_by in DESIGN_DIMS else None
    dims = [chosen] if chosen else (["masking"] if comparison else ["allocation", "masking", "intervention_model"])
    ds = Dataset()
    facts: dict[str, Any] = {"trials_analyzed_derived": _analyzed(ctx), "dimensions": dims}
    order: list[str] = []
    for dim in dims:
        key, dtitle = DESIGN_DIMS[dim]

        def extract(s: Study, key: str = key) -> list[Item]:
            v = s.get(key)
            if isinstance(v, str) and v:
                return [(v, v, [s.ref(key)])]
            return [(NOT_REPORTED, NOT_REPORTED, [s.ref("nct_id")])]

        labels = LabelPicker()
        tallies, _ = _tally(_groups(ctx), extract, labels)
        totals: Counter[str] = Counter()
        for b in tallies.values():
            for k, v in b.items():
                totals[k] += v.count
        keys = top_keys(dict(totals), len(totals))
        if NOT_REPORTED in keys:
            keys.remove(NOT_REPORTED)
            keys.append(NOT_REPORTED)
        dfacts: dict[str, Any] = {}
        for glabel, buckets in tallies.items():
            g: dict[str, int] = {}
            for k in keys:
                b = buckets.get(k)
                if (b is None or not b.count) and not comparison:
                    continue
                cat = k if len(dims) == 1 else f"{dtitle}: {k}"
                row: dict[str, Any] = {
                    "category": cat,
                    "value": k,
                    "dimension": dtitle,
                    "trial_count": b.count if b else 0,
                }
                if comparison:
                    row["cohort"] = glabel
                ds.emit(row, b.refs if b else [])
                g[k] = row["trial_count"]
                if cat not in order:
                    order.append(cat)
            dfacts[glabel] = g
        facts[dim] = dfacts if comparison else next(iter(dfacts.values()), {})
    series = (
        COHORT
        if comparison
        else (Channel(field="dimension", type="nominal", title="Design dimension") if len(dims) > 1 else None)
    )
    xtitle = DESIGN_DIMS[dims[0]][1] if len(dims) == 1 else "Design"
    viz = bar_chart(
        ds.rows,
        title=f"Study design mix: {_title_label(ctx)}",
        x=Channel(field="category", type="nominal", title=xtitle, sort=ordered(order)),
        y=COUNT,
        series=series,
        orientation="horizontal",
        meta=VizMeta(grouping={"by": dims, "aggregate": "count_distinct_trials"}),
    )
    return HandlerResult(
        visualization=viz, notes=_base_notes(ctx), facts=facts, full_refs=ds.full_refs, studies_analyzed=_analyzed(ctx)
    )
