"""Exact counts for cohorts too large to download: one ClinicalTrials.gov ``totalCount`` per bucket.

When a cohort matches more trials than ``max_studies`` (e.g. 11,467 type 2 diabetes trials), counting
the records we managed to download would describe whichever trials the API happened to return first
(the most recently updated ones), not the registry. Instead, for every chart dimension that maps to a
registry field, each bucket is its own filtered search with ``countTotal=true`` and a tiny page:

    cohort search  AND  AREA[StartDate]RANGE[2019-01-01,2019-12-31]   -> totalCount = trials started in 2019

Buckets are mutually exclusive where the field is single-valued (start year, phase combination,
status, sponsor class), and extra "outside the range" / "not reported" buckets are counted too, so
``sum(buckets) == totalCount`` can be checked by the validation step. Multi-valued fields
(intervention type, country) count each trial once per value, exactly like the full-record handlers.

Each datum carries ``query_url`` (the ClinicalTrials.gov search that reproduces the number), a
``derived_count`` citation with that total, and verbatim field citations from a few example trials
returned by the same search. ``nct_ids`` is omitted because the bucket's members were not all fetched.

Supported dimensions: start year (trend), phase, status, lead sponsor class, intervention type, country.
Other intents keep the sampled path and say so (``SAMPLED`` note).
"""

from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ctviz.api.schemas.plan import Intent, QueryPlan, Scope
from ctviz.api.schemas.response import Note, NoteCode
from ctviz.api.schemas.viz import Channel, Sort, VizMeta, VizType
from ctviz.compute.cohort import LabelPicker, country_key
from ctviz.core.provenance import Dataset, FieldRef, Study, derived_count_ref
from ctviz.ctgov.fields import PHASE_LABELS, STATUS_GROUP_LABELS, STATUS_GROUPS, STATUS_ORDER, status_phrase
from ctviz.ctgov.retrieval import budget_left, facet_counts
from ctviz.viz.base import CohortData, HandlerContext, HandlerResult, note
from ctviz.viz.builders.bar_chart import bar_chart, ordered
from ctviz.viz.builders.geo_map import geo_map
from ctviz.viz.builders.pie_chart import pie_chart
from ctviz.viz.builders.time_series import time_series

SAMPLE_PER_BUCKET = 3  # example trials cited per bucket (same call as the count)
YEAR_WINDOW = 20  # default years plotted when the question gives no start year
MIN_YEARS = 8  # never fewer years than this, even when the call budget is tight
DEFAULT_TOP_COUNTRIES = 15

COUNT = Channel(field="trial_count", type="quantitative", title="Trials", unit="trials", format=",d")
COHORT = Channel(field="cohort", type="nominal", title="Cohort")

_P = "AREA[Phase]"
PHASE_BUCKETS: list[tuple[str, str]] = [  # joined-combination semantics, identical to the full-record handler
    ("EARLY_PHASE1", f"{_P}EARLY_PHASE1"),
    ("PHASE1", f"{_P}PHASE1 AND NOT {_P}PHASE2"),
    ("PHASE1/PHASE2", f"{_P}PHASE1 AND {_P}PHASE2"),
    ("PHASE2", f"{_P}PHASE2 AND NOT {_P}PHASE1 AND NOT {_P}PHASE3"),
    ("PHASE2/PHASE3", f"{_P}PHASE2 AND {_P}PHASE3"),
    ("PHASE3", f"{_P}PHASE3 AND NOT {_P}PHASE2"),
    ("PHASE4", f"{_P}PHASE4"),
    ("NA", f"({_P}NA OR {_P}MISSING)"),
]
SPONSOR_CLASSES = ["INDUSTRY", "NIH", "FED", "OTHER_GOV", "OTHER", "NETWORK", "INDIV", "AMBIG", "UNKNOWN"]
INTERVENTION_TYPES = [
    "DRUG",
    "BIOLOGICAL",
    "DEVICE",
    "PROCEDURE",
    "BEHAVIORAL",
    "DIETARY_SUPPLEMENT",
    "RADIATION",
    "GENETIC",
    "DIAGNOSTIC_TEST",
    "COMBINATION_PRODUCT",
    "OTHER",
]
TYPE_QUERY = re.compile(r"intervention\s+types?|types?\s+of\s+(intervention|treatment)", re.IGNORECASE)


SITE_QUERY = re.compile(
    r"\b(?:trial )?sites?\b|\bfacilit(?:y|ies)\b|\bcities\b|\bcity\b|\bhospitals?\b|"
    r"\bclinics?\b|\bwhere exactly\b|\bnear\b",
    re.IGNORECASE,
)


def site_level(plan: QueryPlan, query: str) -> bool:
    """Geography at site level (facility/city points) instead of per-country counts."""
    return plan.intent is Intent.GEOGRAPHIC_DISTRIBUTION and bool(SITE_QUERY.search(query))


def by_intervention_type(plan: QueryPlan, query: str) -> bool:
    return plan.group_by == "intervention_type" or (plan.group_by is None and bool(TYPE_QUERY.search(query)))


def supports(plan: QueryPlan, query: str) -> bool:
    """True when the plan's chart dimension can be counted server-side (exactly) for a large cohort."""
    if plan.scope is Scope.SINGLE_TRIAL or plan.nct_ids:
        return False
    i = plan.intent
    return (
        (i is Intent.TREND_OVER_TIME and plan.series_by is None and (plan.time_granularity or "year") == "year")
        or i in (Intent.PHASE_DISTRIBUTION, Intent.STATUS_DISTRIBUTION)
        or (i is Intent.GEOGRAPHIC_DISTRIBUTION and not site_level(plan, query))
        or (i is Intent.SPONSOR_LANDSCAPE and plan.group_by == "sponsor_class")
        or (i is Intent.INTERVENTION_LANDSCAPE and by_intervention_type(plan, query))
    )


def active(ctx: HandlerContext) -> bool:
    return any(c.counting == "registry_counts" for c in ctx.cohorts)


def _comparison(ctx: HandlerContext) -> bool:
    return ctx.plan.scope is Scope.COMPARISON and len(ctx.cohorts) > 1


def _title_label(ctx: HandlerContext) -> str:
    phrases = {status_phrase(cf.statuses) for cf in ctx.plan.cohorts}
    phrase = phrases.pop() if len(phrases) == 1 else None
    label = " vs ".join(c.label for c in ctx.cohorts)
    return f"{label} ({phrase})" if phrase else label


# -- counting -------------------------------------------------------------------------------------
@dataclass
class Counted:
    key: str
    label: str
    count: int
    url: str
    sample: list[Study]


RefFn = Callable[[Study, str], FieldRef | None]


async def count_buckets(ctx: HandlerContext, cohort: CohortData, buckets: list[tuple[str, str, str]]) -> list[Counted]:
    """``buckets`` = [(key, label, Essie clause)] -> exact count + example trials per bucket (one call each)."""
    res = await facet_counts(ctx.client, cohort.spec, {k: clause for k, _, clause in buckets}, sample=SAMPLE_PER_BUCKET)
    return [Counted(key=k, label=label, count=res[k][0], sample=res[k][1], url=res[k][2]) for k, label, _ in buckets]


def _emit(ds: Dataset, row: dict[str, Any], c: Counted, ref_fn: RefFn) -> None:
    refs = [derived_count_ref(c.count, c.url, c.label)] if c.count else []
    refs += [ref_fn(s, c.key) for s in c.sample] if c.count else []
    ds.emit({**row, "query_url": c.url}, refs, nct_ids=None)


def _registry_note(ctx: HandlerContext, what: str) -> Note:
    totals = ", ".join(f"{c.label}: {c.total_count:,}" for c in ctx.cohorts)
    return note(
        NoteCode.REGISTRY_COUNTS,
        f"Too many matching trials to download ({totals}), so every {what} is an exact count returned by "
        "ClinicalTrials.gov for that filtered search (each datum's query_url reproduces it). Citations show "
        "a few example trials per bar instead of listing every NCT ID.",
    )


# -- reference functions (which field put an example trial in its bucket) ---------------------------
def _ref_start(s: Study, _: str) -> FieldRef | None:
    return s.ref("start_date") or s.ref("nct_id")


def _ref_phase(s: Study, _: str) -> FieldRef | None:
    return s.ref("phases") or s.ref("nct_id")


def _ref_status(s: Study, _: str) -> FieldRef | None:
    return s.ref("status") or s.ref("nct_id")


def _ref_sponsor_class(s: Study, _: str) -> FieldRef | None:
    return s.ref("lead_sponsor_class") or s.ref("nct_id")


def _ref_intervention_type(s: Study, key: str) -> FieldRef | None:
    for _, item, path in s.items("interventions"):
        if isinstance(item, dict) and item.get("type") == key:
            return s.ref(f"{path}.type")
    return s.ref("nct_id")


def _ref_country(s: Study, key: str) -> FieldRef | None:
    for _, loc, path in s.items("locations"):
        if isinstance(loc, dict) and country_key(loc.get("country")) == key:
            return s.ref(f"{path}.country")
    return s.ref("nct_id")


# -- trend ----------------------------------------------------------------------------------------
async def _earliest_year(ctx: HandlerContext, cohort: CohortData) -> int | None:
    page = await ctx.client.search_page(
        cohort.spec.without(sort="StartDate:asc"),
        page_size=1,
        count_total=False,
        fields_override=["NCTId", "StartDate"],
    )
    for raw in page.studies:
        date = ((raw.get("protocolSection") or {}).get("statusModule") or {}).get("startDateStruct", {}).get("date")
        if isinstance(date, str) and date[:4].isdigit():
            return int(date[:4])
    return None


async def trend(ctx: HandlerContext, today: dt.date | None = None) -> HandlerResult:
    """Trials started per year, one exact count per year, reconciled against totalCount."""
    this_year = (today or dt.date.today()).year
    comparison = _comparison(ctx)
    cfs = ctx.plan.cohorts
    req_lo = next((c.start_year for c in cfs if c.start_year), None)
    req_hi = next((c.end_year for c in cfs if c.end_year), None)
    if req_lo is None:
        earliest = [y for y in [await _earliest_year(ctx, c) for c in ctx.cohorts] if y]
        lo = max(min(earliest) if earliest else this_year - YEAR_WINDOW + 1, this_year - YEAR_WINDOW + 1)
    else:
        lo = req_lo
    hi = min(req_hi or this_year, this_year + 5)
    left = budget_left()
    if left is not None and req_lo is None:  # fit the request's call budget: older years join "before"
        per_cohort = left // max(1, len(ctx.cohorts)) - 7  # 3 outside-range + up to 4 planned-start buckets
        lo = max(lo, hi - max(MIN_YEARS, per_cohort) + 1)
    years = list(range(lo, hi + 1))
    ds = Dataset()
    facts: dict[str, Any] = {
        "granularity": "year",
        "first_period": str(lo),
        "last_period": str(hi),
        "trials_analyzed_derived": sum(c.total_count for c in ctx.cohorts),
    }
    per_series: dict[str, dict[str, int]] = {}
    reconcile: dict[str, dict[str, int]] = {}
    # Planned (ESTIMATED) start dates: counted per year where they matter (last year onward); earlier years
    # get one combined count, which splits exactly only when it is 0 (else those bars carry null).
    recent = [y for y in years if y >= this_year - 1]
    early = [y for y in years if y < this_year - 1]
    est = " AND AREA[StartDateType]ESTIMATED"
    notes: list[Note] = []
    for cohort in ctx.cohorts:
        buckets = [(str(y), str(y), f"AREA[StartDate]RANGE[{y}-01-01,{y}-12-31]") for y in years]
        buckets += [
            (f"est:{y}", f"{y} (estimated start)", f"AREA[StartDate]RANGE[{y}-01-01,{y}-12-31]{est}") for y in recent
        ]
        if early:
            buckets.append(
                (
                    "est:early",
                    f"{early[0]}-{early[-1]} (estimated start)",
                    f"AREA[StartDate]RANGE[{early[0]}-01-01,{early[-1]}-12-31]{est}",
                )
            )
        extra: list[tuple[str, str, str]] = []
        if req_lo is None:
            extra.append(("before", f"before {lo}", f"AREA[StartDate]RANGE[MIN,{lo - 1}-12-31]"))
            extra.append(("missing", "no start date", "AREA[StartDate]MISSING"))
        if req_hi is None:
            extra.append(("after", f"after {hi}", f"AREA[StartDate]RANGE[{hi + 1}-01-01,MAX]"))
        counted = await count_buckets(ctx, cohort, buckets + extra)
        by_key = {c.key: c for c in counted}
        early_est = by_key["est:early"].count if early else 0
        if early_est:
            prefix = f"{cohort.label}: " if comparison else ""
            notes.append(
                note(
                    NoteCode.PARTIAL_DATES,
                    f"{prefix}{early_est:,} trial(s) starting {early[0]}-{early[-1]} "
                    "still list an estimated start date; estimated_count is only "
                    f"split by year from {this_year - 1} (null before).",
                )
            )
        series: dict[str, int] = {}
        for y in years:
            c = by_key[str(y)]
            planned = by_key[f"est:{y}"].count if y in recent else (None if early_est else 0)
            row: dict[str, Any] = {
                "period": f"{y}-01-01",
                "year": y,
                "label": str(y),
                "trial_count": c.count,
                "estimated_count": planned,
            }
            if comparison:
                row["cohort"] = cohort.label
            _emit(ds, row, c, _ref_start)
            series[str(y)] = c.count
        per_series[cohort.label] = series
        outside = {k: by_key[k].count for k in ("before", "after", "missing") if k in by_key}
        reconcile[cohort.label] = {
            "plotted_derived": sum(series.values()),
            **{f"{k}_derived": v for k, v in outside.items()},
            "total_derived": cohort.total_count,
        }
    notes.insert(0, _registry_note(ctx, "year"))
    for label, r in reconcile.items():
        parts = [
            f"{r['before_derived']:,} started before {lo}" if r.get("before_derived") else "",
            f"{r['after_derived']:,} have start dates after {hi} (planned)" if r.get("after_derived") else "",
            f"{r['missing_derived']:,} have no start date" if r.get("missing_derived") else "",
        ]
        parts = [p for p in parts if p]
        if parts:
            prefix = f"{label}: " if comparison else ""
            notes.append(note(NoteCode.PARTIAL_DATES, f"{prefix}{'; '.join(parts)}; these are not plotted."))
    facts["reconciliation"] = reconcile
    if comparison:
        facts["by_series"] = per_series
        facts["by_series_peak"] = {k: _peak(v) for k, v in per_series.items()}
    else:
        only = next(iter(per_series.values()))
        facts["counts"] = only
        peak = _peak(only)
        facts.update(peak_period=peak["period"], peak_count=peak["count"])
    viz = time_series(
        ds.rows,
        title=f"Trials started per year: {_title_label(ctx)}",
        x=Channel(field="period", type="temporal", title="Start year", time_format="%Y"),
        y=COUNT,
        granularity="year",
        series=COHORT if comparison else None,
        tooltip=["label", "estimated_count"],
        meta=VizMeta(
            grouping={
                "by": ["start_date"],
                "aggregate": "registry_count",
                "zero_filled": True,
                "each_trial_counted_once": True,
            },
            units={"trial_count": "trials", "estimated_count": "trials"},
        ),
    )
    return HandlerResult(
        visualization=viz,
        notes=notes,
        facts=facts,
        full_refs=ds.full_refs,
        studies_analyzed=sum(c.total_count for c in ctx.cohorts),
    )


def _peak(counts: dict[str, int]) -> dict[str, Any]:
    period, count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return {"period": period, "count": count}


# -- categorical ----------------------------------------------------------------------------------
async def categorical(
    ctx: HandlerContext,
    *,
    field_name: str,
    axis_title: str,
    title: str,
    buckets: list[tuple[str, str, str]],
    ref_fn: RefFn,
    fixed_order: bool,
    what: str,
    remainder_label: str | None = None,
    multi_valued: bool = False,
) -> HandlerResult:
    """Bar (or preferred pie) of exact per-bucket counts; optional remainder = total - sum (exclusive dims)."""
    comparison = _comparison(ctx)
    ds = Dataset()
    per_group: dict[str, dict[str, int]] = {}
    counted_by: dict[str, list[Counted]] = {}
    for cohort in ctx.cohorts:
        counted_by[cohort.label] = await count_buckets(ctx, cohort, buckets)
    totals: Counter[str] = Counter()
    for cs in counted_by.values():
        for c in cs:
            totals[c.key] += c.count
    keys = [k for k, _, _ in buckets if totals[k] or comparison]
    if not fixed_order:
        keys.sort(key=lambda k: -totals[k])
    labels = {k: lab for k, lab, _ in buckets}
    reconcile: dict[str, dict[str, int]] = {}
    for cohort in ctx.cohorts:
        by_key = {c.key: c for c in counted_by[cohort.label]}
        g: dict[str, int] = {}
        for k in keys:
            c = by_key[k]
            if not c.count and not comparison:
                continue
            row: dict[str, Any] = {field_name: labels[k], "trial_count": c.count}
            if comparison:
                row["cohort"] = cohort.label
            _emit(ds, row, c, ref_fn)
            g[labels[k]] = c.count
        summed = sum(c.count for c in counted_by[cohort.label])
        if remainder_label and not multi_valued:
            rest = cohort.total_count - summed
            if rest > 0:
                row = {field_name: remainder_label, "trial_count": rest, "is_other": True}
                if comparison:
                    row["cohort"] = cohort.label
                ds.emit(
                    {**row, "query_url": cohort.search_url, "cite_note": "total minus the counted buckets (derived)"},
                    [],
                    nct_ids=None,
                )
                g[remainder_label] = rest
            reconcile[cohort.label] = {
                "buckets_derived": summed,
                "remainder_derived": max(rest, 0),
                "total_derived": cohort.total_count,
            }
        per_group[cohort.label] = g
    facts: dict[str, Any] = {
        "trials_analyzed_derived": sum(c.total_count for c in ctx.cohorts),
        "categories_total_derived": len(keys),
    }
    if reconcile:
        facts["reconciliation"] = reconcile
    if comparison:
        facts["by_cohort"] = per_group
        facts["trials_by_cohort_derived"] = {c.label: c.total_count for c in ctx.cohorts}
        facts["by_cohort_top"] = {g: _top(v) for g, v in per_group.items()}
    else:
        only = next(iter(per_group.values()), {})
        facts["counts"] = only
        top = _top(only)
        if top:
            facts["top_category"], facts["top_category_count"] = top["category"], top["count"]
    notes = [_registry_note(ctx, what)]
    if multi_valued:
        notes.append(
            note(
                NoteCode.APPROXIMATED,
                f"A trial with several {what}s counts once in each, so bars can sum to more than the number of trials.",
            )
        )
    meta = VizMeta(
        grouping={
            "by": [field_name],
            "aggregate": "registry_count",
            **({"multi_valued": True} if multi_valued else {"each_trial_counted_once": True}),
        }
    )
    shown = [labels[k] for k in keys] + (
        [remainder_label] if remainder_label and any(remainder_label in g for g in per_group.values()) else []
    )
    if not comparison and ctx.plan.preferred_visualization is VizType.PIE_CHART and not multi_valued and ds.rows:
        viz = pie_chart(
            ds.rows,
            title=title,
            label=Channel(field=field_name, type="nominal", title=axis_title),
            value=COUNT,
            meta=meta,
        )
    else:
        x = Channel(
            field=field_name, type="ordinal" if fixed_order else "nominal", title=axis_title, sort=ordered(shown)
        )
        viz = bar_chart(ds.rows, title=title, x=x, y=COUNT, series=COHORT if comparison else None, meta=meta)
    return HandlerResult(
        visualization=viz,
        notes=notes,
        facts=facts,
        full_refs=ds.full_refs,
        studies_analyzed=sum(c.total_count for c in ctx.cohorts),
    )


def _top(counts: dict[str, int]) -> dict[str, Any] | None:
    named = [(k, v) for k, v in counts.items() if v > 0 and not k.startswith("Other")]
    if not named:
        return None
    k, v = max(named, key=lambda kv: kv[1])
    return {"category": k, "count": v}


async def phases(ctx: HandlerContext) -> HandlerResult:
    return await categorical(
        ctx,
        field_name="phase",
        axis_title="Phase",
        title=f"Trials by phase: {_title_label(ctx)}",
        buckets=[(k, PHASE_LABELS[k], clause) for k, clause in PHASE_BUCKETS],
        ref_fn=_ref_phase,
        fixed_order=True,
        what="phase",
        remainder_label="Phase not reported",
    )


async def statuses(ctx: HandlerContext, labels: dict[str, str]) -> HandlerResult:
    return await categorical(
        ctx,
        field_name="status",
        axis_title="Status",
        title=f"Trials by recruitment status: {_title_label(ctx)}",
        buckets=[(s, labels.get(s, s), f"AREA[OverallStatus]{s}") for s in STATUS_ORDER],
        ref_fn=_ref_status,
        fixed_order=True,
        what="status",
        remainder_label="Other statuses (expanded access)",
    )


async def status_groups(ctx: HandlerContext) -> HandlerResult:
    """Active / done / other, each an exact count (one OR-ed status search per group)."""
    return await categorical(
        ctx,
        field_name="status_group",
        axis_title="Status today",
        title=f"Trials by status today (active / done / other): {_title_label(ctx)}",
        buckets=[
            (g, STATUS_GROUP_LABELS[g], f"AREA[OverallStatus]({' OR '.join(members)})")
            for g, members in STATUS_GROUPS.items()
        ],
        ref_fn=_ref_status,
        fixed_order=True,
        what="status group",
        remainder_label="Not reported",
    )


async def sponsor_classes(ctx: HandlerContext, labels: dict[str, str]) -> HandlerResult:
    return await categorical(
        ctx,
        field_name="sponsor_class",
        axis_title="Sponsor class",
        title=f"Trials by lead sponsor class: {_title_label(ctx)}",
        buckets=[(k, labels.get(k, k), f"AREA[LeadSponsorClass]{k}") for k in SPONSOR_CLASSES],
        ref_fn=_ref_sponsor_class,
        fixed_order=False,
        what="sponsor class",
        remainder_label="Not reported",
    )


async def intervention_types(ctx: HandlerContext, labels: dict[str, str]) -> HandlerResult:
    return await categorical(
        ctx,
        field_name="intervention_type",
        axis_title="Intervention type",
        title=f"Trials by intervention type: {_title_label(ctx)}",
        buckets=[(k, labels.get(k, k), f"AREA[InterventionType]{k}") for k in INTERVENTION_TYPES],
        ref_fn=_ref_intervention_type,
        fixed_order=False,
        what="intervention type",
        multi_valued=True,
    )


# -- geography ------------------------------------------------------------------------------------
async def countries(ctx: HandlerContext) -> HandlerResult:
    """Short-list countries from the retrieved first page, then count each one exactly (one trial = one count
    per country, as the registry search counts trials, not sites)."""
    comparison = _comparison(ctx)
    picker = LabelPicker()
    freq: Counter[str] = Counter()
    for c in ctx.cohorts:
        for s in c.studies:
            seen: set[str] = set()
            for _, loc, _p in s.items("locations"):
                raw = loc.get("country") if isinstance(loc, dict) else None
                key = country_key(raw)
                if key and key not in seen:
                    seen.add(key)
                    picker.add(key, " ".join(raw.split()))
                    freq[key] += 1
    top_n = ctx.plan.top_n or ctx.request.top_n or DEFAULT_TOP_COUNTRIES
    keys = [k for k, _ in freq.most_common(top_n)]
    buckets = [(k, picker.label(k), f'AREA[LocationCountry]"{picker.label(k)}"') for k in keys]
    buckets.append(("__none__", "No location listed", "AREA[LocationCountry]MISSING"))
    ds = Dataset()
    per_group: dict[str, dict[str, int]] = {}
    no_loc: dict[str, int] = {}
    for cohort in ctx.cohorts:
        counted = await count_buckets(ctx, cohort, buckets)
        g: dict[str, int] = {}
        for c in counted:
            if c.key == "__none__":
                no_loc[cohort.label] = c.count
                continue
            if not c.count:
                continue
            row: dict[str, Any] = {"country": c.label, "trial_count": c.count}
            if comparison:
                row["cohort"] = cohort.label
            _emit(ds, row, c, _ref_country)
            g[c.label] = c.count
        per_group[cohort.label] = g
    sample_n = sum(len(c.studies) for c in ctx.cohorts)
    notes = [
        _registry_note(ctx, "country"),
        note(
            NoteCode.APPROXIMATED,
            f"The {len(keys)} countries shown were short-listed from the {sample_n:,} "
            "most recently updated matching trials; the count for each one is exact. "
            "A trial counts once per country, however many sites it has there.",
        ),
    ]
    if any(no_loc.values()):
        notes.append(
            note(
                NoteCode.NOT_IN_REGISTRY,
                "; ".join(f"{k}: {v:,}" for k, v in no_loc.items() if v)
                + " trial(s) list no site location and are not mapped.",
            )
        )
    facts: dict[str, Any] = {
        "trials_analyzed_derived": sum(c.total_count for c in ctx.cohorts),
        "trials_without_locations_derived": sum(no_loc.values()),
    }
    if comparison:
        facts["by_cohort"] = per_group
    else:
        counts = next(iter(per_group.values()), {})
        facts["counts"] = counts
        if counts:
            top = max(counts, key=lambda k: counts[k])
            facts.update(top_country=top, top_country_trials=counts[top])
    viz = geo_map(
        ds.rows,
        title=f"Trials by country: {_title_label(ctx)}",
        region=Channel(field="country", type="nominal", title="Country"),
        value=Channel(field="trial_count", type="quantitative", title="Trials", unit="trials", format=",d"),
        series=COHORT if comparison else None,
        meta=VizMeta(
            sort=Sort(by="trial_count", order="desc"),
            grouping={"by": ["country"], "aggregate": "registry_count", "per_trial": "once per country"},
        ),
    )
    return HandlerResult(
        visualization=viz,
        notes=notes,
        facts=facts,
        full_refs=ds.full_refs,
        studies_analyzed=sum(c.total_count for c in ctx.cohorts),
    )
