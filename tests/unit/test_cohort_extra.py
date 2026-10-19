"""Extra coverage for the cohort handlers, compute helpers and chart builders.

* Independent recounts: each handler's headline counts are recomputed here from the raw JSON with
  small, separate implementations (no ctviz helpers) and compared, on all six cohort fixtures.
* Facts audit: every number in ``HandlerResult.facts`` is a value in ``visualization.data`` or is on
  the documented list of derived facts (``DERIVED``); the derivable ones are re-derived from the data.
* Citation invariant for every handler configuration on every fixture (including the full,
  uncapped evidence in ``full_refs``), plus "zero evidence <=> cite_note".
* Empty cohorts and single-study cohorts never crash and still give renderable output.
* Geography counts a trial at most once per country; network edges reference existing nodes and
  their weight is the number of distinct cited trials.

All offline: cohorts come from recorded fixtures, there is no client, DB or LLM.
"""

from __future__ import annotations

import datetime as dt
import functools
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from fnmatch import fnmatch
from typing import Any

import pytest

from ctviz.api.schemas import Intent, Scope, VizType
from ctviz.api.schemas.viz import Channel, Datum, GraphData, Visualization
from ctviz.core.provenance import Dataset, GraphDataset, Study, verify_citation
from ctviz.ctgov.fields import PHASE_LABELS
from ctviz.ctgov.requests import SearchSpec
from ctviz.handlers.cohort import INTERVENTION_TYPE_LABELS, SPONSOR_CLASS_LABELS, STATUS_LABELS, canonical_drug
from ctviz.viz.base import CohortData, EncodingError, check_encoding
from ctviz.viz.builders.bar_chart import bar_chart, ordered
from ctviz.viz.builders.geo_map import geo_map
from ctviz.viz.builders.heatmap import heatmap
from ctviz.viz.builders.histogram import histogram
from ctviz.viz.builders.network_graph import network_graph
from ctviz.viz.builders.pie_chart import pie_chart
from ctviz.viz.builders.scatter_plot import scatter_plot
from ctviz.viz.builders.time_series import time_series
from ctviz.viz.builders.timeline import timeline
from tests.conftest import assert_citations_verified, load_cohort_raw
from tests.unit.cohort_helpers import data_numbers, rows, run

FIXTURES = [
    "melanoma_ph2_3",
    "obesity_since2020",
    "pembrolizumab_since2015",
    "semaglutide",
    "t2d_recruiting",
    "tirzepatide",
]

# name -> (intent, plan kwargs). Covers every cohort handler and every branch inside them.
CONFIGS: dict[str, tuple[Intent, dict[str, Any]]] = {
    "phase": (Intent.PHASE_DISTRIBUTION, {}),
    "status": (Intent.STATUS_DISTRIBUTION, {}),
    "status_pie": (Intent.STATUS_DISTRIBUTION, {"preferred_visualization": VizType.PIE_CHART}),
    "sponsor": (Intent.SPONSOR_LANDSCAPE, {}),
    "sponsor_class": (Intent.SPONSOR_LANDSCAPE, {"group_by": "sponsor_class"}),
    "sponsor_phase": (Intent.SPONSOR_LANDSCAPE, {"series_by": "phase"}),
    "intervention": (Intent.INTERVENTION_LANDSCAPE, {}),
    "intervention_type": (Intent.INTERVENTION_LANDSCAPE, {"group_by": "intervention_type"}),
    "trend_year": (Intent.TREND_OVER_TIME, {}),
    "trend_quarter": (Intent.TREND_OVER_TIME, {"time_granularity": "quarter"}),
    "trend_month": (Intent.TREND_OVER_TIME, {"time_granularity": "month"}),
    "trend_phase": (Intent.TREND_OVER_TIME, {"series_by": "phase"}),
    "trend_status": (Intent.TREND_OVER_TIME, {"series_by": "status"}),
    "phase_status": (Intent.PHASE_STATUS_HEATMAP, {}),
    "gap_intervention_phase": (Intent.LANDSCAPE_GAP_HEATMAP, {}),
    "gap_condition_status": (Intent.LANDSCAPE_GAP_HEATMAP, {"group_by": "condition", "series_by": "status"}),
    "gap_sponsor_year": (Intent.LANDSCAPE_GAP_HEATMAP, {"group_by": "sponsor", "series_by": "year"}),
    "gap_country_phase": (Intent.LANDSCAPE_GAP_HEATMAP, {"group_by": "country"}),
    "geo": (Intent.GEOGRAPHIC_DISTRIBUTION, {}),
    "enroll_hist": (Intent.ENROLLMENT_DISTRIBUTION, {}),
    "enroll_scatter": (Intent.ENROLLMENT_DISTRIBUTION, {"preferred_visualization": VizType.SCATTER_PLOT}),
    "net_sponsor_drug": (Intent.ENTITY_NETWORK, {"network_kind": "sponsor_drug"}),
    "net_drug_drug": (Intent.ENTITY_NETWORK, {"network_kind": "drug_drug"}),
    "net_drug_condition": (Intent.ENTITY_NETWORK, {"network_kind": "drug_condition"}),
    "net_site_investigator": (Intent.ENTITY_NETWORK, {"network_kind": "site_investigator"}),
    "net_sponsor_collaborator": (Intent.ENTITY_NETWORK, {"network_kind": "sponsor_collaborator"}),
    "activity": (Intent.SPONSOR_ACTIVITY, {}),
    "progression": (Intent.DEVELOPMENT_PROGRESSION, {}),
    "design": (Intent.DESIGN_METHODOLOGY, {}),
    "design_masking": (Intent.DESIGN_METHODOLOGY, {"group_by": "masking"}),
}

# Handlers whose output is split per cohort in comparison scope.
COMPARISON_CONFIGS = [
    "phase",
    "status",
    "sponsor",
    "sponsor_class",
    "intervention",
    "intervention_type",
    "trend_year",
    "geo",
    "enroll_hist",
    "enroll_scatter",
    "progression",
    "design",
]

# ------------------------------------------------------------------------------------------------
# Documented derived facts: numbers that are NOT a single value in visualization.data.
# Convention (ctviz/handlers/cohort.py docstring): any fact whose key path has a segment ending in
# ``_derived`` is a documented aggregate; the lists below name them (with the suffix) per family.
# "input" = computed from the studies (not visible in the chart); "data" = an aggregate of data values
# (re-derived in test_derived_facts_match_data).
# ------------------------------------------------------------------------------------------------
_INPUT_COMMON = [
    "trials_analyzed_derived",
    "trials_total_derived",
    "trials_without_*_derived",
    "trials_with_*_derived",
    "trials_skipped_derived",
    "trials_by_cohort_derived.*",
]
DERIVED: dict[str, list[str]] = {
    "categorical": _INPUT_COMMON
    + [
        "categories_total_derived",  # input: distinct categories before top-N truncation
    ],
    "sponsor_phase": _INPUT_COMMON
    + [
        "sponsors_total_derived",  # input: distinct sponsors before top-N
        "sponsor_totals_derived.*",  # data: sum of a sponsor's phase-stacked rows
    ],
    "trend": _INPUT_COMMON
    + [
        "trials_plotted_derived",  # data (single series): sum of trial_count
        "trials_plotted_distinct_derived",  # input (multi-series): distinct trials plotted
        "series_totals_derived.*",  # data: per-series sum of trial_count
    ],
    "heatmap": _INPUT_COMMON
    + [
        "cells_derived",
        "gap_cells_derived",
        "rows_derived",
        "columns_derived",  # data: matrix shape
        "trials_by_phase_derived.*",
        "trials_by_status_derived.*",  # data: column / row sums
        "row_totals_derived.*",  # input: distinct trials per row entity (cells can double count split phases)
    ],
    "geo": _INPUT_COMMON
    + [
        "countries_total_derived",  # input: distinct countries before top-N
        "top_country_trials_distinct_derived",  # input (comparison): distinct trials across cohorts
    ],
    "enrollment": _INPUT_COMMON
    + [
        "stats_derived.*",  # input: actual / estimated enrollment summaries
        "trials_plotted_derived",  # data: number of scatter points
    ],
    "network": _INPUT_COMMON
    + [
        "nodes_derived",
        "edges_derived",  # data: len(nodes), len(edges)
        "nodes_total_derived",
        "edges_total_derived",  # input: graph size before pruning
        "node_type_counts_derived.*",  # data: nodes per type
        "min_edge_weight_derived",  # input: pruning threshold
    ],
    "activity": _INPUT_COMMON
    + [
        "trials_shown_derived",  # data: number of rows
        "trials_by_phase_derived.*",
        "trials_by_status_derived.*",  # data: rows per lane / status
    ],
    "progression": _INPUT_COMMON,
    "design": _INPUT_COMMON,
}
FAMILY = {
    "phase": "categorical",
    "status": "categorical",
    "status_pie": "categorical",
    "sponsor": "categorical",
    "sponsor_class": "categorical",
    "intervention": "categorical",
    "intervention_type": "categorical",
    "sponsor_phase": "sponsor_phase",
    "trend_year": "trend",
    "trend_quarter": "trend",
    "trend_month": "trend",
    "trend_phase": "trend",
    "trend_status": "trend",
    "phase_status": "heatmap",
    "gap_intervention_phase": "heatmap",
    "gap_condition_status": "heatmap",
    "gap_sponsor_year": "heatmap",
    "gap_country_phase": "heatmap",
    "geo": "geo",
    "enroll_hist": "enrollment",
    "enroll_scatter": "enrollment",
    "activity": "activity",
    "progression": "progression",
    "design": "design",
    "design_masking": "design",
    **{k: "network" for k in CONFIGS if k.startswith("net_")},
}


# ------------------------------------------------------------------------------------------------
# Fixture loading (cached: the recorded cohorts are read once per test session)
# ------------------------------------------------------------------------------------------------


@functools.cache
def _raw_studies(name: str) -> tuple[dict, ...]:
    return tuple(load_cohort_raw(name)["studies"])


def load(name: str, label: str | None = None) -> CohortData:
    studies = [Study(r) for r in _raw_studies(name)]
    return CohortData(label=label or name, spec=SearchSpec(), studies=studies, total_count=len(studies))


def make_cohort(label: str, raws: list[dict]) -> CohortData:
    return CohortData(label=label, spec=SearchSpec(), studies=[Study(r) for r in raws], total_count=len(raws))


async def run_config(make_ctx, name: str, cohorts: list[CohortData], scope: Scope | None = None):
    intent, kwargs = CONFIGS[name]
    return await run(intent, make_ctx(intent, cohorts=cohorts, scope=scope, **kwargs))


# ------------------------------------------------------------------------------------------------
# Tiny independent implementations (raw JSON only; no ctviz compute helpers)
# ------------------------------------------------------------------------------------------------


def g(raw: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = raw
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def nct(r: dict) -> str:
    return r["protocolSection"]["identificationModule"]["nctId"]


def phases(r: dict) -> list[str]:
    return g(r, "protocolSection.designModule.phases") or []


def joined_phase(r: dict) -> str:
    return "/".join(phases(r)) if phases(r) else "NA"


def status(r: dict) -> str | None:
    v = g(r, "protocolSection.statusModule.overallStatus")
    return v if isinstance(v, str) else None


def lead(r: dict) -> Any:
    return g(r, "protocolSection.sponsorCollaboratorsModule.leadSponsor.name")


def norm(name: Any) -> str:
    if not isinstance(name, str):
        return ""
    t = unicodedata.normalize("NFKC", name)
    t = re.sub(r"[®™©℠]", "", t).casefold()
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


_PLACEBO = re.compile(r"\b(placebos?|sham|dummy|vehicle)\b", re.IGNORECASE)


def interventions(r: dict) -> list[dict]:
    return [i for i in g(r, "protocolSection.armsInterventionsModule.interventions") or [] if isinstance(i, dict)]


def drug_keys(r: dict, types: set[str] | None = None) -> set[str]:
    out = set()
    for i in interventions(r):
        name = i.get("name")
        if types is not None and i.get("type") not in types:
            continue
        if isinstance(name, str) and norm(name) and not _PLACEBO.search(name):
            out.add(norm(canonical_drug(name) or name))  # variants of one vocabulary drug are one drug
    return out


def countries(r: dict) -> set[str]:
    locs = g(r, "protocolSection.contactsLocationsModule.locations") or []
    return {
        loc["country"]
        for loc in locs
        if isinstance(loc, dict) and isinstance(loc.get("country"), str) and loc["country"].strip()
    }


def conditions(r: dict) -> set[str]:
    return {norm(c) for c in g(r, "protocolSection.conditionsModule.conditions") or [] if norm(c)}


def parse_date(text: Any) -> tuple[dt.date, str] | None:
    if not isinstance(text, str):
        return None
    m = re.fullmatch(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?", text.strip())
    if not m:
        return None
    y, mo, d = m.groups()
    try:
        return dt.date(int(y), int(mo or 1), int(d or 1)), ("day" if d else "month" if mo else "year")
    except ValueError:
        return None


def start(r: dict) -> tuple[dt.date, str] | None:
    return parse_date(g(r, "protocolSection.statusModule.startDateStruct.date"))


def enrollment(r: dict) -> int | None:
    v = g(r, "protocolSection.designModule.enrollmentInfo.count")
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def dedup(raws) -> list[dict]:
    seen: dict[str, dict] = {}
    for r in raws:
        seen.setdefault(nct(r), r)
    return list(seen.values())


def raw_by_id(*cohorts: CohortData) -> dict[str, dict]:
    return {s.nct_id: s.raw for c in cohorts for s in c.studies}


def other_label(counts: dict[str, int]) -> str | None:
    return next((k for k in counts if k.startswith("Other (")), None)


def assert_top_n(facts_counts: dict[str, int], expected: Counter, n: int, trials_with: dict[str, set[str]]):
    """facts['counts'] holds the top-n categories (label -> count) plus an 'Other' bucket of distinct trials."""
    other = other_label(facts_counts)
    shown = {k: v for k, v in facts_counts.items() if k != other}
    for label, v in shown.items():
        assert expected[norm(label)] == v, label
    # The shown categories are the n largest (compare the multiset of counts; ties may pick either name).
    assert sorted(shown.values(), reverse=True) == sorted(expected.values(), reverse=True)[:n]
    rest = set(expected) - {norm(k) for k in shown}
    if rest:
        assert other is not None
        assert facts_counts[other] == len(set().union(*(trials_with[k] for k in rest)))
        assert other == f"Other ({len(rest)} more)"
    else:
        assert other is None


# ================================================================================================
# (1) Independent recounts, every fixture
# ================================================================================================


@pytest.mark.parametrize("name", FIXTURES)
async def test_recount_phase_distribution(make_ctx, name):
    c = load(name)
    res = await run_config(make_ctx, "phase", [c])
    expected = Counter(joined_phase(r) for r in dedup(_raw_studies(name)))
    assert {k.split("|", 1)[1]: v for k, v in res.facts.items() if "|" in k} == {
        PHASE_LABELS.get(p, p): n for p, n in expected.items()
    }
    assert {r["phase"]: r["trial_count"] for r in rows(res.visualization)} == {
        PHASE_LABELS.get(p, p): n for p, n in expected.items()
    }


@pytest.mark.parametrize("name", FIXTURES)
async def test_recount_status_distribution(make_ctx, name):
    res = await run_config(make_ctx, "status", [load(name)])
    expected = Counter(
        STATUS_LABELS.get(status(r), status(r)) if status(r) else "Not reported" for r in dedup(_raw_studies(name))
    )
    assert res.facts["counts"] == dict(expected)
    assert {r["status"]: r["trial_count"] for r in rows(res.visualization)} == dict(expected)
    assert res.facts["top_category_count"] == max(expected.values())


@pytest.mark.parametrize("name", FIXTURES)
async def test_recount_sponsor_landscape(make_ctx, name):
    res = await run_config(make_ctx, "sponsor", [load(name)])
    raws = dedup(_raw_studies(name))
    expected = Counter(norm(lead(r)) for r in raws if norm(lead(r)))
    trials_with: dict[str, set[str]] = defaultdict(set)
    for r in raws:
        if norm(lead(r)):
            trials_with[norm(lead(r))].add(nct(r))
    assert_top_n(res.facts["counts"], expected, 15, trials_with)
    assert res.facts["categories_total_derived"] == len(expected)
    assert res.facts.get("trials_without_value_derived", 0) == sum(1 for r in raws if not norm(lead(r)))


@pytest.mark.parametrize("name", FIXTURES)
async def test_recount_sponsor_class(make_ctx, name):
    res = await run_config(make_ctx, "sponsor_class", [load(name)])
    raws = dedup(_raw_studies(name))
    cls = [g(r, "protocolSection.sponsorCollaboratorsModule.leadSponsor.class") for r in raws]
    expected = Counter(SPONSOR_CLASS_LABELS.get(c, c) for c in cls if isinstance(c, str))
    assert res.facts["counts"] == dict(expected)


@pytest.mark.parametrize("name", FIXTURES)
async def test_recount_sponsor_by_phase(make_ctx, name):
    res = await run_config(make_ctx, "sponsor_phase", [load(name)])
    raws = dedup(_raw_studies(name))
    cells = Counter((norm(lead(r)), PHASE_LABELS.get(joined_phase(r), joined_phase(r))) for r in raws if norm(lead(r)))
    for row in rows(res.visualization):
        assert row["trial_count"] == cells[(norm(row["sponsor"]), row["phase"])]
    totals = Counter(norm(lead(r)) for r in raws if norm(lead(r)))
    for label, v in res.facts["sponsor_totals_derived"].items():
        assert totals[norm(label)] == v


@pytest.mark.parametrize("name", FIXTURES)
async def test_recount_intervention_landscape(make_ctx, name):
    res = await run_config(make_ctx, "intervention", [load(name)])
    raws = dedup(_raw_studies(name))
    trials_with: dict[str, set[str]] = defaultdict(set)
    for r in raws:
        for k in drug_keys(r):
            trials_with[k].add(nct(r))
    expected = Counter({k: len(v) for k, v in trials_with.items()})
    assert_top_n(res.facts["counts"], expected, 15, trials_with)
    assert res.facts.get("trials_without_value_derived", 0) == sum(1 for r in raws if not drug_keys(r))


@pytest.mark.parametrize("name", FIXTURES)
async def test_recount_intervention_type(make_ctx, name):
    res = await run_config(make_ctx, "intervention_type", [load(name)])
    expected: Counter[str] = Counter()
    for r in dedup(_raw_studies(name)):
        for t in {i.get("type") for i in interventions(r) if isinstance(i.get("type"), str)}:
            expected[INTERVENTION_TYPE_LABELS.get(t, t)] += 1
    assert res.facts["counts"] == dict(expected)


def _period(d: dt.date, gran: str) -> str:
    if gran == "year":
        return str(d.year)
    if gran == "quarter":
        return f"{d.year}-Q{(d.month - 1) // 3 + 1}"
    return f"{d.year}-{d.month:02d}"


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("gran", ["year", "quarter", "month"])
async def test_recount_trend_over_time(make_ctx, name, gran):
    res = await run_config(make_ctx, f"trend_{gran}", [load(name)])
    raws = dedup(_raw_studies(name))
    dated = [(r, start(r)) for r in raws]
    expected = Counter(_period(d[0], gran) for _, d in dated if d and (gran == "year" or d[1] != "year"))
    counts = res.facts["counts"]
    assert {k: v for k, v in counts.items() if v} == dict(expected)
    assert res.facts["trials_without_start_date_derived"] == sum(1 for _, d in dated if d is None)
    assert res.facts.get("trials_with_imprecise_start_derived", 0) == (
        0 if gran == "year" else sum(1 for _, d in dated if d and d[1] == "year")
    )
    assert res.facts["first_period"] == min(expected) and res.facts["last_period"] == max(expected)
    assert res.facts["peak_count"] == max(expected.values())


@pytest.mark.parametrize("name", FIXTURES)
async def test_recount_trend_by_phase(make_ctx, name):
    res = await run_config(make_ctx, "trend_phase", [load(name)])
    expected = Counter(
        (PHASE_LABELS.get(joined_phase(r), joined_phase(r)), str(start(r)[0].year))
        for r in dedup(_raw_studies(name))
        if start(r)
    )
    got = {(row["phase"], row["label"]): row["trial_count"] for row in rows(res.visualization) if row["trial_count"]}
    assert got == dict(expected)


@pytest.mark.parametrize("name", FIXTURES)
async def test_recount_phase_status_heatmap(make_ctx, name):
    res = await run_config(make_ctx, "phase_status", [load(name)])
    expected = Counter(
        (
            PHASE_LABELS.get(joined_phase(r), joined_phase(r)),
            STATUS_LABELS.get(status(r), status(r)) if status(r) else "Not reported",
        )
        for r in dedup(_raw_studies(name))
    )
    got = {(row["phase"], row["status"]): row["trial_count"] for row in rows(res.visualization)}
    assert {k: v for k, v in got.items() if v} == dict(expected)
    assert len(got) == res.facts["rows_derived"] * res.facts["columns_derived"]  # full matrix


_PHASE_KEY = {v: k for k, v in PHASE_LABELS.items()}


@pytest.mark.parametrize("name", FIXTURES)
async def test_recount_landscape_gap_heatmap(make_ctx, name):
    res = await run_config(make_ctx, "gap_intervention_phase", [load(name)])
    raws = dedup(_raw_studies(name))
    row_trials: dict[str, set[str]] = defaultdict(set)
    for r in raws:
        for k in drug_keys(r):
            row_trials[k].add(nct(r))
    totals = res.facts["row_totals_derived"]
    for label, v in totals.items():
        assert len(row_trials[norm(label)]) == v
    assert sorted(totals.values(), reverse=True) == sorted((len(v) for v in row_trials.values()), reverse=True)[:15]
    for row in rows(res.visualization):
        p = _PHASE_KEY[row["phase"]]
        want = sum(
            1
            for r in raws
            if norm(row["intervention"]) in drug_keys(r) and (p in phases(r) or (p == "NA" and not phases(r)))
        )
        assert row["trial_count"] == want, row


@pytest.mark.parametrize("name", FIXTURES)
async def test_recount_geographic_distribution(make_ctx, name):
    res = await run_config(make_ctx, "geo", [load(name)])
    raws = dedup(_raw_studies(name))
    expected = Counter(c for r in raws for c in countries(r))
    assert res.facts["counts"] == dict(expected)
    assert res.facts["trials_without_locations_derived"] == sum(1 for r in raws if not countries(r))
    sites = Counter(
        loc["country"]
        for r in raws
        for loc in g(r, "protocolSection.contactsLocationsModule.locations") or []
        if isinstance(loc.get("country"), str)
    )
    assert res.facts["site_counts"] == {k: sites[k] for k in res.facts["site_counts"]}


_EDGES = [0, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]


@pytest.mark.parametrize("name", FIXTURES)
async def test_recount_enrollment_histogram(make_ctx, name):
    res = await run_config(make_ctx, "enroll_hist", [load(name)])
    vals = [n for r in dedup(_raw_studies(name)) if (n := enrollment(r)) is not None]
    per_bin = Counter(max(i for i, lo in enumerate(_EDGES) if n >= lo) for n in vals)
    got = {row["bin_start"]: row["trial_count"] for row in rows(res.visualization)}
    assert {k: v for k, v in got.items() if v} == {_EDGES[i]: n for i, n in per_bin.items()}
    st = res.facts["stats_derived"]
    raws = dedup(_raw_studies(name))

    def summary(xs):
        return {"n": len(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs)} if xs else {"n": 0}

    est = [
        enrollment(r)
        for r in raws
        if enrollment(r) is not None and g(r, "protocolSection.designModule.enrollmentInfo.type") == "ESTIMATED"
    ]
    act = [
        enrollment(r)
        for r in raws
        if enrollment(r) is not None and g(r, "protocolSection.designModule.enrollmentInfo.type") != "ESTIMATED"
    ]
    assert len(est) + len(act) == len(vals)
    assert st == {"actual": {**summary(act), "total_participants_derived": sum(act)}, "estimated": summary(est)}


@pytest.mark.parametrize("name", FIXTURES)
async def test_recount_enrollment_scatter(make_ctx, name):
    res = await run_config(make_ctx, "enroll_scatter", [load(name)])
    raws = dedup(_raw_studies(name))
    ok = [r for r in raws if enrollment(r) is not None and start(r)]
    plotted = [r for r in ok if enrollment(r) > 0]
    assert res.facts["trials_plotted_derived"] == len(plotted)
    assert res.facts["trials_skipped_derived"] == len(raws) - len(ok)
    assert res.facts["trials_zero_enrollment_derived"] == len(ok) - len(plotted)
    assert {row["nct_id"]: row["enrollment"] for row in rows(res.visualization)} == {
        nct(r): enrollment(r) for r in plotted
    }


@pytest.mark.parametrize("name", FIXTURES)
async def test_recount_sponsor_activity(make_ctx, name):
    res = await run_config(make_ctx, "activity", [load(name)])
    raws = dedup(_raw_studies(name))
    dated = sorted(((start(r)[0], nct(r), r) for r in raws if start(r)), key=lambda t: (t[0], t[1]), reverse=True)
    shown = dated[:40]
    assert res.facts["trials_shown_derived"] == len(shown)
    assert res.facts["trials_without_start_date_derived"] == len(raws) - len(dated)
    assert {row["nct_id"] for row in rows(res.visualization)} == {t[1] for t in shown}
    assert res.facts["trials_by_status_derived"] == dict(
        Counter(STATUS_LABELS.get(status(r), status(r)) if status(r) else "Not reported" for *_, r in shown)
    )


@pytest.mark.parametrize("name", FIXTURES)
async def test_recount_development_progression(make_ctx, name):
    res = await run_config(make_ctx, "progression", [load(name)])
    raws = [r for r in dedup(_raw_studies(name)) if start(r)]
    for p in ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"]:
        items = [r for r in raws if p in phases(r)]
        fact = res.facts["phases"].get(PHASE_LABELS[p])
        if not items:
            assert fact is None
            continue
        assert fact["trial_count"] == len(items)
        assert fact["first_start"] == min(start(r)[0] for r in items).isoformat()
        assert fact["latest_start"] == max(start(r)[0] for r in items).isoformat()


@pytest.mark.parametrize("name", FIXTURES)
async def test_recount_design_mix(make_ctx, name):
    res = await run_config(make_ctx, "design", [load(name)])
    raws = dedup(_raw_studies(name))
    paths = {"allocation": "allocation", "masking": "maskingInfo.masking", "intervention_model": "interventionModel"}
    for dim, sub in paths.items():
        vals = [g(r, f"protocolSection.designModule.designInfo.{sub}") for r in raws]
        expected = Counter(v if isinstance(v, str) and v else "Not reported" for v in vals)
        assert res.facts[dim] == dict(expected), dim


async def test_recount_comparison_counts_each_cohort_separately(make_ctx):
    a, b = load("semaglutide", "Sema"), load("tirzepatide", "Tirz")
    res = await run_config(make_ctx, "status", [a, b])
    for c, name in ((a, "semaglutide"), (b, "tirzepatide")):
        expected = Counter(
            STATUS_LABELS.get(status(r), status(r)) if status(r) else "Not reported" for r in dedup(_raw_studies(name))
        )
        got = {k: v for k, v in res.facts["by_cohort"][c.label].items() if v}
        assert got == dict(expected)
    assert res.facts["trials_by_cohort_derived"] == {"Sema": 400, "Tirz": 300}


# ================================================================================================
# (2) Every number in facts is in the data, or documented as derived
# ================================================================================================


def _flat_numbers(d: Any, path: str = ""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _flat_numbers(v, f"{path}.{k}" if path else str(k))
    elif isinstance(d, list):
        for v in d:
            yield from _flat_numbers(v, f"{path}[]")
    elif isinstance(d, int | float) and not isinstance(d, bool):
        yield path, d


def _undocumented(name: str, res, cohort_labels: list[str]) -> list[tuple[str, Any]]:
    nums = data_numbers(res.visualization)
    out = []
    for path, v in _flat_numbers(res.facts):
        if v in nums:
            continue
        p = path
        for lbl in cohort_labels:  # comparison facts are nested under the cohort label
            p = p.replace(f"by_cohort.{lbl}.", "").replace(f"stats.{lbl}.", "stats.")
        documented = any(fnmatch(p, pat) for pat in DERIVED[FAMILY[name]])
        named_derived = any(seg.split("[")[0].endswith("_derived") for seg in path.split("."))
        if not (documented or named_derived):
            out.append((path, v))
    return out


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("config", list(CONFIGS))
async def test_facts_are_data_values_or_documented_derived(make_ctx, config, name):
    res = await run_config(make_ctx, config, [load(name)])
    assert _undocumented(config, res, [name]) == []


@pytest.mark.parametrize("config", COMPARISON_CONFIGS)
async def test_comparison_facts_are_data_values_or_documented_derived(make_ctx, config):
    cohorts = [load("semaglutide", "Sema"), load("obesity_since2020", "Obesity")]
    res = await run_config(make_ctx, config, cohorts)
    assert _undocumented(config, res, ["Sema", "Obesity"]) == []


def _row_sum(rs, key_field, value_field="trial_count") -> Counter:
    out: Counter = Counter()
    for r in rs:
        out[r[key_field]] += r[value_field]
    return out


@pytest.mark.parametrize("name", FIXTURES)
async def test_derived_facts_match_data(make_ctx, name):
    """The 'data' entries of DERIVED really are aggregates of visualization.data."""
    c = [load(name)]
    res = await run_config(make_ctx, "trend_phase", c)
    rs = rows(res.visualization)
    # multi-series scope (review item 2): per-series totals and peaks, distinct plotted trials
    assert "peak_count" not in res.facts and "trials_plotted_derived" not in res.facts
    assert res.facts["series_totals_derived"] == dict(_row_sum(rs, "phase"))
    assert res.facts["trials_plotted_distinct_derived"] == sum(r["trial_count"] for r in rs)  # one phase each
    for label, pk in res.facts["by_series_peak"].items():
        series = [r for r in rs if r["phase"] == label]
        assert pk["count"] == max(r["trial_count"] for r in series)
        assert pk["period"] in {r["label"] for r in series if r["trial_count"] == pk["count"]}
    single = await run_config(make_ctx, "trend_year", c)
    srs = rows(single.visualization)
    assert single.facts["trials_plotted_derived"] == sum(r["trial_count"] for r in srs)
    assert single.facts["peak_count"] == max(r["trial_count"] for r in srs)

    for cfg in ("phase_status", "gap_intervention_phase", "gap_sponsor_year"):
        res = await run_config(make_ctx, cfg, c)
        rs = rows(res.visualization)
        assert res.facts["cells_derived"] == len(rs) == res.facts["rows_derived"] * res.facts["columns_derived"]
        assert res.facts["gap_cells_derived"] == sum(r["is_gap"] for r in rs) == sum(r["trial_count"] == 0 for r in rs)
        if cfg == "phase_status":
            assert res.facts["trials_by_phase_derived"] == dict(_row_sum(rs, "phase"))
            assert res.facts["trials_by_status_derived"] == dict(_row_sum(rs, "status"))
            if res.facts.get("max_cell_count"):
                assert res.facts["max_cell_count"] == max(r["trial_count"] for r in rs)

    res = await run_config(make_ctx, "sponsor_phase", c)
    assert res.facts["sponsor_totals_derived"] == dict(_row_sum(rows(res.visualization), "sponsor"))

    res = await run_config(make_ctx, "activity", c)
    rs = rows(res.visualization)
    assert res.facts["trials_shown_derived"] == len(rs)
    assert res.facts["trials_by_phase_derived"] == dict(Counter(r["lane"] for r in rs))
    assert res.facts["trials_by_status_derived"] == dict(Counter(r["category"] for r in rs))

    for cfg in [k for k in CONFIGS if k.startswith("net_")]:
        res = await run_config(make_ctx, cfg, c)
        data = res.visualization.data
        f = res.facts
        assert f["nodes_derived"] == len(data.nodes) and f["edges_derived"] == len(data.edges)
        assert f["node_type_counts_derived"] == dict(Counter(n.model_extra["type"] for n in data.nodes))
        assert f["nodes_total_derived"] >= f["nodes_derived"] and f["edges_total_derived"] >= f["edges_derived"]
        if data.edges:
            assert min(e.model_extra["weight"] for e in data.edges) >= f["min_edge_weight_derived"]

    res = await run_config(make_ctx, "geo", c)
    if res.facts.get("top_country"):
        assert res.facts["top_country_trials"] == max(r["trial_count"] for r in rows(res.visualization))


# ================================================================================================
# (3) Citation invariant, all handlers x all fixtures
# ================================================================================================


def assert_full_citation_invariant(res, raws: dict[str, dict]) -> int:
    viz = res.visualization
    checked = assert_citations_verified(viz, raws)
    data = viz.data.nodes + viz.data.edges if isinstance(viz.data, GraphData) else viz.data
    assert set(res.full_refs) == {d.id for d in data}, "full_refs must be keyed by datum id"
    for d in data:
        refs = res.full_refs[d.id]
        # displayed citations are a (capped) prefix of the full evidence
        assert [c.field_path for c in d.citations] == [r.field_path for r in refs[: len(d.citations)]]
        if d.citations_total is not None:
            assert d.citations_total == len(refs) > len(d.citations)
        for r in refs:  # the uncapped evidence also verifies
            assert verify_citation(r.to_citation(), raws[r.nct_id]), f"{d.id}: {r.field_path}"
        has_note = "cite_note" in (d.model_extra or {})
        assert has_note == (not d.citations), f"{d.id}: cite_note iff no evidence"
    return checked


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("config", list(CONFIGS))
async def test_citation_invariant_all_handlers(make_ctx, config, name):
    c = load(name)
    res = await run_config(make_ctx, config, [c])
    checked = assert_full_citation_invariant(res, raw_by_id(c))
    data = res.visualization.data
    if data.nodes if isinstance(data, GraphData) else data:
        assert checked > 0


@pytest.mark.parametrize("config", COMPARISON_CONFIGS)
async def test_citation_invariant_comparison(make_ctx, config):
    cohorts = [load("melanoma_ph2_3", "Melanoma"), load("pembrolizumab_since2015", "Pembro")]
    res = await run_config(make_ctx, config, cohorts)
    assert_full_citation_invariant(res, raw_by_id(*cohorts))
    # cohort-tagged rows cite only studies of that cohort
    if res.visualization.encoding.series and res.visualization.encoding.series.field == "cohort":
        members = {c.label: {s.nct_id for s in c.studies} for c in cohorts}
        for d in res.visualization.data:
            for cit in d.citations:
                assert cit.nct_id in members[d.model_extra["cohort"]], d.id


# ================================================================================================
# (4) Empty cohort and single-study cohort: no crash, renderable output
# ================================================================================================


def assert_renderable(res) -> None:
    viz = res.visualization
    check_encoding(viz)
    # What the API actually sends (exclude_none) must parse back and still pass the encoding check.
    again = Visualization.model_validate_json(viz.model_dump_json(exclude_none=True))
    check_encoding(again)
    assert viz.title
    for n in res.notes:
        assert n.message
    data = viz.data.nodes + viz.data.edges if isinstance(viz.data, GraphData) else viz.data
    assert len({d.id for d in data}) == len(data)


@pytest.mark.parametrize("config", list(CONFIGS))
async def test_empty_cohort_every_handler(make_ctx, config):
    res = await run_config(make_ctx, config, [make_cohort("Empty", [])])
    assert_renderable(res)
    assert res.studies_analyzed == 0
    data = res.visualization.data
    if isinstance(data, GraphData):
        assert data.nodes == [] and data.edges == []
    else:
        assert all(
            (d.model_extra or {}).get("trial_count", 0) == 0 or d.model_extra.get("category") == "milestone"
            for d in data
        )
        assert all(not d.citations for d in data)


@pytest.mark.parametrize("config", COMPARISON_CONFIGS)
async def test_comparison_with_one_empty_cohort(make_ctx, config):
    a = load("tirzepatide", "Tirz")
    res = await run_config(make_ctx, config, [a, make_cohort("Empty", [])])
    assert_renderable(res)
    assert_full_citation_invariant(res, raw_by_id(a))


@pytest.mark.parametrize("config", list(CONFIGS))
async def test_no_cohorts_at_all(make_ctx, config):
    res = await run_config(make_ctx, config, [])
    assert_renderable(res)


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("config", list(CONFIGS))
async def test_single_study_cohort_every_handler(make_ctx, config, name):
    raw = _raw_studies(name)[0]
    c = make_cohort("One", [raw])
    res = await run_config(make_ctx, config, [c])
    assert_renderable(res)
    assert_full_citation_invariant(res, raw_by_id(c))
    assert res.studies_analyzed == 1
    data = res.visualization.data
    if not isinstance(data, GraphData):
        for d in data:
            v = (d.model_extra or {}).get("trial_count")
            if v is not None and (d.model_extra or {}).get("category") != "milestone":
                assert v in (0, 1) or FAMILY[config] == "trend", d  # one trial counts at most once per bucket


@pytest.mark.parametrize("config", list(CONFIGS))
async def test_single_study_with_almost_no_fields(make_ctx, config):
    """A record with only an NCT id (every optional field missing) still renders."""
    raw = {"protocolSection": {"identificationModule": {"nctId": "NCT00000001"}}}
    c = make_cohort("Bare", [raw])
    res = await run_config(make_ctx, config, [c])
    assert_renderable(res)
    assert_full_citation_invariant(res, raw_by_id(c))


# ================================================================================================
# (5) Geography: a trial counts at most once per country
# ================================================================================================


def _site(country: str, facility: str = "Site", city: str = "City") -> dict:
    return {"facility": facility, "city": city, "country": country}


def _synthetic(
    nct_id: str,
    *,
    locations=None,
    lead_name="Acme",
    interventions_=None,
    conditions_=None,
    collaborators=None,
    officials=None,
    arms=None,
    phases_=None,
    status_="RECRUITING",
) -> dict:
    ps: dict[str, Any] = {
        "identificationModule": {"nctId": nct_id, "briefTitle": f"Trial {nct_id}"},
        "statusModule": {"overallStatus": status_, "startDateStruct": {"date": "2021-03"}},
        "sponsorCollaboratorsModule": {
            "leadSponsor": {"name": lead_name, "class": "INDUSTRY"},
            **({"collaborators": collaborators} if collaborators else {}),
        },
        "designModule": {"phases": phases_ or ["PHASE2"], "enrollmentInfo": {"count": 100}},
        "conditionsModule": {"conditions": conditions_ or ["Obesity"]},
        "armsInterventionsModule": {
            "interventions": interventions_ or [{"type": "DRUG", "name": "Drug A"}],
            **({"armGroups": arms} if arms else {}),
        },
        "contactsLocationsModule": {
            "locations": locations or [],
            **({"overallOfficials": officials} if officials else {}),
        },
    }
    return {"protocolSection": ps}


@pytest.mark.parametrize("name", FIXTURES)
async def test_geography_once_per_country_per_trial(make_ctx, name):
    c = load(name)
    res = await run_config(make_ctx, "geo", [c])
    raws = raw_by_id(c)
    for d in res.visualization.data:
        refs = res.full_refs[d.id]
        value, sites_n = d.model_extra["trial_count"], d.model_extra["site_count"]
        # one citation per trial (its first site in the country): the count is trials, never sites
        first = [r.nct_id for r in refs]
        assert len(first) == len(set(first)) == value == len(d.nct_ids), d.model_extra["country"]
        assert sites_n >= value
        assert all(d.model_extra["country"] in countries(raws[t]) for t in first)


async def test_geography_many_sites_one_country_counts_once(make_ctx):
    raw = _synthetic(
        "NCT90000001",
        locations=[_site("United States", f"S{i}") for i in range(5)] + [_site("Germany"), _site("Germany", "B")],
    )
    res = await run_config(make_ctx, "geo", [make_cohort("Syn", [raw])])
    got = {r["country"]: (r["trial_count"], r["site_count"]) for r in rows(res.visualization)}
    assert got == {"United States": (1, 5), "Germany": (1, 2)}
    assert res.facts["counts"] == {"United States": 1, "Germany": 1}


async def test_geography_duplicate_trial_in_cohort_counts_once(make_ctx):
    raw = _synthetic("NCT90000002", locations=[_site("France")])
    res = await run_config(
        make_ctx, "geo", [make_cohort("A", [raw]), make_cohort("B", [raw])], scope=Scope.COHORT
    )  # merged scope: cohorts are de-duplicated
    assert {r["country"]: r["trial_count"] for r in rows(res.visualization)} == {"France": 1}


async def test_geography_whitespace_variant_country_counts_once(make_ctx):
    """Fixed: countries are grouped whitespace/case-insensitively (citations keep the raw value)."""
    raw = _synthetic("NCT90000003", locations=[_site("United States"), _site("United States ", "B")])
    res = await run_config(make_ctx, "geo", [make_cohort("Syn", [raw])])
    assert sum(r["trial_count"] for r in rows(res.visualization)) == 1
    (d,) = res.visualization.data
    assert d.model_extra["country"] == "United States" and d.model_extra["site_count"] == 2
    assert [c.excerpt for c in d.citations] == ["United States"]  # one citation per trial


async def test_geography_comparison_top_country_counts_distinct_trials(make_ctx):
    """Fixed: comparison ranks countries by distinct trials across cohorts (key renamed *_derived)."""
    a, b = load("semaglutide", "Sema"), load("obesity_since2020", "Obesity")
    res = await run_config(make_ctx, "geo", [a, b])
    top = res.facts["top_country"]
    distinct = {s.nct_id for c in (a, b) for s in c.studies if top in countries(s.raw)}
    assert res.facts["top_country_trials_distinct_derived"] == len(distinct)
    assert "top_country_trials" not in res.facts


async def test_geography_comparison_rows_are_per_cohort_distinct(make_ctx):
    a, b = load("semaglutide", "Sema"), load("obesity_since2020", "Obesity")
    res = await run_config(make_ctx, "geo", [a, b])
    members = {c.label: c for c in (a, b)}
    for d in res.visualization.data:
        x = d.model_extra
        want = sum(1 for s in members[x["cohort"]].studies if x["country"] in countries(s.raw))
        assert x["trial_count"] == want
        assert len({r.nct_id for r in res.full_refs[d.id]}) == want


# ================================================================================================
# (6) Networks: edges reference nodes; weight = distinct cited trials (independent recount)
# ================================================================================================

NET_KINDS = ["sponsor_drug", "drug_drug", "drug_condition", "site_investigator", "sponsor_collaborator"]
DRUG_TYPES = {"DRUG", "BIOLOGICAL"}


def _nid(kind: str, name: str) -> str:
    return f"{kind}:{norm(name).replace(' ', '-')}"


def expected_edges(kind: str, raws: list[dict]) -> dict[tuple[str, str], set[str]]:
    """Independent edge -> distinct trials map for each network kind."""
    out: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in raws:
        t = nct(r)
        if kind == "sponsor_drug" and norm(lead(r)):
            for d in drug_keys(r, DRUG_TYPES):
                out[(_nid("sponsor", lead(r)), _nid("drug", d))].add(t)
        elif kind == "drug_condition":
            for d in drug_keys(r, DRUG_TYPES):
                for cnd in conditions(r):
                    out[(_nid("drug", d), _nid("condition", cnd))].add(t)
        elif kind == "sponsor_collaborator" and norm(lead(r)):
            for c in g(r, "protocolSection.sponsorCollaboratorsModule.collaborators") or []:
                if isinstance(c, dict) and norm(c.get("name")):
                    a, b = _nid("org", lead(r)), _nid("org", c["name"])
                    if a != b:
                        out[(a, b)].add(t)
        elif kind == "drug_drug":
            for arm in g(r, "protocolSection.armsInterventionsModule.armGroups") or []:
                ds = set()
                for entry in (arm.get("interventionNames") or []) if isinstance(arm, dict) else []:
                    if not isinstance(entry, str) or ":" not in entry:
                        continue
                    typ, _, nm = entry.partition(":")
                    nm = nm.strip()
                    if nm and typ.strip().upper() in DRUG_TYPES and not _PLACEBO.search(nm):
                        ds.add(_nid("drug", canonical_drug(nm) or nm))  # raw arm names: canonicalize once
                ds_sorted = sorted(ds)
                for i in range(len(ds_sorted)):
                    for j in range(i + 1, len(ds_sorted)):
                        out[(ds_sorted[i], ds_sorted[j])].add(t)
        elif kind == "site_investigator":
            loc_mod = g(r, "protocolSection.contactsLocationsModule") or {}
            offs = {
                _nid("investigator", o["name"])
                for o in loc_mod.get("overallOfficials") or []
                if isinstance(o, dict) and norm(o.get("name"))
            }
            sites = set()
            n_sites = 0
            for loc in loc_mod.get("locations") or []:
                fac = loc.get("facility") if isinstance(loc, dict) else None
                if isinstance(fac, str) and norm(fac):
                    n_sites += 1
                    if n_sites > 50:  # handler links at most 50 sites per trial (with a TRUNCATED note)
                        break
                    city = loc.get("city") if isinstance(loc.get("city"), str) else ""
                    sites.add(_nid("site", f"{fac} {city}"))
            for o in offs:
                for s in sites:
                    out[(o, s)].add(t)
    return out


def assert_graph_consistent(res) -> None:
    data = res.visualization.data
    assert isinstance(data, GraphData)
    ids = {n.id for n in data.nodes}
    assert len(ids) == len(data.nodes)
    for e in data.edges:
        x = e.model_extra
        assert x["source"] in ids and x["target"] in ids, e.id
        assert x["source"] != x["target"]
        cited = {r.nct_id for r in res.full_refs[e.id]}
        assert x["weight"] == len(cited), e.id
    for n in data.nodes:
        assert n.model_extra["size"] == len({r.nct_id for r in res.full_refs[n.id]}), n.id


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.parametrize("kind", NET_KINDS)
async def test_network_edges_and_weights(make_ctx, kind, name):
    res = await run_config(make_ctx, f"net_{kind}", [load(name)])
    assert_graph_consistent(res)
    expected = expected_edges(kind, dedup(_raw_studies(name)))
    for e in res.visualization.data.edges:
        x = e.model_extra
        key = (x["source"], x["target"])
        assert key in expected, key
        assert x["weight"] == len(expected[key]), key
        assert {r.nct_id for r in res.full_refs[e.id]} == expected[key]
    assert res.facts["edges_total_derived"] == len(expected)


async def test_network_merged_cohorts_dedupe_shared_trials(make_ctx):
    a, b = load("semaglutide", "Sema"), load("tirzepatide", "Tirz")  # 55 shared trials
    res = await run_config(make_ctx, "net_sponsor_drug", [a, b])
    assert_graph_consistent(res)
    expected = expected_edges(
        "sponsor_drug", dedup(list(_raw_studies("semaglutide")) + list(_raw_studies("tirzepatide")))
    )
    for e in res.visualization.data.edges:
        assert e.model_extra["weight"] == len(expected[(e.model_extra["source"], e.model_extra["target"])])


async def test_network_synthetic_weights_and_self_loops(make_ctx):
    raws = [
        _synthetic(
            "NCT91000001",
            lead_name="Acme",
            interventions_=[
                {"type": "DRUG", "name": "Drug A"},
                {"type": "DRUG", "name": "drug a®"},
                {"type": "DRUG", "name": "Placebo"},
            ],
            collaborators=[{"name": "Acme"}, {"name": "Beta Univ"}],
        ),
        _synthetic(
            "NCT91000002",
            lead_name="ACME",
            interventions_=[{"type": "DRUG", "name": "Drug A"}],
            collaborators=[{"name": "Beta Univ"}],
        ),
        _synthetic("NCT91000003", lead_name="Gamma", interventions_=[{"type": "BIOLOGICAL", "name": "Mab B"}]),
    ]
    c = make_cohort("Syn", raws)
    res = await run_config(make_ctx, "net_sponsor_drug", [c])
    assert_graph_consistent(res)
    w = {
        (e.model_extra["source"], e.model_extra["target"]): e.model_extra["weight"]
        for e in res.visualization.data.edges
    }
    assert w == {("sponsor:acme", "drug:drug-a"): 2, ("sponsor:gamma", "drug:mab-b"): 1}
    assert not any("placebo" in n.id for n in res.visualization.data.nodes)

    res = await run_config(make_ctx, "net_sponsor_collaborator", [c])
    assert_graph_consistent(res)
    w = {
        (e.model_extra["source"], e.model_extra["target"]): e.model_extra["weight"]
        for e in res.visualization.data.edges
    }
    assert w == {("org:acme", "org:beta-univ"): 2}  # Acme listing itself is not a self-loop


def test_network_prune_keeps_edges_between_kept_nodes():
    from ctviz.compute.network import CooccurrenceGraph

    g_ = CooccurrenceGraph(directed=False)
    for i in range(60):
        for t in range(1 + i):  # distinct weights 1..60
            g_.add_node(f"a{i}", f"A{i}", "a", f"T{i}-{t}", "ev")
            g_.add_node("hub", "Hub", "b", f"T{i}-{t}", "ev")
            g_.add_edge(f"a{i}", "hub", "x", f"T{i}-{t}", ["ev"])
    g_.add_edge("hub", "hub", "x", "T0", ["ev"])  # self-loop ignored
    pruned = g_.prune(max_nodes=10, max_edges=5)
    keep = {n.key for n in pruned.nodes}
    assert 0 < len(keep) <= 10 and "hub" in keep and 0 < len(pruned.edges) <= 5
    assert all(e.source in keep and e.target in keep for e in pruned.edges)
    assert all(e.source != e.target for e in pruned.edges)
    assert all(e.weight >= pruned.min_edge_weight for e in pruned.edges)
    assert pruned.total_nodes == 61 and pruned.total_edges == 60


async def test_network_many_equal_weight_edges_not_emptied(make_ctx):
    """Fixed: an over-full lowest-weight tier is cut to max_edges deterministically, with a note."""
    from ctviz.handlers.cohort import MAX_EDGES_SHOWN

    raws = [
        _synthetic(
            f"NCT93{i:06d}",
            lead_name=f"Sponsor {i}",
            interventions_=[{"type": "DRUG", "name": f"Drug {j}"} for j in range(20)],
        )
        for i in range(20)
    ]  # 20 sponsors x 20 drugs = 400 weight-1 edges among 40 nodes
    res = await run_config(make_ctx, "net_sponsor_drug", [make_cohort("Syn", raws)])
    assert_graph_consistent(res)
    assert res.visualization.data.edges, "graph pruned to nothing"
    assert len(res.visualization.data.edges) == MAX_EDGES_SHOWN
    assert any(n.code.value == "TRUNCATED" and "share weight" in n.message for n in res.notes)


def test_cooccurrence_undirected_edges_merge_both_orders():
    from ctviz.compute.network import CooccurrenceGraph

    g_ = CooccurrenceGraph(directed=False)
    g_.add_edge("b", "a", "x", "T1", ["e1"])
    g_.add_edge("a", "b", "x", "T2", ["e2", None])
    g_.add_edge("a", "b", "x", "T2", ["e3"])
    (edge,) = g_.edges.values()
    assert (edge.source, edge.target, edge.weight) == ("a", "b", 2)
    assert edge.evidence == ["e1", "e2", "e3"]
    d = CooccurrenceGraph(directed=True)
    d.add_edge("b", "a", "x", "T1")
    d.add_edge("a", "b", "x", "T1")
    assert len(d.edges) == 2


# ================================================================================================
# compute/cohort.py edge cases
# ================================================================================================


def test_compute_partial_dates_and_buckets():
    from ctviz.compute.cohort import (
        bucket_label,
        bucket_range,
        bucket_start,
        enrollment_bin,
        enrollment_bin_label,
        median,
        parse_partial_date,
        supports_granularity,
    )

    assert parse_partial_date("2021-02-30") is None and parse_partial_date("2021-13") is None
    assert parse_partial_date(None) is None and parse_partial_date(2021) is None
    assert parse_partial_date(" 2021-07 ").precision == "month"
    y = parse_partial_date("2020")
    assert supports_granularity(y, "year") and not supports_granularity(y, "quarter")
    q = bucket_start(dt.date(2020, 12, 31), "quarter")
    assert q == dt.date(2020, 10, 1) and bucket_label(q, "quarter") == "2020-Q4"
    rng = bucket_range(dt.date(2020, 11, 5), dt.date(2021, 2, 1), "month")
    assert [bucket_label(d, "month") for d in rng] == ["2020-11", "2020-12", "2021-01", "2021-02"]
    assert [enrollment_bin(n) for n in (0, 24, 25, 9999, 10000, 10**7)] == [0, 0, 1, 8, 9, 9]
    assert enrollment_bin_label(9) == "10,000+" and enrollment_bin_label(0) == "0–24"
    assert median([]) is None and median([3, 1, 2]) == 2.0 and median([1, 2, 3, 4]) == 2.5


def test_compute_names_placebo_and_arm_split():
    from ctviz.compute.cohort import LabelPicker, is_placebo, normalize_name, split_arm_intervention, top_keys

    assert normalize_name("Ozempic®  (Semaglutide)") == "ozempic semaglutide"
    assert normalize_name(None) == ""
    assert is_placebo("Matching Placebo") and is_placebo("Sham procedure") and not is_placebo("Placebolin")
    assert split_arm_intervention("Drug: Nivolumab ") == ("Drug", "Nivolumab")
    assert split_arm_intervention("Drug:") is None and split_arm_intervention("Nivolumab") is None
    lp = LabelPicker()
    for raw in ("Pfizer", "PFIZER", "Pfizer"):
        lp.add("pfizer", raw)
    assert lp.label("pfizer") == "Pfizer" and lp.label("missing") == "missing"
    assert top_keys({"b": 2, "a": 2, "c": 5}, 2) == ["c", "a"]


# ================================================================================================
# Builders and the encoding contract
# ================================================================================================


def _rows(*dicts) -> list[Datum]:
    ds = Dataset()
    for d in dicts:
        ds.emit(d, [])
    return ds.rows


Q = Channel(field="n", type="quantitative", title="N")
CAT = Channel(field="k", type="nominal", title="K")


def test_builders_produce_valid_specs():
    rs = _rows({"k": "a", "n": 1}, {"k": "b", "n": 0})
    bar = bar_chart(rs, title="t", x=CAT.model_copy(update={"sort": ordered(["a", "b"])}), y=Q)
    assert bar.meta.sort.values == ["a", "b"] and bar.meta.layout.orientation == "vertical"
    pie = pie_chart(rs, title="t", label=CAT, value=Q)
    assert pie.meta.layout.donut is True
    geo = geo_map(rs, title="t", region=CAT, value=Q)
    assert geo.meta.geo.model_dump(exclude_none=True) == {"level": "country", "value_field": "n", "region_field": "k"}
    hm = heatmap(
        _rows({"k": "a", "j": "x", "n": 0}), title="t", x=CAT, y=Channel(field="j", type="nominal", title="J"), value=Q
    )
    assert hm.meta.highlight_zero is True
    hist = histogram(
        _rows({"s": 0, "e": 25, "n": 3}),
        title="t",
        x=Channel(field="s", type="quantitative", title="S"),
        x2=Channel(field="e", type="quantitative", title="E"),
        y=Q,
        summary={"n": 3},
    )
    assert hist.meta.summary == {"n": 3}
    sc = scatter_plot(_rows({"x": 1, "n": 2}), title="t", x=Channel(field="x", type="quantitative", title="X"), y=Q)
    assert sc.type is VizType.SCATTER_PLOT
    ts = time_series(
        _rows({"p": "2020-01-01", "n": 1}),
        title="t",
        x=Channel(field="p", type="temporal", title="P"),
        y=Q,
        granularity="year",
    )
    assert ts.encoding.x.scale.type == "time" and ts.meta.time_granularity == "year"
    tl = timeline(_rows({"task": "T", "start": "2020-01-01", "end": None, "category": "c"}), title="t")
    assert tl.meta.time_axis.model_dump(exclude_none=True) == {"kind": "calendar"} and tl.meta.time_granularity == "day"
    for v in (bar, pie, geo, hm, hist, sc, ts, tl):
        check_encoding(Visualization.model_validate_json(v.model_dump_json(exclude_none=True)))


def test_builders_reject_bad_channels():
    with pytest.raises(ValueError):
        time_series([], title="t", x=CAT, y=Q, granularity="year")
    with pytest.raises(ValueError):
        heatmap([], title="t", x=CAT, y=CAT, value=CAT)


def test_encoding_rejects_null_encoded_field():
    with pytest.raises(EncodingError):
        bar_chart(_rows({"k": None, "n": 1}), title="t", x=CAT, y=Q)


def test_network_builder_rejects_dangling_edge():
    gds = GraphDataset()
    gds.add_node("a", {"label": "A", "type": "x", "size": 1, "degree": 1}, [])
    gds.add_edge({"source": "a", "target": "missing", "weight": 1, "type": "t"}, [])
    with pytest.raises(EncodingError):
        network_graph(gds.to_graph(), title="t", node_types=[], edge_semantics="s")


def test_graph_dataset_merges_repeated_node_ids():
    raw = _synthetic("NCT92000001")
    s = Study(raw)
    gds = GraphDataset()
    gds.add_node("n1", {"label": "A", "type": "x"}, [])
    assert gds.uncited == ["n1"]
    gds.add_node("n1", {"label": "A2", "type": "x"}, [s.ref("lead_sponsor")])
    assert len(gds.nodes.rows) == 1 and gds.uncited == []
    node = gds.nodes.rows[0]
    assert node.model_extra["label"] == "A2" and "cite_note" not in node.model_extra
    assert len(node.citations) == 1
