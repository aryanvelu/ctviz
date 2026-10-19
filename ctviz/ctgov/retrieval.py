"""Retrieval recipes over :class:`~ctviz.ctgov.client.CTGovClient`.

* :func:`fetch_cohort` — page 1 with ``countTotal``; small cohorts are done in one call, cohorts up to
  ``max_studies`` (25,000 by default) are read completely, split into StartDate-year partitions (plus a
  ``MISSING`` partition) fetched concurrently; larger cohorts get a sample stratified by start year
  (each year range contributes in proportion to its size) with ``sampled=True``.
* :func:`fetch_trials` — full records, study store first, then the API; written back to the store.
* :func:`facet_counts` — exact server-side counts per bucket (parallel) with a few sample members.
* :func:`relax_until_nonzero` — deterministic relax ladder driven by cheap count probes.

Every raw record returned is also registered in a request-scoped "seen" map so the Verification
agent can re-check citations against exactly the JSON the handlers used.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import math
from collections import Counter
from typing import Any

from ctviz.api.schemas.plan import CohortFilter
from ctviz.core.background import spawn
from ctviz.core.provenance import Study
from ctviz.ctgov.fields import FieldSet, P
from ctviz.ctgov.paths import path_get
from ctviz.ctgov.requests import SearchSpec
from ctviz.store.cohort_snapshots import load_snapshot, save_snapshot
from ctviz.store.study_store import load_studies, save_studies_background
from ctviz.viz.base import CohortData

log = logging.getLogger(__name__)

PAGE_SIZE = 1000
SAMPLE_SORT = "LastUpdatePostDate:desc"
MISSING_START = "AREA[StartDate]MISSING"
MAX_PARTITIONS = 8
PARTITION_TARGET = 700  # aim below the 1000 page size so most partitions are a single call

DEFAULT_PARTITION_BUDGET = 40  # partition/page calls per request across all cohorts

_seen: contextvars.ContextVar[dict[str, dict[str, Any]] | None] = contextvars.ContextVar(
    "ctviz_seen_studies", default=None
)
_budget: contextvars.ContextVar[list[int] | None] = contextvars.ContextVar("ctviz_partition_budget", default=None)


# -- request-scoped registry of raw records + partition call budget -----------------------------
def begin_seen(partition_budget: int = DEFAULT_PARTITION_BUDGET) -> dict[str, dict[str, Any]]:
    """Start a fresh registry (and partition-call budget) for the current request.

    Call before spawning child tasks so they share the same mutable objects.
    """
    reg: dict[str, dict[str, Any]] = {}
    _seen.set(reg)
    _budget.set([partition_budget])
    return reg


def _take_budget(n: int) -> int:
    """Reserve up to ``n`` partition calls; returns how many were granted (unbounded outside a request)."""
    b = _budget.get()
    if b is None:
        return n
    granted = max(0, min(n, b[0]))
    b[0] -= granted
    return granted


def budget_left() -> int | None:
    """Calls left in this request's budget (None outside a request: unbounded)."""
    b = _budget.get()
    return None if b is None else b[0]


def seen_studies() -> dict[str, dict[str, Any]]:
    """Raw records retrieved so far in this request, keyed by NCT id (full records win)."""
    reg = _seen.get()
    return reg if reg is not None else {}


def _register(raws: list[dict[str, Any]], full: bool) -> None:
    reg = _seen.get()
    if reg is None:
        return
    for raw in raws:
        nct = path_get(raw, P["nct_id"])
        if isinstance(nct, str) and (full or nct not in reg):
            reg[nct] = raw


def _to_studies(raws: list[dict[str, Any]]) -> list[Study]:
    out: list[Study] = []
    seen: set[str] = set()
    for raw in raws:
        try:
            s = Study(raw)
        except ValueError:
            continue
        if s.nct_id not in seen:
            seen.add(s.nct_id)
            out.append(s)
    return out


def _store_level(field_set: FieldSet) -> str:
    return "full" if field_set is FieldSet.FULL else "projected"


# -- cohorts --------------------------------------------------------------------------------------
def _start_year(raw: dict[str, Any]) -> int | None:
    date = path_get(raw, P["start_date"])
    if isinstance(date, str) and len(date) >= 4 and date[:4].isdigit():
        return int(date[:4])
    return None


def year_partitions(
    years: list[int], total: int, *, target: int = PARTITION_TARGET, max_parts: int = MAX_PARTITIONS
) -> list[tuple[int | None, int | None]]:
    """Split the StartDate axis into contiguous year ranges with roughly equal expected counts.

    ``years`` is a sample of start years (page 1). Returns ``(lo, hi)`` inclusive ranges where the first
    ``lo`` and last ``hi`` are ``None`` (open ended) so no study can fall between partitions.
    """
    if not years:
        return [(None, None)]
    parts = max(2, min(max_parts, math.ceil(total / target)))
    counts = Counter(years)
    ordered = sorted(counts)
    per_part = len(years) / parts
    cuts: list[int] = []  # last year included in each partition except the final one
    acc = 0
    for y in ordered[:-1]:
        acc += counts[y]
        if acc >= per_part * (len(cuts) + 1) and len(cuts) < parts - 1:
            cuts.append(y)
    ranges: list[tuple[int | None, int | None]] = []
    lo: int | None = None
    for c in cuts:
        ranges.append((lo, c))
        lo = c + 1
    ranges.append((lo, None))
    return ranges


def _range_clause(lo: int | None, hi: int | None) -> str:
    a = f"{lo}-01-01" if lo is not None else "MIN"
    b = f"{hi}-12-31" if hi is not None else "MAX"
    return f"AREA[StartDate]RANGE[{a},{b}]"


async def _page_all(client: Any, spec: SearchSpec, limit: int, refresh: bool) -> list[dict[str, Any]]:
    """Sequentially page one search (bounded by ``limit`` records and the request's call budget).

    The first page was already reserved by the caller; follow-up pages each take one budget unit.
    """
    page = await client.search_page(spec, page_size=min(PAGE_SIZE, limit), refresh=refresh)
    raws = list(page.studies)
    token = page.next_page_token
    max_pages = math.ceil(limit / PAGE_SIZE) + 1  # hard bound on the loop
    pages = 1
    while token and len(raws) < limit and pages < max_pages and _take_budget(1):
        page = await client.search_page(
            spec, page_size=min(PAGE_SIZE, limit - len(raws)), page_token=token, count_total=False, refresh=refresh
        )
        raws.extend(page.studies)
        token = page.next_page_token
        pages += 1
    return raws[:limit]


async def fetch_cohort(
    client: Any,
    cf: CohortFilter,
    *,
    field_set: FieldSet,
    max_studies: int,
    refresh: bool = False,
    registry_counts_if_large: bool = False,
) -> CohortData:
    """Fetch the trials matching one cohort filter (see module docstring for the strategy).

    With ``registry_counts_if_large`` and more matches than ``max_studies``, only the first page is kept
    (``counting="registry_counts"``): the handler then asks ClinicalTrials.gov for an exact ``totalCount``
    per bucket instead of counting a sample.
    """
    spec = SearchSpec.from_cohort(cf, field_set).without(sort=SAMPLE_SORT)
    first = await client.search_page(spec, page_size=min(PAGE_SIZE, max_studies), refresh=refresh)
    total = int(first.total_count if first.total_count is not None else len(first.studies))
    raws = list(first.studies)
    db = getattr(client, "db", None)
    # Snapshots serve landscape reads only: every stored record carries the landscape fields, but not
    # necessarily the site / outcome / eligibility fields a richer projection asks for.
    if (
        total > len(raws)
        and not refresh
        and field_set is FieldSet.LANDSCAPE
        and (snap := await _from_snapshot(db, spec, cf, total)) is not None
    ):
        return snap
    sampled = False
    method: str | None = None
    if registry_counts_if_large and total > max_studies:
        studies = _to_studies(raws)
        _register(raws, full=field_set is FieldSet.FULL)
        save_studies_background(getattr(client, "db", None), raws, _store_level(field_set))
        return CohortData(
            label=cf.label,
            spec=spec,
            studies=studies,
            total_count=total,
            sampled=True,
            sampling_method=(
                f"{total} matching trials exceed the {max_studies}-record retrieval "
                "limit; each bucket is an exact ClinicalTrials.gov count instead"
            ),
            search_url=spec.ui_url(),
            counting="registry_counts",
        )
    if first.next_page_token and len(raws) < min(total, max_studies):
        if total <= max_studies:
            years = [y for y in (_start_year(r) for r in raws) if y is not None]
            # Partitions + the MISSING partition must fit the per-request call budget (shared by cohorts).
            wanted = len(year_partitions(years, total)) + 1
            granted = _take_budget(wanted)
            ranges = year_partitions(years, total, max_parts=max(1, granted - 1)) if granted >= 2 else []
            clauses = [_range_clause(lo, hi) for lo, hi in ranges] + ([MISSING_START] if ranges else [])
            results = await asyncio.gather(
                *(_page_all(client, spec.with_clause(c), max_studies, refresh) for c in clauses), return_exceptions=True
            )
            failed = sum(1 for r in results if isinstance(r, BaseException))
            for r in results:
                if isinstance(r, BaseException) and not isinstance(r, Exception):
                    raise r  # cancellation / interpreter exit: never swallow
            merged: dict[str, dict[str, Any]] = {}
            for part in [raws, *(r for r in results if not isinstance(r, BaseException))]:
                for r in part:
                    nct = path_get(r, P["nct_id"])
                    if isinstance(nct, str):
                        merged.setdefault(nct, r)
            raws = list(merged.values())[:max_studies]
            if len(raws) < total:
                sampled = True
                why = (
                    f"{failed} of {len(clauses)} StartDate partitions failed"
                    if failed
                    else "per-request call budget reached"
                    if not clauses or granted < wanted
                    else "partitions returned fewer trials than totalCount"
                )
                method = f"partial retrieval: {len(raws)} of {total} matching trials ({why})"
                log.info("cohort %s incomplete: %s", cf.label, method)
        else:
            token = first.next_page_token
            max_pages = math.ceil(max_studies / PAGE_SIZE) + 1
            pages = 1
            while token and len(raws) < max_studies and pages < max_pages:
                page = await client.search_page(
                    spec,
                    page_size=min(PAGE_SIZE, max_studies - len(raws)),
                    page_token=token,
                    count_total=False,
                    refresh=refresh,
                )
                raws.extend(page.studies)
                token = page.next_page_token
                pages += 1
            raws = raws[:max_studies]
    if total > max_studies and not sampled:
        raws = await _stratified_sample(client, spec, raws, total, max_studies, refresh)
        sampled = True
        method = (
            f"{len(raws):,} of {total:,} matching trials, stratified by start year: each start-year range "
            "contributes in proportion to its size (within a range, the most recently updated trials)"
        )
    studies = _to_studies(raws)
    _register(raws, full=field_set is FieldSet.FULL)
    save_studies_background(db, raws, _store_level(field_set))
    if not sampled and total > PAGE_SIZE and len(studies) >= total:  # complete read: remember the members
        ttl = getattr(getattr(client, "settings", None), "ttl_cohort_s", 24 * 3600)
        spawn(
            save_snapshot(db, spec, [s.nct_id for s in studies], total, ttl), name="cohort-snapshot", low_priority=True
        )
    return CohortData(
        label=cf.label,
        spec=spec,
        studies=studies,
        total_count=total,
        sampled=sampled,
        sampling_method=method,
        search_url=spec.ui_url(),
    )


async def _from_snapshot(db: Any, spec: SearchSpec, cf: CohortFilter, total: int) -> CohortData | None:
    """Every member of the cohort from the study store, if a snapshot exists and the registry count still
    matches (records of any projection carry the landscape fields)."""
    snap = await load_snapshot(db, spec)
    if snap is None or snap.get("total") != total:
        return None
    ids = list(snap["nct_ids"])
    found = await load_studies(db, ids)
    if len(found) < len(ids):
        return None
    raws = [found[n] for n in ids]
    _register(raws, full=False)
    return CohortData(
        label=cf.label,
        spec=spec,
        studies=_to_studies(raws),
        total_count=total,
        search_url=spec.ui_url(),
        snapshot_at=float(snap.get("taken_at") or 0.0),
    )


def _strata(this_year: int) -> list[str]:
    """Start-year ranges for a fair sample: wide for old years, single years recently, plus MISSING."""
    spans: list[tuple[int | None, int | None]] = [(None, 1999), (2000, 2004), (2005, 2009), (2010, 2013), (2014, 2016)]
    spans += [(y, y) for y in range(2017, this_year + 1)] + [(this_year + 1, None)]
    return [_range_clause(lo, hi) for lo, hi in spans] + [MISSING_START]


def _quotas(sizes: list[int], n: int) -> list[int]:
    """Largest-remainder apportionment of ``n`` sample slots across strata of ``sizes``."""
    total = sum(sizes) or 1
    exact = [n * s / total for s in sizes]
    out = [min(s, math.floor(e)) for s, e in zip(sizes, exact, strict=True)]
    for i in sorted(range(len(sizes)), key=lambda i: exact[i] - math.floor(exact[i]), reverse=True):
        if sum(out) >= n:
            break
        if out[i] < sizes[i]:
            out[i] += 1
    return out


async def _stratified_sample(
    client: Any, spec: SearchSpec, first: list[dict[str, Any]], total: int, n: int, refresh: bool
) -> list[dict[str, Any]]:
    """``n`` trials spread over start-year strata in proportion to each stratum's totalCount.

    Page 1 of every stratum returns its size and its first records together; only strata whose quota
    exceeds one page are paged further. Falls back to ``first`` if the strata cannot be fetched.
    """
    import datetime as dt

    clauses = _strata(dt.date.today().year)
    if _take_budget(len(clauses)) < len(clauses):
        return first[:n]
    pages = await asyncio.gather(
        *(client.search_page(spec.with_clause(c), page_size=PAGE_SIZE, refresh=refresh) for c in clauses),
        return_exceptions=True,
    )
    if any(isinstance(p, BaseException) for p in pages):
        return first[:n]
    sizes = [int(p.total_count or 0) for p in pages]
    quotas = _quotas(sizes, n)
    more = [(i, c) for i, c in enumerate(clauses) if quotas[i] > len(pages[i].studies)]
    extra = await asyncio.gather(
        *(_page_all(client, spec.with_clause(c), quotas[i], refresh) for i, c in more), return_exceptions=True
    )
    rows: list[list[dict[str, Any]]] = [list(p.studies) for p in pages]
    for (i, _), got in zip(more, extra, strict=True):
        if not isinstance(got, BaseException):
            rows[i] = got
    merged: dict[str, dict[str, Any]] = {}
    for i, part in enumerate(rows):
        for r in part[: quotas[i]]:
            nct = path_get(r, P["nct_id"])
            if isinstance(nct, str):
                merged.setdefault(nct, r)
    return list(merged.values())


async def search_candidates(
    client: Any, cf: CohortFilter, *, limit: int = 10, recruiting_first: bool = False, refresh: bool = False
) -> tuple[list[Study], int, SearchSpec]:
    """A handful of matching trials (most recently updated), for "which trial?" clarifications.

    With ``recruiting_first`` and no status filter, recruiting trials are tried first, then any status.
    Returns ``(studies, totalCount, spec_used)``.
    """
    base = SearchSpec.from_cohort(cf, FieldSet.LANDSCAPE).without(sort=SAMPLE_SORT)
    specs = [base]
    if recruiting_first and not cf.statuses:
        specs.insert(0, base.without(statuses=("RECRUITING", "NOT_YET_RECRUITING")))
    page = None
    spec = base
    for spec in specs:
        page = await client.search_page(spec, page_size=limit, refresh=refresh)
        if page.studies:
            break
    raws = list(page.studies) if page else []
    _register(raws, full=False)
    save_studies_background(getattr(client, "db", None), raws, "projected")
    return _to_studies(raws), int((page.total_count if page else 0) or 0), spec


# -- single trials --------------------------------------------------------------------------------
async def fetch_trials(client: Any, nct_ids: list[str], *, refresh: bool = False) -> list[Study]:
    """Full records for ``nct_ids`` in request order: study store first, then ClinicalTrials.gov."""
    ids = list(dict.fromkeys(n.strip().upper() for n in nct_ids if n))
    db = getattr(client, "db", None)
    found: dict[str, dict[str, Any]] = {} if refresh else await load_studies(db, ids, "full")
    missing = [n for n in ids if n not in found]
    if missing:
        fetched = await client.get_studies(missing, refresh=refresh)
        for raw in fetched:
            nct = path_get(raw, P["nct_id"])
            if isinstance(nct, str):
                found[nct] = raw
        save_studies_background(db, fetched, "full")
    raws = [found[n] for n in ids if n in found]
    _register(raws, full=True)
    return _to_studies(raws)


# -- facets ---------------------------------------------------------------------------------------
async def facet_counts(
    client: Any, spec: SearchSpec, buckets: dict[str, str], *, sample: int = 10
) -> dict[str, tuple[int, list[Study], str]]:
    """Exact count per bucket (``bucket -> Essie clause``), run in parallel.

    Returns ``bucket -> (totalCount, up to `sample` member studies, ClinicalTrials.gov search URL)``.
    """
    names = list(buckets)
    _take_budget(len(names))  # charged (callers size their buckets with budget_left())
    specs = [spec.with_clause(buckets[b]) for b in names]
    pages = await asyncio.gather(*(client.search_page(s, page_size=max(1, sample)) for s in specs))
    out: dict[str, tuple[int, list[Study], str]] = {}
    for name, s, page in zip(names, specs, pages, strict=True):
        raws = list(page.studies)[:sample] if sample > 0 else []
        _register(raws, full=s.field_set is FieldSet.FULL)
        out[name] = (int(page.total_count or 0), _to_studies(raws), s.ui_url())
    return out


# -- relax ladder ---------------------------------------------------------------------------------
def _relax_steps(cf: CohortFilter) -> list[tuple[str, CohortFilter]]:
    """Cumulative relaxations in fixed order; steps that change nothing are skipped."""
    steps: list[tuple[str, CohortFilter]] = []
    cur = cf
    if cur.countries:
        cur = cur.model_copy(update={"countries": []})
        steps.append(("countries", cur))
    if cur.start_year or cur.end_year:
        cur = cur.model_copy(update={"start_year": None, "end_year": None})
        steps.append(("years", cur))
    if cur.phases:
        cur = cur.model_copy(update={"phases": []})
        steps.append(("phases", cur))
    if cur.statuses:
        cur = cur.model_copy(update={"statuses": []})
        steps.append(("statuses", cur))
    if cur.drugs:
        term = " OR ".join([*([cur.term] if cur.term else []), *(f'"{d}"' if " " in d else d for d in cur.drugs)])
        cur = cur.model_copy(update={"drugs": [], "term": term})
        steps.append(("drugs_to_term", cur))
    return steps


async def relax_until_nonzero(client: Any, cf: CohortFilter) -> tuple[CohortFilter, list[str]]:
    """Apply the ladder (countries → years → phases → statuses → drugs-as-term) until a probe is > 0.

    Returns ``(relaxed_filter, relaxed_steps)``. If nothing helps (give up), returns ``(cf, [])``.
    At most five count probes are issued.
    """
    applied: list[str] = []
    for name, candidate in _relax_steps(cf):
        applied.append(name)
        if await client.count(SearchSpec.from_cohort(candidate, FieldSet.LANDSCAPE)) > 0:
            return candidate, applied
    return cf, []
