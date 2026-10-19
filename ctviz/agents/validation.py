"""Final validation: invariants enforced in code on the finished response (after the chart is built).

Each check is pure and returns pass/fail plus a short detail; the results go to ``meta.validation`` so a
reviewer (or the frontend) can see exactly what was enforced. Any failure also adds a
``VALIDATION_FAILED`` warning note. Checks:

* ``retrieval_matches_api_total`` — all_records: every cohort fetched exactly ``totalCount`` distinct trials;
  sample: the response carries a ``SAMPLED`` warning; registry_counts: buckets reconcile to ``totalCount``
  (``sum(buckets) + outside-range + not-reported == totalCount`` for single-valued dimensions).
* ``counts_match_trial_lists`` — every count datum built from records lists exactly that many NCT ids
  (bars, cells, nodes = trials, edges = shared trials); every registry-count datum has a ``query_url``.
* ``citations_from_retrieved_records`` — every cited NCT id is a record retrieved for this request.
* ``summary_numbers_in_data`` — every number in ``answer_summary`` appears in the chart data or facts.
* ``each_trial_in_one_bucket`` — where ``grouping.each_trial_counted_once`` (start year, phase, status, lead
  sponsor...), no NCT id appears in two data rows of the same series, so the bars add up to distinct trials.
* ``value_axes_labeled`` — every quantitative channel says what it measures (title) and count channels
  give a unit, so no chart ranks things by an unnamed number.
"""

from __future__ import annotations

from typing import Any

from ctviz.agents.verification import response_evidence, verify_summary
from ctviz.api.schemas.response import Check, Note, NoteCode, Validation, VisualizeResponse
from ctviz.api.schemas.viz import Datum, GraphData, VizType
from ctviz.viz.base import CohortData, note

COUNT_FIELDS = ("trial_count",)


def _rows(resp: VisualizeResponse) -> list[Datum]:
    viz = resp.visualization
    if viz is None:
        return []
    return viz.data.nodes + viz.data.edges if isinstance(viz.data, GraphData) else list(viz.data)


def check_retrieval(resp: VisualizeResponse, cohorts: list[CohortData], facts: dict[str, Any]) -> Check:
    method = resp.meta.coverage.counting_method
    name = "retrieval_matches_api_total"
    if method == "all_records":
        bad = [
            f"{c.label}: {len({s.nct_id for s in c.studies})} of {c.total_count}"
            for c in cohorts
            if len({s.nct_id for s in c.studies}) != c.total_count
        ]
        if bad:
            return Check(name=name, passed=False, detail="fetched != totalCount (" + "; ".join(bad) + ")")
        return Check(
            name=name,
            passed=True,
            detail="; ".join(f"{c.label}: {c.total_count:,} of {c.total_count:,}" for c in cohorts) or "no cohort",
        )
    if method == "sample":
        flagged = any(n.code is NoteCode.SAMPLED for n in resp.meta.notes)
        return Check(
            name=name,
            passed=flagged,
            detail=(
                f"sample of {resp.meta.coverage.studies_analyzed:,} of {resp.meta.coverage.studies_matched_total:,}; "
                + ("flagged with a SAMPLED note" if flagged else "NOT flagged")
            ),
        )
    if method == "registry_counts":
        rec = facts.get("reconciliation") or {}
        bad, ok = [], []
        for label, r in rec.items():
            parts, total = sum(v for k, v in r.items() if k != "total_derived"), r["total_derived"]
            (ok if parts == total else bad).append(f"{label}: {parts:,} {'=' if parts == total else '!='} {total:,}")
        if bad:
            return Check(name=name, passed=False, detail="buckets do not add up: " + "; ".join(bad))
        return Check(
            name=name,
            passed=True,
            detail=("buckets add up to totalCount (" + "; ".join(ok) + ")")
            if ok
            else "multi-valued dimension: each bucket is an exact count",
        )
    return Check(name=name, passed=True, detail=method)


def check_counts(resp: VisualizeResponse) -> Check:
    method = resp.meta.coverage.counting_method
    # networks count trials (node size / edge weight); a sankey counts participants and is not checked here
    graph = resp.visualization is not None and isinstance(resp.visualization.data, GraphData)
    if graph and resp.visualization.type.value != "network_graph":
        return Check(
            name="counts_match_trial_lists", passed=True, detail="participant flow: values are people, not trial counts"
        )
    bad: list[str] = []
    checked = 0
    for d in _rows(resp):
        extra = d.model_extra or {}
        if graph:
            count = extra.get("weight") if "source" in extra else extra.get("size")
        else:
            count = next((extra[f] for f in COUNT_FIELDS if isinstance(extra.get(f), int)), None)
        if not isinstance(count, int) or isinstance(count, bool) or extra.get("is_other") and d.nct_ids is None:
            continue
        checked += 1
        if d.query_url:
            continue  # registry count: reproducible by its query_url
        if method == "registry_counts":
            bad.append(f"{d.id}: registry count without query_url")
        elif count and d.nct_ids is None:
            bad.append(f"{d.id}: count {count} without a trial list")
        elif d.nct_ids is not None and len(d.nct_ids) != count:
            bad.append(f"{d.id}: count {count} but {len(d.nct_ids)} NCT ids")
    if bad:
        return Check(name="counts_match_trial_lists", passed=False, detail="; ".join(bad[:5]))
    return Check(name="counts_match_trial_lists", passed=True, detail=f"{checked} count datum(s) checked")


def check_exclusive(resp: VisualizeResponse) -> Check:
    name = "each_trial_in_one_bucket"
    viz = resp.visualization
    if viz is None or isinstance(viz.data, GraphData) or not (viz.meta.grouping or {}).get("each_trial_counted_once"):
        return Check(name=name, passed=True, detail="not a single-valued grouping")
    series = viz.encoding.series.field if viz.encoding.series else None
    seen: dict[tuple[Any, str], str] = {}
    dupes: list[str] = []
    for d in viz.data:
        key = (d.model_extra or {}).get(series) if series else None
        for n in d.nct_ids or []:
            if (key, n) in seen:
                dupes.append(f"{n} in {seen[(key, n)]} and {d.id}")
            else:
                seen[(key, n)] = d.id
    if dupes:
        return Check(name=name, passed=False, detail="; ".join(dupes[:5]))
    return Check(
        name=name,
        passed=True,
        detail=f"{len(seen)} trial(s), each in exactly one bucket"
        if seen
        else "registry counts: exclusive buckets, checked by reconciliation",
    )


def check_citations(resp: VisualizeResponse, retrieved: set[str]) -> Check:
    cited = {c.nct_id for d in _rows(resp) for c in d.citations if c.nct_id}
    listed = {n for d in _rows(resp) for n in (d.nct_ids or [])}
    unknown = sorted((cited | listed) - retrieved)
    return Check(
        name="citations_from_retrieved_records",
        passed=not unknown,
        detail=f"{len(cited | listed)} NCT id(s), all retrieved"
        if not unknown
        else f"not retrieved: {', '.join(unknown[:5])}",
    )


def check_summary(resp: VisualizeResponse, facts: dict[str, Any], extra: list[Any]) -> Check:
    if not resp.answer_summary or resp.visualization is None:
        return Check(name="summary_numbers_in_data", passed=True, detail="no numeric summary")
    _, notes = verify_summary(resp.answer_summary, response_evidence(resp, facts, extra))
    return Check(
        name="summary_numbers_in_data",
        passed=not notes,
        detail="every number appears in the data" if not notes else notes[0].message,
    )


def check_axes(resp: VisualizeResponse) -> Check:
    viz = resp.visualization
    if viz is None:
        return Check(name="value_axes_labeled", passed=True, detail="no chart")
    if viz.type is VizType.KPI_CARDS:  # no shared value axis: every card shows its own label (and unit)
        return Check(name="value_axes_labeled", passed=True, detail="kpi cards label each value")
    enc = viz.encoding
    bad = []
    for ch in (enc.x, enc.y, enc.size, enc.series):
        if ch is None or ch.type != "quantitative":
            continue
        if not ch.title or ch.title.strip().lower() in ("value", "count", "number"):
            bad.append(f"{ch.field} has no descriptive title")
        if ch.field in COUNT_FIELDS and not ch.unit:
            bad.append(f"{ch.field} has no unit")
    return Check(
        name="value_axes_labeled", passed=not bad, detail="; ".join(bad) or "all value axes say what they measure"
    )


def validate(
    resp: VisualizeResponse,
    *,
    cohorts: list[CohortData],
    facts: dict[str, Any],
    retrieved: set[str],
    extra_allowed: list[Any] | None = None,
) -> tuple[Validation, list[Note]]:
    """Run every check on an ``ok`` response; returns the validation block and any warning notes."""
    checks = [
        check_retrieval(resp, cohorts, facts),
        check_counts(resp),
        check_exclusive(resp),
        check_citations(resp, retrieved),
        check_summary(resp, facts, extra_allowed or []),
        check_axes(resp),
    ]
    failed = [c.name for c in checks if not c.passed]
    notes = (
        [
            note(
                NoteCode.VALIDATION_FAILED,
                f"Invariant check(s) failed: {', '.join(failed)}; see meta.validation.",
                "warning",
            )
        ]
        if failed
        else []
    )
    return Validation(passed=not failed, checks=checks), notes
