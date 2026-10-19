"""The public JSON a client receives: the assignment's shape, ``{visualization, meta}``, kept minimal.

    {
      "status": "ok",                         # or needs_clarification | unanswerable (then no visualization)
      "visualization": {"type", "title", "subtitle"?, "encoding", "data"},
      "answer_summary": "...",
      "meta": {
        "source", "filters", "query_interpretation", "assumptions", "notes",
        <rendering hints: units, sort, time_granularity, grouping, layout, geo, ...>,
        "coverage", "citation_policy", "validation", "alternatives", "follow_ups",
        "request_id", "created_at", "schema_version", "ctgov_api_version", "data_timestamp"
      },
      "clarification": {...}                  # only when status != ok
    }

``visualization`` holds exactly what is drawn. Everything a renderer needs besides the rows (units,
sorting, time granularity, grouping, layout, map level, network legend, thresholds...) sits in ``meta``
next to the filters and the interpretation, as the assignment lists them under "response metadata".
Empty lists/objects and defaults are dropped. Debug material (pipeline trace, cache counters, how a
chart was computed) appears under ``meta.debug`` only when ``include_trace`` / ``debug`` is requested.
:func:`to_public` is the single place this projection happens (API, CLI, runs/, examples).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from ctviz.api.schemas.citations import MAX_CITATIONS_PER_DATUM, MAX_CITATIONS_SINGLE_TRIAL
from ctviz.api.schemas.response import (
    MAX_NCT_IDS_PER_DATUM,
    Clarification,
    Coverage,
    Note,
    Validation,
    VisualizeResponse,
)
from ctviz.api.schemas.viz import Encoding, GraphData, VizType

FILTER_NAMES = {
    "drugs": "drug_names",
    "conditions": "conditions",
    "sponsors": "sponsors",
    "countries": "countries",
    "phases": "phases",
    "statuses": "statuses",
    "start_year": "start_year",
    "end_year": "end_year",
    "term": "term",
    "interventional_only": "interventional_only",
}
_DEBUG_VIZ_META = {"diagnostics"}
KEEP_NULL = ("estimated_count",)
STUDY_URL = "https://clinicaltrials.gov/study/{nct_id}"
DATUM_SOURCES_URL = "GET /v1/responses/{request_id}/datums/{datum_id}/citations"
# MAX_NCT_IDS_PER_DATUM (schemas/response.py): trial ids inlined per datum (sample / ids_only). A network node can
# stand for hundreds of trials; the complete list stays in the internal response (validation, response cache) and
# is served by DATUM_SOURCES_URL.
# Provenance keys moved to the end of a datum (never data columns). derived_from: an interpolated survival point
# names the reported datums it was computed from.
_PROVENANCE = ("id", "citations", "citations_total", "nct_ids", "query_url", "derived_from")


def _clean(obj: Any) -> Any:
    """Drop None and empty lists/dicts (false and 0 are kept: they carry meaning)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            v = _clean(v)
            if v is None or v == [] or v == {}:
                continue
            out[k] = v
        return out
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _citation(c: dict[str, Any], shared_path: str | None) -> dict[str, Any]:
    """``{nct_id, excerpt}`` plus only what differs from the default: an exact field value at the datum's
    ``field_path``. The study link is ``citation_policy.study_url``; a derived count's link is ``query_url``."""
    out = {k: c[k] for k in ("nct_id", "excerpt") if c.get(k) not in (None, "")}  # derived counts: no nct_id
    if c.get("field_path") != shared_path:
        out["field_path"] = c.get("field_path")
    if c.get("kind") and c["kind"] != "field_value":
        out["kind"] = c["kind"]
    for k in ("char_start", "char_end", "label"):
        if c.get(k) is not None:
            out[k] = c[k]
    return out


def _nct_ids(ids: list[str], detail: str) -> dict[str, Any]:
    """``{nct_ids, nct_ids_total}``: the first ``MAX_NCT_IDS_PER_DATUM`` ids in their existing (sorted) order,
    or all of them with ``detail="full"``; ``nct_ids_total`` is always the complete count."""
    shown = ids if detail == "full" else ids[:MAX_NCT_IDS_PER_DATUM]
    return {"nct_ids": shown, "nct_ids_total": len(ids)}


def _meta_citations(obj: Any) -> Any:
    """Apply the datum citation trimming to every ``citations`` list inside rendering hints (``thresholds``,
    reference lines, stats...): no per-citation ``url``, ``kind`` only when not a field value. There is no
    datum-level ``field_path`` here, so each citation keeps its own."""
    if isinstance(obj, dict):
        return {
            k: [_citation(c, None) if isinstance(c, dict) else c for c in v]
            if k == "citations" and isinstance(v, list)
            else _meta_citations(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_meta_citations(v) for v in obj]
    return obj


def _datum(d: dict[str, Any], detail: str = "sample") -> dict[str, Any]:
    """One public row/node/edge: encoded fields, then provenance keys (capped ``nct_ids`` + ``nct_ids_total``,
    ``query_url``, ``derived_from``, and unless ``ids_only`` the shared ``field_path``, inline ``citations`` and
    their totals)."""
    head = {"id": d["id"]}
    body = {k: v for k, v in d.items() if k not in _PROVENANCE}
    cites = d.get("citations", [])
    paths = Counter(c.get("field_path") for c in cites if c.get("kind", "field_value") == "field_value")
    shared = paths.most_common(1)[0][0] if paths else None
    tail: dict[str, Any] = {}
    if d.get("nct_ids"):
        tail.update(_nct_ids(d["nct_ids"], detail))
    for k in ("query_url", "derived_from"):
        if d.get(k) not in (None, []):
            tail[k] = d[k]
    if detail == "ids_only":
        return {**head, **body, **tail}
    if shared:
        tail["field_path"] = shared
    tail["citations"] = [_citation(c, shared) for c in cites]
    total = d.get("citations_total") or len(cites)
    tail["citations_total"] = total
    tail["excerpts_truncated"] = total > len(cites)
    return {**head, **body, **tail}


def _keep_null(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A field in ``KEEP_NULL`` present on any row is present on every row (null = not computed)."""
    for k in KEEP_NULL:
        if any(k in r for r in rows):
            for r in rows:
                r.setdefault(k, None)
    return rows


def citation_policy(detail: str = "sample") -> dict[str, Any]:
    """What a reader needs to interpret the citation keys, so nobody guesses why 120 trials show 3 excerpts."""
    out: dict[str, Any] = {"detail": detail, "study_url": STUDY_URL}
    if detail == "full":
        out.update(nct_ids="complete", excerpts_per_datum="all", excerpt_order="nct_id")
        return out
    out.update(
        nct_ids=f"first_{MAX_NCT_IDS_PER_DATUM}",
        nct_ids_per_datum=MAX_NCT_IDS_PER_DATUM,
        nct_ids_order="nct_id",
        all_nct_ids=DATUM_SOURCES_URL,
    )
    if detail == "sample":
        out.update(
            excerpts_per_datum=MAX_CITATIONS_PER_DATUM,
            excerpts_single_trial=MAX_CITATIONS_SINGLE_TRIAL,
            excerpt_order="nct_id",
            all_excerpts=DATUM_SOURCES_URL,
        )
    return out


def _filters(interp: dict[str, Any]) -> dict[str, Any]:
    def one(c: dict[str, Any]) -> dict[str, Any]:
        return {FILTER_NAMES[k]: v for k, v in c.items() if k in FILTER_NAMES and v not in (None, [], False, "")}

    cohorts = interp.get("cohorts") or []
    out: dict[str, Any] = {}
    if interp.get("nct_ids"):
        out["nct_ids"] = interp["nct_ids"]
    if len(cohorts) == 1:
        out.update(one(cohorts[0]))
    elif cohorts:
        out["compare"] = [{"label": c["label"], **one(c)} for c in cohorts]
    return out


def to_public(resp: VisualizeResponse, *, debug: bool = False, citation_detail: str = "sample") -> dict[str, Any]:
    """Project the internal response onto the public contract (see module docstring)."""
    raw = resp.model_dump(mode="json", exclude_none=True)
    m = raw["meta"]
    interp = m["interpretation"]
    out: dict[str, Any] = {"status": raw.get("status", "ok")}
    viz_meta: dict[str, Any] = {}
    if resp.visualization is not None:
        v = raw["visualization"]
        viz_meta = dict(v.get("meta") or {})
        approx = viz_meta.pop("approximation", None) or {}
        if approx.get("is_approximation"):
            viz_meta["approximation"] = {k: x for k, x in approx.items() if k != "is_approximation"}
        viz_meta = _meta_citations(viz_meta)
        data = v["data"]
        if isinstance(resp.visualization.data, GraphData):
            data = {
                "nodes": [_datum(n, citation_detail) for n in data["nodes"]],
                "edges": [_datum(e, citation_detail) for e in data["edges"]],
            }
        else:
            data = _keep_null([_datum(d, citation_detail) for d in data])
        out["visualization"] = {
            "type": v["type"],
            "title": v["title"],
            "subtitle": v.get("subtitle"),
            "encoding": _clean(v["encoding"]),
        }
    if raw.get("answer_summary"):
        out["answer_summary"] = raw["answer_summary"]
    if raw.get("clarification"):
        out["clarification"] = raw["clarification"]
    meta: dict[str, Any] = {
        "source": raw.get("source", "clinicaltrials.gov"),
        "filters": _filters(interp),
        "query_interpretation": {
            "intent": interp["intent"],
            "persona": interp["persona"],
            "scope": interp["scope"],
            "confidence": interp["intent_confidence"],
            "parser": interp["parser"],
            "cohorts": [
                {k: x for k, x in c.items() if x not in (None, [], False, "")} for c in interp.get("cohorts", [])
            ],
            "resolved_entities": [
                {"input": e["input"], "resolved": e["resolved"], "kind": e["kind"], "method": e["method"]}
                for e in interp.get("resolved_entities", [])
            ],
        },
        "assumptions": interp.get("assumptions", []),
        "notes": [{k: x for k, x in n.items() if not (k == "affects" and x == ["*"])} for n in m.get("notes", [])],
        **{k: x for k, x in viz_meta.items() if k not in _DEBUG_VIZ_META},
        "coverage": m.get("coverage"),
        "citation_policy": citation_policy(citation_detail) if resp.visualization is not None else None,
        "validation": m.get("validation"),
        "alternatives": m.get("alternatives", []),
        "follow_ups": raw.get("follow_ups", []),
        "request_id": raw["request_id"],
        "created_at": raw["created_at"],
        "schema_version": raw["schema_version"],
        "ctgov_api_version": raw.get("ctgov_api_version"),
        "data_timestamp": raw.get("data_timestamp"),
    }
    if debug:
        meta["debug"] = {
            "api_params": m.get("filters_applied"),
            "cache": m.get("cache"),
            "trace": m.get("trace"),
            "chart_diagnostics": viz_meta.get("diagnostics"),
        }
    out["meta"] = meta
    out = _clean(out)
    if resp.visualization is not None:  # rows are already minimal; added after _clean so KEEP_NULL survives
        out["visualization"]["data"] = data
    return out


# -- documentation models (GET /v1/schema): the public contract, field by field ----------------------
class PublicCitation(BaseModel):
    nct_id: str | None = Field(default=None, description="Trial id, e.g. NCT03548935 (absent for kind=derived_count).")
    excerpt: str = Field(description="Exact field value, or verbatim text excerpt (kind=text_excerpt).")
    field_path: str | None = Field(default=None, description="Only when it differs from the datum's field_path.")
    kind: str | None = Field(default=None, description="Omitted for exact field values; text_excerpt | derived_count.")
    char_start: int | None = None
    char_end: int | None = None


class PublicDatum(BaseModel):
    """One row / node / edge. Every field named by ``encoding`` is present; plus provenance keys."""

    id: str
    nct_ids: list[str] | None = Field(
        default=None,
        description="Trials behind this datum, sorted by nct_id: the first 20 (citation_policy.nct_ids_per_datum), "
        "or all with citation_detail=full. Absent for registry counts.",
    )
    nct_ids_total: int | None = Field(
        default=None,
        description="Number of distinct trials behind this datum (present whenever nct_ids is); for counts built "
        "from retrieved records it equals the count. len(nct_ids) < nct_ids_total means the list was capped; "
        "the complete list via citation_policy.all_nct_ids.",
    )
    query_url: str | None = Field(default=None, description="registry_counts only: search reproducing the value.")
    derived_from: list[str] | None = Field(
        default=None,
        description="survival_curve interpolated points only: ids of the reported datums (same series) this "
        "value was computed from; the point itself has no registry value to cite.",
    )
    field_path: str | None = Field(
        default=None, description="Dot path (into the study JSON) shared by the excerpts below."
    )
    citations: list[PublicCitation] = Field(
        default_factory=list, description="Verbatim excerpts for the first trials by nct_id."
    )
    citations_total: int | None = Field(
        default=None, description="Excerpts available for this datum (absent with citation_detail=ids_only)."
    )
    excerpts_truncated: bool | None = Field(
        default=None, description="citations_total > len(citations); the rest via citation_policy.all_excerpts."
    )


class PublicGraph(BaseModel):
    nodes: list[PublicDatum]
    edges: list[PublicDatum]


class PublicVisualization(BaseModel):
    type: VizType
    title: str
    subtitle: str | None = None
    encoding: Encoding
    data: list[PublicDatum] | PublicGraph


class PublicMeta(BaseModel):
    source: str = "clinicaltrials.gov"
    filters: dict[str, Any] = Field(
        default_factory=dict, description="Filters actually searched (per cohort under 'compare')."
    )
    query_interpretation: dict[str, Any] = Field(
        description="intent, persona, scope, confidence, parser, cohorts, resolved_entities"
    )
    assumptions: list[str] = []
    notes: list[Note] = []
    units: dict[str, str] | None = Field(
        default=None,
        description="Rendering hints from here on (only those that "
        "apply to the chart type are present): units, "
        "sort, time_granularity, grouping, layout, "
        "thresholds, bands, truncation, approximation, "
        "columns, summary, node_types, edge_semantics, "
        "network_kind, directed, network_layout, pinned, "
        "stages, time_axis, lanes, geo, reference_line, "
        "highlight_zero, stats (docs/response_schema.md).",
    )
    coverage: Coverage
    citation_policy: dict[str, Any] | None = Field(
        default=None,
        description="detail; nct_ids (first_20 | complete), nct_ids_per_datum, all_nct_ids; "
        "excerpts_per_datum; excerpt_order; all_excerpts; study_url template.",
    )
    validation: Validation | None = None
    alternatives: list[dict[str, str]] = []
    follow_ups: list[str] = []
    request_id: str
    created_at: str
    schema_version: str
    ctgov_api_version: str | None = None
    data_timestamp: str | None = None


class PublicResponse(BaseModel):
    status: str = Field(description="ok | needs_clarification | unanswerable")
    visualization: PublicVisualization | None = Field(default=None, description="Only when status is ok.")
    answer_summary: str | None = None
    clarification: Clarification | None = Field(default=None, description="Only when status is not ok.")
    meta: PublicMeta
