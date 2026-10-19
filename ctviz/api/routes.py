"""HTTP routes (``/v1``). ``/v1/visualize`` returns the public projection (:func:`ctviz.api.public.to_public`)."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Path, Query, Request
from fastapi.responses import JSONResponse

from ctviz.api.public import PublicResponse, to_public
from ctviz.api.schemas.citations import Citation, CitationKind
from ctviz.api.schemas.request import NCT_PATTERN, VisualizeRequest
from ctviz.api.schemas.response import SCHEMA_VERSION, ErrorResponse, VisualizeResponse
from ctviz.core.intents import load_intents
from ctviz.core.provenance import verify_citation
from ctviz.ctgov.client import CTGovUnavailable
from ctviz.obs.metrics import METRICS
from ctviz.store import response_cache as rc
from ctviz.store.study_store import load_studies
from ctviz.viz.base import registered_intents

router = APIRouter(prefix="/v1")


def _json(model_or_dict: Any, status: int = 200) -> JSONResponse:
    if hasattr(model_or_dict, "model_dump"):
        model_or_dict = model_or_dict.model_dump(mode="json", exclude_none=True)
    return JSONResponse(model_or_dict, status_code=status)


def _err(status: int, error: str, message: str) -> JSONResponse:
    return _json(ErrorResponse(error=error, message=message), status)


@router.post(
    "/visualize",
    response_model=PublicResponse,
    response_model_exclude_none=True,
    summary="Turn a clinical-trial question into a cited visualization spec",
)
async def visualize(body: VisualizeRequest, request: Request) -> JSONResponse:
    """Main endpoint. Never 500s for valid input: problems surface as ``meta.notes``."""
    resp: VisualizeResponse = await request.app.state.orchestrator.run(body)
    # The X-Request-ID header must equal meta.request_id (the id citations are stored under); the guard middleware
    # reads this back after the route returns.
    request.state.request_id = resp.request_id
    meta = resp.meta
    METRICS.inc(f"intent_{meta.interpretation.intent.value}")
    METRICS.inc("response_cache_hits" if meta.cache.response_hit else "response_cache_misses")
    if meta.trace is not None:
        METRICS.observe(f"visualize {'cached' if meta.cache.response_hit else 'computed'}", meta.trace.total_ms)
    request.state.log_fields = {
        "ctviz_request_id": resp.request_id,
        "intent": meta.interpretation.intent.value,
        "parser": meta.interpretation.parser,
        "answer_status": resp.status,
        "viz": resp.visualization.type.value if resp.visualization else None,
        "response_hit": meta.cache.response_hit,
        "notes": [n.code.value for n in meta.notes],
        "total_ms": meta.trace.total_ms if meta.trace else None,
        "llm_calls": len(meta.trace.llm_calls) if meta.trace else None,
        "cost_usd": meta.trace.cost_usd if meta.trace else None,
    }
    out = _json(to_public(resp, debug=body.include_trace, citation_detail=body.citation_detail))
    out.headers["X-Request-ID"] = resp.request_id
    return out


@router.get("/schema", summary="JSON Schemas of the request and response")
async def schema() -> JSONResponse:
    return _json(
        {
            "schema_version": SCHEMA_VERSION,
            "request": VisualizeRequest.model_json_schema(),
            "response": PublicResponse.model_json_schema(),
        }
    )


@router.get("/intents", summary="Supported question types")
async def intents() -> JSONResponse:
    registered = set(registered_intents())
    out = [
        {
            "intent": s.intent.value,
            "description": s.description,
            "personas": [p.value for p in s.personas],
            "scopes": [x.value for x in s.scopes],
            "primary_visualization": s.primary.value,
            "fallbacks": [f.value for f in s.fallbacks],
            "examples": s.examples,
            "registered": s.intent in registered,
        }
        for s in load_intents().values()
    ]
    return _json({"intents": out})


@router.get("/trials/{nct_id}", summary="Raw ClinicalTrials.gov record (what citation field_paths point into)")
async def trial(request: Request, nct_id: str = Path(pattern=NCT_PATTERN)) -> JSONResponse:
    from ctviz.ctgov.retrieval import fetch_trials

    try:
        studies = await fetch_trials(request.app.state.client, [nct_id.upper()])
    except CTGovUnavailable:
        out = _json(
            ErrorResponse(
                error="ctgov_unavailable",
                retry_after_s=30,
                message="ClinicalTrials.gov is unavailable right now; try again shortly.",
            ),
            503,
        )
        out.headers["Retry-After"] = "30"
        return out
    if not studies:
        return _err(404, "not_found", f"{nct_id} was not found on ClinicalTrials.gov")
    return _json(studies[0].raw)


@router.get("/graph/neighbors", summary="Knowledge-graph neighbours of an entity, with evidence")
async def graph_neighbors(
    request: Request,
    name: str = Query(min_length=1, max_length=200),
    node_type: str = Query(default="drug", max_length=40),
    limit: int = Query(default=50, ge=1, le=200),
) -> JSONResponse:
    """``{center, match, nodes, edges}``; ``match`` is ``exact``/``alias``/``fuzzy`` (closest match) or null."""
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        return _err(503, "graph_unavailable", "The knowledge graph is not enabled on this server.")
    from ctviz.compute.graph_ingest import NODE_TYPES

    if node_type not in NODE_TYPES:
        return _err(422, "validation_error", f"node_type must be one of {', '.join(NODE_TYPES)}")
    res = await graph.neighbors(name, node_type, limit)
    return _json(
        {
            "center": res.get("center"),
            "match": res.get("match"),
            "nodes": res.get("nodes", []),
            "edges": res.get("edges", []),
        }
    )


@router.get(
    "/responses/{request_id}/datums/{datum_id}/citations",
    summary="Full evidence for one datum: every citation (re-verified) and the complete nct_ids list",
)
async def datum_citations(
    request: Request, request_id: str = Path(pattern=r"^[0-9a-f]{32}$"), datum_id: str = Path(max_length=64)
) -> JSONResponse:
    """Everything the public JSON caps for one datum.

    ``citations`` is every stored excerpt, each re-checked against the stored raw record (``verified``).
    ``nct_ids`` is the datum's complete, sorted trial list (the response inlines only the first
    ``citation_policy.nct_ids_per_datum``) and ``nct_ids_total`` its length; both are absent for datums
    without trial ids (registry counts) and for responses stored before schema 2.2. If the stored record is
    missing for a response this process produced, it is served from memory (for a response-cache hit, rebuilt
    from the cached response: inline citations only, complete nct_ids).
    """
    # The orchestrator also answers while the background write is pending (or was lost); plain store otherwise.
    resolve = getattr(request.app.state.orchestrator, "evidence", None)
    got = await resolve(request_id) if resolve else await rc.get_datum_sources(request.app.state.db, request_id)
    if got is None:
        return _err(404, "not_found", "Unknown or expired request_id.")
    cites, ncts = got
    if datum_id not in cites and datum_id not in ncts:
        return _err(404, "not_found", f"No datum '{datum_id}' in that response.")
    items = await _reverify(request.app.state.db, cites.get(datum_id, []))
    verified = sum(1 for c in items if c["verified"])
    out: dict[str, Any] = {
        "request_id": request_id,
        "datum_id": datum_id,
        "total": len(items),
        "verified": verified,
        "unverified": len(items) - verified,
        "citations": items,
    }
    ids = ncts.get(datum_id)
    if ids:
        out["nct_ids"] = list(ids)
        out["nct_ids_total"] = len(ids)
    return _json(out)


async def _reverify(db: Any, stored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-check each stored citation against the raw record in the study store (one DB read for all trials).

    Every item gets ``verified``: True only if the excerpt is still exactly supported by the stored record
    (:func:`verify_citation`); False if it is not, the record is no longer stored, or the item is malformed.
    """
    ncts = sorted(
        {c["nct_id"] for c in stored if isinstance(c, dict) and isinstance(c.get("nct_id"), str) and c.get("nct_id")}
    )
    raws = await load_studies(db, ncts) if ncts else {}
    out: list[dict[str, Any]] = []
    for item in stored:
        ok = False
        try:
            cite = Citation.model_validate(item)
            if cite.kind is CitationKind.DERIVED_COUNT:  # a count of a search, not a record field
                ok = verify_citation(cite, {})
            else:
                raw = raws.get(cite.nct_id)
                ok = raw is not None and verify_citation(cite, raw)
        except Exception:  # noqa: BLE001 - a malformed stored item is reported as unverified, never served as proof
            ok = False
        out.append({**item, "verified": ok} if isinstance(item, dict) else {"value": item, "verified": False})
    return out


@router.get("/health", summary="Liveness and configuration flags (never reveals secrets)")
async def health(request: Request) -> JSONResponse:
    state = request.app.state
    settings = state.settings
    version: dict[str, Any] = {}
    client = getattr(state, "client", None)
    if client is not None:
        try:
            version = await asyncio.wait_for(client.version(), timeout=2.0) or {}
        except Exception:
            version = {}
    return _json(
        {
            "status": "ok",
            "schema_version": SCHEMA_VERSION,
            "offline": settings.offline,
            "llm_configured": settings.openai_api_key is not None and settings.llm_enabled,
            "ctgov_api_version": version.get("apiVersion"),
            "data_timestamp": version.get("dataTimestamp"),
            "graph_enabled": getattr(state, "graph", None) is not None,
            "handlers_registered": len(registered_intents()),
        }
    )


@router.get("/metrics", summary="Counters and latency percentiles")
async def metrics() -> JSONResponse:
    return _json(METRICS.snapshot())
