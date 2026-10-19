"""Response cache (ns ``response``) and the per-response citation store (ns ``citations``)."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any

from ctviz.api.schemas.request import VisualizeRequest
from ctviz.api.schemas.response import SCHEMA_VERSION, VisualizeResponse
from ctviz.api.schemas.viz import GraphData
from ctviz.core.provenance import FieldRef

RESPONSE_NS = "response"
CITATIONS_NS = "citations"
_KEY_EXCLUDE = {"include_trace", "refresh"}  # do not change the answer
# Bump when handler logic changes what an answer contains (old cached answers then stop matching).
ANSWER_LOGIC_VERSION = "2026-10-19.15"


def response_key(request: VisualizeRequest) -> str:
    """sha256 of the canonical request JSON (sorted keys, defaults included, trace/refresh flags excluded) plus
    the schema and answer-logic versions, so a code change never serves an answer computed the old way."""
    body = request.model_dump(mode="json", exclude=_KEY_EXCLUDE)
    body["_versions"] = [SCHEMA_VERSION, ANSWER_LOGIC_VERSION]
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


async def get_response(db: Any, key: str) -> dict[str, Any] | None:
    return await db.cache_get(RESPONSE_NS, key)


async def put_response(db: Any, key: str, value: dict[str, Any], ttl_s: float) -> None:
    await db.cache_set(RESPONSE_NS, key, value, ttl_s)


def serialize_refs(full_refs: dict[str, list[FieldRef]]) -> dict[str, list[dict[str, Any]]]:
    """``datum id -> [citation json]`` for every piece of evidence (not just the inline 10)."""
    return {
        did: [r.to_citation().model_dump(mode="json", exclude_none=True) for r in refs]
        for did, refs in full_refs.items()
    }


_RECORD_FORMAT = 2  # {"_format": 2, "citations": {...}, "nct_ids": {...}}; older records are a bare citations map


async def put_citations(
    db: Any,
    request_id: str,
    citations: dict[str, list[dict[str, Any]]],
    ttl_s: float,
    nct_ids: dict[str, list[str]] | None = None,
) -> None:
    """Store one response's evidence: every citation per datum and, per datum, the **complete** ``nct_ids``
    list (the public JSON inlines only the first ``MAX_NCT_IDS_PER_DATUM``; this is where the rest lives)."""
    record = {"_format": _RECORD_FORMAT, "citations": citations, "nct_ids": nct_ids or {}}
    await db.cache_set(CITATIONS_NS, request_id, record, ttl_s)


def _unwrap(record: Any) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]] | None:
    """``(citations, nct_ids)`` from a stored record of either format (a legacy record has no nct_ids)."""
    if not isinstance(record, dict):
        return None
    if record.get("_format") == _RECORD_FORMAT:
        return record.get("citations") or {}, record.get("nct_ids") or {}
    return record, {}


async def get_datum_sources(
    db: Any, request_id: str
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]] | None:
    """``(datum id -> citations, datum id -> complete nct_ids)`` for a stored response, or None if unknown/expired."""
    return _unwrap(await db.cache_get(CITATIONS_NS, request_id))


async def get_citations(db: Any, request_id: str) -> dict[str, list[dict[str, Any]]] | None:
    """``datum id -> citations`` for a stored response, or None if unknown/expired."""
    got = await get_datum_sources(db, request_id)
    return None if got is None else got[0]


async def copy_citations(db: Any, src: str, dst: str, ttl_s: float) -> bool:
    """Store ``src``'s evidence record under ``dst`` too (a response-cache hit gets a new request id).
    Copies the record verbatim, so the complete nct_ids lists come along. Returns False if ``src`` is gone."""
    record = await db.cache_get(CITATIONS_NS, src)
    if record is None:
        return False
    await db.cache_set(CITATIONS_NS, dst, record, ttl_s)
    return True


Evidence = tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]  # (datum id -> citations, -> nct_ids)


def evidence_from_response(resp: VisualizeResponse) -> Evidence:
    """Rebuild a datum-evidence record from an internal response: each datum's inline citations and its
    complete ``nct_ids`` (the internal response is never capped). Used when the stored record is missing;
    citations beyond the inline sample are then not available."""
    viz = resp.visualization
    if viz is None:
        return {}, {}
    rows = viz.data.nodes + viz.data.edges if isinstance(viz.data, GraphData) else viz.data
    cites = {d.id: [c.model_dump(mode="json", exclude_none=True) for c in d.citations] for d in rows}
    return cites, {d.id: list(d.nct_ids) for d in rows if d.nct_ids}


async def copy_or_rebuild(db: Any, src: str | None, dst: str, ttl_s: float, resp: VisualizeResponse) -> None:
    """Give a response-cache hit (request id ``dst``) its evidence record: copy ``src``'s record, or, if that
    is gone (expired, or its write never happened), rebuild one from the cached internal response."""
    if src and await copy_citations(db, src, dst, ttl_s):
        return
    cites, ncts = evidence_from_response(resp)
    await put_citations(db, dst, cites, ttl_s, nct_ids=ncts)


class PendingEvidence:
    """In-memory evidence for responses whose citation-store write has not landed yet (bounded LRU).

    The write is a background task, so a client can ask GET .../datums/{id}/citations before it runs (or after
    it failed). Each entry is either the exact record about to be written, or ``(source request id, cached
    response)`` for a response-cache hit. :meth:`resolve` turns an entry into evidence; :meth:`done` drops it
    once the write succeeded."""

    def __init__(self, max_entries: int = 128):
        self.max_entries = max_entries
        self._items: OrderedDict[str, tuple[Evidence | None, str | None, VisualizeResponse | None]] = OrderedDict()

    def remember(
        self,
        request_id: str,
        *,
        record: Evidence | None = None,
        src: str | None = None,
        resp: VisualizeResponse | None = None,
    ) -> None:
        """Hold evidence for ``request_id`` until :meth:`done`; the oldest entry goes when full."""
        self._items[request_id] = (record, src, resp)
        self._items.move_to_end(request_id)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def done(self, request_id: str) -> None:
        """The stored record exists now; forget the in-memory copy."""
        self._items.pop(request_id, None)

    async def resolve(self, db: Any, request_id: str) -> Evidence | None:
        """Evidence for ``request_id``: the stored record, else the pending record, else the cache-hit source's
        stored record, else a rebuild from the cached response; None if the id is unknown."""
        got = await get_datum_sources(db, request_id)
        if got is not None:
            return got
        item = self._items.get(request_id)
        if item is None:
            return None
        record, src, resp = item
        if record is not None:
            return record
        if src:
            got = await get_datum_sources(db, src)
            if got is not None:
                return got
        return evidence_from_response(resp) if resp is not None else None
