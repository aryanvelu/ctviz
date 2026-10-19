"""Provenance: the only way compute code turns API values into cited datums.

Usage pattern inside an intent handler::

    ds = Dataset()
    for s in studies:                       # Study wraps one raw API record
        ref = s.ref(P["phases"])            # FieldRef with a verbatim excerpt, or None if absent
        ...
    ds.emit({"x": "PHASE2", "y": 12}, refs) # datum gets id + citations (capped) automatically

Rules enforced here:
* ``Study.ref`` builds the excerpt from the raw JSON at creation time, so an excerpt can never
  disagree with its ``field_path``.
* ``Study.text_ref`` only accepts a span that exists verbatim in the field.
* ``verify_citation`` re-checks a citation against the raw record (used by the Verification agent
  and by tests).
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ctviz.api.schemas.citations import (
    MAX_CITATIONS_PER_DATUM,
    MAX_CITATIONS_SINGLE_TRIAL,
    MAX_EXCERPT_CHARS,
    Citation,
    CitationKind,
)
from ctviz.api.schemas.viz import Datum, GraphData
from ctviz.ctgov.fields import P
from ctviz.ctgov.paths import iter_list, path_get

STUDY_URL = "https://clinicaltrials.gov/study/{nct_id}"


def format_leaf(value: Any) -> str:
    """Serialize a JSON leaf exactly as it appears in the API (used for field_value excerpts)."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return json.dumps(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class FieldRef:
    """A pointer to evidence. Convert to a :class:`Citation` with :meth:`to_citation`."""

    nct_id: str
    field_path: str
    excerpt: str
    kind: CitationKind = CitationKind.FIELD_VALUE
    char_start: int | None = None
    char_end: int | None = None
    url: str | None = None
    label: str | None = None

    def to_citation(self) -> Citation:
        return Citation(
            nct_id=self.nct_id,
            excerpt=self.excerpt,
            field_path=self.field_path,
            kind=self.kind,
            char_start=self.char_start,
            char_end=self.char_end,
            url=self.url or STUDY_URL.format(nct_id=self.nct_id),
            label=self.label,
        )

    def key(self) -> tuple:
        return (self.kind, self.nct_id, self.field_path, self.char_start, self.char_end, self.url)


def derived_count_ref(count: int, search_url: str, label: str) -> FieldRef:
    """Evidence for a bucket whose value is a server-side ``totalCount``."""
    return FieldRef(
        nct_id="",
        field_path="$.totalCount",
        excerpt=str(count),
        kind=CitationKind.DERIVED_COUNT,
        url=search_url,
        label=label,
    )


class Study:
    """A raw ClinicalTrials.gov study record plus provenance-aware accessors."""

    __slots__ = ("raw", "nct_id")

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        nct = path_get(raw, P["nct_id"])
        if not isinstance(nct, str):
            raise ValueError("study record has no protocolSection.identificationModule.nctId")
        self.nct_id = nct

    def __repr__(self) -> str:
        return f"Study({self.nct_id})"

    # -- values -------------------------------------------------------------------------------
    def get(self, path_or_key: str, default: Any = None) -> Any:
        return path_get(self.raw, P.get(path_or_key, path_or_key), default)

    def items(self, path_or_key: str) -> Iterable[tuple[int, Any, str]]:
        """Iterate a list field as ``(index, item, item_path)``."""
        return iter_list(self.raw, P.get(path_or_key, path_or_key))

    @property
    def has_results(self) -> bool:
        return bool(self.raw.get("hasResults")) or "resultsSection" in self.raw

    # -- evidence -----------------------------------------------------------------------------
    def ref(self, path_or_key: str, label: str | None = None) -> FieldRef | None:
        """FieldRef for the leaf at the path, or ``None`` if the field is absent/empty."""
        path = P.get(path_or_key, path_or_key)
        value = path_get(self.raw, path)
        if value is None or value == "" or value == []:
            return None
        excerpt = format_leaf(value)
        if len(excerpt) > MAX_EXCERPT_CHARS:
            if isinstance(value, str):
                return self.text_ref(path, 0, MAX_EXCERPT_CHARS, label=label)
            excerpt = excerpt[:MAX_EXCERPT_CHARS]  # long arrays: prefix is still verbatim JSON text
        return FieldRef(self.nct_id, path, excerpt, label=label)

    def text_ref(self, path_or_key: str, start: int, end: int, label: str | None = None) -> FieldRef | None:
        """FieldRef for a verbatim substring ``value[start:end]`` of a text field (≤300 chars)."""
        path = P.get(path_or_key, path_or_key)
        value = path_get(self.raw, path)
        if not isinstance(value, str) or not (0 <= start < end <= len(value)):
            return None
        end = min(end, start + MAX_EXCERPT_CHARS)
        return FieldRef(
            self.nct_id,
            path,
            value[start:end],
            kind=CitationKind.TEXT_EXCERPT,
            char_start=start,
            char_end=end,
            label=label,
        )

    def quote_ref(self, path_or_key: str, quote: str, label: str | None = None) -> FieldRef | None:
        """Locate a (possibly LLM-produced) quote verbatim in a text field; None if not found.

        Whitespace differences are tolerated by matching on a normalized copy and mapping offsets back.
        """
        path = P.get(path_or_key, path_or_key)
        value = path_get(self.raw, path)
        if not isinstance(value, str) or not quote.strip():
            return None
        idx = value.find(quote)
        if idx >= 0:
            return self.text_ref(path, idx, idx + len(quote), label=label)
        span = _find_normalized(value, quote)
        return self.text_ref(path, span[0], span[1], label=label) if span else None


def _find_normalized(haystack: str, needle: str) -> tuple[int, int] | None:
    """Find ``needle`` in ``haystack`` ignoring runs of whitespace; return original offsets."""
    norm_chars: list[str] = []
    index_map: list[int] = []
    prev_space = False
    for i, ch in enumerate(haystack):
        if ch.isspace():
            if prev_space:
                continue
            norm_chars.append(" ")
            prev_space = True
        else:
            norm_chars.append(ch)
            prev_space = False
        index_map.append(i)
    norm = "".join(norm_chars)
    target = " ".join(needle.split())
    pos = norm.find(target)
    if pos < 0 or not target:
        return None
    return index_map[pos], index_map[pos + len(target) - 1] + 1


# Public name for modules that locate LLM quotes (extractors); the returned span is in original offsets,
# so ``haystack[start:end]`` is the verbatim excerpt to cite.
find_normalized = _find_normalized


def verify_citation(citation: Citation, raw: dict[str, Any]) -> bool:
    """True if the citation's excerpt is exactly supported by the raw record."""
    if citation.kind is CitationKind.DERIVED_COUNT:
        return citation.excerpt.isdigit()
    value = path_get(raw, citation.field_path)
    if value is None:
        return False
    if citation.kind is CitationKind.TEXT_EXCERPT:
        if not isinstance(value, str) or citation.char_start is None or citation.char_end is None:
            return False
        return value[citation.char_start : citation.char_end] == citation.excerpt
    full = format_leaf(value)
    if full == citation.excerpt:
        return True
    # Study.ref truncates long non-string leaves to exactly MAX_EXCERPT_CHARS characters.
    return len(citation.excerpt) == MAX_EXCERPT_CHARS and full.startswith(citation.excerpt)


@dataclass
class Dataset:
    """Accumulates cited datums for one visualization."""

    prefix: str = "d"
    max_citations: int = MAX_CITATIONS_PER_DATUM  # datums backed by several trials (they list nct_ids)
    max_citations_single_trial: int = MAX_CITATIONS_SINGLE_TRIAL  # one trial: keep long quotes complete
    rows: list[Datum] = field(default_factory=list)
    full_refs: dict[str, list[FieldRef]] = field(default_factory=dict)
    uncited: list[str] = field(default_factory=list)  # datum ids emitted without evidence (must carry cite_note)
    _counter: itertools.count = field(default_factory=lambda: itertools.count(1))

    def next_id(self) -> str:
        return f"{self.prefix}{next(self._counter)}"

    def make(
        self,
        values: dict[str, Any],
        refs: Sequence[FieldRef | None],
        datum_id: str | None = None,
        nct_ids: Sequence[str] | None | Literal["auto"] = "auto",
        keep_order: bool = False,
    ) -> Datum:
        """Build a datum without appending it.

        A datum without evidence is allowed only for honest empty states (zero-count buckets,
        derived remainders); it must then carry a ``cite_note`` explaining why, and is listed in
        ``uncited`` so verification can report it.

        ``nct_ids`` (default ``"auto"``): every distinct trial in ``refs``, sorted. Count datums built from
        retrieved records cite one ref per counted trial, so ``len(nct_ids)`` equals the count (checked by the
        validation step). Pass ``None`` when the refs are only a sample (registry counts).
        ``keep_order``: the handler ordered ``refs`` by meaning (e.g. first and latest trial first); don't sort.
        """
        unique: dict[tuple, FieldRef] = {}
        for r in refs:
            if r is not None and r.key() not in unique:
                unique[r.key()] = r
        # Sorted by trial (stable, so one trial's refs keep their order; derived counts first): the cited
        # examples are then the same on every run, whatever order the API returned the records in.
        deduped = list(unique.values()) if keep_order else sorted(unique.values(), key=lambda r: r.nct_id or "")
        did = datum_id or self.next_id()
        if did in self.full_refs:
            raise ValueError(f"duplicate datum id {did!r}")
        if not deduped:
            values = {"cite_note": "no supporting record (empty bucket or derived value)", **values}
            self.uncited.append(did)
        self.full_refs[did] = deduped
        trials = {r.nct_id for r in deduped if r.nct_id}
        cap = self.max_citations if len(trials) > 1 else self.max_citations_single_trial
        cites = [r.to_citation() for r in deduped[:cap]]
        total = len(deduped) if len(deduped) > cap else None
        reserved = {"id", "citations", "citations_total", "nct_ids"}
        clean = {k: v for k, v in values.items() if k not in reserved}
        if nct_ids == "auto":
            ids = sorted({r.nct_id for r in deduped if r.nct_id})
        else:
            ids = sorted(set(nct_ids)) if nct_ids is not None else None
        return Datum(id=did, nct_ids=ids or None, citations=cites, citations_total=total, **clean)

    def emit(
        self,
        values: dict[str, Any],
        refs: Sequence[FieldRef | None],
        datum_id: str | None = None,
        nct_ids: Sequence[str] | None | Literal["auto"] = "auto",
        keep_order: bool = False,
    ) -> Datum:
        datum = self.make(values, refs, datum_id, nct_ids, keep_order)
        self.rows.append(datum)
        return datum


def expand_citations(rows: Sequence[Datum], full_refs: dict[str, list[FieldRef]]) -> None:
    """citation_detail=full: every datum carries all its excerpts inline (verified like the sample ones)."""
    for d in rows:
        refs = full_refs.get(d.id)
        if refs and len(refs) > len(d.citations):
            d.citations = [r.to_citation() for r in refs]
            d.citations_total = None


@dataclass
class GraphDataset:
    """Cited nodes and edges for network_graph / sankey."""

    nodes: Dataset = field(default_factory=lambda: Dataset(prefix="n"))
    edges: Dataset = field(default_factory=lambda: Dataset(prefix="e"))

    def add_node(self, node_id: str, values: dict[str, Any], refs: Sequence[FieldRef | None]) -> Datum:
        """Add a node, or merge into an existing node with the same id (values updated, refs unioned)."""
        if node_id in self.nodes.full_refs:
            idx = next(i for i, d in enumerate(self.nodes.rows) if d.id == node_id)
            old = self.nodes.rows[idx]
            merged_vals = {**(old.model_extra or {}), **values}
            merged_vals.pop("cite_note", None)
            merged_refs = [*self.nodes.full_refs.pop(node_id), *refs]
            if node_id in self.nodes.uncited:
                self.nodes.uncited.remove(node_id)
            self.nodes.rows[idx] = self.nodes.make(merged_vals, merged_refs, datum_id=node_id)
            return self.nodes.rows[idx]
        return self.nodes.emit(values, refs, datum_id=node_id)

    def add_edge(self, values: dict[str, Any], refs: Sequence[FieldRef | None]) -> Datum:
        return self.edges.emit(values, refs)

    def to_graph(self) -> GraphData:
        return GraphData(nodes=self.nodes.rows, edges=self.edges.rows)

    @property
    def full_refs(self) -> dict[str, list[FieldRef]]:
        return {**self.nodes.full_refs, **self.edges.full_refs}

    @property
    def uncited(self) -> list[str]:
        return [*self.nodes.uncited, *self.edges.uncited]
