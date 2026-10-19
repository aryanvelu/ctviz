"""Verification agent (pure code): the last gate before a response leaves the server.

1. Every citation is re-verified against the raw ClinicalTrials.gov record it points at
   (:func:`ctviz.core.provenance.verify_citation`); failures are dropped and noted.
2. Every number in ``answer_summary`` must appear (at the token's own precision) among the chart's
   numeric values, the handler fact values or the coverage counts. ``%`` tokens may match fraction
   values x100; digits inside known labels ("Phase 2") are ignored; years may come from dates/labels.
   The question text and the filters are not evidence. Failing sentences are dropped (``SUMMARY_TRIMMED``).
3. The encoding is re-checked (a failing chart becomes a valid table) and the response is re-validated.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ctviz.api.schemas.response import Note, NoteCode, VisualizeResponse
from ctviz.api.schemas.viz import (
    Channel,
    Datum,
    Encoding,
    GraphData,
    TableColumn,
    Visualization,
    VizMeta,
    VizType,
)
from ctviz.core.provenance import verify_citation
from ctviz.viz.base import EncodingError, check_encoding, make_viz, note
from ctviz.viz.selector import empty_kpi

_NCT = re.compile(r"\[?NCT\d{8}\]?")
_NUM = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")


# -- citations ------------------------------------------------------------------------------------
def _verify_rows(rows: list[Datum], raw_by_nct: dict[str, dict]) -> tuple[list[Datum], int, list[str]]:
    dropped = 0
    affected: list[str] = []
    out: list[Datum] = []
    for d in rows:
        kept = [
            c
            for c in d.citations
            if (c.kind.value == "derived_count" and verify_citation(c, {}))
            or (c.nct_id in raw_by_nct and verify_citation(c, raw_by_nct[c.nct_id]))
        ]
        if len(kept) != len(d.citations):
            dropped += len(d.citations) - len(kept)
            affected.append(d.id)
            update: dict[str, Any] = {"citations": kept}
            if d.citations_total is not None:
                update["citations_total"] = max(len(kept), d.citations_total - (len(d.citations) - len(kept)))
            d = d.model_copy(update=update)
        out.append(d)
    return out, dropped, affected


def verify_citations(viz: Visualization, raw_by_nct: dict[str, dict]) -> tuple[Visualization, list[Note]]:
    """Drop citations whose excerpt is not exactly supported by the raw record (or whose record is unknown)."""
    if isinstance(viz.data, GraphData):
        nodes, dn, an = _verify_rows(viz.data.nodes, raw_by_nct)
        edges, de, ae = _verify_rows(viz.data.edges, raw_by_nct)
        dropped, affected = dn + de, an + ae
        new_viz = viz.model_copy(update={"data": GraphData(nodes=nodes, edges=edges)}) if dropped else viz
    else:
        rows, dropped, affected = _verify_rows(viz.data, raw_by_nct)
        new_viz = viz.model_copy(update={"data": rows}) if dropped else viz
    notes = []
    if dropped:
        notes.append(
            note(
                NoteCode.DEGRADED,
                f"{dropped} citation(s) could not be re-verified against the source records and were removed.",
                "warning",
                affected[:50],
            )
        )
    return new_viz, notes


# -- summary numbers ------------------------------------------------------------------------------
def _parse_number(token: str) -> tuple[float, int, bool]:
    """``'1,306'`` → (1306, 0, False); ``'44.2%'`` → (44.2, 1, True)."""
    pct = token.endswith("%")
    t = token.rstrip("%").replace(",", "").lstrip("+")
    decimals = len(t.split(".")[1]) if "." in t else 0
    return float(t), decimals, pct


def numbers_in(text: str) -> list[str]:
    """Number tokens in ``text`` (NCT identifiers excluded)."""
    return _NUM.findall(_NCT.sub(" ", text))


_YEAR = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
_DATE = re.compile(r"(?<!\d)(?:19|20)\d{2}-\d{2}(?:-\d{2})?(?!\d)")
_HAS_LETTER = re.compile(r"[A-Za-z]")


@dataclass
class Evidence:
    """What a summary may mention: numeric field values, fraction-typed values, labels and years."""

    values: set[float] = field(default_factory=set)
    fractions: set[float] = field(default_factory=set)  # non-integer values in [0, 1] (shares, rates)
    labels: set[str] = field(default_factory=set)  # text containing letters; masked before scanning numbers
    years: set[float] = field(default_factory=set)  # 4-digit years appearing inside labels / dates
    dates: set[str] = field(default_factory=set)  # ISO dates in the data; masked like labels

    def add_number(self, v: Any) -> None:
        if isinstance(v, bool) or not isinstance(v, int | float):
            return
        self.values.add(float(v))
        if isinstance(v, float) and 0.0 <= v <= 1.0 and not v.is_integer():
            self.fractions.add(v)

    def add_text(self, text: str) -> None:
        if not text:
            return
        if _HAS_LETTER.search(text):
            self.labels.add(text)
        self.dates.update(_DATE.findall(text))
        self.years.update(float(y) for y in _YEAR.findall(text))

    def add_values(self, obj: Any) -> None:
        """Walk JSON values (never dict keys); skips citations, ids, notes and URLs."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k not in ("citations", "id", "cite_note", "url"):
                    self.add_values(v)
        elif isinstance(obj, list | tuple):
            for v in obj:
                self.add_values(v)
        elif isinstance(obj, str):
            self.add_text(obj)
        else:
            self.add_number(obj)


def _supported(token: str, ev: Evidence) -> bool:
    """A token is supported if a value rounded to the token's own precision equals it.

    ``%`` tokens may also match a fraction-typed value x100 (0.442 -> 44.2%). Four-digit year tokens may
    match a year seen in labels/dates.
    """
    value, decimals, pct = _parse_number(token)
    if any(round(a, decimals) == value for a in ev.values):
        return True
    if pct and any(round(a * 100, decimals) == value for a in ev.fractions):
        return True
    return decimals == 0 and not pct and value in ev.years and len(token.lstrip("+-")) == 4


def _mask_labels(sentence: str, labels: set[str]) -> str:
    """Blank out known labels ('Phase 2', 'COVID-19') so their digits are not checked as claims."""
    for label in sorted(labels, key=len, reverse=True):
        if len(label) >= 2 and label in sentence:
            sentence = sentence.replace(label, " ")
    return sentence


def verify_summary(summary: str | None, evidence: Evidence) -> tuple[str | None, list[Note]]:
    """Drop sentences containing a number that the evidence does not support."""
    if not summary:
        return summary, []
    sentences = [s for s in _SENTENCE.split(summary.strip()) if s.strip()]
    masks = evidence.labels | evidence.dates
    kept = [s for s in sentences if all(_supported(t, evidence) for t in numbers_in(_mask_labels(s, masks)))]
    if len(kept) == len(sentences):
        return summary, []
    trimmed = " ".join(kept).strip() or None
    return trimmed, [
        note(
            NoteCode.SUMMARY_TRIMMED,
            f"{len(sentences) - len(kept)} summary sentence(s) with numbers not found in the chart data were removed.",
        )
    ]


# -- whole response -------------------------------------------------------------------------------
def response_evidence(
    resp: VisualizeResponse, facts: dict[str, Any] | None = None, extra: Iterable[Any] = ()
) -> Evidence:
    """Evidence for the summary: chart values, fact values (keys only as labels), title, coverage counts.

    The user's question and ``filters_applied`` are deliberately excluded.
    """
    ev = Evidence()
    viz = resp.visualization
    rows = viz.data.nodes + viz.data.edges if isinstance(viz.data, GraphData) else viz.data
    ev.add_values([r.model_dump(mode="json") for r in rows])
    for k, v in (facts or {}).items():
        ev.add_values(v)
        for part in str(k).split("|"):
            ev.add_text(part)
    ev.add_text(viz.title)
    ev.add_text(viz.subtitle or "")
    cov = resp.meta.coverage
    for v in (cov.studies_matched_total, cov.studies_analyzed, cov.studies_with_results, *extra):
        ev.add_number(v)
    return ev


_SCALAR = (str, int, float, bool)


def as_table(viz: Visualization) -> Visualization:
    """Re-express any visualization's rows as a plain table (every column present in every row).

    Used when a chart fails the encoding check, so the response stays valid and keeps its citations.
    """
    rows = viz.data.nodes if isinstance(viz.data, GraphData) else viz.data
    fields: list[str] = []
    for r in rows:
        for k, v in (r.model_extra or {}).items():
            if k != "cite_note" and k not in fields and isinstance(v, _SCALAR):
                fields.append(k)
    if not rows or not fields:
        return empty_kpi(viz.title, "No matching data found in ClinicalTrials.gov")
    out = []
    for r in rows:
        extra = dict(r.model_extra or {})
        missing = [f for f in fields if not isinstance(extra.get(f), _SCALAR)]
        for f in missing:
            extra[f] = "\u2014"
        if missing:
            extra["cite_note"] = (str(extra.get("cite_note", "")) + f" not reported: {', '.join(missing)}").strip()
        out.append(Datum(id=r.id, citations=r.citations, citations_total=r.citations_total, **extra))
    cols = [TableColumn(field=f, title=f.replace("_", " ").capitalize()) for f in fields]
    enc = Encoding(label=Channel(field=fields[0], type="nominal", title=cols[0].title))
    return make_viz(VizType.TABLE, viz.title, enc, out, VizMeta(columns=cols), viz.subtitle)


def verify_response(
    resp: VisualizeResponse,
    raw_by_nct: dict[str, dict],
    *,
    facts: dict[str, Any] | None = None,
    extra_allowed: Iterable[Any] = (),
    check_citations: bool = True,
) -> tuple[VisualizeResponse, list[Note]]:
    """Run all checks; returns the (possibly repaired) response and the notes it generated.

    The returned notes are also appended to ``resp.meta.notes``.
    """
    notes: list[Note] = []
    viz = resp.visualization
    if check_citations:
        viz, cnotes = verify_citations(viz, raw_by_nct)
        notes += cnotes
    try:
        check_encoding(viz)
    except EncodingError as exc:
        viz = as_table(viz)
        notes.append(
            note(
                NoteCode.VIZ_DOWNGRADED, f"The chart failed validation ({exc}); showing its data as a table.", "warning"
            )
        )
    resp = resp.model_copy(update={"visualization": viz})
    summary, snotes = verify_summary(resp.answer_summary, response_evidence(resp, facts, extra_allowed))
    notes += snotes
    meta = resp.meta.model_copy(update={"notes": [*resp.meta.notes, *notes]})
    resp = resp.model_copy(update={"answer_summary": summary, "meta": meta})
    resp = VisualizeResponse.model_validate(resp.model_dump(mode="json"))
    return resp, notes
