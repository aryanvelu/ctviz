"""Contract between the orchestrator and intent handlers, plus builder helpers.

Flow: the orchestrator fetches cohorts / full records according to the intent's ``IntentSpec``,
then calls the registered :class:`IntentHandler`. A handler computes cited datums with
:mod:`ctviz.core.provenance` and turns them into a :class:`Visualization` with :func:`make_viz`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ctviz.api.schemas.plan import Intent, QueryPlan
from ctviz.api.schemas.request import VisualizeRequest
from ctviz.api.schemas.response import Alternative, Note, NoteCode
from ctviz.api.schemas.viz import (
    GRAPH_TYPES,
    Datum,
    EdgeEncoding,
    Encoding,
    GraphData,
    NodeEncoding,
    Visualization,
    VizMeta,
    VizType,
)
from ctviz.config import Settings
from ctviz.core.provenance import FieldRef, Study
from ctviz.ctgov.requests import SearchSpec


@dataclass
class CohortData:
    """One fetched cohort (comparison scope has several)."""

    label: str
    spec: SearchSpec
    studies: list[Study]
    total_count: int
    sampled: bool = False
    sampling_method: str | None = None
    search_url: str = ""
    # "all_records" (every match retrieved), "sample" (part of them), or "registry_counts" (only page 1 kept;
    # handlers must count each bucket server-side, see ctviz/handlers/registry_counts.py)
    counting: str = "all_records"
    snapshot_at: float | None = None  # epoch seconds: members read from a stored cohort snapshot

    def __post_init__(self) -> None:
        if self.sampled and self.counting == "all_records":
            self.counting = "sample"


@dataclass
class HandlerContext:
    request: VisualizeRequest
    plan: QueryPlan
    settings: Settings
    client: Any  # ctviz.ctgov.client.CTGovClient (Any to avoid import cycles in tests)
    db: Any  # ctviz.store.db.Database
    llm: Any  # ctviz.agents.llm.LLM
    cohorts: list[CohortData] = field(default_factory=list)
    trials: list[Study] = field(default_factory=list)  # full records for single-trial / comparison intents
    graph: Any = None  # ctviz.store.graph.GraphStore when available

    @property
    def studies(self) -> list[Study]:
        """All cohort studies (deduplicated), or the trials for single-trial scope."""
        if self.trials:
            return self.trials
        seen: dict[str, Study] = {}
        for c in self.cohorts:
            for s in c.studies:
                seen.setdefault(s.nct_id, s)
        return list(seen.values())

    @property
    def persona(self) -> str:
        return self.plan.persona.value


@dataclass
class HandlerResult:
    visualization: Visualization
    notes: list[Note] = field(default_factory=list)
    alternatives: list[Alternative] = field(default_factory=list)
    studies_analyzed: int | None = None
    studies_with_results: int | None = None
    facts: dict[str, Any] = field(default_factory=dict)  # compact facts for the Synthesis agent
    full_refs: dict[str, list[FieldRef]] = field(default_factory=dict)  # datum id -> all evidence


class IntentHandler(Protocol):
    async def __call__(self, ctx: HandlerContext) -> HandlerResult: ...


_HANDLERS: dict[Intent, IntentHandler] = {}


def register(intent: Intent) -> Callable[[IntentHandler], IntentHandler]:
    """Decorator: ``@register(Intent.PHASE_DISTRIBUTION) async def handle(ctx): ...``"""

    def deco(fn: IntentHandler) -> IntentHandler:
        _HANDLERS[intent] = fn
        return fn

    return deco


def get_handler(intent: Intent) -> IntentHandler | None:
    return _HANDLERS.get(intent)


def registered_intents() -> list[Intent]:
    return list(_HANDLERS)


def note(code: NoteCode, message: str, severity: str = "info", affects: list[str] | None = None) -> Note:
    return Note(code=code, message=message, severity=severity, affects=affects or ["*"])  # type: ignore[arg-type]


class EncodingError(ValueError):
    pass


def check_encoding(viz: Visualization) -> None:
    """Raise if an encoded field is missing or null in any datum (renderers must never guess)."""
    enc = viz.encoding

    def require(datum, fields: set[str], what: str) -> None:  # noqa: ANN001
        extra = datum.model_extra or {}
        missing = sorted(f for f in fields if extra.get(f) is None)
        if missing:
            raise EncodingError(f"{what} {datum.id} missing encoded fields {missing}")

    if viz.type in GRAPH_TYPES:
        if not isinstance(viz.data, GraphData):
            raise EncodingError(f"{viz.type} requires graph data")
        ne = enc.nodes or NodeEncoding()
        ee = enc.edges or EdgeEncoding()
        ids = {n.id for n in viz.data.nodes}
        if len(ids) != len(viz.data.nodes):
            raise EncodingError("duplicate node ids")
        for n in viz.data.nodes:
            require(n, {f for f in (ne.label,) if f}, "node")
        for e in viz.data.edges:
            require(e, {f for f in (ee.source, ee.target, ee.weight) if f}, "edge")
            extra = e.model_extra or {}
            if extra.get(ee.source) not in ids or extra.get(ee.target) not in ids:
                raise EncodingError(f"edge {e.id} references unknown node")
        return
    if isinstance(viz.data, GraphData):
        raise EncodingError(f"{viz.type} requires row data")
    for ch in (enc.x, enc.x2, enc.y, enc.y2, enc.series, enc.size, enc.label, enc.error_low, enc.error_high):
        if ch is not None and ch.type == "temporal" and ch.format is not None:
            raise EncodingError(f"temporal channel {ch.field} uses format; dates take time_format")
    channels = [enc.x, enc.y, enc.series, enc.size, enc.label]
    needed = {c.field for c in channels if c is not None}
    if viz.meta.columns:
        needed |= {c.field for c in viz.meta.columns}
    ids = [r.id for r in viz.data]
    if len(set(ids)) != len(ids):
        raise EncodingError("duplicate datum ids")
    for row in viz.data:
        require(row, needed, "datum")


def make_viz(
    vtype: VizType,
    title: str,
    encoding: Encoding,
    data: list[Datum] | GraphData,
    meta: VizMeta | None = None,
    subtitle: str | None = None,
) -> Visualization:
    """Assemble and validate a Visualization."""
    viz = Visualization(
        type=vtype, title=title, subtitle=subtitle, encoding=encoding, data=data, meta=meta or VizMeta()
    )
    check_encoding(viz)
    return viz
