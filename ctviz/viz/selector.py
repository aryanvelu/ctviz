"""Deterministic downgrade rules and alternative suggestions applied after a handler builds a chart."""

from __future__ import annotations

from ctviz.api.schemas.plan import Intent
from ctviz.api.schemas.response import Alternative, Note, NoteCode
from ctviz.api.schemas.viz import Channel, Encoding, GraphData, Visualization, VizMeta, VizType
from ctviz.core.intents import get_intent_spec
from ctviz.viz.base import note

_REASONS = {
    VizType.TABLE: "all values as a sortable table",
    VizType.BAR_CHART: "compare categories side by side",
    VizType.PIE_CHART: "share of the whole",
    VizType.KPI_CARDS: "headline numbers only",
    VizType.HEATMAP: "two-way matrix view",
    VizType.SCATTER_PLOT: "one point per trial",
    VizType.THRESHOLD_LINE: "rates against a threshold",
    VizType.FOREST_PLOT: "effect estimates with confidence intervals",
    VizType.NETWORK_GRAPH: "relationships between entities",
    VizType.TIMELINE: "events on a time axis",
}


def alternatives_for(intent: Intent, current: VizType) -> list[Alternative]:
    spec = get_intent_spec(intent)
    return [
        Alternative(type=t, reason=_REASONS.get(t, t.value.replace("_", " ")))
        for t in spec.compatible_viz
        if t != current
    ]


def _empty(viz: Visualization) -> bool:
    if isinstance(viz.data, GraphData):
        return not viz.data.nodes
    return not viz.data


def empty_kpi(title: str, message: str) -> Visualization:
    """A renderable 'nothing found' card."""
    from ctviz.core.provenance import Dataset

    ds = Dataset()
    ds.emit({"label": "Result", "value": message, "kind": "text"}, [])
    return Visualization(
        type=VizType.KPI_CARDS,
        title=title,
        encoding=Encoding(
            label=Channel(field="label", type="nominal", title="Item"),
            y=Channel(field="value", type="nominal", title="Value"),
        ),
        data=ds.rows,
        meta=VizMeta(),
    )


def apply_downgrades(viz: Visualization) -> tuple[Visualization, list[Note]]:
    """Return a renderable visualization plus notes describing any downgrade."""
    notes: list[Note] = []
    if _empty(viz):
        notes.append(note(NoteCode.ZERO_RESULTS, "No data matched; showing a summary card instead.", "warning"))
        return empty_kpi(viz.title, "No matching data found in ClinicalTrials.gov"), notes
    if viz.type is VizType.TIME_SERIES and isinstance(viz.data, list):
        buckets = {(r.model_extra or {}).get(viz.encoding.x.field) for r in viz.data} if viz.encoding.x else set()
        if len(buckets) < 3:
            viz = viz.model_copy(update={"type": VizType.BAR_CHART})
            notes.append(note(NoteCode.VIZ_DOWNGRADED, "Fewer than 3 time buckets; shown as a bar chart."))
    if viz.type is VizType.NETWORK_GRAPH and isinstance(viz.data, GraphData) and len(viz.data.edges) < 2:
        notes.append(
            note(NoteCode.VIZ_DOWNGRADED, "Too few relationships for a meaningful network; showing nodes as a table.")
        )
        viz = viz.model_copy(
            update={
                "type": VizType.TABLE,
                "data": viz.data.nodes,
                "encoding": Encoding(label=Channel(field="label", type="nominal", title="Entity")),
            }
        )
    return viz, notes
