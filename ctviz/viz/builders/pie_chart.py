"""pie_chart: ``label`` = category, ``y`` = value (share of the whole). Single series only."""

from __future__ import annotations

from ctviz.api.schemas.viz import Channel, Datum, Encoding, Layout, Visualization, VizMeta, VizType
from ctviz.viz.base import make_viz


def pie_chart(
    rows: list[Datum],
    *,
    title: str,
    label: Channel,
    value: Channel,
    donut: bool = True,
    subtitle: str | None = None,
    meta: VizMeta | None = None,
    tooltip: list[str] | None = None,
) -> Visualization:
    """Build a pie/donut chart."""
    meta = meta or VizMeta()
    meta.layout = meta.layout or Layout(donut=donut)
    meta.units.setdefault(value.field, value.unit or "")
    enc = Encoding(label=label, y=value, tooltip=tooltip or [])
    return make_viz(VizType.PIE_CHART, title, enc, rows, meta, subtitle)
