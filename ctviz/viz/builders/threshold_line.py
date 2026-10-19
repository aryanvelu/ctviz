"""threshold_line: ordered categories (e.g. dose levels) with a rate per category and threshold lines."""

from __future__ import annotations

from ctviz.api.schemas.viz import Channel, Datum, Encoding, Threshold, Visualization, VizMeta, VizType
from ctviz.viz.base import make_viz


def threshold_line(
    rows: list[Datum],
    *,
    title: str,
    x: Channel,
    y: Channel,
    thresholds: list[Threshold],
    series: Channel | None = None,
    subtitle: str | None = None,
    meta: VizMeta | None = None,
    tooltip: list[str] | None = None,
) -> Visualization:
    """``x.sort`` carries the category order; thresholds are drawn as horizontal rules."""
    meta = meta or VizMeta()
    meta.thresholds = [*meta.thresholds, *thresholds]
    if meta.sort is None and x.sort is not None:
        meta.sort = x.sort
    meta.units.setdefault(y.field, y.unit or "")
    enc = Encoding(x=x, y=y, series=series, tooltip=tooltip or [])
    return make_viz(VizType.THRESHOLD_LINE, title, enc, rows, meta, subtitle)
