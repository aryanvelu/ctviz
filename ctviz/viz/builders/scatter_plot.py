"""scatter_plot: one point per datum; quantitative/temporal x and y, optional series and size."""

from __future__ import annotations

from ctviz.api.schemas.viz import Channel, Datum, Encoding, Visualization, VizMeta, VizType
from ctviz.viz.base import make_viz


def scatter_plot(
    rows: list[Datum],
    *,
    title: str,
    x: Channel,
    y: Channel,
    series: Channel | None = None,
    size: Channel | None = None,
    label: Channel | None = None,
    subtitle: str | None = None,
    meta: VizMeta | None = None,
    tooltip: list[str] | None = None,
) -> Visualization:
    """Build a scatter plot."""
    meta = meta or VizMeta()
    meta.units.setdefault(y.field, y.unit or "")
    enc = Encoding(x=x, y=y, series=series, size=size, label=label, tooltip=tooltip or [])
    return make_viz(VizType.SCATTER_PLOT, title, enc, rows, meta, subtitle)
