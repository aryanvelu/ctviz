"""bar_chart: categorical x, quantitative y, optional series (grouped/stacked/normalized)."""

from __future__ import annotations

from ctviz.api.schemas.viz import Channel, Datum, Encoding, Layout, Sort, Visualization, VizMeta, VizType
from ctviz.viz.base import make_viz


def bar_chart(
    rows: list[Datum],
    *,
    title: str,
    x: Channel,
    y: Channel,
    series: Channel | None = None,
    subtitle: str | None = None,
    mode: str = "grouped",
    orientation: str = "vertical",
    meta: VizMeta | None = None,
    tooltip: list[str] | None = None,
) -> Visualization:
    meta = meta or VizMeta()
    meta.layout = meta.layout or Layout(orientation=orientation, mode=mode)  # type: ignore[arg-type]
    if meta.sort is None and x.sort is not None:
        meta.sort = x.sort
    meta.units.setdefault(y.field, y.unit or "")
    enc = Encoding(x=x, y=y, series=series, tooltip=tooltip or [])
    return make_viz(VizType.BAR_CHART, title, enc, rows, meta, subtitle)


def ordered(values: list[str]) -> Sort:
    return Sort(values=values)
