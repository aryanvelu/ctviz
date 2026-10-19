"""time_series: temporal x (bucket start ISO date), quantitative y, optional series lines."""

from __future__ import annotations

from ctviz.api.schemas.viz import Channel, Datum, Encoding, Scale, Visualization, VizMeta, VizType
from ctviz.viz.base import make_viz


def time_series(
    rows: list[Datum],
    *,
    title: str,
    x: Channel,
    y: Channel,
    granularity: str,
    series: Channel | None = None,
    subtitle: str | None = None,
    meta: VizMeta | None = None,
    tooltip: list[str] | None = None,
) -> Visualization:
    """Build a time series; ``x`` must be a temporal channel whose values are ISO dates."""
    if x.type != "temporal":
        raise ValueError("time_series x channel must be temporal")
    meta = meta or VizMeta()
    meta.time_granularity = granularity  # type: ignore[assignment]
    meta.units.setdefault(y.field, y.unit or "")
    if x.scale is None:
        x = x.model_copy(update={"scale": Scale(type="time")})
    enc = Encoding(x=x, y=y, series=series, tooltip=tooltip or [])
    return make_viz(VizType.TIME_SERIES, title, enc, rows, meta, subtitle)
