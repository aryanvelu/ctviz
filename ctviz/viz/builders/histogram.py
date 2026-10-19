"""histogram: pre-binned counts. ``x`` = bin start, ``x2`` = bin end (exclusive, null for open-ended)."""

from __future__ import annotations

from typing import Any

from ctviz.api.schemas.viz import Channel, Datum, Encoding, Visualization, VizMeta, VizType
from ctviz.viz.base import make_viz


def histogram(
    rows: list[Datum],
    *,
    title: str,
    x: Channel,
    x2: Channel,
    y: Channel,
    series: Channel | None = None,
    label: Channel | None = None,
    subtitle: str | None = None,
    meta: VizMeta | None = None,
    summary: dict[str, Any] | None = None,
    tooltip: list[str] | None = None,
) -> Visualization:
    """Build a histogram from rows that already carry bin bounds and counts."""
    meta = meta or VizMeta()
    meta.units.setdefault(y.field, y.unit or "")
    meta.units.setdefault(x.field, x.unit or "")
    if summary is not None:
        meta.summary = summary
    enc = Encoding(x=x, x2=x2, y=y, series=series, label=label, tooltip=tooltip or [])
    return make_viz(VizType.HISTOGRAM, title, enc, rows, meta, subtitle)
