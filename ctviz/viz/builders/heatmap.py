"""heatmap: categorical x and y; ``series`` is the quantitative cell value.

Rows are complete (every x × y cell present). Cells with value 0 carry ``is_gap: true`` and no
citations; ``meta.highlight_zero`` asks renderers to style them distinctly.
"""

from __future__ import annotations

from ctviz.api.schemas.viz import Channel, Datum, Encoding, Visualization, VizMeta, VizType
from ctviz.viz.base import make_viz


def heatmap(
    rows: list[Datum],
    *,
    title: str,
    x: Channel,
    y: Channel,
    value: Channel,
    subtitle: str | None = None,
    meta: VizMeta | None = None,
    highlight_zero: bool = True,
    tooltip: list[str] | None = None,
) -> Visualization:
    """Build a heatmap; ``value`` becomes ``encoding.series`` (quantitative)."""
    if value.type != "quantitative":
        raise ValueError("heatmap cell value must be quantitative")
    meta = meta or VizMeta()
    meta.units.setdefault(value.field, value.unit or "")
    meta.highlight_zero = highlight_zero
    enc = Encoding(x=x, y=y, series=value, tooltip=tooltip or [])
    return make_viz(VizType.HEATMAP, title, enc, rows, meta, subtitle)
