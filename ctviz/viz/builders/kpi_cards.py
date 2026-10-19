"""kpi_cards: one card per row ``{label, value, kind, unit?, detail?}``; encoding label -> label, y -> value.

``kind`` tells the renderer how to format ``value``: number | text | date | enum | list (value is a list
of strings). Rows without a ``kind`` get one inferred from the value type (number / list / text).
"""

from __future__ import annotations

from typing import Any

from ctviz.api.schemas.viz import Channel, Datum, Encoding, Layout, Visualization, VizMeta, VizType
from ctviz.viz.base import make_viz

KPI_KINDS = ("number", "text", "date", "enum", "list")


def infer_kind(value: Any) -> str:
    """``number`` for int/float (not bool), ``list`` for lists, else ``text``."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return "number"
    if isinstance(value, list):
        return "list"
    return "text"


def kpi_cards(
    rows: list[Datum],
    *,
    title: str,
    subtitle: str | None = None,
    meta: VizMeta | None = None,
    columns: int | None = None,
) -> Visualization:
    """Build a kpi_cards visualization from rows that each carry ``label`` and ``value`` (and ``kind``).

    An explicit invalid ``kind`` raises; a missing one is inferred with :func:`infer_kind`. ``y`` is
    quantitative only when every card is a number.
    """
    for r in rows:
        extra = r.model_extra or {}
        kind = extra.get("kind")
        if kind is None:
            kind = infer_kind(extra.get("value"))
            setattr(r, "kind", kind)  # noqa: B010  (Datum allows extra fields)
        if kind not in KPI_KINDS:
            raise ValueError(f"kpi row {r.id} has invalid kind {kind!r}")
    meta = meta or VizMeta()
    if columns and meta.layout is None:
        meta.layout = Layout(columns=columns)
    numeric = bool(rows) and all((r.model_extra or {}).get("kind") == "number" for r in rows)
    enc = Encoding(
        label=Channel(field="label", type="nominal", title="Item"),
        y=Channel(field="value", type="quantitative" if numeric else "nominal", title="Value"),
        tooltip=["unit", "detail"],
    )
    return make_viz(VizType.KPI_CARDS, title, enc, rows, meta, subtitle)
