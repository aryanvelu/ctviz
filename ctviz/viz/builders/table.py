"""table: flat rows with an explicit column list in ``meta.columns`` (the first column is the row label).

Every declared column must be present and non-null in every row (``check_encoding`` enforces this), so
handlers declare only uniform columns and pass optional per-row cells as ``tooltip`` fields.
"""

from __future__ import annotations

from ctviz.api.schemas.viz import Channel, Datum, Encoding, TableColumn, Visualization, VizMeta, VizType
from ctviz.viz.base import make_viz


def col(field: str, title: str, type_: str = "nominal", fmt: str | None = None) -> TableColumn:
    """Shorthand for a :class:`TableColumn`."""
    return TableColumn(field=field, title=title, type=type_, format=fmt)  # type: ignore[arg-type]


def table(
    rows: list[Datum],
    *,
    title: str,
    columns: list[TableColumn],
    subtitle: str | None = None,
    meta: VizMeta | None = None,
    tooltip: list[str] | None = None,
) -> Visualization:
    """Build a table. ``columns`` must be filled in every row; optional cells go in ``tooltip``."""
    if not columns:
        raise ValueError("table needs at least one column")
    meta = meta or VizMeta()
    meta.columns = columns
    first = columns[0]
    enc = Encoding(label=Channel(field=first.field, type=first.type, title=first.title), tooltip=tooltip or [])
    return make_viz(VizType.TABLE, title, enc, rows, meta, subtitle)
