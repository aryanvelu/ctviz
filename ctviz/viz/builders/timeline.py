"""timeline: Gantt-style rows.

Single axis (:func:`timeline`): rows ``{task, start, end?, lane, category, status, is_estimated, point}``;
``x`` = start, ``x2`` = end (missing → ongoing / point event), ``y`` = task, ``series`` = category;
``meta.time_axis`` says whether start/end are calendar dates or relative offsets.

Several lanes with different axes (:func:`lane_timeline`): ``y`` = row label, ``series`` = lane name;
each lane's start/end fields and axis kind are in ``meta.lanes`` (no channel mixes dates and offsets).
"""

from __future__ import annotations

from ctviz.api.schemas.viz import Channel, Datum, Encoding, TimeAxis, TimelineLane, Visualization, VizMeta, VizType
from ctviz.viz.base import make_viz

TASK = Channel(field="task", type="nominal", title="Item")
START = Channel(field="start", type="temporal", title="Start", time_format="%Y-%m-%d")
END = Channel(field="end", type="temporal", title="End", time_format="%Y-%m-%d")
CATEGORY = Channel(field="category", type="nominal", title="Category")
LANE = Channel(field="lane", type="nominal", title="Lane")


def timeline(
    rows: list[Datum],
    *,
    title: str,
    task: Channel = TASK,
    start: Channel = START,
    end: Channel = END,
    category: Channel = CATEGORY,
    subtitle: str | None = None,
    meta: VizMeta | None = None,
    tooltip: list[str] | None = None,
    kind: str = "calendar",
    unit: str | None = None,
    origin: str | None = None,
) -> Visualization:
    """Build a timeline on one axis: calendar dates (``kind='calendar'``) or relative offsets."""
    meta = meta or VizMeta()
    meta.time_axis = meta.time_axis or TimeAxis(kind=kind, unit=unit, origin=origin)  # type: ignore[arg-type]
    if kind == "calendar":
        meta.time_granularity = meta.time_granularity or "day"
    enc = Encoding(
        x=start, x2=end, y=task, series=category, tooltip=tooltip or ["lane", "status", "is_estimated", "point"]
    )
    return make_viz(VizType.TIMELINE, title, enc, rows, meta, subtitle)


def lane_timeline(
    rows: list[Datum],
    *,
    title: str,
    lanes: list[TimelineLane],
    label: Channel,
    lane: Channel = LANE,
    subtitle: str | None = None,
    meta: VizMeta | None = None,
    tooltip: list[str] | None = None,
) -> Visualization:
    """Build a multi-lane timeline; every row's ``lane.field`` value must name one of ``lanes``."""
    names = {ln.name for ln in lanes}
    for r in rows:
        if (r.model_extra or {}).get(lane.field) not in names:
            raise ValueError(f"timeline row {r.id} is not in a declared lane")
    meta = meta or VizMeta()
    meta.lanes = lanes
    enc = Encoding(y=label, series=lane, tooltip=tooltip or [])
    return make_viz(VizType.TIMELINE, title, enc, rows, meta, subtitle)
