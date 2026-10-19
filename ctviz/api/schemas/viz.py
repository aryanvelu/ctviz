"""Visualization specification: the frontend rendering contract.

A renderer reads ``visualization.type``, then for each channel in ``encoding`` looks up
``channel.field`` in every element of ``data`` (rows), or in ``data.nodes`` / ``data.edges`` for
graph types. Every datum has an ``id`` and a ``citations`` list.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ctviz.api.schemas.citations import Citation


class VizType(StrEnum):
    KPI_CARDS = "kpi_cards"
    TABLE = "table"
    BAR_CHART = "bar_chart"
    HISTOGRAM = "histogram"
    TIME_SERIES = "time_series"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    PIE_CHART = "pie_chart"
    NETWORK_GRAPH = "network_graph"
    SANKEY = "sankey"
    TIMELINE = "timeline"
    SURVIVAL_CURVE = "survival_curve"
    THRESHOLD_LINE = "threshold_line"
    POWER_CURVE = "power_curve"
    FOREST_PLOT = "forest_plot"
    GEO_MAP = "geo_map"
    ELIGIBILITY_CHECKLIST = "eligibility_checklist"


GRAPH_TYPES = frozenset({VizType.NETWORK_GRAPH, VizType.SANKEY})

FieldType = Literal["quantitative", "nominal", "ordinal", "temporal"]


class Sort(BaseModel):
    """Either sort by a field (``by`` + ``order``) or use an explicit category ``values`` order."""

    by: str | None = None
    order: Literal["asc", "desc"] | None = None
    values: list[str] | None = None


class Scale(BaseModel):
    type: Literal["linear", "log", "sqrt", "time", "band"] = "linear"
    domain: list[float | str] | None = None
    zero: bool | None = None


class Channel(BaseModel):
    """Maps one data field to one visual channel."""

    field: str
    type: FieldType
    title: str
    unit: str | None = Field(default=None, description="e.g. 'trials', 'participants', '%', 'days'")
    format: str | None = Field(default=None, description="d3-format number format, e.g. ',d', '.1%'")
    time_format: str | None = Field(default=None, description="d3-time-format for temporal fields, e.g. '%Y'")
    sort: Sort | None = None
    scale: Scale | None = None


class NodeEncoding(BaseModel):
    id: str = "id"
    label: str = "label"
    type: str | None = "type"
    size: str | None = "size"


class EdgeEncoding(BaseModel):
    source: str = "source"
    target: str = "target"
    weight: str | None = "weight"
    type: str | None = "type"


class Encoding(BaseModel):
    """Only the channels relevant to the visualization type are set."""

    x: Channel | None = None
    x2: Channel | None = Field(default=None, description="Range end (histogram bin_end, timeline end).")
    y: Channel | None = None
    y2: Channel | None = None
    series: Channel | None = Field(
        default=None, description="Color/legend grouping. For heatmap it is the quantitative cell value."
    )
    size: Channel | None = None
    label: Channel | None = Field(default=None, description="Text label (kpi_cards, checklist, table rows).")
    error_low: Channel | None = None
    error_high: Channel | None = None
    nodes: NodeEncoding | None = None
    edges: EdgeEncoding | None = None
    tooltip: list[str] = Field(default_factory=list, description="Extra datum fields to show on hover.")


class Datum(BaseModel):
    """One row / node / edge. Fields named by the encoding appear as extra keys."""

    model_config = ConfigDict(extra="allow")

    id: str
    nct_ids: list[str] | None = Field(
        default=None,
        description="Every trial behind this datum (distinct, plain NCT ids). For a count datum "
        "computed from retrieved records, len(nct_ids) == the count.",
    )
    query_url: str | None = Field(
        default=None, description="For registry_counts: the ClinicalTrials.gov search whose totalCount is this value."
    )
    citations: list[Citation] = Field(default_factory=list, description="Up to 3 full excerpts.")
    citations_total: int | None = Field(
        default=None, description="Present when citations were truncated; full list via the citations endpoint."
    )


class GraphData(BaseModel):
    nodes: list[Datum]
    edges: list[Datum]


class Threshold(BaseModel):
    axis: Literal["x", "y"] = "y"
    value: float
    label: str
    source: Literal["registry", "convention", "user"] = "convention"
    citations: list[Citation] = Field(default_factory=list, description="Evidence when source is 'registry'.")


class ReferenceLine(BaseModel):
    """Forest plot 'no effect' line: 1 for ratios (log axis), 0 for differences (linear axis)."""

    value: float
    label: str = "No effect"


class GeoSpec(BaseModel):
    """geo_map: country-level rows use region_field; site-level rows use lat_field/lon_field."""

    level: Literal["country", "site"]
    region_field: str | None = None
    value_field: str | None = None
    lat_field: str | None = None
    lon_field: str | None = None


class TimeAxis(BaseModel):
    kind: Literal["calendar", "relative"]
    unit: Literal["days", "weeks", "months", "years"] | None = Field(
        default=None, description="Unit of numeric offsets when kind='relative'."
    )
    origin: str | None = Field(default=None, description="What offset 0 means, e.g. 'baseline'.")


class TimelineLane(BaseModel):
    """One lane of a timeline. Rows belong to a lane via their encoding.series field value."""

    name: str
    axis: TimeAxis
    start_field: str
    end_field: str | None = Field(default=None, description="Missing in a row = point event or still ongoing.")
    approximated: bool = False


class NodeType(BaseModel):
    type: str
    label: str
    count: int | None = None


class Band(BaseModel):
    axis: Literal["x", "y"] = "y"
    low: float
    high: float
    label: str


class Approximation(BaseModel):
    is_approximation: bool = False
    method: str | None = None
    assumptions: list[str] = Field(default_factory=list)


class Truncation(BaseModel):
    top_n: int
    other_bucket: bool = False
    omitted_count: int = 0


class Layout(BaseModel):
    orientation: Literal["vertical", "horizontal"] = "vertical"
    mode: Literal["grouped", "stacked", "normalized"] = "grouped"
    donut: bool = False
    columns: int | None = None


class TableColumn(BaseModel):
    field: str
    title: str
    type: FieldType = "nominal"
    format: str | None = None


class VizMeta(BaseModel):
    """Rendering hints. Every key is documented here; undocumented keys are rejected.

    ``stats`` holds extra display-only figures (e.g. dropout rate per arm, minimum detectable
    effect) and ``diagnostics`` explains how the chart was computed (e.g. which measures the
    analysis agent selected). Renderers may show both as plain key/value lists.
    """

    model_config = ConfigDict(extra="forbid")

    units: dict[str, str] = Field(default_factory=dict)
    sort: Sort | None = None
    time_granularity: Literal["year", "quarter", "month", "day"] | None = None
    grouping: dict[str, Any] | None = None
    layout: Layout | None = None
    thresholds: list[Threshold] = Field(default_factory=list)
    bands: list[Band] = Field(default_factory=list)
    truncation: Truncation | None = None
    approximation: Approximation = Field(default_factory=Approximation)
    columns: list[TableColumn] | None = Field(default=None, description="table only")
    summary: dict[str, Any] | None = Field(default=None, description="histogram stats / checklist counts")
    node_types: list[NodeType] | None = Field(default=None, description="network_graph legend")
    edge_semantics: str | None = Field(default=None, description="network_graph: what an edge means")
    network_kind: str | None = Field(default=None, description="network_graph: sponsor_drug, drug_drug, ...")
    directed: bool | None = Field(default=None, description="network_graph: edges have a direction")
    min_edge_weight: int | None = Field(default=None, description="network_graph: weakest edge kept")
    pinned: list[str] | None = Field(default=None, description="network_graph: node ids the user named")
    network_layout: Literal["force", "ego"] | None = Field(
        default=None, description="network_graph: 'ego' = every node links to the pinned node; draw it centered"
    )
    stages: list[str] | None = Field(default=None, description="sankey stage names, index = node.stage")
    time_axis: TimeAxis | None = Field(default=None, description="timeline/time_series with one axis kind")
    lanes: list[TimelineLane] | None = Field(default=None, description="timeline with several lanes")
    geo: GeoSpec | None = Field(default=None, description="geo_map")
    reference_line: ReferenceLine | None = Field(default=None, description="forest_plot")
    highlight_zero: bool | None = Field(default=None, description="heatmap: emphasize is_gap cells")
    stats: dict[str, Any] = Field(default_factory=dict, description="Extra display-only figures.")
    diagnostics: dict[str, Any] = Field(default_factory=dict, description="How the chart was computed.")


class Visualization(BaseModel):
    type: VizType
    title: str
    subtitle: str | None = None
    encoding: Encoding
    data: list[Datum] | GraphData
    meta: VizMeta = Field(default_factory=VizMeta)
