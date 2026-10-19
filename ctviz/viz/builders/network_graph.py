"""network_graph: ``data = {nodes, edges}``.

Nodes ``{id, label, type, size, degree}``; edges ``{id, source, target, weight, type}``.
``meta.node_types`` is the legend and ``meta.edge_semantics`` says what an edge means.
"""

from __future__ import annotations

from typing import Any

from ctviz.api.schemas.viz import (
    EdgeEncoding,
    Encoding,
    GraphData,
    NodeEncoding,
    NodeType,
    Visualization,
    VizMeta,
    VizType,
)
from ctviz.viz.base import make_viz


def network_graph(
    data: GraphData,
    *,
    title: str,
    node_types: list[NodeType | dict[str, Any]],
    edge_semantics: str,
    subtitle: str | None = None,
    meta: VizMeta | None = None,
    tooltip: list[str] | None = None,
) -> Visualization:
    """Build a network graph spec (edge endpoints are validated against node ids)."""
    meta = meta or VizMeta()
    meta.node_types = [t if isinstance(t, NodeType) else NodeType(**t) for t in node_types]
    meta.edge_semantics = edge_semantics
    meta.units.setdefault("weight", "trials")
    meta.units.setdefault("size", "trials")
    enc = Encoding(nodes=NodeEncoding(), edges=EdgeEncoding(), tooltip=tooltip or ["degree"])
    return make_viz(VizType.NETWORK_GRAPH, title, enc, data, meta, subtitle)
