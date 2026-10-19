"""sankey: staged flow graph. Nodes {id,label,stage,group,value}; edges {source,target,weight,type,...}."""

from __future__ import annotations

from ctviz.api.schemas.viz import EdgeEncoding, Encoding, GraphData, NodeEncoding, Visualization, VizMeta, VizType
from ctviz.viz.base import make_viz


def sankey(
    graph: GraphData, *, title: str, stages: list[str], subtitle: str | None = None, meta: VizMeta | None = None
) -> Visualization:
    """Build a sankey; ``meta.stages[i]`` names the column of nodes with ``stage == i``."""
    meta = meta or VizMeta()
    meta.stages = stages
    meta.units.setdefault("weight", "participants")
    enc = Encoding(
        nodes=NodeEncoding(id="id", label="label", type="group", size="value"),
        edges=EdgeEncoding(source="source", target="target", weight="weight", type="type"),
        tooltip=["share_of_started", "group"],
    )
    return make_viz(VizType.SANKEY, title, enc, graph, meta, subtitle)
