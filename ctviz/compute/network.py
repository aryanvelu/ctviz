"""Co-occurrence network accumulation and pruning (pure; evidence objects are opaque here).

Handlers add nodes and edges per trial together with their evidence (``FieldRef`` objects in
practice). Edge weight is the number of *distinct trials* in which the two entities co-occur.
:meth:`CooccurrenceGraph.prune` keeps the graph readable: at most ``max_nodes`` nodes chosen by
degree (with a per-type quota so bipartite graphs keep both sides), at most ``max_edges`` edges
(raising the minimum edge weight until it fits), and no isolated nodes. Pinned nodes (the entities a
user asked about) are exempt: always kept, and their edges are never removed by the weight threshold.

Evidence objects are opaque: handlers pass cheap lazy handles and resolve them to citations only for
the nodes/edges that survive pruning.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodeAcc:
    key: str
    label: str
    type: str
    trials: set[str] = field(default_factory=set)
    evidence: list[Any] = field(default_factory=list)
    degree: int = 0


@dataclass
class EdgeAcc:
    source: str
    target: str
    type: str
    trials: set[str] = field(default_factory=set)
    evidence: list[Any] = field(default_factory=list)

    @property
    def weight(self) -> int:
        return len(self.trials)


@dataclass
class PrunedGraph:
    nodes: list[NodeAcc]
    edges: list[EdgeAcc]
    min_edge_weight: int
    total_nodes: int
    total_edges: int
    tie_truncated: bool = False  # edges of weight == min_edge_weight were cut to fit max_edges


class CooccurrenceGraph:
    """Accumulates entity nodes and co-occurrence edges across trials."""

    def __init__(self, directed: bool = False):
        self.directed = directed
        self.nodes: dict[str, NodeAcc] = {}
        self.edges: dict[tuple[str, str, str], EdgeAcc] = {}

    def add_node(self, key: str, label: str, node_type: str, trial: str, evidence: Any = None) -> NodeAcc:
        """Register an entity occurrence in ``trial`` (idempotent per trial for counting)."""
        node = self.nodes.get(key)
        if node is None:
            node = self.nodes[key] = NodeAcc(key=key, label=label, type=node_type)
        node.trials.add(trial)
        if evidence is not None:
            node.evidence.append(evidence)
        return node

    def add_edge(self, a: str, b: str, edge_type: str, trial: str, evidence: list[Any] | None = None) -> None:
        """Record that ``a`` and ``b`` co-occur in ``trial``. Self-loops are ignored."""
        if a == b:
            return
        if not self.directed and b < a:
            a, b = b, a
        k = (a, b, edge_type)
        edge = self.edges.get(k)
        if edge is None:
            edge = self.edges[k] = EdgeAcc(source=a, target=b, type=edge_type)
        edge.trials.add(trial)
        edge.evidence.extend(e for e in evidence or [] if e is not None)

    def prune(
        self,
        max_nodes: int = 40,
        max_edges: int = 300,
        min_edge_weight: int = 1,
        pinned: set[str] | frozenset[str] = frozenset(),
        ego: bool = False,
    ) -> PrunedGraph:
        """Select the most connected nodes and strongest edges; see module docstring.

        ``pinned`` keys (the entities the user asked about) are always kept, together with their edges
        to the other kept nodes; half of the remaining node budget goes to their strongest neighbours.
        The minimum edge weight is only raised for edges that do not touch a pinned node.
        With ``ego`` (the question is "what co-occurs WITH X"), the whole budget goes to the pinned nodes'
        direct neighbours, so the graph shows X's partners and the links among them, nothing else.
        """
        # Links below ``min_edge_weight`` (e.g. one shared trial) never decide which entities are shown.
        floor = max(1, min_edge_weight)
        strong = [e for e in self.edges.values() if e.weight >= floor]
        degree = _degrees(strong, set(self.nodes))
        strength = _strengths(strong, set(self.nodes))  # shared trials summed over links: rank by this
        pins = {k for k in pinned if k in self.nodes}
        connected = [n for n in self.nodes.values() if degree[n.key] > 0 and n.key not in pins]
        budget = max(0, max_nodes - len(pins))
        keep = set(pins)
        if pins and budget:
            to_pin: dict[str, int] = defaultdict(int)
            for e in strong:
                if e.source in pins and e.target not in pins:
                    to_pin[e.target] = max(to_pin[e.target], e.weight)
                elif e.target in pins and e.source not in pins:
                    to_pin[e.source] = max(to_pin[e.source], e.weight)
            ranked = sorted(to_pin, key=lambda k: (-to_pin[k], -strength[k], self.nodes[k].label))
            keep |= set(ranked[: budget if ego else budget // 2])
        rest = [] if ego and pins else [n for n in connected if n.key not in keep]
        keep |= _select_nodes(rest, strength, max_nodes - len(keep))
        candidates = [e for e in strong if e.source in keep and e.target in keep]
        pinned_edges = sorted(
            (e for e in candidates if e.source in pins or e.target in pins),
            key=lambda e: (-e.weight, e.source, e.target),
        )[:max_edges]
        others = [e for e in candidates if not (e.source in pins or e.target in pins)]
        weight = max(1, min_edge_weight)
        while True:
            edges = pinned_edges + [e for e in others if e.weight >= weight]
            if len(edges) <= max_edges:
                break
            weight += 1
        tie_truncated = False
        if weight > max(1, min_edge_weight) and len(edges) < max_edges:
            # The weight-1 tier did not fit as a whole (raising the threshold may even have emptied the
            # graph): fill the remaining budget from that tier, best-connected endpoints first, then by key.
            tier = sorted(
                (e for e in others if e.weight == weight - 1),
                key=lambda e: (-(degree[e.source] + degree[e.target]), e.source, e.target),
            )
            fill = tier[: max_edges - len(edges)]
            if fill:
                edges += fill
                weight -= 1
                tie_truncated = len(fill) < len(tier)
        edges.sort(key=lambda e: (-e.weight, e.source, e.target))
        final_degree = _degrees(edges, keep)
        nodes = [self.nodes[k] for k in keep if final_degree[k] > 0 or k in pins]
        for n in nodes:
            n.degree = final_degree[n.key]
        nodes.sort(key=lambda n: (-n.degree, -len(n.trials), n.label))
        return PrunedGraph(
            nodes=nodes,
            edges=edges,
            min_edge_weight=weight,
            total_nodes=len(self.nodes),
            total_edges=len(self.edges),
            tie_truncated=tie_truncated,
        )


def partners(graph: CooccurrenceGraph, key: str, limit: int = 5) -> list[tuple[str, int]]:
    """``(neighbour label, shared trials)`` for ``key``'s strongest links, over the whole (unpruned) graph."""
    out = []
    for e in graph.edges.values():
        if key in (e.source, e.target):
            other = e.target if e.source == key else e.source
            out.append((graph.nodes[other].label, e.weight))
    out.sort(key=lambda t: (-t[1], t[0]))
    return out[:limit]


def _strengths(edges, keys: set[str]) -> dict[str, int]:
    """Shared trials summed over each node's links (a hub of one-trial links ranks below a real partner)."""
    out: dict[str, int] = defaultdict(int)
    for e in edges:
        out[e.source] += e.weight
        out[e.target] += e.weight
    return {k: out.get(k, 0) for k in keys}


def _degrees(edges, keys: set[str]) -> dict[str, int]:
    """Number of distinct neighbours per node."""
    neigh: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        neigh[e.source].add(e.target)
        neigh[e.target].add(e.source)
    return {k: len(neigh.get(k, ())) for k in keys}


def _select_nodes(nodes: list[NodeAcc], degree: dict[str, int], max_nodes: int) -> set[str]:
    """Top nodes by ``degree`` (any per-node score; callers pass link strength) with a proportional per-type
    quota (each type gets at least a quarter)."""
    if max_nodes <= 0:
        return set()
    if len(nodes) <= max_nodes:
        return {n.key for n in nodes}
    by_type: dict[str, list[NodeAcc]] = defaultdict(list)
    for n in nodes:
        by_type[n.type].append(n)
    for group in by_type.values():
        group.sort(key=lambda n: (-degree[n.key], -len(n.trials), n.label))
    floor = max_nodes // (4 if len(by_type) > 1 else 1)
    quota = {t: min(len(g), max(floor, round(max_nodes * len(g) / len(nodes)))) for t, g in by_type.items()}
    # Trim the largest quotas until the total fits.
    while sum(quota.values()) > max_nodes:
        t = max(quota, key=lambda k: quota[k])
        quota[t] -= 1
    keep = {n.key for t, g in by_type.items() for n in g[: quota[t]]}
    # Spend any leftover budget on the best remaining nodes overall.
    rest = sorted((n for n in nodes if n.key not in keep), key=lambda n: (-degree[n.key], -len(n.trials), n.label))
    for n in rest[: max_nodes - len(keep)]:
        keep.add(n.key)
    return keep
