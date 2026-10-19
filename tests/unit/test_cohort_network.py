"""entity_network (all five kinds)."""

from __future__ import annotations

import re

import pytest

from ctviz.api.schemas import Intent, VizType
from ctviz.api.schemas.viz import GraphData
from ctviz.compute.cohort import is_placebo
from tests.unit.cohort_helpers import assert_valid, run

_ARM = re.compile(r"\.armGroups\[(\d+)\]\.interventionNames\[\d+\]$")


def _check_graph(viz, max_nodes=40):
    assert viz.type is VizType.NETWORK_GRAPH and isinstance(viz.data, GraphData)
    nodes, edges = viz.data.nodes, viz.data.edges
    assert 0 < len(nodes) <= max_nodes and 0 < len(edges) <= 300
    ids = {n.id for n in nodes}
    degree = {i: set() for i in ids}
    for e in edges:
        x = e.model_extra
        assert {"source", "target", "weight", "type"} <= set(x)
        assert x["source"] in ids and x["target"] in ids and x["weight"] >= 1
        degree[x["source"]].add(x["target"])
        degree[x["target"]].add(x["source"])
        # weight = distinct trials; every shared trial is cited with both fields (2 refs per trial)
        trials = {c.nct_id for c in e.citations}
        if e.citations_total is None:
            assert len(trials) == x["weight"]
    for n in nodes:
        x = n.model_extra
        assert {"label", "type", "size", "degree"} <= set(x)
        assert x["degree"] == len(degree[n.id]) > 0
    assert viz.meta.edge_semantics and viz.meta.node_types


async def test_sponsor_drug_network_default(make_ctx, cohort):
    c = cohort("obesity_since2020")
    res = await run(Intent.ENTITY_NETWORK, make_ctx(Intent.ENTITY_NETWORK, cohorts=[c]))
    viz = res.visualization
    _check_graph(viz)
    assert res.facts["network_kind"] == "sponsor_drug"
    assert {n.model_extra["type"] for n in viz.data.nodes} == {"sponsor", "drug"}
    for e in viz.data.edges:
        tails = {cit.field_path.rsplit(".", 1)[-1] for cit in e.citations}
        assert tails == {"name"}  # leadSponsor.name + interventions[i].name
        assert any("leadSponsor" in cit.field_path for cit in e.citations)
        assert any(".interventions[" in cit.field_path for cit in e.citations)
    assert_valid(viz, c)
    assert res.facts["nodes_derived"] == len(viz.data.nodes) and res.facts["edges_derived"] == len(viz.data.edges)


async def test_drug_drug_network_melanoma(make_ctx, cohort):
    c = cohort("melanoma_ph2_3")
    res = await run(Intent.ENTITY_NETWORK, make_ctx(Intent.ENTITY_NETWORK, cohorts=[c], network_kind="drug_drug"))
    viz = res.visualization
    _check_graph(viz)
    labels = {n.id: n.model_extra["label"].lower() for n in viz.data.nodes}
    assert not any(is_placebo(lbl) for lbl in labels.values())
    assert "nivolumab" in labels.values() and "ipilimumab" in labels.values()
    by_id = {s.nct_id: s for s in c.studies}
    for e in viz.data.edges:
        # each cited entry is an arm interventionNames item; both drugs appear in the same arm
        per_arm: dict[tuple[str, int], int] = {}
        for cit in e.citations:
            m = _ARM.search(cit.field_path)
            assert m, cit.field_path
            assert cit.excerpt.split(":")[0] in ("Drug", "Biological")
            per_arm[(cit.nct_id, int(m.group(1)))] = per_arm.get((cit.nct_id, int(m.group(1))), 0) + 1
            assert cit.nct_id in by_id
        if e.citations_total is None:
            assert all(v >= 2 for v in per_arm.values())
    assert_valid(viz, c)
    top = res.facts["top_edges"][0]
    assert top["weight"] == max(e.model_extra["weight"] for e in viz.data.edges)


@pytest.mark.parametrize(
    "kind,types",
    [
        ("drug_condition", {"drug", "condition"}),
        ("sponsor_collaborator", {"lead_sponsor", "collaborator"}),
        ("site_investigator", {"investigator", "site"}),
    ],
)
async def test_other_network_kinds(make_ctx, cohort, kind, types):
    c = cohort("melanoma_ph2_3")
    res = await run(Intent.ENTITY_NETWORK, make_ctx(Intent.ENTITY_NETWORK, cohorts=[c], network_kind=kind))
    viz = res.visualization
    _check_graph(viz)
    assert {n.model_extra["type"] for n in viz.data.nodes} <= types
    assert viz.meta.network_kind == kind
    assert_valid(viz, c)


async def test_network_auto_raises_min_weight(make_ctx, cohort):
    c = cohort("melanoma_ph2_3")
    ctx = make_ctx(Intent.ENTITY_NETWORK, cohorts=[c], network_kind="site_investigator")
    res = await run(Intent.ENTITY_NETWORK, ctx)
    assert res.facts["edges_total_derived"] > 300
    assert len(res.visualization.data.edges) <= 300
    assert any(n.code.value == "TRUNCATED" for n in res.notes)
