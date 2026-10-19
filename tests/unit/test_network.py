"""Co-occurrence graph pruning: what decides which entities and links are drawn."""

from __future__ import annotations


def test_floor_drops_single_trial_links_and_ranks_by_shared_trials():
    from ctviz.compute.network import CooccurrenceGraph

    g = CooccurrenceGraph()
    # hub: linked once each to 12 different drugs (noise); pair: A+B together in 9 trials
    for i in range(12):
        g.add_node("hub", "Hub", "drug", f"t{i}")
        g.add_node(f"x{i}", f"X{i}", "drug", f"t{i}")
        g.add_edge("hub", f"x{i}", "combined_in_arm", f"t{i}")
    for i in range(9):
        for k in ("a", "b"):
            g.add_node(k, k.upper(), "drug", f"p{i}")
        g.add_edge("a", "b", "combined_in_arm", f"p{i}")
    pruned = g.prune(max_nodes=2, max_edges=10, min_edge_weight=2)
    assert {n.key for n in pruned.nodes} == {"a", "b"}  # 12 one-trial links never outrank a real pair
    assert all(e.weight >= 2 for e in pruned.edges)


def test_drug_name_variants_merge_but_combinations_and_unknowns_stay():
    from ctviz.handlers.cohort import canonical_drug

    for variant in (
        "Pembrolizumab (MK-3475)",
        "pembrolizumab [KEYTRUDA]",
        "Pembrolizumab IV 200 mg",
        "MK-3475",
        "Pembrolizumab Injection [Keytruda]",
        "Keytruda",
    ):
        assert canonical_drug(variant) == "pembrolizumab", variant
    assert canonical_drug("Semaglutide 2.4 mg") == "semaglutide"
    for kept in ("Pembrolizumab, Eftilagimod alfa", "LITT + Pembrolizumab", "BNT327", "Placebo"):
        assert canonical_drug(kept) is None, kept  # a wrong merge is worse than a split
