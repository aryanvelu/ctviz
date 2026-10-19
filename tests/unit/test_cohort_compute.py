"""Pure helpers in ctviz.compute.cohort and ctviz.compute.network."""

from __future__ import annotations

import datetime as dt

import pytest

from ctviz.compute.cohort import (
    LabelPicker,
    bucket_label,
    bucket_range,
    bucket_start,
    enrollment_bin,
    enrollment_bin_label,
    is_placebo,
    median,
    normalize_name,
    parse_partial_date,
    split_arm_intervention,
    supports_granularity,
)
from ctviz.compute.network import CooccurrenceGraph


@pytest.mark.parametrize(
    "text,iso,precision",
    [
        ("2015", "2015-01-01", "year"),
        ("2015-07", "2015-07-01", "month"),
        ("2015-07-09", "2015-07-09", "day"),
    ],
)
def test_parse_partial_date(text, iso, precision):
    pd = parse_partial_date(text)
    assert pd is not None and pd.iso == iso and pd.precision == precision


@pytest.mark.parametrize("bad", [None, "", "July 2015", "2015-13", "2015-02-30", 2015])
def test_parse_partial_date_rejects(bad):
    assert parse_partial_date(bad) is None


def test_granularity_support_and_buckets():
    year_only = parse_partial_date("2015")
    assert supports_granularity(year_only, "year") and not supports_granularity(year_only, "month")
    d = dt.date(2015, 8, 17)
    assert bucket_start(d, "year") == dt.date(2015, 1, 1)
    assert bucket_start(d, "quarter") == dt.date(2015, 7, 1)
    assert bucket_label(dt.date(2015, 7, 1), "quarter") == "2015-Q3"
    assert bucket_label(dt.date(2015, 8, 1), "month") == "2015-08"
    rng = bucket_range(dt.date(2015, 11, 3), dt.date(2016, 2, 1), "month")
    assert [bucket_label(r, "month") for r in rng] == ["2015-11", "2015-12", "2016-01", "2016-02"]
    assert len(bucket_range(dt.date(2015, 1, 1), dt.date(2016, 12, 1), "quarter")) == 8


def test_normalize_name_is_conservative():
    assert normalize_name("Keytruda®") == normalize_name("KEYTRUDA") == "keytruda"
    assert normalize_name("Novo Nordisk A/S") == "novo nordisk a s"
    assert normalize_name("Novo Nordisk A/S") != normalize_name("Novo Nordisk")
    assert normalize_name(None) == ""


def test_placebo_and_arm_split():
    assert is_placebo("Placebo") and is_placebo("matching placebo tablets") and is_placebo("Sham procedure")
    assert not is_placebo("Pembrolizumab")
    assert split_arm_intervention("Biological: Nivolumab") == ("Biological", "Nivolumab")
    assert split_arm_intervention("no prefix") is None


def test_label_picker_and_median_and_bins():
    lp = LabelPicker()
    for raw in ["Merck", "MERCK", "Merck"]:
        lp.add("merck", raw)
    assert lp.label("merck") == "Merck"
    assert median([3, 1, 2]) == 2 and median([1, 2, 3, 4]) == 2.5 and median([]) is None
    assert enrollment_bin(0) == 0 and enrollment_bin(25) == 1 and enrollment_bin(10**6) == 9
    assert enrollment_bin_label(1) == "25–49" and enrollment_bin_label(9) == "10,000+"


def test_cooccurrence_weight_is_distinct_trials_and_prune():
    g = CooccurrenceGraph()
    for trial in ["T1", "T2"]:
        g.add_node("a", "A", "drug", trial, f"{trial}-a")
        g.add_node("b", "B", "drug", trial, f"{trial}-b")
        g.add_edge("a", "b", "x", trial, [f"{trial}-a", f"{trial}-b"])
    g.add_edge("b", "a", "x", "T1", [])  # same pair again in T1, reversed: still weight 2
    g.add_node("c", "C", "drug", "T3")  # isolated: pruned
    pruned = g.prune()
    assert [e.weight for e in pruned.edges] == [2]
    assert {n.key for n in pruned.nodes} == {"a", "b"}
    assert all(n.degree == 1 for n in pruned.nodes)


def test_prune_raises_min_weight_and_keeps_both_types():
    g = CooccurrenceGraph(directed=True)
    for i in range(30):
        for j in range(30):
            trials = [f"T{i}-{j}-{k}" for k in range(1 + (i + j) % 3)]
            for t in trials:
                g.add_node(f"s{i}", f"S{i}", "sponsor", t)
                g.add_node(f"d{j}", f"D{j}", "drug", t)
                g.add_edge(f"s{i}", f"d{j}", "tests", t)
    pruned = g.prune(max_nodes=40, max_edges=300)
    assert len(pruned.nodes) <= 40 and len(pruned.edges) <= 300
    # the lowest tier that does not fit whole is used to top the budget up (deterministically)
    assert len(pruned.edges) == 300 and pruned.tie_truncated
    assert all(e.weight >= pruned.min_edge_weight for e in pruned.edges)
    assert pruned.edges == g.prune(max_nodes=40, max_edges=300).edges  # deterministic
    assert {n.type for n in pruned.nodes} == {"sponsor", "drug"}
    keep = {n.key for n in pruned.nodes}
    assert all(e.source in keep and e.target in keep for e in pruned.edges)
