"""Pure knowledge-graph extraction (no database)."""

from __future__ import annotations

import copy

import pytest

from ctviz.api.schemas.citations import CitationKind
from ctviz.compute.graph_ingest import (
    PAIR_NODE_TYPES,
    affiliation_matches,
    clean_drug_name,
    extract,
    normalize_drug,
    observed_families,
    strip_arm_prefix,
)
from ctviz.core.provenance import FieldRef, Study, verify_citation
from ctviz.ctgov.paths import path_get


@pytest.mark.parametrize(
    ("affiliation", "facility", "expected"),
    [
        ("Mayo Clinic", "Mayo Clinic", True),
        ("Memorial Sloan Kettering Cancer Center", "Memorial Sloan-Kettering Cancer Center", True),
        ("Dana-Farber Cancer Institute", "Dana Farber Cancer Institute, Boston", True),
        ("University Hospital", "University Hospital Basel", False),  # raw substring; generic tokens only
        ("University of Arizona", "University of Arizona Cancer Center", False),  # one meaningful token
        ("Mayo", "Mayo Clinic Hospital Rochester", False),
        ("Bristol-Myers Squibb", "Local Institution - 0085", False),
        ("", "Mayo Clinic", False),
    ],
)
def test_affiliation_matches(affiliation, facility, expected):
    assert affiliation_matches(affiliation, facility) is expected


def test_official_paired_only_with_matching_site(study):
    raw = copy.deepcopy(study("NCT01844505").raw)
    raw["protocolSection"]["contactsLocationsModule"]["overallOfficials"] = [
        {"name": "Dr A", "affiliation": "Arizona", "role": "PRINCIPAL_INVESTIGATOR"},
        {"name": "Dr B", "affiliation": "University of Arizona Cancer Center", "role": "STUDY_CHAIR"},
    ]
    facts = extract(Study(raw))
    si = [(p.a[1], p.b[1]) for p in facts.pairs if p.pair_type == "site_investigator"]
    assert si == [("university of arizona cancer center, tucson, united states", "dr b")]


def test_single_site_pairs_only_principal_investigator(study):
    raw = copy.deepcopy(study("NCT01844505").raw)
    cl = raw["protocolSection"]["contactsLocationsModule"]
    cl["locations"] = cl["locations"][:1]
    cl["overallOfficials"] = [
        {"name": "Dr PI", "affiliation": "Elsewhere", "role": "PRINCIPAL_INVESTIGATOR"},
        {"name": "Dr Chair", "affiliation": "Elsewhere", "role": "STUDY_CHAIR"},
    ]
    si = [p.b[1] for p in extract(Study(raw)).pairs if p.pair_type == "site_investigator"]
    assert si == ["dr pi"]


def test_observed_families(study):
    s = study("NCT01844505")
    assert observed_families(s) >= {
        "drug",
        "arm_drug",
        "condition",
        "sponsor",
        "collaborator",
        "investigator",
        "site",
        "country",
    }
    raw = copy.deepcopy(s.raw)
    cl = raw["protocolSection"]["contactsLocationsModule"]
    cl["locations"] = [{"country": x["country"]} for x in cl["locations"]]
    del cl["overallOfficials"]
    obs = observed_families(Study(raw))
    assert "country" in obs and "site" not in obs and "investigator" not in obs


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Semaglutide 2.4 mg", "semaglutide"),
        ("Wegovy®", "wegovy"),
        ("Ozempic™ 1.0 mg (subcutaneous)", "ozempic (subcutaneous)"),
        ("Pembrolizumab 200mg", "pembrolizumab"),
        ("Nivolumab 3 mg/kg", "nivolumab"),
        ("Tirzepatide 5-15 mg", "tirzepatide"),
        ("Vitamin B12", "vitamin b12"),
        ("IL-2", "il-2"),
        ("5-FU", "5-fu"),
        ("  Metformin   XR  ", "metformin xr"),
    ],
)
def test_normalize_drug(raw, expected):
    assert normalize_drug(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "Placebo",
        "Placebo for Nivolumab",
        "semaglutide placebo",
        "Standard of Care",
        "Standard care",
        "Best Supportive Care",
        "SOC",
        "Sham injection",
        "",
        "2.4 mg",
    ],
)
def test_normalize_drug_skips_non_drugs(raw):
    assert normalize_drug(raw) is None


def test_clean_drug_name_keeps_case():
    assert clean_drug_name("Semaglutide® 2.4 mg") == "Semaglutide"


def test_strip_arm_prefix():
    assert strip_arm_prefix("Drug: Nivolumab") == ("Drug", "Nivolumab")
    assert strip_arm_prefix("Combination Product: Foo: Bar") == ("Combination Product", "Foo: Bar")
    assert strip_arm_prefix("Nivolumab") == (None, "Nivolumab")


def _assert_mentions_verify(facts, raw):
    for m in facts.mentions:
        value = path_get(raw, m.field_path)
        assert isinstance(value, str), m
        assert m.excerpt and m.excerpt in value, m
        ref = FieldRef(facts.nct_id, m.field_path, m.excerpt, kind=CitationKind.FIELD_VALUE)
        assert verify_citation(ref.to_citation(), raw), m


@pytest.mark.parametrize(
    "nct", ["NCT01844505", "NCT03548935", "NCT02578680", "NCT02142738", "NCT07654361", "NCT01284335"]
)
def test_mentions_cite_raw_json(study, nct):
    s = study(nct)
    facts = extract(s)
    assert facts.mentions
    _assert_mentions_verify(facts, s.raw)


def test_checkmate067_entities_and_pairs(study):
    s = study("NCT01844505")
    facts = extract(s)
    drugs = {m.canonical_norm for m in facts.mentions if m.entity_type == "drug" and m.rel == "drug"}
    assert drugs == {"nivolumab", "ipilimumab"}  # placebos skipped
    dd = [p for p in facts.pairs if p.pair_type == "drug_drug"]
    assert [(p.a, p.b) for p in dd] == [(("drug", "ipilimumab"), ("drug", "nivolumab"))]
    sponsor = [m for m in facts.mentions if m.entity_type == "sponsor"]
    assert [m.canonical for m in sponsor] == ["Bristol-Myers Squibb"]
    assert any(p.pair_type == "sponsor_drug" and p.b == ("drug", "nivolumab") for p in facts.pairs)
    assert any(p.pair_type == "drug_condition" for p in facts.pairs)
    # countries are deduplicated per trial even though the trial has 150 sites
    countries = [m.canonical_norm for m in facts.mentions if m.entity_type == "country"]
    assert len(countries) == len(set(countries))
    # otherNames become aliases
    assert any(a.alias_norm == "yervoy" and a.canonical == "Ipilimumab" for a in facts.aliases)


def test_pair_orientation(cohort):
    for s in cohort("melanoma_ph2_3").studies:
        for p in extract(s).pairs:
            first, second = PAIR_NODE_TYPES[p.pair_type]
            assert (p.a[0], p.b[0]) == (first, second)
            assert p.a != p.b
            if p.pair_type == "drug_drug":
                assert p.a < p.b


def test_cohort_mentions_verify(cohort):
    for s in cohort("melanoma_ph2_3").studies[:120]:
        _assert_mentions_verify(extract(s), s.raw)


def test_site_investigator_only_when_both_present(cohort):
    for s in cohort("obesity_since2020").studies:
        facts = extract(s)
        types = {m.entity_type for m in facts.mentions}
        if any(p.pair_type == "site_investigator" for p in facts.pairs):
            assert {"site", "investigator"} <= types
