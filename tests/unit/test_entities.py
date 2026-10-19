"""EntityResolver: alias table, drug-class expansion, scanning, spelling, MeSH growth."""

from __future__ import annotations

import json

import pytest

from ctviz.core.entities import ALIASES_FILE, EntityResolver, get_resolver
from tests.conftest import load_study_raw


@pytest.fixture
def r() -> EntityResolver:
    return EntityResolver()


def test_alias_table_size_and_shape():
    table = json.loads(ALIASES_FILE.read_text())
    n = sum(len(c["aliases"]) for c in table["drug_classes"].values()) + sum(
        len(table[k]) for k in ("drugs", "conditions", "sponsors", "countries")
    )
    assert 150 <= n <= 450
    for cls in table["drug_classes"].values():
        assert cls["members"] and all(m == m.lower() for m in cls["members"])


@pytest.mark.parametrize(
    ("brand", "generic"),
    [
        ("Ozempic", "semaglutide"),
        ("Wegovy", "semaglutide"),
        ("Keytruda", "pembrolizumab"),
        ("Mounjaro", "tirzepatide"),
        ("Zepbound", "tirzepatide"),
        ("Opdivo", "nivolumab"),
        ("MK-3475", "pembrolizumab"),
    ],
)
def test_brand_to_generic(r, brand, generic):
    e = r.resolve(brand, "drug")
    assert e.resolved == generic and e.method == "alias"


def test_generic_is_exact_and_carries_brand_search_terms(r):
    e = r.resolve("semaglutide", "drug")
    assert e.method == "exact"
    assert e.search_terms[0] == "semaglutide" and "ozempic" in e.search_terms


def test_glp1_class_expansion(r):
    members = r.expand_drug_class("GLP-1")
    assert members[:6] == ["semaglutide", "liraglutide", "dulaglutide", "exenatide", "tirzepatide", "lixisenatide"]
    assert {"orforglipron", "retatrutide", "survodutide", "mazdutide"} <= set(members)
    assert r.expand_drug_class("glp-1 receptor agonists") == members
    assert r.expand_drug_class("semaglutide") == []
    e = r.resolve("GLP-1", "drug")
    assert e.resolved == "GLP-1 receptor agonists" and e.search_terms == members


def test_pd1_class(r):
    m = r.expand_drug_class("PD-1 inhibitors")
    assert {"pembrolizumab", "nivolumab", "cemiplimab"} <= set(m)


@pytest.mark.parametrize(
    ("abbr", "canon"),
    [
        ("T2D", "type 2 diabetes"),
        ("NSCLC", "non-small cell lung cancer"),
        ("COPD", "chronic obstructive pulmonary disease"),
        ("ALS", "amyotrophic lateral sclerosis"),
    ],
)
def test_condition_abbreviations(r, abbr, canon):
    assert r.resolve(abbr, "condition").resolved == canon


def test_sponsor_and_country(r):
    assert r.resolve("Lilly", "sponsor").resolved == "Eli Lilly"
    assert r.resolve("USA", "country").resolved == "United States"


def test_spelling_correction(r):
    e = r.resolve("pembrolizumb", "drug")
    assert e.resolved == "pembrolizumab" and e.method == "spelling_correction"


def test_unknown_is_passthrough(r):
    e = r.resolve("Zorblaxumab", "drug")
    assert e.resolved == "Zorblaxumab" and e.method == "passthrough"


def test_structured_method_override(r):
    assert r.resolve("Keytruda", "drug", method="structured_field").method == "structured_field"


def test_scan_longest_match_and_case_rules(r):
    ms = r.scan("Compare type 2 diabetes vs NSCLC trials in the US for Ozempic and GLP-1 agonists")
    got = {(m.kind, m.entity.resolved) for m in ms}
    assert ("condition", "type 2 diabetes") in got
    assert ("condition", "non-small cell lung cancer") in got
    assert ("country", "United States") in got
    assert ("drug", "semaglutide") in got
    assert any(m.is_class for m in ms)
    assert not any(m.entity.resolved == "diabetes" for m in ms)


def test_scan_ignores_lowercase_short_abbreviations(r):
    # "us" (pronoun) and "als" inside text must not resolve
    assert r.scan("tell us about trials") == []


def test_scan_skips_ambiguous_terms(r):
    assert r.scan("What PK/PD are tracked in MS trials?") == []
    assert r.is_ambiguous("MS")


def test_guess_kind(r):
    assert r.guess_kind("Acme Pharmaceuticals") == ("sponsor", True)
    assert r.guess_kind("frobnicate syndrome") == ("condition", True)
    assert r.guess_kind("zorblaxumab") == ("drug", True)
    assert r.guess_kind("blue widgets")[1] is False


def test_learn_from_study_mesh():
    r = EntityResolver()
    raw = load_study_raw("NCT03548935")
    before = dict(r.learned["condition"])
    r.learn_from_study(raw)
    assert r.learned["condition"].keys() >= before.keys()
    # synthetic 1:1 pair is learned and then resolves via "mesh"
    fake = {
        "protocolSection": {"conditionsModule": {"conditions": ["Glorbish Fever"]}},
        "derivedSection": {"conditionBrowseModule": {"meshes": [{"id": "D1", "term": "Fever"}]}},
    }
    assert r.learn_from_study(fake) == 1
    e = r.resolve("glorbish fever", "condition")
    assert e.resolved == "Fever" and e.method == "mesh"
    assert any(m.entity.resolved == "Fever" for m in r.scan("trials for Glorbish Fever"))


def test_get_resolver_is_cached():
    assert get_resolver() is get_resolver()
