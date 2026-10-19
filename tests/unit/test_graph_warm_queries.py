"""data/warm_queries.yaml: every body is a valid request; coverage of personas and appendix classes."""

from __future__ import annotations

from pathlib import Path

import yaml

from ctviz.api.schemas import VisualizeRequest

WARM = Path(__file__).resolve().parents[2] / "data" / "warm_queries.yaml"
INTENTS = Path(__file__).resolve().parents[2] / "data" / "intents.yaml"


def _load():
    return yaml.safe_load(WARM.read_text())["queries"]


def test_bodies_validate():
    queries = _load()
    assert 12 <= len(queries) <= 25
    ids = [q["id"] for q in queries]
    assert len(ids) == len(set(ids))
    for q in queries:
        VisualizeRequest.model_validate(q["body"])


def test_coverage():
    queries = _load()
    classes = {q["class"] for q in queries}
    assert {
        "time_trend",
        "distribution",
        "comparison",
        "geographic",
        "network_sponsor_drug",
        "network_drug_drug",
        "single_trial",
    } <= classes
    assert {q["persona"] for q in queries} == {"patient", "researcher"}
    single = {q["body"].get("nct_id") for q in queries if q["class"] == "single_trial"}
    assert {"NCT03548935", "NCT02578680"} <= single


def test_expected_intents_exist():
    known = set(yaml.safe_load(INTENTS.read_text()))
    for q in _load():
        assert q["expect_intent"] in known, q["id"]
