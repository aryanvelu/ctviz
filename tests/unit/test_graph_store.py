"""GraphStore: idempotent ingest, retraction of changed records, co-occurrence, neighbors, resolution, locking."""

from __future__ import annotations

import copy
import json
import threading

import pytest

from ctviz.api.schemas.citations import CitationKind
from ctviz.compute.graph_ingest import extract
from ctviz.core.provenance import FieldRef, Study, verify_citation
from ctviz.ctgov.paths import path_get
from ctviz.store.db import Database
from ctviz.store.fts import TextIndex
from ctviz.store.graph import MAX_NCT_IDS, GraphStore


@pytest.fixture
def graph(tmp_path):
    db = Database(tmp_path / "g.db").connect()
    yield GraphStore(db)
    db.close()


def _snapshot(g: GraphStore):
    conn = g.db.conn
    cooccur = conn.execute(
        """SELECT na.type, na.canonical_norm, nb.type, nb.canonical_norm, c.pair_type, c.weight, c.trial_ids
           FROM kg_cooccur c JOIN kg_nodes na ON na.id=c.a JOIN kg_nodes nb ON nb.id=c.b ORDER BY 1,2,3,4,5"""
    ).fetchall()
    members = conn.execute("SELECT COUNT(*) FROM kg_cooccur_trials").fetchone()[0]
    edges = conn.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
    return cooccur, members, edges, g.stats_sync()


def _verify_evidence(evidence, raws):
    assert evidence
    for ev in evidence:
        raw = raws[ev["nct_id"]]
        value = path_get(raw, ev["field_path"])
        assert isinstance(value, str) and ev["excerpt"] in value, ev
        cite = FieldRef(ev["nct_id"], ev["field_path"], ev["excerpt"], kind=CitationKind.FIELD_VALUE).to_citation()
        assert verify_citation(cite, raw), ev


def _all_edges_verify(g: GraphStore, raws: dict[str, dict]) -> None:
    rows = g.db.conn.execute(
        "SELECT t.canonical, e.field_path, e.excerpt FROM kg_edges e JOIN kg_nodes t ON t.id=e.trial_id"
    ).fetchall()
    _verify_evidence([{"nct_id": n, "field_path": p, "excerpt": x} for n, p, x in rows], raws)


def _weights(g: GraphStore) -> None:
    """Invariant: kg_cooccur.weight == COUNT(memberships) and every summary row has members."""
    rows = g.db.conn.execute(
        """SELECT c.weight, (SELECT COUNT(*) FROM kg_cooccur_trials t
                             WHERE t.a=c.a AND t.b=c.b AND t.pair_type=c.pair_type), c.trial_ids
           FROM kg_cooccur c"""
    ).fetchall()
    for weight, count, sample in rows:
        assert weight == count > 0
        assert len(json.loads(sample)) == min(count, MAX_NCT_IDS)
    orphans = g.db.conn.execute(
        """SELECT COUNT(*) FROM kg_cooccur_trials t LEFT JOIN kg_cooccur c
           ON c.a=t.a AND c.b=t.b AND c.pair_type=t.pair_type WHERE c.a IS NULL"""
    ).fetchone()[0]
    assert orphans == 0


# -- idempotency ----------------------------------------------------------------------------------
def test_ingest_twice_same_weights(graph, cohort):
    studies = cohort("melanoma_ph2_3").studies
    assert graph.ingest_sync(studies) == {"ingested": 400, "skipped": 0, "replaced": 0}
    snap = _snapshot(graph)
    assert graph.ingest_sync(studies) == {"ingested": 0, "skipped": 400, "replaced": 0}
    assert _snapshot(graph) == snap
    _weights(graph)


def test_forced_reingest_does_not_double_count(graph, cohort):
    """Bypass the fingerprint fast path: retraction + set membership keep weights exact."""
    studies = cohort("melanoma_ph2_3").studies
    graph.ingest_sync(studies)
    snap = _snapshot(graph)
    graph.db.conn.execute("UPDATE kg_nodes SET meta=NULL WHERE type='trial'")
    graph.db.conn.commit()
    graph.ingest_facts_sync([extract(s) for s in studies[200:]])
    graph.ingest_facts_sync([extract(s) for s in studies[::-1]])
    assert _snapshot(graph) == snap
    _weights(graph)


def test_duplicate_trial_in_batch(graph, study):
    s = study("NCT01844505")
    graph.ingest_sync([s, s])
    rows = graph.db.conn.execute("SELECT weight FROM kg_cooccur").fetchall()
    assert rows and all(w == 1 for (w,) in rows)


# -- fix 1: changed records retract stale facts ---------------------------------------------------
def test_changed_record_retracts_stale_edges_and_pairs(graph, study):
    old = study("NCT01844505")
    graph.ingest_sync([old])
    assert graph.cooccurrence_sync([old.nct_id], "drug_drug")
    raw = copy.deepcopy(old.raw)
    aim = raw["protocolSection"]["armsInterventionsModule"]
    del aim["armGroups"][1]  # the nivolumab + ipilimumab arm
    aim["interventions"].reverse()  # every interventions[i] index moves
    raw["protocolSection"]["contactsLocationsModule"]["locations"] = raw["protocolSection"]["contactsLocationsModule"][
        "locations"
    ][::-1][:10]
    new = Study(raw)
    assert graph.ingest_sync([new])["replaced"] == 1
    _all_edges_verify(graph, {new.nct_id: new.raw})  # no evidence points at a stale array index
    assert graph.cooccurrence_sync([new.nct_id], "drug_drug") == []
    sites = graph.db.conn.execute("SELECT COUNT(DISTINCT entity_id) FROM kg_edges WHERE rel='site'").fetchone()[0]
    assert sites == 10
    _weights(graph)
    # drug_drug summary row removed when its last member is retracted
    assert graph.stats_sync()["cooccur"].get("drug_drug", 0) == 0


def test_retraction_only_affects_the_changed_trial(graph, study):
    a, b = study("NCT01844505"), study("NCT02578680")
    raw_b2 = copy.deepcopy(a.raw)  # a second trial with the same nivolumab + ipilimumab arms
    raw_b2["protocolSection"]["identificationModule"]["nctId"] = "NCT99999999"
    twin = Study(raw_b2)
    graph.ingest_sync([a, b, twin])
    pair = graph.cooccurrence_sync([a.nct_id, twin.nct_id], "drug_drug")
    assert pair[0]["weight"] == 2
    raw_a2 = copy.deepcopy(a.raw)
    del raw_a2["protocolSection"]["armsInterventionsModule"]["armGroups"][1]
    graph.ingest_sync([Study(raw_a2)])
    pair = graph.cooccurrence_sync([a.nct_id, twin.nct_id], "drug_drug")
    assert [(p["weight"], p["nct_ids"]) for p in pair] == [(1, ["NCT99999999"])]
    _weights(graph)


def test_sparse_projection_keeps_unobserved_families(graph, study):
    full = study("NCT01844505")
    graph.ingest_sync([full])
    before_sites = graph.db.conn.execute("SELECT COUNT(*) FROM kg_edges WHERE rel='site'").fetchone()[0]
    assert before_sites > 0
    raw = copy.deepcopy(full.raw)
    locs = raw["protocolSection"]["contactsLocationsModule"]["locations"]
    raw["protocolSection"]["contactsLocationsModule"]["locations"] = [{"country": x["country"]} for x in locs]
    graph.ingest_sync([Study(raw)])  # landscape-style record: countries only, no facilities
    after_sites = graph.db.conn.execute("SELECT COUNT(*) FROM kg_edges WHERE rel='site'").fetchone()[0]
    assert after_sites == before_sites
    _weights(graph)


# -- fix 2+3: normalized membership, indexed cohort query, capped nct_ids ---------------------------
def test_melanoma_drug_drug_cooccurrence(graph, cohort):
    c = cohort("melanoma_ph2_3")
    graph.ingest_sync(c.studies)
    ncts = [s.nct_id for s in c.studies]
    pairs = graph.cooccurrence_sync(ncts, "drug_drug", limit=50)
    labels = {frozenset((p["source_label"].casefold(), p["target_label"].casefold())): p for p in pairs}
    nivo_ipi = labels[frozenset(("nivolumab", "ipilimumab"))]
    expected = sum(
        (("drug", "ipilimumab"), ("drug", "nivolumab")) in {(p.a, p.b) for p in extract(s).pairs} for s in c.studies
    )
    assert nivo_ipi["weight"] == nivo_ipi["nct_ids_total"] == expected > MAX_NCT_IDS
    assert len(nivo_ipi["nct_ids"]) == MAX_NCT_IDS
    raws = {s.nct_id: s.raw for s in c.studies}
    for p in pairs[:10]:
        _verify_evidence(p["evidence"], raws)
        assert all(".interventionNames[" in e["field_path"] for e in p["evidence"])
    weights = [p["weight"] for p in pairs]
    assert weights == sorted(weights, reverse=True)
    _weights(graph)


def test_cooccurrence_uses_nct_index(graph, cohort):
    graph.ingest_sync(cohort("tirzepatide").studies)
    plan = graph.db.conn.execute(
        """EXPLAIN QUERY PLAN SELECT a, b, COUNT(*), json_group_array(nct_id) FROM kg_cooccur_trials
           WHERE nct_id IN (SELECT value FROM json_each(?)) AND pair_type = ? GROUP BY a, b""",
        ('["NCT1"]', "sponsor_drug"),
    ).fetchall()
    detail = " ".join(r[-1] for r in plan)
    assert "kg_cooccur_trials_nct" in detail and "SCAN kg_cooccur_trials" not in detail


def test_cooccurrence_restricted_to_trials(graph, cohort):
    c = cohort("melanoma_ph2_3")
    graph.ingest_sync(c.studies)
    subset = [s.nct_id for s in c.studies[:100]]
    full = {
        (p["source"], p["target"]): p["weight"]
        for p in graph.cooccurrence_sync([s.nct_id for s in c.studies], "sponsor_drug", limit=200)
    }
    part = graph.cooccurrence_sync(subset, "sponsor_drug", limit=200)
    assert part
    for p in part:
        assert set(p["nct_ids"]) <= set(subset)
        assert p["weight"] == p["nct_ids_total"] >= len(p["nct_ids"])
        if (p["source"], p["target"]) in full:
            assert p["weight"] <= full[(p["source"], p["target"])]
        assert p["source_type"] == "sponsor" and p["target_type"] == "drug"
    assert graph.cooccurrence_sync([], "sponsor_drug") == []
    assert graph.cooccurrence_sync(["NCT00000000"], "sponsor_drug") == []
    with pytest.raises(ValueError):
        graph.cooccurrence_sync(subset, "drug_sponsor")


@pytest.mark.parametrize("pair_type", ["sponsor_drug", "drug_condition", "sponsor_collaborator"])
def test_cooccurrence_evidence_verifies(graph, cohort, pair_type):
    c = cohort("melanoma_ph2_3")
    graph.ingest_sync(c.studies)
    pairs = graph.cooccurrence_sync([s.nct_id for s in c.studies], pair_type, limit=15)
    assert pairs
    raws = {s.nct_id: s.raw for s in c.studies}
    for p in pairs:
        _verify_evidence(p["evidence"], raws)


def test_site_investigator_evidence_verifies(graph, cohort):
    c = cohort("melanoma_ph2_3")
    graph.ingest_sync(c.studies)
    pairs = graph.cooccurrence_sync([s.nct_id for s in c.studies], "site_investigator", limit=15)
    raws = {s.nct_id: s.raw for s in c.studies}
    for p in pairs:
        _verify_evidence(p["evidence"], raws)


# -- neighbors + fix 5: resolution match kinds -------------------------------------------------------
def test_neighbors_evidence_verifies(graph, cohort, study):
    c = cohort("melanoma_ph2_3")
    full = study("NCT01844505")
    graph.ingest_sync(c.studies + [full])
    raws = {s.nct_id: s.raw for s in c.studies} | {full.nct_id: full.raw}
    res = graph.neighbors_sync("Nivolumab", "drug", limit=20)
    assert res["match"] == "exact" and res["center"]["id"] == "drug:nivolumab"
    assert res["center"]["trial_count"] >= 10
    assert 0 < len(res["edges"]) <= 20
    ids = {n["id"] for n in res["nodes"]}
    for e in res["edges"]:
        assert e["source"] in ids and e["target"] in ids
        assert "drug:nivolumab" in (e["source"], e["target"])
        assert len(e["nct_ids"]) <= MAX_NCT_IDS and e["nct_ids_total"] == e["weight"]
        _verify_evidence(e["evidence"], raws)
    assert any(e["pair_type"] == "drug_drug" for e in res["edges"])


def test_resolve_match_kinds(graph, cohort, study):
    graph.ingest_sync(cohort("melanoma_ph2_3").studies + [study("NCT01844505")])
    assert graph.resolve_sync("Nivolumab", "drug")["match"] == "exact"
    alias = graph.resolve_sync("Yervoy", "drug")
    assert alias["match"] == "alias" and alias["label"] == "Ipilimumab"
    fuzzy = graph.resolve_sync("uveal metastatic", "condition")
    assert fuzzy["match"] == "fuzzy" and {"uveal", "metastatic"} <= set(fuzzy["id"].split(":", 1)[1].split())
    assert graph.resolve_sync("noma", "condition") is None  # raw substring of "melanoma" is not a match
    assert graph.resolve_sync("ivolumab", "drug") is None
    assert graph.resolve_sync("NCT01844505", "trial")["match"] == "exact"
    assert graph.resolve_sync("NCT0184450", "trial") is None  # no prefix/substring guessing for trial IDs
    res = graph.neighbors_sync("uveal metastatic", "condition", limit=3)
    assert res["match"] == "fuzzy" and res["center"] is not None


def test_neighbors_alias_and_trial(graph, study):
    s = study("NCT01844505")
    graph.ingest_sync([s])
    via_alias = graph.neighbors_sync("Yervoy", "drug")
    assert via_alias["center"]["label"] == "Ipilimumab" and via_alias["match"] == "alias"
    assert graph.aliases_sync("BMS-936558", "drug") == ["Nivolumab"]
    trial = graph.neighbors_sync("nct01844505", "trial", limit=200)
    assert trial["center"]["id"] == "trial:NCT01844505" and trial["match"] == "exact"
    _verify_evidence([ev for e in trial["edges"] for ev in e["evidence"]], {s.nct_id: s.raw})
    assert graph.neighbors_sync("no such drug xyz", "drug") == {"center": None, "match": None, "nodes": [], "edges": []}
    with pytest.raises(ValueError):
        graph.neighbors_sync("x", "planet")


# -- fix 8: argument validation ----------------------------------------------------------------------
def test_limits_are_clamped(graph, cohort):
    c = cohort("melanoma_ph2_3")
    graph.ingest_sync(c.studies)
    ncts = [s.nct_id for s in c.studies]
    assert len(graph.cooccurrence_sync(ncts, "drug_condition", limit=10_000)) == 200
    assert len(graph.cooccurrence_sync(ncts, "drug_condition", limit=0)) == 1
    assert len(graph.cooccurrence_sync(ncts, "drug_condition", limit=-5)) == 1
    assert len(graph.neighbors_sync("Nivolumab", "drug", limit=10_000)["edges"]) <= 200
    assert len(graph.neighbors_sync("Nivolumab", "drug", limit=0)["edges"]) == 1
    with pytest.raises(ValueError):
        graph.neighbors_sync("Nivolumab", "drug", limit="many")


# -- fix 6: locking ---------------------------------------------------------------------------------
def test_shared_reentrant_lock(graph):
    assert graph.lock is graph.db.lock is TextIndex(graph.db).lock
    with graph.lock:  # re-entrant: public methods called while holding it do not deadlock
        assert graph.stats_sync()["edges"] == 0


def test_public_sync_methods_wait_for_lock(graph, study):
    graph.ingest_sync([study("NCT01844505")])
    started, done = threading.Event(), threading.Event()

    def reader():
        started.set()
        graph.stats_sync()
        done.set()

    with graph.lock:
        t = threading.Thread(target=reader)
        t.start()
        started.wait(2)
        assert not done.wait(0.2)  # blocked while another thread holds the lock
    t.join(2)
    assert done.is_set()


def test_concurrent_readers_and_writer(graph, cohort):
    c = cohort("melanoma_ph2_3")
    ncts = [s.nct_id for s in c.studies]
    errors: list[BaseException] = []

    def read():
        try:
            for _ in range(20):
                graph.cooccurrence_sync(ncts, "drug_drug", limit=20)
                graph.neighbors_sync("Nivolumab", "drug", limit=10)
                graph.stats_sync()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=read) for _ in range(4)]
    for t in threads:
        t.start()
    for i in range(0, 400, 50):
        graph.ingest_sync(c.studies[i : i + 50])
    for t in threads:
        t.join()
    assert not errors
    _weights(graph)


# -- misc -------------------------------------------------------------------------------------------
def test_stats_and_compounding(graph, cohort):
    graph.ingest_sync(cohort("semaglutide").studies)
    s1 = graph.stats_sync()
    assert s1["nodes"]["trial"] == 400 and s1["nodes"]["drug"] > 0 and s1["aliases"] > 0
    assert s1["cooccur_memberships"] >= sum(s1["cooccur"].values())
    graph.ingest_sync(cohort("tirzepatide").studies)
    s2 = graph.stats_sync()
    assert s2["nodes"]["trial"] > s1["nodes"]["trial"] and s2["edges"] > s1["edges"]


def test_mesh_aliases(graph, cohort):
    graph.ingest_sync(cohort("obesity_since2020").studies)
    rows = graph.db.conn.execute("SELECT alias_norm, kind, canonical, source FROM entity_aliases").fetchall()
    assert any(src == "mesh" and kind == "condition" for _, kind, _, src in rows)


async def test_async_api(graph, cohort):
    c = cohort("tirzepatide")
    await graph.ingest(c.studies)
    assert (await graph.stats())["nodes"]["trial"] == len(c.studies)
    res = await graph.neighbors("tirzepatide", "drug", limit=5)
    assert res["center"]["id"] == "drug:tirzepatide" and len(res["edges"]) <= 5
    assert (await graph.resolve("Tirzepatide", "drug"))["match"] == "exact"
    pairs = await graph.cooccurrence([s.nct_id for s in c.studies], "sponsor_drug", limit=5)
    assert pairs and pairs[0]["source_label"]
