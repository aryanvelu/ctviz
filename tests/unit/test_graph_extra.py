"""Extra coverage for the knowledge graph, FTS index and vector re-ranker (test-writer pass).

Focus: citation invariants over *every* evidence item / search hit across all recorded fixtures, concurrent
ingest determinism, sparse records, and the "embeddings only reorder FTS candidates" contract. Offline only.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from types import SimpleNamespace

import httpx
import numpy as np
import openai
import pytest

from ctviz.api.schemas.citations import CitationKind
from ctviz.compute.graph_ingest import NODE_TYPES, PAIR_NODE_TYPES, PAIR_TYPES, extract
from ctviz.config import Settings
from ctviz.core.provenance import FieldRef, Study, verify_citation
from ctviz.ctgov.paths import path_get
from ctviz.store.db import Database
from ctviz.store.fts import SNIPPET_CHARS, TextIndex
from ctviz.store.graph import MAX_EVIDENCE_PER_EDGE, MAX_NCT_IDS, GraphStore
from ctviz.store.vectors import VectorIndex
from tests.conftest import load_cohort_raw, load_study_raw

FULL = ["NCT03548935", "NCT02578680", "NCT02142738", "NCT01844505", "NCT07654361", "NCT01284335"]
COHORTS = ["melanoma_ph2_3", "obesity_since2020"]
CHUNK_ID_RE = re.compile(r"^(NCT\d{8}):(.+):(\d+)-(\d+)$")


# -- shared corpus (module scoped: ingest once, query many times) -------------------------------
@pytest.fixture(scope="module")
def corpus():
    studies = [Study(load_study_raw(n)) for n in FULL]
    for name in COHORTS:
        studies += [Study(s) for s in load_cohort_raw(name)["studies"]]
    raws: dict[str, dict] = {}
    for s in studies:
        raws.setdefault(s.nct_id, s.raw)  # full record listed first wins; graph keeps the union anyway
    return studies, raws


@pytest.fixture(scope="module")
def loaded(tmp_path_factory, corpus):
    studies, raws = corpus
    db = Database(tmp_path_factory.mktemp("extra") / "kg.db").connect()
    graph, text = GraphStore(db), TextIndex(db)
    graph.ingest_sync(studies)
    text.index_sync(studies)
    yield graph, text, studies, raws
    db.close()


def _all_raws_for(studies, nct):
    return [s.raw for s in studies if s.nct_id == nct]


def _assert_evidence_cites(evidence, studies, allowed_ncts):
    """Every evidence item is a verbatim JSON leaf of a record the graph actually ingested."""
    assert evidence, "edge without evidence"
    for ev in evidence:
        assert set(ev) == {"nct_id", "field_path", "excerpt"}, ev
        assert ev["nct_id"] in allowed_ncts, ev
        cite = FieldRef(ev["nct_id"], ev["field_path"], ev["excerpt"], kind=CitationKind.FIELD_VALUE).to_citation()
        # the same NCT may appear in a full record and a cohort projection; the leaf must exist in one of them
        candidates = _all_raws_for(studies, ev["nct_id"])
        assert candidates, ev
        assert any(verify_citation(cite, raw) for raw in candidates), ev
        assert any(isinstance(path_get(raw, ev["field_path"]), str) for raw in candidates), ev


# -- (1) citation invariant over every neighbors()/cooccurrence() evidence item ------------------
@pytest.mark.parametrize("pair_type", PAIR_TYPES)
def test_cooccurrence_every_evidence_item_cites(loaded, pair_type):
    graph, _, studies, raws = loaded
    edges = graph.cooccurrence_sync(list(raws), pair_type, limit=1_000_000)
    assert edges, pair_type
    first, second = PAIR_NODE_TYPES[pair_type]
    for e in edges:
        assert (e["source_type"], e["target_type"]) == (first, second)
        # nct_ids is a capped sample; nct_ids_total carries the full count (= weight)
        assert e["weight"] == e["nct_ids_total"]
        assert len(e["nct_ids"]) == len(set(e["nct_ids"])) == min(e["weight"], MAX_NCT_IDS)
        assert set(e["nct_ids"]) <= set(raws)
        # one evidence item per endpoint per trial, at most MAX_EVIDENCE_PER_EDGE trials
        assert len(e["evidence"]) <= 2 * MAX_EVIDENCE_PER_EDGE
        _assert_evidence_cites(e["evidence"], studies, set(e["nct_ids"]))


@pytest.mark.parametrize("pair_type", PAIR_TYPES)
def test_cooccurrence_per_fixture_study_cites(loaded, pair_type):
    """Restricting to each single full-record study: evidence only ever comes from that trial."""
    graph, _, studies, _ = loaded
    for nct in FULL:
        for e in graph.cooccurrence_sync([nct], pair_type, limit=10_000):
            assert e["nct_ids"] == [nct] and e["weight"] == 1
            _assert_evidence_cites(e["evidence"], studies, {nct})


def _top_nodes(graph, node_type, n):
    return [
        r[0]
        for r in graph.db.conn.execute(
            """SELECT n.canonical FROM kg_nodes n JOIN kg_edges e ON e.entity_id = n.id WHERE n.type=?
               GROUP BY n.id ORDER BY COUNT(DISTINCT e.trial_id) DESC, n.canonical LIMIT ?""",
            (node_type, n),
        )
    ]


@pytest.mark.parametrize("node_type", [t for t in NODE_TYPES if t != "trial"])
def test_neighbors_every_evidence_item_cites(loaded, node_type):
    graph, _, studies, raws = loaded
    names = _top_nodes(graph, node_type, 8)
    assert names, node_type
    for name in names:
        res = graph.neighbors_sync(name, node_type, limit=200)
        assert res["center"] is not None and res["center"]["type"] == node_type, name
        ids = {n["id"] for n in res["nodes"]}
        assert len(ids) == len(res["nodes"])  # no duplicate nodes
        for e in res["edges"]:
            assert e["source"] in ids and e["target"] in ids
            assert res["center"]["id"] in (e["source"], e["target"])
            assert e["weight"] == e["nct_ids_total"]
            assert len(e["nct_ids"]) == len(set(e["nct_ids"])) == min(e["weight"], MAX_NCT_IDS)
            _assert_evidence_cites(e["evidence"], studies, set(e["nct_ids"]))


@pytest.mark.parametrize("nct", FULL)
def test_trial_neighbors_every_evidence_item_cites(loaded, nct):
    graph, _, studies, _ = loaded
    res = graph.neighbors_sync(nct, "trial", limit=10_000)
    assert res["center"]["id"] == f"trial:{nct}"
    assert res["edges"]
    for e in res["edges"]:
        assert e["nct_ids"] == [nct]
        assert len(e["evidence"]) <= MAX_EVIDENCE_PER_EDGE
        _assert_evidence_cites(e["evidence"], studies, {nct})


def test_cohort_trial_neighbors_cite(loaded):
    graph, _, studies, raws = loaded
    cohort_ncts = sorted(set(raws) - set(FULL))
    for nct in cohort_ncts[::20]:  # deterministic sample of ~40 cohort trials
        res = graph.neighbors_sync(nct, "trial", limit=10_000)
        for e in res["edges"]:
            _assert_evidence_cites(e["evidence"], studies, {nct})


# -- (2) every TextIndex.search hit is a verbatim slice with a well-formed chunk id -------------
_QUERIES = [
    ("pregnant", "all", None),
    ("pregnancy breastfeeding", "any", None),
    ("melanoma", "all", None),
    ("brain metastases", "all", None),
    ("body mass index", "all", ["inclusion"]),
    ('"type 2 diabetes"', "all", None),
    ("pregnant hepatitis", "any", ["exclusion", "inclusion", "eligibility"]),  # cohorts carry no criteria text
    ("insulin metformin", "any", None),
    ("overall survival", "all", ["outcome"]),
    ("nivolumab ipilimumab", "any", ["arm", "intervention"]),
    ("weight loss", "all", ["summary", "description"]),
    ("hepatitis HIV", "any", None),
    ("ECOG performance status", "all", None),
    ("adverse events", "all", None),
    ("bariatric surgery", "all", None),
]


@pytest.mark.parametrize(("query", "mode", "sections"), _QUERIES)
def test_every_search_hit_is_verbatim(loaded, query, mode, sections):
    _, text, studies, _ = loaded
    hits = text.search_sync(query, k=200, mode=mode, sections=sections)
    assert hits, query
    seen = set()
    for h in hits:
        m = CHUNK_ID_RE.match(h.chunk_id)
        assert m, h.chunk_id
        assert m.groups() == (h.nct_id, h.field_path, str(h.char_start), str(h.char_end))
        assert h.chunk_id not in seen  # no duplicate chunks returned
        seen.add(h.chunk_id)
        if sections:
            assert h.section in sections
        assert 0 <= h.char_start < h.char_end
        assert h.char_start <= h.snippet_start < h.snippet_end <= h.char_end
        assert 0 < len(h.snippet) <= SNIPPET_CHARS
        assert h.snippet == h.snippet.strip()
        candidates = [s for s in studies if s.nct_id == h.nct_id]
        ok_text = ok_snip = False
        for s in candidates:
            value = path_get(s.raw, h.field_path)
            if not isinstance(value, str):
                continue
            ok_text |= value[h.char_start : h.char_end] == h.text
            if value[h.snippet_start : h.snippet_end] == h.snippet:
                ref = s.text_ref(h.field_path, h.snippet_start, h.snippet_end)
                ok_snip |= ref is not None and ref.excerpt == h.snippet and verify_citation(ref.to_citation(), s.raw)
        assert ok_text, h.chunk_id
        assert ok_snip, h.chunk_id


def test_search_scoped_to_each_cohort_stays_in_scope(loaded):
    _, text, _, _ = loaded
    for name in COHORTS:
        scope = {s["protocolSection"]["identificationModule"]["nctId"] for s in load_cohort_raw(name)["studies"]}
        hits = text.search_sync("criteria", nct_ids=sorted(scope), k=300, mode="any")
        assert hits and {h.nct_id for h in hits} <= scope


# -- (3) concurrent ingest == sequential ingest -------------------------------------------------
def _canonical_snapshot(graph: GraphStore):
    conn = graph.db.conn
    cooccur = sorted(
        (pt, *sorted([(ta, na), (tb, nb)]), w, tids)
        for ta, na, tb, nb, pt, w, tids in conn.execute(
            """SELECT na.type, na.canonical_norm, nb.type, nb.canonical_norm, c.pair_type, c.weight, c.trial_ids
               FROM kg_cooccur c JOIN kg_nodes na ON na.id=c.a JOIN kg_nodes nb ON nb.id=c.b"""
        )
    )
    edges = sorted(
        conn.execute(
            """SELECT t.canonical_norm, n.type, n.canonical_norm, e.rel, e.field_path, e.excerpt
               FROM kg_edges e JOIN kg_nodes t ON t.id=e.trial_id JOIN kg_nodes n ON n.id=e.entity_id"""
        ).fetchall()
    )
    nodes = sorted(conn.execute("SELECT type, canonical_norm FROM kg_nodes").fetchall())
    aliases = sorted(conn.execute("SELECT alias_norm, kind, canonical, source FROM entity_aliases").fetchall())
    return cooccur, edges, nodes, aliases


def _overlapping_batches():
    mel = [Study(s) for s in load_cohort_raw("melanoma_ph2_3")["studies"]]
    obe = [Study(s) for s in load_cohort_raw("obesity_since2020")["studies"]]
    full = [Study(load_study_raw(n)) for n in FULL]
    a = mel[:260] + obe[:150] + full[:4]
    b = mel[140:] + obe[100:] + full[2:]  # overlaps a on mel[140:260], obe[100:150], full[2:4]
    return a, b


async def test_concurrent_ingest_matches_sequential(tmp_path):
    a, b = _overlapping_batches()

    seq_db = Database(tmp_path / "seq.db").connect()
    seq = GraphStore(seq_db)
    seq.ingest_sync(a)
    seq.ingest_sync(b)

    con_db = Database(tmp_path / "con.db").connect()
    con = GraphStore(con_db)
    r1, r2 = await asyncio.gather(con.ingest(a), con.ingest(b))
    assert r1["ingested"] + r1["skipped"] == len({s.nct_id for s in a})
    assert r2["ingested"] + r2["skipped"] == len({s.nct_id for s in b})

    assert _canonical_snapshot(con) == _canonical_snapshot(seq)
    for *_, w, tids in _canonical_snapshot(con)[0]:
        assert min(w, MAX_NCT_IDS) == len(json.loads(tids))
    seq_db.close()
    con_db.close()


async def test_concurrent_ingest_order_independent_weights(tmp_path):
    """Weights / edges don't depend on which batch wins the write lock (b before a, many tasks)."""
    a, b = _overlapping_batches()
    ref_db = Database(tmp_path / "ref.db").connect()
    ref = GraphStore(ref_db)
    ref.ingest_sync(a + b)

    db = Database(tmp_path / "many.db").connect()
    g = GraphStore(db)
    await asyncio.gather(g.ingest(b), g.ingest(a), g.ingest(a[::2]), g.ingest(b[::-1]))
    got, want = _canonical_snapshot(g), _canonical_snapshot(ref)
    assert got[0] == want[0]  # co-occurrence weights + trial_ids
    assert got[1] == want[1]  # evidence edges
    assert got[2] == want[2]  # node keys
    ref_db.close()
    db.close()


async def test_concurrent_graph_and_text_index_writes(tmp_path):
    a, b = _overlapping_batches()
    db = Database(tmp_path / "mix.db").connect()
    g, t = GraphStore(db), TextIndex(db)
    await asyncio.gather(g.ingest(a), t.index(a), g.ingest(b), t.index(b))
    ref_db = Database(tmp_path / "mixref.db").connect()
    rt = TextIndex(ref_db)
    rt.index_sync(a)
    rt.index_sync(b)
    q = "SELECT nct_id, field_path, char_start, char_end, section, text FROM trial_text"
    assert sorted(db.conn.execute(q).fetchall()) == sorted(ref_db.conn.execute(q).fetchall())
    db.close()
    ref_db.close()


# -- (4) studies with missing modules ingest cleanly --------------------------------------------
def _sparse(nct: str, **protocol) -> Study:
    return Study({"protocolSection": {"identificationModule": {"nctId": nct, "briefTitle": "Sparse"}, **protocol}})


SPARSE = [
    _sparse("NCT09000001"),  # identification only
    _sparse("NCT09000002", conditionsModule={"conditions": ["Melanoma"]}),
    _sparse(
        "NCT09000003",
        conditionsModule={"conditions": ["Obesity", ""]},
        armsInterventionsModule={"interventions": [{"type": "DRUG", "name": "Semaglutide 2.4 mg"}]},
    ),
    _sparse(
        "NCT09000004",
        sponsorCollaboratorsModule={},  # module present, leadSponsor missing
        armsInterventionsModule={"armGroups": [], "interventions": []},
        contactsLocationsModule={"locations": [], "overallOfficials": [{"name": "Dr. Solo"}]},
        eligibilityModule={},
        descriptionModule={"briefSummary": "   "},
        outcomesModule={"primaryOutcomes": [{"measure": "Weight change"}]},
    ),
    _sparse(
        "NCT09000005",
        contactsLocationsModule={"locations": [{"city": "Oslo"}, {"facility": "Site A", "contacts": None}]},
        armsInterventionsModule={"armGroups": [{"label": "A", "interventionNames": ["Drug: Unlisted"]}]},
    ),
]


def test_sparse_studies_extract_ingest_and_index(tmp_path):
    for s in SPARSE:
        facts = extract(s)
        for m in facts.mentions:
            assert verify_citation(
                FieldRef(s.nct_id, m.field_path, m.excerpt, kind=CitationKind.FIELD_VALUE).to_citation(), s.raw
            ), m
        assert not [p for p in facts.pairs if p.pair_type in ("sponsor_drug", "sponsor_collaborator")]

    db = Database(tmp_path / "sparse.db").connect()
    g, t = GraphStore(db), TextIndex(db)
    assert g.ingest_sync(SPARSE) == {"ingested": len(SPARSE), "skipped": 0, "replaced": 0}
    assert g.ingest_sync(SPARSE) == {"ingested": 0, "skipped": len(SPARSE), "replaced": 0}
    t.index_sync(SPARSE)
    stats = g.stats_sync()
    assert stats["nodes"]["trial"] == len(SPARSE)
    assert "sponsor" not in stats["nodes"]

    bare = g.neighbors_sync("NCT09000001", "trial")
    assert bare["center"]["id"] == "trial:NCT09000001" and bare["edges"] == []
    ncts = [s.nct_id for s in SPARSE]
    assert g.cooccurrence_sync(ncts, "sponsor_drug") == []
    dc = g.cooccurrence_sync(ncts, "drug_condition")
    assert [(e["source_label"], e["target_label"]) for e in dc] == [("Semaglutide", "Obesity")]
    _assert_evidence_cites(dc[0]["evidence"], SPARSE, {"NCT09000003"})
    for e in g.neighbors_sync("NCT09000005", "trial")["edges"]:
        _assert_evidence_cites(e["evidence"], SPARSE, {"NCT09000005"})

    hits = t.search_sync("weight change", k=5)
    assert [h.chunk_id for h in hits] == ["NCT09000004:protocolSection.outcomesModule.primaryOutcomes[0].measure:0-13"]
    assert t.search_sync("Sparse", k=5) == []  # titles are not indexed; whitespace summary skipped
    db.close()


async def test_sparse_studies_async_ingest(tmp_path):
    db = Database(tmp_path / "sparse_async.db").connect()
    g = GraphStore(db)
    await asyncio.gather(g.ingest(SPARSE[:3]), g.ingest(SPARSE[2:]))
    assert (await g.stats())["nodes"]["trial"] == len(SPARSE)
    db.close()


# -- (5) vectors only reorder FTS candidates and fail closed -----------------------------------
def _vsettings(tmp_path) -> Settings:
    return Settings(db_path=tmp_path / "v.db", embeddings_enabled=True, offline=False, _env_file=None)  # type: ignore[call-arg]


class _HashEmbeddings:
    """Deterministic pseudo-embeddings; optional misbehaviour to model a bad API response."""

    def __init__(self, extra: int = 0, drop: int = 0):
        self.extra, self.drop, self.calls = extra, drop, 0

    async def create(self, model, input):  # noqa: A002
        self.calls += 1
        out = []
        for i, t in enumerate(input):
            rng = np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16))
            out.append(SimpleNamespace(index=i, embedding=rng.normal(size=8).tolist()))
        out = out[: len(out) - self.drop] if self.drop else out
        out += [SimpleNamespace(index=len(input) + j, embedding=[1.0] * 8) for j in range(self.extra)]
        return SimpleNamespace(data=out)


def _permission_denied() -> openai.PermissionDeniedError:
    req = httpx.Request("POST", "https://api.openai.invalid/v1/embeddings")
    resp = httpx.Response(403, request=req, json={"error": {"message": "model not allowed"}})
    return openai.PermissionDeniedError("model not allowed", response=resp, body=None)


class _Raising:
    def __init__(self, exc):
        self.exc, self.calls = exc, 0

    async def create(self, model, input):  # noqa: A002
        self.calls += 1
        raise self.exc


async def test_rerank_never_leaves_fts_candidates(loaded, tmp_path):
    _, text, _, _ = loaded
    fake = _HashEmbeddings()
    vi = VectorIndex(_vsettings(tmp_path), client_factory=lambda: SimpleNamespace(embeddings=fake))
    for query, mode, sections in _QUERIES[:6]:
        hits = text.search_sync(query, k=25, mode=mode, sections=sections)
        order = await vi.rerank(query, [h.text for h in hits])
        assert sorted(order) == list(range(len(hits)))  # a permutation: nothing added, nothing dropped
        reranked = [hits[i] for i in order]
        assert {h.chunk_id for h in reranked} == {h.chunk_id for h in hits}
        assert {h.nct_id for h in reranked} <= {h.nct_id for h in hits}
    assert fake.calls >= 1 and vi.enabled


async def test_rerank_disables_on_openai_403(loaded, tmp_path):
    _, text, _, _ = loaded
    fake = _Raising(_permission_denied())
    db = Database(tmp_path / "v403.db").connect()
    vi = VectorIndex(_vsettings(tmp_path), db=db, client_factory=lambda: SimpleNamespace(embeddings=fake))
    hits = text.search_sync("pregnant", k=10)
    assert await vi.rerank("pregnant", [h.text for h in hits]) == list(range(len(hits)))
    assert not vi.enabled and "PermissionDeniedError" in (vi.disabled_reason or "")
    assert await vi.rerank("melanoma", ["a", "b"]) == [0, 1]
    assert await vi.embed(["x"]) is None
    assert fake.calls == 1  # disabled for the process: no further API calls
    assert db.conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 0  # nothing cached on failure
    db.close()


@pytest.mark.parametrize(
    "exc", [httpx.ConnectTimeout("t"), openai.APIConnectionError(request=httpx.Request("POST", "https://x.invalid"))]
)
async def test_rerank_disables_on_network_errors(tmp_path, exc):
    fake = _Raising(exc)
    vi = VectorIndex(_vsettings(tmp_path), client_factory=lambda: SimpleNamespace(embeddings=fake))
    assert await vi.rerank("q", ["a", "b", "c"]) == [0, 1, 2]
    assert not vi.enabled


@pytest.mark.parametrize("bad", [{"extra": 1}, {"drop": 1}])
async def test_rerank_fails_closed_on_wrong_embedding_count(tmp_path, bad):
    fake = _HashEmbeddings(**bad)
    vi = VectorIndex(_vsettings(tmp_path), client_factory=lambda: SimpleNamespace(embeddings=fake))
    assert await vi.rerank("q", ["a", "b", "c"]) == [0, 1, 2]
    assert not vi.enabled


async def test_rerank_short_inputs_skip_api(tmp_path):
    fake = _HashEmbeddings()
    vi = VectorIndex(_vsettings(tmp_path), client_factory=lambda: SimpleNamespace(embeddings=fake))
    assert await vi.rerank("q", []) == []
    assert await vi.rerank("q", ["only"]) == [0]
    assert await vi.embed([]) is None
    assert fake.calls == 0
