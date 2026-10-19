"""Schema 2.2: the public JSON inlines at most MAX_NCT_IDS_PER_DATUM trial ids per datum (plus the full count);
the complete list stays internal and is served by GET /v1/responses/{request_id}/datums/{datum_id}/citations."""

from __future__ import annotations

from ctviz.api.public import DATUM_SOURCES_URL, MAX_NCT_IDS_PER_DATUM, PublicResponse, to_public
from ctviz.api.schemas.citations import Citation
from ctviz.api.schemas.plan import Intent, Persona, Scope
from ctviz.api.schemas.request import VisualizeRequest
from ctviz.api.schemas.response import SCHEMA_VERSION, Coverage, Interpretation, ResponseMeta, VisualizeResponse
from ctviz.api.schemas.viz import Datum, EdgeEncoding, Encoding, GraphData, NodeEncoding, Visualization, VizType
from ctviz.core.background import drain
from ctviz.store import response_cache as rc
from tests.unit.test_api import make_client  # noqa: F401 - pytest fixture
from tests.unit.test_orchestrator import comparison_plan, env, make  # noqa: F401 - env is a pytest fixture

IDS = [f"NCT{n:08d}" for n in range(1, 51)]  # 50 sorted ids


def _cite(nct: str) -> Citation:
    return Citation(
        nct_id=nct,
        excerpt="Pembrolizumab",
        field_path="protocolSection.armsInterventionsModule.interventions[0].name",
        url=f"https://clinicaltrials.gov/study/{nct}",
    )


def _network() -> VisualizeResponse:
    nodes = [
        Datum(
            id="n1",
            label="Pembrolizumab",
            type="drug",
            size=50,
            nct_ids=IDS,
            citations=[_cite(IDS[0])],
            citations_total=50,
        ),
        Datum(id="n2", label="Ipilimumab", type="drug", size=5, nct_ids=IDS[:5], citations=[_cite(IDS[0])]),
        Datum(id="n3", label="Merck", type="sponsor", size=20, nct_ids=IDS[:20], citations=[_cite(IDS[0])]),
    ]
    edges = [
        Datum(
            id="e1", source="n1", target="n2", weight=21, type="co_tested", nct_ids=IDS[:21], citations=[_cite(IDS[0])]
        )
    ]
    viz = Visualization(
        type=VizType.NETWORK_GRAPH,
        title="Melanoma drug network",
        encoding=Encoding(nodes=NodeEncoding(), edges=EdgeEncoding()),
        data=GraphData(nodes=nodes, edges=edges),
    )
    interp = Interpretation(
        intent=Intent.PHASE_DISTRIBUTION,
        intent_confidence=1,
        persona=Persona.RESEARCHER,
        scope=Scope.COHORT,
        parser="rules",
    )
    return VisualizeResponse(
        request_id="a" * 32,
        created_at="2026-10-19T00:00:00+00:00",
        visualization=viz,
        meta=ResponseMeta(interpretation=interp, coverage=Coverage(studies_matched_total=50, studies_analyzed=50)),
    )


def _by_id(pub: dict) -> dict[str, dict]:
    data = pub["visualization"]["data"]
    return {d["id"]: d for d in data["nodes"] + data["edges"]}


# -- serializer --------------------------------------------------------------------------------------
def test_cap_applies_to_nodes_and_edges_in_default_and_ids_only_modes():
    resp = _network()
    for detail in ("sample", "ids_only"):
        pub = to_public(resp, citation_detail=detail)
        PublicResponse.model_validate(pub)
        d = _by_id(pub)
        assert d["n1"]["nct_ids"] == IDS[:MAX_NCT_IDS_PER_DATUM] and d["n1"]["nct_ids_total"] == 50
        assert d["n2"]["nct_ids"] == IDS[:5] and d["n2"]["nct_ids_total"] == 5  # under the cap: complete
        assert d["n3"]["nct_ids"] == IDS[:20] and d["n3"]["nct_ids_total"] == 20  # exactly the cap: complete
        assert len(d["e1"]["nct_ids"]) == MAX_NCT_IDS_PER_DATUM and d["e1"]["nct_ids_total"] == 21 == d["e1"]["weight"]
    # the internal response is untouched (validation and the response cache keep complete lists)
    assert resp.visualization.data.nodes[0].nct_ids == IDS


def test_citation_detail_full_is_uncapped():
    pub = to_public(_network(), citation_detail="full")
    PublicResponse.model_validate(pub)
    d = _by_id(pub)
    assert d["n1"]["nct_ids"] == IDS and d["n1"]["nct_ids_total"] == 50
    assert d["e1"]["nct_ids"] == IDS[:21] and d["e1"]["nct_ids_total"] == 21
    assert pub["meta"]["citation_policy"]["nct_ids"] == "complete"
    assert "nct_ids_per_datum" not in pub["meta"]["citation_policy"]


def test_citation_policy_describes_the_cap():
    for detail in ("sample", "ids_only"):
        pol = to_public(_network(), citation_detail=detail)["meta"]["citation_policy"]
        assert pol["nct_ids"] == f"first_{MAX_NCT_IDS_PER_DATUM}" and pol["nct_ids_per_datum"] == 20
        assert pol["all_nct_ids"] == DATUM_SOURCES_URL and pol["nct_ids_order"] == "nct_id"
    assert "all_excerpts" in to_public(_network())["meta"]["citation_policy"]
    assert to_public(_network())["meta"]["schema_version"] == SCHEMA_VERSION == "2.2"


def test_datum_without_trial_ids_has_no_total():
    resp = _network()
    resp.visualization.data.nodes[1] = Datum(id="n2", label="x", type="drug", size=3, query_url="https://q")
    d = _by_id(to_public(resp))
    assert "nct_ids" not in d["n2"] and "nct_ids_total" not in d["n2"] and d["n2"]["query_url"] == "https://q"


def test_public_schema_documents_nct_ids_total():
    props = PublicResponse.model_json_schema()["$defs"]["PublicDatum"]["properties"]
    assert "nct_ids_total" in props and "first 20" in props["nct_ids"]["description"]


# -- end to end: capped in the response, complete in the store --------------------------------------
async def test_orchestrator_stores_complete_lists_behind_capped_public_rows(env):  # noqa: F811
    o = make(env, comparison_plan())
    resp = await o.run(VisualizeRequest(query="Compare phases for semaglutide vs tirzepatide"))
    await drain()
    pub = to_public(resp)
    PublicResponse.model_validate(pub)
    internal = {d.id: d for d in resp.visualization.data}
    big = [d for d in pub["visualization"]["data"] if d.get("nct_ids_total", 0) > MAX_NCT_IDS_PER_DATUM]
    assert big, "fixture cohort should have a bar with more than 20 trials"
    got = await rc.get_datum_sources(env[1], resp.request_id)
    assert got is not None
    cites, ncts = got
    for d in pub["visualization"]["data"]:
        full = internal[d["id"]].nct_ids
        if not full:
            assert "nct_ids_total" not in d
            continue
        assert d["nct_ids"] == full[:MAX_NCT_IDS_PER_DATUM] and d["nct_ids_total"] == len(full) == d["trial_count"]
        assert ncts[d["id"]] == full
    assert await rc.get_citations(env[1], resp.request_id) == cites


async def test_response_cache_alias_keeps_complete_lists(env):  # noqa: F811
    o = make(env, comparison_plan())
    req = VisualizeRequest(query="Compare phases for semaglutide vs tirzepatide")
    first = await o.run(req)
    await drain()
    second = await o.run(req)
    assert second.meta.cache.response_hit
    await drain()
    assert await rc.get_datum_sources(env[1], second.request_id) == await rc.get_datum_sources(env[1], first.request_id)


# -- the endpoint -------------------------------------------------------------------------------------
def test_endpoint_returns_complete_nct_ids(make_client):  # noqa: F811
    tc, orch = make_client()
    rid = "b" * 32
    cites = {"n1": [{"nct_id": "NCT00000001", "excerpt": "x", "field_path": "p", "url": "u"}]}
    tc.portal.call(lambda: rc.put_citations(orch.db, rid, cites, 60, nct_ids={"n1": IDS, "n9": IDS[:2]}))
    body = tc.get(f"/v1/responses/{rid}/datums/n1/citations").json()
    assert body["nct_ids"] == IDS and body["nct_ids_total"] == 50 and body["total"] == 1
    only_ids = tc.get(f"/v1/responses/{rid}/datums/n9/citations").json()  # ids but no stored excerpts
    assert only_ids["nct_ids"] == IDS[:2] and only_ids["citations"] == []
    assert tc.get(f"/v1/responses/{rid}/datums/zz/citations").status_code == 404


def test_endpoint_reads_legacy_records_without_ids(make_client):  # noqa: F811
    tc, orch = make_client()
    rid = "c" * 32
    legacy = {"d1": [{"nct_id": "NCT00000001", "excerpt": "x"}]}  # schema <= 2.1: a bare citations map
    tc.portal.call(lambda: orch.db.cache_set(rc.CITATIONS_NS, rid, legacy, 60))
    body = tc.get(f"/v1/responses/{rid}/datums/d1/citations").json()
    assert body["total"] == 1 and "nct_ids" not in body


# -- evidence for a cache-hit request id survives a lost or not-yet-run background write -------------------
def _drop_spawn(coro, name="background", *, low_priority=False):
    """Stand-in for background.spawn that loses the work (queue full / crash before the write)."""
    coro.close()


async def test_cache_hit_evidence_served_when_alias_write_is_dropped(env, monkeypatch):  # noqa: F811
    from ctviz.core import orchestrator as orch_mod

    o = make(env, comparison_plan())
    req = VisualizeRequest(query="Compare phases for semaglutide vs tirzepatide")
    first = await o.run(req)
    await drain()
    monkeypatch.setattr(orch_mod, "spawn", _drop_spawn)
    second = await o.run(req)  # cache hit; its evidence copy is dropped, and we do not drain
    assert second.meta.cache.response_hit
    assert await rc.get_datum_sources(env[1], second.request_id) is None  # nothing was written
    got = await o.evidence(second.request_id)
    assert got == await rc.get_datum_sources(env[1], first.request_id)  # the source record, full excerpts


async def test_cache_hit_evidence_rebuilt_from_cached_response_when_source_is_gone(env, monkeypatch):  # noqa: F811
    from ctviz.core import orchestrator as orch_mod

    o = make(env, comparison_plan())
    req = VisualizeRequest(query="Compare phases for semaglutide vs tirzepatide")
    first = await o.run(req)
    await drain()
    await env[1].cache_set(rc.CITATIONS_NS, first.request_id, None, -1)  # the source record has expired
    monkeypatch.setattr(orch_mod, "spawn", _drop_spawn)
    second = await o.run(req)
    cites, ncts = await o.evidence(second.request_id)
    for d in second.visualization.data:
        assert ncts.get(d.id, []) == (d.nct_ids or [])  # complete lists from the cached internal response
        assert len(cites[d.id]) == len(d.citations)  # inline excerpts only
    assert await o.evidence("f" * 32) is None


async def test_cache_hit_alias_write_lands_at_normal_priority(env, monkeypatch):  # noqa: F811
    from ctviz.core import background

    o = make(env, comparison_plan())
    req = VisualizeRequest(query="Compare phases for semaglutide vs tirzepatide")
    await o.run(req)
    await drain()
    monkeypatch.setattr(background, "pending", lambda: background.MAX_PENDING + 1)  # queue "full"
    second = await o.run(req)
    await drain()
    assert await rc.get_datum_sources(env[1], second.request_id) is not None  # not dropped
    assert second.request_id not in o.pending_evidence._items  # forgotten once written


def test_route_serves_pending_evidence(make_client, monkeypatch):  # noqa: F811
    tc, orch = make_client()
    rid = "d" * 32

    async def evidence(request_id):
        return ({"n1": []}, {"n1": IDS}) if request_id == rid else None

    monkeypatch.setattr(orch, "evidence", evidence, raising=False)
    body = tc.get(f"/v1/responses/{rid}/datums/n1/citations").json()
    assert body["nct_ids"] == IDS and body["nct_ids_total"] == 50 and body["total"] == 0
    assert tc.get(f"/v1/responses/{'e' * 32}/datums/n1/citations").status_code == 404


# -- derived_from is a provenance key ----------------------------------------------------------------------
def test_derived_from_is_reserved_provenance_key():
    resp = _network()
    resp.visualization.data.nodes[1] = Datum(
        id="n2", label="x", type="drug", size=5, nct_ids=IDS[:5], derived_from=["n1", "n3"]
    )
    for detail in ("sample", "ids_only", "full"):
        d = _by_id(to_public(resp, citation_detail=detail))["n2"]
        keys = list(d)
        assert d["derived_from"] == ["n1", "n3"]
        assert keys.index("derived_from") > keys.index("nct_ids_total")  # after the data fields, with provenance
    assert "derived_from" in PublicResponse.model_json_schema()["$defs"]["PublicDatum"]["properties"]


def test_graph_neighbors_uses_the_same_cap():
    from ctviz.store.graph import MAX_NCT_IDS

    assert MAX_NCT_IDS == MAX_NCT_IDS_PER_DATUM == 20


# -- citations inside meta hints get the same trimming as datum citations --------------------------------
def test_meta_threshold_citations_are_trimmed():
    from ctviz.api.schemas.viz import Threshold

    resp = _network()
    resp.visualization.meta.thresholds = [
        Threshold(axis="y", value=0.33, label="DLT", source="registry", citations=[_cite(IDS[0])])
    ]
    for detail in ("sample", "ids_only", "full"):
        c = to_public(resp, citation_detail=detail)["meta"]["thresholds"][0]["citations"][0]
        assert c == {"nct_id": IDS[0], "excerpt": "Pembrolizumab", "field_path": _cite(IDS[0]).field_path}


def test_meta_citations_trim_any_nested_hint():
    from ctviz.api.public import _meta_citations

    raw = {
        "stats": {
            "refs": [
                {
                    "citations": [
                        {
                            "nct_id": "N",
                            "excerpt": "1",
                            "field_path": "p",
                            "url": "u",
                            "kind": "text_excerpt",
                            "char_start": 0,
                            "char_end": 1,
                        }
                    ]
                }
            ]
        }
    }
    out = _meta_citations(raw)["stats"]["refs"][0]["citations"][0]
    assert out == {
        "nct_id": "N",
        "excerpt": "1",
        "field_path": "p",
        "kind": "text_excerpt",
        "char_start": 0,
        "char_end": 1,
    }


# -- needs_clarification: one unmatched term probed as drug and condition is listed once ------------------
def test_unmatched_terms_are_deduplicated_case_insensitively():
    from ctviz.core.answerability import unmatched

    c = unmatched(["flurbozzle", "Flurbozzle", "zorp", "flurbozzle"], {"flurbozzle": ["flurbiprofen"]})
    assert c.unmatched_terms == ["flurbozzle", "zorp"]
    assert "'flurbozzle', 'zorp'" in c.message and c.message.count("flurbozzle") == 1
    assert c.suggestions.count("Did you mean 'flurbiprofen'?") == 1
