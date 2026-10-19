"""Regression tests for the Phase 0 code-review findings."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from ctviz.api.schemas import Intent, QueryPlan
from ctviz.api.schemas.citations import Citation, CitationKind
from ctviz.api.schemas.plan import CohortFilter
from ctviz.core.provenance import Dataset, GraphDataset, verify_citation
from ctviz.ctgov.client import CTGovClient, CTGovError
from ctviz.ctgov.limiter import AsyncTokenBucket
from ctviz.ctgov.requests import SearchSpec
from ctviz.store.db import Database
from tests.conftest import load_study_raw


def test_uncited_datum_gets_note_and_is_tracked():
    ds = Dataset()
    d = ds.emit({"x": "Phase 4", "y": 0}, [])
    assert ds.uncited == [d.id] and d.model_extra["cite_note"]


def test_duplicate_datum_id_rejected(study):
    ds = Dataset()
    ds.emit({"x": 1}, [study("NCT03548935").ref("phases")], datum_id="a")
    with pytest.raises(ValueError):
        ds.emit({"x": 2}, [], datum_id="a")


def test_graph_node_upsert_merges_refs(study):
    s1, s2 = study("NCT03548935"), study("NCT02578680")
    g = GraphDataset()
    g.add_node("sponsor:x", {"label": "X", "size": 1}, [s1.ref("lead_sponsor")])
    g.add_node("sponsor:x", {"label": "X", "size": 2}, [s2.ref("lead_sponsor")])
    assert len(g.nodes.rows) == 1
    node = g.nodes.rows[0]
    assert node.model_extra["size"] == 2 and len(node.citations) == 2


def test_verify_rejects_short_prefix_of_long_leaf():
    raw = load_study_raw("NCT03548935")
    good = Citation(
        nct_id="NCT03548935", excerpt='["PHASE3"]', field_path="protocolSection.designModule.phases", url="u"
    )
    stub = good.model_copy(update={"excerpt": "["})
    assert verify_citation(good, raw) and not verify_citation(stub, raw)


def test_derived_count_refs_with_different_urls_both_kept():
    from ctviz.core.provenance import derived_count_ref

    ds = Dataset()
    d = ds.emit({"y": 3}, [derived_count_ref(1, "u1", "a"), derived_count_ref(2, "u2", "b")])
    assert len(d.citations) == 2 and all(c.kind is CitationKind.DERIVED_COUNT for c in d.citations)


def test_plan_drops_malformed_nct_ids_and_normalizes_filters():
    plan = QueryPlan(
        intent=Intent.SINGLE_TRIAL_OVERVIEW, intent_confidence=1, nct_ids=["nct03548935", "../version", "NCT123"]
    )
    assert plan.nct_ids == ["NCT03548935"]
    cf = CohortFilter(label="x", phases=["Phase 2", "III", "bogus"], statuses=["Recruiting", "nope"])
    assert cf.phases == ["PHASE2", "PHASE3"] and cf.statuses == ["RECRUITING"]


@pytest.mark.parametrize("term", ["anti-PD-1 (pembrolizumab", "A AND B", "drug[x]", 'say "hi"'])
def test_essie_terms_are_quoted(term):
    p = SearchSpec.from_cohort(CohortFilter(label="x", drugs=[term])).params()
    assert p["query.intr"].startswith('"') and p["query.intr"].endswith('"')


def test_penalty_drains_bucket_and_blocks():
    now = [0.0]
    b = AsyncTokenBucket(rate_per_s=1.0, capacity=10, clock=lambda: now[0])
    b.penalize(5)
    assert b.try_acquire()[0] is False
    now[0] = 5.5
    ok, _ = b.try_acquire()
    assert ok is False or b._tokens < 1  # no burst of 10 right after the cooldown
    b.adapt(0.5)
    assert b.rate == 0.5
    for _ in range(30):
        b.adapt(1.05)
    assert b.rate == 1.0


async def test_client_single_flight_and_bad_nct(settings, tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"totalCount": 1, "studies": []})

    db = Database(tmp_path / "c.db").connect()
    client = CTGovClient(settings.model_copy(update={"offline": False}), db, transport=httpx.MockTransport(handler))
    spec = SearchSpec(cond="obesity")
    await asyncio.gather(*(client.search_page(spec) for _ in range(5)))
    assert calls["n"] == 1
    with pytest.raises(CTGovError):
        await client.get_study("../version")
    await client.aclose()


async def test_single_flight_owner_cancel_does_not_cancel_waiters(settings, tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            started.set()
            await release.wait()  # first fetch hangs until cancelled
        return httpx.Response(200, json={"totalCount": 3, "studies": []})

    db = Database(tmp_path / "s.db").connect()
    client = CTGovClient(settings.model_copy(update={"offline": False}), db, transport=httpx.MockTransport(handler))
    spec = SearchSpec(cond="asthma")
    owner = asyncio.create_task(client.search_page(spec))
    await started.wait()
    waiter = asyncio.create_task(client.search_page(spec))
    await asyncio.sleep(0)
    owner.cancel()
    page = await asyncio.wait_for(waiter, 5)
    assert page.total_count == 3 and calls["n"] == 2
    await client.aclose()
