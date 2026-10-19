"""Retrieval recipes against a respx-mocked ClinicalTrials.gov (no network)."""

from __future__ import annotations

import re
from collections import Counter

import httpx
import pytest
import respx

from ctviz.api.schemas.plan import CohortFilter
from ctviz.config import Settings
from ctviz.core.background import drain
from ctviz.ctgov.client import CTGovClient
from ctviz.ctgov.fields import FieldSet
from ctviz.ctgov.requests import SearchSpec
from ctviz.ctgov.retrieval import (
    MISSING_START,
    begin_seen,
    facet_counts,
    fetch_cohort,
    fetch_trials,
    relax_until_nonzero,
    seen_studies,
    year_partitions,
)
from ctviz.store.db import Database
from tests.conftest import load_study_raw

BASE = "https://clinicaltrials.gov/api/v2"
_RANGE = re.compile(r"AREA\[StartDate\]RANGE\[(MIN|\d{4})-?\d*-?\d*,(MAX|\d{4})-?\d*-?\d*\]")


def make_universe(n: int, missing: int = 0) -> list[dict]:
    """Synthetic minimal study records with start years spread over 2000–2024."""
    out = []
    for i in range(n):
        status = {"overallStatus": "COMPLETED"}
        if i >= missing:
            status["startDateStruct"] = {"date": f"{2000 + (i * 7) % 25}-03"}
        out.append({"protocolSection": {"identificationModule": {"nctId": f"NCT{i:08d}"}, "statusModule": status}})
    return out


def _year(raw: dict) -> int | None:
    d = raw["protocolSection"]["statusModule"].get("startDateStruct", {}).get("date")
    return int(d[:4]) if d else None


def search_handler(universe: list[dict], calls: list[dict]):
    """A /studies responder that honours StartDate RANGE/MISSING clauses, pageSize and pageToken."""

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        calls.append(params)
        adv = params.get("filter.advanced", "")
        rows = universe
        if MISSING_START in adv:
            rows = [r for r in rows if _year(r) is None]
        for lo, hi in _RANGE.findall(adv):
            lo_y = -1 if lo == "MIN" else int(lo)
            hi_y = 10_000 if hi == "MAX" else int(hi)
            rows = [r for r in rows if _year(r) is not None and lo_y <= _year(r) <= hi_y]
        size = int(params.get("pageSize", 10))
        start = int(params.get("pageToken", 0))
        page = rows[start : start + size]
        body: dict = {"studies": page}
        if params.get("countTotal") == "true":
            body["totalCount"] = len(rows)
        if start + size < len(rows):
            body["nextPageToken"] = str(start + size)
        return httpx.Response(200, json=body)

    return handler


@pytest.fixture
def online_settings(tmp_path) -> Settings:
    return Settings(offline=False, db_path=tmp_path / "r.db", ctgov_rpm=600, ctgov_burst=100, _env_file=None)  # type: ignore[call-arg]


@pytest.fixture
async def client(online_settings):
    db = Database(online_settings.db_path).connect()
    c = CTGovClient(online_settings, db, transport=httpx.AsyncHTTPTransport())
    yield c
    await drain()
    await c.aclose()
    db.close()


def test_year_partitions_cover_axis():
    years = [2000 + (i % 20) for i in range(1000)]
    parts = year_partitions(years, 2500)
    assert parts[0][0] is None and parts[-1][1] is None
    assert len(parts) == 4
    for (_, hi), (lo, _) in zip(parts, parts[1:], strict=False):
        assert lo == hi + 1  # contiguous: nothing falls between partitions


async def test_small_cohort_single_call(client):
    calls: list[dict] = []
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{BASE}/studies").mock(side_effect=search_handler(make_universe(300), calls))
        data = await fetch_cohort(
            client, CohortFilter(label="x", drugs=["x"]), field_set=FieldSet.LANDSCAPE, max_studies=3000
        )
    assert len(calls) == 1 and calls[0]["countTotal"] == "true"
    assert data.total_count == 300 and len(data.studies) == 300 and not data.sampled
    assert data.search_url.startswith("https://clinicaltrials.gov/search?")


async def test_medium_cohort_partitions_by_start_year(client):
    calls: list[dict] = []
    universe = make_universe(2500, missing=40)
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{BASE}/studies").mock(side_effect=search_handler(universe, calls))
        data = await fetch_cohort(
            client, CohortFilter(label="x", conditions=["c"]), field_set=FieldSet.LANDSCAPE, max_studies=3000
        )
    assert data.total_count == 2500
    assert len({s.nct_id for s in data.studies}) == 2500 == len(data.studies)
    assert not data.sampled
    advs = [c.get("filter.advanced", "") for c in calls[1:]]
    assert any(MISSING_START in a for a in advs)
    assert sum("RANGE[" in a for a in advs) >= 3


async def test_large_cohort_is_sampled(client):
    calls: list[dict] = []
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{BASE}/studies").mock(side_effect=search_handler(make_universe(6000), calls))
        data = await fetch_cohort(
            client, CohortFilter(label="x", drugs=["x"]), field_set=FieldSet.LANDSCAPE, max_studies=1500
        )
    assert data.sampled and data.total_count == 6000 and len(data.studies) == 1500
    assert "stratified by start year" in (data.sampling_method or "")
    # fair: every start year keeps its share of the cohort (universe years are uniform over 2000-2024)
    universe = Counter(_year(r) for r in make_universe(6000))
    got = Counter(int(s.get("start_date")[:4]) for s in data.studies)
    for y, n in universe.items():
        assert abs(got[y] - n * 1500 / 6000) <= 2, (y, got[y], n)


def test_quotas_are_proportional_and_exact():
    from ctviz.ctgov.retrieval import _quotas

    q = _quotas([5000, 3000, 1500, 500, 0], 1000)
    assert sum(q) == 1000 and q == [500, 300, 150, 50, 0]
    assert sum(_quotas([7, 7, 7], 10)) == 10 and max(_quotas([7, 7, 7], 10)) <= 7
    assert _quotas([3, 2], 100) == [3, 2]  # never more than a stratum holds


async def test_relax_ladder_order(client):
    """Zero unless countries AND years are dropped → relaxed = [countries, years]."""

    def handler(request: httpx.Request) -> httpx.Response:
        p = dict(request.url.params)
        zero = "query.locn" in p or "StartDate" in p.get("filter.advanced", "")
        return httpx.Response(200, json={"studies": [], "totalCount": 0 if zero else 7})

    cf = CohortFilter(label="x", drugs=["d"], countries=["Atlantis"], start_year=2030, phases=["PHASE3"])
    with respx.mock(assert_all_called=False) as router:
        route = router.get(f"{BASE}/studies").mock(side_effect=handler)
        relaxed, steps = await relax_until_nonzero(client, cf)
    assert steps == ["countries", "years"]
    assert relaxed.countries == [] and relaxed.start_year is None and relaxed.phases == ["PHASE3"]
    assert route.call_count == 2


async def test_relax_ladder_gives_up(client):
    cf = CohortFilter(label="x", drugs=["d"], statuses=["RECRUITING"])
    with respx.mock(assert_all_called=False) as router:
        route = router.get(f"{BASE}/studies").mock(
            return_value=httpx.Response(200, json={"studies": [], "totalCount": 0})
        )
        relaxed, steps = await relax_until_nonzero(client, cf)
    assert steps == [] and relaxed == cf
    assert route.call_count == 2  # statuses, then drugs → term
    assert "query.term" in dict(route.calls[-1].request.url.params)


async def test_retry_after_429_then_200(client):
    raw = load_study_raw("NCT07654361")
    with respx.mock(assert_all_called=False) as router:
        route = router.get(f"{BASE}/studies/NCT07654361").mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "1"}, text="<html>Too Many Requests</html>"),
                httpx.Response(200, json=raw),
            ]
        )
        studies = await fetch_trials(client, ["NCT07654361"])
    assert route.call_count == 2
    assert [s.nct_id for s in studies] == ["NCT07654361"]


async def test_fetch_trials_store_first(client):
    raw = load_study_raw("NCT03548935")
    await client.db.put_studies([("NCT03548935", "full", None, raw)])
    begin_seen()
    with respx.mock(assert_all_called=False) as router:
        route = router.get(f"{BASE}/studies/NCT03548935").mock(return_value=httpx.Response(500))
        studies = await fetch_trials(client, ["nct03548935"])
    assert route.call_count == 0
    assert studies[0].nct_id == "NCT03548935"
    assert "NCT03548935" in seen_studies()


async def test_fetch_trials_writes_back(client):
    raw = load_study_raw("NCT07654361")
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{BASE}/studies/NCT07654361").mock(return_value=httpx.Response(200, json=raw))
        await fetch_trials(client, ["NCT07654361"])
    await drain()
    assert "NCT07654361" in await client.db.get_studies(["NCT07654361"], "full")


async def test_item8_failed_partition_kept_partial_and_marked(client, monkeypatch):
    calls: list[dict] = []
    universe = make_universe(2500, missing=40)
    ok = search_handler(universe, calls)

    def flaky(request: httpx.Request) -> httpx.Response:
        if "RANGE[MIN" in dict(request.url.params).get("filter.advanced", ""):
            return httpx.Response(400, text="bad")  # non-retryable failure for the earliest-years partition
        return ok(request)

    with respx.mock(assert_all_called=False) as router:
        router.get(f"{BASE}/studies").mock(side_effect=flaky)
        data = await fetch_cohort(
            client, CohortFilter(label="x", conditions=["c"]), field_set=FieldSet.LANDSCAPE, max_studies=3000
        )
    assert data.total_count == 2500 and 1000 <= len(data.studies) < 2500  # other partitions kept
    assert data.sampled and "1 of" in data.sampling_method and "partitions failed" in data.sampling_method


async def test_item8_partition_budget_shared_across_cohorts(client):
    calls: list[dict] = []
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{BASE}/studies").mock(side_effect=search_handler(make_universe(2500), calls))
        begin_seen(partition_budget=3)
        a = await fetch_cohort(
            client, CohortFilter(label="a", conditions=["a"]), field_set=FieldSet.LANDSCAPE, max_studies=3000
        )
        b = await fetch_cohort(
            client, CohortFilter(label="b", conditions=["b"]), field_set=FieldSet.LANDSCAPE, max_studies=3000
        )
    partition_calls = [c for c in calls if "StartDate" in c.get("filter.advanced", "")]
    assert len(partition_calls) <= 3
    assert b.sampled and "budget" in (b.sampling_method or "")
    assert len(b.studies) == 1000  # page 1 only
    assert a.total_count == 2500


async def test_facet_counts_parallel(client):
    universe = make_universe(500)
    calls: list[dict] = []
    buckets = {"early": "AREA[StartDate]RANGE[MIN,2009-12-31]", "late": "AREA[StartDate]RANGE[2010-01-01,MAX]"}
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{BASE}/studies").mock(side_effect=search_handler(universe, calls))
        out = await facet_counts(client, SearchSpec(cond="c"), buckets, sample=3)
    early = sum(1 for r in universe if _year(r) < 2010)
    assert out["early"][0] == early and out["late"][0] == 500 - early
    assert len(out["early"][1]) == 3
    assert out["late"][2].startswith("https://clinicaltrials.gov/search?")
    assert len(calls) == 2


async def test_complete_read_is_snapshotted_and_reused_while_count_matches(client):
    calls: list[dict] = []
    universe = make_universe(2500, missing=40)
    cf = CohortFilter(label="x", conditions=["c"])
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{BASE}/studies").mock(side_effect=search_handler(universe, calls))
        first = await fetch_cohort(client, cf, field_set=FieldSet.LANDSCAPE, max_studies=3000)
        await drain()  # study records + snapshot are written in the background
        n_first = len(calls)
        again = await fetch_cohort(client, cf, field_set=FieldSet.LANDSCAPE, max_studies=3000)
        assert len(calls) <= n_first + 1  # at most page 1 (its totalCount confirms the snapshot), no partitions
        assert again.snapshot_at and {s.nct_id for s in again.studies} == {s.nct_id for s in first.studies}
        universe.append(make_universe(2501)[-1])  # the registry changed: the snapshot is not used
        client.db.conn.execute("DELETE FROM cache WHERE ns='http'")  # page 1 no longer cached
        client.db.conn.commit()  # reads use separate WAL reader connections: they see committed state only
        changed = await fetch_cohort(client, cf, field_set=FieldSet.LANDSCAPE, max_studies=3000)
    assert changed.snapshot_at is None and changed.total_count == 2501 == len(changed.studies)


async def test_snapshot_lets_a_cohort_above_the_live_limit_be_read_completely(client):
    calls: list[dict] = []
    universe = make_universe(2500)
    cf = CohortFilter(label="x", conditions=["c"])
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{BASE}/studies").mock(side_effect=search_handler(universe, calls))
        await fetch_cohort(client, cf, field_set=FieldSet.LANDSCAPE, max_studies=10_000)  # `ctviz warm`
        await drain()
        live = await fetch_cohort(client, cf, field_set=FieldSet.LANDSCAPE, max_studies=1000)
    assert not live.sampled and len(live.studies) == 2500 and live.counting == "all_records"


async def test_snapshot_is_not_used_for_richer_projections(client):
    calls: list[dict] = []
    universe = make_universe(2500)
    cf = CohortFilter(label="x", conditions=["c"])
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{BASE}/studies").mock(side_effect=search_handler(universe, calls))
        await fetch_cohort(client, cf, field_set=FieldSet.LANDSCAPE, max_studies=10_000)
        await drain()
        people = await fetch_cohort(client, cf, field_set=FieldSet.PEOPLE, max_studies=10_000)
    assert people.snapshot_at is None  # site maps need location fields the snapshot records may lack
