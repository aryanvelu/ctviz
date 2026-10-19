"""The readable disk cache: every live response is saved as JSON and can be replayed offline."""

from __future__ import annotations

import json

import httpx

from ctviz.ctgov.client import CTGovClient
from ctviz.ctgov.requests import SearchSpec
from ctviz.obs.trace import trace_scope
from ctviz.store.db import Database


async def test_live_response_written_to_disk_and_replayed(settings, tmp_path):
    cache_dir = tmp_path / "cache"
    live = settings.model_copy(update={"offline": False, "cache_dir": cache_dir})
    body = {"totalCount": 7, "studies": []}
    client = CTGovClient(
        live,
        Database(tmp_path / "a.db").connect(),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)),
    )
    with trace_scope() as tr:
        page = await client.search_page(SearchSpec(cond="asthma"))
    await client.aclose()
    assert page.total_count == 7
    files = list(cache_dir.glob("*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved["body"] == body and "query.cond=asthma" in saved["url"] and saved["fetched_at"]
    assert tr.ctgov_calls_log[0]["cached"] is False and tr.ctgov_calls_log[0]["cache_file"] == str(files[0])

    # Replay: fresh database (no SQLite cache), no network, same answer from the JSON file.
    replay = settings.model_copy(update={"offline": False, "replay": True, "cache_dir": cache_dir})
    client2 = CTGovClient(replay, Database(tmp_path / "b.db").connect())
    page2 = await client2.search_page(SearchSpec(cond="asthma"))
    await client2.aclose()
    assert page2.total_count == 7


async def test_replay_missing_entry_is_a_clear_error(settings, tmp_path):
    import pytest

    from ctviz.ctgov.client import CTGovError

    replay = settings.model_copy(update={"offline": False, "replay": True, "cache_dir": tmp_path / "empty"})
    client = CTGovClient(replay, Database(tmp_path / "c.db").connect())
    with pytest.raises(CTGovError, match="404"):
        await client.search_page(SearchSpec(cond="never fetched"))
    await client.aclose()
