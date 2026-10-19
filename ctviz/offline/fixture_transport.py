"""Offline mode and fixture recording for ClinicalTrials.gov HTTP traffic.

Fixtures are gzip JSON files named by the SHA-256 of the canonical request URL (sorted params),
containing ``{"url", "status", "body"}``. ``FixtureTransport`` replays them; ``RecordingTransport``
wraps a real transport and writes every response it sees.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

import httpx


def canonical_url(url: str | httpx.URL) -> str:
    parts = urlsplit(str(url))
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return f"{parts.scheme}://{parts.netloc}{parts.path}" + (f"?{query}" if query else "")


def url_hash(url: str | httpx.URL) -> str:
    return hashlib.sha256(canonical_url(url).encode()).hexdigest()[:24]


def fixture_name(url: str | httpx.URL) -> str:
    return url_hash(url) + ".json.gz"


def cache_file_path(directory: Path, url: str | httpx.URL) -> Path:
    """Readable on-disk copy of one API response: <dir>/<hash>.json."""
    return directory / f"{url_hash(url)}.json"


def write_cache_file(directory: Path, url: str, status: int, body: object) -> Path:
    """Save ``{url, fetched_at, status, body}`` as pretty JSON so a person can open and inspect it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = cache_file_path(directory, url)
    payload = {
        "url": canonical_url(url),
        "fetched_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "status": status,
        "body": body,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    tmp.replace(path)  # atomic: readers never see a half-written file
    return path


def write_fixture(directory: Path, url: str, status: int, body: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / fixture_name(url)
    payload = {"url": canonical_url(url), "status": status, "body": json.loads(body) if body else None}
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return path


class FixtureTransport(httpx.AsyncBaseTransport):
    """Serves recorded responses; unknown URLs get a 404 explaining how to record them."""

    def __init__(self, directory: Path):
        self.directory = directory

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        plain = cache_file_path(self.directory, request.url)
        if plain.exists():  # replay from the readable disk cache
            payload = json.loads(plain.read_text())
            return httpx.Response(payload["status"], json=payload["body"], request=request)
        path = self.directory / fixture_name(request.url)
        if not path.exists():
            msg = {
                "error": "fixture_missing",
                "url": canonical_url(request.url),
                "hint": "run `ctviz record` online (or CTVIZ_RECORD=1) to capture this request",
            }
            return httpx.Response(404, json=msg, request=request)
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        return httpx.Response(payload["status"], json=payload["body"], request=request)


class RecordingTransport(httpx.AsyncBaseTransport):
    """Passes requests through to a real transport and records 200 responses as fixtures."""

    def __init__(self, directory: Path, inner: httpx.AsyncBaseTransport | None = None):
        self.directory = directory
        self.inner = inner or httpx.AsyncHTTPTransport(http2=True, retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self.inner.handle_async_request(request)
        body = await response.aread()
        if response.status_code == 200:
            write_fixture(self.directory, str(request.url), 200, body)
        # aread() already decoded the body: drop the encoding headers so httpx does not decode it twice
        headers = [
            (k, v)
            for k, v in response.headers.multi_items()
            if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")
        ]
        return httpx.Response(response.status_code, headers=headers, content=body, request=request)

    async def aclose(self) -> None:
        await self.inner.aclose()
