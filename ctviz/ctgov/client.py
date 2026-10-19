"""The only module that talks to ClinicalTrials.gov.

Responsibilities: HTTP cache (SQLite, keyed by canonical URL), outbound rate limiting, retries
with exponential backoff + jitter (honoring Retry-After), offline/record transports, and per-request
call accounting in the trace.
"""

from __future__ import annotations

import asyncio
import email.utils
import hashlib
import logging
import math
import random
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
import orjson

from ctviz.config import Settings
from ctviz.core.budget import active_budget
from ctviz.ctgov.limiter import LimiterWaitTooLong, OutboundLimiter
from ctviz.ctgov.requests import SearchSpec
from ctviz.obs.trace import current_trace
from ctviz.offline.fixture_transport import (
    FixtureTransport,
    RecordingTransport,
    cache_file_path,
    canonical_url,
    write_cache_file,
)
from ctviz.store.db import Database

log = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}
_NCT = re.compile(r"^NCT\d{8}$")
_OWNER_CANCELLED = object()  # sentinel result for single-flight waiters
MAX_ATTEMPTS = 4
MAX_RETRY_WAIT_S = 30.0  # never sleep (or block the shared limiter) longer than this after a 429
VERSION_TTL_S = 3600.0  # a successful /version answer is kept this long
VERSION_FAILURE_TTL_S = 30.0  # a failed /version lookup is retried after this long (never cached forever)


class CTGovError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, url: str | None = None):
        super().__init__(message)
        self.status = status
        self.url = url


class CTGovUnavailable(CTGovError):
    """ClinicalTrials.gov could not answer (5xx / 429 / network errors after retries, or a Retry-After longer
    than the request can wait). Distinct from a definitive answer such as 404 "no such trial"."""

    def __init__(
        self,
        message: str,
        status: int | None = None,
        url: str | None = None,
        partial: list[dict[str, Any]] | None = None,
    ):
        super().__init__(message, status, url)
        self.partial: list[dict[str, Any]] = partial or []  # records that did arrive (get_studies)


def parse_retry_after(value: str | None, now: float | None = None) -> float | None:
    """Seconds from a ``Retry-After`` header (delta-seconds or HTTP-date); None if absent/unparseable."""
    if not value:
        return None
    value = value.strip()
    try:
        secs = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if when is None:
            return None
        secs = when.timestamp() - (time.time() if now is None else now)
    if math.isnan(secs) or math.isinf(secs):
        return None
    return max(0.0, secs)


def _remaining_request_s() -> float:
    """Wall-clock left in the current request's budget (infinite outside a request)."""
    b = active_budget()
    return b.remaining_s() if b is not None else math.inf


@dataclass
class SearchPage:
    studies: list[dict[str, Any]]
    total_count: int | None
    next_page_token: str | None
    url: str


class CTGovClient:
    """Async ClinicalTrials.gov v2 client. Create once per app; share across requests."""

    def __init__(self, settings: Settings, db: Database, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.db = db
        self.base = settings.ctgov_base.rstrip("/")
        if transport is None:
            if settings.replay:
                transport = FixtureTransport(settings.cache_dir)
            elif settings.offline:
                transport = FixtureTransport(settings.fixtures_dir)
            elif settings.record:
                transport = RecordingTransport(settings.fixtures_dir)
            else:
                transport = httpx.AsyncHTTPTransport(http2=True, retries=0)
        self.http = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=10.0),
            headers={
                "User-Agent": "ctviz/0.1 (clinical-trials visualization research tool)",
                "Accept": "application/json",
            },
        )
        self.limiter = OutboundLimiter(settings.ctgov_rpm, settings.ctgov_burst, settings.ctgov_concurrency)
        self._version: dict[str, Any] | None = None
        self._version_expires = 0.0  # monotonic time after which _version is looked up again
        self._inflight: dict[str, asyncio.Future] = {}

    async def aclose(self) -> None:
        await self.http.aclose()

    # -- core GET -----------------------------------------------------------------------------
    async def get_json(
        self, path: str, params: dict[str, str] | None = None, *, ttl_s: float, refresh: bool = False
    ) -> Any:
        url = canonical_url(str(httpx.URL(self.base + path, params=params or {})))
        key = hashlib.sha256(url.encode()).hexdigest()
        tr = current_trace()
        disk = cache_file_path(self.settings.cache_dir, url)
        if not refresh and not self.settings.record:
            t0 = time.perf_counter()
            cached = await self.db.cache_get("http", key)
            if cached is not None:
                tr.ctgov_hits += 1
                tr.ctgov_calls_log.append(
                    {
                        "url": url,
                        "cached": True,
                        "status": 200,
                        "ms": round((time.perf_counter() - t0) * 1000, 1),
                        "cache_file": str(disk) if disk.exists() else None,
                    }
                )
                return cached
        # Single-flight: concurrent identical misses share one network call.
        pending = self._inflight.get(key)
        if pending is not None:
            result = await asyncio.shield(pending)
            if result is _OWNER_CANCELLED:  # the request that owned the fetch was cancelled; fetch ourselves
                return await self.get_json(path, params, ttl_s=ttl_s, refresh=refresh)
            tr.ctgov_hits += 1
            return result
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[key] = fut
        try:
            t0 = time.perf_counter()
            data = await self._fetch(url)
            tr.ctgov_misses += 1
            await self.db.cache_set("http", key, data, ttl_s)
            if not self.settings.offline and not self.settings.replay:
                await asyncio.to_thread(write_cache_file, self.settings.cache_dir, url, 200, data)
            tr.ctgov_calls_log.append(
                {
                    "url": url,
                    "cached": False,
                    "status": 200,
                    "ms": round((time.perf_counter() - t0) * 1000, 1),
                    "cache_file": str(disk) if disk.exists() else None,
                }
            )
            fut.set_result(data)
            return data
        except asyncio.CancelledError:
            # Don't propagate one request's cancellation to other requests sharing this fetch.
            fut.set_result(_OWNER_CANCELLED)
            raise
        except BaseException as exc:
            tr.ctgov_calls_log.append(
                {
                    "url": url,
                    "cached": False,
                    "status": getattr(exc, "status", None) or "error",
                    "ms": round((time.perf_counter() - t0) * 1000, 1),
                    "cache_file": None,
                    "error": type(exc).__name__,
                }
            )
            fut.set_exception(exc)
            fut.exception()  # mark retrieved so an unawaited future doesn't warn
            raise
        finally:
            self._inflight.pop(key, None)

    def _timeout(self) -> httpx.Timeout:
        """The client's HTTP timeouts, each capped at the time left in the current request."""
        left = max(0.1, _remaining_request_s())
        t = self.http.timeout
        return httpx.Timeout(
            connect=min(t.connect or left, left),
            read=min(t.read or left, left),
            write=min(t.write or left, left),
            pool=min(t.pool or left, left),
        )

    async def _fetch(self, url: str) -> Any:
        tr = current_trace()
        delay = 0.5
        for attempt in range(1, MAX_ATTEMPTS + 1):
            # Never queue for the shared limiter (429 cooldown, reduced rate, busy slots) past this request's
            # deadline: fail fast so the caller degrades with a note instead of burning the whole budget.
            try:
                await self.limiter.enter(_remaining_request_s())
            except LimiterWaitTooLong as exc:
                raise CTGovUnavailable(
                    f"ClinicalTrials.gov rate limit cooldown outlasts the time left in the request ({exc})", 429, url
                ) from exc
            try:
                tr.ctgov_calls += 1
                t0 = time.perf_counter()
                try:
                    resp = await self.http.get(url, timeout=self._timeout())
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    resp = None
                    err: Exception | None = exc
                else:
                    err = None
                log.debug(
                    "ctgov GET %s -> %s in %.0f ms",
                    url,
                    resp.status_code if resp else "ERR",
                    (time.perf_counter() - t0) * 1000,
                )
            finally:
                self.limiter.release()
            if resp is not None and resp.status_code == 200:
                tr.ctgov_urls.append(url)
                self.limiter.bucket.adapt(1.05)  # slowly recover after any earlier 429 slowdown
                try:
                    return orjson.loads(resp.content)
                except orjson.JSONDecodeError as exc:
                    raise CTGovError("ClinicalTrials.gov returned a non-JSON body", 200, url) from exc
            if resp is not None and resp.status_code not in RETRY_STATUSES:
                raise CTGovError(
                    f"ClinicalTrials.gov returned {resp.status_code}: {resp.text[:200]}", resp.status_code, url
                )
            status = resp.status_code if resp is not None else None
            if attempt == MAX_ATTEMPTS:
                raise CTGovUnavailable(
                    f"ClinicalTrials.gov unavailable after {attempt} attempts ({status or err})", status, url
                )
            wait = delay * (1 + random.random())
            if status == 429:
                retry_after = parse_retry_after(resp.headers.get("Retry-After"))  # type: ignore[union-attr]
                asked = retry_after if retry_after is not None else 5.0
                # Cap the cooldown: a hostile/huge Retry-After (e.g. 600 s) must not freeze all CT.gov traffic.
                wait = min(max(wait, asked), MAX_RETRY_WAIT_S)
                self.limiter.bucket.penalize(wait)
                self.limiter.bucket.adapt(0.5)
                remaining = _remaining_request_s()
                if asked > MAX_RETRY_WAIT_S or wait >= remaining:
                    # The server wants us gone longer than we may wait: fail this call now (the caller degrades).
                    raise CTGovUnavailable(
                        f"ClinicalTrials.gov rate-limited this call (Retry-After {asked:.0f} s; "
                        f"{min(remaining, 1e9):.0f} s left in the request)",
                        429,
                        url,
                    )
            elif wait >= _remaining_request_s():
                raise CTGovUnavailable(
                    f"ClinicalTrials.gov unavailable ({status or err}); no time left to retry", status, url
                )
            await asyncio.sleep(wait)
            delay *= 2
        raise AssertionError("unreachable")

    # -- endpoints ----------------------------------------------------------------------------
    async def search_page(
        self,
        spec: SearchSpec,
        *,
        page_size: int = 1000,
        page_token: str | None = None,
        count_total: bool = True,
        refresh: bool = False,
        fields_override: list[str] | None = None,
    ) -> SearchPage:
        params = spec.params(
            page_size=page_size, count_total=count_total, page_token=page_token, fields_override=fields_override
        )
        data = await self.get_json("/studies", params, ttl_s=self.settings.ttl_http_list_s, refresh=refresh)
        return SearchPage(
            studies=data.get("studies", []),
            total_count=data.get("totalCount"),
            next_page_token=data.get("nextPageToken"),
            url=canonical_url(str(httpx.URL(self.base + "/studies", params=params))),
        )

    async def count(self, spec: SearchSpec, *, refresh: bool = False) -> int:
        page = await self.search_page(spec, page_size=1, count_total=True, refresh=refresh, fields_override=["NCTId"])
        return int(page.total_count or 0)

    async def search_all(
        self, spec: SearchSpec, *, max_studies: int, refresh: bool = False
    ) -> tuple[list[dict[str, Any]], int]:
        """Sequentially page a search up to ``max_studies``. Returns (studies, totalCount)."""
        first = await self.search_page(spec, page_size=min(1000, max_studies), refresh=refresh)
        studies = list(first.studies)
        total = int(first.total_count or len(studies))
        token = first.next_page_token
        while token and len(studies) < max_studies:
            page = await self.search_page(
                spec,
                page_size=min(1000, max_studies - len(studies)),
                page_token=token,
                count_total=False,
                refresh=refresh,
            )
            studies.extend(page.studies)
            token = page.next_page_token
        return studies[:max_studies], total

    async def get_study(self, nct_id: str, *, refresh: bool = False) -> dict[str, Any]:
        if not _NCT.match(nct_id):
            raise CTGovError(f"invalid NCT id {nct_id!r}", 400)
        return await self.get_json(f"/studies/{nct_id}", None, ttl_s=self.settings.ttl_http_study_s, refresh=refresh)

    async def get_studies(self, nct_ids: list[str], *, refresh: bool = False) -> list[dict[str, Any]]:
        """Fetch full records concurrently (bounded by the limiter).

        Trials ClinicalTrials.gov definitively lacks (404, invalid id) are skipped. If any fetch failed because the
        service was unavailable, raises :class:`CTGovUnavailable` (with ``.partial`` = records that did arrive) so
        callers can say "ClinicalTrials.gov unavailable" instead of "trial not found".
        """
        unavailable: list[CTGovUnavailable] = []

        async def one(n: str) -> dict[str, Any] | None:
            try:
                return await self.get_study(n, refresh=refresh)
            except CTGovUnavailable as exc:
                log.warning("full record %s unavailable (service): %s", n, exc)
                unavailable.append(exc)
                return None
            except CTGovError as exc:
                log.warning("full record %s not available: %s", n, exc)
                return None

        results = await asyncio.gather(*(one(n) for n in nct_ids))
        found = [r for r in results if r]
        if unavailable:
            first = unavailable[0]
            raise CTGovUnavailable(
                f"{len(unavailable)} of {len(nct_ids)} trial record(s) could not be fetched: {first}",
                first.status,
                first.url,
                partial=found,
            )
        return found

    async def version(self) -> dict[str, Any]:
        """``/version`` (API version + data timestamp). Successes are kept for an hour; failures only briefly."""
        now = time.monotonic()
        if self._version is None or now >= self._version_expires:
            try:
                self._version = await self.get_json("/version", None, ttl_s=VERSION_TTL_S)
                self._version_expires = now + VERSION_TTL_S
            except CTGovError:
                self._version = {}
                self._version_expires = now + VERSION_FAILURE_TTL_S
        return self._version
