"""HTTP guard middleware: request ids, inbound rate limiting, in-flight cap, request log + metrics.

* ``X-Request-ID`` is echoed (if sane) or generated, and returned on every response. When a route produces its own
  request id (``POST /v1/visualize``: the orchestrator's ``meta.request_id``, under which citations are stored), the
  route stores it in ``request.state.request_id`` and that id is returned instead, so header and body always agree.
* Token bucket per client IP on ``/v1/*`` (a configured ``X-API-Key`` gets its own bucket) → 429 +
  ``Retry-After``. Behind ``trust_proxy`` the right-most ``X-Forwarded-For`` hop is the client.
* At most ``settings.max_inflight`` concurrent guarded requests → 503 + ``Retry-After``.
* Health, metrics, schema docs and static files are exempt.
* One JSON log line per request (never bodies, keys or patient profiles).
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import time
import uuid
from collections import OrderedDict
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ctviz.api.schemas.response import ErrorResponse
from ctviz.ctgov.limiter import AsyncTokenBucket
from ctviz.obs.logging import log_request
from ctviz.obs.metrics import METRICS

EXEMPT_PATHS = frozenset({"/v1/health", "/v1/metrics", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"})
_REQ_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
MAX_BUCKETS = 10_000


class InboundLimiter:
    """Per-caller token buckets (LRU-bounded)."""

    def __init__(self, rpm: int, burst: int):
        self.rate = rpm / 60.0
        self.burst = burst
        self._buckets: OrderedDict[str, AsyncTokenBucket] = OrderedDict()

    def check(self, key: str) -> tuple[bool, float]:
        """(allowed, retry_after_seconds)."""
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = AsyncTokenBucket(self.rate, self.burst)
            self._buckets[key] = bucket
            if len(self._buckets) > MAX_BUCKETS:
                self._buckets.popitem(last=False)
        else:
            self._buckets.move_to_end(key)
        return bucket.try_acquire()


def client_ip(request: Request, trust_proxy: bool = False) -> str:
    """The peer address; behind a trusted proxy, the right-most ``X-Forwarded-For`` hop (the one our
    proxy appended — left-most entries are client-controlled and spoofable)."""
    if trust_proxy:
        hops = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]
        if hops:
            return hops[-1]
    return request.client.host if request.client else "unknown"


def caller_key(request: Request, trust_proxy: bool = False, api_keys: frozenset[str] = frozenset()) -> str:
    """Rate-limit identity.

    Callers are bucketed by client IP. Only an ``X-API-Key`` that matches a configured key
    (``settings.api_keys``) gets its own bucket — arbitrary header values cannot mint fresh buckets.
    The key itself is never stored (only a hash prefix).
    """
    api_key = request.headers.get("x-api-key")
    if api_key and any(hmac.compare_digest(api_key, k) for k in api_keys):
        return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:16]
    return "ip:" + client_ip(request, trust_proxy)


def _route_key(path: str) -> str:
    """Collapse path parameters so metrics keys stay bounded (``/v1/trials/NCT…`` → ``/v1/trials``)."""
    parts = path.rstrip("/").split("/")
    return "/".join(parts[:3]) if len(parts) > 3 else path


def _error(status: int, error: str, message: str, retry_after: float | None, request_id: str) -> JSONResponse:
    body = ErrorResponse(
        error=error, message=message, retry_after_s=round(retry_after, 2) if retry_after is not None else None
    )
    headers = {"X-Request-ID": request_id}
    if retry_after is not None:
        headers["Retry-After"] = str(max(1, math.ceil(retry_after)))
    return JSONResponse(body.model_dump(exclude_none=True), status_code=status, headers=headers)


class GuardMiddleware(BaseHTTPMiddleware):
    """Reads limits from ``app.state.settings`` (created lazily on first request)."""

    def __init__(self, app: Any):
        super().__init__(app)
        self._limiter: InboundLimiter | None = None
        self._inflight = 0
        self._api_keys: frozenset[str] | None = None

    def _limiter_for(self, settings: Any) -> InboundLimiter:
        if self._limiter is None:
            self._limiter = InboundLimiter(settings.inbound_rpm, settings.inbound_burst)
        return self._limiter

    def _keys(self, settings: Any) -> frozenset[str]:
        if self._api_keys is None:
            self._api_keys = frozenset(k.get_secret_value() for k in getattr(settings, "api_keys", []) or [])
        return self._api_keys

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _REQ_ID.match(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        request.state.log_fields = {}
        path = request.url.path
        guarded = path.startswith("/v1/") and path not in EXEMPT_PATHS
        settings = getattr(request.app.state, "settings", None)
        t0 = time.perf_counter()
        status = 500
        try:
            if guarded and settings is not None:
                who = caller_key(request, settings.trust_proxy, self._keys(settings))
                ok, wait = self._limiter_for(settings).check(who)
                if not ok:
                    METRICS.inc("rate_limited")
                    status = 429
                    return _error(429, "rate_limited", "Too many requests; slow down.", wait, request_id)
                if self._inflight >= settings.max_inflight:
                    METRICS.inc("overloaded")
                    status = 503
                    return _error(503, "overloaded", "Server is busy; retry shortly.", 1.0, request_id)
            if guarded:
                self._inflight += 1
            try:
                response = await call_next(request)
            finally:
                if guarded:
                    self._inflight -= 1
            status = response.status_code
            request_id = getattr(request.state, "request_id", None) or request_id  # the route's id wins
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            ms = round((time.perf_counter() - t0) * 1000, 1)
            METRICS.inc("http_requests")
            METRICS.inc(f"http_status_{status}")
            if path.startswith("/v1/"):
                METRICS.observe(f"{request.method} {_route_key(path)}", ms)
                log_request(
                    request_id=request_id,
                    method=request.method,
                    path=path,
                    status=status,
                    ms=ms,
                    **getattr(request.state, "log_fields", {}),
                )
