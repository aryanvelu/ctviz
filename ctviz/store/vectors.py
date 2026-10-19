"""Optional embedding re-ranker for FTS hits (``CTVIZ_EMBEDDINGS_ENABLED=1``; off by default).

Lexical FTS5 retrieval is the source of truth: embeddings can only *reorder* hits that FTS already returned,
never add new ones, so cited chunk IDs always come from retrieval. Each embeddings call is bounded by
``settings.llm_timeout_s`` (client timeout and ``asyncio.wait_for``) and retried at most twice with exponential
backoff; permission/auth errors (401/403/404) are not retried. When a call still fails, or the response is
malformed (missing indices, wrong length), the index disables itself for the process and every call degrades to a
no-op (``None`` / FTS order). Vectors are cached in the generic ``cache`` table (namespace ``embedding``).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ctviz.config import Settings
from ctviz.store.db import Database

log = logging.getLogger(__name__)
_TTL = 365 * 24 * 3600
MAX_RETRIES = 2
_NO_RETRY_STATUS = {400, 401, 403, 404, 422}


class EmbeddingResponseError(RuntimeError):
    """The embeddings response did not line up with the request."""


class VectorIndex:
    """Embeddings behind a feature flag that fail closed to no-ops."""

    def __init__(
        self,
        settings: Settings,
        db: Database | None = None,
        client_factory: Callable[[], Any] | None = None,
        backoff_s: float = 0.5,
    ):
        self.settings = settings
        self.db = db
        self.backoff_s = backoff_s
        self._client_factory = client_factory
        self._client: Any = None
        self._disabled_reason: str | None = None
        if not settings.embeddings_enabled:
            self._disabled_reason = "disabled by CTVIZ_EMBEDDINGS_ENABLED"
        elif settings.offline:
            self._disabled_reason = "offline mode"
        elif client_factory is None and settings.openai_api_key is None:
            self._disabled_reason = "no OPENAI_API_KEY"

    @property
    def enabled(self) -> bool:
        return self._disabled_reason is None

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    def _disable(self, reason: str) -> None:
        self._disabled_reason = reason
        log.warning("embeddings disabled: %s", reason)

    def _get_client(self) -> Any:
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(
                    api_key=self.settings.require_llm_key(), max_retries=0, timeout=self.settings.llm_timeout_s
                )
        return self._client

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.settings.embed_model}\x00{text}".encode()).hexdigest()

    async def _call(self, inputs: list[str]) -> list[list[float]]:
        """One embeddings request with timeout + bounded retries; vectors are mapped back by ``d.index``."""
        attempt = 0
        while True:
            try:
                resp = await asyncio.wait_for(
                    self._get_client().embeddings.create(model=self.settings.embed_model, input=inputs),
                    timeout=self.settings.llm_timeout_s,
                )
                by_index: dict[int, list[float]] = {}
                for d in resp.data:
                    by_index[int(d.index)] = [float(x) for x in d.embedding]
                if sorted(by_index) != list(range(len(inputs))):
                    raise EmbeddingResponseError(f"expected {len(inputs)} indexed vectors, got {sorted(by_index)}")
                dims = {len(v) for v in by_index.values()}
                if len(dims) != 1 or 0 in dims:
                    raise EmbeddingResponseError("inconsistent embedding dimensions")
                return [by_index[i] for i in range(len(inputs))]
            except EmbeddingResponseError:
                raise
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if attempt >= MAX_RETRIES or status in _NO_RETRY_STATUS:
                    raise
                await asyncio.sleep(self.backoff_s * (2**attempt))
                attempt += 1

    async def embed(self, texts: Sequence[str]) -> np.ndarray | None:
        """Unit-normalized ``(n, d)`` float32 matrix, or ``None`` if disabled / on any failure."""
        if not self.enabled or not texts:
            return None
        vecs: dict[int, list[float]] = {}
        if self.db is not None:
            for i, t in enumerate(texts):
                hit = await self.db.cache_get("embedding", self._key(t))
                if hit is not None:
                    vecs[i] = hit
        missing = [i for i in range(len(texts)) if i not in vecs]
        if missing:
            try:
                fresh = await self._call([texts[i] for i in missing])
                for i, v in zip(missing, fresh, strict=True):
                    vecs[i] = [round(x, 5) for x in v]
            except Exception as exc:  # timeout, 403, network, malformed response...
                self._disable(f"embedding call failed ({type(exc).__name__})")
                return None
            if self.db is not None:
                for i in missing:
                    await self.db.cache_set("embedding", self._key(texts[i]), vecs[i], _TTL)
        try:
            mat = np.asarray([vecs[i] for i in range(len(texts))], dtype=np.float32)
        except ValueError:  # cached vectors of a different dimension
            self._disable("inconsistent cached embedding dimensions")
            return None
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.where(norms == 0, 1, norms)

    async def rerank(self, query: str, texts: Sequence[str]) -> list[int]:
        """Indices of ``texts`` ordered by cosine similarity to ``query``; FTS (identity) order when disabled."""
        identity = list(range(len(texts)))
        if not self.enabled or len(texts) < 2:
            return identity
        mat = await self.embed([query, *texts])
        if mat is None or mat.shape[0] != len(texts) + 1:
            return identity
        sims = mat[1:] @ mat[0]
        return [int(i) for i in np.argsort(-sims, kind="stable")]
