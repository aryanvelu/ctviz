"""Optional embeddings: off by default, bounded (timeout + retries), fail closed to FTS order. No network."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ctviz.config import Settings
from ctviz.store.db import Database
from ctviz.store.vectors import MAX_RETRIES, VectorIndex

VOCAB = ["pregnan", "insulin", "cancer"]


def _settings(tmp_path, **kw) -> Settings:
    kw.setdefault("offline", False)
    return Settings(db_path=tmp_path / "v.db", _env_file=None, **kw)  # type: ignore[call-arg]


def _vec(text: str) -> list[float]:
    return [float(w in text.lower()) + 0.01 for w in VOCAB]


class _StatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _FakeEmbeddings:
    """Scripted embeddings endpoint: ``script`` items are exceptions to raise, ``"slow"``, or ``"ok"``."""

    def __init__(self, script=None, *, reverse=False, drop_last=False):
        self.script = list(script or [])
        self.calls = 0
        self.reverse = reverse
        self.drop_last = drop_last

    async def create(self, model, input):  # noqa: A002
        self.calls += 1
        step = self.script.pop(0) if self.script else "ok"
        if isinstance(step, BaseException):
            raise step
        if step == "slow":
            await asyncio.sleep(5)
        data = [SimpleNamespace(index=i, embedding=_vec(t)) for i, t in enumerate(input)]
        if self.reverse:
            data.reverse()
        if self.drop_last:
            data = data[:-1]
        return SimpleNamespace(data=data)


def _index(tmp_path, fake, db=None, **kw) -> VectorIndex:
    return VectorIndex(
        _settings(tmp_path, embeddings_enabled=True, **kw),
        db=db,
        client_factory=lambda: SimpleNamespace(embeddings=fake),
        backoff_s=0.001,
    )


TEXTS = ["insulin pump", "prior cancer", "pregnant or breastfeeding"]


async def test_default_off_is_noop(tmp_path):
    vi = VectorIndex(_settings(tmp_path, offline=True))
    assert not vi.enabled and vi.disabled_reason
    assert await vi.embed(["x"]) is None
    assert await vi.rerank("q", ["a", "b", "c"]) == [0, 1, 2]


async def test_permission_error_not_retried_and_disables(tmp_path):
    fake = _FakeEmbeddings([_StatusError(403)])
    vi = _index(tmp_path, fake)
    assert vi.enabled
    assert await vi.rerank("pregnant", ["insulin use", "pregnant women"]) == [0, 1]
    assert not vi.enabled and "_StatusError" in vi.disabled_reason
    assert fake.calls == 1
    assert await vi.embed(["again"]) is None
    assert fake.calls == 1  # disabled: never called again


async def test_transient_errors_retried_with_backoff(tmp_path):
    fake = _FakeEmbeddings([ConnectionError("reset"), _StatusError(500)])
    vi = _index(tmp_path, fake)
    assert (await vi.rerank("pregnant", TEXTS))[0] == 2
    assert fake.calls == 3 and vi.enabled


async def test_retries_exhausted_disables(tmp_path):
    fake = _FakeEmbeddings([ConnectionError("x")] * 5)
    vi = _index(tmp_path, fake)
    assert await vi.rerank("pregnant", TEXTS) == [0, 1, 2]
    assert fake.calls == MAX_RETRIES + 1 and not vi.enabled


async def test_timeout_disables(tmp_path):
    fake = _FakeEmbeddings(["slow"] * 5)
    vi = _index(tmp_path, fake, llm_timeout_s=0.05)
    assert await vi.rerank("pregnant", TEXTS) == [0, 1, 2]
    assert not vi.enabled and "Timeout" in vi.disabled_reason
    assert fake.calls == MAX_RETRIES + 1


async def test_vectors_mapped_by_index(tmp_path):
    vi = _index(tmp_path, _FakeEmbeddings(reverse=True))
    assert (await vi.rerank("pregnant", TEXTS))[0] == 2  # out-of-order response data still lines up


async def test_length_mismatch_disables_and_keeps_fts_order(tmp_path):
    vi = _index(tmp_path, _FakeEmbeddings(drop_last=True))
    assert await vi.rerank("pregnant", TEXTS) == [0, 1, 2]
    assert not vi.enabled and "EmbeddingResponseError" in vi.disabled_reason


async def test_rerank_and_cache(tmp_path):
    fake = _FakeEmbeddings()
    db = Database(tmp_path / "v.db").connect()
    vi = _index(tmp_path, fake, db=db)
    assert (await vi.rerank("pregnant", TEXTS))[0] == 2
    await vi.rerank("pregnant", TEXTS)
    assert fake.calls == 1  # second call served from the cache table
    db.close()
