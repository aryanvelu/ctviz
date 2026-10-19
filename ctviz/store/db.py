"""SQLite storage: one file, WAL mode. Holds caches, the study store, the knowledge graph and FTS.

All access goes through :class:`Database`. Values are zlib-compressed JSON.

Concurrency: one *writer* connection, guarded by ``Database.lock`` (SQLite allows one writer) and, for async
callers, an asyncio write lock (:meth:`Database.write`). Reads issued through :meth:`Database.run` use a separate
per-thread *reader* connection: in WAL mode readers see the last committed state and never wait for a writer, so a
long background write (study-store upserts, graph ingest) does not block request-path reads. Inside such a read
call ``Database.conn`` is the thread's reader connection and ``Database.lock`` is a no-op. Large study-store
upserts are also committed in chunks so queued writes (e.g. the HTTP cache) interleave with them.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
import zlib
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

-- Generic TTL cache: namespaces http | cohort | plan | response | extraction | analysis
CREATE TABLE IF NOT EXISTS cache (
  ns TEXT NOT NULL, key TEXT NOT NULL, value BLOB NOT NULL,
  created_at REAL NOT NULL, expires_at REAL NOT NULL,
  PRIMARY KEY (ns, key)
);
CREATE INDEX IF NOT EXISTS cache_expiry ON cache(expires_at);

-- Study store: latest raw record per trial (projected or full)
CREATE TABLE IF NOT EXISTS studies (
  nct_id TEXT PRIMARY KEY,
  level TEXT NOT NULL,                 -- 'projected' | 'full'
  last_update TEXT,                    -- lastUpdatePostDate from the record
  raw BLOB NOT NULL,
  fetched_at REAL NOT NULL
);

-- Entity alias table (grows from MeSH terms / otherNames)
CREATE TABLE IF NOT EXISTS entity_aliases (
  alias_norm TEXT NOT NULL, kind TEXT NOT NULL, canonical TEXT NOT NULL, source TEXT NOT NULL,
  PRIMARY KEY (alias_norm, kind, canonical)
);

-- Knowledge graph
CREATE TABLE IF NOT EXISTS kg_nodes (
  id INTEGER PRIMARY KEY,
  type TEXT NOT NULL,                  -- trial|drug|condition|sponsor|collaborator|site|investigator|country
  canonical TEXT NOT NULL,
  canonical_norm TEXT NOT NULL,
  meta TEXT,
  first_seen REAL NOT NULL, last_seen REAL NOT NULL,
  UNIQUE(type, canonical_norm)
);
CREATE TABLE IF NOT EXISTS kg_edges (       -- trial -> entity with evidence
  trial_id INTEGER NOT NULL, entity_id INTEGER NOT NULL, rel TEXT NOT NULL,
  field_path TEXT NOT NULL, excerpt TEXT NOT NULL,
  PRIMARY KEY (trial_id, entity_id, rel, field_path)
);
CREATE INDEX IF NOT EXISTS kg_edges_entity ON kg_edges(entity_id, rel);
CREATE TABLE IF NOT EXISTS kg_cooccur (     -- undirected entity pairs, a < b
  a INTEGER NOT NULL, b INTEGER NOT NULL, pair_type TEXT NOT NULL,
  weight INTEGER NOT NULL, trial_ids TEXT NOT NULL, updated_at REAL NOT NULL,
  PRIMARY KEY (a, b, pair_type)
);
CREATE INDEX IF NOT EXISTS kg_cooccur_b ON kg_cooccur(b, pair_type);
CREATE TABLE IF NOT EXISTS kg_cooccur_trials ( -- normalized pair membership; kg_cooccur.weight = COUNT here
  a INTEGER NOT NULL, b INTEGER NOT NULL, pair_type TEXT NOT NULL, nct_id TEXT NOT NULL,
  PRIMARY KEY (a, b, pair_type, nct_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS kg_cooccur_trials_nct ON kg_cooccur_trials(nct_id, pair_type);

-- Full-text index over trial text with provenance
CREATE VIRTUAL TABLE IF NOT EXISTS trial_text USING fts5(
  nct_id UNINDEXED, field_path UNINDEXED, char_start UNINDEXED, char_end UNINDEXED, section UNINDEXED, text
);
"""


PUT_STUDIES_CHUNK = 500  # records per write transaction in put_studies (the write lock is released in between)

_tls = threading.local()  # .reader_of = the Database whose read call this worker thread is running


class _WriterLock:
    """Re-entrant lock for the shared writer connection; a no-op inside a :meth:`Database.run` read call, where the
    thread uses its own reader connection and must not wait behind a long write."""

    def __init__(self, db: Database):
        self._db = db
        self._lock = threading.RLock()

    def _bypass(self) -> bool:
        return getattr(_tls, "reader_of", None) is self._db

    def __enter__(self) -> _WriterLock:
        if not self._bypass():
            self._lock.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        if not self._bypass():
            self._lock.release()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        self._lock.release()


def _pack(value: Any) -> bytes:
    return zlib.compress(json.dumps(value, separators=(",", ":")).encode(), 3)


def _unpack(blob: bytes) -> Any:
    return json.loads(zlib.decompress(blob))


class Database:
    """Thin async wrapper over sqlite3 (calls run in a worker thread)."""

    def __init__(self, path: Path | str):
        self.path = str(path)
        self._conn: sqlite3.Connection | None = None
        self._write_lock = asyncio.Lock()
        # The writer connection is shared across worker threads; every statement on it runs under this lock.
        self.lock = _WriterLock(self)
        self._readers: dict[int, sqlite3.Connection] = {}  # thread id -> reader connection (WAL snapshot reads)
        self._readers_lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------------------------
    def connect(self) -> Database:
        with self.lock:
            return self._connect_locked()

    def _connect_locked(self) -> Database:
        if self._conn is None:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(SCHEMA)
        return self

    @property
    def conn(self) -> sqlite3.Connection:
        """The writer connection, or this thread's reader connection inside a :meth:`run` read call."""
        if getattr(_tls, "reader_of", None) is self:
            return self._reader()
        return self.connect()._conn  # type: ignore[return-value]

    @property
    def readers_enabled(self) -> bool:
        return self.path != ":memory:" and not self.path.startswith("file::memory:")  # :memory: is per-connection

    def _reader(self) -> sqlite3.Connection:
        tid = threading.get_ident()
        with self._readers_lock:
            conn = self._readers.get(tid)
        if conn is None:
            if self._conn is None:
                with self.lock._lock:  # real lock even inside a read call: the writer creates schema + WAL first
                    self._connect_locked()
            conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA query_only=ON")  # a read call can never write by accident
            with self._readers_lock:
                self._readers[tid] = conn
        return conn

    def close(self) -> None:
        with self._readers_lock:
            readers, self._readers = list(self._readers.values()), {}
        for r in readers:
            r.close()
        with self.lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _locked(self, fn, *args):  # noqa: ANN001, ANN202
        with self.lock:
            return fn(*args)

    def _read_call(self, fn, *args):  # noqa: ANN001, ANN202
        prev = getattr(_tls, "reader_of", None)
        _tls.reader_of = self
        try:
            return fn(*args)
        finally:
            _tls.reader_of = prev

    async def run(self, fn, *args):  # noqa: ANN001
        """Run a read-only sync DB function in a worker thread on that thread's reader connection (never blocked by
        a write in progress). For ``:memory:`` databases it falls back to the locked writer connection."""
        if not self.readers_enabled:
            return await asyncio.to_thread(self._locked, fn, *args)
        return await asyncio.to_thread(self._read_call, fn, *args)

    async def write(self, fn, *args):  # noqa: ANN001
        async with self._write_lock:
            return await asyncio.to_thread(self._locked, fn, *args)

    # -- generic TTL cache --------------------------------------------------------------------
    def cache_get_sync(self, ns: str, key: str) -> Any | None:
        row = self.conn.execute("SELECT value, expires_at FROM cache WHERE ns=? AND key=?", (ns, key)).fetchone()
        if row is None or row[1] < time.time():
            return None
        return _unpack(row[0])

    def cache_set_sync(self, ns: str, key: str, value: Any, ttl_s: float) -> None:
        now = time.time()
        self.conn.execute(
            "INSERT OR REPLACE INTO cache(ns, key, value, created_at, expires_at) VALUES (?,?,?,?,?)",
            (ns, key, _pack(value), now, now + ttl_s),
        )
        self.conn.commit()

    async def cache_get(self, ns: str, key: str) -> Any | None:
        return await self.run(self.cache_get_sync, ns, key)

    async def cache_set(self, ns: str, key: str, value: Any, ttl_s: float) -> None:
        await self.write(self.cache_set_sync, ns, key, value, ttl_s)

    def sweep_sync(self) -> int:
        cur = self.conn.execute("DELETE FROM cache WHERE expires_at < ?", (time.time(),))
        self.conn.commit()
        return cur.rowcount

    # -- study store --------------------------------------------------------------------------
    def put_studies_sync(self, records: list[tuple[str, str, str | None, Any]]) -> None:
        """Upsert ``(nct_id, level, last_update, raw)``; never downgrade a full record to projected."""
        now = time.time()
        self.conn.executemany(
            """INSERT INTO studies(nct_id, level, last_update, raw, fetched_at) VALUES (?,?,?,?,?)
               ON CONFLICT(nct_id) DO UPDATE SET level=excluded.level, last_update=excluded.last_update,
                 raw=excluded.raw, fetched_at=excluded.fetched_at
               WHERE studies.level='projected' OR excluded.level='full'""",
            [(n, lvl, upd, _pack(raw), now) for n, lvl, upd, raw in records],
        )
        self.conn.commit()

    def get_studies_sync(self, nct_ids: list[str], level: str | None = None) -> dict[str, Any]:
        if not nct_ids:
            return {}
        out: dict[str, Any] = {}
        for i in range(0, len(nct_ids), 500):
            chunk = nct_ids[i : i + 500]
            q = f"SELECT nct_id, level, raw FROM studies WHERE nct_id IN ({','.join('?' * len(chunk))})"
            for nct, lvl, raw in self.conn.execute(q, chunk):
                if level is None or lvl == level:
                    out[nct] = _unpack(raw)
        return out

    async def put_studies(self, records: list[tuple[str, str, str | None, Any]]) -> None:
        """Upsert in chunks of :data:`PUT_STUDIES_CHUNK` so other writes interleave with a large background save."""
        for i in range(0, len(records), PUT_STUDIES_CHUNK):
            await self.write(self.put_studies_sync, records[i : i + PUT_STUDIES_CHUNK])

    async def get_studies(self, nct_ids: list[str], level: str | None = None) -> dict[str, Any]:
        return await self.run(self.get_studies_sync, nct_ids, level)
