"""Compounding knowledge graph over every trial the service has seen (SQLite tables kg_*).

* ``kg_nodes``          -- one row per ``(type, canonical_norm)``: trial, drug, condition, sponsor, collaborator,
  site, investigator, country.
* ``kg_edges``          -- trial -> entity with evidence ``(rel, field_path, excerpt)``; the excerpt is the verbatim
  JSON leaf at ``field_path``, so it becomes a field-value citation as is.
* ``kg_cooccur_trials`` -- normalized pair membership ``(a, b, pair_type, nct_id)`` (indexed on
  ``(nct_id, pair_type)``). It is the source of truth for co-occurrence.
* ``kg_cooccur``        -- materialized summary per pair: ``weight = COUNT(membership)``; ``trial_ids`` holds only a
  capped, sorted sample (:data:`MAX_NCT_IDS`) of NCT IDs. Pairs are stored with ``a < b`` (node row ids);
  callers get them oriented by type.

Ingest semantics: facts accumulate across *different* trials. When a trial's record changes (fingerprint
mismatch) its previous edges and pair memberships are retracted before the new facts are written, for every
mention family the new record actually contains (``TrialFacts.observed``). A sparse projection, such as a
landscape record without facilities, therefore cannot erase sites learned from a full record, and evidence never
points at stale array indices of a field that was re-read. Re-ingesting an identical record is a no-op.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from ctviz.api.schemas.response import MAX_NCT_IDS_PER_DATUM
from ctviz.compute.graph_ingest import (
    NODE_TYPES,
    PAIR_DEPENDS,
    PAIR_NODE_TYPES,
    PAIR_TYPES,
    EntityKey,
    TrialFacts,
    clean_drug_name,
    extract,
    norm_text,
    normalize_drug,
)
from ctviz.core.provenance import Study
from ctviz.store.db import Database

MAX_EVIDENCE_PER_EDGE = 5
MAX_NCT_IDS = MAX_NCT_IDS_PER_DATUM  # NCT IDs listed per edge, as in /v1/visualize (``nct_ids_total`` = all)
MAX_LIMIT = 200
_CHUNK = 500
_FUZZY_CANDIDATES = 2000
_FUZZY_STOP = frozenset({"a", "an", "and", "of", "the", "in", "with", "for", "or", "to"})
_TOKEN = re.compile(r"[^\W_]+")

MatchKind = Literal["exact", "alias", "fuzzy"]


def _fingerprint(facts: TrialFacts) -> str:
    payload = json.dumps(
        [
            sorted((m.entity_type, m.canonical_norm, m.rel, m.field_path, m.excerpt) for m in facts.mentions),
            sorted((p.pair_type, p.a, p.b) for p in facts.pairs),
            sorted((a.alias_norm, a.kind, a.canonical) for a in facts.aliases),
            sorted(facts.observed),
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode()).hexdigest()


def _node_key_norm(name: str, node_type: str) -> str:
    if node_type == "drug":
        return normalize_drug(name) or norm_text(clean_drug_name(name))
    if node_type == "trial":
        return name.strip().upper()
    return norm_text(name)


def _tokens(norm: str) -> set[str]:
    return {t for t in _TOKEN.findall(norm) if t not in _FUZZY_STOP}


def _chunks(seq: Sequence[Any], n: int = _CHUNK) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def clamp_limit(limit: Any, hi: int = MAX_LIMIT) -> int:
    """Coerce a caller-supplied limit into ``1..hi``."""
    try:
        value = int(limit)
    except (TypeError, ValueError):
        raise ValueError(f"limit must be an integer, got {limit!r}") from None
    return max(1, min(hi, value))


WRITE_CHUNK = 200  # trials per ingest transaction


class GraphStore:
    """Knowledge-graph persistence and query API (lock-guarded ``*_sync`` methods + async wrappers)."""

    def __init__(self, db: Database):
        self.db = db
        self.lock = db.lock  # the Database's re-entrant connection lock (all kg_* tables are in db.SCHEMA)

    # ---------------------------------------------------------------------------------- ingest
    async def ingest(self, studies: Sequence[Study]) -> dict[str, int]:
        """Upsert nodes/edges/co-occurrences/aliases for ``studies`` (idempotent). Runs under the write lock."""
        facts = [extract(s) for s in studies]  # pure CPU work outside the write lock
        totals: dict[str, int] = {}
        for i in range(0, len(facts), WRITE_CHUNK):  # short transactions: request-path writes can interleave
            for k, v in (await self.db.write(self.ingest_facts_sync, facts[i : i + WRITE_CHUNK])).items():
                totals[k] = totals.get(k, 0) + v
        return totals

    def ingest_sync(self, studies: Sequence[Study]) -> dict[str, int]:
        """Synchronous :meth:`ingest` (CLI warmers, tests)."""
        return self.ingest_facts_sync([extract(s) for s in studies])

    def ingest_facts_sync(self, facts_list: Sequence[TrialFacts]) -> dict[str, int]:
        """Persist pre-extracted facts in one transaction. Returns counts of ingested / skipped / replaced trials."""
        with self.lock:
            return self._ingest_facts(facts_list)

    def _ingest_facts(self, facts_list: Sequence[TrialFacts]) -> dict[str, int]:
        conn = self.db.conn
        now = time.time()
        stats = {"ingested": 0, "skipped": 0, "replaced": 0}
        by_nct = {f.nct_id: f for f in facts_list}  # last occurrence wins within a batch
        try:
            existing = self._trial_fingerprints(conn, list(by_nct))
            fps = {n: _fingerprint(f) for n, f in by_nct.items()}
            todo = [f for n, f in by_nct.items() if n not in existing or existing[n] != fps[n]]
            stats["skipped"] = len(by_nct) - len(todo)
            if not todo:
                return stats
            ids: dict[EntityKey, int] = {}

            def node_id(key: EntityKey, label: str, meta: dict | None = None) -> int:
                nid = ids.get(key)
                if nid is None:
                    row = conn.execute(
                        """INSERT INTO kg_nodes(type, canonical, canonical_norm, meta, first_seen, last_seen)
                           VALUES (?,?,?,?,?,?)
                           ON CONFLICT(type, canonical_norm) DO UPDATE SET last_seen=excluded.last_seen,
                             meta=COALESCE(kg_nodes.meta, excluded.meta)
                           RETURNING id""",
                        (key[0], label, key[1], json.dumps(meta) if meta else None, now, now),
                    ).fetchone()
                    nid = ids[key] = row[0]
                return nid

            touched: set[tuple[int, int, str]] = set()
            members: set[tuple[int, int, str, str]] = set()
            edge_rows: list[tuple[int, int, str, str, str]] = []
            alias_rows: set[tuple[str, str, str, str]] = set()
            for f in todo:
                tid = node_id(("trial", f.nct_id), f.nct_id)
                if f.nct_id in existing:
                    touched |= self._retract(conn, tid, f)
                    stats["replaced"] += 1
                conn.execute(
                    "UPDATE kg_nodes SET meta=? WHERE id=?", (json.dumps({"title": f.title, "fp": fps[f.nct_id]}), tid)
                )
                for m in f.mentions:
                    meta = {k: v for k, v in (m.meta or {}).items() if v is not None and k != "role"} or None
                    edge_rows.append((tid, node_id(m.key, m.canonical, meta), m.rel, m.field_path, m.excerpt))
                for p in f.pairs:
                    a = node_id(p.a, f.labels.get(p.a, p.a[1]))
                    b = node_id(p.b, f.labels.get(p.b, p.b[1]))
                    key = (min(a, b), max(a, b), p.pair_type)
                    members.add((*key, f.nct_id))
                    touched.add(key)
                alias_rows.update((a.alias_norm, a.kind, a.canonical, a.source) for a in f.aliases)

            conn.executemany(
                "INSERT OR REPLACE INTO kg_edges(trial_id, entity_id, rel, field_path, excerpt) VALUES (?,?,?,?,?)",
                edge_rows,
            )
            conn.executemany(
                "INSERT OR IGNORE INTO kg_cooccur_trials(a, b, pair_type, nct_id) VALUES (?,?,?,?)", sorted(members)
            )
            self._refresh_cooccur(conn, touched, now)
            conn.executemany(
                "INSERT OR IGNORE INTO entity_aliases(alias_norm, kind, canonical, source) VALUES (?,?,?,?)",
                sorted(alias_rows),
            )
            conn.commit()
            stats["ingested"] = len(todo)
            return stats
        except BaseException:
            conn.rollback()
            raise

    @staticmethod
    def _retract(conn: sqlite3.Connection, tid: int, facts: TrialFacts) -> set[tuple[int, int, str]]:
        """Remove a changed trial's old edges / pair memberships for every family its new record contains."""
        rels = sorted(facts.observed)
        if rels:
            conn.execute(
                f"DELETE FROM kg_edges WHERE trial_id=? AND rel IN ({','.join('?' * len(rels))})", [tid, *rels]
            )
        pts = [pt for pt, deps in PAIR_DEPENDS.items() if deps <= facts.observed]
        if not pts:
            return set()
        marks = ",".join("?" * len(pts))
        params = [facts.nct_id, *pts]
        old = conn.execute(
            f"SELECT a, b, pair_type FROM kg_cooccur_trials WHERE nct_id=? AND pair_type IN ({marks})", params
        ).fetchall()
        conn.execute(f"DELETE FROM kg_cooccur_trials WHERE nct_id=? AND pair_type IN ({marks})", params)
        return {tuple(r) for r in old}

    @staticmethod
    def _refresh_cooccur(conn: sqlite3.Connection, keys: set[tuple[int, int, str]], now: float) -> None:
        """Recompute ``kg_cooccur`` (weight, NCT sample) for ``keys`` from the membership table."""
        members: dict[tuple[int, int, str], list[str]] = defaultdict(list)
        ordered = sorted(keys)
        for chunk in _chunks(ordered, 300):
            values = ",".join("(?,?,?)" for _ in chunk)
            q = (
                f"SELECT a, b, pair_type, nct_id FROM kg_cooccur_trials WHERE (a, b, pair_type) IN (VALUES {values}) "
                f"ORDER BY a, b, pair_type, nct_id"
            )
            for a, b, pt, nct in conn.execute(q, [x for k in chunk for x in k]):
                members[(a, b, pt)].append(nct)
        upserts = [(*k, len(v), json.dumps(v[:MAX_NCT_IDS]), now) for k, v in members.items()]
        gone = [k for k in ordered if k not in members]
        conn.executemany(
            "INSERT OR REPLACE INTO kg_cooccur(a, b, pair_type, weight, trial_ids, updated_at) VALUES (?,?,?,?,?,?)",
            upserts,
        )
        conn.executemany("DELETE FROM kg_cooccur WHERE a=? AND b=? AND pair_type=?", gone)

    @staticmethod
    def _trial_fingerprints(conn: sqlite3.Connection, ncts: list[str]) -> dict[str, str | None]:
        """``{nct_id: fingerprint or None}`` for trials already in the graph."""
        out: dict[str, str | None] = {}
        for chunk in _chunks(ncts):
            q = (
                f"SELECT canonical_norm, meta FROM kg_nodes WHERE type='trial' AND canonical_norm IN "
                f"({','.join('?' * len(chunk))})"
            )
            for nct, meta in conn.execute(q, list(chunk)):
                out[nct] = json.loads(meta).get("fp") if meta else None
        return out

    # ---------------------------------------------------------------------------------- lookup
    def resolve_sync(self, name: str, node_type: str) -> dict | None:
        """Find a node by name. The result's ``match`` is ``exact`` (normalized key), ``alias`` (alias table) or
        ``fuzzy`` (every query token appears as a whole token of the node name; most-connected wins). Trials are
        matched exactly only. Callers should surface ``fuzzy`` as "closest match"."""
        if node_type not in NODE_TYPES:
            raise ValueError(f"unknown node_type {node_type!r}; use one of {NODE_TYPES}")
        with self.lock:
            return self._resolve(name, node_type)

    def _resolve(self, name: str, node_type: str) -> dict | None:
        conn = self.db.conn
        sel = "SELECT id, type, canonical, canonical_norm, meta FROM kg_nodes WHERE type=? AND canonical_norm=?"
        norm = _node_key_norm(name, node_type)
        if not norm:
            return None
        row = conn.execute(sel, (node_type, norm)).fetchone()
        if row:
            return {**self._node_dict(row), "match": "exact"}
        if node_type == "trial":
            return None
        for (canonical,) in conn.execute(
            "SELECT canonical FROM entity_aliases WHERE alias_norm=? AND kind=? ORDER BY canonical", (norm, node_type)
        ).fetchall():
            row = conn.execute(sel, (node_type, _node_key_norm(canonical, node_type))).fetchone()
            if row:
                return {**self._node_dict(row), "match": "alias"}
        return self._fuzzy(norm, node_type)

    def _fuzzy(self, norm: str, node_type: str) -> dict | None:
        want = _tokens(norm)
        if not want:
            return None
        probe = max(want, key=len)
        like = "%" + probe.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        cands = [
            r
            for r in self.db.conn.execute(
                """SELECT id, type, canonical, canonical_norm, meta FROM kg_nodes
                   WHERE type=? AND canonical_norm LIKE ? ESCAPE '\\' LIMIT ?""",
                (node_type, like, _FUZZY_CANDIDATES),
            )
            if want <= _tokens(r[3])
        ]
        if not cands:
            return None
        counts: dict[int, int] = {}
        for chunk in _chunks([r[0] for r in cands]):
            q = (
                f"SELECT entity_id, COUNT(DISTINCT trial_id) FROM kg_edges WHERE entity_id IN "
                f"({','.join('?' * len(chunk))}) GROUP BY entity_id"
            )
            counts.update(dict(self.db.conn.execute(q, list(chunk))))
        best = min(cands, key=lambda r: (-counts.get(r[0], 0), len(_tokens(r[3])), r[3]))
        return {**self._node_dict(best), "match": "fuzzy"}

    @staticmethod
    def _node_dict(row: Sequence[Any]) -> dict:
        nid, ntype, canonical, norm, meta = row
        out = {"id": f"{ntype}:{norm}", "type": ntype, "label": canonical, "_rowid": nid}
        if meta:
            m = json.loads(meta)
            m.pop("fp", None)
            if m:
                out["meta"] = m
        return out

    def _nodes_by_rowid(self, rowids: Iterable[int]) -> dict[int, dict]:
        rowids = list(set(rowids))
        out: dict[int, dict] = {}
        for chunk in _chunks(rowids):
            q = (
                f"SELECT id, type, canonical, canonical_norm, meta FROM kg_nodes WHERE id IN "
                f"({','.join('?' * len(chunk))})"
            )
            for row in self.db.conn.execute(q, list(chunk)):
                out[row[0]] = self._node_dict(row)
        return out

    def _evidence(self, pairs: Iterable[tuple[int, int]]) -> dict[tuple[int, int], list[dict]]:
        """Map ``(trial_rowid, entity_rowid)`` -> evidence dicts (with ``rel``)."""
        pairs = list(set(pairs))
        out: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for chunk in _chunks(pairs, 400):
            values = ",".join("(?,?)" for _ in chunk)
            q = f"""SELECT e.trial_id, e.entity_id, t.canonical, e.rel, e.field_path, e.excerpt
                    FROM kg_edges e JOIN kg_nodes t ON t.id = e.trial_id
                    WHERE (e.trial_id, e.entity_id) IN (VALUES {values}) ORDER BY e.field_path"""
            for tid, eid, nct, rel, path, excerpt in self.db.conn.execute(q, [x for p in chunk for x in p]):
                out[(tid, eid)].append({"nct_id": nct, "field_path": path, "excerpt": excerpt, "rel": rel})
        return out

    def _trial_rowids(self, ncts: Iterable[str]) -> dict[str, int]:
        ncts = list(set(ncts))
        out: dict[str, int] = {}
        for chunk in _chunks(ncts):
            q = (
                f"SELECT canonical_norm, id FROM kg_nodes WHERE type='trial' AND canonical_norm IN "
                f"({','.join('?' * len(chunk))})"
            )
            out.update(dict(self.db.conn.execute(q, list(chunk))))
        return out

    # ---------------------------------------------------------------------------------- queries
    def neighbors_sync(self, name: str, node_type: str, limit: int = 50) -> dict:
        """Ego network of one entity for ``GET /v1/graph/neighbors``.

        Returns ``{center, match, nodes, edges}``. ``match`` is ``exact|alias|fuzzy`` (``None`` if not found); a
        ``fuzzy`` center should be presented as the closest match. For entities, edges are co-occurrence partners
        across all pair types, strongest first; for ``node_type='trial'`` they are the trial's entities. Every edge
        carries ``evidence`` = ``[{nct_id, field_path, excerpt}]`` (verbatim JSON leaves), ``nct_ids`` (at most
        :data:`MAX_NCT_IDS`) and ``nct_ids_total``. ``limit`` is clamped to 1..200.
        """
        limit = clamp_limit(limit)
        with self.lock:
            center = self.resolve_sync(name, node_type)
            if center is None:
                return {"center": None, "match": None, "nodes": [], "edges": []}
            match = center.pop("match")
            if node_type == "trial":
                return {**self._trial_neighbors(center, limit), "match": match}
            conn = self.db.conn
            cid = center["_rowid"]
            rows = conn.execute(
                """SELECT a, b, pair_type, weight FROM kg_cooccur WHERE a=? OR b=?
                   ORDER BY weight DESC, pair_type, a, b LIMIT ?""",
                (cid, cid, limit),
            ).fetchall()
            full_rows = [(a, b, pt, w, self._pair_ncts(a, b, pt)) for a, b, pt, w in rows]
            nodes = self._nodes_by_rowid([cid] + [r[0] for r in rows] + [r[1] for r in rows])
            edges = self._pair_edges(full_rows, nodes)
            trial_count = conn.execute(
                "SELECT COUNT(DISTINCT trial_id) FROM kg_edges WHERE entity_id=?", (cid,)
            ).fetchone()[0]
            center = {**nodes[cid], "trial_count": trial_count}
            out_nodes = [center] + [n for k, n in nodes.items() if k != cid]
            return {
                "center": _public(center),
                "match": match,
                "nodes": [_public(n) for n in out_nodes],
                "edges": edges,
            }

    def _pair_ncts(self, a: int, b: int, pair_type: str) -> list[str]:
        rows = self.db.conn.execute(
            "SELECT nct_id FROM kg_cooccur_trials WHERE a=? AND b=? AND pair_type=? ORDER BY nct_id LIMIT ?",
            (a, b, pair_type, MAX_NCT_IDS),
        )
        return [r[0] for r in rows]

    def _trial_neighbors(self, center: dict, limit: int) -> dict:
        tid = center["_rowid"]
        rows = self.db.conn.execute(
            "SELECT entity_id, rel, field_path, excerpt FROM kg_edges WHERE trial_id=? ORDER BY rel, field_path",
            (tid,),
        ).fetchall()
        grouped: dict[tuple[int, str], list[dict]] = {}
        for eid, rel, path, excerpt in rows:
            ev = grouped.setdefault((eid, rel), [])
            if len(ev) < MAX_EVIDENCE_PER_EDGE:
                ev.append({"nct_id": center["label"], "field_path": path, "excerpt": excerpt})
        keys = list(grouped)[:limit]
        nodes = self._nodes_by_rowid([eid for eid, _ in keys])
        edges = [
            {
                "source": center["id"],
                "target": nodes[eid]["id"],
                "rel": rel,
                "weight": 1,
                "nct_ids": [center["label"]],
                "nct_ids_total": 1,
                "evidence": grouped[(eid, rel)],
            }
            for eid, rel in keys
        ]
        uniq = {center["id"]: center, **{n["id"]: n for n in nodes.values()}}
        return {"center": _public(center), "nodes": [_public(n) for n in uniq.values()], "edges": edges}

    def _pair_edges(self, rows: Sequence[tuple[int, int, str, int, list[str]]], nodes: dict[int, dict]) -> list[dict]:
        """Build oriented, evidence-bearing edges from ``(a, b, pair_type, weight, nct_ids_sample)`` rows."""
        trial_rowids = self._trial_rowids(n for r in rows for n in r[4][:MAX_EVIDENCE_PER_EDGE])
        need: list[tuple[int, int]] = []
        for a, b, _pt, _w, ncts in rows:
            for nct in ncts[:MAX_EVIDENCE_PER_EDGE]:
                t = trial_rowids.get(nct)
                if t is not None:
                    need += [(t, a), (t, b)]
        ev = self._evidence(need)
        edges = []
        for a, b, pt, w, ncts in rows:
            na, nb = nodes[a], nodes[b]
            first, second = PAIR_NODE_TYPES[pt]
            if first != second and na["type"] != first:  # orient typed pairs: source has the pair's first type
                na, nb, a, b = nb, na, b, a
            evidence: list[dict] = []
            for nct in ncts[:MAX_EVIDENCE_PER_EDGE]:
                t = trial_rowids.get(nct)
                if pt == "drug_drug":
                    arm_ev = _same_arm_evidence(ev.get((t, a), []), ev.get((t, b), []))
                    if arm_ev:
                        evidence += arm_ev
                        continue
                for ent in (a, b):
                    items = ev.get((t, ent), [])
                    if pt == "drug_drug":
                        items = [x for x in items if x["rel"] == "arm_drug"] or items
                    if items:
                        x = items[0]
                        evidence.append({"nct_id": x["nct_id"], "field_path": x["field_path"], "excerpt": x["excerpt"]})
            edges.append(
                {
                    "source": na["id"],
                    "target": nb["id"],
                    "pair_type": pt,
                    "weight": w,
                    "nct_ids": ncts[:MAX_NCT_IDS],
                    "nct_ids_total": w,
                    "evidence": evidence,
                }
            )
        return edges

    def cooccurrence_sync(self, nct_ids: Sequence[str], pair_type: str, limit: int = MAX_LIMIT) -> list[dict]:
        """Co-occurring entity pairs of ``pair_type`` restricted to ``nct_ids`` (weight = trials in the set).

        Uses the ``(nct_id, pair_type)`` index of ``kg_cooccur_trials``; no full scan. Returns
        ``[{source, target, source_label, target_label, source_type, target_type, pair_type, weight, nct_ids,
        nct_ids_total, evidence}]`` sorted by weight desc then labels; ``source`` has the pair's first type (e.g. the
        sponsor for ``sponsor_drug``); ``nct_ids`` lists at most :data:`MAX_NCT_IDS`. ``limit`` is clamped to 1..200.
        """
        if pair_type not in PAIR_TYPES:
            raise ValueError(f"unknown pair_type {pair_type!r}; use one of {PAIR_TYPES}")
        limit = clamp_limit(limit)
        ncts = sorted({n.strip().upper() for n in nct_ids if n and n.strip()})
        if not ncts:
            return []
        with self.lock:
            rows = self.db.conn.execute(
                """SELECT a, b, COUNT(*) AS w, json_group_array(nct_id) AS tids FROM kg_cooccur_trials
                   WHERE nct_id IN (SELECT value FROM json_each(?)) AND pair_type = ?
                   GROUP BY a, b""",
                (json.dumps(ncts), pair_type),
            ).fetchall()
            nodes = self._nodes_by_rowid([r[0] for r in rows] + [r[1] for r in rows])
            rows.sort(key=lambda r: (-r[2], nodes[r[0]]["label"].casefold(), nodes[r[1]]["label"].casefold(), r[:2]))
            top = [(a, b, pair_type, w, sorted(json.loads(t))) for a, b, w, t in rows[:limit]]
            edges = self._pair_edges(top, nodes)
        by_id = {n["id"]: n for n in nodes.values()}
        for e in edges:
            s, t = by_id[e["source"]], by_id[e["target"]]
            e.update(source_label=s["label"], target_label=t["label"], source_type=s["type"], target_type=t["type"])
        return edges

    def stats_sync(self) -> dict[str, Any]:
        """Counts per node type, edges, co-occurrence rows per pair type, memberships and aliases."""
        with self.lock:
            conn = self.db.conn
            return {
                "nodes": dict(conn.execute("SELECT type, COUNT(*) FROM kg_nodes GROUP BY type ORDER BY type")),
                "edges": conn.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0],
                "cooccur": dict(
                    conn.execute("SELECT pair_type, COUNT(*) FROM kg_cooccur GROUP BY pair_type ORDER BY pair_type")
                ),
                "cooccur_memberships": conn.execute("SELECT COUNT(*) FROM kg_cooccur_trials").fetchone()[0],
                "aliases": conn.execute("SELECT COUNT(*) FROM entity_aliases").fetchone()[0],
            }

    def aliases_sync(self, alias: str, kind: str) -> list[str]:
        """Canonical names known for an alias (e.g. ``"Yervoy"`` -> ``["Ipilimumab"]``)."""
        norm = _node_key_norm(alias, kind)
        with self.lock:
            rows = self.db.conn.execute(
                "SELECT DISTINCT canonical FROM entity_aliases WHERE alias_norm=? AND kind=? ORDER BY canonical",
                (norm, kind),
            ).fetchall()
        return [r[0] for r in rows]

    def close_names_sync(self, name: str, node_type: str, n: int = 2) -> list[str]:
        """Spelling suggestions from every entity name seen in retrieved trials (best-connected first)."""
        import difflib

        with self.lock:
            rows = self.db.conn.execute("SELECT canonical FROM kg_nodes WHERE type=?", (node_type,)).fetchall()
        names = {r[0].casefold(): r[0] for r in rows}
        hits = difflib.get_close_matches(name.casefold(), list(names), n=n, cutoff=0.8)
        return [names[h] for h in hits]

    # ---------------------------------------------------------------------------------- async API
    async def neighbors(self, name: str, node_type: str, limit: int = 50) -> dict:
        """Async :meth:`neighbors_sync`."""
        return await self.db.run(self.neighbors_sync, name, node_type, limit)

    async def cooccurrence(self, nct_ids: list[str], pair_type: str, limit: int = MAX_LIMIT) -> list[dict]:
        """Async :meth:`cooccurrence_sync`."""
        return await self.db.run(self.cooccurrence_sync, nct_ids, pair_type, limit)

    async def resolve(self, name: str, node_type: str) -> dict | None:
        """Async :meth:`resolve_sync`."""
        return await self.db.run(self.resolve_sync, name, node_type)

    async def stats(self) -> dict[str, Any]:
        """Async :meth:`stats_sync`."""
        return await self.db.run(self.stats_sync)

    async def aliases(self, alias: str, kind: str) -> list[str]:
        """Async :meth:`aliases_sync`."""
        return await self.db.run(self.aliases_sync, alias, kind)

    async def close_names(self, name: str, node_type: str, n: int = 2) -> list[str]:
        """Async :meth:`close_names_sync`."""
        return await self.db.run(self.close_names_sync, name, node_type, n)


def _same_arm_evidence(items_a: list[dict], items_b: list[dict]) -> list[dict]:
    """Evidence entries for two drugs listed in the *same* arm group (the combination itself), if any."""

    def by_arm(items: list[dict]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for x in items:
            if x["rel"] == "arm_drug":
                out.setdefault(x["field_path"].split(".interventionNames", 1)[0], x)
        return out

    arms_a, arms_b = by_arm(items_a), by_arm(items_b)
    common = sorted(set(arms_a) & set(arms_b), key=lambda p: (len(p), p))
    if not common:
        return []
    return [{k: x[k] for k in ("nct_id", "field_path", "excerpt")} for x in (arms_a[common[0]], arms_b[common[0]])]


def _public(node: dict) -> dict:
    return {k: v for k, v in node.items() if not k.startswith("_")}
