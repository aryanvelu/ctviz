"""FTS5 index over trial text with exact provenance (table ``trial_text``).

Every row is a chunk ``raw_field[char_start:char_end]`` of one text field (``text`` is that exact slice, so
it can be cited) tagged with a ``section``: ``eligibility`` (preamble), ``inclusion``, ``exclusion``,
``summary``, ``description``, ``outcome``, ``arm``, ``intervention``. Chunks are at most :data:`MAX_CHUNK`
characters and break at paragraph / line / sentence / word boundaries.

Each chunk has a stable ``chunk_id = f"{nct_id}:{field_path}:{char_start}-{char_end}"``. Answers may cite
only chunk IDs returned by :meth:`TextIndex.search`, so callers can check every cited ID was retrieved.

Re-indexing a trial replaces its rows per *source group* (e.g. all outcome rows) only when the new record
contains that group, so a sparse projection never wipes text learned from a full record.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from typing import NamedTuple

from ctviz.api.schemas.citations import MAX_EXCERPT_CHARS
from ctviz.core.provenance import Study
from ctviz.ctgov.fields import P
from ctviz.ctgov.paths import path_get
from ctviz.obs.trace import current_trace
from ctviz.store.db import Database

MAX_CHUNK = 1200
SNIPPET_CHARS = MAX_EXCERPT_CHARS
MAX_K = 200
MODES = ("all", "any")
SECTIONS = ("eligibility", "inclusion", "exclusion", "summary", "description", "outcome", "arm", "intervention")

# source group -> (container path that must exist in the record, field_path prefix of its rows)
_GROUPS: dict[str, tuple[str, str]] = {
    "eligibility": (P["eligibility_criteria"], P["eligibility_criteria"]),
    "summary": (P["brief_summary"], P["brief_summary"]),
    "description": (P["detailed_description"], P["detailed_description"]),
    "primary_outcomes": (P["primary_outcomes"], P["primary_outcomes"] + "["),
    "secondary_outcomes": (P["secondary_outcomes"], P["secondary_outcomes"] + "["),
    "other_outcomes": ("protocolSection.outcomesModule.otherOutcomes", "protocolSection.outcomesModule.otherOutcomes["),
    "arms": (P["arm_groups"], P["arm_groups"] + "["),
    "interventions": (P["interventions"], P["interventions"] + "["),
}
_HEADING = re.compile(r"(?im)^[ \t*#-]*(?:key\s+)?(inclusion|exclusion)\s+criteria\b[^\n]*$")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "do",
        "does",
        "can",
        "my",
        "me",
        "am",
    }
)
_TOKEN = re.compile(r"\w+", re.UNICODE)
_PHRASE = re.compile(r'"([^"]+)"')


def make_chunk_id(nct_id: str, field_path: str, char_start: int, char_end: int) -> str:
    """Stable identifier for one indexed chunk."""
    return f"{nct_id}:{field_path}:{char_start}-{char_end}"


class Chunk(NamedTuple):
    nct_id: str
    field_path: str
    char_start: int
    char_end: int
    section: str
    text: str

    @property
    def chunk_id(self) -> str:
        return make_chunk_id(self.nct_id, self.field_path, self.char_start, self.char_end)


class SearchHit(NamedTuple):
    """One FTS hit. ``char_start/char_end`` delimit the chunk (``raw_field[char_start:char_end] == text``);
    ``snippet`` is a verbatim excerpt (<= 300 chars) of it at ``raw_field[snippet_start:snippet_end]``,
    directly usable as a text-excerpt citation."""

    nct_id: str
    field_path: str
    char_start: int
    char_end: int
    snippet: str
    chunk_id: str
    section: str
    score: float
    snippet_start: int
    snippet_end: int
    text: str


# -- chunking ------------------------------------------------------------------------------------
def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def chunk_spans(text: str, start: int = 0, end: int | None = None, max_len: int = MAX_CHUNK) -> list[tuple[int, int]]:
    """Split ``text[start:end]`` into whitespace-trimmed spans of at most ``max_len`` chars."""
    end = len(text) if end is None else end
    out: list[tuple[int, int]] = []
    s, e = _trim(text, start, end)
    while s < e:
        if e - s <= max_len:
            out.append((s, e))
            break
        cut = -1
        for sep in ("\n\n", "\n", ". ", "; ", " "):
            idx = text.rfind(sep, s + max_len // 3, s + max_len)
            if idx > s:
                cut = idx + len(sep)
                break
        if cut <= s:
            cut = s + max_len
        cs, ce = _trim(text, s, cut)
        if ce > cs:
            out.append((cs, ce))
        s, e = _trim(text, cut, e)
    return out


def eligibility_sections(text: str) -> list[tuple[str, int, int]]:
    """``(section, start, end)`` spans: preamble ``eligibility``, then ``inclusion`` / ``exclusion`` blocks."""
    heads = list(_HEADING.finditer(text))
    if not heads:
        return [("eligibility", 0, len(text))]
    spans = []
    if text[: heads[0].start()].strip():
        spans.append(("eligibility", 0, heads[0].start()))
    for i, m in enumerate(heads):
        stop = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        spans.append((m.group(1).lower(), m.start(), stop))
    return spans


def study_chunks(study: Study) -> list[Chunk]:
    """All indexable chunks of one study, with exact offsets into each raw text field."""
    raw, nct = study.raw, study.nct_id
    out: list[Chunk] = []

    def add(path: str, section: str, spans: Iterable[tuple[int, int]] | None = None) -> None:
        value = path_get(raw, path)
        if not isinstance(value, str) or not value.strip():
            return
        for s, e in spans if spans is not None else chunk_spans(value):
            out.append(Chunk(nct, path, s, e, section, value[s:e]))

    crit = path_get(raw, P["eligibility_criteria"])
    if isinstance(crit, str):
        for section, s, e in eligibility_sections(crit):
            add(P["eligibility_criteria"], section, chunk_spans(crit, s, e))
    add(P["brief_summary"], "summary")
    add(P["detailed_description"], "description")
    for key in ("primary_outcomes", "secondary_outcomes", "other_outcomes"):
        container = _GROUPS[key][0]
        for i, _ in enumerate(path_get(raw, container) or []):
            for leaf in ("measure", "description", "timeFrame"):
                add(f"{container}[{i}].{leaf}", "outcome")
    for i, _ in enumerate(path_get(raw, P["arm_groups"]) or []):
        add(f"{P['arm_groups']}[{i}].description", "arm")
    for i, _ in enumerate(path_get(raw, P["interventions"]) or []):
        add(f"{P['interventions']}[{i}].description", "intervention")
    return out


def _group_of(field_path: str) -> str | None:
    for name, (_, prefix) in _GROUPS.items():
        if field_path == prefix or field_path.startswith(prefix):
            return name
    return None


# -- query building ------------------------------------------------------------------------------
def _word_tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text) if any(c.isalnum() for c in t)]


def build_match(query: str, mode: str = "all") -> tuple[str | None, list[str]]:
    """Turn free user text into a safe FTS5 MATCH expression.

    Only word tokens survive and every term is double-quoted, so quotes, parentheses, hyphens, ``*``, ``:``,
    ``NEAR`` or ``AND/OR/NOT`` in user text can never act as FTS syntax. ``"quoted phrases"`` in the query are
    kept as phrases. ``mode='all'`` ANDs terms, ``'any'`` ORs them. Tokens without a letter or digit (e.g. ``___``)
    are dropped. Returns ``(expr or None, terms)``; raises ``ValueError`` for an unknown ``mode``.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    parts: list[str] = []
    terms: list[str] = []
    for phrase in _PHRASE.findall(query):
        toks = _word_tokens(phrase)
        if toks:
            parts.append('"' + " ".join(toks) + '"')
            terms += toks
    rest = _PHRASE.sub(" ", query)
    for tok in _word_tokens(rest):
        if tok.casefold() in _STOPWORDS or tok.casefold() in (t.casefold() for t in terms):
            continue
        parts.append(f'"{tok}"')
        terms.append(tok)
    if not parts:
        return None, []
    return (" AND " if mode == "all" else " OR ").join(parts), terms


def _snippet_window(text: str, terms: Sequence[str], size: int = SNIPPET_CHARS) -> tuple[int, int]:
    """Local ``(start, end)`` of a <= ``size`` window of ``text`` around the first matching term."""
    if len(text) <= size:
        return 0, len(text)
    pos = len(text)
    for t in terms:
        m = re.search(rf"(?<!\w){re.escape(t)}", text, re.I)
        if m and m.start() < pos:
            pos = m.start()
    if pos == len(text):
        pos = 0
    s = max(0, pos - size // 3)
    if s > 0:
        sp = text.find(" ", s, pos) if pos > s else -1
        s = sp + 1 if sp >= 0 else s
    e = min(len(text), s + size)
    if e < len(text):
        sp = text.rfind(" ", s + size // 2, e)
        e = sp if sp > s else e
    return _trim(text, s, e)


class TextIndex:
    """FTS5 lexical retrieval over trial text (the RAG store)."""

    def __init__(self, db: Database):
        self.db = db
        self.lock = db.lock

    def index_sync(self, studies: Sequence[Study]) -> int:
        """Index (or re-index) studies in one transaction; returns rows inserted."""
        with self.lock:
            return self._index(studies)

    def _index(self, studies: Sequence[Study]) -> int:
        conn = self.db.conn
        by_nct = {s.nct_id: s for s in studies}
        if not by_nct:
            return 0
        rows: list[Chunk] = []
        present: dict[str, set[str]] = {}
        for nct, s in by_nct.items():
            present[nct] = {
                g for g, (container, _) in _GROUPS.items() if path_get(s.raw, container) not in (None, "", [])
            }
            rows += study_chunks(s)
        try:
            stale = [
                rowid
                for rowid, nct, path in conn.execute(
                    "SELECT rowid, nct_id, field_path FROM trial_text WHERE nct_id IN (SELECT value FROM json_each(?))",
                    (json.dumps(list(by_nct)),),
                )
                if _group_of(path) in present.get(nct, set())
            ]
            conn.executemany("DELETE FROM trial_text WHERE rowid=?", [(r,) for r in stale])
            conn.executemany(
                "INSERT INTO trial_text(nct_id, field_path, char_start, char_end, section, text) VALUES (?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        return len(rows)

    async def index(self, studies: Sequence[Study]) -> int:
        """Async :meth:`index_sync` under the database write lock, in short chunks so request-path writes
        (HTTP cache) can interleave."""
        n = 0
        for i in range(0, len(studies), 200):
            n += await self.db.write(self.index_sync, studies[i : i + 200])
        return n

    def search_sync(
        self,
        query: str,
        nct_ids: Sequence[str] | None = None,
        k: int = 20,
        mode: str = "all",
        sections: Sequence[str] | None = None,
    ) -> list[SearchHit]:
        """BM25-ranked hits for ``query`` (optionally restricted to ``nct_ids`` / ``sections``).

        Returns ``[]`` when nothing matches -- there is no fuzzy or fallback retrieval. ``k`` is clamped to
        1..200; ``mode`` must be ``all|any`` and ``sections`` known :data:`SECTIONS` names (else ``ValueError``).
        Each search is recorded as a ``fts_search`` step in the current request trace (query, k, ms, hits).
        """
        try:
            k = max(1, min(MAX_K, int(k)))
        except (TypeError, ValueError):
            raise ValueError(f"k must be an integer, got {k!r}") from None
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if sections is not None:
            unknown = sorted(set(sections) - set(SECTIONS))
            if unknown:
                raise ValueError(f"unknown sections {unknown}; use any of {SECTIONS}")
        with current_trace().step("fts_search", query=query[:200], k=k) as info, self.lock:
            hits = self._search(query, nct_ids, k, mode, sections)
            info["hits"] = len(hits)
            return hits

    def _search(
        self, query: str, nct_ids: Sequence[str] | None, k: int, mode: str, sections: Sequence[str] | None
    ) -> list[SearchHit]:
        expr, terms = build_match(query, mode)
        if expr is None or (nct_ids is not None and not nct_ids):
            return []
        sql = (
            "SELECT nct_id, field_path, char_start, char_end, section, text, bm25(trial_text) AS score "
            "FROM trial_text WHERE trial_text MATCH ?"
        )
        params: list = [expr]
        if nct_ids is not None:
            sql += " AND nct_id IN (SELECT value FROM json_each(?))"
            params.append(json.dumps([n.strip().upper() for n in nct_ids]))
        if sections:
            sql += " AND section IN (SELECT value FROM json_each(?))"
            params.append(json.dumps(list(sections)))
        sql += " ORDER BY score, nct_id, field_path, char_start LIMIT ?"
        params.append(int(k))
        hits: list[SearchHit] = []
        for nct, path, cs, ce, section, text, score in self.db.conn.execute(sql, params):
            ls, le = _snippet_window(text, terms)
            hits.append(
                SearchHit(
                    nct,
                    path,
                    int(cs),
                    int(ce),
                    text[ls:le],
                    make_chunk_id(nct, path, int(cs), int(ce)),
                    section,
                    round(float(score), 4),
                    int(cs) + ls,
                    int(cs) + le,
                    text,
                )
            )
        return hits

    async def search(
        self,
        query: str,
        nct_ids: Sequence[str] | None = None,
        k: int = 20,
        mode: str = "all",
        sections: Sequence[str] | None = None,
    ) -> list[SearchHit]:
        """Async :meth:`search_sync` (the worker thread shares the request trace via the copied context)."""
        return await self.db.run(self.search_sync, query, nct_ids, k, mode, sections)

    def count_sync(self, nct_id: str | None = None) -> int:
        """Number of indexed chunks (optionally for one trial)."""
        with self.lock:
            if nct_id is None:
                return self.db.conn.execute("SELECT COUNT(*) FROM trial_text").fetchone()[0]
            return self.db.conn.execute("SELECT COUNT(*) FROM trial_text WHERE nct_id=?", (nct_id,)).fetchone()[0]
