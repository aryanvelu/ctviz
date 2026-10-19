"""TextIndex: exact offsets, chunk IDs, safe query escaping, scoped search."""

from __future__ import annotations

import pytest

from ctviz.core.provenance import verify_citation
from ctviz.ctgov.fields import P
from ctviz.ctgov.paths import path_get
from ctviz.obs.trace import trace_scope
from ctviz.store.db import Database
from ctviz.store.fts import MAX_CHUNK, TextIndex, build_match, chunk_spans, eligibility_sections, make_chunk_id

FULL = ["NCT03548935", "NCT02578680", "NCT02142738", "NCT01844505", "NCT07654361", "NCT01284335"]


def test_k_clamped_and_args_validated(tmp_path, study):
    db = Database(tmp_path / "v.db").connect()
    ti = TextIndex(db)
    ti.index_sync([study(n) for n in FULL])
    assert len(ti.search_sync("criteria", k=0, mode="any")) == 1
    assert len(ti.search_sync("criteria", k=-3, mode="any")) == 1
    many = ti.search_sync("the criteria participants study", k=10_000, mode="any")
    assert 1 <= len(many) <= 200
    with pytest.raises(ValueError):
        ti.search_sync("pregnant", mode="phrase")
    with pytest.raises(ValueError):
        build_match("pregnant", mode="ALL")
    with pytest.raises(ValueError):
        ti.search_sync("pregnant", sections=["exclusions"])
    with pytest.raises(ValueError):
        ti.search_sync("pregnant", k="lots")
    assert ti.search_sync("pregnant", sections=["exclusion", "inclusion"])
    db.close()


def test_underscore_only_tokens_dropped():
    assert build_match("___") == (None, [])
    assert build_match("_ __ ___ ( )") == (None, [])
    expr, terms = build_match("___ hba1c __ foo_bar")
    assert terms == ["hba1c", "foo_bar"] and expr == '"hba1c" AND "foo_bar"'
    assert build_match('"___" pregnant')[1] == ["pregnant"]


@pytest.fixture
def index(tmp_path, study):
    db = Database(tmp_path / "g.db").connect()
    ti = TextIndex(db)
    ti.index_sync([study(n) for n in FULL])
    yield ti
    db.close()


def _rows(ti: TextIndex):
    return ti.db.conn.execute(
        "SELECT nct_id, field_path, char_start, char_end, section, text FROM trial_text"
    ).fetchall()


def test_offsets_exact(index, study):
    raws = {n: study(n).raw for n in FULL}
    rows = _rows(index)
    assert len(rows) > 50
    for nct, path, cs, ce, _section, text in rows:
        value = path_get(raws[nct], path)
        assert value[int(cs) : int(ce)] == text
        assert 0 < len(text) <= MAX_CHUNK
        assert text == text.strip()


def test_sections(index):
    sections = {r[4] for r in _rows(index) if r[0] == "NCT03548935"}
    assert {"inclusion", "exclusion", "summary", "outcome"} <= sections
    incl = [r for r in _rows(index) if r[0] == "NCT03548935" and r[4] == "inclusion"]
    excl = [r for r in _rows(index) if r[0] == "NCT03548935" and r[4] == "exclusion"]
    assert incl[0][5].startswith("Inclusion Criteria")
    assert excl[0][5].startswith("Exclusion Criteria")
    assert max(int(r[3]) for r in incl) <= min(int(r[2]) for r in excl)


def test_chunk_spans_cover_text():
    text = ("word " * 900).strip() + "\n\n" + "Sentence one. " * 200
    spans = chunk_spans(text, max_len=500)
    assert all(0 < e - s <= 500 for s, e in spans)
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
    covered = "".join(text[s:e] for s, e in spans)
    assert covered.replace(" ", "").replace("\n", "") == text.replace(" ", "").replace("\n", "")
    assert chunk_spans("   ") == []
    assert chunk_spans("x" * 2500, max_len=1000) == [(0, 1000), (1000, 2000), (2000, 2500)]


def test_eligibility_sections_without_headings():
    assert eligibility_sections("Adults with BMI > 30") == [("eligibility", 0, 20)]
    secs = eligibility_sections("Key Inclusion Criteria:\n* a\n\nKey Exclusion Criteria:\n* b")
    assert [s for s, *_ in secs] == ["inclusion", "exclusion"]
    assert secs[0][1] == 0 and secs[0][2] == secs[1][1]


def test_search_hits_cite_and_have_chunk_ids(index, study):
    hits = index.search_sync("pregnant", k=10)
    assert hits
    stored = {make_chunk_id(r[0], r[1], int(r[2]), int(r[3])) for r in _rows(index)}
    for h in hits:
        assert h.chunk_id == f"{h.nct_id}:{h.field_path}:{h.char_start}-{h.char_end}"
        assert h.chunk_id in stored
        s = study(h.nct_id)
        value = path_get(s.raw, h.field_path)
        assert value[h.char_start : h.char_end] == h.text
        assert value[h.snippet_start : h.snippet_end] == h.snippet
        assert len(h.snippet) <= 300 and "pregnan" in h.text.lower()
        ref = s.text_ref(h.field_path, h.snippet_start, h.snippet_end)
        assert ref is not None and ref.excerpt == h.snippet and verify_citation(ref.to_citation(), s.raw)
    scores = [h.score for h in hits]
    assert scores == sorted(scores)  # bm25: lower is better


@pytest.mark.parametrize(
    "query",
    [
        'pregnant "',
        '"pregnant',
        "(pregnant",
        "pregnant)",
        "non-small-cell",
        "HbA1c (glycated)",
        "x AND OR NOT",
        "NEAR(a b)",
        "col:pregnant",
        "pregn*",
        "^start",
        "'single' \"double\"",
        "-",
        '""',
        "()",
        "a + b",
        "50%",
        "≥ 18 years",
        "drop table trial_text;--",
    ],
)
def test_query_escaping_never_raises(index, query):
    expr, _ = build_match(query)
    if expr is not None:
        assert expr.count('"') % 2 == 0
    assert isinstance(index.search_sync(query, k=5), list)


def test_escaped_hyphen_and_parens_still_match(index):
    assert index.search_sync("non-small cell (lung)", k=5)
    assert index.search_sync('"body mass index"', k=5)
    expr, terms = build_match('BMI "body mass index" (and) OR x')
    assert expr == '"body mass index" AND "BMI" AND "x"'  # bare operators/stopwords dropped, never syntax
    assert terms == ["body", "mass", "index", "BMI", "x"]


def test_no_relevant_hits_is_empty(index):
    assert index.search_sync("zzqxvy") == []
    assert index.search_sync("the and of") == []  # only stopwords
    assert index.search_sync("") == []
    assert index.search_sync("pregnant zzqxvy") == []  # AND semantics: no guessing
    assert index.search_sync("pregnant zzqxvy", mode="any")


def test_nct_and_section_filters(index):
    hits = index.search_sync("pregnant", nct_ids=["nct03548935"], k=50)
    assert hits and {h.nct_id for h in hits} == {"NCT03548935"}
    assert index.search_sync("pregnant", nct_ids=[], k=5) == []
    excl = index.search_sync("pregnant", sections=["exclusion"], k=50)
    assert excl and {h.section for h in excl} == {"exclusion"}


def test_reindex_idempotent_and_projection_safe(index, study, cohort):
    before = sorted(_rows(index))
    index.index_sync([study(n) for n in FULL])
    assert sorted(_rows(index)) == before
    # a sparse record (no eligibility/description) must not wipe what the full record taught us
    full = study("NCT03548935")
    sparse_raw = {"protocolSection": {"identificationModule": full.raw["protocolSection"]["identificationModule"]}}
    from ctviz.core.provenance import Study

    index.index_sync([Study(sparse_raw)])
    assert index.count_sync("NCT03548935") == sum(1 for r in before if r[0] == "NCT03548935")


def test_search_is_traced(index):
    with trace_scope() as tr:
        index.search_sync("pregnant", k=3)
        index.search_sync("zzqxvy", k=7)
    steps = [s for s in tr.steps if s.name == "fts_search"]
    assert len(steps) == 2
    assert steps[0].detail["query"] == "pregnant" and steps[0].detail["k"] == 3 and steps[0].detail["hits"] == 3
    assert steps[1].detail["hits"] == 0 and steps[1].ms >= 0


async def test_async_index_and_search(tmp_path, study):
    db = Database(tmp_path / "a.db").connect()
    ti = TextIndex(db)
    assert await ti.index([study("NCT02578680")]) > 0
    with trace_scope() as tr:
        hits = await ti.search("pemetrexed", k=3)
    assert hits and all(h.nct_id == "NCT02578680" for h in hits)
    assert any(s.name == "fts_search" for s in tr.steps)
    db.close()


def test_outcome_and_arm_paths(index):
    paths = {r[1] for r in _rows(index)}
    assert any(p.startswith(P["primary_outcomes"] + "[") and p.endswith(".measure") for p in paths)
    assert any(p.startswith(P["arm_groups"] + "[") and p.endswith(".description") for p in paths)
