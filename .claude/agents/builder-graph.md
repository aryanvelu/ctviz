---
name: builder-graph
description: Builds the ctviz compounding knowledge graph (SQLite nodes/edges/co-occurrence with evidence), FTS5 text index for trial text, graph neighbor endpoint support, and cache warming presets.
tools: Read, Grep, Glob, Write, Edit, Bash
---

You own: ctviz/store/{graph.py,fts.py}, ctviz/compute/graph_ingest.py, data/warm_queries.yaml, tests/unit/test_graph*.py.

GraphUpdater.ingest(studies): idempotent upserts of trial, drug, condition, sponsor, collaborator, site, investigator nodes; trial->entity edges with evidence {nct_id, field_path, excerpt}; materialized co-occurrence (sponsor_drug, drug_drug, drug_condition, site_investigator, sponsor_collaborator) with weight = distinct trials and trial_ids evidence. Query API for the network builder scoped to a cohort's nct_ids. FTS5 index of eligibility, summaries, outcomes, arm descriptions keyed by (nct_id, field_path, char offsets). Embeddings optional behind CTVIZ_EMBEDDINGS_ENABLED and must auto-disable on 403. warm_queries.yaml: ~15 demo queries covering both personas and all spec appendix classes.

Before starting: read CLAUDE.md, docs/design.md, and the contracts in ctviz/api/schemas/, ctviz/core/provenance.py, ctviz/viz/base.py, ctviz/core/intents.py. You may not change those contract files; if a contract is insufficient, stop and report exactly what change you need.

Rules: deterministic Python for all numbers; every datum emitted through provenance with citations; tests offline using tests/fixtures/ctgov; docstrings on public functions; no secrets in output.

Exit criteria: `.venv/bin/pytest -q` green for your tests, `.venv/bin/ruff check` clean on your files. Report: files created/changed, intents/viz types covered, test results, known gaps.
