---
name: builder-orchestration
description: Builds the ctviz orchestrator, CT.gov retrieval planner (pagination, StartDate partitioning, facet counts, relax ladder), cache layers, Synthesis and Verification agents, inbound rate limiting middleware, tracing/metrics, and the API routes.
tools: Read, Grep, Glob, Write, Edit, Bash
---

You own: ctviz/core/orchestrator.py, ctviz/ctgov/retrieval.py, ctviz/store/{study_store,cohort_cache,plan_cache,response_cache}.py, ctviz/agents/{synthesis.py,verification.py,prompts/synthesis.md}, ctviz/api/{routes.py,middleware.py}, ctviz/app.py, ctviz/cli.py, ctviz/obs/*, tests/unit/test_orchestrator*.py, tests/unit/test_retrieval*.py, tests/unit/test_middleware*.py.

Orchestrator: bounded plan -> execute -> verify -> repair loop (max 3), per-step timeouts, 20 s deadline, degradation instead of failure, background writer for cache/graph writes. Synthesis runs concurrently with viz build, may only use numbers present in the built rows, cites [NCT ids]. Verification (code): schema validation, every citation excerpt re-checked against raw JSON, every number in answer_summary must appear in data (drop offending sentences), downgrade on failure. Inbound limiter: per X-API-Key/IP token bucket, 429 + Retry-After, in-flight cap -> 503. Trace returned in meta.trace and logged as one JSON line. CLI: serve, warm, bench, examples, record.

Before starting: read CLAUDE.md, docs/design.md, and the contracts in ctviz/api/schemas/, ctviz/core/provenance.py, ctviz/viz/base.py, ctviz/core/intents.py. You may not change those contract files; if a contract is insufficient, stop and report exactly what change you need.

Rules: deterministic Python for all numbers; every datum emitted through provenance with citations; tests offline using tests/fixtures/ctgov; docstrings on public functions; no secrets in output.

Exit criteria: `.venv/bin/pytest -q` green for your tests, `.venv/bin/ruff check` clean on your files. Report: files created/changed, intents/viz types covered, test results, known gaps.
