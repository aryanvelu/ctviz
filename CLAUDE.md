# CLAUDE.md — ctviz (ClinicalTrials.gov Query-to-Visualization Agent)

Claude Code reads this file every session. It is the source of truth for rules, structure, and status.
Full design rationale lives in `docs/design.md` (copied from the approved plan).

## What we are building

A FastAPI backend that turns a natural-language
clinical-trial question plus optional structured fields into a **visualization spec**
(`type`, `title`, `encoding`, `data`, `meta`) backed by the ClinicalTrials.gov API v2, where
**every datum carries citations** (`nct_id` + exact excerpt / field value). Serves two personas:
patients (eligibility, duration, visits, risks, results) and researchers (landscapes, trends,
networks, design, endpoints, dropout, power, survival).

## Working rules from the user (highest priority)

1. Build one small piece at a time. Before writing code for a new piece, list the design choices
   with tradeoffs and **wait for approval**. Do not add features, frameworks, or files not asked for.
2. After each piece, explain every function in plain language and say what could break.
3. Any agent loop has a max step count, a timeout on every API call, and retries with exponential backoff.
4. Validate every argument an LLM passes to a tool before running it (types, ranges, allowed values).
5. Answers come only from retrieved sources. Every claim cites a chunk/record ID; code checks each
   cited ID was actually retrieved; if nothing relevant is found, say so instead of guessing.
6. Log every step: tool, inputs, duration, and what it returned.
7. pytest with a mocked LLM; run tests and fix failures before moving on.
8. Never hardcode secrets. Keys live in `.env` (OPENAI_API_KEY only; no Anthropic key), loaded with
   python-dotenv. Never print or commit them.
9. Never delete files or run destructive commands without asking first.
10. **The LLM never writes summaries or counts.** `answer_summary` is a deterministic template from computed facts.
11. ClinicalTrials.gov responses are cached on disk so any run can be replayed and inspected.
12. After each major milestone, walk the user through input → JSON end to end, naming files and data.

## Non-negotiable rules

1. **No LLM-generated numbers.** Every number in `visualization.data` comes from deterministic
   Python over ClinicalTrials.gov JSON. LLMs only: plan the query, pick outcome measures/arms,
   and extract verbatim text with cited chunk IDs. No LLM-written summaries.
2. **Every datum cites.** Build rows only through `core/provenance.py` (`Dataset.emit(row, prov)`).
   An excerpt is either the verbatim JSON leaf value or a verbatim substring (≤300 chars) of a text
   field with `char_start/char_end`. LLM quotes are accepted only after substring verification.
3. **Honesty over coverage.** Unanswerable parts get a `notes[]` entry (`NOT_IN_REGISTRY`,
   `APPROXIMATED`, `SAMPLED`, `NO_RESULTS_POSTED`, ...). Never fabricate survival curves, visit
   counts, costs, or power calculations; label derived curves with `meta.approximation`.
4. **Pydantic schemas in `ctviz/api/schemas/` are the contract.** Change them deliberately; regenerate
   `docs/response_schema.md` and examples when they change.
5. **Simple JSON.** Flat rows, small encodings (`{field, type, title, unit?, format?, sort?}`).
   Responses are serialized with `exclude_none=True`. Don't add nesting a frontend engineer would have to guess about.
6. **LLM and tool budget.** ≤ 4 LLM calls and ≤ 6 agent tool calls per request. Quality beats latency
   (user decision, 2026-10-18): cold cohort answers of 15–30 s are acceptable; warm answers come from
   the response cache / snapshots in ~1–3 s. Never add an LLM call to fix something code can decide.
7. **Secrets.** Never print, log, echo, or commit `OPENAI_API_KEY` or any credential. `.env` is
   gitignored; `.env.example` holds placeholders only. Before every commit:
   `scripts/secret_scan.sh` (checks .env is ignored and no key-like strings are staged).
8. **Allowed OpenAI models only:** gpt-4o-mini, gpt-4o-2024-08-06, gpt-4.1-nano, gpt-4.1-mini,
   gpt-4.1, gpt-5-mini, gpt-5, gpt-5.1, gpt-5.2, gpt-5.4-nano, gpt-5.4-mini, gpt-5.4.
   Defaults: nano for planner/synthesis/extraction, mini for analysis/escalation, gpt-5.4 only for
   `quality=max`. Embeddings are off by default (`CTVIZ_EMBEDDINGS_ENABLED=0`).
9. **Be polite to ClinicalTrials.gov** (it DOES rate-limit: ~50 req/min/IP, 429 returned as HTML): all calls go through `ctgov/client.py` (token bucket
   30/min + burst 20 so any minute stays ≤ 50, semaphore 8, backoff with jitter, http cache). Never call `httpx` directly elsewhere.
10. **Offline mode must always work** (`CTVIZ_OFFLINE=1`): fixture transport + rule planner + stub
    LLM. Tests never hit the network or OpenAI.

## Runtime agents (inside the product)

| Agent | Kind | Fires when | Budget |
|---|---|---|---|
| Router/Planner | LLM (nano), structured output, validator, rule fallback | every request | 1 call (+1 escalation) |
| Chart planner | LLM (mini) picks from a code-checked menu of buildable views | open questions only (no view named) | ≤1 call |
| Retrieval | code; LLM tool loop only on ambiguity/zero results | every request | 0–1 call, ≤3 tools |
| Analysis | LLM (mini) + deterministic stats tools | survival, power, AE, results, dropout intents | ≤1 call, ≤2 tools |
| Visualization | code (selector + 17 builders) | every request | 0 |
| Citation | code (`CitationCollector`) | every request | 0 |
| Synthesis | code (deterministic template from facts) | when `narrative=true` | 0 |
| Verification | code; LLM only to re-extract failed quotes | every request | 0–1 call |
| Extraction | LLM (nano), cached by criteria hash | eligibility / medication / biomarker intents | ≤1 per trial |

Pipeline: validate → response cache → entity resolution → plan → retrieve → (extract) → (analyze)
→ build viz + cite ‖ synthesize → verify → respond → background cache/graph writes.

## Key design decisions (settled; don't re-litigate without the user)

- **Read everything.** Read every matching trial (lean field sets, start-date partitioning) up to 25,000;
  above that use exact registry counts per bucket, a warmed cohort snapshot (`ctviz warm`), or a fair
  start-year-stratified sample, and say which in `meta.coverage` + a `SAMPLED`/`SNAPSHOT` note.
- **Status means status today.** Filters/groups: Active (recruiting, not yet, by invitation, active not
  recruiting), Done (completed, terminated, withdrawn), Other. Titles say "(status today: …)". Never split
  by status unless asked.
- **Refuse or clarify, never widen.** `unanswerable` for verdicts ("which drug is safest"); only the
  deterministic `is_subjective` regex can refuse; an LLM-only "subjective" flag is overruled and logged.
  `needs_clarification` for no entity, unknown entity, or a required trial; gibberish never becomes
  "All trials".
- **Survival and results.** Curves are reconstructed from reported medians/landmark rates (piecewise
  exponential), labelled approximate (`meta.approximation`, `APPROXIMATED`), never invented. Cross-trial
  questions read only trials with posted results (largest first), one endpoint per chart (OS default, PFS
  if asked), ≤ 8 trials / 16 arms / point budget; interpolated points carry one citation + `derived_from`
  (anchor datum ids). Forest plots show reported effect estimates, following the endpoint the question
  names; duplicates removed. A named trial without results + drug/condition words → answer for the cohort
  with a `NO_RESULTS_POSTED` note.
- **Networks are meaningful.** Same-arm co-occurrence only, placebo/saline excluded, edge weight ≥ 2,
  labelled weights, drug name variants merged (`clean_drug_name` + aliases + otherNames); the viewer uses a
  deterministic precomputed layout (no live force simulation).
- **Public JSON (schema 2.2, `ctviz/api/public.py`).** Flat rows; provenance keys last on each datum
  (`nct_ids` capped at 20 + `nct_ids_total`, `field_path`, `citations`, `citations_total`,
  `excerpts_truncated`, `derived_from`); full lists via
  `GET /v1/responses/{rid}/datums/{did}/citations` or `citation_detail=full`. Pie slices use
  `encoding.label`. Heatmap `highlight_zero` only for the landscape gap heatmap.
- **Summaries are templates over all rows** (cohort facts such as CI-below/above/crossing counts), checked
  number-by-number against the data; a sentence that fails is trimmed (`SUMMARY_TRIMMED`).
- **Every response passes a final validation step** (`meta.validation`: retrieval total, counts = trial
  lists, each trial in one bucket, citations from retrieved records, summary numbers in data, labelled
  axes); failures are flagged and never cached.
- **Caches are versioned.** Bump `ANSWER_LOGIC_VERSION` (`store/response_cache.py`) whenever an answer's
  content can change, `PLAN_CACHE_VERSION` when planning changes, `SCHEMA_VERSION` when the public JSON
  changes.

## Coding principles

- Match the surrounding code: naming, comment density, idioms; small pure functions in `compute/`,
  handlers stay thin, builders only shape data.
- Deterministic first: if code can decide it, code decides it; LLM output is an enum/ID choice that is
  validated against an allow-list before use, with a rules fallback.
- Every new behaviour gets an offline test (mocked LLM, recorded fixtures); a bug found by a reviewer or
  the user gets a regression test that would have caught it. Never weaken a test to make it pass.
- Fix root causes, not symptoms; when a contract changes, change the producer, the docs, the viewer and the
  samples together (the empty-pie bug came from a renderer that drifted from the documented contract).
- No silent fallbacks: every downgrade, truncation, sample or skipped trial is a note with a code.
- After changing behaviour: full pytest + vitest + ruff, regenerate `examples/` and run
  **viz-spec-validator**, rebuild `frontend/dist`, restart the server (or run it with `--reload`).

## Repository map

```
ctviz/config.py            settings (CTVIZ_* env, .env via python-dotenv), fail-fast key check
ctviz/app.py, cli.py       FastAPI factory; `ctviz serve|warm|bench|examples|record`
ctviz/api/                 routes, middleware (inbound rate limit, request id), schemas/ (THE contract)
ctviz/core/                orchestrator, intents registry (data/intents.yaml), entities, budget, provenance
ctviz/agents/              llm client + router, planner, rules_planner, retrieval/analysis/synthesis/verification, extractors, prompts/
ctviz/ctgov/               client, limiter, Essie request builder, field sets, retrieval (partition/facets/relax), normalize, json paths
ctviz/store/               SQLite (WAL): http/study/cohort/plan/response caches, knowledge graph, FTS
ctviz/compute/             pure functions: eligibility, timeframes, adverse_events, flow, survival, power, network, text_scan, endpoints
ctviz/viz/                 base.py (HandlerContext/HandlerResult/register/make_viz), selector.py (downgrades), builders/<one file per viz type>
ctviz/handlers/            intent handlers: cohort.py (reference: phase_distribution), trial_text.py, results.py
ctviz/obs/, offline/       trace/metrics/pricing; fixture transport, stub LLM, recorder
data/                      aliases.json, intents.yaml, warm_queries.yaml
tests/                     unit, contract, golden, bench, fixtures/ctgov (recorded API JSON)
examples/                  real request/response pairs for the README
frontend/                  Vite + React + ECharts demo renderer
scripts/try_queries.sh     test the API without the frontend
```

## API

`POST /v1/visualize` (main) · `GET /v1/schema` · `GET /v1/intents` · `GET /v1/trials/{nct_id}` ·
`GET /v1/graph/neighbors` · `GET /v1/responses/{request_id}/datums/{datum_id}/citations` ·
`GET /v1/health` · `GET /v1/metrics` · `/docs` (OpenAPI UI).

Viz types (17): kpi_cards, table, bar_chart, histogram, time_series, scatter_plot, heatmap,
pie_chart, network_graph, sankey, timeline, survival_curve, threshold_line, power_curve,
forest_plot, geo_map, eligibility_checklist.

## Commands

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # install
cp .env.example .env   # then set OPENAI_API_KEY
.venv/bin/ctviz serve --port 8000 --reload                    # run API + viewer (http://localhost:8000)
CTVIZ_OFFLINE=1 .venv/bin/ctviz serve                         # no key / no network
.venv/bin/pytest -q                                           # tests (offline)
.venv/bin/ruff check . && .venv/bin/ruff format --check .     # lint
scripts/try_queries.sh                                        # smoke-test queries via curl
.venv/bin/ctviz record                                        # refresh CT.gov fixtures
.venv/bin/ctviz examples                                      # regenerate examples/*.json
```

## Development workflow (Claude Code subagents in `.claude/agents/`)

- After each major module: run **code-reviewer** and **test-writer** in parallel on it, then fix findings.
- Whenever a test fails: run **debugger** (reproduce → root cause → minimal fix).
- Before committing examples or schema changes: run **viz-spec-validator**.
- To exercise a running server without the frontend: run **api-tester**.
- Builder subagents (`builder-*`) own disjoint files; they must not edit `api/schemas/` or
  `core/provenance.py` without the orchestrating session's approval.
- Commit after each phase with a descriptive message. Never commit `.env`, `*.db`, or `explanations.md`.

## Deliverables checklist

- [x] Code runs from a fresh clone (`pip install -e .` + `ctviz serve`), online and offline
- [x] README: run, request schema, response schema, design + tradeoffs, limitations, AI usage/validation
- [x] `docs/response_schema.md` (renderer notes per viz type) + `docs/response_schema_conventions.md`; intents are listed by `GET /v1/intents` (no separate `docs/intents.md`)
- [x] ≥ 5 real example runs in `examples/` (incl. spec's Pembrolizumab example + all appendix classes)
- [x] Deep citations on every datum, verified by tests
- [x] Tests green; latency bench table in README
- [x] Frontend demo (bonus, served at `/`) — [ ] demo video, optional deploy
- [x] Zip excludes `.env`, `.venv`, `node_modules`

