# ctviz design detail

Part 1 is the shipped design in more detail than the README. Part 2 is the original approved plan, kept for
the record.

# Part 1: Shipped design

## Pipeline

One pipeline for every question (`ctviz/core/orchestrator.py`). Each step is logged with its inputs, duration
and result (`ctviz/obs/`, `runs/<request_id>/steps.jsonl`).

| # | Step | What happens | Where |
|---|---|---|---|
| 1 | Validate | Pydantic request model; inbound per-IP rate limit and in-flight cap | `api/schemas/request.py`, `api/middleware.py` |
| 2 | Response cache | Same request (minus trace/refresh flags) within 1 h → stored answer | `store/response_cache.py` |
| 3 | Plan | Rule plan always computed; LLM planner (gpt-5.4-nano, structured output) proposes a `QueryPlan`; a validator grounds every entity in the alias table, the question text or a verified span, and drops filters the user did not ask for. LLM failure → rule plan | `agents/planner.py`, `agents/rules_planner.py` |
| 4 | Answerability | "Which drug is safest?" → `unanswerable`. Only the deterministic `is_subjective` check can refuse; an LLM-only "subjective" flag is overruled and logged as a repair | `core/answerability.py` |
| 5 | Entity verification | Names outside the vocabulary get a count probe. Zero trials → `needs_clarification` with "did you mean" suggestions (vocabulary + every name the knowledge graph has seen). A name that trial records list as another name of a known drug (e.g. a code name) is searched under both | `orchestrator.py::_verify_entities`, `_kg_other_names` |
| 6 | Retrieval | Read every matching trial up to 25,000 (see below) | `ctgov/retrieval.py` |
| 7 | Chart planner | One validated LLM call (gpt-5.4-mini) sees a category profile of the retrieved cohort and picks the most informative view from a menu the code has checked is buildable. It cannot override a view the question names or a requested chart type, and never sees trial records | `agents/chart_planner.py` |
| 8 | Handler | Deterministic compute over the records; rows built only through `Dataset.emit(values, refs)` so each datum carries its evidence | `handlers/`, `compute/`, `core/provenance.py` |
| 9 | Summary | `answer_summary` from templates over computed facts. No LLM text reaches the response | `agents/synthesis.py` |
| 10 | Verification | Every inline citation re-checked against the raw record (exact leaf or exact substring at `char_start:char_end`); every number in the summary must appear in the data, else the sentence is dropped | `agents/verification.py` |
| 11 | Validation | Six invariant checks (`meta.validation`); a failure adds a warning and the answer is not cached | `agents/validation.py` |
| 12 | Projection | Internal model → public JSON | `api/public.py` |
| 13 | Background | Citation store, response cache, knowledge-graph ingest, FTS index | `orchestrator.py::_background` |

## Retrieval: read everything, up to 25,000 trials

ClinicalTrials.gov has no aggregation API, and pages hold at most 1,000 records with sequential page tokens.
`fetch_cohort` (`ctgov/retrieval.py`):

1. Page 1 with `countTotal`. Done if it holds every match.
2. Up to 25,000 matches: split the `StartDate` axis into up to 8 year ranges of similar size (from page 1's
   start years) plus a `MISSING` partition, and page them concurrently. A per-request budget of 40 partition
   calls applies; if it runs out, the result is flagged as a partial sample.
3. More than 25,000: if a **cohort snapshot** exists for the same search and ClinicalTrials.gov still reports
   the same `totalCount`, every member is read from the local study store (`SNAPSHOT` note). Otherwise:
   * for one-dimensional views that ClinicalTrials.gov can count server-side (trend per year, phase, status,
     country, sponsor class, intervention type), only page 1 is kept and each bucket is an **exact
     registry count** (one filtered `countTotal` call per bucket, `query_url` per datum, reconciled against
     the total by the validation step);
   * otherwise (networks, heatmaps, histograms…), a start-year-stratified sample, flagged `SAMPLED`.

## Citations per datum

* Every count is checked against its complete trial list (`counts_match_trial_lists`). The public datum
  carries the first 20 `nct_ids` plus `nct_ids_total` (= the count); the complete list is one request away
  (citations endpoint) or inline with `citation_detail=full`. Interpolated survival points carry
  `derived_from` (the reported anchor datums) instead of pretending to have their own source.
* Excerpts are verbatim: the JSON leaf at `field_path`, or a substring with offsets (≤ 300 chars).
  `derived_count` cites a registry count by its search URL.
* Inline excerpts are capped (default 3, sorted by NCT id; 10 for single-trial datums) to keep payloads small;
  `citation_detail=full` inlines all, and the citations endpoint serves all for 24 h.
* Excerpts are re-verified against the raw records before sending (`core/provenance.py::verify_citation`).

## Networks

Edges are co-occurrence within one trial (sponsor tests drug, drugs given in the same arm, drug studied for
condition, sponsor–collaborator, investigator–site); edge weight and node size are **distinct trials**, and
each lists its NCT ids. To stay readable (`compute/network.py`): single-trial links are dropped when at least
10 links repeat across trials; at most 30 nodes by default (40 max) ranked by summed link weight; at most 120
edges, raising the minimum weight until they fit; entities the user named are pinned. "Which drugs are combined
with X" becomes an ego network around X. `meta.min_edge_weight` and `TRUNCATED` notes say what was cut.

## Storage and caching (`ctviz/store/`, one SQLite file in WAL mode)

| Store | Contents | Lifetime |
|---|---|---|
| HTTP cache | ClinicalTrials.gov responses (also as readable JSON in `.cache/ctgov/`) | 6 h for searches, 24 h for single studies |
| Plan cache | Validated LLM plans | 7 days, versioned |
| Response cache | Final responses, keyed by request + schema version + answer-logic version | 1 h; skipped for patient, degraded or failed-validation answers |
| Citation store | Every excerpt per datum, for the citations endpoint | 24 h |
| Study store | Latest raw record per trial (projected or full) | refreshed on fetch |
| Cohort snapshots | Member list of a fully read search (`store/cohort_snapshots.py`) | 24 h; used only while `totalCount` matches. `ctviz warm` builds them for broad conditions above the live limit |
| Knowledge graph | Trials, drugs, conditions, sponsors, sites, investigators and co-occurrences, with field-path evidence (`store/graph.py`) | Grows in the background from every answered cohort. Used for other-name resolution and spelling suggestions; exposed at `/v1/graph/neighbors` |
| FTS5 index | Trial text chunks with exact offsets (`store/fts.py`) | Written in the background; not read on the answer path |

## Rate limiting, budgets and failure handling

* Outbound (`ctgov/limiter.py`, `ctgov/client.py`): token bucket of 30 requests/min with a burst of 20, so any
  60 s window stays under ClinicalTrials.gov's ~50/min; at most 8 concurrent calls; 4 attempts with
  exponential backoff and jitter on 429/5xx/timeouts; on a 429 the bucket pauses for `Retry-After` (≥ 5 s) and
  halves its rate, then recovers by 5 % per success. Single-flight de-duplicates identical concurrent requests.
* Inbound (`api/middleware.py`): 30 requests/min per IP (burst 10) on `/v1/*` → `429` with `Retry-After`;
  more than 16 requests in flight → `503`. `X-Request-ID` on every response.
* Budgets (`config.py`, `core/budget.py`, `agents/llm.py`): ≤ 4 LLM calls and ≤ $0.05 per request, 8 s timeout
  per LLM call, one retry. The 60 s request deadline is a hard bound: limiter waits and HTTP timeouts are capped
  by the time left.
* The planner's plan is cached for 7 days, keyed by the normalized request, the model and a hash of the
  prompt, alias table and intent registry, plus `PLAN_CACHE_VERSION`. Patient requests are never cached.
* A failed dependency becomes a `DEGRADED` note and a smaller answer, not a 500.

## Latency (measured during development, not guaranteed)

| Question | Trials | Cache state | Time |
|---|---|---|---|
| Obesity trials by phase | 15,065 read | cold (20 API calls, 3 LLM calls, $0.007) | 18.4 s |
| Alzheimer's phase × status heatmap | 4,230 read | cold | 11.7 s |
| Lung cancer sponsor–drug network | 10,649 read | cold API cache | 6.5 s |
| Outcomes of one trial (NCT06973564) | 1 | cold | 2.5 s |
| Any repeated question | – | response cache hit | < 10 ms |

Cold broad-condition questions typically take 5–60 s, dominated by ClinicalTrials.gov paging under the rate
limit. `ctviz warm` moves that cost off the request path.

## CLI

```bash
.venv/bin/ctviz ask "Which drugs are most often combined with pembrolizumab?" --explain --out f.json
.venv/bin/ctviz ask "Am I eligible?" --nct-id NCT03548935 --field 'patient_profile={"age":45,"sex":"female"}'
.venv/bin/ctviz ask "..." --citations full     # citation_detail: ids_only | sample | full
.venv/bin/ctviz warm                           # build cohort snapshots + run data/warm_queries.yaml
.venv/bin/ctviz bench | examples | record      # latency table, regenerate examples, record fixtures
```

`--explain` prints the plan, every step with its timing, the ClinicalTrials.gov calls, the computed facts and
the notes. Every `ask` writes `runs/<request_id>/` (request, plan, steps, API calls, handler result, public JSON).

## Smaller limitations

* The LLM-chosen chart can differ between runs of the same question (the chart planner is not cached on its
  own; the response cache hides this for an hour).
* The FTS index is filled but not searched; `store/vectors.py` is not wired in (embeddings are off).
* `agents/retrieval_agent.py` supports an LLM rewrite for zero-result searches, but the orchestrator shows
  "0 trials" with suggestions instead.
* Sponsor names are normalized without fuzzy merging, so name variants of one company can appear separately.
* Citations-endpoint evidence for a response-cache hit is held in memory until its background write lands.
* SQLite storage and the inbound limiter are per process; several workers need a shared store and limiter.
* `FILTERS_RELAXED` and `CITATION_DROPPED` remain in the note enum but are unused.

# Part 2: Original plan (Cheiron take-home)

> **This is the original approved plan, kept for the record.** Main differences from this plan:
> `answer_summary` is a deterministic template (no synthesis LLM call); a chart planner agent (one validated
> LLM call after retrieval) was added; retrieval reads every matching trial up to 25,000 (not 3,000/5,000),
> with exact registry counts or a flagged sample beyond that, plus cohort snapshots from `ctviz warm`; the
> request deadline is 60 s, not 20 s; the outbound limiter is 30/min with a burst of 20; zero results and
> unknown entities return "0 trials" or `needs_clarification` instead of relaxing filters; the public JSON is
> the 2.2 projection in `ctviz/api/public.py` (no per-citation `url`, `nct_ids` on every datum, capped at 20);
> the FTS index is written but not used on the answer path; there is no Dockerfile.

## Context

Cheiron's take-home (full spec read from 6 screenshots in `the assignment PDF`; time expectation ~24 h) asks for an AI-enabled backend that turns a natural-language clinical-trial question (plus optional structured fields) into a **structured visualization specification** backed by ClinicalTrials.gov API v2, with documented request/response schemas, 3–5 example runs with real JSON output, a README (how to run; schema docs; key design decisions/tradeoffs; limitations + improvements; **integrity section: which AI tools were used, how correctness was validated, what was designed deliberately vs generated/adapted**), and bonus **deep citations** (every datum references `nct_id` + exact text excerpt / field value). Optional demo: small UI, deployed endpoint, short video.

**Grading:** System Design 35% (clear decisions, maintainable/extensible, sensible handling of real API data) · AI/Agent Design 20% (avoid hallucination-prone steps, validation/constraints, sensible planning + tools) · Code Quality 20% · Query & Viz Coverage 15% (breadth, no one-off hacks, meaningful network graphs) · I/O Design 10% (unambiguous, frontend-friendly schemas) · bonus citations. They reward "evidence of thoughtful construction, testing, and iteration."

**Spec's appendix query classes (will be tested):** time trends ("trials for [drug] per year since 2015", "trials started each year for [condition]"); distributions (phases for [condition]; most common intervention types); comparisons ("phases for Drug A vs Drug B", "sponsor categories across two conditions"); geographic ("countries with the most recruiting trials for [condition]"); networks ("sponsors ↔ drugs for [condition]", "drugs that co-occur in combination studies").

**User requirements on top:** patient + researcher personas; low latency (seconds); a real multi-agent runtime (router → retrieval → analysis → visualization → citation → synthesis → verification) but with **simple JSON and minimal tool calls**; model routing + rate limiting; compounding knowledge graph + RAG cache; small React demo; build with parallel Claude Code subagents the user can watch; keep the user in the loop with progress updates; don't stop at the deadline tiers, keep going. **Deadline: submit by 12:00 PM Sunday 2026-10-19.** Stack: Python 3.14 + FastAPI; SQLite (Postgres not needed; store is pluggable).

**Security:** secrets live only in a gitignored `.env`; `.env.example` holds placeholders. Nothing logs or commits keys.

## Verified facts (checked 2026-10-18)

- CT.gov API v2.0.5, **no API key**, base `https://clinicaltrials.gov/api/v2`. Endpoints: `/studies`, `/studies/{nctId}`, `/studies/metadata` (341 leaf fields), `/studies/search-areas`, `/studies/enums`, `/stats/field/values?fields=Phase,OverallStatus`, `/stats/size`, `/version`.
- `/studies` params confirmed: `query.cond|intr|term|spons|locn|outc|titles|id|patient`, `filter.overallStatus`, `filter.ids`, `filter.advanced` (Essie, e.g. `AREA[StartDate]RANGE[2020-01-01,MAX]`), `aggFilters=phase:2,status:rec`, `sort=StartDate:desc`, `fields=NCTId,BriefTitle,...` (piece names), `pageSize` ≤ 1000, `pageToken`/`nextPageToken` (**sequential only**), `countTotal=true`.
- Latency: 0.15–0.3 s small requests; 1000 studies × 12 fields = 1.4 MB in 1.4 s. 40-request burst → no 429; limit undocumented (~50/min cited) → stay polite.
- Full record (NCT03548935 = STEP 1, 239 KB) `resultsSection`: participantFlow (milestones + dropWithdraws reasons per group), baselineCharacteristics, outcomeMeasures (per-group measurements + analyses with pValue/CI/method/paramValue), adverseEvents (frequencyThreshold, eventGroups, per-term `stats[{groupId,numAffected,numAtRisk}]`, MedDRA organSystem). Design: allocation, interventionModel, masking + whoMasked, enrollment {count,type}. Eligibility free text with "Inclusion Criteria:"/"Exclusion Criteria:". Locations have geoPoint. `derivedSection` has MeSH terms.
- **No patient-level time-to-event data** → survival curves only from reported medians/landmarks/HR, labeled approximated.
- **OpenAI models allowed on the user's key:** gpt-4o-mini, gpt-4o-2024-08-06, gpt-4.1-nano, gpt-4.1-mini, gpt-4.1, gpt-5-mini, gpt-5, gpt-5.1, gpt-5.2, gpt-5.4-nano, gpt-5.4-mini, gpt-5.4. **No gpt-6 family, no embedding models listed** → embeddings are an optional flag that auto-disables on 403; retrieval is FTS5-lexical by default. Responses API + structured outputs (`client.responses.parse(text_format=Model)`); reasoning effort param made configurable per model (try `none`/`minimal`, fall back if rejected). PyPI: `openai` 3.35.1, `fastapi` 0.146.0, `pydantic` 2.14.1, `numpy` 2.5.4.
- Local: Python 3.14.3, Node 24.16, Docker (+ Postgres available, unused). No `uv`/`pnpm`. Project dir empty, **not a git repo**.

## Design principles (mapped to grading)

1. **One pipeline, intent-parameterized**: every query → `QueryPlan` → retrieval recipe → deterministic compute → viz builder → citations. 31 intents, 17 viz types, one encoding grammar, one citation model. Comparison scope (2–5 cohorts/trials → `series`) is first-class.
2. **Agents are specialists with hard budgets.** The hot path is 1 LLM call (router/planner); analysis, synthesis and verification agents add at most 2 more and run in parallel where possible. ≤ 4 LLM calls and ≤ 6 tool calls per request. Numbers are never LLM-generated.
3. **Honesty is structural**: `notes[]` with codes (`NOT_IN_REGISTRY`, `APPROXIMATED`, `SAMPLED`, `NO_RESULTS_POSTED`, `LLM_EVALUATED`, `ZERO_RESULTS`, `TRIAL_REQUIRED`, `ENTITY_AMBIGUOUS`, `FILTERS_RELAXED`), `meta.approximation` on derived curves, builder `feasibility()` gates, `kpi_cards`/`table` universal fallbacks.
4. **Every datum cites, in the spec's shape**: inline `citations: [{nct_id, excerpt, field_path, kind, char_start?, char_end?, url}]` per row/node/edge (cap 10, `citations_total` when truncated; full list via `GET /v1/responses/{request_id}/datums/{id}/citations`). Excerpts are verbatim JSON leaf values or verbatim substrings (≤300 chars) with offsets; LLM quotes accepted only after substring verification.
5. **Keep JSON simple**: flat rows, a handful of top-level keys, no nested encodings beyond `{field, type, title, unit?, format?, sort?}`.
6. **Compounding**: every fetch grows the alias table, study store, knowledge graph, and FTS index.

## Runtime agent roster (the product's agents; LLM-backed vs code is explicit)

| Agent | Kind | Role | Budget |
|---|---|---|---|
| **Router/Planner Agent** | LLM (gpt-5.4-nano, structured output) | Interprets the question → `QueryPlan{intent, persona, scope, entities, filters, viz prefs, needs}`; validated + repaired deterministically; rule-based fallback | 1 call, always |
| **Retrieval Agent** | Code by default; LLM-with-tools only on ambiguity | Executes the intent's retrieval recipe against CT.gov (search, paginate/partition, full records, facet counts, relax ladder). When the planner flags `ambiguous_entity` or all probes return zero, an LLM loop with tools `probe_count`, `search_studies`, `get_study` (max 3 tool calls) picks the right query | 0 LLM calls (default) / 1 |
| **Analysis Agent** | LLM (gpt-5.4-mini) + math tools, only for statistical intents | Chooses which outcome measures/groups/arms to analyze (a language task: "which outcome is overall survival?"), then calls deterministic tools `reconstruct_survival`, `power_curve`, `ae_rates`, `flow_table`, `forest_rows`; outputs a typed `AnalysisPlan`; every number comes from the tool with provenance | ≤ 1 call + ≤ 2 tool calls |
| **Visualization Agent** | Code (selector + 17 builders); no LLM | Picks builder by intent + feasibility + user preference; builds `encoding`, `data`, `meta`; emits downgrades and `alternatives` | 0 |
| **Citation Agent** | Code | `CitationCollector`: every datum's provenance → inline citations; dedupe, excerpt rules, caps, `derived_count` for facet buckets | 0 |
| **Synthesis Agent** | LLM (gpt-5.4-nano), overlapped with viz build | 2–4 sentence `answer_summary` citing `[nct_id]`s and numbers only from the built data rows; persona-adapted wording | 1 call, skippable (`narrative=false`) |
| **Verification Agent** | Code first; LLM only when flagged | Schema validation; every citation excerpt re-checked as substring of raw record; every number in `answer_summary` must appear in data rows (else sentence dropped); eligibility verdicts clamped; on failure → repair or downgrade, never silent | 0 (default) / 1 (gpt-5.4-mini re-extract) |
| **Extraction Agent** | LLM (gpt-5.4-nano), cached per criteria hash | Eligibility criteria → structured rules; medication/biomarker claims with verbatim quotes (verified) | ≤ 1 call per trial, ≤ 8 parallel |

Orchestrator = bounded plan → execute → verify → repair loop (max 3 iterations), per-step timeouts, hard 20 s deadline, degradation instead of failure. Target latency: warm cohort < 1.5 s, cold cohort 3–5 s, single trial < 2.5 s, statistical intents < 6 s.

**Model routing:** nano for planner/synthesis/extraction; mini for analysis agent and escalations (confidence < 0.6, schema failures, < 50% verified quotes, "complex" pre-classifier); gpt-5.4 only behind `quality=max`. Per-request cost cap $0.05. Models overridable via env.

## API contract

`POST /v1/visualize` · `GET /v1/schema` · `GET /v1/intents` · `GET /v1/trials/{nct_id}` · `GET /v1/graph/neighbors` · `GET /v1/responses/{id}/datums/{id}/citations` · `GET /v1/health` · `GET /v1/metrics` · `/docs`.

**Request:** `query` (required, 1–2000) · `nct_id` (`^NCT\d{8}$`) · `nct_ids` (≤5) · `drug_name` · `drug_names` (≤5) · `condition` · `conditions` (≤5) · `trial_phase[]` · `status[]` · `sponsor` · `country` · `start_year`/`end_year` (1990..current+5) · `persona` (patient|researcher|auto) · `patient_profile{age, sex, conditions[], medications[], notes}` · `preferred_visualization` · `group_by`/`series_by`/`time_granularity`/`top_n` · `max_studies` (≤5000) · `narrative` · `include_trace` · `refresh`. Structured fields override parsed entities.

**Response** (superset of the spec's example):
```
{ request_id, created_at, schema_version, source: "clinicaltrials.gov", ctgov_api_version,
  visualization: { type, title, subtitle, encoding, data, meta },
  meta: { filters_applied, interpretation{intent, intent_confidence, persona, scope, nct_ids, cohorts[], resolved_entities[], assumptions[], clarification, parser}, coverage{...}, notes[], alternatives[], cache{...}, trace{...} },
  answer_summary }
```
`visualization.data` = array of flat rows (or `{nodes[], edges[]}` for network/sankey); every row/node/edge has `id` and `citations[]`. Encoding = `{x, x2, y, y2, series, size, label, error_low, error_high, nodes{id,label,type,size}, edges{source,target,weight,type}, tooltip[]}` with `Channel{field, type, title, unit?, format?, sort?}`. `visualization.meta`: `units, sort, time_granularity, grouping, layout, thresholds[], truncation, approximation` + type-specific keys.

**VizType (17):** `kpi_cards, table, bar_chart, histogram, time_series, scatter_plot, heatmap, pie_chart, network_graph, sankey, timeline, survival_curve, threshold_line, power_curve, forest_plot, geo_map, eligibility_checklist`. Deterministic downgrades (0 rows → kpi; <3 buckets → bar; <2 edges → table; survival w/o points → forest → table; cardinality > top_n → "Other").

## Intent taxonomy (31) → primary viz

Cohort: `trend_over_time`→time_series · `phase_distribution`→bar · `status_distribution`→bar · `phase_status_heatmap`→heatmap · `landscape_gap_heatmap`→heatmap · `sponsor_landscape`→bar/pie · `sponsor_activity`→timeline · `geographic_distribution`→geo_map/bar · `entity_network`→network_graph (sponsor–drug, drug–drug combos from arms with ≥2 drugs, drug–condition, site–investigator, collaborators) · `enrollment_distribution`→histogram/scatter · `intervention_landscape`→bar · `development_progression`→timeline.
Single-trial: `single_trial_overview`→kpi · `study_rationale`→kpi · `eligibility_check`→eligibility_checklist · `concomitant_medications`→checklist · `trial_journey_timeline`→timeline · `visit_burden`→timeline/bar · `risks_adverse_events`→bar/threshold_line/table · `results_outcomes`→forest_plot/bar/kpi · `survival_summary`→survival_curve · `dropout_flow`→sankey · `demographics`→bar · `enrollment_vs_power`→power_curve · `sites_and_contacts`→geo_map/table · `safety_provisions`→table · `study_limitations`→table · `design_methodology`→kpi/bar · `endpoints_catalog`→table/bar · `biomarkers_pkpd`→table/bar · `not_in_registry_fallback`→kpi (costs/reimbursement/insurance/regular doctor: honest note + text scan + contacts).

## Retrieval, caching, limits

Cohort: page 1 with `countTotal`; ≤1000 done; ≤ max_studies (3000 / researcher 5000) → StartDate-partitioned parallel pages; beyond → facet counts for 1-D enum intents (≤10 parallel, exact, 10 sample members + `derived_count` citation) else most-recent sample + SAMPLED. Zero results → deterministic relax ladder → `FILTERS_RELAXED` → Retrieval Agent LLM loop → clarification. Field sets `P_LANDSCAPE`, `P_OUTCOMES`, `P_PEOPLE`, `P_ELIG`, `FULL`; validated against `/studies/metadata` at startup.

Storage: single SQLite (WAL, one background writer): `http_cache`, `studies`, `cohorts`, `plans`, `responses`, `extractions`, `chunks` (FTS5; optional float16 vectors), `kg_nodes`/`kg_edges`/`kg_cooccur`. Semantic plan cache only if embeddings available; else exact-normalized-query cache. Offline mode = fixture transport + stub LLM + rule planner; warmed `ctviz.db` ships in the zip.

Rate limits: inbound per key/IP 30/min burst 10 (429 + Retry-After), in-flight cap 16; outbound bucket 45/min burst 10 + semaphore 8 + backoff/jitter + adaptive halving; ≤ 8 outbound calls/request.

## Repository layout

```
.gitignore .env.example (placeholders only) .env (ignored) CLAUDE.md explanations.md (ignored) README.md pyproject.toml Dockerfile docker-compose.yml
.claude/agents/{code-reviewer,debugger,test-writer,viz-spec-validator,api-tester,builder-cohort-viz,builder-trial-text,builder-results-stats,builder-orchestration,builder-planner,builder-graph,builder-frontend,builder-docs}.md
ctviz/ app.py cli.py config.py
  api/{routes.py,middleware.py,schemas/}  core/{orchestrator.py,intents.py,entities.py,budget.py,provenance.py}
  agents/{llm.py,router.py,planner.py,rules_planner.py,retrieval_agent.py,analysis_agent.py,synthesis.py,verification.py,extractors.py,prompts/}
  ctgov/{client.py,limiter.py,requests.py,fields.py,retrieval.py,normalize.py,paths.py}
  store/{db.py,http_cache.py,study_store.py,cohort_cache.py,plan_cache.py,response_cache.py,graph.py,fts.py}
  compute/{eligibility.py,timeframes.py,adverse_events.py,flow.py,survival.py,power.py,network.py,text_scan.py,endpoints.py}
  viz/{base.py,registry.py,selector.py,citations.py,builders/<17>}  obs/{trace.py,logging.py,pricing.py,metrics.py}  offline/
data/{aliases.json,intents.yaml,warm_queries.yaml}  tests/{unit,contract,golden,bench,fixtures/ctgov}  examples/  docs/  frontend/  scripts/try_queries.sh
```

**Config (`ctviz/config.py`, pydantic-settings + python-dotenv):** loads `.env`; fails at startup with a clear message if `OPENAI_API_KEY` is missing and `CTVIZ_OFFLINE` is not set; never logs the key (masked `sk-…last4` in `/v1/health` only as "configured: true").

**`.claude/agents/` subagents (all with name/description/tools frontmatter):**
- `code-reviewer` (Read, Grep, Glob): reviews diffs for bugs, unclear naming, missing docstrings, unsafe handling of API data; reports only.
- `debugger` (Read, Grep, Glob, Bash, Edit): reproduces a failing test/error, finds root cause, proposes minimal fix.
- `test-writer` (Read, Grep, Glob, Write, Edit, Bash): pytest tests incl. respx mocks of CT.gov and the citation-excerpt-in-raw-record invariant.
- `viz-spec-validator` (Read, Grep, Glob, Bash): validates every `examples/*.json` against the exported schema + renderer rules.
- `api-tester` (Bash, Read): hits the running server with curl for a query list, pretty-prints type/title/first rows/citations/latency, flags anomalies — the user's way to test without the frontend (`scripts/try_queries.sh` does the same manually).
- `builder-*` (full tools): one per Phase 1/3 work package, each with its contract, files it owns, and "run code-reviewer + test-writer, then fix" exit criteria.
Workflow rule: after each major module, run `code-reviewer` and `test-writer` in parallel, fix findings; run `debugger` whenever a test fails.

**CLAUDE.md (repo root, committed):** project purpose + rubric; non-negotiable rules (no LLM-generated numbers; every datum cites; schemas are source of truth; simple JSON; ≤ 4 LLM calls/≤ 6 tool calls per request; never log secrets; Python 3.14/FastAPI/SQLite; allowed OpenAI models); module map; runtime agent roster; subagent roster + workflow; commands (`ctviz serve|warm|bench|examples|record`, `pytest`, `scripts/try_queries.sh`); deliverables checklist; current phase/status pointer.

**explanations.md (root, gitignored):** plain-language walkthrough for the user: phases, every endpoint (method, path, purpose, example), the agents and when each fires, how caching/rate limiting work, how to test without the frontend, what to open in other terminals.

## Build plan (parallel subagents; keep user updated at each phase boundary)

**Step 0 (me, first):** `git init`; `.gitignore` (`.env`, `.env.*`, `!.env.example`, `__pycache__/`, `.venv/`, `node_modules/`, `.DS_Store`, `*.log`, `.pytest_cache/`, `.ruff_cache/`, `*.db`, `.cache/`, `explanations.md`, `frontend/dist/`); `.env.example` with placeholders (`OPENAI_API_KEY=sk-...`, `CTVIZ_*` names); `.env` with the real key (user supplies it by editing the file or `! pbpaste > .env`-style, not via chat); `ctviz/config.py`; `git status` + `git check-ignore -v .env` to prove ignore; `CLAUDE.md`; `.claude/agents/*.md`; `explanations.md` v1; first commit. Save memory (deadline, rubric, allowed models, user preferences).
**Phase 0 (me, contracts):** pyproject; schemas + `/v1/schema`; intents registry; viz base/registry/selector; provenance; ctgov client/limiter/requests/fields/normalize/paths + http cache; store DDL; offline transport + `ctviz record` fixtures (NCT03548935; a recruiting no-results trial; pembrolizumab/GLP-1/obesity/semaglutide/tirzepatide cohorts; facet counts). Verify Python 3.14 wheels first. Run code-reviewer + test-writer; commit. **Update user.**
**Phase 1 (6 builder subagents in worktrees, parallel):** planner+entities+retrieval agent · cohort builders+intents · trial-text intents+extraction · results/stats intents+analysis agent · orchestration+caches+synthesis+verification+middleware+trace · knowledge graph+FTS+warm. Each ends with code-reviewer + test-writer, fixes, commit. **Update user.**
**Phase 2 (me):** merge; contract/golden tests; 14 end-to-end queries (6 patient, 8 researcher incl. all appendix classes); api-tester run; bench; ruff. **Update user with latency table.**
**Phase 3 (3 subagents, parallel):** frontend (Vite+React+ECharts renderer for all 17 types, persona toggle, dropdowns, citation panel, alternatives switcher; built assets served at `/`) · README+docs+examples (10 real outputs incl. the spec's pembrolizumab example)+Docker+`.env.example` · hardening tests. viz-spec-validator on examples. **Update user.**
**Phase 4 (me):** rubric review of README; warm db; fresh-clone install test; zip; push to GitHub via `gh` once the user has run `gh auth login`; optional deploy + video. Then continue with stretch items (embeddings flag, deploy, video) rather than stopping.

Scope tiers if integration slips: **Must** = Step 0, Phase 0, Phase 1 packages 1–5, Phase 2, README + examples + static renderer. **Should** = graph package, React demo, bench, Docker. **Stretch** = embeddings/semantic cache, deploy, video.

## Verification

- `pytest`: unit (builders on fixtures; citation invariant: every `field_path` resolves and excerpt is a substring of the raw value), contract (schema validation of every example), golden (offline regeneration), limiter (429 → backoff; concurrency ≤ 8), verification agent (drops unverifiable quotes; drops summary sentences with numbers absent from data).
- `ctviz bench --mode cold|warm` vs targets.
- `api-tester` subagent / `scripts/try_queries.sh`: run the 14 queries against `ctviz serve`, print type/title/rows/citations/latency; with and without `OPENAI_API_KEY` (offline).
- Manual: `/docs`; frontend renders each viz type; click citations and confirm excerpts on ClinicalTrials.gov; notes appear for costs / regular-doctor / no-results.
- Final checklist vs spec §6 and rubric §7–8; `git check-ignore -v .env` before zipping; grep the repo for `sk-` before commit and before zip.
