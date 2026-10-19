# Clinical Trial Backend (`ctviz`)

A FastAPI service that turns a clinical-trial question, in plain language plus optional structured fields,
into a **visualization spec** (`type`, `title`, `encoding`, `data`, `meta`) built from the
[ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api). Every number in the chart is computed
in Python from registry records, and every datum lists the trials behind it and quotes the exact registry
fields that support it.

It serves **patients** (eligibility, visits, risks, results of one trial) and **researchers** (landscapes,
trends, comparisons, networks, design, endpoints, dropout, power, survival): 31 question types
(`GET /v1/intents`) mapped onto 17 chart types.

Contents: [Folder structure](#folder-structure) · [Run it](#how-to-run) · [Request](#request-schema) ·
[Response](#response-schema) · [Design](#design-decisions-and-tradeoffs) ·
[Limitations](#limitations-and-what-id-improve) · [Examples](#example-runs) · [AI usage](#ai-usage-and-integrity)

Demo: the viewer at <http://localhost:8000/> once the server runs.

---

## Folder structure

```
ctviz/                 the backend package
  app.py, cli.py       FastAPI app factory; `ctviz serve | ask | warm | examples | record | bench`
  config.py            settings from CTVIZ_* env vars and .env
  api/                 routes, middleware (rate limit, request id), schemas/ (the request/response contract),
                       public.py (internal model → public JSON)
  core/                orchestrator.py (the pipeline), provenance.py (every row built with its citations),
                       answerability.py (refuse/clarify), budget.py (deadline, LLM budget), entities.py
  agents/              LLM planner + rule-based fallback, chart planner, analysis agent (results), eligibility
                       extractor, summary templates, citation/summary verification, final validation; prompts/
  ctgov/               ClinicalTrials.gov client, rate limiter, query builder, cohort retrieval, field paths
  handlers/            one handler per question type: cohort.py (landscapes), trial_text.py (one trial's
                       text), results.py (posted results: survival, forest plots, adverse events, dropout)
  compute/             pure functions: survival reconstruction, power, adverse events, participant flow, networks
  viz/                 one builder per chart type + the selector that downgrades a chart that can't be drawn
  store/               SQLite caches, cohort snapshots, knowledge graph, full-text index
  obs/, offline/       logging and metrics; stub LLM + recorded-fixture transport for offline mode
data/                  drug/condition aliases, question types (intents.yaml), warm-up queries
frontend/              React + ECharts viewer; the built copy in frontend/dist is served at /
examples/              real request/response JSON: requests.json + one file per run
tests/                 unit, contract and golden tests + recorded ClinicalTrials.gov fixtures (all offline)
docs/                  response schema reference, conventions, design detail
scripts/               curl smoke test, question runner, secret scan
CLAUDE.md, .claude/    the rules and subagents used while building it (see AI usage)
```

---

## How to run

Requires Python ≥ 3.12 (developed on 3.14). Node 24 only if you rebuild the viewer.

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # [dev] adds pytest/ruff
cp .env.example .env          # then set OPENAI_API_KEY (the only secret)
.venv/bin/ctviz serve --port 8000
```

* Viewer: <http://localhost:8000/> · API docs: <http://localhost:8000/docs>
* Rebuild the viewer: `cd frontend && npm ci && npm run build`
* Tests: `.venv/bin/pytest -q` (offline, mocked LLM) · `cd frontend && npx vitest run`
* One question from the terminal: `.venv/bin/ctviz ask "Which drugs are combined with pembrolizumab?" --explain`

`.env` is gitignored and read with python-dotenv. The key is held as a `SecretStr`, never logged, and
`/v1/health` reports only `llm_configured: true/false`.

**Offline mode** (`CTVIZ_OFFLINE=1 .venv/bin/ctviz serve`): no key, no network. A stub LLM (the rule-based
planner answers) and recorded ClinicalTrials.gov fixtures replace the live services. The 14 example requests in
`examples/requests.json` replay end to end; other questions return a "ClinicalTrials.gov unavailable" card.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/visualize` | Main endpoint: question → visualization spec |
| GET | `/v1/schema` | JSON Schemas of the request and the public response |
| GET | `/v1/intents` | Supported question types, with example questions and chart types |
| GET | `/v1/trials/{nct_id}` | The raw ClinicalTrials.gov record that citation paths point into |
| GET | `/v1/responses/{request_id}/datums/{datum_id}/citations` | Every trial id and excerpt for one datum (kept 24 h), each re-verified |
| GET | `/v1/graph/neighbors?name=&node_type=` | Knowledge-graph neighbours of an entity, with evidence |
| GET | `/v1/health`, `/v1/metrics` | Liveness/config flags; counters and latency percentiles |

---

## Request schema

`POST /v1/visualize` (`ctviz/api/schemas/request.py`). Only `query` is required. Unknown fields are rejected
and invalid values return `422` with a message per field. Structured fields override what the planner reads
from the question.

| Field | Type | Validation / meaning |
|---|---|---|
| `query` | string | **required**, 1–2000 characters |
| `nct_id` / `nct_ids` | string / string[] | `^NCT\d{8}$` (case-insensitive); ≤ 5 trials to compare |
| `drug_name` / `drug_names` | string / string[] | ≤ 5 drugs to compare |
| `condition` / `conditions` | string / string[] | ≤ 5 conditions to compare |
| `trial_phase` | string[] | `EARLY_PHASE1`, `PHASE1`…`PHASE4`, `NA`; `"2"`, `"II"`, `"Phase 2"` are normalized |
| `status` | string[] | `RECRUITING`, `COMPLETED`, … (status **today**) |
| `sponsor`, `country` | string | |
| `start_year` / `end_year` | int | ≥ 1990, ≤ current year + 5, `start_year ≤ end_year` |
| `persona` | `patient` \| `researcher` \| `auto` | default `auto` |
| `patient_profile` | object | `{age, sex, conditions, medications, notes}`; never sent to the planner LLM, never cached |
| `preferred_visualization` | one of the 17 types | used when it fits the question, else a `VIZ_PREFERENCE_IGNORED` note |
| `group_by` / `series_by` | enum | `phase`, `status`, `sponsor`, `sponsor_class`, `condition`, `intervention`, `country`, `year`, … |
| `time_granularity` | `year` \| `quarter` \| `month` | |
| `top_n` | int | 1–100 |
| `max_studies` | int | 1–25,000 (default 25,000): records read per cohort |
| `citation_detail` | `ids_only` \| `sample` \| `full` | excerpts per datum: none / first 3 / all (default `sample`) |
| `narrative`, `include_trace`, `refresh` | bool | summary on/off; add `meta.debug`; bypass caches |

```bash
curl -s localhost:8000/v1/visualize -H 'content-type: application/json' -d '{
  "query": "How has the number of trials for this drug changed over time?",
  "drug_name": "Pembrolizumab", "start_year": 2015 }'
```

---

## Response schema

Full renderer contract: **[docs/response_schema.md](docs/response_schema.md)** (every chart type, meta key,
note code), one-page summary in [docs/response_schema_conventions.md](docs/response_schema_conventions.md),
JSON Schema at `GET /v1/schema`. `schema_version`: **2.2**. Real responses are shown under
[Example runs](#example-runs).

```jsonc
{
  "status": "ok",                  // or "needs_clarification" | "unanswerable" (then "clarification", no chart)
  "visualization": {
    "type": "time_series",          // one of 17
    "title": "Trials started per year: Pembrolizumab",
    "encoding": {"x": {"field": "period", ...}, "y": {"field": "trial_count", ...}},
    "data": [ {"id": "d1", "period": "2015-01-01", "trial_count": 120, /* provenance keys last */ ...} ]
  },
  "answer_summary": "2022 has the most trials (299), ...",
  "meta": {"filters": ..., "coverage": ..., "notes": [...], "validation": ..., "citation_policy": ..., "schema_version": "2.2"}
}
```

How to read it:

* `visualization` holds only what is drawn; anything else a renderer needs (units, sort order, map level,
  network legend, thresholds) is in `meta`. Rows are flat; network and sankey use `data: {nodes, edges}`.
* `encoding.<channel>.field` names the row key, so a renderer never guesses field names. Pie slices use
  `encoding.label`.
* Each datum ends with its provenance: `nct_ids` (first 20) + `nct_ids_total` (the exact count), `field_path`,
  `citations` (verbatim excerpts), `citations_total`, `excerpts_truncated`, and on interpolated survival points
  `derived_from` (the reported datums it was computed from). The full lists are at the citations endpoint.
* `meta.coverage` says how the numbers were obtained (all records, exact registry counts, sample);
  `meta.notes` says what is approximate, sampled, missing or truncated; `meta.validation` lists the checks run.

---

## Design decisions and tradeoffs

More detail (pipeline table, retrieval, caching, rate limits, latency): [docs/design.md](docs/design.md).

**One pipeline.** validate → response cache → plan → answerability → entity check → retrieve → chart planner →
handler → summary template → verify → validate → public JSON → background cache/graph writes. Every step is
logged with inputs, duration and result.

**LLMs only where language is the problem.** Three bounded LLM roles (≤ 4 calls per request, 8 s timeout,
one retry, $0.05 cap): the **planner** (gpt-5.4-nano) turns the question into a `QueryPlan`; the **chart
planner** (gpt-5.4-mini) picks a view from a menu the code has already checked is buildable; the **analysis
agent** (gpt-5.4-mini) picks outcome measures and arms for results questions; eligibility judging uses nano.
Every output is a Pydantic model whose values are checked against allow-lists, and entities must appear in the
alias table or the question. They never produce a number, label or sentence that reaches the response. A
rule-based planner is the fallback, so an LLM outage degrades answers instead of failing them (tradeoff: two
planners to maintain).

**No LLM text in answers.** `answer_summary` is a template over computed facts, and every number in it is
checked against the data; a sentence that fails is dropped (`SUMMARY_TRIMMED`).

**Read everything.** ClinicalTrials.gov has no aggregation API, so the service reads every matching trial up
to 25,000 (start-date partitions paged concurrently). Above that it uses exact registry counts per bucket, a
cohort snapshot (`ctviz warm`), or a start-year-stratified sample, and says which. Tradeoff: correctness over
latency; a cold broad question takes 5–60 s, a repeated one milliseconds.

**Deep citations.** Rows can only be built through `Dataset.emit(row, evidence)`. Each excerpt is the
verbatim JSON leaf or a verbatim substring with offsets, re-verified against the raw record before sending.
Counts are checked against their trial lists.

**Honesty over coverage.** Three outcomes: `ok`, `needs_clarification` (unknown entity, gibberish, or a
question that needs a trial), `unanswerable` (verdicts like "which drug is safest"; only a deterministic
check can refuse). The service never silently widens a search. Partial answers carry notes: `SAMPLED`,
`APPROXIMATED`, `NO_RESULTS_POSTED`, `TRUNCATED`, …

**Survival.** The registry stores reported medians and landmark rates, not patient-level data. Curves are
reconstructed from those (piecewise exponential), labelled approximate, and interpolated points link to their
reported anchors. Across trials, only trials with posted results count: a forest plot of reported hazard
ratios by default, curves for up to 8 trials when asked for "Kaplan Meier". A named trial with no results
falls back to the cohort with a `NO_RESULTS_POSTED` note.

**Meaningful networks.** Edges are same-trial co-occurrence (drugs in the same arm, sponsor tests drug, …),
weighted by distinct trials, placebo excluded, drug name variants merged, at least 2 trials per edge when the
graph is large, at most 30 nodes / 120 edges with named entities pinned. The viewer uses a fixed, precomputed
layout.

**Status means today.** The registry keeps no status history, so status filters describe trials as they are
now and titles say "(status today: recruiting)".

**Polite to the API.** All calls go through one client: 30 requests/min with a burst of 20 (under the ~50/min
limit), 8 concurrent, backoff with jitter, `Retry-After` honoured, single-flight de-duplication, disk cache.
Inbound: 30 requests/min per IP. A failed dependency becomes a `DEGRADED` note, not a 500.

---

## Limitations and what I'd improve

* **Above 25,000 matching trials without a snapshot,** networks and heatmaps use a flagged sample, so only
  one-dimensional counts stay exact.
* **Survival curves are reconstructions**, and most registry trials post no results, so many cohorts give few
  curves. Power curves approximate from sample size, not the trial's registered calculation.
* **"Melanoma and Type 2 Diabetes" for example, in one Condition field is one condition.** Use `conditions: [...]`, commas,
  or name both in the question.
* **A bare "best" is refused as a verdict,** even when the user meant "largest" or "most recent".
* **Cold broad questions are slow** (one API call per 1,000 records under the rate limit).
* **Status is today's status,** not historical.
* **Offline mode** replays only the 14 recorded example requests.
* **Single process:** SQLite and the inbound limiter are per process.

Smaller items are listed in [docs/design.md](docs/design.md#smaller-limitations).

**Next Steps:** use the full-text index (or embeddings) for text questions across a cohort; refresh snapshots
incrementally so large cohorts stay exact; cache the chart planner's choice; Postgres + Redis for several
workers; merge sponsor-name variants. Develop this further to include an agent for data analysis for even more advanced questions.

---

## Example runs

One real run per query class in the spec's appendix. Each file is the unedited output of `ctviz examples`
(requests in [`examples/requests.json`](examples/requests.json)). The excerpts below were cut from those files
by a script: 2 rows (or 2 nodes and 2 edges), `nct_ids` cut to 3, 2 citations per datum, `meta` trimmed.
All 14 files validate against the response model, and all 959 inline citations re-verify against the raw
ClinicalTrials.gov records.

### Time trends: How has the number of trials for Pembrolizumab changed over time since 2015?

Request `{"query": "How has the number of trials for Pembrolizumab changed over time since 2015?", "drug_name": "Pembrolizumab", "start_year": 2015, "persona": "researcher"}` → [`examples/pembrolizumab_trend_since_2015.json`](examples/pembrolizumab_trend_since_2015.json)

```jsonc
{
  "status": "ok",
  "visualization": {
    "type": "time_series", "title": "Trials started per year: Pembrolizumab",
    "encoding": {"x": {"field": "period", "type": "temporal", "title": "Start year", "time_format": "%Y", "scale": {"type": "time"}}, "y": {"field": "trial_count", "type": "quantitative", "title": "Trials", "unit": "trials", "format": ",d"}, "tooltip": ["label", "estimated_count"]},
    "data": [
      {"id": "d1", "period": "2015-01-01", "year": 2015, "label": "2015", "trial_count": 120, "estimated_count": 0, "nct_ids": ["NCT02260440", "NCT02268825", "NCT02296684", "…"], "nct_ids_total": 120, "field_path": "protocolSection.statusModule.startDateStruct.date", "citations": [{"nct_id": "NCT02260440", "excerpt": "2015-01"}, {"nct_id": "NCT02268825", "excerpt": "2015-01-23"}], "citations_total": 120, "excerpts_truncated": true},
      {"id": "d2", "period": "2016-01-01", "year": 2016, "label": "2016", "trial_count": 195, "estimated_count": 0, "nct_ids": ["NCT01676753", "NCT02289209", "NCT02432963", "…"], "nct_ids_total": 195, "field_path": "protocolSection.statusModule.startDateStruct.date", "citations": [{"nct_id": "NCT01676753", "excerpt": "2016-12-28"}, {"nct_id": "NCT02289209", "excerpt": "2016-03-09"}], "citations_total": 195, "excerpts_truncated": true},
      "… 11 more rows"]
  },
  "answer_summary": "2022 has the most trials (299), then 2021 (265), then 2020 (263). Based on 2,916 trials from ClinicalTrials.gov.",
  "meta": {"filters": {"drug_names": ["pembrolizumab"], "start_year": 2015}, "coverage": {"studies_matched_total": 2916, "studies_analyzed": 2916, "counting_method": "all_records", "sampled": false, "queries": ["https://clinicaltrials.gov/search?intr=pembrolizumab&advanced=AREA%5BStartDate%5DRANGE%5B2015-01-01%2CMAX%5D"]}, "validation": {"passed": true, "checks": ["retrieval_matches_api_total", "counts_match_trial_lists", "each_trial_in_one_bucket", "citations_from_retrieved_records", "summary_numbers_in_data", "value_axes_labeled"]}, "schema_version": "2.2", "…": "…"}
}
```

### Distributions: How are melanoma trials distributed across phases?

Request `{"query": "How are melanoma trials distributed across phases?", "condition": "melanoma", "persona": "researcher"}` → [`examples/melanoma_phase_distribution.json`](examples/melanoma_phase_distribution.json)

```jsonc
{
  "status": "ok",
  "visualization": {
    "type": "bar_chart", "title": "Trials by phase: Melanoma",
    "encoding": {"x": {"field": "phase", "type": "ordinal", "title": "Phase", "sort": {"values": ["Early Phase 1", "Phase 1", "Phase 1/2", "Phase 2", "Phase 2/3", "Phase 3", "Phase 4", "Not applicable"]}}, "y": {"field": "trial_count", "type": "quantitative", "title": "Trials", "unit": "trials", "format": ",d"}},
    "data": [
      {"id": "d1", "phase": "Early Phase 1", "trial_count": 66, "nct_ids": ["NCT00276835", "NCT00436410", "NCT00470379", "…"], "nct_ids_total": 66, "field_path": "protocolSection.designModule.phases", "citations": [{"nct_id": "NCT00276835", "excerpt": "[\"EARLY_PHASE1\"]"}, {"nct_id": "NCT00436410", "excerpt": "[\"EARLY_PHASE1\"]"}], "citations_total": 66, "excerpts_truncated": true},
      {"id": "d2", "phase": "Phase 1", "trial_count": 871, "nct_ids": ["NCT00001439", "NCT00001577", "NCT00001683", "…"], "nct_ids_total": 871, "field_path": "protocolSection.designModule.phases", "citations": [{"nct_id": "NCT00001439", "excerpt": "[\"PHASE1\"]"}, {"nct_id": "NCT00001577", "excerpt": "[\"PHASE1\"]"}], "citations_total": 871, "excerpts_truncated": true},
      "… 6 more rows"]
  },
  "answer_summary": "For Melanoma, Not applicable has the most trials (1,059), then Phase 2 (1,042), then Phase 1 (871). Based on 3,776 trials from ClinicalTrials.gov.",
  "meta": {"filters": {"conditions": ["melanoma"]}, "coverage": {"studies_matched_total": 3776, "studies_analyzed": 3776, "counting_method": "all_records", "sampled": false, "queries": ["https://clinicaltrials.gov/search?cond=melanoma"]}, "validation": {"passed": true, "checks": ["retrieval_matches_api_total", "counts_match_trial_lists", "each_trial_in_one_bucket", "citations_from_retrieved_records", "summary_numbers_in_data", "value_axes_labeled"]}, "schema_version": "2.2", "…": "…"}
}
```

### Comparisons: Compare sponsor categories (industry, academic, government) for lung cancer vs breast cancer trials

Request `{"query": "Compare sponsor categories (industry, academic, government) for lung cancer vs breast cancer trials", "conditions": ["lung cancer", "breast cancer"], "persona": "researcher"}` → [`examples/sponsor_categories_lung_vs_breast_cancer.json`](examples/sponsor_categories_lung_vs_breast_cancer.json)

```jsonc
{
  "status": "ok",
  "visualization": {
    "type": "bar_chart", "title": "Trials by lead sponsor class: Lung cancer vs Breast cancer",
    "encoding": {"x": {"field": "sponsor_class", "type": "nominal", "title": "Sponsor class", "sort": {"values": ["Other (academic, hospital, non-profit)", "Industry", "NIH", "Other government", "Network", "Other U.S. federal", "Individual", "Unknown"]}}, "y": {"field": "trial_count", "type": "quantitative", "title": "Trials", "unit": "trials", "format": ",d"}, "series": {"field": "cohort", "type": "nominal", "title": "Cohort"}},
    "data": [
      {"id": "d1", "sponsor_class": "Other (academic, hospital, non-profit)", "trial_count": 8187, "cohort": "Lung cancer", "nct_ids": ["NCT00002465", "NCT00002470", "NCT00002475", "…"], "nct_ids_total": 8187, "field_path": "protocolSection.sponsorCollaboratorsModule.leadSponsor.class", "citations": [{"nct_id": "NCT00002465", "excerpt": "OTHER"}, {"nct_id": "NCT00002470", "excerpt": "OTHER"}], "citations_total": 8187, "excerpts_truncated": true},
      {"id": "d2", "sponsor_class": "Industry", "trial_count": 4321, "cohort": "Lung cancer", "nct_ids": ["NCT00002911", "NCT00003034", "NCT00003562", "…"], "nct_ids_total": 4321, "field_path": "protocolSection.sponsorCollaboratorsModule.leadSponsor.class", "citations": [{"nct_id": "NCT00002911", "excerpt": "INDUSTRY"}, {"nct_id": "NCT00003034", "excerpt": "INDUSTRY"}], "citations_total": 4321, "excerpts_truncated": true},
      "… 14 more rows"]
  },
  "answer_summary": "For Lung cancer, Other (academic, hospital, non-profit) has the most trials (8,187), then Industry (4,321), then NIH (355). For Breast cancer, Other (academic, hospital, non-profit) has the most trials (12,260), then Industry (3,390), then NIH (376). Based on 30,128 trials from ClinicalTrials.gov.",
  "meta": {"filters": {"compare": [{"label": "Lung cancer", "conditions": ["lung cancer"]}, {"label": "Breast cancer", "conditions": ["breast cancer"]}]}, "coverage": {"studies_matched_total": 30128, "studies_analyzed": 30128, "counting_method": "all_records", "sampled": false, "queries": ["https://clinicaltrials.gov/search?cond=%22lung+cancer%22", "https://clinicaltrials.gov/search?cond=%22breast+cancer%22"]}, "validation": {"passed": true, "checks": ["retrieval_matches_api_total", "counts_match_trial_lists", "each_trial_in_one_bucket", "citations_from_retrieved_records", "summary_numbers_in_data", "value_axes_labeled"]}, "schema_version": "2.2", "…": "…"}
}
```

### Geographic patterns: Which countries have the most recruiting trials for Alzheimer's disease?

Request `{"query": "Which countries have the most recruiting trials for Alzheimer's disease?", "condition": "Alzheimer's disease", "status": ["RECRUITING"], "persona": "researcher"}` → [`examples/alzheimers_recruiting_countries.json`](examples/alzheimers_recruiting_countries.json)

```jsonc
{
  "status": "ok",
  "visualization": {
    "type": "geo_map", "title": "Trials by country: Alzheimer's disease (status today: recruiting)",
    "encoding": {"x": {"field": "country", "type": "nominal", "title": "Country"}, "y": {"field": "trial_count", "type": "quantitative", "title": "Trials", "unit": "trials", "format": ",d"}, "label": {"field": "country", "type": "nominal", "title": "Country"}, "tooltip": ["site_count"]},
    "data": [
      {"id": "d1", "country": "United States", "trial_count": 278, "site_count": 1646, "nct_ids": ["NCT00064870", "NCT00869817", "NCT02022943", "…"], "nct_ids_total": 278, "field_path": "protocolSection.contactsLocationsModule.locations[0].country", "citations": [{"nct_id": "NCT00064870", "excerpt": "United States"}, {"nct_id": "NCT00869817", "excerpt": "United States"}], "citations_total": 278, "excerpts_truncated": true},
      {"id": "d2", "country": "China", "trial_count": 98, "site_count": 509, "nct_ids": ["NCT03653156", "NCT03657732", "NCT03810794", "…"], "nct_ids_total": 98, "field_path": "protocolSection.contactsLocationsModule.locations[0].country", "citations": [{"nct_id": "NCT03653156", "excerpt": "China"}, {"nct_id": "NCT03657732", "excerpt": "China"}], "citations_total": 98, "excerpts_truncated": true},
      "… 8 more rows"]
  },
  "answer_summary": "United States has the most trials (278), then China (98), then France (69). Based on 614 trials from ClinicalTrials.gov (status today: recruiting).",
  "meta": {"filters": {"conditions": ["Alzheimer's disease"], "statuses": ["RECRUITING"]}, "coverage": {"studies_matched_total": 614, "studies_analyzed": 614, "counting_method": "all_records", "sampled": false, "queries": ["https://clinicaltrials.gov/search?cond=%22Alzheimer%27s+disease%22&aggFilters=status%3Arec"]}, "validation": {"passed": true, "checks": ["retrieval_matches_api_total", "counts_match_trial_lists", "each_trial_in_one_bucket", "citations_from_retrieved_records", "summary_numbers_in_data", "value_axes_labeled"]}, "schema_version": "2.2", "…": "…"}
}
```

### Relationships / networks: Show the network of sponsors and the drugs they test in obesity trials

Request `{"query": "Show the network of sponsors and the drugs they test in obesity trials", "condition": "obesity", "persona": "researcher"}` → [`examples/obesity_sponsor_drug_network.json`](examples/obesity_sponsor_drug_network.json)

```jsonc
{
  "status": "ok",
  "visualization": {
    "type": "network_graph", "title": "Sponsors and the drugs they test: Obesity",
    "encoding": {"nodes": {"id": "id", "label": "label", "type": "type", "size": "size"}, "edges": {"source": "source", "target": "target", "weight": "weight", "type": "type"}, "tooltip": ["degree"]},
    "data": {"nodes": [
      {"id": "sponsor:eli-lilly-and-company", "label": "Eli Lilly and Company", "type": "sponsor", "size": 118, "degree": 7, "nct_ids": ["NCT00044187", "NCT00090012", "NCT00630825", "…"], "nct_ids_total": 118, "field_path": "protocolSection.sponsorCollaboratorsModule.leadSponsor.name", "citations": [{"nct_id": "NCT00044187", "excerpt": "Eli Lilly and Company"}, {"nct_id": "NCT00090012", "excerpt": "Eli Lilly and Company"}], "citations_total": 118, "excerpts_truncated": true},
      {"id": "drug:semaglutide", "label": "Semaglutide", "type": "drug", "size": 186, "degree": 6, "nct_ids": ["NCT02079870", "NCT02453711", "NCT03292315", "…"], "nct_ids_total": 186, "field_path": "protocolSection.armsInterventionsModule.interventions[0].name", "citations": [{"nct_id": "NCT02079870", "excerpt": "semaglutide"}, {"nct_id": "NCT02453711", "excerpt": "semaglutide"}], "citations_total": 186, "excerpts_truncated": true},
      "… 25 more nodes"], "edges": [
      {"id": "e1", "source": "sponsor:novo-nordisk-a-s", "target": "drug:semaglutide", "weight": 66, "type": "sponsor_tests_drug", "nct_ids": ["NCT02079870", "NCT02453711", "NCT03548935", "…"], "nct_ids_total": 66, "field_path": "protocolSection.sponsorCollaboratorsModule.leadSponsor.name", "citations": [{"nct_id": "NCT02079870", "excerpt": "Novo Nordisk A/S"}, {"nct_id": "NCT02079870", "excerpt": "semaglutide", "field_path": "protocolSection.armsInterventionsModule.interventions[0].name"}], "citations_total": 132, "excerpts_truncated": true},
      {"id": "e2", "source": "sponsor:eli-lilly-and-company", "target": "drug:tirzepatide", "weight": 41, "type": "sponsor_tests_drug", "nct_ids": ["NCT04081337", "NCT04184622", "NCT04311411", "…"], "nct_ids_total": 41, "field_path": "protocolSection.sponsorCollaboratorsModule.leadSponsor.name", "citations": [{"nct_id": "NCT04081337", "excerpt": "Eli Lilly and Company"}, {"nct_id": "NCT04081337", "excerpt": "Tirzepatide", "field_path": "protocolSection.armsInterventionsModule.interventions[0].name"}], "citations_total": 82, "excerpts_truncated": true},
      "… 28 more edges"]}
  },
  "answer_summary": "The sponsor–drug pairs with the most trials are Novo Nordisk A/S – Semaglutide (66 trials), Eli Lilly and Company – Tirzepatide (41 trials), Eli Lilly and Company – Orforglipron (25 trials). The sponsors with the most trials here are Novo Nordisk A/S (174 trials), Eli Lilly and Company (118 trials). Based on 11,989 trials from ClinicalTrials.gov.",
  "meta": {"filters": {"conditions": ["obesity"], "interventional_only": true}, "truncation": {"top_n": 27, "other_bucket": false, "omitted_count": 5252}, "min_edge_weight": 2, "coverage": {"studies_matched_total": 11989, "studies_analyzed": 11989, "counting_method": "all_records", "sampled": false, "queries": ["https://clinicaltrials.gov/search?cond=obesity&advanced=AREA%5BStudyType%5DINTERVENTIONAL"]}, "validation": {"passed": true, "checks": ["retrieval_matches_api_total", "counts_match_trial_lists", "each_trial_in_one_bucket", "citations_from_retrieved_records", "summary_numbers_in_data", "value_axes_labeled"]}, "schema_version": "2.2", "…": "…"}
}
```

### More examples

* [`keynote189_survival.json`](examples/keynote189_survival.json): approximate overall/progression-free survival curves for KEYNOTE-189, reconstructed from reported medians and landmark rates.
* [`step1_dropout_flow.json`](examples/step1_dropout_flow.json): sankey of participant flow and dropout reasons per arm (STEP 1).
* [`melanoma_drug_combination_network.json`](examples/melanoma_drug_combination_network.json): drugs given together in the same arm (melanoma).
* [`step1_patient_eligibility.json`](examples/step1_patient_eligibility.json), [`keynote189_patient_eligibility.json`](examples/keynote189_patient_eligibility.json): eligibility checklists, each verdict quoting the criterion.
* [`step1_patient_side_effects.json`](examples/step1_patient_side_effects.json): adverse events by arm for a patient.
* [`semaglutide_vs_tirzepatide_phases.json`](examples/semaglutide_vs_tirzepatide_phases.json), [`breast_cancer_intervention_types.json`](examples/breast_cancer_intervention_types.json), [`obesity_trials_started_per_year.json`](examples/obesity_trials_started_per_year.json).

Reproduce one against a running server:

```bash
jq '.request' examples/alzheimers_recruiting_countries.json |
  curl -s localhost:8000/v1/visualize -H 'content-type: application/json' -d @- | jq .
```

`ctviz examples` regenerates every file. Live numbers drift as the registry changes; with `CTVIZ_OFFLINE=1`
the same requests replay from the recorded fixtures.

---

## AI usage and integrity

**How it was built.** Claude Code wrote most of the implementation code. I ran it as an orchestrating session with subagents. At runtime the service calls OpenAI
`gpt-5.4-nano` (planner, eligibility extraction) and `gpt-5.4-mini` (chart planner, results analysis,
escalation), from an allow-list; embeddings are off.

**What I designed.** I set the architecture and the rules in `CLAUDE.md`, and approved every piece before it
was built: for each agent I chose the design choices and limitations. The parts I deliberately designed:

* The agent pipeline: an LLM planner with a rule-based fallback and a validator that grounds every entity and
  drops filters the user did not ask for; a chart planner that may only pick from a menu the code has checked is
  buildable; an analysis agent bounded to choosing outcome measures and arms (≤ 1 call, validated arguments);
  verification of every citation and every summary number before a response leaves.
* The survival approach: curves reconstructed from reported medians and landmark rates, labelled approximate
  and never invented (interpolated points link to their reported anchors); a forest plot of reported hazard
  ratios across trials; a cohort fallback when a named trial has no results.
* The integrity rules: no LLM-generated numbers or summary text, and every datum cites the records behind it.
* Refusal and clarification: `unanswerable` for verdicts the registry cannot support, `needs_clarification`
  for unknown or ambiguous entities, never a silently widened search.

**How I checked correctness**

1. Offline tests with a mocked LLM and recorded ClinicalTrials.gov data: **2,687 pytest tests** (unit,
   contract, golden) and **391 frontend tests** (vitest). They cover the citation invariant, count/NCT-list
   agreement, the planner validator, rate limiting and verification.
2. Independent subagent passes: debugger audits, code reviews, a test writer whose end-to-end tests found real
   bugs, and the viz-spec validator, which re-checks the example files against the schema and every inline
   example citation against the raw records (959 citations in the current files, all verified).
3. Rounds of manual testing of real questions in the viewer. My feedback for multiple edge cases drove fixes, for
   example: a stale NCT id field now falls back to the cohort question; an empty pie caused by an encoding
   mismatch; false "subjective" refusals; misleading gap marking on heatmaps; 300 KB survival payloads slimmed
   to about 40 KB.
4. On every response, in code: citation re-verification against raw records, summary numbers checked against
   the data, and the six `meta.validation` checks. A response that fails is flagged and never cached.

Limits of this checking: tests use recorded data, so they do not catch changes in the live registry; live runs
covered the questions listed, not every combination of fields; LLM-dependent steps (planning, chart choice,
eligibility judgments) are validated for shape and grounding, not for being the best possible choice.
