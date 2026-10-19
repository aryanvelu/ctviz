# Response schema (v2.2): how to render any `POST /v1/visualize` response

This page is the renderer contract. Everything here is taken from the code:

| What | Where |
|---|---|
| Public JSON projection (what a client receives) | `ctviz/api/public.py` (`to_public`, documentation models `PublicResponse`, `PublicDatum`, `PublicCitation`) |
| Encoding, channels, rendering hints | `ctviz/api/schemas/viz.py` |
| Status, notes, coverage, validation | `ctviz/api/schemas/response.py` |
| Citations (internal model) | `ctviz/api/schemas/citations.py` |
| Final invariant checks | `ctviz/agents/validation.py` |
| Per-type channel wiring | `ctviz/viz/builders/<type>.py`, guarantees in `ctviz/viz/base.py::check_encoding` |

`GET /v1/schema` returns `{"schema_version": "2.2", "request": <JSON Schema of VisualizeRequest>, "response":
<JSON Schema of PublicResponse>}`.

**Serialization rule.** Keys whose value is `null`, `[]` or `{}` are dropped from `meta`, `encoding` and
`clarification`. `false` and `0` are kept. Datum rows are projected separately (they are already minimal), so
two things can appear inside `data` that are dropped elsewhere: `citations: []` on a datum with no evidence,
and `estimated_count: null` on trend rows (section 4.5). A missing key means "no value". For example, a
histogram bin without `bin_end` is open-ended, and a timeline row without `end` is a point event or is still
ongoing.

### What changed in 2.2 (from 2.1)

| Area | 2.1 | 2.2 |
|---|---|---|
| `datum.nct_ids` | Complete list on every datum | The **first 20** ids (sorted by `nct_id`, the same order as before) with `citation_detail` `sample` (default) or `ids_only`; complete with `full` |
| `datum.nct_ids_total` | absent | Always present when `nct_ids` is: the number of distinct trials behind the datum. `len(nct_ids) < nct_ids_total` means the list was capped (there is no separate flag) |
| Complete trial list | inline | `GET /v1/responses/{request_id}/datums/{datum_id}/citations` now also returns `nct_ids` (complete) and `nct_ids_total` |
| `derived_from` | (not emitted) | Provenance key on interpolated `survival_curve` points: the reported datum ids they were computed from |
| `GET /v1/graph/neighbors` | up to 25 `nct_ids` per edge | Same cap as datums: up to 20 `nct_ids` plus `nct_ids_total` |
| `meta.citation_policy` | `nct_ids: "complete"` | `nct_ids: "first_20"`, `nct_ids_per_datum: 20`, `nct_ids_order: "nct_id"`, `all_nct_ids` (endpoint template); `nct_ids: "complete"` only with `citation_detail=full` |

Why: a network node such as "Pembrolizumab" in a melanoma network stood for 284 trials, all inlined. The cap
keeps responses small; the count stays exact (`nct_ids_total`) and the full list is one request away. The
internal response, the validation checks (`counts_match_trial_lists` etc.) and the response cache still use
the complete lists.

### What changed in 2.1 (from 2.0)

| Area | 2.0 | 2.1 |
|---|---|---|
| Citation link | `url` on every citation | No per-citation `url`. Build it from `meta.citation_policy.study_url` (`https://clinicaltrials.gov/study/{nct_id}`). A registry-count datum has `query_url` |
| Field path | `field_path` on every citation | `field_path` once per datum (the path shared by its field-value excerpts). A citation repeats `field_path` only when it differs |
| `kind` | omitted for `field_value` | unchanged: only `text_excerpt` and `derived_count` are written |
| Excerpt counts | `citations_total` only when truncated | `citations_total` and `excerpts_truncated` on every datum (except `citation_detail=ids_only`) |
| Request option | none | `citation_detail`: `ids_only` \| `sample` (default) \| `full` |
| `meta.citation_policy` | absent | Describes the citation mode for this response |
| Trend rows | `trial_count` | plus `estimated_count` (trials whose start date is still an estimate) |
| Grouping | | `meta.grouping.each_trial_counted_once` and the validation check `each_trial_in_one_bucket` |
| Status | `status` values | plus `status_group` (active / done / other) for `group_by` / `series_by`; titles say "(status today: ...)" |
| Notes | | new code `SNAPSHOT` |
| `include_trace` | default `true` | default `false` |

---

## 1. Top-level shape

```jsonc
{
  "status": "ok" | "needs_clarification" | "unanswerable",
  "visualization": { "type", "title", "subtitle"?, "encoding", "data" },   // only when status == "ok"
  "answer_summary": "…",          // optional; deterministic template from computed values, never LLM text
  "clarification": { … },         // only when status != "ok"
  "meta": { … }                   // always present (section 5)
}
```

| `status` | `visualization` | `clarification` | What to show |
|---|---|---|---|
| `ok` | present | absent | The chart, plus `answer_summary`, `meta.notes` and the citations panel |
| `needs_clarification` | absent | present | `clarification.message`, `unmatched_terms`, `suggestions` (clickable), and `options` (pick a trial) |
| `unanswerable` | absent | present | `clarification.message` and `suggestions`. The registry cannot support this judgment (for example "which drug is safest") |

`clarification` fields:

| Field | Type | Meaning |
|---|---|---|
| `reason` | `unrecognized_entity` \| `no_entity` \| `subjective_question` \| `trial_required` \| `ambiguous_entity` | Why no chart was made |
| `message` | string | Text for the user (fixed template, never LLM text) |
| `unmatched_terms` | string[] | Terms that matched nothing in ClinicalTrials.gov |
| `suggestions` | string[] | Spelling suggestions ("Did you mean 'Tirzepatide'?") and questions this service can answer instead |
| `options` | `{nct_id, title, status}`[] | Candidate trials when the user must pick one |

The service never widens a search on its own. If a named entity matches no trial, the response is
`needs_clarification`. If the filters together match no trial, the response is `ok` with a "No matching
trials" `kpi_cards` chart and follow-up questions that name the filter to drop.

### `ok` (real live run, `"How many semaglutide trials started each year?"`; data trimmed to 2 of 22 rows, `nct_ids` trimmed)

```json
{
  "status": "ok",
  "visualization": {
    "type": "time_series",
    "title": "Trials started per year: Semaglutide",
    "encoding": {
      "x": {"field": "period", "type": "temporal", "title": "Start year", "time_format": "%Y", "scale": {"type": "time"}},
      "y": {"field": "trial_count", "type": "quantitative", "title": "Trials", "unit": "trials", "format": ",d"},
      "tooltip": ["label", "estimated_count"]
    },
    "data": [
      {"id": "d21", "period": "2027-01-01", "year": 2027, "label": "2027", "trial_count": 11, "estimated_count": 11,
       "nct_ids": ["NCT05714683", "NCT07430332", "NCT07462663", "…8 more"], "nct_ids_total": 11,
       "field_path": "protocolSection.statusModule.startDateStruct.date",
       "citations": [{"nct_id": "NCT05714683", "excerpt": "2027-01-31"},
                     {"nct_id": "NCT07430332", "excerpt": "2027-09-01"},
                     {"nct_id": "NCT07462663", "excerpt": "2027-01"}],
       "citations_total": 11, "excerpts_truncated": true},
      {"id": "d22", "period": "2028-01-01", "year": 2028, "label": "2028", "trial_count": 1, "estimated_count": 1,
       "nct_ids": ["NCT07586150"], "nct_ids_total": 1,
       "field_path": "protocolSection.statusModule.startDateStruct.date",
       "citations": [{"nct_id": "NCT07586150", "excerpt": "2028-10"}],
       "citations_total": 1, "excerpts_truncated": false}
    ]
  },
  "answer_summary": "2026 has the most trials (163), then 2025 (132), then 2024 (120). Based on 785 trials from ClinicalTrials.gov.",
  "meta": {
    "source": "clinicaltrials.gov",
    "filters": {"drug_names": ["semaglutide"]},
    "query_interpretation": {"intent": "trend_over_time", "persona": "researcher", "scope": "cohort", "confidence": 0.93,
                             "parser": "cache", "cohorts": [{"label": "Semaglutide", "drugs": ["semaglutide"]}],
                             "resolved_entities": [{"input": "semaglutide", "resolved": "semaglutide", "kind": "drug", "method": "exact"}]},
    "notes": [{"code": "PARTIAL_DATES", "severity": "info", "message": "1 trial(s) have no start date and are excluded."},
              {"code": "PARTIAL_DATES", "severity": "info", "message": "139 trial(s) have an estimated (anticipated) start date; future buckets reflect plans, not actual starts."}],
    "units": {"trial_count": "trials", "estimated_count": "trials"},
    "time_granularity": "year",
    "grouping": {"by": ["start_date"], "aggregate": "count_distinct_trials", "zero_filled": true, "each_trial_counted_once": true},
    "coverage": {"studies_matched_total": 785, "studies_analyzed": 785, "counting_method": "all_records", "sampled": false,
                 "queries": ["https://clinicaltrials.gov/search?intr=semaglutide"]},
    "citation_policy": {"detail": "sample", "study_url": "https://clinicaltrials.gov/study/{nct_id}",
                        "nct_ids": "first_20", "nct_ids_per_datum": 20, "nct_ids_order": "nct_id",
                        "all_nct_ids": "GET /v1/responses/{request_id}/datums/{datum_id}/citations",
                        "excerpts_per_datum": 3, "excerpts_single_trial": 10, "excerpt_order": "nct_id",
                        "all_excerpts": "GET /v1/responses/{request_id}/datums/{datum_id}/citations"},
    "validation": {"passed": true, "checks": [
      {"name": "retrieval_matches_api_total", "passed": true, "detail": "Semaglutide: 785 of 785"},
      {"name": "counts_match_trial_lists", "passed": true, "detail": "22 count datum(s) checked"},
      {"name": "each_trial_in_one_bucket", "passed": true, "detail": "784 trial(s), each in exactly one bucket"},
      {"name": "citations_from_retrieved_records", "passed": true, "detail": "784 NCT id(s), all retrieved"},
      {"name": "summary_numbers_in_data", "passed": true, "detail": "every number appears in the data"},
      {"name": "value_axes_labeled", "passed": true, "detail": "all value axes say what they measure"}]},
    "alternatives": [{"type": "bar_chart", "reason": "compare categories side by side"}, {"type": "table", "reason": "all values as a sortable table"}],
    "follow_ups": ["How many Semaglutide trials started each year, split by phase?"],
    "request_id": "ac63f7dc21a9424f8fd67516b016ffcf", "created_at": "2026-10-19T03:02:22+00:00",
    "schema_version": "2.2", "ctgov_api_version": "2.0.5", "data_timestamp": "2026-10-16T09:00:05"
  }
}
```

### `needs_clarification` (real live run, `"How many trials for tirzepatid started each year?"`)

```json
{
  "status": "needs_clarification",
  "clarification": {
    "reason": "unrecognized_entity",
    "message": "I could not find any ClinicalTrials.gov trials for 'tirzepatid', so I stopped instead of answering about a different set of trials. Can you check the name?",
    "unmatched_terms": ["tirzepatid"],
    "suggestions": ["Did you mean 'Tirzepatide'?",
                    "Check the spelling, or use the generic drug name or the official condition name (e.g. 'type 2 diabetes', 'pembrolizumab')."]
  },
  "meta": {"source": "clinicaltrials.gov", "filters": {"conditions": ["tirzepatid"]},
           "query_interpretation": {"intent": "trend_over_time", "persona": "researcher", "scope": "cohort", "confidence": 0.9, "parser": "llm",
                                    "cohorts": [{"label": "Tirzepatid", "conditions": ["tirzepatid"]}],
                                    "resolved_entities": [{"input": "tirzepatid", "resolved": "tirzepatid", "kind": "condition", "method": "passthrough"}]},
           "coverage": {"studies_matched_total": 0, "studies_analyzed": 0, "counting_method": "none", "sampled": false},
           "request_id": "b43b8e7a42bf46c785e3ebd73d51f529", "created_at": "2026-10-19T04:31:00+00:00", "schema_version": "2.2"}
}
```

### `unanswerable` (offline run, `"Which drug is safest?"`; message and meta trimmed)

```json
{
  "status": "unanswerable",
  "clarification": {
    "reason": "subjective_question",
    "message": "ClinicalTrials.gov cannot answer which treatment is safest, best or right for someone. …",
    "suggestions": ["Show the adverse events reported for one trial (give its NCT ID, e.g. NCT03548935)"]
  },
  "meta": {"source": "clinicaltrials.gov",
           "coverage": {"studies_matched_total": 0, "studies_analyzed": 0, "counting_method": "none", "sampled": false},
           "request_id": "399ad69b2a184930ac22b320594c39a4", "created_at": "2026-10-19T01:39:31+00:00", "schema_version": "2.2"}
}
```

Errors (HTTP 4xx/5xx) use a different body: `{"error", "message", "retry_after_s"?}`. Request validation
errors are `422` with `error: "validation_error"` and a `detail` list.

---

## 2. Renderer algorithm

1. **Read `status`.** If it is not `ok`, render `clarification` (section 1) and stop.
2. **Read `visualization.type`** (one of the 17 types in section 4) and pick the component. Show `title`,
   and show `subtitle` when present. Approximated charts use the subtitle to say so.
3. **Get the marks.** For row types, `data` is an array and each element is one mark (bar, point, cell,
   card, row). For graph types (`network_graph`, `sankey`), `data` is `{nodes: [...], edges: [...]}`.
4. **Map channels.** For each channel present in `encoding` (`x`, `x2`, `y`, `y2`, `series`, `size`,
   `label`, `error_low`, `error_high`), read `datum[channel.field]`. Use `channel.type` to pick the scale,
   `channel.title` and `unit` for the axis title, `format` / `time_format` to format ticks, `sort` to order
   categories, and `scale` to override the scale. A channel that is absent is not used. Never infer a field name.
5. **For graph types**, read the field names from `encoding.nodes` (`id`, `label`, `type`, `size`) and
   `encoding.edges` (`source`, `target`, `weight`, `type`). Edge `source` and `target` hold node `id` values.
6. **Apply `meta` rendering hints** that apply to the type: `units`, `sort`, `time_granularity`, `layout`
   (orientation / mode / donut / columns), `thresholds`, `bands`, `reference_line`, `geo`, `lanes`,
   `time_axis`, `stages`, `node_types`, `network_layout`, `pinned`, `highlight_zero`, `columns`, `summary`,
   `truncation`, `approximation`, `stats`. Section 4 lists which hints each type uses. Hints that do not
   apply to a type are absent.
7. **Tooltips.** Show every encoded field, plus each field named in `encoding.tooltip` that is present in
   that datum (tooltip fields may be missing in some rows).
8. **Citations panel.** When a mark is clicked:
   * list `datum.nct_ids` as links built from `meta.citation_policy.study_url`, headed with
     `datum.nct_ids_total` trials; if `nct_ids_total > len(nct_ids)` (capped at 20), add a "show all" link that
     reads `nct_ids` from `GET /v1/responses/{request_id}/datums/{id}/citations` (section 6);
   * show each `datum.citations[]` excerpt with its path (`citation.field_path`, else `datum.field_path`),
     linking `nct_id` the same way; a `derived_count` citation links to `datum.query_url`;
   * show `datum.cite_note` when the datum has no evidence;
   * if `excerpts_truncated` is `true`, add a "show all excerpts" link to
     `GET /v1/responses/{request_id}/datums/{id}/citations` (section 6).
9. **Footer.** Show `answer_summary`, then `meta.notes` (a `warning` severity gets a badge; `affects` lists
   datum ids to mark, and a missing `affects` means the whole chart), the `meta.coverage` line (for example
   "785 of 785 matching trials"), and `meta.approximation` when present. `meta.alternatives` can become
   "view as…" buttons, and `meta.follow_ups` can become suggested questions.

---

## 3. Channel object and datum keys

### Channel (`encoding.x`, `y`, …)

| Key | Type | Meaning |
|---|---|---|
| `field` | string | Datum key holding the value. **Always present and non-null in every row** for `x`, `y`, `series`, `size` and `label`, and for every `meta.columns` field (`check_encoding`). `x2`, `y2`, `error_low` and `error_high` may be missing in a row (for example an open-ended bin) |
| `type` | `nominal` \| `ordinal` \| `quantitative` \| `temporal` | Scale kind. `ordinal` means the order matters (use `sort.values`). `temporal` values are ISO date strings (`YYYY-MM-DD`, or `YYYY-MM` in a few raw fields) |
| `title` | string | Axis or legend title |
| `unit` | string? | For example `trials`, `participants`, `%`, `days`, `Cohen's d`. Value axes for `trial_count` always have one |
| `format` | string? | d3-format for numbers (`,d`, `.1%`, `.2f`). `%` formats mean the value is a fraction (0.098 shows as 9.8%). Never set on temporal channels |
| `time_format` | string? | d3-time-format for temporal channels (`%Y`, `%Y-%m`, `%Y-%m-%d`) |
| `sort` | `{values: [...]}` or `{by, order}` | `values` is the explicit category order (use it as the domain). `by` + `order` (`asc`/`desc`) sorts marks by a field |
| `scale` | `{type, domain?, zero?}` | `type`: `linear` \| `log` \| `sqrt` \| `time` \| `band`. `domain` fixes the range (for example `[0,1]`). `zero` makes the axis start at 0 |

`encoding.tooltip` is a string[] of extra datum keys.

**Axis convention.** `x` is always the category or time axis, and `y` is the value. For
`meta.layout.orientation = "horizontal"`, draw `x` categories down the vertical axis and the `y` values
horizontally. A threshold with `axis: "y"` is always on the **value** axis.

### Datum keys

Every row, node and edge is a flat object: `id`, then the fields named by the encoding / tooltip / columns,
then the provenance keys. The projection is `ctviz/api/public.py::_datum`.

**Provenance keys** (in this order, at the end of every datum):

| Key | Present | Meaning |
|---|---|---|
| `nct_ids` | record-backed datums | Distinct trials behind this datum, sorted by `nct_id`. **The first 20** (`meta.citation_policy.nct_ids_per_datum`) with `citation_detail` `sample` / `ids_only`; complete with `full`. The complete list: `GET /v1/responses/{request_id}/datums/{datum_id}/citations` (section 6). **Absent for registry counts** |
| `nct_ids_total` | whenever `nct_ids` is | Number of distinct trials behind this datum. Invariant: for a `trial_count` (or network `size` / `weight`) built from retrieved records, `nct_ids_total == value` (the server checks the complete list: `counts_match_trial_lists`). `len(nct_ids) < nct_ids_total` means the inline list was capped |
| `query_url` | registry counts | The ClinicalTrials.gov search whose `totalCount` equals the value (`coverage.counting_method = "registry_counts"`). `nct_ids` is omitted because the bucket's members were not all downloaded |
| `derived_from` | `survival_curve` interpolated points | Ids of the reported datums (same series, normally the two neighbouring anchors) this value was computed from. The point has no registry value of its own; follow the ids to the cited anchors. Never a data column. Kept in every `citation_detail` mode |
| `field_path` | when the datum has field-value excerpts | The dot path (into the study JSON) shared by the excerpts below, hoisted once. When excerpts sit at different paths, the most common one is used here and the others carry their own |
| `citations` | always, except `ids_only` | Excerpts: `{nct_id, excerpt}` plus `field_path` only when it differs from the datum's, `kind` only when not a field value, `char_start`/`char_end` for text excerpts, and an optional `label`. May be `[]` (then see `cite_note`) |
| `citations_total` | always, except `ids_only` | How many excerpts exist for this datum (inline or not) |
| `excerpts_truncated` | always, except `ids_only` | `true` when `citations_total > len(citations)`; fetch the rest from the citations endpoint |

With `citation_detail = "ids_only"` a datum carries only `nct_ids` / `nct_ids_total` / `query_url` / `derived_from` (no `field_path`, `citations`,
`citations_total`, `excerpts_truncated`).

**Other reserved keys:**

| Key | Where | Meaning |
|---|---|---|
| `id` | all | Unique within the visualization (`d1`, `s12`, `e4`, or graph node keys such as `sponsor:novo-nordisk-a-s`) |
| `cite_note` | uncited datums | Why there is no evidence, for example `"no supporting record (empty bucket or derived value)"` or `"total minus the counted buckets (derived)"` |
| `is_other` | bar / pie | `true` for the "Other" bucket that groups truncated categories (see `meta.truncation`). In registry-count mode it is a derived remainder with `cite_note` and `query_url` |

Type-specific reserved keys: `is_gap` (heatmap), `source` (survival_curve: `reported` \| `interpolated`),
`stage` (sankey nodes), `section` / `status` / `evaluated_by` (eligibility_checklist), `kind` (kpi_cards),
`is_open_ended` (histogram), `point` / `is_estimated` (timeline), `row_type` (single-trial site map),
`pinned` (network nodes), `estimated_count` (time_series).

---

## 4. Viz types

Notation: field names are the actual keys the handlers emit. "Hints" are `meta.*` keys. Provenance keys
(section 3) are trimmed with `…` in the examples.

### 4.1 `kpi_cards`

| Channel | Field | Meaning |
|---|---|---|
| `label` | `label` | Card title |
| `y` | `value` | Card value. `type` is `quantitative` only when every card is a number, otherwise `nominal` |
| tooltip | `unit`, `detail` | Optional unit suffix and secondary line |

Rows: `{label, value, kind, unit?, detail?}`. `kind` is `number` (format the number), `text`, `date`
(ISO string), `enum` (a registry code such as `COMPLETED`; show it as a badge), or `list` (`value` is a
string[]; render it as chips or bullets). Hints: `layout.columns` (grid columns). kpi_cards is also the
**empty / degraded state**: a single card with `cite_note`, `citations: []` and a `DEGRADED`,
`ZERO_RESULTS` or `NOT_IN_REGISTRY` note.

```json
{"type": "kpi_cards", "encoding": {"label": {"field": "label", "type": "nominal", "title": "Item"},
  "y": {"field": "value", "type": "nominal", "title": "Value"}, "tooltip": ["unit", "detail"]},
 "data": [{"id": "d2", "label": "Status", "value": "COMPLETED", "kind": "enum", "nct_ids": ["NCT03548935"], "nct_ids_total": 1,
           "field_path": "protocolSection.statusModule.overallStatus",
           "citations": [{"nct_id": "NCT03548935", "excerpt": "COMPLETED", "label": "Status"}],
           "citations_total": 1, "excerpts_truncated": false}]}
```

### 4.2 `table`

| Channel | Field | Meaning |
|---|---|---|
| `label` | first column's field | Row header |
| tooltip | handler-specific | Optional per-row cells (may be missing) |

Columns come from **`meta.columns`**: `[{field, title, type, format?}]`, in display order. Every column
field is present and non-null in every row. Typical rows are `{topic, text, source}` (record text tables)
or `{statement, source}` (results statements). Text cells are verbatim registry text, so preserve line
breaks.

**Registered outcomes (trial without posted results).** When a results question (outcomes, survival) is
asked about a trial that has no `resultsSection`, the answer is a table of the trial's *registered* outcome
measures instead of a chart of results (`ctviz/handlers/results.py::_registered_outcomes`): title
`"Registered outcomes: <trial>"`, subtitle `"<NCT>: no results posted · status … · enrollment …"`, columns
`type` (Primary / Secondary), `measure`, `time_frame`, with `description` (shortened, cited as a
`text_excerpt`) in the tooltip. Secondary outcomes are capped (a `TRUNCATED` note says how many were left
out). It always carries a `NO_RESULTS_POSTED` warning.

```json
{"type": "table", "title": "Registered outcomes: NCT06973564",
 "subtitle": "NCT06973564: no results posted · status Recruiting · enrollment 344 (estimated)",
 "encoding": {"label": {"field": "type", "type": "nominal", "title": "Type"}, "tooltip": ["description"]},
 "data": [{"id": "d1", "type": "Primary", "measure": "Part A: Number and proportion of patients who experience at least 1 dose-limiting toxicity (DLT) …",
           "time_frame": "Up to 1 year", "description": "A DLT is defined as any toxicities that meet the criteria for a DLT defined in the protocol",
           "nct_ids": ["NCT06973564"], "nct_ids_total": 1, "field_path": "protocolSection.outcomesModule.primaryOutcomes[0].measure",
           "citations": [{"nct_id": "NCT06973564", "excerpt": "Part A: Number and proportion …", "label": "Primary outcome"},
                         {"nct_id": "NCT06973564", "excerpt": "Up to 1 year", "field_path": "protocolSection.outcomesModule.primaryOutcomes[0].timeFrame", "label": "time frame"},
                         "…"], "…": "…"}]}
// meta.columns: [{"field":"type","title":"Type","type":"nominal"},{"field":"measure","title":"Outcome measure","type":"nominal"},{"field":"time_frame","title":"Time frame","type":"nominal"}]
// meta.notes: [{"code":"NO_RESULTS_POSTED","severity":"warning","message":"NCT06973564 has no results posted on ClinicalTrials.gov, so outcome results cannot be shown; the registered outcome measures are listed instead."}]
```

### 4.3 `bar_chart`

| Channel | Field (examples) | Meaning |
|---|---|---|
| `x` | `phase`, `status`, `status_group`, `sponsor`, `term`, `category`, `bar`, `country`, `intervention_type`, … | Category. Use `x.sort.values` as the category order |
| `y` | `trial_count` (unit `trials`), `rate` (`%`, format `.1%`), `n`, `value`, `endpoint_count`, `dropout_rate` | Bar length |
| `series` | `cohort`, `arm`, `phase`, `category`, `dimension`, `outcome_type` | Optional color / legend grouping |
| `error_low` / `error_high` | `error_low`, `error_high` | Optional error bars (outcome results); may be missing in a row |

Hints: `layout.orientation` (`vertical` \| `horizontal`), `layout.mode` (`grouped` \| `stacked` \|
`normalized`), `sort` (same as `x.sort`), `units`, `grouping` (how values were computed; display only),
`truncation` `{top_n, other_bucket, omitted_count}`, and `thresholds` (adverse events: the registry's
5 % reporting threshold, `source: "registry"`, with its own citations). Rows with `is_other: true` are the
"Other" bucket. Comparison charts have one row per (category, cohort).

`grouping.each_trial_counted_once: true` means the dimension is single-valued (phase, status, status group,
lead sponsor, sponsor class): each trial sits in exactly one bar per series, so the bars add up to the number
of distinct trials. `grouping.multi_valued: true` (intervention types, interventions, conditions) means a
trial can sit in several bars, so bars can add up to more than the trial total.

Registry-count bar (real live run, pancreatic cancer intervention types; `counting_method = "registry_counts"`):

```json
{"id": "d1", "intervention_type": "Drug", "trial_count": 2478,
 "query_url": "https://clinicaltrials.gov/search?cond=%22pancreatic+cancer%22&advanced=AREA%5BInterventionType%5DDRUG",
 "field_path": "protocolSection.armsInterventionsModule.interventions[0].type",
 "citations": [{"excerpt": "2478", "field_path": "$.totalCount", "kind": "derived_count", "label": "Drug"},
               {"nct_id": "NCT06885697", "excerpt": "DRUG"},
               {"nct_id": "NCT06973564", "excerpt": "DRUG"}],
 "citations_total": 4, "excerpts_truncated": true}
```

Adverse-event bar (single trial):

```json
{"type": "bar_chart", "encoding": {"x": {"field": "term", "type": "nominal", "title": "Adverse event", "sort": {"values": ["Any serious adverse event", "Nausea", "…"]}},
  "y": {"field": "rate", "type": "quantitative", "title": "Participants affected", "unit": "%", "format": ".1%"},
  "series": {"field": "arm", "type": "nominal", "title": "Arm"}, "tooltip": ["n_affected", "n_at_risk", "organ_system", "seriousness"]},
 "data": [{"id": "d3", "term": "Nausea", "arm": "Sema 2.4 mg", "rate": 0.441, "n_affected": 576, "n_at_risk": 1306,
           "organ_system": "Gastrointestinal disorders", "seriousness": "other", "nct_ids": ["NCT03548935"], "nct_ids_total": 1,
           "field_path": "resultsSection.adverseEventsModule.otherEvents[18].stats[0].numAffected",
           "citations": [{"nct_id": "NCT03548935", "excerpt": "576", "label": "Nausea / Sema 2.4 mg: affected"}, "…"], "…": "…"}]}
// meta: {"layout": {"orientation": "horizontal", "mode": "grouped", "donut": false}, "thresholds": [{"axis": "y", "value": 0.05, "label": "Registry reporting threshold for non-serious AEs (5%)", "source": "registry", "citations": ["…"]}]}
```

### 4.4 `histogram`

| Channel | Field | Meaning |
|---|---|---|
| `x` | `bin_start` | Bin start (inclusive), quantitative, linear scale with `zero: true` |
| `x2` | `bin_end` | Bin end (**exclusive**). **Missing on the open-ended last bin** |
| `y` | `trial_count` | Trials in the bin |
| `label` | `bin_label` | Ordinal label, for example `"100–249"` or `"10,000+"`. Use it as the tick label, or as a categorical axis fallback |
| `series` | `cohort` | Comparison only |

Rows: `{bin_start, bin_end?, bin_label, trial_count, is_open_ended, cohort?}`. Bins are `[start, end)` and
**unequal width** (edges 0, 25, 50, 100, 250, 500, 1,000, 2,500, 5,000, 10,000, then open-ended), so either
draw each bar from `bin_start` to `bin_end`, or draw equal-width ordinal bars labelled `bin_label`. For the
open-ended bin (`is_open_ended: true`), draw a fixed-width bar and use its label. Empty bins are present with
`trial_count: 0` and a `cite_note`. Hints: `summary` (display-only statistics such as median and quartiles of
enrollment; in comparisons it is keyed by cohort), `grouping` (`bin_interval: "[start, end)"`,
`open_ended_last_bin: true`), `units`.

```json
{"type": "histogram", "encoding": {"x": {"field": "bin_start", "type": "quantitative", "title": "Participants", "unit": "participants", "format": ",d", "scale": {"type": "linear", "zero": true}},
  "x2": {"field": "bin_end", "type": "quantitative", "title": "Bin end (exclusive)", "unit": "participants", "format": ",d"},
  "y": {"field": "trial_count", "type": "quantitative", "title": "Trials", "unit": "trials", "format": ",d"},
  "label": {"field": "bin_label", "type": "ordinal", "title": "Participants"}},
 "data": [{"id": "d1", "bin_start": 0, "bin_end": 25, "bin_label": "0–24", "trial_count": 12, "is_open_ended": false, "nct_ids": ["…"], "nct_ids_total": 12, "…": "…"},
          {"id": "d10", "bin_start": 10000, "bin_label": "10,000+", "trial_count": 3, "is_open_ended": true, "nct_ids": ["…"], "nct_ids_total": 3, "…": "…"}]}
// shape from the builder and handler code; values are illustrative
```

### 4.5 `time_series`

| Channel | Field | Meaning |
|---|---|---|
| `x` | `period` | Bucket start as an ISO date (`2015-01-01`). `temporal`, `scale.type: "time"`, `time_format` `%Y` (year) or `%Y-%m` (quarter / month) |
| `y` | `trial_count` | Trials whose start date falls in the bucket |
| `series` | `cohort`, or the `series_by` dimension (`phase`, `status`, `status_group`) | Optional: one line per value |
| tooltip | `label`, `estimated_count` | Display label of the bucket; trials in the bucket whose start date is still an estimate |

Rows: `{period, year, label, trial_count, estimated_count, cohort? | <series_by field>?}`.

**`estimated_count`** = trials in this bucket whose `protocolSection.statusModule.startDateStruct.type` is
`ESTIMATED` (a planned start, not an actual one). It is part of `trial_count`, not added to it. A renderer can
draw it as a dashed second series ("of which planned"). `null` means "not computed": in registry-count mode
(`ctviz/handlers/registry_counts.py::trend`) it is counted per year only from last year onward; earlier years
get one combined count, and if that count is not zero those years carry `null` and a `PARTIAL_DATES` note
says so. If any row has `estimated_count`, every row has it (`public.py`, `KEEP_NULL`).

Hints: `time_granularity` (`year` \| `quarter` \| `month`) gives the bucket width, `units` (`trial_count`,
`estimated_count`), and `grouping` (`zero_filled: true`, so empty buckets are present with 0 and the line is
continuous; `each_trial_counted_once: true`; `aggregate` is `count_distinct_trials` or `registry_count`). In
registry-count mode, rows carry `query_url` instead of `nct_ids`, and `PARTIAL_DATES` notes give the trials
outside the plotted range (before / after / no start date) so the total reconciles.

See section 1 for a full example.

### 4.6 `scatter_plot`

| Channel | Field | Meaning |
|---|---|---|
| `x` | `start` | Trial start date (`temporal`, `time_format` `%Y`) |
| `y` | `enrollment` | Participants, **`scale.type: "log"`**. Zero enrollments are not plotted (`TRUNCATED` note) |
| `series` | `phase`, or `cohort` in comparisons | Point color |
| `label` | `nct_id` | Point label / tooltip title |
| `size` | (optional) | Point size when set |

Rows: `{nct_id, start, start_raw, start_precision, enrollment, phase, cohort?}`. Dates with only a month
or year are placed on the first day of that period (`start_precision` is `month` or `year`, with a
`PARTIAL_DATES` note). Each point has exactly one trial in `nct_ids`.

### 4.7 `heatmap`

| Channel | Field (examples) | Meaning |
|---|---|---|
| `x` | `status`, `phase`, `year`, … | Column category (`ordinal`; order = `x.sort.values`) |
| `y` | `phase`, `condition`, `sponsor`, … | Row category (`ordinal`; order = `y.sort.values`) |
| `series` | `trial_count` | **The cell value** (quantitative) and drives the color scale |
| tooltip | `is_gap` | |

The grid is **complete**: every x × y cell is present. Cells with value 0 have `is_gap: true`, no
citations (`citations: []`), and a `cite_note`. Hints: `highlight_zero` (`true` only for the landscape gap
heatmap, `landscape_gap_heatmap`: style gap cells distinctly, for example hatched; they mean "no trials
here". The phase × status heatmap sends `false`: many empty cells there are structural, such as
expanded-access statuses that never carry a phase, so draw them as plain 0), `units`, `grouping`.

```json
{"type": "heatmap", "encoding": {"x": {"field": "phase", "type": "ordinal", "title": "Phase", "sort": {"values": ["…"]}},
  "y": {"field": "status", "type": "ordinal", "title": "Status", "sort": {"values": ["…"]}},
  "series": {"field": "trial_count", "type": "quantitative", "title": "Trials", "unit": "trials", "format": ",d"}, "tooltip": ["is_gap"]},
 "data": [{"id": "d1", "phase": "…", "status": "…", "trial_count": 0, "is_gap": true, "cite_note": "no supporting record (empty bucket or derived value)",
           "citations": [], "citations_total": 0, "excerpts_truncated": false}]}
// shape from the builder code; values are illustrative
```

### 4.8 `pie_chart`

| Channel | Field | Meaning |
|---|---|---|
| `label` | the category field (for example `phase`, `status`, `intervention_type`) | Slice name. **Pie uses `label`, not `x`** |
| `y` | `trial_count` | Slice value. Compute shares as value / sum(values) |

Hints: `layout.donut` (`true` by default, so draw a ring), `truncation` / `is_other` (the "Other" slice),
`units`. Pie charts always have a single series (comparisons are never drawn as pies). Some dimensions,
such as intervention types, are **multi-valued** (`grouping.multi_valued: true`): one trial can be counted
in several slices, so slices can sum to more than the trial total. Do not label slices "% of trials" in
that case.

### 4.9 `network_graph`

`data = {nodes, edges}`. `encoding.nodes = {id: "id", label: "label", type: "type", size: "size"}` and
`encoding.edges = {source: "source", target: "target", weight: "weight", type: "type"}`.

| Element | Fields | Meaning |
|---|---|---|
| node | `id`, `label`, `type`, `size`, `degree`, `pinned?` | `type` is the node class (legend in `meta.node_types`). `size` = distinct trials the entity appears in (unit `trials`); `nct_ids_total == size` (inline `nct_ids` capped at 20). `degree` = links shown. `pinned: true` = an entity the user named |
| edge | `id`, `source`, `target`, `weight`, `type` | `weight` = **distinct trials** in which the two entities co-occur (unit `trials`); `nct_ids_total == weight`. `type` is for example `sponsor_tests_drug` |

Network kinds (`meta.network_kind`), titles and edge meaning (`ctviz/handlers/cohort.py::NETWORKS`):

| `network_kind` | Title prefix | Node types | Edge (`type`): meaning |
|---|---|---|---|
| `sponsor_drug` | "Sponsors and the drugs they test" | `sponsor`, `drug` | `sponsor_tests_drug`: the lead sponsor runs at least one trial testing the drug (directed) |
| `drug_drug` | "Drugs given together in the same arm" | `drug`, `biological` | `combined_in_arm`: both are given in the same arm (`armGroups[].interventionNames`, ≥ 2 drugs/biologicals, placebo excluded) |
| `drug_condition` | "Drugs and the conditions they are tested for" | `drug`, `condition` | `drug_studied_for` (directed) |
| `sponsor_collaborator` | "Lead sponsors and their collaborators" | `lead_sponsor`, `collaborator` | `collaborates_with` (directed) |
| `site_investigator` | "Investigators and their trial sites" | `investigator`, `site` | `official_of_trial_at_site` (directed; at most 50 sites per trial) |

The title is `"<prefix>: <cohort label>"`, plus `"(status today: …)"` when a status filter was applied.

**Which nodes and edges are kept** (`ctviz/compute/network.py::CooccurrenceGraph.prune`, called from
`entity_network`):

1. **Minimum edge weight.** If the cohort has at least 10 links seen in 2 or more trials, links seen in a
   single trial are dropped first (floor = 2); otherwise the floor is 1.
2. **Nodes.** At most `top_n` (default 30, max 40) nodes, ranked by their summed link weight. Entities the
   user named are pinned: always kept, half of the remaining node budget goes to their strongest neighbours,
   and their links are never removed by the weight threshold.
3. **Edges.** At most 120 links. If more remain, the minimum weight is raised until they fit; if raising it
   leaves room, the remaining slots are filled from the next-weaker tier, preferring better-connected
   entities (ties broken by name). Nodes left without links are removed (pinned nodes stay).
4. **Ego layout.** A `drug_drug` question about one named drug ("which drugs are combined with X") keeps only
   X's direct partners (at most 21 nodes) and sets `network_layout: "ego"`.

`meta.min_edge_weight` is the weakest weight kept. `TRUNCATED` notes say how many entities and links were
left out and the weight a link needed.

Hints: `node_types` `[{type, label, count}]` (legend), `edge_semantics` (a sentence saying what an edge
means; show it under the legend), `network_kind`, `directed` (draw arrows from `source` to `target`),
`min_edge_weight`, `pinned` (node ids), `network_layout` (`ego`: place the pinned node at the center and the
others around it), `truncation` (`top_n` = the number of nodes actually shown, which can be below the
requested limit after pruning; `omitted_count` = entities left out), `grouping: {"weight": "distinct_trials"}`, and
`units: {weight: "trials", size: "trials"}`. Edge endpoints always reference existing node ids.

Real live run, `"Show the network of sponsors and drugs in lung cancer trials"` (10,649 trials, all read):

```json
{"type": "network_graph", "title": "Sponsors and the drugs they test: Lung cancer",
 "encoding": {"nodes": {"id": "id", "label": "label", "type": "type", "size": "size"},
  "edges": {"source": "source", "target": "target", "weight": "weight", "type": "type"}, "tooltip": ["degree"]},
 "data": {"nodes": [{"id": "sponsor:national-cancer-institute-nci", "label": "National Cancer Institute (NCI)", "type": "sponsor",
                     "size": 297, "degree": 18, "nct_ids": ["NCT00001450", "…19 more"], "nct_ids_total": 297, "…": "…"}],
          "edges": [{"id": "e1", "source": "sponsor:eli-lilly-and-company", "target": "drug:pemetrexed", "weight": 62,
                     "type": "sponsor_tests_drug", "nct_ids": ["NCT00061451", "…19 more"], "nct_ids_total": 62, "…": "…"}]}}
// meta: {"node_types": [{"type": "sponsor", "label": "Lead sponsor", "count": 11}, {"type": "drug", "label": "Drug / biological", "count": 19}],
//        "edge_semantics": "Lead sponsor runs at least one trial testing the drug; weight = distinct trials.",
//        "network_kind": "sponsor_drug", "directed": true, "min_edge_weight": 3, "grouping": {"weight": "distinct_trials"},
//        "truncation": {"top_n": 30, "other_bucket": false, "omitted_count": 8794}, "units": {"weight": "trials", "size": "trials"}}
// notes: TRUNCATED "Showing 30 of 8824 entities (most connected) and 120 of 11983 links; other links need weight >= 3."
```

### 4.10 `sankey`

`encoding.nodes = {id: "id", label: "label", type: "group", size: "value"}` and
`encoding.edges = {source, target, weight, type}`.

| Element | Fields | Meaning |
|---|---|---|
| node | `id`, `label`, `stage`, `group`, `value` | `stage` is the **column index** into `meta.stages`. `group` = arm (color by it). `value` = participants |
| edge | `id`, `source`, `target`, `weight`, `type`, `group`, `share_of_started` | `weight` = participants. `type` is `completed` \| `dropout` \| `unexplained`. `share_of_started` = weight / participants who started the arm (0–1) |

Hints: `stages` (column names, for example `["Started", "End of study"]`), `units.weight: "participants"`,
`stats.dropout_rate` (per arm, 0–1), `stats.periods`. `unexplained` edges and nodes ("reason not
reported") are derived remainders with no citation and a `cite_note`. Arms are independent flows, so lay
out one band per `group`.

```json
{"type": "sankey", "encoding": {"nodes": {"id": "id", "label": "label", "type": "group", "size": "value"},
  "edges": {"source": "source", "target": "target", "weight": "weight", "type": "type"}, "tooltip": ["share_of_started", "group"]},
 "data": {"nodes": [{"id": "FG000:start", "label": "Started", "stage": 0, "group": "Semaglutide 2.4 mg", "value": 1306, "nct_ids": ["NCT03548935"], "nct_ids_total": 1,
                     "field_path": "resultsSection.participantFlowModule.periods[0].milestones[0].achievements[0].numSubjects",
                     "citations": [{"nct_id": "NCT03548935", "excerpt": "1306", "label": "Semaglutide 2.4 mg: started"}], "…": "…"}],
          "edges": [{"id": "e2", "source": "FG000:start", "target": "FG000:p0:r0", "weight": 26, "type": "dropout", "group": "Semaglutide 2.4 mg",
                     "share_of_started": 0.0199, "nct_ids": ["NCT03548935"], "nct_ids_total": 1, "…": "…"}]}}
// meta: {"stages": ["Started", "End of study"], "units": {"weight": "participants"}, "stats": {"dropout_rate": {"Semaglutide 2.4 mg": 0.0505, "Placebo": 0.0702}}}
```

### 4.11 `timeline`

There are two forms. Tell them apart by `meta.lanes`.

**A. Single axis** (no `meta.lanes`; `meta.time_axis` is present). Used for trial activity and development progression.

| Channel | Field | Meaning |
|---|---|---|
| `y` | `task` | Row label (for example `"STEP 1 · NCT03548935"`, or a phase) |
| `x` | `start` | Bar start (`temporal`, `%Y-%m-%d`) |
| `x2` | `end` | Bar end. **Missing → point event or still ongoing** (`point: true`) |
| `series` | `category` | Color (status or phase). Use `series.sort.values` when present |
| tooltip | `title`, `lane`, `status`, `is_estimated`, `point`, `trial_count`, `first_nct`, … | |

Rows also carry `start_raw` / `start_precision` and `end_raw` / `end_precision` (`day` \| `month` \|
`year`). `is_estimated: true` marks planned dates, so draw them hatched or dashed. Hints: `time_axis`
`{kind: "calendar"}`, `time_granularity: "day"`, and `grouping.lane_order` (group rows by `lane` in that
order).

**B. Multi-lane** (`meta.lanes` present). Used for the single-trial journey. `encoding` has only
`y = event` (row label) and `series = lane`. Each row's `lane` value names a lane in `meta.lanes`, and
**that lane gives the row's fields and axis**:

```json
"lanes": [
  {"name": "registry_milestones", "axis": {"kind": "calendar"}, "start_field": "date", "approximated": false},
  {"name": "participant_journey", "axis": {"kind": "relative", "unit": "days", "origin": "baseline / start of participation"},
   "start_field": "day", "end_field": "day_end", "approximated": true}
]
```

Draw each lane on **its own x axis**: calendar dates for `kind: "calendar"`, and numeric offsets in
`axis.unit` from `axis.origin` for `kind: "relative"`. Never put dates and offsets on one axis. A row
without its lane's `end_field` is a point. When `day == day_end`, the row is also a point. A lane with
`approximated: true` should be styled as approximate (see `meta.approximation` and the `APPROXIMATED` note,
whose `affects` lists those rows). Tooltip fields: `date_type` (`ACTUAL` \| `ESTIMATED`), `detail`.

```json
{"type": "timeline", "encoding": {"y": {"field": "event", "type": "nominal", "title": "Event"}, "series": {"field": "lane", "type": "nominal", "title": "Lane"}, "tooltip": ["date_type", "detail"]},
 "data": [{"id": "d1", "lane": "registry_milestones", "event": "Study start", "date": "2018-06-04", "date_type": "ACTUAL", "nct_ids": ["NCT03548935"], "nct_ids_total": 1,
           "field_path": "protocolSection.statusModule.startDateStruct.date",
           "citations": [{"nct_id": "NCT03548935", "excerpt": "2018-06-04", "label": "Study start"}], "…": "…"},
          {"id": "d6", "lane": "participant_journey", "event": "Assessment (…)", "day": 0, "day_end": 0, "detail": "…", "…": "…"}]}
```

### 4.12 `survival_curve`

| Channel | Field | Meaning |
|---|---|---|
| `x` | `time` | Time in `x.unit` (`months` by default), `scale.zero: true` |
| `y` | `survival` | S(t), 0–1, format `.0%`, domain `[0,1]` |
| `series` | `series` | One line per "endpoint · arm" |
| tooltip | `source`, `ci_low`, `ci_high`, `label` | |

Rows: `{series, time, survival, source, endpoint, group, ci_low?, ci_high?, time_ci_low?, time_ci_high?,
label}`. **These curves are approximations**, rebuilt from reported medians and landmark rates. They are
not Kaplan–Meier data. `meta.approximation` `{method, assumptions[]}` is always present, together with a
`warning` `APPROXIMATED` note. Draw step or line segments: solid through `source: "reported"` points (these
are cited), and **dashed** for `source: "interpolated"` points (listed in the note's `affects`). An
interpolated point cites only the measure title and carries `derived_from: [datum ids]`, the reported anchor
points it was computed from; show them as its sources. The point
at t=0 (S=1) is "by definition". `stats.analyses` may list reported hazard ratios. If no anchors exist,
the handler returns a `forest_plot` or `table` instead; a trial with no posted results gets the
registered-outcomes table (4.2).

### 4.13 `threshold_line`

| Channel | Field | Meaning |
|---|---|---|
| `x` | `group` | Dose group (`ordinal`, order = `x.sort.values`) |
| `y` | `rate` | Event rate 0–1 (format `.0%`, domain `[0,1]`) |
| `series` | (optional) | |

Rows: `{group, dose_order, rate, n_events, n_evaluable, dose?}`. Draw points joined by a line in category
order, and draw each `meta.thresholds[]` entry as a horizontal rule at `value` on the value axis, with its
`label`. `source` is `convention` (for example "33% DLT rate (3+3 design convention; not from the
registry)"), `registry` (cited, with its own `citations`), or `user`. Show the source next to the label.

### 4.14 `power_curve`

| Channel | Field | Meaning |
|---|---|---|
| `x` | `effect_size` | Cohen's d, format `.2f` |
| `y` | `power` | 0–1, format `.0%`, domain `[0,1]` |
| `series` | `series` | Sample-size scenario, for example `"n/2: 436 per arm"` |
| tooltip | `n_per_arm` | |

Hints: `thresholds` (`{axis: "y", value: 0.8, label: "80% power (convention)", source: "convention"}`),
`stats.mde_at_80` (minimum detectable effect per scenario), `stats.power_inputs`, `approximation` (always
present: method plus assumptions), and the `subtitle` "Approximation from the sample size; not the trial's
registered power calculation". The citations point to the sample size that was used.

### 4.15 `forest_plot`

| Channel | Field | Meaning |
|---|---|---|
| `y` | `label` | Comparison / analysis label (one row each) |
| `x` | `estimate` | Point estimate. **`scale.type: "log"` for ratios** (HR, OR, RR), `linear` for differences |
| `error_low` / `error_high` | `ci_low` / `ci_high` | Confidence interval whiskers |
| tooltip | `p_value`, `ci_pct`, `param_type`, `method` | `p_value` is the raw registry string (for example `"<0.001"`) |

Rows: `{label, estimate, ci_low, ci_high, ci_pct, p_value?, param_type, null_value?, method?, measure}`.
Hints: **`reference_line` `{value: 1 (ratio) | 0 (difference), label: "No difference"}`** (draw a vertical
rule), `diagnostics` (which measures were selected; display only). Some rows may lack a CI bound; draw the
point without a whisker when the bound is missing.

### 4.16 `geo_map`

Read **`meta.geo.level`** first.

**Country level** (`geo.level = "country"`): choropleth or bubbles.

| Channel | Field | Meaning |
|---|---|---|
| `x` / `label` | `country` | Region name as written in the registry (for example `"United States"`). Join it to shapes by name. `meta.geo.region_field` = `"country"` |
| `y` | `trial_count` | Value (`meta.geo.value_field`). **Distinct trials** with at least one site in the country |
| `series` | `cohort` | Comparison only (small multiples or a toggle) |
| tooltip | `site_count` | Sites in that country (unit `sites`, `meta.units.site_count`) |

A trial with 20 U.S. sites adds 1 to the United States. Hints: `geo`, `sort` (`{by: "trial_count", order:
"desc"}`), `units`, `grouping.country_grouping`, `grouping.per_trial`. Because `x` and `y` are set, a bar chart fallback needs no
changes. Country counts are multi-valued (a trial with sites in 3 countries counts in each).

**Site level** (`geo.level = "site"`): points. `meta.geo.lat_field` / `lon_field` are `"lat"` / `"lon"`.

| Channel | Field | Meaning |
|---|---|---|
| `label` | `site` (cohort) or `facility` (single trial) | Marker label |
| `size` | `trial_count` (cohort only) | Marker size, trials at the site |
| tooltip | `city`, `state`, `country`, and for a single trial also `status`, `contact_name`, `contact_phone`, `contact_email`, `text` | |

**Rows without `lat`/`lon` are listed but not drawn** (a `NOT_IN_REGISTRY` note gives the number; show
them in a side list). Single-trial maps also carry `row_type`: `site` (has coordinates when known),
`central_contact` or `record_mention` (never drawn; put them in a contacts panel). `meta.stats.mappable_rows`
is the number of drawable rows.

```json
{"type": "geo_map", "encoding": {"label": {"field": "facility", "type": "nominal", "title": "Site"},
  "tooltip": ["city", "state", "country", "status", "contact_name", "contact_phone", "contact_email", "text"]},
 "data": [{"id": "s1", "row_type": "site", "facility": "Novo Nordisk Investigational Site", "city": "Anniston", "state": "Alabama",
           "country": "United States", "lat": 33.65983, "lon": -85.83163, "nct_ids": ["NCT03548935"], "nct_ids_total": 1,
           "field_path": "protocolSection.contactsLocationsModule.locations[0].facility",
           "citations": [{"nct_id": "NCT03548935", "excerpt": "Novo Nordisk Investigational Site"}], "…": "…"}]}
// meta: {"geo": {"level": "site", "lat_field": "lat", "lon_field": "lon"}, "stats": {"mappable_rows": 120}}
// country level meta.geo: {"level": "country", "region_field": "country", "value_field": "trial_count"}
```

### 4.17 `eligibility_checklist`

| Channel | Field | Meaning |
|---|---|---|
| `label` | `text` | Criterion text (verbatim, or a structured summary such as "Age: 18 Years or older") |
| `series` | `status` | `met` \| `unmet` \| `needs_review` \| `unknown` (order fixed in `series.sort.values`) |
| tooltip | `section`, `reason`, `evaluated_by`, `relevance` | |

Rows: `{section, index, text, status, reason, evaluated_by, relevance?, subheading?, parent_text?}`.

| Key | Values | Meaning |
|---|---|---|
| `section` | `structured` \| `inclusion` \| `exclusion` \| `unknown` | Group rows under these headings. `unknown` = the record has no clear heading |
| `status` | `met` = does not block · `unmet` = blocks · `needs_review` = confirm with the study team · `unknown` = not evaluated | For **exclusion** rows, `met` means the person does *not* have the excluding condition |
| `evaluated_by` | `rule` \| `llm` \| `none` | `llm` rows come with an `LLM_EVALUATED` note; their quoted text is substring-verified |
| `parent_text` | string | The list item a nested criterion sits under (indent it) |

Hints: `summary` `{total, by_status, by_section, by_evaluator}` (header counts). Unevaluated free-text
criteria show as `unknown` with a `DEGRADED` note. Use neutral wording ("the study team decides"), not a
verdict.

---

## 5. `meta` reference

| Key | Always? | Meaning |
|---|---|---|
| `source` | yes | `"clinicaltrials.gov"` |
| `filters` | when filtered | The filters actually searched: `drug_names`, `conditions`, `sponsors`, `countries`, `phases`, `statuses`, `start_year`, `end_year`, `term`, `interventional_only`, `nct_ids`. For comparisons: `{compare: [{label, …filters}]}` with one entry per cohort |
| `query_interpretation` | yes | `{intent, persona (patient\|researcher), scope (cohort\|comparison\|single_trial), confidence (0–1), parser (llm\|rules\|cache), cohorts[], resolved_entities[{input, resolved, kind, method}]}`. `method`: `exact`, `alias`, `mesh`, `structured_field`, `passthrough` (searched as written), `spelling_correction` (show "searched X for Y"), `knowledge_graph` (trial records list the input as another name of `resolved`; both were searched) |
| `assumptions` | when any | string[]: defaults the planner applied, and the chart planner's choice ("Chart chosen after profiling the retrieved trials: …"). Show them as "assumed: …" |
| `notes` | when any | `[{code, severity (info\|warning), message, affects?}]`. `affects` = datum ids. If it is absent, the note applies to the whole chart |
| rendering hints | per type | `units`, `sort`, `time_granularity`, `grouping`, `layout`, `thresholds`, `bands` `[{axis, low, high, label}]` (shade a range), `truncation`, `approximation`, `columns`, `summary`, `node_types`, `edge_semantics`, `network_kind`, `directed`, `min_edge_weight`, `pinned`, `network_layout`, `stages`, `time_axis`, `lanes`, `geo`, `reference_line`, `highlight_zero`, `stats` (section 4) |
| `coverage` | yes | `{studies_matched_total, studies_analyzed, counting_method, studies_with_results?, sampled, sampling_method?, queries[]}`. `queries` are ClinicalTrials.gov search URLs that reproduce the cohort |
| `citation_policy` | `ok` responses | `{detail, study_url, nct_ids}`. `nct_ids` is `"first_20"` for `sample` / `ids_only`, together with `nct_ids_per_datum: 20`, `nct_ids_order: "nct_id"` and `all_nct_ids` (the endpoint template); it is `"complete"` for `full`. For `sample` also `excerpts_per_datum: 3`, `excerpts_single_trial: 10`, `excerpt_order: "nct_id"`, `all_excerpts` (the endpoint template); for `full`, `excerpts_per_datum: "all"` |
| `validation` | `ok` responses | `{passed, checks: [{name, passed, detail}]}` (below) |
| `alternatives` | when any | `[{type, reason}]`: other compatible viz types (re-request with `preferred_visualization`) |
| `follow_ups` | when any | Suggested next questions |
| `request_id`, `created_at`, `schema_version`, `ctgov_api_version`, `data_timestamp` | ids always; versions when known | `data_timestamp` = the ClinicalTrials.gov data snapshot time |
| `debug` | only with `include_trace: true` | `{api_params, cache, trace, chart_diagnostics}`: pipeline steps, timings, LLM calls and cost. Off by default. Never needed to draw |

`stats` holds extra display-only figures (for example dropout rate per arm, minimum detectable effect).
`approximation` (`{method, assumptions[]}`) is present **only** for approximated charts, so show an
"approximation" banner whenever it exists.

`grouping` keys a renderer may show: `by`, `aggregate` (`count`, `count_distinct_trials`,
`registry_count`), `zero_filled`, `each_trial_counted_once` (single-valued dimension: bars of one series add
up to distinct trials), `multi_valued` (a trial can count in several buckets), `phase_combination`
(`joined`: "Phase 1/2" is its own bucket), `weight: "distinct_trials"` (networks), `per_trial` (geo_map by
country: `"once per country"`, a trial with sites in several countries counts once in each), `rate`
(adverse events and cross-trial dropout: the formula behind the rate, for example
`"numAffected / numAtRisk"`).

### Status: "status today"

`overallStatus` in ClinicalTrials.gov is the trial's status **now**. The registry keeps no status history,
so a filter on status describes trials as they are today, not as they were in the year they started.

* Request `status[]` accepts `RECRUITING`, `NOT_YET_RECRUITING`, `ENROLLING_BY_INVITATION`,
  `ACTIVE_NOT_RECRUITING`, `SUSPENDED`, `TERMINATED`, `COMPLETED`, `WITHDRAWN`, `UNKNOWN`. The planner also
  reads phrases such as "recruiting trials", "completed studies", "active/ongoing trials", "finished trials".
* Status groups (`ctviz/ctgov/fields.py::STATUS_GROUPS`):

  | Group | Label | Statuses |
  |---|---|---|
  | `ACTIVE` | Active | `NOT_YET_RECRUITING`, `RECRUITING`, `ENROLLING_BY_INVITATION`, `ACTIVE_NOT_RECRUITING` |
  | `DONE` | Done (completed, terminated, withdrawn) | `COMPLETED`, `TERMINATED`, `WITHDRAWN` |
  | `OTHER` | Other (suspended, unknown, expanded access) | `SUSPENDED`, `UNKNOWN`, expanded-access statuses, `WITHHELD` |

* `group_by: "status_group"` (status distribution, field `status_group`, title "Trials by status today
  (active / done / other): …") and `series_by: "status_group"` (trends, series title "Status today
  (group)") split by these groups. "active vs completed" in a question is read as this split, not as a filter.
* When a status filter is applied, chart titles end with `"(status today: <statuses>)"`, for example
  `"Trials started per year: Obesity (status today: recruiting)"`; a whole group reads as
  `"status today: active"` or `"status today: completed, terminated or withdrawn"`.

### Note codes

| Code | Meaning |
|---|---|
| `NOT_IN_REGISTRY` | The registry does not record this (for example visit schedule, coordinates for some sites, AEs below 5 %) |
| `APPROXIMATED` | Values derived by a stated method (survival curves, journey offsets, power), or a stated reading of the data (multi-valued buckets, an ambiguous name resolved to one meaning) |
| `SAMPLED` | Only part of the matching trials was analyzed (see `coverage.sampling_method`) |
| `SNAPSHOT` | All trial records came from a stored cohort snapshot (built by `ctviz warm` or an earlier complete read); ClinicalTrials.gov still reports the same `totalCount` for the search. The message gives when it was taken |
| `NO_RESULTS_POSTED` | The trial has no posted results section (results questions then show the registered outcomes, 4.2) |
| `LLM_EVALUATED` | Some rows were judged by an LLM; quoted text was verified as a substring of the record |
| `ZERO_RESULTS` | The search or computation produced nothing to draw |
| `TRIAL_REQUIRED` | The question needs a specific NCT ID |
| `ENTITY_AMBIGUOUS` | A name matched several entities; the one used is stated |
| `TRUNCATED` | Top-N shown; the rest are omitted or grouped as "Other" (see `meta.truncation`) |
| `PARTIAL_DATES` | Month- or year-only, missing, or estimated dates were handled as described |
| `VIZ_DOWNGRADED` | The preferred chart could not be built from this data; a simpler type was used |
| `VIZ_PREFERENCE_IGNORED` | `preferred_visualization` is incompatible with the intent |
| `DEGRADED` | A dependency failed (ClinicalTrials.gov, LLM), or citations failed re-verification and were removed. Part of the answer is a fallback |
| `SUMMARY_TRIMMED` | A sentence of `answer_summary` was dropped because its numbers were not in the data |
| `REGISTRY_COUNTS` | Each bucket is an exact server-side `totalCount` (`datum.query_url`), not a count of downloaded records |
| `VALIDATION_FAILED` | An invariant check failed; see `meta.validation` (`warning`) |
| `TERM_SEARCHED_VERBATIM` | A term was searched as typed (not in the built-in vocabulary, but ClinicalTrials.gov has trials for it) |
| `FILTERS_RELAXED`, `CITATION_DROPPED` | Defined in the enum but not emitted by the current code (the service no longer loosens filters; dropped citations are reported with `DEGRADED`) |

### `coverage.counting_method`

| Value | Meaning | What the datums carry |
|---|---|---|
| `all_records` | Every matching record was retrieved and counted (`studies_analyzed == studies_matched_total`), live or from a cohort snapshot (`SNAPSHOT` note) | `nct_ids` (first 20) + `nct_ids_total`; `nct_ids_total == trial_count` |
| `registry_counts` | Too many matches to download for this chart. Each bucket is an exact ClinicalTrials.gov `totalCount` for a filtered search | `query_url`, a `derived_count` citation plus a few example-trial excerpts; no `nct_ids` |
| `sample` | Only part of the matches was analyzed (`sampled: true`, `sampling_method` says how, `SAMPLED` warning) | `nct_ids` within the sample |
| `single_trials` | Specific NCT records | `nct_ids` = that trial |
| `none` | No data was needed or fetched (clarifications, degraded states) | n/a |

### `validation.checks` (`ctviz/agents/validation.py`)

| `name` | Passes when |
|---|---|
| `retrieval_matches_api_total` | `all_records`: the number of distinct records fetched equals `totalCount` for every cohort. `sample`: a `SAMPLED` note is present. `registry_counts`: buckets + outside-range + no-value counts add up to `totalCount` (or the dimension is multi-valued) |
| `counts_match_trial_lists` | Every `trial_count` datum's complete trial list (internal; public `nct_ids_total`) has exactly `trial_count` ids, or the datum has a `query_url` (registry count). For networks, node `size` and edge `weight` are checked the same way (a sankey counts participants and is skipped) |
| `each_trial_in_one_bucket` | When `grouping.each_trial_counted_once` is set: no NCT id appears in two rows of the same series, so the bars add up to distinct trials. Registry-count charts are checked by reconciliation instead |
| `citations_from_retrieved_records` | Every NCT id in `citations` and `nct_ids` was actually retrieved for this request |
| `summary_numbers_in_data` | Every number in `answer_summary` appears in the computed data or facts |
| `value_axes_labeled` | Every quantitative `x`/`y`/`size`/`series` channel has a descriptive title, and count fields have a unit |

Any failure adds a `VALIDATION_FAILED` warning, and the response is not stored in the response cache.

---

## 6. Citations

Request option **`citation_detail`** (default `sample`) controls how much evidence is inline:

| Mode | Per datum | Use |
|---|---|---|
| `ids_only` | first 20 `nct_ids` + `nct_ids_total` / `query_url` only | Drawing; smallest payload |
| `sample` | first 20 `nct_ids` + `nct_ids_total` + the first 3 excerpts by `nct_id` (up to 10 for a datum backed by one trial) | Default |
| `full` | **all** `nct_ids` + `nct_ids_total` + every excerpt inline; `excerpts_truncated` is `false` | Audits, offline review. Large for big cohorts |

`full` is the explicit opt-in for complete inline lists. In the other modes the trial list is capped at
`meta.citation_policy.nct_ids_per_datum` (20) ids, in `nct_id` order; `nct_ids_total` always gives the full
count, and the endpoint below returns the complete list.

Inline excerpts are re-verified against the raw records before the response is sent (in `full` mode that is
every excerpt). `meta.citation_policy` repeats the mode, so a reader knows why a bar of 120 trials shows 3
excerpts.

**Citation keys** (public projection):

| Key | Meaning |
|---|---|
| `nct_id` | Trial id. Absent for `derived_count` (the count comes from the search in `query_url`) |
| `excerpt` | The evidence string (see kinds) |
| `field_path` | Only when it differs from the datum's `field_path`. Dot path into the study JSON, with list indices, for example `protocolSection.contactsLocationsModule.locations[0].facility`. `$.totalCount` for `derived_count` |
| `kind` | **Omitted for `field_value`**. Otherwise `text_excerpt` or `derived_count`. Citations nested inside rendering hints (`meta.thresholds[].citations`, …) are trimmed the same way: no `url`, `kind` only when not a field value; they always carry their own `field_path`. |
| `char_start`, `char_end` | `text_excerpt` only: offsets in the field string, end exclusive |
| `label` | Optional human context, for example `"Nausea / Sema 2.4 mg: affected"` |

There is no per-citation `url`. Link a trial with `meta.citation_policy.study_url` (replace `{nct_id}`), and a
`derived_count` with the datum's `query_url`.

| Kind | `excerpt` is |
|---|---|
| `field_value` (default) | The verbatim JSON leaf at `field_path`: strings as-is, numbers and booleans as JSON text (`"1306"`, `"false"`), and lists of scalars as compact JSON (`"[\"PHASE3\"]"`). Long non-string leaves are cut to their first 300 characters |
| `text_excerpt` | A verbatim substring `value[char_start:char_end]` of the string at `field_path`, at most 300 characters (eligibility criteria, descriptions, duration phrases) |
| `derived_count` | The `totalCount` returned by the search in the datum's `query_url` (registry-count buckets) |

**How to verify one.** Fetch `https://clinicaltrials.gov/api/v2/studies/{nct_id}` (or `GET /v1/trials/{nct_id}`),
walk the effective `field_path` (citation's, else datum's; split on `.`; `name[i]` indexes a list), then:

- `field_value`: compare against the leaf, JSON-encoded for non-strings.
- `text_excerpt`: check `leaf[char_start:char_end] == excerpt`.
- `derived_count`: open `query_url` (or the equivalent API search with `countTotal=true`) and compare its total
  with `excerpt`.

The server re-checks `field_value` and `text_excerpt` citations this way before responding
(`ctviz/core/provenance.py::verify_citation`); `derived_count` excerpts are checked to be a count, not
re-queried. Citations that fail are removed and reported with a `DEGRADED` warning whose `affects` lists the
datums.

**All excerpts and all trials for one datum.** When `excerpts_truncated` is `true` or
`nct_ids_total > len(nct_ids)`:

```
GET /v1/responses/{request_id}/datums/{datum_id}/citations
→ {"request_id": "...", "datum_id": "d20", "total": 163, "verified": 163, "unverified": 0,
   "citations": [ {nct_id, excerpt, field_path, kind, url, label?, char_start?, char_end?, verified}, ... ],
   "nct_ids": ["NCT…", ...],   // complete, sorted by nct_id
   "nct_ids_total": 163}
```

The evidence is written in the background right after the response. Until that write lands (or if it is
lost), the server answers from memory; for a response-cache hit whose source record is gone too, it rebuilds
the answer from the cached response (complete `nct_ids`, inline excerpts only).
`total` counts excerpts; `nct_ids_total` counts trials (they differ when a trial has several excerpts or a
datum has none). `nct_ids` / `nct_ids_total` are absent for a datum without trial ids (registry counts).

This endpoint returns the internal citation objects, so every item has its own `field_path`, `kind` and `url`.
Before serving, each excerpt is re-checked against the stored raw record (`verified: false` when it no longer
matches or the record is not in the study store); nothing is dropped.
`404 not_found` is returned for an unknown or expired `request_id` (citation lists are kept for 24 hours) or
an unknown `datum_id`.
