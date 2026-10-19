# Response conventions (schema v2.2)

The full renderer contract, with every viz type, meta key, note code and citation kind, is in
**[response_schema.md](response_schema.md)**. This page is a quick summary that matches it. The Pydantic
models in `ctviz/api/schemas/` and the projection in `ctviz/api/public.py` are the source of truth, and
`GET /v1/schema` exports them.

## Datums
- `null`, `[]` and `{}` values are dropped. A missing key means "no value". For example, a timeline row
  without its end field is a point event or still ongoing, and a histogram bin without `bin_end` is open-ended.
  Two exceptions inside `data`: `citations: []` on a datum with no evidence, and `estimated_count: null`
  on trend rows ("not computed").
- Provenance keys come last in every row, node and edge:

| Key | Present | Meaning |
|---|---|---|
| `nct_ids` | record-backed datums | Trials behind the datum, sorted by `nct_id`: the first 20 (all with `citation_detail=full`; full list from `GET /v1/responses/{request_id}/datums/{datum_id}/citations`) |
| `nct_ids_total` | whenever `nct_ids` is | Number of distinct trials behind the datum. For `trial_count` (and network `size` / `weight`) from retrieved records, `nct_ids_total == value`; `len(nct_ids) < nct_ids_total` means the list was capped |
| `query_url` | registry counts | The search whose `totalCount` is the value (then no `nct_ids`) |
| `field_path` | datums with field-value excerpts | Path shared by the excerpts, written once |
| `citations` | all, except `citation_detail=ids_only` | `{nct_id, excerpt}`; `field_path` only when it differs from the datum's; `kind` only for `text_excerpt` (with `char_start`/`char_end`) or `derived_count` (no `nct_id`; its link is `query_url`); optional `label`. No `url`: use `meta.citation_policy.study_url` or `query_url` |
| `citations_total` | all, except `ids_only` | Excerpts available for the datum |
| `excerpts_truncated` | all, except `ids_only` | `true` when more excerpts exist than are inline; get them from `GET /v1/responses/{request_id}/datums/{id}/citations` |

- `citation_detail`: `ids_only` (no excerpts), `sample` (default: first 3 excerpts by `nct_id`, up to 10 for a
  single-trial datum), `full` (every excerpt inline, all re-verified). `meta.citation_policy` states the mode.
- Other reserved keys (never used as channel fields):

| Key | Where | Meaning |
|---|---|---|
| `id` | all | Unique within the visualization |
| `cite_note` | uncited datums | Why there are no citations (empty bucket, derived remainder) |
| `is_other` | bar / pie | The "Other" bucket for truncated categories |
| `is_gap` | heatmap | Zero-count cell. Style it with `meta.highlight_zero` |
| `is_open_ended` | histogram | Last bin has no `bin_end` |
| `estimated_count` | time_series | Trials in the bucket whose start date is still an estimate (part of `trial_count`); `null` = not computed |
| `source` | survival_curve | `reported` (cited value) or `interpolated` (draw dashed) |
| `stage` | sankey nodes | Column index into `meta.stages` |
| `section` | eligibility_checklist | `structured`, `inclusion`, `exclusion`, `unknown` |
| `status` | eligibility_checklist | `met`, `unmet`, `needs_review`, `unknown` |
| `evaluated_by` | eligibility_checklist | `rule`, `llm`, `none` |
| `parent_text` | eligibility_checklist | Parent line of a nested criterion |
| `kind` | kpi_cards | `number`, `text`, `date`, `enum`, `list` |
| `row_type` | geo_map (single trial) | `site`, `central_contact`, `record_mention` (only sites are drawn) |
| `pinned` | network nodes | An entity the user named |

## Channels
- `x` is always the category (or time) axis. `meta.layout.orientation = "horizontal"` only means draw it vertically.
- `format` is a d3-format string for numbers. `time_format` is a d3-time-format string for dates. Temporal channels never use `format`.
- `heatmap`: `x` and `y` are categorical. `series` (quantitative) is the cell value and color.
- `pie_chart`: `label` is the category and `y` is the value.
- `geo_map`: see `meta.geo`. At `level` country, rows use `region_field`/`value_field` (and `x`/`y` are set too).
  At site level, rows use `lat_field`/`lon_field`; rows without coordinates are listed but not drawn.
- `timeline`: there are two forms. Single axis: `x` start, `x2` end, `y` task, `series` category, plus `meta.time_axis`.
  Multi-lane: `y` is the row label, `series` is the lane, and each lane's fields and axis (calendar vs relative) are in `meta.lanes`.
- `forest_plot`: `y` is the label, `x` the estimate (log scale for ratios), and `error_low`/`error_high` the CI. `meta.reference_line` gives the null value.
- Graph types: field names come from `encoding.nodes` and `encoding.edges`. Edges reference node ids.
  Network edge `weight` and node `size` count distinct trials.

## Grouping and status
- `meta.grouping.each_trial_counted_once: true`: single-valued dimension (phase, status, status group, start
  year, lead sponsor); bars of one series add up to distinct trials (checked by `each_trial_in_one_bucket`).
  `multi_valued: true`: a trial can count in several buckets.
- Status is the trial's status **today** (ClinicalTrials.gov keeps no history). Titles with a status filter end
  in `(status today: …)`. `status_group` splits into Active / Done (completed, terminated, withdrawn) / Other.

## Meta catch-alls
- `meta.stats`: extra display-only figures (dropout rate per arm, minimum detectable effect, ...).
- `meta.debug` (only when the request sets `include_trace: true`; default `false`): trace, cache, API params and
  chart diagnostics. It is never needed to draw.

## Notes
`meta.notes[].code` is one of the `NoteCode` values (see response_schema.md §5). `affects` lists datum ids.
If it is absent, the note applies to the whole chart. Citations that fail re-verification against the raw
record are removed and reported with a `DEGRADED` warning. `SNAPSHOT` means the records came from a stored
cohort snapshot whose `totalCount` still matches ClinicalTrials.gov.
