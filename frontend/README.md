# ctviz frontend

A small Vite + React + TypeScript demo that renders every `POST /v1/visualize` response with
ECharts. It renders only from the response contract (`ctviz/api/schemas/`), with no per-query code.

## Run (one server, no Node at runtime)

```bash
cd frontend && npm install && npm run build && cd ..     # writes frontend/dist
CTVIZ_OFFLINE=1 .venv/bin/ctviz serve --port 8000          # or without CTVIZ_OFFLINE for live data
open http://localhost:8000/                                # the viewer; the API stays under /v1
```

`ctviz serve` mounts `frontend/dist` at `/` whenever it exists (`ctviz/app.py`). Rebuild after
frontend changes; the API works the same with or without it.

## Develop (hot reload)

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173 ; /v1 is proxied to http://localhost:8000
```

Start the API in another terminal (`.venv/bin/ctviz serve --port 8000`, or with `CTVIZ_OFFLINE=1`).
Without a backend, use the dev-only **Load sample** dropdown (top right). It renders the bundled
public responses in `src/samples/` (`live_*` were captured from the running backend; the others are
hand-written, one per visualization type, converted with the backend's `to_public`). Production builds leave it out.

```bash
npm run build      # type-check + bundle to frontend/dist (FastAPI serves it at /)
npm run lint       # tsc --noEmit
npm test           # vitest: schema validation of samples + render smoke tests
```

Refresh `src/schema.json` whenever the Pydantic contract changes. Run this from the repo root:

```bash
PYTHONPATH=. .venv/bin/python -c "import json; from ctviz.api.schemas import VisualizeRequest; from ctviz.api.public import PublicResponse; print(json.dumps({'request': VisualizeRequest.model_json_schema(), 'response': PublicResponse.model_json_schema()}))" > frontend/src/schema.json
```

## What the page shows (public JSON, schema 2.0)

- `status: ok` → the chart, then `answer_summary`, `meta.notes` (warnings highlighted), `meta.follow_ups`
  as chips that re-ask, and `meta.alternatives` as "View as" chips (re-ask with `preferred_visualization`).
- `status: needs_clarification | unanswerable` → no chart; a card with `clarification.message`,
  `unmatched_terms`, `suggestions` (click to put one in the question box) and `options` (candidate
  trials; click to re-ask the same question with that `nct_id`).
- Clicking a datum opens the side panel: `datum.nct_ids` (each linked to its study page),
  `datum.query_url` (the registry search reproducing a registry count) and the citations.
- **Validation** panel: `meta.validation.checks` (green/red) and `meta.coverage` with the counting method in plain words.
- Tabs: **Public JSON** (the body exactly as received) and **Debug** (`meta.debug`: steps, LLM calls,
  cache, API parameters; only when "Include debug" is ticked, which sends `include_trace: true`).

## How rendering works

The server flattens rendering hints (`units`, `layout`, `geo`, `node_types`, `network_layout`, ...)
into the response `meta` and drops empty lists/objects. `src/adapt.ts::toRenderViz` gathers the hints
back into one `viz.meta` object for the chart builders, so the per-type code below reads `meta.*` as
before; every list is defaulted when absent.

- `src/viz/Renderer.tsx` holds one `<Renderer>`. It picks a component from `visualization.type`.
  Chart types are built as ECharts options by pure functions in `src/viz/options.ts`.
  `kpi_cards`, `table` and `eligibility_checklist` are plain React (`src/viz/Plain.tsx`).
- Field names come only from `visualization.encoding` and the typed `meta` hints (`geo`, `lanes`,
  `reference_line`, ...). The only data keys read directly are the documented reserved keys
  (`RESERVED_KEYS` in `src/types.ts`). A test scans the renderer source to enforce this.
- Numbers use `channel.format` (d3-format) and dates use `channel.time_format`. Units come from
  `channel.unit`, else `meta.units[field]`. Categories follow `sort.values`, then `sort.by`
  (ascending unless `order: "desc"`).
- A missing number is never treated as 0. That datum is left off the chart but still appears in the data table.
- A log scale is used only when every value is positive. Otherwise the chart uses a linear scale and says so.
- Every chart has a **Data table** underneath (nodes and edges for graph types). It is the
  keyboard and screen-reader way to reach each datum's sources.
- Clicking a datum opens the citation panel, and the panel takes focus. Escape closes it while it has
  focus. Only http(s) URLs become links. A `derived_count` citation links to its search URL, and shows
  as plain text if there is no URL. "+N more" loads `GET /v1/responses/{request_id}/datums/{datum_id}/citations`
  and disappears once loaded. A request that is still running when you switch datums is cancelled and its result ignored.
- `meta.approximation` → "Approximation" badge. `meta.truncation` → "Top N" badge. `meta.notes[]` →
  banners. `meta.thresholds[].citations` → a "sources" button next to the line's label.
  `meta.summary` / `meta.stats` / `meta.diagnostics` → key/value lists; only scalar entries are shown.
- If a renderer crashes, an error boundary shows the raw visualization JSON instead.

### Per-type conventions

| type | read from |
|---|---|
| bar_chart | `x` category, `y` value, `series`, `error_low/high`; `meta.layout.{orientation,mode}` (orientation only flips the drawing) |
| histogram | `x` bin start, `x2` bin end (missing → open-ended, drawn dashed), `y` count |
| time_series | `x` temporal (`time_format`), `y`, `series` |
| scatter_plot | `x`, `y`, `series`, `size`, `label` |
| heatmap | `x`, `y` categories; `series` cell value; reserved `is_gap` outlined when `meta.highlight_zero` |
| pie_chart | `x` category, `y` value; `meta.layout.donut` |
| network_graph | `encoding.nodes/edges` (node size from `nodes.size`, edge width from `edges.weight`); `meta.node_types[{type,label,count}]` legend, `edge_semantics`, `directed` (arrows), `pinned` (bold), `min_edge_weight`; force layout, or `network_layout: "ego"` → pinned node centered, others on rings (strongest links innermost) |
| sankey | `encoding.nodes/edges`; reserved node key `stage` (index into `meta.stages`) |
| timeline | `y` row label, `series` lane; `meta.lanes[{name, axis{kind,unit,origin}, start_field, end_field?, approximated}]`. Lanes are grouped by axis kind, each group gets its own x axis, and a missing end value draws a point. Without lanes: `x`/`x2` plus `meta.time_axis`. |
| survival_curve | `x` time, `y` survival, `series` arm, `error_low/high` CI band; reserved `source: "interpolated"` → dashed segment |
| threshold_line / power_curve | `x`, `y`, `series`; `meta.thresholds` → markLine, `meta.bands` → markArea |
| forest_plot | `y` label, `x` estimate, `error_low/high`, `size`; `x.scale.type = "log"`; `meta.reference_line{value,label}` |
| geo_map | `meta.geo`. Country level: sorted bar of `region_field` × `value_field` (else `encoding.x` × `encoding.y`). Site level: scatter of `lon_field`/`lat_field` named by `encoding.label`, point size from `encoding.size` (e.g. trial_count), axes fitted to the points, plus the table. |
| eligibility_checklist | `label` criterion text; reserved `section`, `status`, `evaluated_by`, `parent_text` |
| kpi_cards | `label` title, `y` value; reserved `kind` (number / text / date / enum / list) |
| table | `meta.columns`, else the encoded channels |

Colors follow `prefers-color-scheme` (light and dark tokens in `src/styles.css` and `src/theme.ts`).
The layout works down to phone width. There are no runtime CDNs, because all assets are bundled.
