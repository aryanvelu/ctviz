// Bridges the public JSON (schema 2.0) and the renderer. The server flattens rendering hints into
// response.meta; the chart builders read them from one `viz.meta` object, so gather them back here.
import type { Datum, PublicMeta, VisualizeResponse, Visualization, VizMeta } from "./types";
import { isGraphData } from "./types";

/** Every rendering hint the server may flatten into response.meta (ctviz/api/schemas/viz.py::VizMeta). */
export const HINT_KEYS = [
  "units", "sort", "time_granularity", "grouping", "layout", "thresholds", "bands", "truncation",
  "columns", "summary", "node_types", "edge_semantics", "network_kind", "directed", "min_edge_weight", "pinned",
  "network_layout", "stages", "time_axis", "lanes", "geo", "reference_line", "highlight_zero", "stats",
] as const satisfies readonly (keyof VizMeta)[];

/** The rendering hints of a response, in the shape the chart builders read. */
export function hintsOf(meta: PublicMeta | undefined): VizMeta {
  const out: Record<string, unknown> = {};
  if (!meta) return out;
  for (const k of HINT_KEYS) if (meta[k] !== undefined) out[k] = meta[k];
  // The server sends `approximation` only when the chart is one (and drops the is_approximation flag).
  if (meta.approximation) out.approximation = { is_approximation: true, ...meta.approximation };
  if (meta.debug?.chart_diagnostics) out.diagnostics = meta.debug.chart_diagnostics;
  return out as VizMeta;
}

/** The visualization to draw, or null when the response has none (status != ok). */
export function toRenderViz(resp: VisualizeResponse): Visualization | null {
  const v = resp.visualization;
  if (!v) return null;
  return { ...v, meta: hintsOf(resp.meta) };
}

export function allDatums(viz: { data: Visualization["data"] }): Datum[] {
  return isGraphData(viz.data) ? [...viz.data.nodes, ...viz.data.edges] : viz.data;
}
