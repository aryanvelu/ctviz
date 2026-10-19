// TypeScript mirror of ctviz/api/schemas (the Pydantic models are the source of truth;
// src/schema.json is generated from them and the sample test validates against it).
// Responses are serialized with exclude_none, so every optional field may be absent.

export type VizType =
  | "kpi_cards" | "table" | "bar_chart" | "histogram" | "time_series" | "scatter_plot"
  | "heatmap" | "pie_chart" | "network_graph" | "sankey" | "timeline" | "survival_curve"
  | "threshold_line" | "power_curve" | "forest_plot" | "geo_map" | "eligibility_checklist";

export const VIZ_TYPES: VizType[] = [
  "kpi_cards", "table", "bar_chart", "histogram", "time_series", "scatter_plot", "heatmap",
  "pie_chart", "network_graph", "sankey", "timeline", "survival_curve", "threshold_line",
  "power_curve", "forest_plot", "geo_map", "eligibility_checklist",
];

export type FieldType = "quantitative" | "nominal" | "ordinal" | "temporal";
export type CitationKind = "field_value" | "text_excerpt" | "derived_count";

/**
 * Schema 2.1: `kind` is omitted for exact field values; `field_path` defaults to the datum's; there is no
 * `url` (link a trial as citation_policy.study_url, a derived_count as datum.query_url). Older samples and
 * the "all citations" endpoint may still send `url` and `field_path` on every item.
 */
export interface Citation {
  /** Absent for kind=derived_count (the excerpt is a registry total, not one trial's field). */
  nct_id?: string;
  excerpt: string;
  field_path?: string;
  kind?: CitationKind;
  char_start?: number;
  char_end?: number;
  /** Legacy (schema 2.0) or from the all-citations endpoint. */
  url?: string;
  label?: string;
  /** All-citations endpoint only: false when the excerpt is no longer exactly supported by the stored record. */
  verified?: boolean;
}

export interface Sort { by?: string; order?: "asc" | "desc"; values?: string[] }
export interface Scale { type?: "linear" | "log" | "sqrt" | "time" | "band"; domain?: (number | string)[]; zero?: boolean }

export interface Channel {
  field: string;
  type: FieldType;
  title: string;
  unit?: string;
  /** d3-format for numbers */
  format?: string;
  /** d3-time-format for temporal fields */
  time_format?: string;
  sort?: Sort;
  scale?: Scale;
}

export interface NodeEncoding { id?: string; label?: string; type?: string; size?: string }
export interface EdgeEncoding { source?: string; target?: string; weight?: string; type?: string }

export interface Encoding {
  x?: Channel; x2?: Channel; y?: Channel; y2?: Channel; series?: Channel; size?: Channel;
  label?: Channel; error_low?: Channel; error_high?: Channel;
  nodes?: NodeEncoding; edges?: EdgeEncoding; tooltip?: string[];
}

/** One row / node / edge. Encoded fields appear as extra keys. */
export interface Datum {
  id: string;
  /** Every trial behind this datum (for counts built from retrieved records, length == the count). */
  nct_ids?: string[];
  /** Schema 2.2+: how many trials are behind the datum when `nct_ids` lists only the first few. */
  nct_ids_total?: number;
  /** registry_counts only: the ClinicalTrials.gov search whose totalCount is this value. */
  query_url?: string;
  /** Field path shared by the excerpts (a citation's own field_path wins). */
  field_path?: string;
  /** The first excerpts by nct_id; the rest come from GET .../datums/{id}/citations. */
  citations?: Citation[];
  citations_total?: number;
  excerpts_truncated?: boolean;
  cite_note?: string;
  /** Interpolated values (survival_curve): ids of the reported datums this value lies between. */
  derived_from?: string[];
  [key: string]: unknown;
}

/**
 * Reserved datum keys (docs/response_schema_conventions.md). These are the only data keys a
 * renderer may read without going through `encoding`; they are never channel fields.
 */
export const RESERVED_KEYS = [
  "id", "nct_ids", "nct_ids_total", "query_url", "field_path", "citations", "citations_total", "excerpts_truncated", "cite_note", // all types
  "estimated_count", // time_series of trial starts: how many have an estimated (planned) start date
  "is_gap", // heatmap
  "source", // survival_curve: reported | interpolated
  "derived_from", // interpolated points: the anchor datum ids
  "stage", // sankey nodes
  "section", "status", "evaluated_by", "parent_text", // eligibility_checklist
  "kind", // kpi_cards
] as const;

export interface GraphData { nodes: Datum[]; edges: Datum[] }

export interface Threshold { axis?: "x" | "y"; value: number; label: string; source?: "registry" | "convention" | "user"; citations?: Citation[] }
export interface ReferenceLine { value: number; label?: string }
export interface GeoSpec {
  level: "country" | "site";
  region_field?: string;
  value_field?: string;
  lat_field?: string;
  lon_field?: string;
}
export interface TimeAxis { kind: "calendar" | "relative"; unit?: "days" | "weeks" | "months" | "years"; origin?: string }
export interface TimelineLane { name: string; axis: TimeAxis; start_field: string; end_field?: string; approximated?: boolean }
export interface NodeType { type: string; label: string; count?: number }
export interface Band { axis?: "x" | "y"; low: number; high: number; label: string }
export interface Approximation { is_approximation?: boolean; method?: string; assumptions?: string[] }
export interface Truncation { top_n: number; other_bucket?: boolean; omitted_count?: number }
export interface Layout { orientation?: "vertical" | "horizontal"; mode?: "grouped" | "stacked" | "normalized"; donut?: boolean; columns?: number }
export interface TableColumn { field: string; title: string; type?: FieldType; format?: string }

export interface VizMeta {
  units?: Record<string, string>;
  sort?: Sort;
  time_granularity?: "year" | "quarter" | "month" | "day";
  grouping?: Record<string, unknown>;
  layout?: Layout;
  thresholds?: Threshold[];
  bands?: Band[];
  truncation?: Truncation;
  approximation?: Approximation;
  columns?: TableColumn[];
  summary?: Record<string, unknown>;
  node_types?: NodeType[];
  edge_semantics?: string;
  network_kind?: string;
  directed?: boolean;
  min_edge_weight?: number;
  pinned?: string[];
  network_layout?: "force" | "ego";
  stages?: string[];
  time_axis?: TimeAxis;
  lanes?: TimelineLane[];
  geo?: GeoSpec;
  reference_line?: ReferenceLine;
  highlight_zero?: boolean;
  stats?: Record<string, unknown>;
  diagnostics?: Record<string, unknown>;
}

/** What the renderer draws: the public visualization plus its hints gathered back from response meta. */
export interface Visualization {
  type: VizType;
  title: string;
  subtitle?: string;
  encoding: Encoding;
  data: Datum[] | GraphData;
  meta?: VizMeta;
}

export type NoteCode =
  | "NOT_IN_REGISTRY" | "APPROXIMATED" | "SAMPLED" | "NO_RESULTS_POSTED" | "LLM_EVALUATED"
  | "ZERO_RESULTS" | "TRIAL_REQUIRED" | "ENTITY_AMBIGUOUS" | "FILTERS_RELAXED" | "TRUNCATED"
  | "PARTIAL_DATES" | "VIZ_DOWNGRADED" | "VIZ_PREFERENCE_IGNORED" | "DEGRADED" | "SUMMARY_TRIMMED"
  | "CITATION_DROPPED" | "REGISTRY_COUNTS" | "VALIDATION_FAILED" | "TERM_SEARCHED_VERBATIM";

export interface Note { code: NoteCode; severity?: "info" | "warning"; message: string; affects?: string[] }

export interface ResolvedEntity { kind: string; input: string; resolved: string; method: string; search_terms?: string[] }

export interface CohortFilter {
  label: string; drugs?: string[]; conditions?: string[]; sponsors?: string[]; countries?: string[];
  phases?: string[]; statuses?: string[]; start_year?: number; end_year?: number; term?: string;
  interventional_only?: boolean;
}

export type CountingMethod = "all_records" | "registry_counts" | "sample" | "single_trials" | "none";

export interface Coverage {
  studies_matched_total: number;
  studies_analyzed: number;
  counting_method?: CountingMethod;
  studies_with_results?: number;
  sampled?: boolean;
  sampling_method?: string;
  queries?: string[];
}

/** How datums cite their sources (schema 2.1). */
export interface CitationPolicy {
  /** How datum.nct_ids is filled (e.g. complete, or the first N with nct_ids_total). */
  nct_ids?: string;
  /** Schema 2.2+: at most this many nct_ids are listed inline per datum. */
  nct_ids_per_datum?: number;
  nct_ids_order?: string;
  /** Where the complete list comes from (the all-citations endpoint). */
  all_nct_ids?: string;
  excerpts_per_datum?: number;
  excerpts_single_trial?: number;
  excerpt_order?: string;
  /** Study link template, e.g. "https://clinicaltrials.gov/study/{nct_id}". */
  study_url?: string;
  all_excerpts?: string;
}

export interface Alternative { type: VizType; reason: string }
export interface CacheInfo { response_hit?: boolean; ctgov_hits?: number; ctgov_misses?: number }
export interface StepTiming { name: string; ms: number; status?: string; detail?: Record<string, unknown> }
export interface LLMCallInfo {
  agent: string; model: string; input_tokens?: number; cached_tokens?: number; output_tokens?: number;
  ms?: number; cost_usd?: number; tool_calls?: number;
}
export interface Trace { total_ms: number; steps?: StepTiming[]; llm_calls?: LLMCallInfo[]; ctgov_calls?: number; cost_usd?: number }

/* ------------------------------------------------------------------------------------------------
 * Public JSON (schema 2.0): what POST /v1/visualize returns (ctviz/api/public.py::to_public).
 * Empty lists/objects are omitted by the server, so every list/object below may be absent.
 * ---------------------------------------------------------------------------------------------- */

export type ResponseStatus = "ok" | "needs_clarification" | "unanswerable";

export interface TrialOption { nct_id: string; title?: string; status?: string }

export interface Clarification {
  reason: "unrecognized_entity" | "no_entity" | "subjective_question" | "trial_required" | "ambiguous_entity" | string;
  message: string;
  unmatched_terms?: string[];
  suggestions?: string[];
  options?: TrialOption[];
}

export interface Check { name: string; passed: boolean; detail?: string }
export interface Validation { passed: boolean; checks?: Check[] }

export interface QueryInterpretation {
  intent: string;
  persona: "patient" | "researcher";
  scope: "single_trial" | "cohort" | "comparison";
  confidence: number;
  parser: "llm" | "rules" | "cache";
  cohorts?: CohortFilter[];
  resolved_entities?: { input: string; resolved: string; kind: string; method: string }[];
}

/** Only when the request set include_trace=true. */
export interface DebugInfo {
  api_params?: Record<string, unknown>;
  cache?: CacheInfo;
  trace?: Trace;
  chart_diagnostics?: Record<string, unknown>;
}

/**
 * Response metadata. Rendering hints (units, sort, layout, geo, node_types, network_layout, ...) are
 * flattened in here next to the filters; only those that apply to the chart are present.
 */
export interface PublicMeta extends Omit<VizMeta, "approximation" | "diagnostics"> {
  source?: string;
  filters?: Record<string, unknown>;
  query_interpretation?: QueryInterpretation;
  assumptions?: string[];
  notes?: Note[];
  /** Present only when the chart is an approximation. */
  approximation?: { method?: string; assumptions?: string[] };
  network_layout?: "force" | "ego";
  coverage?: Coverage;
  validation?: Validation;
  alternatives?: Alternative[];
  follow_ups?: string[];
  citation_policy?: CitationPolicy;
  request_id: string;
  created_at: string;
  schema_version?: string;
  ctgov_api_version?: string;
  data_timestamp?: string;
  debug?: DebugInfo;
}

/** The drawable part only: rendering hints live in the response meta. */
export interface PublicVisualization {
  type: VizType;
  title: string;
  subtitle?: string;
  encoding: Encoding;
  data: Datum[] | GraphData;
}

export interface VisualizeResponse {
  status?: ResponseStatus;
  visualization?: PublicVisualization;
  answer_summary?: string;
  clarification?: Clarification;
  meta: PublicMeta;
}

export interface PatientProfile {
  age?: number;
  sex?: "female" | "male" | "other";
  conditions?: string[];
  medications?: string[];
  notes?: string;
}

export interface VisualizeRequest {
  query: string;
  nct_id?: string;
  nct_ids?: string[];
  drug_name?: string;
  drug_names?: string[];
  condition?: string;
  conditions?: string[];
  trial_phase?: string[];
  status?: string[];
  sponsor?: string;
  country?: string;
  start_year?: number;
  end_year?: number;
  persona?: "patient" | "researcher" | "auto";
  patient_profile?: PatientProfile;
  preferred_visualization?: VizType;
  narrative?: boolean;
  include_trace?: boolean;
  refresh?: boolean;
}

export const PHASES = ["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"] as const;
export const STATUSES = [
  "RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION", "ACTIVE_NOT_RECRUITING",
  "SUSPENDED", "TERMINATED", "COMPLETED", "WITHDRAWN", "UNKNOWN",
] as const;

export function isGraphData(d: Visualization["data"]): d is GraphData {
  return !Array.isArray(d) && typeof d === "object" && d !== null && "nodes" in d;
}
