"""Response schema for ``POST /v1/visualize``.

Top level stays small:

* ``status`` — ``ok`` (a chart answers the question), ``needs_clarification`` (a named entity could not be
  found, or the question needs a specific trial) or ``unanswerable`` (the registry cannot support the
  judgment asked for, e.g. "which drug is safest"). Only ``ok`` responses carry a ``visualization``;
  the others carry ``clarification`` and never silently widen the search.
* ``answer_summary`` — deterministic sentences built from computed values (never LLM-written).
* ``meta`` — how the question was interpreted, how counts were obtained, notes, and the invariant checks
  run on the final response (``meta.validation``). Debug-only keys (``trace``, ``cache``,
  ``filters_applied``) appear only when ``include_trace`` is set.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from ctviz.api.schemas.plan import CohortFilter, Intent, Persona, Scope
from ctviz.api.schemas.viz import Visualization, VizType

SCHEMA_VERSION = "2.2"
# Trial ids inlined per datum in public JSON (citation_detail sample / ids_only) and per edge in
# GET /v1/graph/neighbors; ``nct_ids_total`` always carries the full count.
MAX_NCT_IDS_PER_DATUM = 20


class NoteCode(StrEnum):
    NOT_IN_REGISTRY = "NOT_IN_REGISTRY"
    APPROXIMATED = "APPROXIMATED"
    SAMPLED = "SAMPLED"
    NO_RESULTS_POSTED = "NO_RESULTS_POSTED"
    LLM_EVALUATED = "LLM_EVALUATED"
    ZERO_RESULTS = "ZERO_RESULTS"
    TRIAL_REQUIRED = "TRIAL_REQUIRED"
    ENTITY_AMBIGUOUS = "ENTITY_AMBIGUOUS"
    FILTERS_RELAXED = "FILTERS_RELAXED"
    TRUNCATED = "TRUNCATED"
    PARTIAL_DATES = "PARTIAL_DATES"
    VIZ_DOWNGRADED = "VIZ_DOWNGRADED"
    VIZ_PREFERENCE_IGNORED = "VIZ_PREFERENCE_IGNORED"
    DEGRADED = "DEGRADED"
    SUMMARY_TRIMMED = "SUMMARY_TRIMMED"
    CITATION_DROPPED = "CITATION_DROPPED"
    REGISTRY_COUNTS = "REGISTRY_COUNTS"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    TERM_SEARCHED_VERBATIM = "TERM_SEARCHED_VERBATIM"
    SNAPSHOT = "SNAPSHOT"


class Note(BaseModel):
    code: NoteCode
    severity: Literal["info", "warning"] = "info"
    message: str
    affects: list[str] = Field(default_factory=lambda: ["*"], description="Datum ids, or ['*'] for all.")


class ResolvedEntity(BaseModel):
    kind: Literal["drug", "condition", "sponsor", "country", "nct_id"]
    input: str
    resolved: str
    method: Literal[
        "exact", "alias", "mesh", "structured_field", "passthrough", "spelling_correction", "knowledge_graph"
    ]
    search_terms: list[str] = Field(default_factory=list)


class Interpretation(BaseModel):
    intent: Intent
    intent_confidence: float
    persona: Persona
    scope: Scope
    nct_ids: list[str] = Field(default_factory=list)
    cohorts: list[CohortFilter] = Field(default_factory=list)
    resolved_entities: list[ResolvedEntity] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    parser: Literal["llm", "rules", "cache"]


CountingMethod = Literal["all_records", "registry_counts", "sample", "single_trials", "none"]


class Coverage(BaseModel):
    studies_matched_total: int = Field(description="totalCount reported by ClinicalTrials.gov.")
    studies_analyzed: int = Field(
        description="Studies whose records were used to compute the chart "
        "(with registry_counts: the total the counts reconcile to)."
    )
    counting_method: CountingMethod = Field(
        default="all_records",
        description="all_records: every matching record was retrieved and counted; registry_counts: each bucket "
        "is an exact ClinicalTrials.gov totalCount for a filtered search (datum.query_url); "
        "sample: only part of the matching trials was analyzed (see sampling_method); "
        "single_trials: specific NCT records; none: no data was needed.",
    )
    studies_with_results: int | None = None
    sampled: bool = False
    sampling_method: str | None = None
    queries: list[str] = Field(default_factory=list, description="ClinicalTrials.gov search URLs used.")


class Alternative(BaseModel):
    type: VizType
    reason: str


class CacheInfo(BaseModel):
    response_hit: bool = False
    ctgov_hits: int = 0
    ctgov_misses: int = 0


class StepTiming(BaseModel):
    name: str
    ms: float
    status: Literal["ok", "skipped", "timeout", "error", "degraded", "cached"] = "ok"
    detail: dict[str, Any] = Field(default_factory=dict)


class LLMCallInfo(BaseModel):
    agent: str
    model: str
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    ms: float = 0
    cost_usd: float = 0
    tool_calls: int = 0


class Clarification(BaseModel):
    """Why no chart was produced and what the user can do instead (status != ok)."""

    reason: Literal["unrecognized_entity", "no_entity", "subjective_question", "trial_required", "ambiguous_entity"]
    message: str
    unmatched_terms: list[str] = Field(
        default_factory=list, description="Terms from the question that matched nothing in ClinicalTrials.gov."
    )
    suggestions: list[str] = Field(default_factory=list, description="Questions this service can answer instead.")
    options: list[dict[str, str]] = Field(
        default_factory=list, description="Candidate trials ({nct_id, title, status}) when a trial must be chosen."
    )


class Check(BaseModel):
    name: str
    passed: bool
    detail: str | None = None


class Validation(BaseModel):
    """Invariants enforced in code on the final response (see ctviz/agents/validation.py)."""

    passed: bool
    checks: list[Check] = Field(default_factory=list)


class Trace(BaseModel):
    total_ms: float
    steps: list[StepTiming] = Field(default_factory=list)
    llm_calls: list[LLMCallInfo] = Field(default_factory=list)
    ctgov_calls: int = 0
    cost_usd: float = 0


class ResponseMeta(BaseModel):
    interpretation: Interpretation
    coverage: Coverage
    notes: list[Note] = Field(default_factory=list)
    alternatives: list[Alternative] = Field(default_factory=list)
    validation: Validation | None = None
    # debug-only (stripped from the public JSON unless include_trace)
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    cache: CacheInfo = Field(default_factory=CacheInfo)
    trace: Trace | None = None


class VisualizeResponse(BaseModel):
    request_id: str
    created_at: str
    schema_version: str = SCHEMA_VERSION
    source: Literal["clinicaltrials.gov"] = "clinicaltrials.gov"
    ctgov_api_version: str | None = None
    data_timestamp: str | None = Field(default=None, description="ClinicalTrials.gov data snapshot time.")
    status: Literal["ok", "needs_clarification", "unanswerable"] = "ok"
    answer_summary: str | None = Field(
        default=None, description="2–4 deterministic sentences; every number appears in visualization.data."
    )
    visualization: Visualization | None = Field(default=None, description="Present only when status is 'ok'.")
    clarification: Clarification | None = Field(default=None, description="Present when status is not 'ok'.")
    follow_ups: list[str] = Field(default_factory=list, description="Related questions the user may want next.")
    meta: ResponseMeta


class ErrorResponse(BaseModel):
    error: str
    message: str
    retry_after_s: float | None = None
