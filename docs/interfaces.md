# Cross-module interfaces (as implemented)

The orchestrator calls these signatures. Contracts in `ctviz/api/schemas/` (schema v1.1),
`ctviz/core/provenance.py` and `ctviz/viz/base.py` are the source of truth; this page mirrors the code.

## Planner — `ctviz/agents/planner.py`

```python
async def plan_query(request: VisualizeRequest, *, settings: Settings, llm: LLM, db: Database | None = None
                     ) -> tuple[QueryPlan, list[ResolvedEntity]]
```
- Rule extraction first (`rule_plan_with_extraction`), then one LLM call (structured `LLMPlan`) validated and
  repaired by `PlanValidator`; at most one escalation (low confidence / schema failure) within the request budget.
- Falls back to the rule plan on `LLMUnavailable`/`LLMError`; `plan.parser` is `llm`, `rules` or `cache`.
- Exact-normalized plan cache (`db`), never for patient requests (`cacheable()`).
- `plan.clarification` and `plan.assumptions` are fixed templates (`TEMPLATES`), rendered as-is.
- Never raises for normal input; low confidence → clarification text but still a usable plan.

`ctviz/agents/rules_planner.py`: `rule_plan(request) -> QueryPlan` (pure, sync);
`rule_plan_with_extraction(request, resolver=None, today=None) -> tuple[QueryPlan, Extraction]`;
`extract(request, resolver=None, today=None) -> Extraction` (`Extraction.ambiguous: dict[surface, meanings]`).

`ctviz/core/entities.py`: `EntityResolver` with `resolve(text, kind, *, method=None) -> ResolvedEntity`,
`expand_drug_class(name) -> list[str]`, `is_ambiguous(text) -> list[str]`, `guess_kind(text) -> (kind, confident)`,
`aliases_of(kind, canonical) -> list[str]`; `get_resolver()` returns the shared instance.

`ctviz/agents/router.py`: `ModelRouter(settings).model_for(task, *, quality="default", complex_query=False) -> str`,
`escalation_model(current) -> str | None`, `should_escalate(*, confidence=None, schema_failures=0, ...) -> bool`.

`ctviz/core/budget.py`: `budget_scope(settings) -> ContextManager[RequestBudget]` (one per request, installed by
the orchestrator next to `trace_scope`); `current_budget()`; `RequestBudget.can_call_llm()`, `charge_tool()`
(raises `BudgetExceeded`; at most 6 tool calls), `timeout_for(default_s)`.

## Retrieval agent — `ctviz/agents/retrieval_agent.py`

```python
@dataclass
class RetrievalTools:
    probe_count: Callable[[CohortFilter], Awaitable[int]]
    search_studies: Callable[[CohortFilter, int], Awaitable[list[dict]]]   # [{nct_id, title}]
    get_study: Callable[[str], Awaitable[dict]]                             # raw record

async def resolve_retrieval(plan: QueryPlan, question: str, cohort: CohortFilter, *, llm: LLM | None,
                            settings: Settings, tools: RetrievalTools, ambiguous: dict[str, list[str]] | None = None,
                            max_steps: int = 3, tool_timeout_s: float = 5.0, tool_retries: int = 2,
                            backoff_s: float = 0.2) -> RetrievalDecision
```
`RetrievalDecision{cohort: CohortFilter | None, count, nct_ids, calls, rejected, notes: list[Note],
clarification: str | None, used_llm, rationale}`.
The orchestrator calls it only (a) when the question names an ambiguous entity the plan did not settle
(deterministic probes, no LLM) or (b) when every cohort is still empty after `relax_until_nonzero`
(≤ 1 LLM call, ≤ 3 validated tool calls). Tools are bound to `CTGovClient.count` / `search_page` / `get_study`.
`decision.notes` → `meta.notes`; `decision.clarification` → `meta.interpretation.clarification`;
`decision.cohort` replaces the planned cohort.

## Retrieval — `ctviz/ctgov/retrieval.py`

```python
def begin_seen(partition_budget: int = DEFAULT_PARTITION_BUDGET) -> dict[str, dict]   # per-request registry
def seen_studies() -> dict[str, dict]                                                 # nct_id -> raw (this request)
async def fetch_cohort(client, cf: CohortFilter, *, field_set: FieldSet, max_studies: int,
                       refresh: bool = False) -> CohortData
async def fetch_trials(client, nct_ids: list[str], *, refresh: bool = False) -> list[Study]
async def search_candidates(client, cf: CohortFilter, *, limit: int = 10, recruiting_first: bool = False,
                            refresh: bool = False) -> tuple[list[Study], int, SearchSpec]
async def facet_counts(client, spec: SearchSpec, buckets: dict[str, str], *, sample: int = 10
                       ) -> dict[str, tuple[int, list[Study], str]]   # bucket -> (count, sample studies, search url)
async def relax_until_nonzero(client, cf: CohortFilter) -> tuple[CohortFilter, list[str]]  # (relaxed, steps)
```
Handlers may call `facet_counts` / `fetch_trials` through `ctx.client` when they need more data.

## Orchestrator — `ctviz/core/orchestrator.py`

```python
class Orchestrator:
    def __init__(self, settings, db, client, llm, graph=None, *, text_index=None, planner=None,
                 runs_dir="default", use_response_cache=True)
    async def run(self, request: VisualizeRequest) -> VisualizeResponse
    async def run_recorded(self, request) -> tuple[VisualizeResponse, RunRecorder]
```
Order: response cache → `plan_query` → (ambiguity → retrieval agent) → fetch (cohorts per `IntentSpec.field_set`;
`fetch_trials` for `plan.nct_ids`; single-trial intent without an id → TRIAL_REQUIRED candidate table) →
zero results → relax ladder → retrieval agent → `get_handler(intent)(ctx)` (≤ 1 repair) → `apply_downgrades`
→ synthesis ‖ citation verification → `verify_response` → assemble → background writes (citations, response cache
unless private/degraded, `graph.ingest`, `text_index.index`). Patient/profile requests are never response-cached
and profile-derived strings are hashed in `runs/<id>/`.

## Synthesis + Verification — `ctviz/agents/synthesis.py`, `ctviz/agents/verification.py`

```python
async def synthesize(plan: QueryPlan, result: HandlerResult, *, llm=None, settings=None,
                     viz: Visualization | None = None) -> str | None     # deterministic template, no LLM
def template_summary(plan, result, viz=None) -> str | None
def verify_citations(viz: Visualization, raw_by_nct: dict[str, dict]) -> tuple[Visualization, list[Note]]
def verify_response(resp: VisualizeResponse, raw_by_nct: dict[str, dict], *, facts: dict | None = None,
                    extra_allowed: Iterable = (), check_citations: bool = True
                    ) -> tuple[VisualizeResponse, list[Note]]
```
Verification drops citations that fail `verify_citation` (note `CITATION_DROPPED`), turns a chart that fails the
encoding check into a valid table (`VIZ_DOWNGRADED`), and drops summary sentences whose numbers do not appear in
`visualization.data`, `facts` or `extra_allowed` (note `SUMMARY_TRIMMED`). The template states "Based on N
trials" only when N is a handler fact (`trials_analyzed_derived`, ...).

## Extraction — `ctviz/agents/extractors.py`

```python
async def extract_criteria(study, *, llm, settings, profile: PatientProfile | None, focus_terms: list[str],
                           mode: Literal["eligibility", "medications"] = "eligibility") -> list[CriterionVerdict]
async def extract_criteria_detailed(...same...) -> ExtractionOutcome   # verdicts + LLM audit
def span_refs(study, path, start, end, label=None) -> list[FieldRef]   # <=300-char refs covering a span
```
`CriterionVerdict{section, index, text, status: met|unmet|needs_review|unknown, reason, evaluated_by: rule|llm|none,
refs: list[FieldRef], relevance, subheading, chunk_id, matched_terms, parent_text}`. LLM quotes are located with
`provenance.find_normalized` and accepted only if they are verbatim substrings of the cited chunk.

## Analysis agent — `ctviz/agents/analysis_agent.py`

```python
async def select_measures(study: Study, question: str, purpose: Literal["survival", "results", "ae", "flow"],
                          *, llm: LLM | None, settings: Settings) -> AnalysisSelection
```
`AnalysisSelection{purpose, measure_indices: list[int], group_ids: list[str] (empty = all), method: llm|rules,
rationale (fixed text), rejected (fixed codes)}`. IDs only; handlers record it in `meta.diagnostics.analysis`.

## Knowledge graph + text index — `ctviz/store/graph.py`, `ctviz/store/fts.py`

```python
class GraphStore:
    def __init__(self, db: Database)                    # uses db.lock; tables are in db.SCHEMA
    async def ingest(self, studies) -> dict[str, int]   # {ingested, skipped, replaced}; idempotent
    async def neighbors(self, name, node_type, limit=50) -> dict
    async def cooccurrence(self, nct_ids, pair_type, limit=200) -> list[dict]
    async def resolve(self, name, node_type) -> dict | None   # node + match: exact|alias|fuzzy
    async def stats(self) -> dict; async def aliases(self, alias, kind) -> list[str]
```
(each has a `*_sync` twin). `neighbors` → `{center, match, nodes, edges}`; `match` is `exact|alias|fuzzy|None`
(fuzzy = closest match). Edges: `{source, target, pair_type | rel, weight, nct_ids (≤25), nct_ids_total,
evidence: [{nct_id, field_path, excerpt}]}`. `cooccurrence` edges add `source_label, target_label, source_type,
target_type`; `source` has the pair's first type (sponsor for `sponsor_drug`). `limit` is clamped to 1..200.

```python
class TextIndex:
    def __init__(self, db: Database)
    async def index(self, studies) -> int                              # rows inserted
    async def search(self, query, nct_ids=None, k=20, mode="all", sections=None) -> list[SearchHit]
```
`SearchHit(nct_id, field_path, char_start, char_end, snippet, chunk_id, section, score, snippet_start,
snippet_end, text)`; `raw_field[snippet_start:snippet_end] == snippet` (≤ 300 chars, usable as a citation).

## Background — `ctviz/core/background.py`

`spawn(coro, name="background", *, low_priority=False) -> Task | None` (low-priority work is dropped when the
queue is full); `await drain(timeout_s=10.0)`; `pending() -> int`.

## API routes — `ctviz/api/routes.py`

`POST /v1/visualize`, `GET /v1/schema`, `GET /v1/intents`, `GET /v1/trials/{nct_id}`,
`GET /v1/graph/neighbors?name=&node_type=&limit=` (`{center, match, nodes, edges}`; 422 on unknown node_type,
503 without a graph), `GET /v1/responses/{request_id}/datums/{datum_id}/citations`, `GET /v1/health`,
`GET /v1/metrics`. Responses serialized with `exclude_none=True`.
