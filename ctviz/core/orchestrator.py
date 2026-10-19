"""Orchestrator: one bounded pipeline per request.

    response cache → plan → retrieve (cohorts | trials | candidates) → handler (≤1 repair) → downgrades
    → synthesis ‖ citation verification → summary verification → assemble → background writes

Guarantees: a hard deadline (``settings.request_deadline_s``) after which whatever is available is
returned with a ``DEGRADED`` note; at most one handler attempt plus one repair; no unbounded loops;
normal input never raises (unknown/failed handlers degrade to a cited table or KPI card).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import itertools
import logging
import re
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ctviz.handlers  # noqa: F401  (registers intent handlers)
from ctviz.agents import chart_planner
from ctviz.agents.rules_planner import detect_chart
from ctviz.agents.synthesis import synthesize
from ctviz.agents.validation import validate
from ctviz.agents.verification import verify_citations, verify_response
from ctviz.api.public import to_public
from ctviz.api.schemas.plan import CohortFilter, Intent, Persona, QueryPlan, Scope
from ctviz.api.schemas.request import VisualizeRequest
from ctviz.api.schemas.response import (
    CacheInfo,
    Clarification,
    Coverage,
    Interpretation,
    Note,
    NoteCode,
    ResolvedEntity,
    ResponseMeta,
    VisualizeResponse,
)
from ctviz.api.schemas.viz import Channel, Encoding, GraphData, TableColumn, Visualization, VizMeta, VizType
from ctviz.config import Settings
from ctviz.core import answerability
from ctviz.core.background import spawn
from ctviz.core.budget import await_within, budget_scope
from ctviz.core.entities import get_resolver
from ctviz.core.intents import IntentSpec, get_intent_spec
from ctviz.core.provenance import STUDY_URL, Dataset, Study, derived_count_ref, expand_citations
from ctviz.ctgov.fields import PHASE_LABELS, FieldSet
from ctviz.ctgov.requests import SearchSpec
from ctviz.ctgov.retrieval import (
    begin_seen,
    fetch_cohort,
    fetch_trials,
    relax_until_nonzero,
    search_candidates,
    seen_studies,
)
from ctviz.handlers import registry_counts
from ctviz.obs.runs import RunRecorder, query_digest, run_scope, step
from ctviz.obs.trace import RequestTrace, trace_scope
from ctviz.store import response_cache as rc
from ctviz.store.study_store import load_studies
from ctviz.viz.base import CohortData, HandlerContext, HandlerResult, get_handler, make_viz, note
from ctviz.viz.selector import alternatives_for, apply_downgrades, empty_kpi

log = logging.getLogger(__name__)

PlannerFn = Callable[..., Awaitable[tuple[QueryPlan, list[ResolvedEntity]]]]
_NCT_RE = re.compile(r"\bNCT\d{8}\b", re.I)
_TREND_RE = re.compile(r"over time|per year|each year|by year|trend|since \d{4}|annual", re.I)
MISSING_VALUE = "\u2014"  # placeholder for a field the registry record does not report
CANDIDATE_LIMIT = 10
FALLBACK_TABLE_ROWS = 50
FETCH_RESERVE_S = 1.5  # time kept back after retrieval for handler + verification
HANDLER_RESERVE_S = 0.5  # time kept back after the handler for verification + assembly
CACHE_READ_TIMEOUT_S = 2.0
RUN_GRACE_S = 2.0  # past request_deadline_s, the whole pipeline is abandoned and a degraded answer returned


def is_private(request: VisualizeRequest, plan: QueryPlan | None = None) -> bool:
    """Patient questions: never cached, and the question text is hashed in logs/artifacts."""
    return (
        request.patient_profile is not None
        or request.persona == "patient"
        or (plan is not None and plan.persona is Persona.PATIENT)
    )


def _request_artifact(request: VisualizeRequest, private: bool) -> dict[str, Any]:
    body = request.model_dump(mode="json", exclude_none=True)
    if private:
        body["query"] = query_digest(request.query)
    return body


def profile_terms(request: VisualizeRequest) -> frozenset[str]:
    """Patient-profile strings (conditions, medications, notes), casefolded; scrubbed from runs/ and logs."""
    prof = request.patient_profile
    if prof is None:
        return frozenset()
    vals = [*prof.conditions, *prof.medications, *([prof.notes] if prof.notes else [])]
    return frozenset(v.strip().casefold() for v in vals if v and v.strip())


async def _none() -> None:
    return None


def _trial_only(spec: IntentSpec) -> bool:
    return Scope.SINGLE_TRIAL in spec.scopes and not ({Scope.COHORT, Scope.COMPARISON} & set(spec.scopes))


# Intents whose answer exists only in posted results (no registered-protocol substitute worth showing).
RESULTS_ONLY_INTENTS = frozenset({Intent.SURVIVAL_SUMMARY, Intent.RESULTS_OUTCOMES, Intent.DROPOUT_FLOW})
_COHORT_KINDS = ("drug", "condition", "sponsor", "country")


def _entity_terms(e: ResolvedEntity) -> list[str]:
    """Search terms for a resolved entity: its name, or its members for a class ("PD-1 inhibitors")."""
    return [e.resolved] if e.resolved in e.search_terms or not e.search_terms else list(e.search_terms)


def cohort_instead(
    plan: QueryPlan,
    spec: IntentSpec,
    trials: list[Study],
    entities: list[ResolvedEntity],
    request: VisualizeRequest,
) -> tuple[QueryPlan, Note] | None:
    """A named trial without posted results, but the question names a drug/condition: answer for that cohort.

    Typical case: an NCT id left in a form while the question asks about "nivolumab in melanoma". Applies
    only to results-only intents, only when NONE of the named trials has posted results, and only when the
    resolved entities include a drug or condition. Returns the cohort plan (NCT ids removed) and the note that
    says why, or ``None`` to keep the single-trial answer (which states that no results are posted).
    """
    if plan.intent not in RESULTS_ONLY_INTENTS or Scope.COHORT not in spec.scopes or not trials:
        return None
    if any(t.has_results for t in trials):
        return None
    by_kind: dict[str, list[ResolvedEntity]] = {k: [] for k in _COHORT_KINDS}
    for e in entities:
        if e.kind in by_kind and all(e.resolved.casefold() != x.resolved.casefold() for x in by_kind[e.kind]):
            by_kind[e.kind].append(e)
    if not (by_kind["drug"] or by_kind["condition"]):
        return None

    def terms(kind: str) -> list[str]:
        return list(dict.fromkeys(t for e in by_kind[kind] for t in _entity_terms(e)))

    drugs = " / ".join(e.resolved for e in by_kind["drug"])
    conds = " / ".join(e.resolved for e in by_kind["condition"])
    label = f"{drugs} in {conds}" if drugs and conds else drugs or conds
    described = (
        f"trials of {drugs} in {conds}" if drugs and conds else (f"trials of {drugs}" if drugs else f"{conds} trials")
    )
    cohort = CohortFilter(
        label=label,
        drugs=terms("drug"),
        conditions=terms("condition"),
        sponsors=terms("sponsor"),
        countries=terms("country"),
        phases=request.trial_phase,
        statuses=request.status,
        start_year=request.start_year,
        end_year=request.end_year,
    )
    ids = ", ".join(t.nct_id for t in trials)
    has, it = ("has", "it") if len(trials) == 1 else ("have", "they")
    message = (
        f"{ids} {has} no results posted on ClinicalTrials.gov (hasResults: false), so {it} cannot answer this; "
        f"showing {described} that do."
    )
    new_plan = plan.model_copy(
        update={
            "nct_ids": [],
            "scope": Scope.COHORT,
            "cohorts": [cohort],
            "assumptions": [*plan.assumptions, f"Answered for {described} with posted results instead of {ids}."],
        }
    )
    return new_plan, note(NoteCode.NO_RESULTS_POSTED, message, "warning")


def _dedupe_notes(notes: list[Note]) -> list[Note]:
    seen: set[tuple[str, str]] = set()
    out = []
    for n in notes:
        k = (n.code.value, n.message)
        if k not in seen:
            seen.add(k)
            out.append(n)
    return out


# -- fallback planning (used when the planner module is unavailable or fails) ---------------------
def minimal_plan(request: VisualizeRequest) -> QueryPlan:
    """A deterministic plan straight from the structured fields (last-resort fallback)."""
    ncts = list(
        dict.fromkeys(
            [
                *([request.nct_id] if request.nct_id else []),
                *request.nct_ids,
                *(m.upper() for m in _NCT_RE.findall(request.query)),
            ]
        )
    )[:5]
    persona = (
        Persona.PATIENT
        if request.persona == "patient" or (request.persona == "auto" and request.patient_profile is not None)
        else Persona.RESEARCHER
    )
    if ncts:
        return QueryPlan(
            intent=Intent.SINGLE_TRIAL_OVERVIEW,
            intent_confidence=0.3,
            persona=persona,
            scope=Scope.COMPARISON if len(ncts) > 1 else Scope.SINGLE_TRIAL,
            nct_ids=ncts,
            assumptions=["Planner unavailable; showing a trial overview."],
            parser="rules",
        )
    common = dict(
        sponsors=[request.sponsor] if request.sponsor else [],
        countries=[request.country] if request.country else [],
        phases=request.trial_phase,
        statuses=request.status,
        start_year=request.start_year,
        end_year=request.end_year,
    )
    cohorts: list[CohortFilter] = []
    if len(request.drug_names) > 1:
        cohorts = [CohortFilter(label=d, drugs=[d], **common) for d in request.drug_names]
    elif len(request.conditions) > 1:
        cohorts = [CohortFilter(label=c, conditions=[c], **common) for c in request.conditions]
    else:
        drugs = request.drug_names or ([request.drug_name] if request.drug_name else [])
        conds = request.conditions or ([request.condition] if request.condition else [])
        if drugs or conds or request.sponsor:
            cohorts = [
                CohortFilter(
                    label=" / ".join(drugs + conds) or request.sponsor or "Trials",
                    drugs=drugs,
                    conditions=conds,
                    **common,
                )
            ]
    intent = Intent.TREND_OVER_TIME if _TREND_RE.search(request.query) else Intent.PHASE_DISTRIBUTION
    return QueryPlan(
        intent=intent,
        intent_confidence=0.3,
        persona=persona,
        scope=Scope.COMPARISON if len(cohorts) > 1 else Scope.COHORT,
        cohorts=cohorts,
        group_by=request.group_by,
        series_by=request.series_by,
        top_n=request.top_n,
        time_granularity=request.time_granularity,
        preferred_visualization=request.preferred_visualization,
        assumptions=["Planner unavailable; used structured fields only."],
        parser="rules",
        clarification=None if cohorts else "Name a drug, condition, sponsor or NCT id.",
    )


async def default_planner(
    request: VisualizeRequest, *, settings: Settings, llm: Any, db: Any
) -> tuple[QueryPlan, list[ResolvedEntity]]:
    """``ctviz.agents.planner.plan_query`` when present, else the rule planner, else :func:`minimal_plan`."""
    try:
        from ctviz.agents.planner import plan_query
    except ImportError:
        try:
            from ctviz.agents.rules_planner import rule_plan
        except ImportError:
            return minimal_plan(request), []
        return rule_plan(request), []
    return await plan_query(request, settings=settings, llm=llm, db=db)


# -- generic cited visualizations ------------------------------------------------------------------
_TABLE_FIELDS = [
    ("nct_id", "NCT ID", "nct_id"),
    ("title", "Title", "brief_title"),
    ("status", "Status", "status"),
    ("phase", "Phase", "phases"),
    ("sponsor", "Lead sponsor", "lead_sponsor"),
    ("start_date", "Start date", "start_date"),
    ("enrollment", "Enrollment", "enrollment_count"),
]


def trial_table(studies: list[Study], title: str, subtitle: str | None = None) -> tuple[Visualization, Dataset]:
    """A cited table, one row per trial (universal fallback and TRIAL_REQUIRED candidates)."""
    ds = Dataset(prefix="t")
    numeric_enrollment = True
    for s in studies:
        row: dict[str, Any] = {}
        refs = []
        missing: list[str] = []
        for fname, _, key in _TABLE_FIELDS:
            value = s.get(key)
            if key == "phases":
                value = ", ".join(PHASE_LABELS.get(p, p) for p in value) if value else None
            if value is None or value == "" or value == []:
                row[fname] = MISSING_VALUE  # every declared column is present in every row
                missing.append(fname)
            else:
                row[fname] = value
                refs.append(s.ref(key))
        if "enrollment" in missing:
            numeric_enrollment = False
        if missing:
            row["cite_note"] = f"not reported in the registry record: {', '.join(missing)}"
        row["nct_id"] = s.nct_id
        row["url"] = STUDY_URL.format(nct_id=s.nct_id)
        ds.emit(row, refs)
    cols = [
        TableColumn(
            field=f,
            title=t,
            type="quantitative" if f == "enrollment" and numeric_enrollment else "nominal",
            format=",d" if f == "enrollment" and numeric_enrollment else None,
        )
        for f, t, _ in _TABLE_FIELDS
    ]
    viz = make_viz(
        VizType.TABLE,
        title,
        Encoding(label=Channel(field="nct_id", type="nominal", title="Trial")),
        ds.rows,
        VizMeta(columns=cols),
        subtitle,
    )
    return viz, ds


def count_cards(cohorts: list[CohortData], trials: list[Study], title: str) -> tuple[Visualization, Dataset]:
    """KPI cards with server-side match counts (used when an intent has no handler)."""
    ds = Dataset(prefix="k")
    for c in cohorts:
        ds.emit(
            {"label": f"{c.label}: matching trials", "value": c.total_count, "kind": "count"},
            [derived_count_ref(c.total_count, c.search_url, c.label)],
        )
    for s in trials:
        ds.emit(
            {"label": s.get("brief_title") or s.nct_id, "value": s.nct_id, "kind": "text"},
            [s.ref("nct_id"), s.ref("brief_title")],
        )
    if not ds.rows:
        return empty_kpi(title, "No matching data found in ClinicalTrials.gov"), ds
    viz = make_viz(
        VizType.KPI_CARDS,
        title,
        Encoding(
            label=Channel(field="label", type="nominal", title="Item"),
            y=Channel(field="value", type="nominal", title="Value"),
        ),
        ds.rows,
    )
    return viz, ds


class NoChart(Exception):  # noqa: N818  (control flow, not an error)
    """Stop the pipeline and answer with a clarification instead of a chart (never widen the search)."""

    def __init__(self, status: str, clarification: Clarification):
        super().__init__(clarification.reason)
        self.status = status
        self.clarification = clarification


_FIELD_OF_KIND = {"drug": "drugs", "condition": "conditions", "sponsor": "sponsors", "country": "countries"}


def counting_method(cohorts: list[CohortData], trials: list[Study]) -> str:
    if any(c.counting == "registry_counts" for c in cohorts):
        return "registry_counts"
    if any(c.sampled for c in cohorts):
        return "sample"
    if cohorts:
        return "all_records"
    return "single_trials" if trials else "none"


def _follow_ups(plan: QueryPlan, result: HandlerResult) -> list[str]:
    """Fixed-template follow-up questions the service can answer (shown under the chart)."""
    out: list[str] = []
    label = plan.cohorts[0].label if plan.cohorts and plan.cohorts[0].label != "All trials" else None
    top = result.facts.get("top_country")
    if plan.intent is Intent.GEOGRAPHIC_DISTRIBUTION and top and label and not result.facts.get("site_level"):
        out.append(f"Show the trial sites (facility and city) in {top} for {label} trials")
    if plan.intent is Intent.TREND_OVER_TIME and label and not plan.series_by:
        out.append(f"How many {label} trials started each year, split by phase?")
    if plan.intent is Intent.ENTITY_NETWORK and label and plan.network_kind != "sponsor_drug":
        out.append(f"Which sponsors are testing which drugs in {label} trials?")
    return out


def _kpi_result(title: str, message: str, notes: list[Note]) -> HandlerResult:
    return HandlerResult(visualization=empty_kpi(title, message), notes=notes, studies_analyzed=0)


def _compact_rows(viz: Visualization, limit: int = 5) -> list[dict[str, Any]]:
    rows = viz.data.nodes if isinstance(viz.data, GraphData) else viz.data
    return [dict(r.model_extra or {}) for r in rows[:limit]]


# -- per-request state -----------------------------------------------------------------------------
@dataclass
class _Run:
    request: VisualizeRequest
    plan: QueryPlan
    entities: list[ResolvedEntity]
    ctx: HandlerContext
    notes: list[Note] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    matched_total: int = 0
    degraded: bool = False

    def degrade(self, message: str) -> None:
        self.degraded = True
        self.notes.append(note(NoteCode.DEGRADED, message, "warning"))


class Orchestrator:
    """Runs one :class:`VisualizeRequest` end to end. Create once per app and share."""

    def __init__(
        self,
        settings: Settings,
        db: Any,
        client: Any,
        llm: Any,
        graph: Any = None,
        *,
        text_index: Any = None,
        planner: PlannerFn | None = None,
        runs_dir: Path | None | str = "default",
        use_response_cache: bool = True,
    ):
        self.settings = settings
        self.db = db
        self.client = client
        self.llm = llm
        self.graph = graph  # ctviz.store.graph.GraphStore: grows from every fetch (background ingest)
        self.text_index = text_index  # ctviz.store.fts.TextIndex: FTS over every fetched record
        self.planner: PlannerFn = planner or default_planner
        if runs_dir == "default":
            self.runs_dir: Path | None = settings.runs_dir if settings.runs_active else None
        else:
            self.runs_dir = Path(runs_dir) if runs_dir else None
        self.use_response_cache = use_response_cache
        self.pending_evidence = rc.PendingEvidence()  # evidence whose background write has not landed yet
        self.chart_planner = settings.chart_planner_enabled

    # -- public -----------------------------------------------------------------------------------
    async def run(self, request: VisualizeRequest) -> VisualizeResponse:
        """Answer one request. Never raises for normal input."""
        resp, _ = await self.run_recorded(request)
        return resp

    async def run_recorded(self, request: VisualizeRequest) -> tuple[VisualizeResponse, RunRecorder]:
        """Like :meth:`run`, also returning the run record (steps, ClinicalTrials.gov calls, artifacts)."""
        request_id = uuid.uuid4().hex
        rec = RunRecorder(
            request_id, self.runs_dir / request_id if self.runs_dir else None, private_terms=profile_terms(request)
        )
        with trace_scope() as tr, budget_scope(self.settings), run_scope(rec):
            begin_seen(self.settings.max_ctgov_calls_per_request)
            rec.set("request", _request_artifact(request, is_private(request)))
            try:
                # Hard cap: whatever an inner step does, the caller gets an answer shortly after the deadline.
                resp = await await_within(
                    self._run(request, request_id, tr, rec), self.settings.request_deadline_s + RUN_GRACE_S
                )
            except TimeoutError:
                log.warning("request exceeded the hard deadline; returning a degraded response")
                resp = self._emergency(
                    request,
                    request_id,
                    tr,
                    "The request deadline was reached before an answer was ready; retry shortly.",
                )
            except Exception as exc:  # last resort: still return a valid, honest response
                log.exception("orchestrator failure (%s)", type(exc).__name__)
                resp = self._emergency(request, request_id, tr)
            rec.set("response", resp.model_dump(mode="json", exclude_none=True))
            # what a client receives (runs/<id>/public.json)
            rec.set("public", to_public(resp, citation_detail=request.citation_detail))
            rec.capture_calls()
            if rec.directory is not None:
                spawn(asyncio.to_thread(rec.flush), name="run-artifacts")
            return resp, rec

    # -- pipeline ---------------------------------------------------------------------------------
    async def _run(
        self, request: VisualizeRequest, request_id: str, tr: RequestTrace, rec: RunRecorder
    ) -> VisualizeResponse:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.request_deadline_s

        def remaining() -> float:
            return max(0.0, deadline - loop.time())

        version_task = asyncio.create_task(self._version())
        key = rc.response_key(request)
        if self.use_response_cache and not request.refresh:
            with step("response_cache", key=key[:16]) as info:
                try:
                    cached = await asyncio.wait_for(
                        rc.get_response(self.db, key), timeout=max(0.05, min(CACHE_READ_TIMEOUT_S, remaining()))
                    )
                except TimeoutError:  # a slow cache is a miss, not a failure
                    cached = None
                    info["timeout"] = True
                info["summary"] = {"hit": cached is not None}
                if cached is not None:
                    info["status"] = "cached"
            if cached is not None:
                version_task.cancel()
                return self._from_cache(request, request_id, cached, tr)

        with step("plan", query=query_digest(request.query), persona=request.persona) as info:
            plan, entities = await self._plan(request, remaining)
            info["summary"] = {
                "intent": plan.intent.value,
                "scope": plan.scope.value,
                "parser": plan.parser,
                "confidence": plan.intent_confidence,
                "cohorts": [c.label for c in plan.cohorts],
                "nct_ids": plan.nct_ids,
            }
        private = is_private(request, plan)
        rec.set(
            "plan",
            {"plan": plan.model_dump(mode="json"), "resolved_entities": [e.model_dump(mode="json") for e in entities]},
        )
        if private:
            rec.set("request", _request_artifact(request, True))
        spec = get_intent_spec(plan.intent)
        ctx = HandlerContext(
            request=request,
            plan=plan,
            settings=self.settings,
            client=self.client,
            db=self.db,
            llm=self.llm,
            graph=self.graph,
        )
        st = _Run(request=request, plan=plan, entities=entities, ctx=ctx)

        try:
            if plan.answerability == "subjective_judgment":
                with step("answerability", verdict=plan.answerability) as info:
                    info["summary"] = {"status": "unanswerable"}
                raise NoChart("unanswerable", answerability.refusal(plan))
            await self._verify_entities(st, remaining)
            result = await self._execute(st, spec, remaining)
        except NoChart as stop:
            version_task.cancel()
            return self._no_chart(request_id, st, stop, tr)
        st.notes.extend(result.notes)
        rec.set(
            "handler_result",
            {
                "visualization": result.visualization.model_dump(mode="json", exclude_none=True),
                "facts": result.facts,
                "notes": [n.model_dump(mode="json") for n in result.notes],
                "studies_analyzed": result.studies_analyzed,
                "studies_with_results": result.studies_with_results,
            },
        )

        with step("downgrades", type=result.visualization.type.value) as info:
            viz, dnotes = apply_downgrades(result.visualization)
            st.notes.extend(dnotes)
            info["summary"] = {"type": viz.type.value, "notes": [n.code.value for n in dnotes]}
        pref = st.plan.preferred_visualization or request.preferred_visualization
        if pref is not None and pref != viz.type and pref not in spec.compatible_viz:
            st.notes.append(
                note(
                    NoteCode.VIZ_PREFERENCE_IGNORED,
                    f"'{pref.value}' does not fit {st.plan.intent.value}; showing {viz.type.value}.",
                )
            )
        alternatives = result.alternatives or alternatives_for(st.plan.intent, viz.type)
        if pref is not None and pref != viz.type and pref in spec.compatible_viz:
            alternatives = sorted(alternatives, key=lambda a: a.type != pref)

        if request.citation_detail == "full":
            data = viz.data
            expand_citations(data.nodes + data.edges if isinstance(data, GraphData) else data, result.full_refs)
        raw_by_nct = await self._raw_records(viz, ctx, remaining)
        counting = counting_method(ctx.cohorts, ctx.trials)
        with step("synthesis_and_verify_citations", narrative=request.narrative) as info:
            synth = (
                synthesize(st.plan, result, llm=self.llm, settings=self.settings, viz=viz, counting=counting)
                if request.narrative
                else None
            )
            summary, verified = await asyncio.gather(
                synth if synth is not None else _none(), self._verify_citations_bounded(viz, raw_by_nct, remaining)
            )
            vnotes: list[Note] = []
            if verified is None:
                st.degrade("Citation re-verification did not finish within the time budget.")
            else:
                viz, vnotes = verified
                st.notes.extend(vnotes)
            info["summary"] = {
                "summary_chars": len(summary or ""),
                "citation_notes": len(vnotes),
                "citation_check": "skipped" if verified is None else "done",
            }

        version = await self._await_version(version_task, remaining)
        studies_analyzed = result.studies_analyzed if result.studies_analyzed is not None else len(ctx.studies)
        coverage = Coverage(
            studies_matched_total=st.matched_total,
            studies_analyzed=studies_analyzed,
            counting_method=counting,  # type: ignore[arg-type]
            studies_with_results=result.studies_with_results,
            sampled=counting == "sample",
            sampling_method="; ".join(f"{c.label}: {c.sampling_method}" for c in ctx.cohorts if c.sampling_method)
            or None,
            queries=list(dict.fromkeys(st.queries)),
        )
        if coverage.sampled and not any(n.code is NoteCode.SAMPLED for n in st.notes):
            st.notes.append(
                note(
                    NoteCode.SAMPLED,
                    "Not every matching trial was retrieved; the chart uses "
                    f"{coverage.studies_analyzed} of {coverage.studies_matched_total}. "
                    "See meta.coverage.sampling_method.",
                    "warning",
                )
            )
        if remaining() <= 0 and not st.degraded:
            st.degrade("Request deadline reached; the response may be incomplete.")
        resp = VisualizeResponse(
            request_id=request_id,
            created_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            ctgov_api_version=version.get("apiVersion"),
            data_timestamp=version.get("dataTimestamp"),
            visualization=viz,
            answer_summary=summary,
            follow_ups=_follow_ups(st.plan, result),
            meta=ResponseMeta(
                filters_applied=st.filters,
                interpretation=self._interpretation(st.plan, entities),
                coverage=coverage,
                notes=_dedupe_notes(st.notes),
                alternatives=alternatives,
                cache=CacheInfo(response_hit=False, ctgov_hits=tr.ctgov_hits, ctgov_misses=tr.ctgov_misses),
            ),
        )
        with step("verify_response") as info:
            resp, snotes = verify_response(
                resp, raw_by_nct, facts=result.facts, extra_allowed=[request.query], check_citations=False
            )
            info["summary"] = {"notes": [n.code.value for n in snotes]}
        with step("validate") as info:
            validation, vnotes = validate(
                resp, cohorts=ctx.cohorts, facts=result.facts, retrieved=set(raw_by_nct), extra_allowed=[request.query]
            )
            resp = resp.model_copy(
                update={
                    "meta": resp.meta.model_copy(
                        update={"validation": validation, "notes": [*resp.meta.notes, *vnotes]}
                    )
                }
            )
            info["summary"] = {
                "passed": validation.passed,
                "failed": [c.name for c in validation.checks if not c.passed],
            }
            if not validation.passed:
                info["status"] = "degraded"
        if request.include_trace and self.settings.include_trace:
            resp = resp.model_copy(update={"meta": resp.meta.model_copy(update={"trace": tr.to_model()})})
        self._background(
            request_id, key, resp, result, ctx, cacheable=not st.degraded and not private and validation.passed
        )
        return resp

    # -- no-chart answers (refusal / clarification) ------------------------------------------------
    def _no_chart(self, request_id: str, st: _Run, stop: NoChart, tr: RequestTrace) -> VisualizeResponse:
        """status=unanswerable|needs_clarification: no chart, no silent widening, nothing cached."""
        rec_step = {"status": stop.status, "reason": stop.clarification.reason}
        with step("no_chart", **rec_step) as info:
            info["summary"] = rec_step
        return VisualizeResponse(
            request_id=request_id,
            created_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            status=stop.status,  # type: ignore[arg-type]
            clarification=stop.clarification,
            meta=ResponseMeta(
                filters_applied=st.filters,
                interpretation=self._interpretation(st.plan, st.entities),
                coverage=Coverage(
                    studies_matched_total=st.matched_total,
                    studies_analyzed=0,
                    counting_method="none",
                    queries=list(dict.fromkeys(st.queries)),
                ),
                notes=_dedupe_notes(st.notes),
                cache=CacheInfo(ctgov_hits=tr.ctgov_hits, ctgov_misses=tr.ctgov_misses),
                trace=tr.to_model() if st.request.include_trace and self.settings.include_trace else None,
            ),
        )

    async def _verify_entities(self, st: _Run, remaining: Callable[[], float]) -> None:
        """Confirm every name the vocabulary could not ground with a ClinicalTrials.gov count probe.

        * LLM-named terms that grounding dropped (``plan.unverified_terms``) are added back to their cohort
          when the registry has trials for them (searched as written, ``TERM_SEARCHED_VERBATIM`` note);
        * pass-through names already in a cohort must match at least one trial on their own.
        Any name with zero matches stops the request with ``needs_clarification``: the search is never
        widened to "all trials" and never silently drops what the user asked about.
        """
        plan = st.plan
        if plan.nct_ids:
            return
        # names outside the vocabulary, matched by value: the rules may tag a typo as a condition while the
        # cohort searches it as a drug, so the probe uses the field the cohort actually searches
        unknown = {e.resolved.casefold() for e in st.entities if e.method == "passthrough"}
        in_cohorts = {
            (k, v) for c in plan.cohorts for k, f in _FIELD_OF_KIND.items() if k != "country" for v in getattr(c, f)
        }
        terms: list[tuple[str, str, int | None]] = [(u.kind, u.text, u.cohort_index) for u in plan.unverified_terms]
        terms += [(k, v, None) for k, v in sorted(in_cohorts) if v.casefold() in unknown]
        if not terms:
            return
        with step("verify_entities", terms=[t for _, t, _ in terms]) as info:
            probes = [
                self.client.count(
                    SearchSpec.from_cohort(CohortFilter(label=t, **{_FIELD_OF_KIND[k]: [t]})),
                    refresh=st.request.refresh,
                )
                for k, t, _ in terms
            ]
            counts = await self._bounded(asyncio.gather(*probes), remaining() - FETCH_RESERVE_S, st, "entity check")
            info["summary"] = {t: n for (_, t, _), n in zip(terms, counts or [None] * len(terms), strict=True)}
        if counts is None:  # could not check: refuse to guess rather than widen
            raise NoChart(
                "needs_clarification",
                answerability.unmatched([t for _, t, i in terms if i is not None] or [t for _, t, _ in terms]),
            )
        unmatched = [(k, t) for (k, t, _), n in zip(terms, counts, strict=True) if n == 0]
        if unmatched:
            resolver = get_resolver()
            close = {t: resolver.close_matches(t, k) for k, t in unmatched}  # type: ignore[arg-type]
            for k, t in unmatched:  # the knowledge graph knows every name seen in retrieved trials
                close[t] = list(dict.fromkeys(close[t] + await self._kg_close(t, k)))[:2]
            raise NoChart("needs_clarification", answerability.unmatched([t for _, t in unmatched], close))
        cohorts = [c.model_copy(deep=True) for c in plan.cohorts] or [CohortFilter(label="All trials")]
        added: list[str] = []
        for (kind, text, idx), n in zip(terms, counts, strict=True):
            if idx is None:
                continue
            c = cohorts[min(idx, len(cohorts) - 1)]
            getattr(c, _FIELD_OF_KIND[kind]).append(text)
            if c.label == "All trials":
                c.label = text
            added.append(text)
            st.entities.append(
                ResolvedEntity(
                    kind=kind,
                    input=text,
                    resolved=text,
                    method="passthrough",  # type: ignore[arg-type]
                    search_terms=[text],
                )
            )
            st.notes.append(
                note(
                    NoteCode.TERM_SEARCHED_VERBATIM,
                    f"'{text}' is not in the built-in vocabulary; ClinicalTrials.gov was searched for it "
                    f"as written ({n:,} matching trials).",
                )
            )
        added += await self._kg_other_names(st, cohorts, [(k, t) for k, t, _ in terms])
        if added:
            assumptions = [a for a in st.plan.assumptions if "all registered trials" not in a]
            st.plan = st.plan.model_copy(update={"cohorts": cohorts, "assumptions": assumptions})
            st.ctx.plan = st.plan

    async def _kg_close(self, text: str, kind: str) -> list[str]:
        if self.graph is None or kind not in ("drug", "condition", "sponsor"):
            return []
        try:
            return await asyncio.wait_for(self.graph.close_names(text, kind), timeout=1.0)
        except Exception:  # noqa: BLE001 - suggestions are optional
            return []

    async def _kg_other_names(self, st: _Run, cohorts: list[CohortFilter], terms: list[tuple[str, str]]) -> list[str]:
        """A name outside the vocabulary that trial records list as another name of one known entity (e.g.
        the code "MK-3475" in interventions[].otherNames) is also searched under that entity's name."""
        if self.graph is None:
            return []
        added: list[str] = []
        for kind, text in terms:
            if kind not in ("drug", "condition"):
                continue
            try:
                canon = await asyncio.wait_for(self.graph.aliases(text, kind), timeout=1.0)
            except Exception:  # noqa: BLE001 - the alias lookup only widens recall when it works
                continue
            # The alias table maps an alias to the raw intervention names it was seen with ("Pembrolizumab
            # (MK-3475)", "pembrolizumab (KEYTRUDA)", ...): keep the vocabulary name a clear majority resolves to.
            resolver = get_resolver()
            votes = Counter(
                next(iter({m.entity.resolved for m in resolver.scan(c) if m.kind == kind}), None) for c in canon
            )
            votes.pop(None, None)
            top = votes.most_common(1)
            if not top or top[0][1] < 0.6 * sum(votes.values()) or top[0][0].casefold() == text.casefold():
                continue
            canon = [top[0][0]]
            field = _FIELD_OF_KIND[kind]
            for c in cohorts:
                values = getattr(c, field)
                if any(v.casefold() == text.casefold() for v in values) and canon[0] not in values:
                    values.append(canon[0])
                    added.append(canon[0])
            st.entities.append(
                ResolvedEntity(
                    kind=kind,
                    input=text,
                    resolved=canon[0],  # type: ignore[arg-type]
                    method="knowledge_graph",
                    search_terms=[text, canon[0]],
                )
            )
            st.plan = st.plan.model_copy(
                update={
                    "assumptions": [
                        *st.plan.assumptions,
                        (
                            f"Read '{text}' as {kind} '{canon[0]}': trial records list it as another name; both were "
                            "searched."
                        ),
                    ]
                }
            )
        return added

    async def _execute(self, st: _Run, spec: IntentSpec, remaining: Callable[[], float]) -> HandlerResult:
        """Retrieve data for the plan and run the handler (bounded: one attempt + at most one repair)."""
        plan, req, ctx = st.plan, st.request, st.ctx
        if plan.nct_ids:
            with step("fetch_trials", nct_ids=plan.nct_ids) as info:
                trials = await self._bounded(
                    fetch_trials(self.client, plan.nct_ids, refresh=req.refresh),
                    remaining() - FETCH_RESERVE_S,
                    st,
                    "trial fetch",
                )
                info["summary"] = {"found": [t.nct_id for t in trials or []]}
            st.filters = {"nct_ids": plan.nct_ids}
            if trials is None:  # fetch failed / timed out (CTGovUnavailable...); _bounded already noted DEGRADED
                return _kpi_result(
                    "ClinicalTrials.gov unavailable",
                    "ClinicalTrials.gov could not be reached; the trial record was not retrieved",
                    [
                        note(
                            NoteCode.DEGRADED,
                            f"ClinicalTrials.gov was unavailable or too slow, so "
                            f"{', '.join(plan.nct_ids)} could not be retrieved. This does not mean the "
                            "trial does not exist; try again shortly.",
                            "warning",
                        )
                    ],
                )
            if not trials:
                return _kpi_result(
                    "Trial not found",
                    "No matching trial record in ClinicalTrials.gov",
                    [note(NoteCode.ZERO_RESULTS, f"Could not retrieve {', '.join(plan.nct_ids)}.", "warning")],
                )
            swap = cohort_instead(plan, spec, trials, st.entities, req)
            if swap is not None:  # named trial(s) without results, question names a drug/condition
                with step("cohort_instead", nct_ids=plan.nct_ids) as info:
                    st.plan, no_results = swap
                    ctx.plan = st.plan
                    st.notes.append(no_results)
                    st.filters = {}
                    info["summary"] = {"cohort": st.plan.cohorts[0].label}
                await self._verify_entities(st, remaining)  # the cohort's names get the usual registry check
                return await self._execute(st, get_intent_spec(st.plan.intent), remaining)
            st.matched_total = len(trials)
            st.queries += [STUDY_URL.format(nct_id=t.nct_id) for t in trials]
            if Scope.SINGLE_TRIAL in spec.scopes or spec.needs.full_records:
                ctx.trials = trials
            else:
                ids = tuple(t.nct_id for t in trials)
                ctx.cohorts = [
                    CohortData(
                        label="Selected trials",
                        spec=SearchSpec(ids=ids),
                        studies=trials,
                        total_count=len(trials),
                        search_url=STUDY_URL.format(nct_id=ids[0]),
                    )
                ]
        elif _trial_only(spec):
            return await self._trial_required(st, remaining)
        else:
            await self._disambiguate(st, remaining)
            plan = st.plan
            if not plan.cohorts:
                raise NoChart("needs_clarification", answerability.no_entity())
            ok = await self._retrieve_cohorts(st, spec, remaining)
            if ok is not None:
                return ok
            spec = await self._choose_chart(st, spec)
            if spec.needs.results:
                await self._fetch_result_records(st, remaining)
        return await self._call_handler(st, spec, remaining)

    async def _choose_chart(self, st: _Run, spec: IntentSpec) -> IntentSpec:
        """Chart planner agent: with the cohort in hand, pick the most informative view (validated menu)."""
        if not self.chart_planner:
            return spec
        decision = await chart_planner.choose(
            st.plan,
            st.request.query,
            st.ctx.cohorts,
            llm=self.llm,
            settings=self.settings,
            explicit_viz=(st.request.preferred_visualization or detect_chart(st.request.query)) is not None,
        )
        if decision.method != "llm" or decision.plan == st.plan:
            return spec
        changed = decision.plan.intent is not st.plan.intent
        n = sum(len(c.studies) for c in st.ctx.cohorts)
        view = (decision.plan.preferred_visualization or get_intent_spec(decision.plan.intent).primary).value
        st.plan = decision.plan.model_copy(
            update={
                "assumptions": [  # template only: no LLM text in the response
                    *decision.plan.assumptions,
                    f"The question does not name a view, so the chart was chosen after "
                    f"profiling the {n:,} retrieved trials: {decision.plan.intent.value.replace('_', ' ')} as "
                    f"{view.replace('_', ' ')}.",
                ]
            }
        )
        st.ctx.plan = st.plan
        return get_intent_spec(st.plan.intent) if changed else spec

    async def _retrieve_cohorts(
        self, st: _Run, spec: IntentSpec, remaining: Callable[[], float]
    ) -> HandlerResult | None:
        """Fill ``ctx.cohorts``; returns an early result for zero results / outages, else None."""
        req, ctx = st.request, st.ctx
        field_set = spec.field_set if spec.field_set is not FieldSet.FULL else FieldSet.LANDSCAPE
        if registry_counts.site_level(st.plan, req.query):
            field_set = FieldSet.PEOPLE  # facility, city, state, geo point
        max_studies = req.max_studies or self.settings.cohort_limit
        registry = registry_counts.supports(st.plan, req.query)
        cohorts = await self._fetch_cohorts(st, st.plan.cohorts, field_set, max_studies, remaining, registry)
        if not cohorts:
            return _kpi_result("ClinicalTrials.gov unavailable", "Could not retrieve trials right now; try again", [])
        if all(c.total_count == 0 for c in cohorts):
            return await self._zero_results(st, cohorts, remaining)
        ctx.cohorts = cohorts
        st.matched_total = sum(c.total_count for c in cohorts)
        st.queries += [c.search_url for c in cohorts if c.search_url]
        st.filters = self._filters(cohorts)
        for c in cohorts:
            if c.snapshot_at:
                taken = dt.datetime.fromtimestamp(c.snapshot_at, dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
                st.notes.append(
                    note(
                        NoteCode.SNAPSHOT,
                        f"{c.label}: all {c.total_count:,} trial records read from "
                        f"the stored cohort snapshot taken {taken}; ClinicalTrials.gov still reports "
                        "the same number of matching trials.",
                    )
                )
        return None

    async def _zero_results(self, st: _Run, cohorts: list[CohortData], remaining: Callable[[], float]) -> HandlerResult:
        """Zero matches is an answer, not a reason to widen: show 0 (cited by the search URL) and offer, as
        follow-up questions, the filters whose removal would find trials (count probes only)."""
        with step("relax_probe", cohorts=[c.label for c in cohorts]) as info:
            relaxed = (
                await self._bounded(
                    asyncio.gather(*(relax_until_nonzero(self.client, cf) for cf in st.plan.cohorts)),
                    remaining() - FETCH_RESERVE_S,
                    st,
                    "filter check",
                )
                or []
            )
            info["summary"] = {cf.label: steps for (cf, steps) in relaxed}
        st.matched_total = 0
        st.queries += [c.search_url for c in cohorts]
        st.filters = self._filters(cohorts)
        names = {
            "countries": "country",
            "years": "year range",
            "phases": "phase",
            "statuses": "status",
            "drugs_to_term": "drug-field",
        }
        hints = [
            f"{cf.label}: ask again without the {', '.join(names.get(x, x) for x in steps)} filter"
            for cf, steps in relaxed
            if steps
        ]
        viz, ds = count_cards(cohorts, [], "No matching trials")
        return HandlerResult(
            visualization=viz,
            studies_analyzed=0,
            full_refs=ds.full_refs,
            facts={"matching_trials": 0, "relax_hints": hints},
            notes=[
                note(
                    NoteCode.ZERO_RESULTS,
                    "ClinicalTrials.gov has no trials matching every "
                    "filter in the question; nothing was widened."
                    + (" Removing a filter would find trials: " + "; ".join(hints) + "." if hints else ""),
                    "warning",
                )
            ],
        )

    async def _fetch_cohorts(
        self,
        st: _Run,
        cfs: list[CohortFilter],
        field_set: FieldSet,
        max_studies: int,
        remaining: Callable[[], float],
        registry: bool = False,
    ) -> list[CohortData]:
        with step(
            "fetch_cohorts",
            cohorts=[cf.model_dump(exclude_defaults=True) for cf in cfs],
            field_set=field_set.value,
            max_studies=max_studies,
            registry_counts_if_large=registry,
        ) as info:
            tasks = [
                asyncio.create_task(
                    fetch_cohort(
                        self.client,
                        cf,
                        field_set=field_set,
                        max_studies=max_studies,
                        refresh=st.request.refresh,
                        registry_counts_if_large=registry,
                    )
                )
                for cf in cfs
            ]
            done, pending = await asyncio.wait(tasks, timeout=max(0.1, remaining() - FETCH_RESERVE_S))
            for t in pending:
                t.cancel()
            out: list[CohortData] = []
            for cf, t in zip(cfs, tasks, strict=True):
                if t in pending:
                    st.degrade(f"Retrieval for '{cf.label}' exceeded the time budget and was skipped.")
                elif t.cancelled():
                    st.degrade(f"Retrieval for '{cf.label}' was cancelled.")
                elif t.exception() is not None:
                    st.degrade(f"ClinicalTrials.gov request for '{cf.label}' failed ({type(t.exception()).__name__}).")
                else:
                    out.append(t.result())
            info["summary"] = {
                c.label: {"total": c.total_count, "fetched": len(c.studies), "counting": c.counting} for c in out
            }
            if pending or len(out) < len(cfs):
                info["status"] = "degraded"
        return out

    # -- retrieval agent (only on ambiguity or zero results) ---------------------------------------
    def _retrieval_tools(self, refresh: bool) -> Any:
        """:class:`RetrievalTools` bound to this client's ``count`` / ``search_page`` / ``get_study``."""
        from ctviz.agents.retrieval_agent import RetrievalTools

        client = self.client

        async def probe_count(cf: CohortFilter) -> int:
            return await client.count(SearchSpec.from_cohort(cf), refresh=refresh)

        async def search_studies(cf: CohortFilter, page_size: int) -> list[dict[str, Any]]:
            page = await client.search_page(SearchSpec.from_cohort(cf), page_size=page_size, refresh=refresh)
            out = []
            for raw in page.studies:
                try:
                    s = Study(raw)
                except ValueError:  # record without an NCT id: not citable, skip
                    continue
                out.append({"nct_id": s.nct_id, "title": s.get("brief_title")})
            return out

        async def get_study(nct_id: str) -> dict[str, Any]:
            return await client.get_study(nct_id, refresh=refresh)

        return RetrievalTools(probe_count=probe_count, search_studies=search_studies, get_study=get_study)

    async def _retrieval_agent(
        self,
        st: _Run,
        cohort: CohortFilter,
        remaining: Callable[[], float],
        *,
        ambiguous: dict[str, list[str]] | None = None,
    ) -> Any:
        """Run ``resolve_retrieval`` within the deadline; ``None`` when unavailable or it failed."""
        try:
            from ctviz.agents.retrieval_agent import resolve_retrieval
        except ImportError:
            return None
        with step("retrieval_agent", cohort=cohort.label, ambiguous=sorted(ambiguous or {})) as info:
            decision = await self._bounded(
                resolve_retrieval(
                    st.plan,
                    st.request.query,
                    cohort,
                    llm=self.llm,
                    settings=self.settings,
                    tools=self._retrieval_tools(st.request.refresh),
                    ambiguous=ambiguous,
                ),
                remaining() - FETCH_RESERVE_S,
                st,
                "retrieval agent",
            )
            if decision is not None:
                info["summary"] = {
                    "resolved": decision.cohort is not None,
                    "count": decision.count,
                    "tool_calls": decision.tool_calls,
                    "rejected": len(decision.rejected),
                    "used_llm": decision.used_llm,
                    "rationale": decision.rationale,
                }
        return decision

    def _apply_decision(self, st: _Run, index: int, decision: Any) -> bool:
        """Surface the agent's notes/clarification; swap in its cohort. True when a cohort was replaced."""
        st.notes.extend(decision.notes)
        update: dict[str, Any] = {}
        if decision.clarification:
            update["clarification"] = decision.clarification
        if decision.cohort is not None:
            cohorts = list(st.plan.cohorts)
            if index < len(cohorts):
                cohorts[index] = decision.cohort
            else:
                cohorts.append(decision.cohort)
            update["cohorts"] = cohorts
        if update:
            st.plan = st.plan.model_copy(update=update)
            st.ctx.plan = st.plan
        return decision.cohort is not None

    async def _disambiguate(self, st: _Run, remaining: Callable[[], float]) -> None:
        """When the question names an ambiguous entity ("Merck", "MS"), probe each meaning (no LLM)."""
        try:
            from ctviz.agents.rules_planner import extract
        except ImportError:
            return
        named = {v.casefold() for c in st.plan.cohorts for v in (*c.drugs, *c.conditions, *c.sponsors, *c.countries)}
        # a term is still open only if the plan did not already settle on one of its meanings
        ambiguous = {
            k: v for k, v in extract(st.request).ambiguous.items() if not any(m.casefold() in named for m in v)
        }
        if not ambiguous:
            return
        base = st.plan.cohorts[0] if st.plan.cohorts else CohortFilter(label=next(iter(ambiguous)))
        decision = await self._retrieval_agent(st, base, remaining, ambiguous=ambiguous)
        if decision is not None and decision.cohort is None and decision.options:
            # several meanings have trials: ask instead of picking one (or merging them) for the user
            st.notes.extend(decision.notes)
            raise NoChart(
                "needs_clarification",
                Clarification(
                    reason="ambiguous_entity",
                    message=decision.clarification or "Which meaning did you intend?",
                    unmatched_terms=list(ambiguous),
                    suggestions=[f"Ask again naming '{m}'" for m in decision.options],
                ),
            )
        if decision is not None:
            self._apply_decision(st, 0, decision)

    async def _repair_zero_results(self, st: _Run, remaining: Callable[[], float]) -> bool:
        """Every cohort is still empty after the relax ladder: let the retrieval agent rewrite each one."""
        repaired = False
        for i, cf in enumerate(list(st.plan.cohorts)):
            decision = await self._retrieval_agent(st, cf, remaining)
            if decision is not None:
                repaired = self._apply_decision(st, i, decision) or repaired
        return repaired

    async def _fetch_result_records(self, st: _Run, remaining: Callable[[], float]) -> None:
        """Cohort-scope results intents: full records for up to ``max_full_records`` result-bearing trials."""

        # Per cohort, largest enrollment first (pivotal trials), interleaved so a comparison is balanced.
        def enrollment(s: Study) -> int:
            n = s.get("enrollment_count")
            return n if isinstance(n, int) else 0

        ranked = [
            sorted((s for s in c.studies if s.has_results), key=lambda s: (-enrollment(s), s.nct_id))
            for c in st.ctx.cohorts
        ]
        candidates = list(
            dict.fromkeys(s.nct_id for tier in itertools.zip_longest(*ranked) for s in tier if s is not None)
        )[: self.settings.max_full_records]
        with step("fetch_result_records", candidates=len(candidates)) as info:
            if candidates:
                trials = await self._bounded(
                    fetch_trials(self.client, candidates, refresh=st.request.refresh),
                    remaining() - FETCH_RESERVE_S,
                    st,
                    "full-record fetch",
                )
                st.ctx.trials = trials or []
            info["summary"] = {"fetched": len(st.ctx.trials)}

    async def _trial_required(self, st: _Run, remaining: Callable[[], float]) -> HandlerResult:
        """Single-trial intent without an NCT id: ask which trial, listing matching candidates (no chart)."""
        plan = st.plan
        if not plan.cohorts:
            raise NoChart("needs_clarification", answerability.trial_required([], None))
        cf = plan.cohorts[0]
        with step("search_candidates", cohort=cf.model_dump(exclude_defaults=True)) as info:
            found = await self._bounded(
                search_candidates(
                    self.client,
                    cf,
                    limit=CANDIDATE_LIMIT,
                    recruiting_first=plan.persona is Persona.PATIENT,
                    refresh=st.request.refresh,
                ),
                remaining() - FETCH_RESERVE_S,
                st,
                "candidate search",
            )
            studies, total, spec = found if found else ([], 0, SearchSpec.from_cohort(cf))
            info["summary"] = {"total": total, "candidates": [s.nct_id for s in studies]}
        st.matched_total = total
        st.queries.append(spec.ui_url())
        st.filters = spec.describe()
        options = [
            {"nct_id": s.nct_id, "title": str(s.get("brief_title") or ""), "status": str(s.get("status") or "")}
            for s in studies
        ]
        raise NoChart("needs_clarification", answerability.trial_required(options, cf.label))

    async def _call_handler(self, st: _Run, spec: IntentSpec, remaining: Callable[[], float]) -> HandlerResult:
        ctx = st.ctx
        handler = get_handler(st.plan.intent)
        if handler is None:
            viz, ds = count_cards(ctx.cohorts, ctx.trials, f"{spec.description.split('.')[0]}")
            return HandlerResult(
                visualization=viz,
                studies_analyzed=len(ctx.studies),
                full_refs=ds.full_refs,
                notes=[
                    note(
                        NoteCode.DEGRADED,
                        f"The '{st.plan.intent.value}' view is not "
                        "available yet; showing matching-trial counts instead.",
                        "warning",
                    )
                ],
            )
        for attempt in (1, 2):  # one attempt + at most one repair
            try:
                with step(
                    "handler",
                    intent=st.plan.intent.value,
                    attempt=attempt,
                    studies=len(ctx.studies),
                    trials=len(ctx.trials),
                ) as info:
                    result = await await_within(handler(ctx), max(0.2, remaining() - HANDLER_RESERVE_S))
                    info["summary"] = {
                        "type": result.visualization.type.value,
                        "rows": _compact_rows(result.visualization, 3),
                        "notes": [n.code.value for n in result.notes],
                    }
                return result
            except TimeoutError:
                st.degrade("The analysis step exceeded the time budget; showing the matching trials instead.")
                break
            except Exception as exc:
                log.warning("handler %s failed: %s", st.plan.intent.value, type(exc).__name__)
                if attempt == 1 and self._repairable(ctx, spec):
                    await self._fetch_result_records(
                        st, remaining
                    ) if spec.needs.results else await self._fetch_full_records(st, remaining)
                    if ctx.trials:
                        continue
                st.degrade(
                    f"The '{st.plan.intent.value}' view could not be built from this data "
                    f"({type(exc).__name__}); showing the matching trials instead."
                )
                break
        studies = ctx.studies[:FALLBACK_TABLE_ROWS]
        if not studies:
            return _kpi_result("No data", "No matching data found in ClinicalTrials.gov", [])
        viz, ds = trial_table(studies, "Matching trials")
        return HandlerResult(visualization=viz, studies_analyzed=len(studies), full_refs=ds.full_refs)

    @staticmethod
    def _repairable(ctx: HandlerContext, spec: IntentSpec) -> bool:
        return not ctx.trials and bool(ctx.cohorts) and spec.field_set is FieldSet.FULL

    async def _fetch_full_records(self, st: _Run, remaining: Callable[[], float]) -> None:
        ids = [s.nct_id for s in st.ctx.studies][: self.settings.max_full_records]
        with step("repair_fetch_full_records", candidates=len(ids)) as info:
            st.ctx.trials = (
                await self._bounded(
                    fetch_trials(self.client, ids, refresh=st.request.refresh),
                    remaining() - FETCH_RESERVE_S,
                    st,
                    "full-record fetch",
                )
                or []
            )
            info["summary"] = {"fetched": len(st.ctx.trials)}

    # -- helpers ----------------------------------------------------------------------------------
    async def _bounded(self, aw: Awaitable[Any], timeout: float, st: _Run, what: str) -> Any:
        """Await with a timeout; on timeout/error mark the run degraded and return None."""
        try:
            return await await_within(aw, max(0.1, timeout))
        except TimeoutError:
            st.degrade(f"The {what} exceeded the time budget.")
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise  # this request itself is being cancelled: never swallow that
            st.degrade(f"The {what} was cancelled.")  # a child task was cancelled underneath us
        except Exception as exc:
            st.degrade(f"The {what} failed ({type(exc).__name__}).")
        return None

    async def _plan(
        self, request: VisualizeRequest, remaining: Callable[[], float]
    ) -> tuple[QueryPlan, list[ResolvedEntity]]:
        timeout = max(0.5, min(remaining() - 3.0, self.settings.llm_timeout_s + 4.0))
        try:
            plan, entities = await await_within(
                self.planner(request, settings=self.settings, llm=self.llm, db=self.db), timeout
            )
            return plan, list(entities)
        except Exception as exc:  # incl. timeout: fall back to the deterministic rule plan
            log.warning("planner failed (%s); using the rule plan", type(exc).__name__)
            try:
                from ctviz.agents.rules_planner import rule_plan

                plan = rule_plan(request)
            except Exception:  # noqa: BLE001 - last resort
                plan = minimal_plan(request)
            return plan.model_copy(
                update={
                    "assumptions": [
                        *plan.assumptions,
                        f"Planner error ({type(exc).__name__}); the rule-based plan was used.",
                    ]
                }
            ), []

    async def _version(self) -> dict[str, Any]:
        try:
            return await self.client.version() or {}
        except Exception:
            return {}

    async def _await_version(
        self, task: asyncio.Task[dict[str, Any]], remaining: Callable[[], float]
    ) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=max(0.05, min(1.0, remaining())))
        except Exception:
            return {}

    async def _raw_records(
        self, viz: Visualization, ctx: HandlerContext, remaining: Callable[[], float]
    ) -> dict[str, dict]:
        """Raw JSON for every cited trial: this request's retrievals first, then the study store."""
        raw: dict[str, dict] = dict(seen_studies())
        for s in [*ctx.trials, *(s for c in ctx.cohorts for s in c.studies)]:
            raw.setdefault(s.nct_id, s.raw)
        rows = viz.data.nodes + viz.data.edges if isinstance(viz.data, GraphData) else viz.data
        missing = sorted({c.nct_id for r in rows for c in r.citations if c.nct_id and c.nct_id not in raw})
        if missing:
            try:
                raw.update(
                    await asyncio.wait_for(
                        load_studies(self.db, missing), timeout=max(0.05, min(CACHE_READ_TIMEOUT_S, remaining()))
                    )
                )
            except TimeoutError:
                log.info("study store lookup timed out; %d citations cannot be re-verified", len(missing))
        return raw

    @staticmethod
    async def _verify_citations_bounded(
        viz: Visualization, raw_by_nct: dict[str, dict], remaining: Callable[[], float]
    ) -> tuple[Visualization, list[Note]] | None:
        """Citation re-verification in a worker thread, bounded by the deadline (None = skipped)."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(verify_citations, viz, raw_by_nct), timeout=max(0.2, remaining())
            )
        except TimeoutError:
            return None

    @staticmethod
    def _filters(cohorts: list[CohortData]) -> dict[str, Any]:
        if len(cohorts) == 1:
            return cohorts[0].spec.describe()
        return {"cohorts": {c.label: c.spec.describe() for c in cohorts}}

    @staticmethod
    def _interpretation(plan: QueryPlan, entities: list[ResolvedEntity], parser: str | None = None) -> Interpretation:
        # Non-blocking planner remarks (low confidence, ambiguous wording) are assumptions; blocking cases
        # are answered with status != ok and a top-level clarification instead.
        assumptions = [*plan.assumptions, *([plan.clarification] if plan.clarification else [])]
        used = {v.casefold() for c in plan.cohorts for v in (*c.drugs, *c.conditions, *c.sponsors, *c.countries)}
        used |= {n.casefold() for n in plan.nct_ids}
        entities = [
            e
            for e in entities
            if e.resolved.casefold() in used or e.kind == "nct_id" or any(t.casefold() in used for t in e.search_terms)
        ]
        return Interpretation(
            intent=plan.intent,
            intent_confidence=plan.intent_confidence,
            persona=plan.persona,
            scope=plan.scope,
            nct_ids=plan.nct_ids,
            cohorts=plan.cohorts,
            resolved_entities=entities,
            assumptions=list(dict.fromkeys(assumptions)),
            parser=parser or plan.parser,
        )  # type: ignore[arg-type]

    def _from_cache(
        self, request: VisualizeRequest, request_id: str, cached: dict[str, Any], tr: RequestTrace
    ) -> VisualizeResponse:
        resp = VisualizeResponse.model_validate(cached["response"])
        interp = resp.meta.interpretation.model_copy(update={"parser": "cache"})
        meta = resp.meta.model_copy(
            update={
                "interpretation": interp,
                "cache": CacheInfo(response_hit=True, ctgov_hits=tr.ctgov_hits, ctgov_misses=tr.ctgov_misses),
                "trace": tr.to_model() if request.include_trace and self.settings.include_trace else None,
            }
        )
        resp = resp.model_copy(
            update={
                "request_id": request_id,
                "meta": meta,
                "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            }
        )
        # Normal priority (never dropped): "show all" sources must work for this request id. Until the write
        # lands, evidence() serves it from memory (source record, else rebuilt from this cached response).
        orig = cached.get("request_id")
        self.pending_evidence.remember(request_id, src=orig, resp=resp)
        spawn(self._copy_citations(orig, request_id, resp), name="citations-alias")
        return resp

    async def _copy_citations(self, src: str | None, dst: str, resp: VisualizeResponse) -> None:
        await rc.copy_or_rebuild(self.db, src, dst, self.settings.ttl_response_s * 24, resp)
        self.pending_evidence.done(dst)

    async def _put_evidence(self, request_id: str, cites: dict[str, Any], ncts: dict[str, list[str]]) -> None:
        await rc.put_citations(self.db, request_id, cites, self.settings.ttl_response_s * 24, nct_ids=ncts)
        self.pending_evidence.done(request_id)

    async def evidence(self, request_id: str) -> rc.Evidence | None:
        """``(datum id -> citations, datum id -> complete nct_ids)`` for a response this service produced, also
        while (or if) its background write has not landed; None for an unknown or expired id."""
        return await self.pending_evidence.resolve(self.db, request_id)

    def _background(
        self,
        request_id: str,
        key: str,
        resp: VisualizeResponse,
        result: HandlerResult,
        ctx: HandlerContext,
        *,
        cacheable: bool,
    ) -> None:
        """Citation store, response cache and graph ingest; never on the response path."""
        rows = resp.visualization.data
        all_rows = rows.nodes + rows.edges if isinstance(rows, GraphData) else rows
        cites = rc.serialize_refs({k: v for k, v in result.full_refs.items()})
        for d in all_rows:  # datums without full refs (or after verification drops) use their inline list
            if d.id not in cites or d.citations_total is None:
                cites[d.id] = [c.model_dump(mode="json", exclude_none=True) for c in d.citations]
        # Complete per-datum trial lists: the public JSON caps nct_ids, the citations endpoint serves the rest.
        ncts = {d.id: list(d.nct_ids) for d in all_rows if d.nct_ids}
        self.pending_evidence.remember(request_id, record=(cites, ncts))
        spawn(self._put_evidence(request_id, cites, ncts), name="citations")
        if cacheable and self.use_response_cache:
            body = resp.model_copy(update={"meta": resp.meta.model_copy(update={"trace": None})})
            spawn(
                rc.put_response(
                    self.db,
                    key,
                    {"request_id": request_id, "response": body.model_dump(mode="json", exclude_none=True)},
                    self.settings.ttl_response_s,
                ),
                name="response-cache",
            )
        studies = ctx.studies
        if self.graph is not None and studies:
            spawn(self.graph.ingest(studies), name="graph-ingest", low_priority=True)
        if self.text_index is not None and studies:
            spawn(self.text_index.index(studies), name="fts-index", low_priority=True)

    def _emergency(
        self,
        request: VisualizeRequest,
        request_id: str,
        tr: RequestTrace,
        message: str = "Internal error while answering; no data is shown.",
    ) -> VisualizeResponse:
        plan = minimal_plan(request)
        return VisualizeResponse(
            request_id=request_id,
            created_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            visualization=empty_kpi("Something went wrong", "The request could not be completed; please retry"),
            meta=ResponseMeta(
                interpretation=self._interpretation(plan, []),
                coverage=Coverage(studies_matched_total=0, studies_analyzed=0),
                notes=[note(NoteCode.DEGRADED, message, "warning")],
                cache=CacheInfo(ctgov_hits=tr.ctgov_hits, ctgov_misses=tr.ctgov_misses),
                trace=tr.to_model() if request.include_trace and self.settings.include_trace else None,
            ),
        )
