"""Retrieval agent: a small, bounded repair step used ONLY on ambiguity or zero results.

The default retrieval path is pure code (``ctviz/ctgov/retrieval.py``). The orchestrator calls
:func:`resolve_retrieval` when the planner flagged an ambiguous entity or when every deterministic
relaxation still returned zero studies.

* **Ambiguity** is resolved deterministically: each listed meaning is probed under its own kind
  (``Merck`` -> sponsor, ``MS`` -> condition). Exactly one meaning with trials -> used, with an
  ``APPROXIMATED`` note; more than one -> no cohort, a clarification listing the meanings.
* **Zero results**: at most ONE LLM call (``0-1`` per CLAUDE.md). The model proposes one tool call
  plus up to two alternative cohorts, ordered most-faithful first. Invalid proposals are rejected and
  we fall back to rules (no second LLM call).
* Every proposed tool argument is validated with Pydantic BEFORE execution (NCT regex, phase/status
  enums, years 1990..current+5, page size <= 1000, term <= 200 chars, no Essie operators such as
  ``AREA[`` / ``OR``) and checked for faithfulness: rewritten entities must be the originals, their
  aliases / spelling corrections / class members, or listed meanings; filters may only be removed.
* At most 3 tool calls; every call has a timeout; transient errors are retried with exponential
  backoff while the request deadline allows; every call and rejection is a trace step.
* The chosen rewrite is described in an ``APPROXIMATED`` note (what changed, nothing silent).

Tools are plain async callables (see :class:`RetrievalTools`) so everything is testable offline.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ctviz.agents.llm import LLM, LLMError
from ctviz.agents.router import ModelRouter
from ctviz.api.schemas.plan import CohortFilter, QueryPlan
from ctviz.api.schemas.request import NCT_PATTERN, PHASES, STATUSES
from ctviz.api.schemas.response import Note, NoteCode
from ctviz.config import Settings
from ctviz.core.budget import BudgetExceeded, current_budget
from ctviz.core.entities import EntityResolver, get_resolver, norm
from ctviz.obs.trace import current_trace

MAX_STEPS = 3  # tool calls
MAX_ALTERNATIVES = 2
MAX_PAGE_SIZE = 1000
MAX_TERM_LEN = 200
MIN_YEAR = 1990
TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}
_ESSIE = re.compile(r"AREA\[|RANGE\[|SEARCH\[|EXPANSION\[|COVERAGE\[|[\[\]{}\"]|\b(AND|OR|NOT)\b")
_KINDS = (("drug", "drugs"), ("condition", "conditions"), ("sponsor", "sponsors"), ("country", "countries"))

ToolName = Literal["probe_count", "search_studies", "get_study"]


# ---------------------------------------------------------------------------------------- tool args
def _max_year() -> int:
    return dt.date.today().year + 5


def _no_operators(text: str) -> str:
    if _ESSIE.search(text):
        raise ValueError("search operators (AREA[, RANGE[, AND/OR/NOT, brackets, quotes) are not allowed")
    return text


class ToolCohort(BaseModel):
    """Validated cohort arguments for search tools (strict: unknown keys rejected)."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(default="candidate", max_length=80)
    drugs: list[str] = Field(default_factory=list, max_length=10)
    conditions: list[str] = Field(default_factory=list, max_length=10)
    sponsors: list[str] = Field(default_factory=list, max_length=5)
    countries: list[str] = Field(default_factory=list, max_length=5)
    phases: list[str] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=list)
    start_year: int | None = None
    end_year: int | None = None
    term: str | None = Field(default=None, max_length=MAX_TERM_LEN)

    @field_validator("drugs", "conditions", "sponsors", "countries")
    @classmethod
    def _names(cls, v: list[str]) -> list[str]:
        out = []
        for x in v:
            x = x.strip()
            if not x:
                raise ValueError("empty entity name")
            if len(x) > MAX_TERM_LEN:
                raise ValueError(f"entity name longer than {MAX_TERM_LEN} chars")
            out.append(_no_operators(x))
        return out

    @field_validator("term")
    @classmethod
    def _term(cls, v: str | None) -> str | None:
        return _no_operators(v.strip()) if v else None

    @field_validator("phases")
    @classmethod
    def _phases(cls, v: list[str]) -> list[str]:
        bad = [p for p in v if p not in PHASES]
        if bad:
            raise ValueError(f"unknown phases {bad}; allowed {list(PHASES)}")
        return v

    @field_validator("statuses")
    @classmethod
    def _statuses(cls, v: list[str]) -> list[str]:
        bad = [s for s in v if s not in STATUSES]
        if bad:
            raise ValueError(f"unknown statuses {bad}; allowed {list(STATUSES)}")
        return v

    @field_validator("start_year", "end_year")
    @classmethod
    def _year(cls, v: int | None) -> int | None:
        if v is not None and not MIN_YEAR <= v <= _max_year():
            raise ValueError(f"year {v} outside {MIN_YEAR}..{_max_year()}")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> ToolCohort:
        if self.start_year and self.end_year and self.start_year > self.end_year:
            raise ValueError("start_year must be <= end_year")
        if not (self.drugs or self.conditions or self.sponsors or self.countries or self.term):
            raise ValueError("cohort needs at least one of drugs/conditions/sponsors/countries/term")
        return self

    def to_filter(self) -> CohortFilter:
        return CohortFilter(**self.model_dump())


class ProbeCountArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cohort: ToolCohort


class SearchStudiesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cohort: ToolCohort
    page_size: int = Field(default=10, ge=1, le=MAX_PAGE_SIZE)


class GetStudyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nct_id: str = Field(pattern=NCT_PATTERN)

    @field_validator("nct_id", mode="before")
    @classmethod
    def _upper(cls, v: Any) -> Any:
        return v.strip().upper() if isinstance(v, str) else v


TOOL_ARGS: dict[str, type[BaseModel]] = {
    "probe_count": ProbeCountArgs,
    "search_studies": SearchStudiesArgs,
    "get_study": GetStudyArgs,
}


# ---------------------------------------------------------------------------------------- LLM schema
class StepCohort(BaseModel):
    """Raw cohort the model proposes (validation happens in :class:`ToolCohort`)."""

    drugs: list[str]
    conditions: list[str]
    sponsors: list[str]
    countries: list[str]
    phases: list[str]
    statuses: list[str]
    start_year: int | None
    end_year: int | None
    term: str | None


class AgentStep(BaseModel):
    """The model's single proposal: one tool call (+ up to 2 alternative cohorts, most faithful first)."""

    action: Literal["probe_count", "search_studies", "get_study", "finish"]
    reason: str
    cohort: StepCohort | None
    page_size: int | None
    nct_id: str | None
    alternatives: list[StepCohort] | None = None


SYSTEM_PROMPT = """You repair a ClinicalTrials.gov search that returned zero studies. You get ONE turn.
Return one action:
- probe_count(cohort): count matching studies for your best rewrite; put up to 2 fallback rewrites in
  `alternatives`, ordered most faithful first.
- search_studies(cohort, page_size<=1000) or get_study(nct_id = NCT + 8 digits) only if the question
  names a specific trial.
- finish when no faithful rewrite exists.
Allowed rewrites only: fix spelling, use the generic or brand name of the SAME drug, expand a drug
class to its members, or remove a phase/status/year filter. Never add a drug, condition, sponsor or
country that is not in the original cohort. No search syntax (AREA[, RANGE[, AND/OR/NOT, quotes).
Cohort fields: drugs, conditions, sponsors, countries, phases (EARLY_PHASE1, PHASE1, PHASE2, PHASE3,
PHASE4, NA), statuses (registry enum), start_year/end_year, term (<= 200 chars).
Unused fields must be null or empty lists."""


# ---------------------------------------------------------------------------------------- tools + result
@dataclass
class RetrievalTools:
    """Async tool implementations (wired to CTGovClient by the orchestrator, fakes in tests)."""

    probe_count: Callable[[CohortFilter], Awaitable[int]]
    search_studies: Callable[[CohortFilter, int], Awaitable[list[dict[str, Any]]]]
    get_study: Callable[[str], Awaitable[dict[str, Any]]]


class TransientToolError(RuntimeError):
    """Raise from a tool implementation to request a retry with backoff."""


@dataclass
class ToolCallRecord:
    tool: str
    args: dict[str, Any] | None
    ok: bool
    ms: float = 0.0
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class RetrievalDecision:
    """Outcome: a (possibly rewritten) cohort with a nonzero count, or nothing (+ clarification)."""

    cohort: CohortFilter | None
    count: int | None
    nct_ids: list[str] = field(default_factory=list)
    calls: list[ToolCallRecord] = field(default_factory=list)
    rejected: list[ToolCallRecord] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    clarification: str | None = None
    options: list[str] = field(default_factory=list)  # meanings with trials when the user must choose
    used_llm: bool = False
    rationale: str = ""

    @property
    def tool_calls(self) -> int:
        return len(self.calls)


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, TransientToolError | TimeoutError | ConnectionError):
        return True
    status = getattr(exc, "status", None)
    return isinstance(status, int) and status in TRANSIENT_STATUS


def _summary(tool: str, result: Any) -> dict[str, Any]:
    if tool == "probe_count":
        return {"count": int(result)}
    if tool == "search_studies":
        return {"n": len(result), "nct_ids": [r.get("nct_id") for r in result[:5]]}
    proto = (result or {}).get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    return {"nct_id": ident.get("nctId"), "title": (ident.get("briefTitle") or "")[:80]}


def _compact_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5])
    return str(exc)


Check = Callable[[BaseModel], str | None]


class ToolRunner:
    """Validates, executes (timeout + retry/backoff), charges budget and logs every tool call."""

    def __init__(
        self,
        tools: RetrievalTools,
        *,
        timeout_s: float = 5.0,
        retries: int = 2,
        base_delay_s: float = 0.2,
        max_calls: int = MAX_STEPS,
    ):
        self.tools = tools
        self.timeout_s = timeout_s
        self.retries = retries
        self.base_delay_s = base_delay_s
        self.max_calls = max_calls
        self.calls: list[ToolCallRecord] = []
        self.rejected: list[ToolCallRecord] = []

    @staticmethod
    def validate(tool: str, raw_args: dict[str, Any]) -> BaseModel:
        """Pydantic-validate tool args; raises ``ValueError`` (incl. ValidationError) when invalid."""
        model = TOOL_ARGS.get(tool)
        if model is None:
            raise ValueError(f"unknown tool {tool!r}")
        return model.model_validate(raw_args)

    def reject(self, tool: str, raw_args: dict[str, Any] | None, error: str) -> ToolCallRecord:
        rec = ToolCallRecord(tool=tool, args=raw_args, ok=False, error=error[:300])
        self.rejected.append(rec)
        with current_trace().step(f"tool:{tool}", rejected=True, error=rec.error) as info:
            info["status"] = "error"
        return rec

    @property
    def remaining(self) -> int:
        return self.max_calls - len(self.calls)

    async def run(self, tool: str, raw_args: dict[str, Any], check: Check | None = None) -> ToolCallRecord:
        """Validate (schema + optional ``check``) then execute ``tool``; invalid -> rejected, not executed."""
        try:
            args = self.validate(tool, raw_args)
        except (ValidationError, ValueError) as exc:
            return self.reject(tool, raw_args, _compact_error(exc))
        if check is not None and (why := check(args)):
            return self.reject(tool, raw_args, why)
        if len(self.calls) >= self.max_calls:
            return self.reject(tool, raw_args, f"tool-call limit {self.max_calls} reached")
        try:
            current_budget().charge_tool()
        except BudgetExceeded as exc:
            return self.reject(tool, raw_args, str(exc))
        validated = args.model_dump(exclude_defaults=False)
        rec = ToolCallRecord(tool=tool, args=validated, ok=False)
        with current_trace().step(f"tool:{tool}", args=validated) as info:
            t0 = asyncio.get_running_loop().time()
            try:
                result = await self._execute(args, info)
                rec.ok, rec.summary = True, _summary(tool, result)
                info["result"] = rec.summary
            except Exception as exc:  # tool failures are data, not crashes
                rec.error = f"{type(exc).__name__}: {str(exc)[:200]}"
                info.update(status="error", error=rec.error)
            rec.ms = round((asyncio.get_running_loop().time() - t0) * 1000, 1)
        self.calls.append(rec)
        return rec

    async def _execute(self, args: BaseModel, info: dict[str, Any]) -> Any:
        attempt = 0
        while True:
            try:
                coro: Awaitable[Any]
                if isinstance(args, ProbeCountArgs):
                    coro = self.tools.probe_count(args.cohort.to_filter())
                elif isinstance(args, SearchStudiesArgs):
                    coro = self.tools.search_studies(args.cohort.to_filter(), args.page_size)
                else:
                    assert isinstance(args, GetStudyArgs)
                    coro = self.tools.get_study(args.nct_id)
                return await asyncio.wait_for(coro, timeout=current_budget().timeout_for(self.timeout_s))
            except Exception as exc:
                delay = self.base_delay_s * (2**attempt) * (1 + random.random() * 0.25)
                # retry only transient errors, and only while the request deadline leaves room
                if attempt >= self.retries or not _is_transient(exc) or current_budget().remaining_s() <= delay:
                    raise
                attempt += 1
                info["retries"] = attempt
                await asyncio.sleep(delay)


# ---------------------------------------------------------------------------------------- faithfulness
def _allowed_names(
    original: CohortFilter, resolver: EntityResolver, ambiguous: dict[str, list[str]] | None
) -> dict[str, set[str]]:
    allowed: dict[str, set[str]] = {}
    for kind, fld in _KINDS:
        names: set[str] = set()
        for o in getattr(original, fld):
            ent = resolver.resolve(o, kind)  # type: ignore[arg-type]
            names |= {norm(o), norm(ent.resolved), *(norm(a) for a in resolver.aliases_of(kind, ent.resolved))}  # type: ignore[arg-type]
            if kind == "drug":
                names |= {norm(m) for m in resolver.expand_drug_class(o)}
        allowed[fld] = names
    for meanings in (ambiguous or {}).values():
        for m in meanings:
            kind = _meaning_kind(m, resolver)
            if kind:
                allowed[dict(_KINDS)[kind]].add(norm(m))
    return allowed


def faithfulness_error(
    proposed: ToolCohort,
    original: CohortFilter,
    resolver: EntityResolver | None = None,
    ambiguous: dict[str, list[str]] | None = None,
) -> str | None:
    """Why ``proposed`` is not a faithful rewrite of ``original`` (``None`` when it is)."""
    resolver = resolver or get_resolver()
    allowed = _allowed_names(original, resolver, ambiguous)
    vocab = {t for names in allowed.values() for n in names for t in re.findall(r"[a-z0-9]+", n)}
    vocab |= set(re.findall(r"[a-z0-9]+", norm(original.term or "")))
    term_tokens = set(re.findall(r"[a-z0-9]+", norm(proposed.term or "")))
    if term_tokens - vocab:
        return f"term adds words not in the original search: {sorted(term_tokens - vocab)[:5]}"
    for _, fld in _KINDS:
        new = getattr(proposed, fld)
        bad = [n for n in new if norm(n) not in allowed[fld]]
        if bad:
            return f"{fld} {bad[:3]} are not aliases, class members or listed meanings of the original {fld}"
        if getattr(original, fld) and not new and not proposed.term:
            return f"{fld} were removed from the search (that would widen it)"
    if not set(proposed.phases) <= set(original.phases):
        return "phase filters may only be removed, not added"
    if not set(proposed.statuses) <= set(original.statuses):
        return "status filters may only be removed, not added"
    for y in ("start_year", "end_year"):
        if getattr(proposed, y) not in (None, getattr(original, y)):
            return f"{y} may only be removed, not changed"
    return None


def describe_rewrite(original: CohortFilter, new: CohortFilter) -> str:
    """Human description of what a rewrite changed (for the APPROXIMATED note)."""
    parts: list[str] = []
    for _, fld in _KINDS:
        a, b = getattr(original, fld), getattr(new, fld)
        if a != b:
            parts.append(f"{fld} {a or '[]'} -> {b or '[]'}")
    if original.term != new.term:
        parts.append(f"free-text term '{new.term}'" if new.term else "free-text term removed")
    for fld in ("phases", "statuses"):
        removed = [x for x in getattr(original, fld) if x not in getattr(new, fld)]
        if removed:
            parts.append(f"removed {fld[:-1] if fld == 'phases' else 'status'} filter {removed}")
    for y in ("start_year", "end_year"):
        if getattr(original, y) is not None and getattr(new, y) is None:
            parts.append(f"removed {y} {getattr(original, y)}")
    return "; ".join(parts) or "no change"


def _meaning_kind(meaning: str, resolver: EntityResolver) -> str | None:
    kind, confident = resolver.guess_kind(meaning)
    return kind if confident else None


# ---------------------------------------------------------------------------------------- deterministic paths
def candidate_rewrites(cf: CohortFilter, resolver: EntityResolver | None = None) -> list[CohortFilter]:
    """Rule-based rewrites for a zero-result cohort, most faithful first."""
    resolver = resolver or get_resolver()
    out: list[CohortFilter] = []

    def add(c: CohortFilter) -> None:
        if (c.drugs or c.conditions or c.sponsors or c.countries or c.term) and c not in out and c != cf:
            out.append(c)

    fixed = cf.model_copy(deep=True)
    fixed.drugs = list(
        dict.fromkeys(
            t for d in cf.drugs for t in (resolver.expand_drug_class(d) or [resolver.resolve(d, "drug").resolved])
        )
    )
    fixed.conditions = [resolver.resolve(c, "condition").resolved for c in cf.conditions]
    fixed.sponsors = [resolver.resolve(s, "sponsor").resolved for s in cf.sponsors]
    add(fixed)
    add(fixed.model_copy(update={"phases": [], "statuses": [], "start_year": None, "end_year": None}))
    words = [*cf.drugs, *cf.conditions, *cf.sponsors]
    if words:
        add(CohortFilter(label=cf.label, term=" ".join(words)[:MAX_TERM_LEN]))
    return out


def _cohort_args(cf: CohortFilter) -> dict[str, Any]:
    data = cf.model_dump(exclude={"interventional_only", "label"})
    return {k: v for k, v in data.items() if v not in (None, [])}


def _note(original: CohortFilter, new: CohortFilter, why: str) -> Note:
    return Note(
        code=NoteCode.APPROXIMATED,
        severity="warning",
        message=f"The original search found no trials; {why}: {describe_rewrite(original, new)}.",
    )


async def _resolve_ambiguity(
    cohort: CohortFilter,
    ambiguous: dict[str, list[str]],
    runner: ToolRunner,
    resolver: EntityResolver,
    decision: RetrievalDecision,
) -> None:
    hits: list[tuple[str, str, CohortFilter, int]] = []
    tried: list[str] = []
    for surface, meanings in ambiguous.items():
        for meaning in meanings:
            kind = _meaning_kind(meaning, resolver)
            if kind is None or runner.remaining <= 0:
                continue
            fld = dict(_KINDS)[kind]
            cand = cohort.model_copy(deep=True)
            setattr(cand, fld, list(dict.fromkeys([*getattr(cand, fld), meaning])))
            rec = await runner.run("probe_count", {"cohort": _cohort_args(cand)})
            tried.append(meaning)
            if rec.ok and rec.summary.get("count"):
                hits.append((surface, meaning, cand, int(rec.summary["count"])))
    if len(hits) == 1:
        surface, meaning, cand, n = hits[0]
        cand.label = cohort.label
        decision.cohort, decision.count = cand, n
        decision.notes.append(
            Note(
                code=NoteCode.APPROXIMATED,
                severity="warning",
                message=(
                    f"'{surface}' is ambiguous; interpreted as '{meaning}', the only meaning tried with registered "
                    f"trials (tried: {', '.join(tried)})."
                ),
            )
        )
        decision.rationale = "ambiguity: single meaning with results"
    elif len(hits) > 1:
        opts = ", ".join(f"'{m}' ({n} trials)" for _, m, _, n in hits)
        decision.clarification = f"'{hits[0][0]}' is ambiguous. Which did you mean: {opts}?"
        decision.options = [m for _, m, _, _ in hits]
        decision.notes.append(Note(code=NoteCode.ENTITY_AMBIGUOUS, severity="warning", message=decision.clarification))
        decision.rationale = "ambiguity: several meanings have results; asking"
    else:
        decision.rationale = "ambiguity: no meaning has registered trials"


# ---------------------------------------------------------------------------------------- entry point
async def resolve_retrieval(
    plan: QueryPlan,
    question: str,
    cohort: CohortFilter,
    *,
    llm: LLM | None,
    settings: Settings,
    tools: RetrievalTools,
    ambiguous: dict[str, list[str]] | None = None,
    max_steps: int = MAX_STEPS,
    tool_timeout_s: float = 5.0,
    tool_retries: int = 2,
    backoff_s: float = 0.2,
) -> RetrievalDecision:
    """Disambiguate or repair ``cohort`` with at most 3 tool calls and at most 1 LLM call."""
    resolver = get_resolver()
    runner = ToolRunner(
        tools,
        timeout_s=tool_timeout_s,
        retries=tool_retries,
        base_delay_s=backoff_s,
        max_calls=min(max_steps, MAX_STEPS),
    )
    decision = RetrievalDecision(cohort=None, count=None)
    try:
        if ambiguous:
            await _resolve_ambiguity(cohort, ambiguous, runner, resolver, decision)
            return decision

        def check(args: BaseModel) -> str | None:
            c = getattr(args, "cohort", None)
            return faithfulness_error(c, cohort, resolver) if c is not None else None

        fetched: list[str] = []
        if llm is not None and current_budget().can_call_llm():
            step = await _llm_step(llm, ModelRouter(settings).model_for("retrieval"), question, plan, cohort, settings)
            if step is None:
                decision.rationale = "LLM unavailable; rules fallback"
            else:
                decision.used_llm = True
                await _run_llm_step(step, runner, check, cohort, decision, fetched)
        if decision.cohort is None and runner.remaining > 0:
            for cand in candidate_rewrites(cohort, resolver):
                if runner.remaining <= 0 or not current_budget().can_call_tool():
                    break
                rec = await runner.run("probe_count", {"cohort": _cohort_args(cand)})
                if rec.ok and rec.summary.get("count"):
                    cand.label = cohort.label
                    decision.cohort, decision.count = cand, int(rec.summary["count"])
                    decision.notes.append(_note(cohort, cand, "the search was rewritten by rule"))
                    decision.rationale = (decision.rationale + "; " if decision.rationale else "") + "rule rewrite"
                    break
        decision.nct_ids = [n for n in dict.fromkeys(fetched) if re.match(NCT_PATTERN, n or "")]
        return decision
    finally:
        decision.calls, decision.rejected = runner.calls, runner.rejected


async def _run_llm_step(
    step: AgentStep,
    runner: ToolRunner,
    check: Check,
    cohort: CohortFilter,
    decision: RetrievalDecision,
    fetched: list[str],
) -> None:
    """Execute the model's proposal; the first faithful candidate with results (in its order) wins."""
    if step.action == "get_study":
        rec = await runner.run("get_study", {"nct_id": step.nct_id})
        if rec.ok and rec.summary.get("nct_id"):
            fetched.append(rec.summary["nct_id"])
        return
    if step.action == "finish" and step.cohort is None:
        decision.rationale = "model found no faithful rewrite"
        return
    candidates = [c for c in [step.cohort, *(step.alternatives or [])[:MAX_ALTERNATIVES]] if c is not None]
    if not candidates:
        runner.reject(step.action, None, "cohort is required")
        return
    for sc in candidates:
        raw = {k: v for k, v in sc.model_dump().items() if v not in (None, [])}
        if step.action == "search_studies":
            args: dict[str, Any] = {"cohort": raw}
            if step.page_size is not None:
                args["page_size"] = step.page_size
            rec = await runner.run("search_studies", args, check)
            if rec.ok:
                fetched.extend(n for n in rec.summary.get("nct_ids", []) if n)
            break  # search results inform; selection needs a probe
        rec = await runner.run("probe_count", {"cohort": raw}, check)
        if rec.ok and rec.summary.get("count"):
            chosen = ToolCohort.model_validate(rec.args["cohort"]).to_filter()  # type: ignore[index]
            chosen.label = cohort.label
            chosen.interventional_only = cohort.interventional_only
            decision.cohort, decision.count = chosen, int(rec.summary["count"])
            decision.notes.append(_note(cohort, chosen, "the search was rewritten"))
            decision.rationale = step.reason[:200]
            return
        if runner.remaining <= 0:
            return
    if runner.rejected and not runner.calls:
        decision.rationale = "model proposed invalid tool arguments; rules fallback"


async def _llm_step(
    llm: LLM, model: str, question: str, plan: QueryPlan, cohort: CohortFilter, settings: Settings
) -> AgentStep | None:
    user = json.dumps(
        {
            "question": question,
            "intent": plan.intent.value,
            "original_cohort": cohort.model_dump(exclude_defaults=True),
        },
        ensure_ascii=False,
    )
    try:
        return await llm.structured(
            agent="retrieval",
            model=model,
            system=SYSTEM_PROMPT,
            user=user,
            schema=AgentStep,
            max_output_tokens=500,
            timeout_s=current_budget().timeout_for(settings.llm_timeout_s),
        )
    except (LLMError, ValidationError):
        return None
