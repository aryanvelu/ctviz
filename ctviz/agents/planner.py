"""Router/Planner agent: VisualizeRequest -> validated QueryPlan.

Flow (one LLM call on the hot path):

1. Deterministic extraction + rule plan (always computed; it is the fallback and the grounding set).
2. Exact-normalized plan cache (LLM plans only; never for patient requests).
3. LLM structured output (:class:`LLMPlan`) on the planner model; escalate once to the escalation
   model on schema failure or confidence < 0.6. ``LLMUnavailable`` / other ``LLMError`` -> rules.
4. :class:`PlanValidator` repairs the LLM plan deterministically. Principle: never silently widen,
   narrow or relabel what the user asked.
   * entities must be grounded (alias table / query text / structured field / a verified verbatim
     ``source_text`` span); if a kind loses every entity, the grounded rule cohorts are used, or the
     plan asks for clarification;
   * phases / statuses / years only when the request contains them (structured field or literal cue);
     never a filter on the charted dimension;
   * no LLM free text reaches the response: labels come from grounded names, clarifications and
     assumptions from templates, focus terms must be verbatim substrings of the query/profile.

Privacy: the patient profile is never sent to the LLM and patient plans are never cached.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import difflib
import hashlib
import json
import logging
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from ctviz.agents.llm import LLM, LLMError, LLMUnavailable
from ctviz.agents.router import ModelRouter, is_complex
from ctviz.agents.rules_planner import (
    GENERIC_VIZ,
    PHASE_DIM,
    STATUS_DIM,
    TEMPLATES,
    Extraction,
    clamp_years,
    detect_network_kind,
    display_name,
    entities_of,
    preferred_chart,
    rule_plan_with_extraction,
)
from ctviz.api.schemas.plan import (
    Answerability,
    CohortFilter,
    GroupBy,
    Intent,
    NetworkKind,
    Persona,
    QueryPlan,
    Scope,
    UnverifiedTerm,
)
from ctviz.api.schemas.request import PHASES, STATUSES, VisualizeRequest
from ctviz.api.schemas.response import ResolvedEntity
from ctviz.api.schemas.viz import VizType
from ctviz.config import Settings
from ctviz.core.answerability import is_subjective
from ctviz.core.budget import current_budget
from ctviz.core.entities import ALIASES_FILE, EntityResolver, get_resolver
from ctviz.core.entities import norm as _norm
from ctviz.core.intents import INTENTS_FILE, IntentSpec, load_intents
from ctviz.obs.trace import current_trace

log = logging.getLogger(__name__)

PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "planner.md"
PLAN_CACHE_NS = "plan"
PLAN_CACHE_VERSION = "p8"  # bump whenever plan repair logic changes
ESCALATION_MIN_S = 2.0  # do not start an escalation call with less time than this left

Phase = Literal["EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"]
Status = Literal[
    "RECRUITING",
    "NOT_YET_RECRUITING",
    "ENROLLING_BY_INVITATION",
    "ACTIVE_NOT_RECRUITING",
    "SUSPENDED",
    "TERMINATED",
    "COMPLETED",
    "WITHDRAWN",
    "UNKNOWN",
]
EntityKindLit = Literal["drug", "condition", "sponsor", "country"]

# literal cue in the query that licenses an LLM status filter
_STATUS_CUES = {
    "RECRUITING": r"\brecruiting\b",
    "NOT_YET_RECRUITING": r"not yet recruiting",
    "ENROLLING_BY_INVITATION": r"by invitation",
    "ACTIVE_NOT_RECRUITING": r"active,? not recruiting",
    "SUSPENDED": r"\bsuspended\b",
    "TERMINATED": r"\bterminated\b",
    "COMPLETED": r"\bcompleted?\b",
    "WITHDRAWN": r"\bwithdrawn\b",
    "UNKNOWN": r"\bunknown status\b",
}
_KIND_FIELD = {"drug": "drugs", "condition": "conditions", "sponsor": "sponsors", "country": "countries"}


# ---------------------------------------------------------------------------------------- LLM schema
# No numeric constraints; nullable fields default to None (the OpenAI SDK still marks them required).
class LLMSpan(BaseModel):
    """Where in the question an entity came from (``source_text`` must be copied verbatim)."""

    kind: EntityKindLit
    name: str
    source_text: str


class LLMCohort(BaseModel):
    label: str
    drugs: list[str]
    conditions: list[str]
    sponsors: list[str]
    countries: list[str]
    phases: list[Phase]
    statuses: list[Status]
    start_year: int | None
    end_year: int | None
    entity_spans: list[LLMSpan] | None = None


class LLMPlan(BaseModel):
    """What the planner LLM fills. Mirrors :class:`QueryPlan` minus fields code owns (parser)."""

    intent: Intent
    intent_confidence: float
    persona: Persona
    scope: Scope
    nct_ids: list[str]
    cohorts: list[LLMCohort]
    group_by: GroupBy | None
    series_by: GroupBy | None
    network_kind: NetworkKind | None
    time_granularity: Literal["year", "quarter", "month"] | None
    top_n: int | None
    preferred_visualization: VizType | None
    focus_terms: list[str]
    ambiguous_entities: list[str]
    answerability: Answerability = "answerable"
    assumptions: list[str]
    clarification: str | None


@lru_cache(maxsize=1)
def system_prompt() -> str:
    """The static planner system prompt (kept byte-identical across requests for prompt caching)."""
    return PROMPT_FILE.read_text()


# fields the planner needs for routing; the patient profile is deliberately excluded (privacy)
_LLM_FIELDS = {
    "nct_id",
    "nct_ids",
    "drug_name",
    "drug_names",
    "condition",
    "conditions",
    "trial_phase",
    "status",
    "sponsor",
    "country",
    "start_year",
    "end_year",
    "persona",
    "preferred_visualization",
    "group_by",
    "series_by",
    "time_granularity",
    "top_n",
}


def build_user_message(request: VisualizeRequest, ex: Extraction, today: dt.date | None = None) -> str:
    """Per-request user message: the question, routing fields, and alias-table entity hints."""
    fields = request.model_dump(mode="json", exclude_none=True, exclude_defaults=True, include=_LLM_FIELDS)
    hints = [
        {
            "text": f.entity.input,
            "kind": f.kind,
            "resolved": f.entity.resolved,
            **({"members": f.entity.search_terms} if f.is_class else {}),
        }
        for f in ex.found
    ]
    msg = {
        "query": request.query,
        "structured_fields": fields,
        "entity_hints": hints,
        "ambiguous_terms": ex.ambiguous,
        "nct_ids_in_request": ex.nct_ids,
        "current_year": (today or dt.date.today()).year,
    }
    return json.dumps(msg, ensure_ascii=False)


# ---------------------------------------------------------------------------------------- validator
class PlanValidator:
    """Deterministic repair of an :class:`LLMPlan` into a valid, grounded :class:`QueryPlan`."""

    def __init__(
        self,
        request: VisualizeRequest,
        ex: Extraction,
        rules: QueryPlan,
        resolver: EntityResolver | None = None,
        today: dt.date | None = None,
    ):
        self.request = request
        self.ex = ex
        self.rules = rules
        self.resolver = resolver or get_resolver()
        self.today = today
        self.repairs: list[str] = []
        self.extra_entities: list[ResolvedEntity] = []
        self.unverified: list[tuple[str, str]] = []  # (kind, term) named by the LLM but not grounded
        self.readings: list[str] = []  # "Read 'X' as 'Y'": every normalization the user should see
        self._query = _norm(request.query)
        grounded: set[str] = set()
        for f in ex.found:
            grounded |= {_norm(f.entity.input), _norm(f.entity.resolved), *(_norm(t) for t in f.entity.search_terms)}
        self._grounded = grounded

    # -------------------------------------------------------------- grounding helpers
    def _in_query(self, text: str) -> bool:
        n = _norm(text)
        return bool(n) and re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", self._query) is not None

    def _span_ok(self, span: LLMSpan, kind: str, name: str) -> bool:
        """A verified span: verbatim in the query, not ambiguous, and plausibly normalizes to ``name``."""
        src = span.source_text.strip()
        if span.kind != kind or _norm(span.name) != _norm(name) or not src or len(src) > 100:
            return False
        if not self._in_query(src) or self.resolver.is_ambiguous(src):
            return False
        if _norm(self.resolver.resolve(src, kind).resolved) == _norm(name):  # type: ignore[arg-type]
            return True
        a, b = _norm(src), _norm(name)
        shared = {t for t in re.findall(r"[a-z0-9]{3,}", a)} & {t for t in re.findall(r"[a-z0-9]{3,}", b)}
        return bool(shared) or difflib.SequenceMatcher(None, a, b).ratio() >= 0.75

    def _ground(self, names: list[str], kind: EntityKindLit, spans: list[LLMSpan], parts: list[str]) -> list[str]:
        """Grounded search terms for ``names``; ``parts`` collects label names (class name for classes)."""
        out: list[str] = []
        places = {_norm(pl) for pl in self.ex.places} if kind == "country" else set()
        for name in names:
            if not name or not name.strip() or len(name) > 200:
                continue
            if _norm(name) in places:  # a city/state the rules flagged: not a country filter (noted instead)
                self.repairs.append("dropped city/state used as a country")
                continue
            ent = self.resolver.resolve(name, kind)
            grounded = _norm(name) in self._grounded or _norm(ent.resolved) in self._grounded
            if kind == "drug" and self.resolver.is_drug_class(name):
                if grounded or self._in_query(name):
                    out.extend(ent.search_terms)
                    parts.append(ent.resolved)
                else:
                    self.repairs.append("dropped ungrounded drug class")
                continue
            if self.resolver.is_ambiguous(name):
                self.repairs.append(f"dropped ambiguous {kind}")
                continue
            if grounded or self._in_query(name):
                value = ent.resolved if ent.method in ("alias", "exact") else name.strip()
                if not grounded:
                    self.extra_entities.append(ent)
            elif any(self._span_ok(s, kind, name) for s in spans):
                value = name.strip()  # LLM normalization of a verified verbatim span
                src = next(s.source_text for s in spans if self._span_ok(s, kind, name))
                self.extra_entities.append(
                    ResolvedEntity(kind=kind, input=src, resolved=value, method="passthrough", search_terms=[value])
                )
                self.repairs.append(f"kept {kind} via verified source span")
                if _norm(src) != _norm(value):  # a spelling fix or renaming is never silent
                    self.readings.append(f"Read '{src.strip()}' as the {kind} '{value}'.")
            else:
                self.repairs.append(f"dropped ungrounded {kind}")
                src = next(
                    (
                        sp.source_text.strip()
                        for sp in spans
                        if sp.kind == kind and _norm(sp.name) == _norm(name) and self._in_query(sp.source_text)
                    ),
                    None,
                )
                self.unverified.append((kind, src or name.strip()))
                continue
            out.append(value)
            parts.append(value)
        return list(dict.fromkeys(out))

    def _phases(self, phases: list[str]) -> list[str]:
        ok = [p for p in dict.fromkeys(phases) if p in PHASES and p in self.ex.phases]
        if len(ok) < len(set(phases)):
            self.repairs.append("dropped phase filter not in the request")
        return ok

    def _statuses(self, statuses: list[str]) -> list[str]:
        ok = [
            s
            for s in dict.fromkeys(statuses)
            if s in STATUSES and (s in self.ex.statuses or re.search(_STATUS_CUES[s], self.request.query, re.I))
        ]
        if len(ok) < len(set(statuses)):
            self.repairs.append("dropped status filter not in the request")
        if not ok and self.ex.statuses:  # the question says "recruiting" / "finished" but the LLM left it out
            self.repairs.append("status filter taken from the question's wording")
            return list(self.ex.statuses)
        return ok

    def _year(self, y: int | None) -> int | None:
        if y is None or y in (self.ex.start_year, self.ex.end_year) or re.search(rf"\b{y}\b", self.request.query):
            return y
        self.repairs.append("dropped year filter not in the request")
        return None

    def _cohort(self, c: LLMCohort) -> tuple[CohortFilter, list[str]]:
        """Grounded cohort plus the kinds that lost every entity."""
        spans = c.entity_spans or []
        parts: list[str] = []
        lists: dict[str, list[str]] = {}
        lost: list[str] = []
        for kind in ("sponsor", "drug", "condition", "country"):
            names = getattr(c, _KIND_FIELD[kind])
            kept = self._ground(names, kind, spans, parts)  # type: ignore[arg-type]
            lists[_KIND_FIELD[kind]] = kept
            if names and not kept:
                lost.append(kind)
        lo, hi = clamp_years(self._year(c.start_year), self._year(c.end_year), self.today)
        cf = CohortFilter(
            label=self._label(parts),
            phases=self._phases(c.phases),
            statuses=self._statuses(c.statuses),
            start_year=lo,
            end_year=hi,
            **lists,
        )
        return cf, lost

    @staticmethod
    def _label(parts: list[str]) -> str:
        names = [display_name(p) for p in dict.fromkeys(parts)]
        return " · ".join(names[:3]) if names else "All trials"

    @staticmethod
    def _has_entities(c: CohortFilter) -> bool:
        return bool(c.drugs or c.conditions or c.sponsors or c.countries or c.term)

    def _rule_cohorts(self) -> list[CohortFilter]:
        return [c.model_copy(deep=True) for c in self.rules.cohorts]

    def _apply_structured(self, cohorts: list[CohortFilter]) -> list[CohortFilter]:
        r = self.request
        if len(r.drug_names) >= 2 or len(r.conditions) >= 2:
            self.repairs.append("structured comparison fields override LLM cohorts")
            return self._rule_cohorts()
        for kind in ("drug", "condition", "sponsor", "country"):
            terms = [
                t for f in self.ex.found if f.kind == kind and f.entity.method == "structured_field" for t in f.terms
            ]
            if terms:
                for c in cohorts:
                    setattr(c, _KIND_FIELD[kind], terms)
        for c in cohorts:
            if r.trial_phase:
                c.phases = list(r.trial_phase)
            if r.status:
                c.statuses = list(r.status)
            if r.start_year is not None:
                c.start_year = r.start_year
            if r.end_year is not None:
                c.end_year = r.end_year
        return cohorts

    # -------------------------------------------------------------- repair
    def repair(self, p: LLMPlan) -> QueryPlan:
        """Return a valid QueryPlan; every change is listed in ``self.repairs``."""
        specs = load_intents()
        intent = p.intent
        ncts = list(self.ex.nct_ids)
        if any(n.strip().upper() not in ncts for n in p.nct_ids):
            self.repairs.append("dropped NCT ids not present in the request")
        # "not in the registry" must be corroborated by the high-precision rules (costs, insurance, parking,
        # regular doctor), and when those rules fire they win: the honest card never depends on the LLM alone.
        if intent is Intent.NOT_IN_REGISTRY and self.rules.intent is not Intent.NOT_IN_REGISTRY:
            self.repairs.append(f"not_in_registry not supported by rules; using {self.rules.intent.value}")
            intent = self.rules.intent
        elif self.rules.intent is Intent.NOT_IN_REGISTRY and intent is not Intent.NOT_IN_REGISTRY:
            self.repairs.append("rules: question asks for something the registry does not record")
            intent = Intent.NOT_IN_REGISTRY
        if ncts and Scope.SINGLE_TRIAL not in specs[intent].scopes:
            self.repairs.append(f"intent {intent.value} cannot answer a single trial; using rule intent")
            intent = self.rules.intent
        intent = self._rule_overrides(intent, specs)
        spec = specs[intent]
        assumptions: list[str] = []
        clarification: str | None = None

        built = []
        unverified: list[UnverifiedTerm] = []
        for i, c in enumerate(p.cohorts[:5]):
            mark = len(self.unverified)
            built.append(self._cohort(c))
            unverified += [
                UnverifiedTerm(kind=k, text=t[:200], cohort_index=i)  # type: ignore[arg-type]
                for k, t in self.unverified[mark:]
            ]
        cohorts = [c for c, _ in built]
        lost = sorted({k for _, ks in built for k in ks})
        if lost and any(self._has_entities(c) for c in self.rules.cohorts):
            cohorts = self._rule_cohorts()  # grounded by construction
            self.repairs.append(f"lost every LLM {'/'.join(lost)}: using rule cohorts")
        # Never widen silently: every dropped name is checked against ClinicalTrials.gov by the orchestrator
        # (count probe) unless the cohorts already search for it; zero matches -> needs_clarification.
        searched = {_norm(v) for c in cohorts for v in (*c.drugs, *c.conditions, *c.sponsors, *c.countries)}
        unverified = [u for u in unverified if _norm(u.text) not in searched]
        keep = [c for c in cohorts if self._has_entities(c)]
        if keep and len(keep) < len(cohorts):  # re-point cohort indexes after dropping empty cohorts
            idx = {id(c): j for j, c in enumerate(keep)}
            unverified = [
                u.model_copy(update={"cohort_index": idx.get(id(cohorts[u.cohort_index]), 0)})
                if u.cohort_index < len(cohorts)
                else u
                for u in unverified
            ]
        cohorts = keep or cohorts[:1]
        unverified = [
            u.model_copy(update={"cohort_index": min(u.cohort_index, max(0, len(cohorts) - 1))}) for u in unverified
        ]
        if not any(self._has_entities(c) for c in cohorts) and self.ex.found:
            cohorts = self._rule_cohorts()
        cohorts = self._apply_structured(cohorts)

        if ncts:
            scope = Scope.COMPARISON if len(ncts) > 1 else Scope.SINGLE_TRIAL
            cohorts = []
        else:
            scope = p.scope
            has_entities = any(self._has_entities(c) for c in cohorts)
            if scope is Scope.SINGLE_TRIAL and Scope.COHORT in spec.scopes and has_entities:
                scope = Scope.COHORT
                self.repairs.append("no NCT id: single_trial -> cohort")
            if scope is Scope.SINGLE_TRIAL:
                # no NCT id yet: keep the grounded cohort so the user is offered matching candidate trials
                cohorts = [c for c in cohorts if self._has_entities(c)][:1]
                if Scope.SINGLE_TRIAL not in spec.scopes:
                    scope = spec.scopes[0]
                elif not clarification:
                    clarification = TEMPLATES["TRIAL_REQUIRED"]
            if (
                scope is not Scope.SINGLE_TRIAL
                and self.ex.unintelligible
                and not any(self._has_entities(c) for c in cohorts)
            ):
                # nothing recognizable in the question and nothing grounded by the LLM: ask, never "All trials"
                cohorts = []
                clarification = TEMPLATES["UNCLEAR"]
                self.repairs.append("rules: nothing recognizable in the question; no cohort")
            elif scope is not Scope.SINGLE_TRIAL:
                if not cohorts:
                    cohorts = self._rule_cohorts() or [CohortFilter(label="All trials")]
                if len(cohorts) >= 2 and Scope.COMPARISON in spec.scopes:
                    scope = Scope.COMPARISON
                elif len(cohorts) >= 2:
                    cohorts = [self._merge(cohorts)]
                    scope = Scope.COHORT
                    assumptions.append(TEMPLATES["MERGED"].format(intent=intent.value))
                elif scope is Scope.COMPARISON:
                    scope = Scope.COHORT
                    self.repairs.append("comparison needs >= 2 cohorts: scope -> cohort")
                if not any(self._has_entities(c) for c in cohorts) and not unverified:
                    assumptions.append(TEMPLATES["ALL_TRIALS"])
            if scope not in spec.scopes:
                scope = spec.scopes[0]
        for c in cohorts:
            if intent in (Intent.ENTITY_NETWORK, Intent.DEVELOPMENT_PROGRESSION):
                c.interventional_only = True
            if intent in PHASE_DIM and not self.ex.phases_from_request and c.phases:
                c.phases = []
                self.repairs.append("no filter on the charted phase dimension")
            if intent in STATUS_DIM and not self.ex.statuses_from_request and c.statuses:
                c.statuses = []
                self.repairs.append("no filter on the charted status dimension")

        ambiguous = list(self.ex.ambiguous) + [
            a for a in p.ambiguous_entities if a and len(a) <= 60 and self._in_query(a)
        ]
        ambiguous = list(dict.fromkeys(ambiguous))
        if ambiguous:  # asking which meaning comes first; keep any other question after it
            ask = TEMPLATES["AMBIGUOUS"].format(terms=", ".join(f"'{a}'" for a in ambiguous))
            clarification = f"{ask} {clarification}" if clarification else ask
        if intent is self.rules.intent:  # rule notes that explain the view (broad question, city not a country)
            assumptions += [
                a
                for a in self.rules.assumptions
                if a.startswith("Broad question") or "is not a country: the registry" in a
            ]
        conf = min(1.0, max(0.0, float(p.intent_confidence)))
        conf = max(0.0, conf - 0.1 * sum(1 for r in self.repairs if r.startswith(("dropped", "intent", "lost"))))
        if not clarification and (conf < 0.6 or p.clarification):
            clarification = TEMPLATES["LOW_CONFIDENCE"].format(description=spec.description)
        persona = Persona(self.request.persona) if self.request.persona != "auto" else p.persona
        network_kind = p.network_kind
        if intent is Intent.ENTITY_NETWORK and network_kind is None:
            network_kind = detect_network_kind(self.request.query)  # type: ignore[assignment]
        elif intent is not Intent.ENTITY_NETWORK:
            network_kind = None
        top_n = self.request.top_n or (max(1, min(100, p.top_n)) if p.top_n else None)
        answerability: Answerability = "answerable"
        if is_subjective(self.request.query):  # only the deterministic check may refuse
            answerability = "subjective_judgment"
        elif p.answerability == "subjective_judgment":
            self.repairs.append(
                "planner flagged this as a judgment question; the wording asks for registry data, so it was answered"
            )
        return QueryPlan(
            intent=intent,
            intent_confidence=round(conf, 3),
            persona=persona,
            scope=scope,
            nct_ids=ncts,
            cohorts=cohorts,
            group_by=self.request.group_by
            or p.group_by
            or (self.rules.group_by if intent is self.rules.intent is Intent.LANDSCAPE_GAP_HEATMAP else None),
            # rules backstop: "split by phase" etc. is a literal cue the LLM sometimes leaves out
            series_by=self.request.series_by
            or p.series_by
            or (self.rules.series_by if intent is self.rules.intent else None),
            network_kind=network_kind,
            time_granularity=self.request.time_granularity
            or p.time_granularity
            or ("year" if intent is Intent.TREND_OVER_TIME else None),
            top_n=top_n,
            preferred_visualization=self.request.preferred_visualization
            or (preferred_chart(self.ex, intent) if self.ex.chart else p.preferred_visualization),
            focus_terms=self._focus(p.focus_terms),
            assumptions=list(dict.fromkeys([*self.readings, *assumptions])),
            clarification=clarification,
            answerability=answerability,
            unverified_terms=unverified if not ncts else [],
            parser="llm",
        )

    def _rule_overrides(self, intent: Intent, specs: dict[Intent, IntentSpec]) -> Intent:
        """Deterministic rule decisions the LLM may not overturn (each one is listed in ``repairs``).

        * a chart the question names ("sankey", "map", preferred_visualization) must be drawable by the intent;
        * a heatmap whose rows are named ("sponsors by phase") is the gap heatmap, not phase x status;
        * a broad question about an entity without an NCT id is not a single-trial question.
        """
        ex, rule = self.ex, self.rules.intent
        chart = ex.chart
        if (
            chart is not None
            and chart not in GENERIC_VIZ
            and chart not in specs[intent].compatible_viz
            and chart in specs[rule].compatible_viz
        ):
            self.repairs.append(f"rules: {intent.value} cannot draw the requested {chart.value}; using {rule.value}")
            return rule
        if intent is Intent.PHASE_STATUS_HEATMAP and rule is Intent.LANDSCAPE_GAP_HEATMAP and ex.heat_dims[0]:
            self.repairs.append("rules: heatmap rows named; using landscape_gap_heatmap")
            return rule
        if (
            not ex.nct_ids
            and ex.rule == "landscape"
            and specs[intent].scopes == [Scope.SINGLE_TRIAL]
            and not ex.trial_hint
        ):
            self.repairs.append(f"rules: broad question without an NCT id; using {rule.value}")
            return rule
        return intent

    def _focus(self, terms: list[str]) -> list[str]:
        """Only verbatim substrings of the query or the patient profile."""
        prof = self.request.patient_profile
        allowed = self._query
        if prof:
            allowed += " | " + _norm(" | ".join([*prof.medications, *prof.conditions]))
        out = [t.strip() for t in terms if t and t.strip() and len(t) <= 80 and _norm(t) in allowed]
        if len(out) < len([t for t in terms if t and t.strip()]):
            self.repairs.append("dropped focus terms not in the request")
        return list(dict.fromkeys(out))[:10]

    @staticmethod
    def _merge(cohorts: list[CohortFilter]) -> CohortFilter:
        m = cohorts[0].model_copy(deep=True)
        for c in cohorts[1:]:
            m.drugs = list(dict.fromkeys(m.drugs + c.drugs))
            m.conditions = list(dict.fromkeys(m.conditions + c.conditions))
            m.sponsors = list(dict.fromkeys(m.sponsors + c.sponsors))
            m.countries = list(dict.fromkeys(m.countries + c.countries))
        m.label = " + ".join(c.label for c in cohorts)
        return m


# ---------------------------------------------------------------------------------------- cache
@lru_cache(maxsize=1)
def _static_fingerprint() -> str:
    """Hash of everything static that shapes a plan: prompt, alias table, intent registry."""
    h = hashlib.sha256()
    for path in (PROMPT_FILE, ALIASES_FILE, INTENTS_FILE):
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def plan_cache_key(request: VisualizeRequest, *, model: str | None = None, today: dt.date | None = None) -> str:
    """Cache key: request fields + whitespace-normalized query (case kept) + model + static fingerprint + year."""
    data = request.model_dump(
        mode="json", exclude={"narrative", "include_trace", "refresh", "max_studies", "citation_detail"}
    )
    data["query"] = re.sub(r"\s+", " ", request.query.strip())
    data["_model"] = model or Settings.model_fields["model_planner"].default
    data["_static"] = _static_fingerprint()
    data["_year"] = (today or dt.date.today()).year
    blob = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return f"{PLAN_CACHE_VERSION}:{hashlib.sha256(blob.encode()).hexdigest()}"


def cacheable(request: VisualizeRequest, plan: QueryPlan | None = None) -> bool:
    """Patient requests (profile or patient persona) are never cached (privacy)."""
    if request.patient_profile is not None or request.persona == "patient":
        return False
    return plan is None or plan.persona is not Persona.PATIENT


# ---------------------------------------------------------------------------------------- entry point
def _is_schema_failure(exc: Exception) -> bool:
    if isinstance(exc, ValidationError):
        return True
    cause = exc.__cause__
    return isinstance(cause, ValidationError) or "no parsable output" in str(exc) or "ValidationError" in str(exc)


async def _call(
    llm: LLM, model: str, request: VisualizeRequest, ex: Extraction, settings: Settings, deadline: float | None = None
) -> LLMPlan:
    timeout = current_budget().timeout_for(settings.llm_timeout_s)
    if deadline is not None:
        timeout = max(0.1, min(timeout, deadline - time.monotonic()))
    return await llm.structured(
        agent="planner",
        model=model,
        system=system_prompt(),
        user=build_user_message(request, ex),
        schema=LLMPlan,
        max_output_tokens=900,
        effort="low",
        timeout_s=timeout,
    )


def planner_deadline_s(settings: Settings) -> float:
    """Seconds the planner may spend on LLM calls: below the orchestrator's planner timeout
    (``min(remaining - 3, llm_timeout_s + 4)``) so the rule plan is returned before that timeout fires."""
    return max(0.2, min(settings.llm_timeout_s + 2.0, current_budget().remaining_s() - 4.0))


async def _llm_plan(
    llm: LLM,
    router: ModelRouter,
    model: str,
    request: VisualizeRequest,
    ex: Extraction,
    settings: Settings,
    info: dict,
    deadline: float,
) -> tuple[LLMPlan | None, bool]:
    """First LLM call plus at most one escalation, never past ``deadline`` (``time.monotonic()``)."""
    result: LLMPlan | None = None
    escalated = False
    try:
        result = await _call(llm, model, request, ex, settings, deadline)
        need = router.should_escalate(confidence=result.intent_confidence)
        failure = "low_confidence" if need else None
    except LLMUnavailable:
        raise
    except (LLMError, ValidationError) as exc:
        if not _is_schema_failure(exc):
            raise
        need, failure = True, "schema_failure"
    esc = router.escalation_model(model) if need else None
    if esc and current_budget().can_call_llm() and deadline - time.monotonic() >= ESCALATION_MIN_S:
        escalated = True
        info.update(escalated_to=esc, escalation_reason=failure)
        try:
            better = await _call(llm, esc, request, ex, settings, deadline)
            if result is None or better.intent_confidence >= result.intent_confidence:
                result = better
        except (LLMError, ValidationError) as exc:
            info["escalation_error"] = type(exc).__name__
    elif esc:
        info["escalation_skipped"] = "no time left"
    return result, escalated


async def plan_query(
    request: VisualizeRequest,
    *,
    settings: Settings,
    llm: LLM,
    db=None,  # noqa: ANN001
) -> tuple[QueryPlan, list[ResolvedEntity]]:
    """Plan a request. Never raises for normal input; falls back to the rule planner.

    ``db`` (a :class:`ctviz.store.db.Database`, optional) enables the exact-normalized plan cache.
    """
    rules, ex = rule_plan_with_extraction(request)
    entities = entities_of(ex)
    tr = current_trace()
    with tr.step("plan") as info:
        router = ModelRouter(settings)
        model = router.model_for("planner", complex_query=is_complex(request.query))
        info["model"] = model
        use_cache = db is not None and cacheable(request)
        key = plan_cache_key(request, model=model)
        if use_cache and not request.refresh:
            try:
                hit = await db.cache_get(PLAN_CACHE_NS, key)
            except Exception:  # cache is best-effort
                hit = None
            if hit:
                plan = QueryPlan.model_validate({**hit["plan"], "parser": "cache"})
                info.update(parser="cache", status="cached")
                return plan, [ResolvedEntity.model_validate(e) for e in hit.get("entities", [])]

        # One overall deadline for the LLM work (first call + optional escalation), safely inside the
        # orchestrator's planner timeout: a slow or failing LLM must end in the RULE plan, never in a
        # last-resort plan built from structured fields only.
        deadline = planner_deadline_s(settings)
        info["deadline_s"] = round(deadline, 2)
        try:
            result, escalated = await asyncio.wait_for(
                _llm_plan(llm, router, model, request, ex, settings, info, time.monotonic() + deadline),
                timeout=deadline,
            )
        except TimeoutError:
            info.update(parser="rules", fallback_reason="timeout")
            return rules, entities
        except (LLMError, ValidationError) as exc:
            info.update(parser="rules", fallback_reason=type(exc).__name__)
            return rules, entities

        if result is None:
            info.update(parser="rules", fallback_reason="schema_failure")
            return rules, entities
        validator = PlanValidator(request, ex, rules)
        try:
            plan = validator.repair(result)
        except Exception as exc:  # a repair bug must not cost the user the (valid) rule plan
            log.warning("plan repair failed (%s); using the rule plan", type(exc).__name__)
            info.update(parser="rules", fallback_reason=f"repair_error:{type(exc).__name__}")
            return rules, entities
        if validator.repairs:
            info["repairs"] = validator.repairs[:10]
        info.update(parser="llm", escalated=escalated, intent=plan.intent.value, rule_intent=rules.intent.value)
        seen = {(e.kind, e.resolved.lower()) for e in entities}
        for e in validator.extra_entities:
            if (e.kind, e.resolved.lower()) not in seen:
                entities.append(e)
                seen.add((e.kind, e.resolved.lower()))
        if use_cache and cacheable(request, plan):
            try:
                await db.cache_set(
                    PLAN_CACHE_NS,
                    key,
                    {"plan": plan.model_dump(mode="json"), "entities": [e.model_dump(mode="json") for e in entities]},
                    settings.ttl_plan_s,
                )
            except Exception:
                log.debug("plan cache write failed", exc_info=True)
        return plan, entities
