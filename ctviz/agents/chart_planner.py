"""Chart planner agent: after retrieval, pick the visualization that best answers the question for THIS data.

The router/planner decides the intent before any trial is seen. Once the cohort is in, a profile of it
(how many trials, how many distinct phases / statuses / sponsors / drugs / countries / start years, how
concentrated they are, how many drug combinations exist) says which charts would actually be informative:
a phase bar for a cohort with one phase says nothing, a 3,000-drug network needs a threshold, a vague
question ("tell me about obesity trials") deserves the most informative overview rather than a default.

One validated LLM call (``model_analysis``) chooses from a menu of options the code has already checked
are buildable from the records in hand (same field set, no new retrieval):

* ``intent``     only when the question does not name a view itself (no intent keyword in the question);
* ``visualization`` one of that intent's compatible types;
* ``split``      phase or status group, only when the question mentions that dimension (never by default);
* ``top_n``      5-40.

The LLM sees counts of categories and question text, never trial records, and no LLM text reaches the
response: the pick is an enum choice, the assumption shown to the client is a template, and the model's
one-line reason is kept in the debug trace only. It is called only when there is a real choice (the question
does not name its view, so the menu holds several intents); otherwise no extra LLM call is made. Invalid or
failed picks fall back to the plan unchanged (``method="rules"``).
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ctviz.agents.llm import LLMError, LLMUnavailable
from ctviz.api.schemas.plan import Intent, QueryPlan, Scope
from ctviz.api.schemas.viz import VizType
from ctviz.core.intents import IntentSpec, load_intents
from ctviz.ctgov.fields import FieldSet, status_group
from ctviz.obs.runs import step
from ctviz.viz.base import CohortData

# Cohort views that can be re-chosen after retrieval (landscape records, no results/extraction needed).
MENU: tuple[Intent, ...] = (
    Intent.TREND_OVER_TIME,
    Intent.PHASE_DISTRIBUTION,
    Intent.STATUS_DISTRIBUTION,
    Intent.PHASE_STATUS_HEATMAP,
    Intent.SPONSOR_LANDSCAPE,
    Intent.INTERVENTION_LANDSCAPE,
    Intent.GEOGRAPHIC_DISTRIBUTION,
    Intent.ENTITY_NETWORK,
    Intent.ENROLLMENT_DISTRIBUTION,
)
# Views ClinicalTrials.gov can count exactly bucket by bucket when only page 1 was kept (registry counts).
REGISTRY_MENU = (
    Intent.TREND_OVER_TIME,
    Intent.PHASE_DISTRIBUTION,
    Intent.STATUS_DISTRIBUTION,
    Intent.GEOGRAPHIC_DISTRIBUTION,
)
SPLIT_WORDS = {"phase": r"\bphases?\b", "status_group": r"\b(status|active|ongoing|completed|finished|done)\b"}
Split = Literal["none", "phase", "status_group"]


class ChartPick(BaseModel):
    """What the LLM returns (validated against the menu before use)."""

    intent: Intent
    visualization: VizType
    split: Split = "none"
    top_n: int | None = None
    reason: str = Field(description="One sentence: why this view answers the question best for this data.")


class ChartDecision(BaseModel):
    plan: QueryPlan
    method: Literal["llm", "rules"]
    reason: str | None = None
    rejected: list[str] = Field(default_factory=list)


# -- profile ----------------------------------------------------------------------------------------
def _distinct(values: list[Any], top: int = 3) -> dict[str, Any]:
    c = Counter(v for v in values if v)
    total = sum(c.values()) or 1
    return {
        "distinct": len(c),
        "top_share": round(c.most_common(1)[0][1] / total, 2) if c else 0.0,
        "top": [k for k, _ in c.most_common(top)],
    }


def profile(cohorts: list[CohortData]) -> dict[str, Any]:
    """Category counts only (no record content beyond the top few category names)."""
    studies = [s for c in cohorts for s in c.studies]
    years = [str(s.get("start_date") or "")[:4] for s in studies]
    drugs, combos = [], 0
    for s in studies:
        names = {
            str(i.get("name")).casefold()
            for _, i, _ in s.items("interventions")
            if isinstance(i, dict) and i.get("type") in ("DRUG", "BIOLOGICAL") and i.get("name")
        }
        drugs += list(names)
        combos += any(
            isinstance(a, dict) and len(a.get("interventionNames") or []) >= 2 for _, a, _ in s.items("arm_groups")
        )
    countries = [
        str(loc.get("country"))
        for s in studies
        for _, loc, _ in s.items("locations")
        if isinstance(loc, dict) and loc.get("country")
    ]
    phases = ["/".join(s.get("phases") or []) or "NA" for s in studies]
    return {
        "trials": len(studies),
        "trials_matched": sum(c.total_count for c in cohorts),
        "cohorts": [c.label for c in cohorts],
        "counting": sorted({c.counting for c in cohorts}),
        "start_years": {
            "distinct": len({y for y in years if y.isdigit()}),
            "first": min((y for y in years if y.isdigit()), default=None),
            "last": max((y for y in years if y.isdigit()), default=None),
        },
        "phases": _distinct(phases),
        "status_groups": _distinct([status_group(s.get("status")) for s in studies]),
        "lead_sponsors": _distinct([s.get("lead_sponsor") for s in studies]),
        "sponsor_classes": _distinct([s.get("lead_sponsor_class") for s in studies]),
        "drugs": _distinct(drugs),
        "trials_with_combination_arms": combos,
        "countries": _distinct(list(set(countries)) if len(studies) == 1 else countries),
    }


# -- menu + validation --------------------------------------------------------------------------------
def anchored(question: str, intent: Intent, specs: dict[Intent, IntentSpec]) -> bool:
    """The question names its view itself (an intent keyword is in the text): keep the planner's intent."""
    return any(re.search(k, question, re.I) for k in specs[intent].keywords)


def menu(plan: QueryPlan, question: str, cohorts: list[CohortData]) -> dict[str, Any]:
    specs = load_intents()
    registry = any(c.counting == "registry_counts" for c in cohorts)
    allowed = [i for i in (REGISTRY_MENU if registry else MENU) if specs[i].field_set is FieldSet.LANDSCAPE]
    if anchored(question, plan.intent, specs) or plan.intent not in allowed:
        allowed = [plan.intent] if plan.intent in MENU else []
    splits = ["none"] + ([] if registry else [k for k, rx in SPLIT_WORDS.items() if re.search(rx, question, re.I)])
    return {
        "intents": {i.value: [v.value for v in specs[i].compatible_viz] for i in allowed},
        "splits": splits,
        "current": {
            "intent": plan.intent.value,
            "visualization": (plan.preferred_visualization or specs[plan.intent].primary).value,
        },
    }


def validate_pick(pick: ChartPick, options: dict[str, Any]) -> list[str]:
    problems = []
    if pick.intent.value not in options["intents"]:
        problems.append(f"intent {pick.intent.value} not offered")
    elif pick.visualization.value not in options["intents"][pick.intent.value]:
        problems.append(f"{pick.visualization.value} does not fit {pick.intent.value}")
    if pick.split not in options["splits"]:
        problems.append(f"split {pick.split} not asked for")
    if pick.top_n is not None and not 5 <= pick.top_n <= 40:
        problems.append("top_n out of range")
    return problems


def apply_pick(plan: QueryPlan, pick: ChartPick) -> QueryPlan:
    update: dict[str, Any] = {"intent": pick.intent, "preferred_visualization": pick.visualization}
    if pick.split != "none" and pick.intent in (Intent.TREND_OVER_TIME,):
        update["series_by"] = pick.split
    elif pick.split == "status_group" and pick.intent is Intent.STATUS_DISTRIBUTION:
        update["group_by"] = "status_group"
    if pick.top_n:
        update["top_n"] = pick.top_n
    if pick.intent is not plan.intent:  # dimension choices of the old view do not carry over
        update.setdefault("group_by", None)
        update.setdefault("series_by", None)
    return plan.model_copy(update=update)


_SYSTEM = """You choose the chart that best answers a clinical-trials question for the data already retrieved.
You get the question, a profile of the retrieved trials (category counts only), the current choice and a menu
of allowed choices. Pick exactly one intent and one visualization from the menu, a split only if offered, and
top_n (5-40) if the chart ranks categories.
Rules: keep the current choice unless the profile shows it would be uninformative (e.g. one category holds
almost everything, or a network with no repeated sponsors/drugs/combinations). For broad overview questions
prefer rich multi-dimensional views (a sponsor-drug or drug-combination network, a phase x status heatmap)
over a single bar or line when the cohort has many sponsors and drugs. Reply with a one-sentence reason.
Never invent numbers."""


async def choose(
    plan: QueryPlan, question: str, cohorts: list[CohortData], *, llm: Any, settings: Any, explicit_viz: bool
) -> ChartDecision:
    """Validated LLM pick; the unchanged plan when the LLM is unavailable, slow, or picks off-menu."""
    if plan.scope is Scope.SINGLE_TRIAL or plan.nct_ids or not cohorts or llm is None:
        return ChartDecision(plan=plan, method="rules")
    options = menu(plan, question, cohorts)
    if len(options["intents"]) < 2:  # the question names its view: nothing to choose, no LLM call
        return ChartDecision(plan=plan, method="rules")
    with step("chart_planner", options=options["intents"], splits=options["splits"]) as info:
        user = json.dumps(
            {"question": question[:500], "profile": profile(cohorts), "menu": options}, ensure_ascii=False, default=str
        )
        timeout = settings.llm_timeout_s
        try:
            pick = await asyncio.wait_for(
                llm.structured(
                    agent="chart_planner",
                    model=settings.model_analysis,
                    system=_SYSTEM,
                    user=user,
                    schema=ChartPick,
                    max_output_tokens=800,
                    effort="low",
                    timeout_s=timeout,
                ),
                timeout=timeout + 1.0,
            )
            if not isinstance(pick, ChartPick):
                pick = ChartPick.model_validate(pick if isinstance(pick, dict) else pick.model_dump())
        except (TimeoutError, LLMUnavailable, LLMError, ValidationError) as exc:
            info["summary"] = {"method": "rules", "rejected": [type(exc).__name__]}
            return ChartDecision(plan=plan, method="rules", rejected=[type(exc).__name__])
        problems = validate_pick(pick, options)
        if explicit_viz and pick.visualization.value != options["current"]["visualization"]:
            problems.append("the request fixed the visualization")
        if problems:
            info["summary"] = {"method": "rules", "rejected": problems}
            return ChartDecision(plan=plan, method="rules", rejected=problems)
        info["summary"] = {
            "method": "llm",
            "intent": pick.intent.value,
            "viz": pick.visualization.value,
            "split": pick.split,
            "top_n": pick.top_n,
            "reason": pick.reason[:300],
        }  # trace only
        return ChartDecision(plan=apply_pick(plan, pick), method="llm", reason=pick.reason.strip()[:300])
