"""When the service must NOT draw a chart: subjective verdicts and unverifiable entities.

Two outcomes replace the chart (``VisualizeResponse.status``):

* ``unanswerable`` — the question asks for a judgment the registry cannot support ("which drug is safest",
  "is X safe", "should I take X"). Registry records report outcomes trial by trial, for different
  populations, doses and follow-up times, so ranking drugs from them would be misleading. The planner LLM
  may flag these (``plan.answerability``), but only :func:`is_subjective` decides: the refusal fires
  when this deterministic check matches, never on the LLM's flag alone (a valid question such as "overall
  survival across pembrolizumab trials" must not be refused because the planner misread it).
* ``needs_clarification`` — an entity named in the question matches nothing in ClinicalTrials.gov (or no
  entity / no trial was named where one is required). The search is never widened silently.

Every message here is a fixed template filled with grounded names; no LLM text reaches the response.
"""

from __future__ import annotations

import re

from ctviz.api.schemas.plan import QueryPlan
from ctviz.api.schemas.response import Clarification

# Verdict words: ranking or judging treatments, or asking for personal medical advice.
# Standard endpoint / comparator / method names are excluded (any hyphen/space variant): "best overall
# response", "best objective response", "best supportive care", "best-corrected visual acuity",
# "worst-case imputation", "non-inferior to". "best available ..." stays a verdict.
_S = r"[\s-]+"  # separator between words of an excluded term
_BEST_OK = rf"known|practice|studied|(?:(?:overall|objective){_S})?response|supportive{_S}care|corrected"
_SUBJECTIVE = re.compile(
    r"\b(safest|safer|least safe|most dangerous|more dangerous|riskiest|"
    r"most effective|more effective|less effective|least effective|works? best|works? better|"
    rf"best(?!{_S}(?:{_BEST_OK})\b)|"
    r"(?:better|worse) than|(?:is|are|works?|one) (?:\w+ )?(?:better|worse)|"
    rf"worst(?!{_S}case\b)|(?<!non-)(?<!non)(?<!non )(?:superior|inferior) to|"
    r"should i (?:take|use|stop|switch|try|start|join|enrol+)|"
    r"do you recommend|would you recommend|recommended for me)\b"
    # ranking: "which drug has the best overall response rate" (an endpoint name after "best" is still a ranking)
    r"|\bwhich (?:\w+ ){0,2}(?:drugs?|treatments?|therap(?:y|ies)|medications?|medicines?|regimens?)\b"
    r"[^?.]{0,40}\bbest\b"
    r"|\bhow (?:safe|dangerous|harmful|effective|good) (?:is|are)\b"
    r"|\bis (?:it|this|that|[\w\-]+(?: [\w\-]+){0,3}) (?:drug |medication |treatment |trial |study )?"
    r"(?:safe|dangerous|harmful|good|bad|effective|worth it)\b"
    r"|\b(?:good|bad) (?:drug|medication|medicine|treatment)\b",
    re.IGNORECASE,
)


def is_subjective(text: str) -> bool:
    """True when the question asks for a safety/efficacy verdict or personal advice."""
    return _SUBJECTIVE.search(text) is not None


def _names(plan: QueryPlan, kind: str) -> list[str]:
    field = {"drug": "drugs", "condition": "conditions"}[kind]
    return list(dict.fromkeys(v for c in plan.cohorts for v in getattr(c, field)))


def refusal(plan: QueryPlan) -> Clarification:
    """Clarification for a subjective question, with answerable alternatives built from the named entities."""
    drugs, conds = _names(plan, "drug"), _names(plan, "condition")
    for_cond = f" for {conds[0]}" if conds else ""
    suggestions: list[str] = []
    for d in drugs[:2]:
        suggestions.append(f"What serious adverse events were reported in completed {d} trials{for_cond}?")
    if len(drugs) >= 2:
        suggestions.append(f"Compare trial phases for {drugs[0]} vs {drugs[1]}{for_cond}")
    if drugs:
        suggestions.append(f"How many {drugs[0]} trials{for_cond} started each year?")
    if not drugs and conds:
        suggestions.append(f"What are the most common interventions in {conds[0]} trials?")
    suggestions.append("Show the adverse events reported for one trial (give its NCT ID, e.g. NCT03548935)")
    return Clarification(
        reason="subjective_question",
        message=(
            "ClinicalTrials.gov cannot answer which treatment is safest, best or right for someone. Registry "
            "records report outcomes and adverse events trial by trial, for different patients, doses and "
            "follow-up times, so ranking drugs from them would be misleading, and no single number in the "
            "registry measures 'safety'. A doctor can weigh that evidence for a specific person. "
            "I can show what the registry does record instead."
        ),
        suggestions=list(dict.fromkeys(suggestions))[:4],
    )


def unmatched(terms: list[str], close: dict[str, list[str]] | None = None) -> Clarification:
    """Clarification for named entities that matched no ClinicalTrials.gov record.

    ``terms`` is deduplicated case-insensitively, first spelling kept (one term can be probed as both a drug and a
    condition)."""
    seen: set[str] = set()
    terms = [t for t in terms if not (t.casefold() in seen or seen.add(t.casefold()))]
    quoted = ", ".join(f"'{t}'" for t in terms)
    suggestions = list(dict.fromkeys(f"Did you mean '{m}'?" for t in terms for m in (close or {}).get(t, [])[:2]))
    suggestions.append(
        "Check the spelling, or use the generic drug name or the official condition name "
        "(e.g. 'type 2 diabetes', 'pembrolizumab')."
    )
    return Clarification(
        reason="unrecognized_entity",
        message=(
            f"I could not find any ClinicalTrials.gov trials for {quoted}, so I stopped instead of answering "
            "about a different set of trials. Can you check the name?"
        ),
        unmatched_terms=terms,
        suggestions=suggestions,
    )


def no_entity() -> Clarification:
    return Clarification(
        reason="no_entity",
        message="Which drug, condition, sponsor or trial should I look at?",
        suggestions=[
            "How many trials for pembrolizumab started each year since 2015?",
            "Which countries have the most recruiting trials for type 2 diabetes?",
        ],
    )


def trial_required(options: list[dict[str, str]], label: str | None) -> Clarification:
    about = f" for {label}" if label else ""
    return Clarification(
        reason="trial_required",
        message=(
            f"This question is about one specific trial. Pick one of the matching trials{about} and ask again "
            "with its NCT ID."
            if options
            else "This question is about one specific trial. Please give its NCT ID (e.g. NCT03548935)."
        ),
        options=options,
    )
