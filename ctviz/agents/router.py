"""ModelRouter: which OpenAI model each agent task uses, and when to escalate.

Defaults (from :class:`~ctviz.config.Settings`): nano for planner / synthesis / extraction, mini for
analysis and escalations, the large model only for ``quality="max"``. Escalation triggers:
low self-reported confidence (< 0.6), repeated schema failures, or a "complex" query pre-classifier.
"""

from __future__ import annotations

import re
from typing import Literal

from ctviz.config import Settings

Task = Literal["planner", "synthesis", "extract", "analysis", "retrieval", "escalation"]
Quality = Literal["fast", "default", "max"]

CONFIDENCE_FLOOR = 0.6
MAX_SCHEMA_FAILURES = 2

_COMPARE = re.compile(r"\b(vs\.?|versus|compare|comparing|between)\b", re.I)


def is_complex(query: str) -> bool:
    """Cheap pre-classifier for questions the nano model tends to mis-plan.

    Complex = very long, several questions at once, or a comparison combined with extra constraints.
    """
    words = len(query.split())
    questions = query.count("?")
    constraints = len(re.findall(r"\b(since|between|phase|recruiting|in \w+ only|excluding|except)\b", query, re.I))
    return words > 45 or questions >= 3 or (bool(_COMPARE.search(query)) and constraints >= 3)


class ModelRouter:
    """Map an agent task to a model name and decide escalations."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def model_for(self, task: Task, *, quality: Quality = "default", complex_query: bool = False) -> str:
        """Model for ``task``. ``complex_query`` starts nano tasks on the escalation model directly."""
        s = self.settings
        if quality == "max":
            return s.model_max
        base = {
            "planner": s.model_planner,
            "synthesis": s.model_synthesis,
            "extract": s.model_extract,
            "analysis": s.model_analysis,
            "retrieval": s.model_planner,
            "escalation": s.model_escalation,
        }[task]
        if complex_query and task in ("planner", "retrieval") and quality != "fast":
            return s.model_escalation
        return base

    def escalation_model(self, current: str) -> str | None:
        """Model to retry with, or ``None`` when already at (or above) the escalation tier."""
        s = self.settings
        return None if current in (s.model_escalation, s.model_max) else s.model_escalation

    @staticmethod
    def should_escalate(
        *, confidence: float | None = None, schema_failures: int = 0, verified_quote_ratio: float | None = None
    ) -> bool:
        """True when the last result is not trustworthy enough to use as-is."""
        if confidence is not None and confidence < CONFIDENCE_FLOOR:
            return True
        if schema_failures >= 1:
            return True
        return verified_quote_ratio is not None and verified_quote_ratio < 0.5
