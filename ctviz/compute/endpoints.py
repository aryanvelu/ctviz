"""Regex classifier for outcome measures: safety | dose | efficacy | pk | pd | biomarker | qol | other.

Rules are ordered from most to least specific; the first matching rule wins and the matched term is
returned so the classification can be explained (and cited via the measure text).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CATEGORY_ORDER = ["efficacy", "safety", "dose", "pk", "pd", "biomarker", "qol", "other"]
CATEGORY_LABELS = {
    "efficacy": "Efficacy",
    "safety": "Safety",
    "dose": "Dose finding",
    "pk": "Pharmacokinetics (PK)",
    "pd": "Pharmacodynamics (PD)",
    "biomarker": "Biomarker",
    "qol": "Quality of life / PRO",
    "other": "Other",
}

_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "dose",
        re.compile(
            r"dose[- ]limiting|maximum tolerated dose|\bmtd\b|recommended (?:phase 2 )?dose|\brp2d\b|"
            r"dose[- ](?:escalation|finding|selection|response)|optimal dose",
            re.I,
        ),
    ),
    (
        "pk",
        re.compile(
            r"pharmacokinetic|\bpk\b|\bc ?max\b|\bt ?max\b|\bauc\w*|half[- ]life|\bt1/2\b|clearance|"
            r"volume of distribution|(?:plasma|serum|trough) concentration|concentration maximum|"
            r"area under the (?:concentration|curve)|exposure",
            re.I,
        ),
    ),
    (
        "pd",
        re.compile(
            r"pharmacodynamic|\bpd\b(?!-?l)|receptor occupancy|target engagement|"
            r"immunogenicity|anti-?drug antibod",
            re.I,
        ),
    ),
    ("efficacy", re.compile(r"surviv|objective response|response rate|remission|time to progression", re.I)),
    (
        "biomarker",
        re.compile(
            r"biomarker|pd-?l1|ctdna|\bhba1c\b|glycated|fasting (?:plasma )?glucose|"
            r"\bldl\b|\bhdl\b|cholesterol|triglycerid|c-reactive|\bcrp\b|insulin|"
            r"\bexpression\b|mutation|gene|tumou?r mutational|\btmb\b|lipid|lipoprotein|leptin|"
            r"fatty acid|"
            r"plasminogen|adiponectin|cytokine|interleukin",
            re.I,
        ),
    ),
    (
        "qol",
        re.compile(
            r"quality of life|\bqol\b|\bhrqol\b|sf-36|sf-12|eortc|qlq|eq-5d|iwqol|\bfact-\w|"
            r"patient[- ]reported|\bpros?\b|questionnaire|well-being|physical function",
            re.I,
        ),
    ),
    (
        "safety",
        re.compile(
            r"adverse|\baes?\b|\bsaes?\b|safety|toxicit|tolerab|side effect|"
            r"laboratory|vital sign|amylase|lipase|calcitonin|pulse|heart rate|electrocardiogram|\becg\b|"
            r"discontinu\w* (?:\w+ )*due to|death",
            re.I,
        ),
    ),
    (
        "efficacy",
        re.compile(
            r"surviv|response|remission|progression|recurrence|relapse|efficacy|"
            r"body weight|weight (?:loss|reduction)|waist|blood pressure|reduction|improvement|"
            r"change (?:from baseline )?in|time to|duration of|rate of|percentage of participants|"
            r"number of participants who achieve|event[- ]free|cure|clearance of",
            re.I,
        ),
    ),
]


@dataclass(frozen=True, slots=True)
class EndpointClass:
    """Classification result: category plus the term (and its span in the measure) that triggered it."""

    category: str
    term: str | None = None
    start: int | None = None
    end: int | None = None


def classify_endpoint(measure: str | None, description: str | None = None) -> EndpointClass:
    """Classify an outcome by its measure title (description used only if the title is inconclusive)."""
    for text, use_span in ((measure or "", True), (description or "", False)):
        for cat, rx in _RULES:
            m = rx.search(text)
            if m:
                return EndpointClass(cat, m.group(0), m.start() if use_span else None, m.end() if use_span else None)
    return EndpointClass("other")
