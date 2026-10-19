"""Sentence splitting with char offsets and keyword-category scanning over study text fields.

Used by rationale / safety / cost / visit / medication intents to surface verbatim, citable
sentences. Pure and deterministic: a hit is a sentence span plus the keyword that matched.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from ctviz.core.provenance import Study
from ctviz.ctgov.fields import P

MAX_SENTENCE = 300

CATEGORIES: dict[str, re.Pattern[str]] = {
    "rationale": re.compile(
        r"\b(hypothes\w*|mechanism\w*|rationale|is thought to|believed? to|designed to|may (?:help|improve|reduce|"
        r"prevent|slow|increase|lower)|expected to|aims? (?:to|of)|purpose of (?:this|the) (?:study|trial)|"
        r"objective|works? by|inhibit\w*|agonist|antagonist|block\w*|target\w*)\b",
        re.IGNORECASE,
    ),
    "safety": re.compile(
        r"\b(rescue|discontinu\w*|withdraw\w*|stop(?:ped|ping)? (?:the )?(?:study|treatment|trial)|monitor\w*|"
        r"data (?:and )?(?:safety )?monitoring|dsmb|\bdmc\b|safety (?:follow[- ]up|review|committee)|worsen\w*|"
        r"(?:disease )?progression|unacceptable toxicit\w*|dose (?:reduction|interruption|modification|delay)\w*|"
        r"adverse event|serious adverse|emergency|hospitali[sz]\w*)\b",
        re.IGNORECASE,
    ),
    "costs": re.compile(
        r"\b(reimburs\w*|compensat\w*|stipend\w*|costs?|insurance|insurer|travel\w*|parking|free of charge|"
        r"at no (?:cost|charge)|payment\w*|paid|out[- ]of[- ]pocket)\b",
        re.IGNORECASE,
    ),
    "visits": re.compile(
        r"\b(visits?|clinic\w*|hospital\w*|overnight|in-?patient|out-?patient|admission|admitted|"
        r"phone calls?|telephone|site (?:visit|attendance)|on[- ]site|home visits?|stay\w* (?:at|in))\b",
        re.IGNORECASE,
    ),
    "medication": re.compile(
        r"\b(medications?|medicines?|drugs?|concomitant|prior (?:\w+ )?(?:therapy|treatment)|washout|"
        r"corticosteroids?|steroids?|immunosuppress\w*|anticoagula\w*|insulin|inhibitors?|agonists?|"
        r"supplements?|vaccin\w*|chemotherap\w*|radiotherap\w*|investigational (?:agent|drug|product|device)|"
        r"treatment with|treated with|therapy with|taking|receiv\w+ (?:any )?(?:prior )?(?:treatment|therapy))\b",
        re.IGNORECASE,
    ),
    "biomarker": re.compile(
        r"\b(biomarkers?|pd-?l1|ctdna|hba1c|glycated|egfr|alk|her2|brca\w*|kras|mutation\w*|expression|"
        r"immunohistochem\w*|\bihc\b|genotyp\w*|c-reactive|crp|ldl|hdl|cholesterol|triglycerides?|"
        r"glucose|insulin resistance|troponin|psa|tumou?r (?:tissue|sample)|tissue for)\b",
        re.IGNORECASE,
    ),
    "doctor": re.compile(
        r"\b(personal physician|primary care|your (?:own |regular )?doctor|regular doctor|family doctor|"
        r"treating physician|referring physician|general practitioner)\b",
        re.IGNORECASE,
    ),
}

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])|\n+")


@dataclass(frozen=True, slots=True)
class Hit:
    """A sentence in a field that matched a category keyword."""

    category: str
    field_path: str
    start: int
    end: int
    term: str
    text: str


def split_sentences(text: str | None) -> list[tuple[int, int]]:
    """(start, end) spans of sentences/lines, whitespace- and bullet-trimmed, each <= 300 chars."""
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in [*_SPLIT.finditer(text), None]:
        end = m.start() if m else len(text)
        s, e = pos, end
        while s < e and (text[s].isspace() or text[s] in "*-•·"):
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        while e - s > MAX_SENTENCE:  # chunk very long sentences at a word boundary
            cut = text.rfind(" ", s, s + MAX_SENTENCE)
            cut = cut if cut > s else s + MAX_SENTENCE
            spans.append((s, cut))
            s = cut
            while s < e and text[s].isspace():
                s += 1
        if e > s:
            spans.append((s, e))
        if m:
            pos = m.end()
    return spans


def scan_text(text: str | None, field_path: str, categories: Iterable[str]) -> list[Hit]:
    """Sentences of ``text`` matching any of ``categories`` (first matching category wins)."""
    cats = [(c, CATEGORIES[c]) for c in categories]
    hits = []
    for s, e in split_sentences(text):
        sentence = text[s:e]  # type: ignore[index]
        for name, rx in cats:
            m = rx.search(sentence)
            if m:
                hits.append(Hit(name, field_path, s, e, m.group(0), sentence))
                break
    return hits


def study_text_fields(
    study: Study, *, eligibility: bool = False, outcomes: bool = False, arms: bool = True, locations: bool = False
) -> list[tuple[str, str]]:
    """(field_path, text) for the narrative text fields of a study, in a stable order."""
    out: list[tuple[str, str]] = []
    for key in ("brief_summary", "detailed_description"):
        v = study.get(key)
        if isinstance(v, str) and v.strip():
            out.append((P[key], v))
    if arms:
        for _, item, path in study.items("arm_groups"):
            if isinstance(item, dict) and isinstance(item.get("description"), str):
                out.append((f"{path}.description", item["description"]))
        for _, item, path in study.items("interventions"):
            if isinstance(item, dict) and isinstance(item.get("description"), str):
                out.append((f"{path}.description", item["description"]))
    if eligibility:
        v = study.get("eligibility_criteria")
        if isinstance(v, str) and v.strip():
            out.append((P["eligibility_criteria"], v))
    if outcomes:
        for key in ("primary_outcomes", "secondary_outcomes"):
            for _, item, path in study.items(key):
                if isinstance(item, dict) and isinstance(item.get("description"), str):
                    out.append((f"{path}.description", item["description"]))
    if locations:
        for _, item, path in study.items("locations"):
            if isinstance(item, dict) and isinstance(item.get("facility"), str):
                out.append((f"{path}.facility", item["facility"]))
    return out


def scan_study(study: Study, categories: Iterable[str], **field_opts: bool) -> list[Hit]:
    """Scan a study's text fields for category hits (see :func:`study_text_fields` for options)."""
    cats = list(categories)
    hits: list[Hit] = []
    seen: set[tuple[str, str]] = set()
    for path, text in study_text_fields(study, **field_opts):
        for h in scan_text(text, path, cats):
            # Many locations repeat the same boilerplate; keep the first occurrence of each sentence.
            if (h.category, h.text) in seen:
                continue
            seen.add((h.category, h.text))
            hits.append(h)
    return hits
