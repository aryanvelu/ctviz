"""Eligibility criteria parsing and structured-rule evaluation (pure, deterministic).

``split_criteria`` turns the free-text ``eligibilityCriteria`` field into individual criteria with
exact character spans (``text[c.start:c.end] == c.text``), so every checklist row can cite a verbatim
excerpt. ``structured_rules`` evaluates the structured eligibility fields (minimum/maximum age, sex,
healthy volunteers) against an optional patient profile.

Status semantics used throughout: ``met`` = the criterion does not block the person's eligibility,
``unmet`` = it does, ``unknown`` = not evaluated, ``needs_review`` = evaluated but not confidently.
A wrong ``met`` is the worst possible error, so every ambiguous case resolves to ``needs_review``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Section = Literal["inclusion", "exclusion", "unknown"]
Status = Literal["met", "unmet", "unknown", "needs_review"]

_QUALIFIER = r"(?:(?:key|main|principal|general|major|additional|other)\s+)?"
# "Inclusion Criteria:", "Key Exclusion Criteria (all cohorts):", "Exclusion criteria include:"
_HEADER_CRITERIA = re.compile(rf"^\s*{_QUALIFIER}(inclusion|exclusion)s?\s+criteri(?:a|on)\b[^\n]*$", re.IGNORECASE)
# "Inclusion:", "Exclusions", "Key Exclusion", "EXCLUSION -"
_HEADER_BARE = re.compile(rf"^\s*{_QUALIFIER}(inclusion|exclusion)s?\s*[:\-–]?\s*$", re.IGNORECASE)
# Headings that do not say which section follows ("Inclusion/Exclusion Criteria:", "Eligibility Criteria:")
_HEADER_AMBIGUOUS = re.compile(
    r"^\s*(?:inclusion\s*(?:/|and|&)\s*exclusion|eligibility)(?:\s+criteria)?\s*[:\-–]?\s*$", re.IGNORECASE
)
_BULLET = re.compile(r"^(\s*)(?:[*\-•·▪◦]|\d{1,3}[.)]|\(\d{1,3}\)|[a-zA-Z][.)](?=\s))\s+")
_SUBHEADING_MAX = 80


def heading_section(line: str) -> Section | None:
    """Section a heading line introduces (``unknown`` for ambiguous headings), or None if not a heading."""
    m = _HEADER_CRITERIA.match(line) or _HEADER_BARE.match(line)
    if m:
        return "inclusion" if m.group(1).lower() == "inclusion" else "exclusion"
    return "unknown" if _HEADER_AMBIGUOUS.match(line) else None


@dataclass(frozen=True, slots=True)
class Criterion:
    """One criterion with its exact span in the eligibilityCriteria string.

    ``section`` is ``unknown`` when the text has no inclusion/exclusion heading or the heading is
    ambiguous. ``parent_text`` is set for nested items (sub-bullets, or items under a parent ending in
    ':' such as "Must not have any of the following:"): their meaning depends on the parent.
    """

    section: Section
    index: int  # 1-based within its section
    text: str
    start: int
    end: int
    subheading: str | None = None
    headed: bool = True  # False when the text had no Inclusion/Exclusion heading at all
    parent_text: str | None = None
    parent_index: int | None = None


def _lines(text: str) -> list[tuple[int, int]]:
    """(start, end) of each line, end excluding the newline."""
    spans, pos = [], 0
    for line in text.split("\n"):
        spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return spans


def split_criteria(text: str | None) -> list[Criterion]:
    """Split eligibility free text into inclusion/exclusion criteria with exact char spans.

    Handles headings ("Inclusion Criteria:", "Key Exclusion Criteria", "Exclusion:", "Exclusions"),
    sub-headings such as "Main phase:", bullets (``*``, ``-``, ``•``), numbered items (``1.``, ``2)``),
    nested items (which keep their parent's text) and wrapped continuation lines. Text before the
    first heading (e.g. sponsor boilerplate) is skipped unless the text has no headings at all, in
    which case every item is reported with section ``unknown`` and ``headed=False``.
    """
    if not text or not text.strip():
        return []
    has_header = any(heading_section(text[s:e]) for s, e in _lines(text))
    section: Section | None = None if has_header else "unknown"
    subheading: str | None = None
    items: list[dict] = []
    open_item = False  # whether the next plain line may continue the previous item
    parent: dict | None = None  # current top-level item that nested items attach to
    for ls, le in _lines(text):
        line = text[ls:le]
        stripped = line.strip()
        if not stripped:
            open_item = False
            continue
        sec = heading_section(line)
        if sec is not None:
            section, subheading, open_item, parent = sec, None, False, None
            continue
        if section is None:  # preamble before the first heading
            continue
        b = _BULLET.match(line)
        if b:
            indent = len(b.group(1).expandtabs(4))
            start, end = ls + b.end(), ls + len(line.rstrip())
            if start >= end:
                continue
            # A child is indented deeper than its parent, or (flat lists) follows a bullet ending in ':' at the
            # same indent -- unless that parent already had deeper-indented children, which closes its scope.
            is_child = parent is not None and (
                indent > parent["indent"]
                or (
                    parent["colon"]
                    and parent["bullet"]
                    and indent == parent["indent"]
                    and parent.get("child_indent", indent) == indent
                )
            )
            item = {
                "section": section,
                "start": start,
                "end": end,
                "sub": subheading,
                "indent": indent,
                "parent": parent if is_child else None,
                "colon": text[start:end].endswith(":"),
                "bullet": True,
            }
            items.append(item)
            if is_child:
                parent.setdefault("child_indent", indent)
            else:
                parent = item
            open_item = True
            continue
        if stripped.endswith(":") and len(stripped) <= _SUBHEADING_MAX and not open_item:
            subheading, parent = stripped[:-1].strip(), None
            continue
        start = ls + (len(line) - len(line.lstrip()))
        end = ls + len(line.rstrip())
        continuation = line[:1].isspace() or stripped[:1].islower()
        if open_item and continuation and items and items[-1]["section"] == section:
            items[-1]["end"] = end  # wrapped continuation of the previous item
            items[-1]["colon"] = text[items[-1]["start"] : end].endswith(":")
        else:
            item = {
                "section": section,
                "start": start,
                "end": end,
                "sub": subheading,
                "indent": len(line) - len(line.lstrip()),
                "parent": None,
                "colon": stripped.endswith(":"),
                "bullet": False,
            }
            items.append(item)
            parent = item
            open_item = True
    out: list[Criterion] = []
    counters: dict[str, int] = {"inclusion": 0, "exclusion": 0, "unknown": 0}
    index_of: dict[int, int] = {}
    for it in items:
        counters[it["section"]] += 1
        index_of[id(it)] = counters[it["section"]]
        p = it["parent"]
        out.append(
            Criterion(
                section=it["section"],
                index=counters[it["section"]],
                text=text[it["start"] : it["end"]],
                start=it["start"],
                end=it["end"],
                subheading=it["sub"],
                headed=has_header,
                parent_text=text[p["start"] : p["end"]] if p else None,
                parent_index=index_of.get(id(p)) if p else None,
            )
        )
    return out


# -- structured fields -------------------------------------------------------------------------

_AGE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(year|month|week|day|hour|minute)s?\s*$", re.IGNORECASE)
_AGE_UNIT_YEARS = {
    "year": 1.0,
    "month": 1 / 12,
    "week": 7 / 365.25,
    "day": 1 / 365.25,
    "hour": 1 / (365.25 * 24),
    "minute": 1 / (365.25 * 24 * 60),
}


def parse_age(value: str | None) -> float | None:
    """Registry age string ("18 Years", "6 Months", "N/A") -> age in years, or None."""
    if not value:
        return None
    m = _AGE.match(value)
    if not m:
        return None
    return float(m.group(1)) * _AGE_UNIT_YEARS[m.group(2).lower()]


def _whole_years(value: str | None) -> bool:
    """True for limits stated in whole years ("18 Years"), False for "18 Months", "6.5 Years"."""
    m = _AGE.match(value or "")
    return bool(m and m.group(2).lower() == "year" and float(m.group(1)).is_integer())


def age_status(profile_age: int, min_age: str | None, max_age: str | None) -> tuple[Status, str]:
    """Compare a whole-year profile age with registry limits.

    A profile age ``a`` means the true age lies in ``[a, a + 1)``. Whole-year limits are inclusive
    (the registry convention: "79 Years" admits anyone aged 79). Sub-year limits ("18 Months") are
    decided only when the whole year is unambiguous; otherwise the result is ``needs_review``.
    """
    lo, hi = parse_age(min_age), parse_age(max_age)
    if (min_age and lo is None) or (max_age and hi is None):
        return "needs_review", "The registry age limit could not be parsed."
    a = profile_age
    if lo is not None:
        if a + 1 <= lo or (_whole_years(min_age) and a < lo):
            return "unmet", f"Age {a} is below the minimum of {min_age}."
        if a < lo:
            return "needs_review", f"Age {a} is close to the minimum of {min_age}; exact age needed."
    if hi is not None:
        if a > hi:
            return "unmet", f"Age {a} is above the maximum of {max_age}."
        if not _whole_years(max_age) and a + 1 > hi:
            return "needs_review", f"Age {a} is close to the maximum of {max_age}; exact age needed."
    return "met", f"Age {a} is within the registered age range."


@dataclass(frozen=True, slots=True)
class StructuredRule:
    """A rule-evaluated criterion from structured eligibility fields."""

    key: Literal["age", "sex", "healthy_volunteers"]
    text: str
    status: Status
    reason: str
    fields: tuple[str, ...] = field(default_factory=tuple)  # P keys that support this rule


def structured_rules(
    *,
    min_age: str | None,
    max_age: str | None,
    sex: str | None,
    healthy_volunteers: bool | None,
    profile_age: int | None = None,
    profile_sex: str | None = None,
    profile_conditions: list[str] | None = None,
) -> list[StructuredRule]:
    """Evaluate structured eligibility fields against an optional profile (deterministic rules)."""
    rules: list[StructuredRule] = []
    if min_age or max_age:
        if min_age and max_age:
            text = f"Age: {min_age} to {max_age}"
        elif min_age:
            text = f"Age: {min_age} or older"
        else:
            text = f"Age: up to {max_age}"
        fields = tuple(k for k, v in (("min_age", min_age), ("max_age", max_age)) if v)
        if profile_age is None:
            status, reason = "unknown", "No age given in the patient profile."
        else:
            status, reason = age_status(profile_age, min_age, max_age)
        rules.append(StructuredRule("age", text, status, reason, fields))
    if sex:
        su = sex.upper()
        if su == "ALL":
            rules.append(StructuredRule("sex", "Sex: all", "met", "The trial is open to all sexes.", ("sex",)))
        else:
            want = {"FEMALE": "female", "MALE": "male"}.get(su)
            if profile_sex is None:
                status, reason = "unknown", "No sex given in the patient profile."
            elif want is None or profile_sex == "other":
                status, reason = "needs_review", "Sex eligibility needs confirmation by the study team."
            elif profile_sex == want:
                status, reason = "met", f"The trial enrolls {want} participants."
            else:
                status, reason = "unmet", f"The trial enrolls only {want} participants."
            rules.append(StructuredRule("sex", f"Sex: {su.lower()} only", status, reason, ("sex",)))
    if healthy_volunteers is not None:
        if healthy_volunteers:
            rules.append(
                StructuredRule(
                    "healthy_volunteers",
                    "Accepts healthy volunteers",
                    "met",
                    "Healthy volunteers may take part.",
                    ("healthy_volunteers",),
                )
            )
        elif profile_conditions:
            # Reporting a condition does not show it is the condition studied: never a final 'met'.
            rules.append(
                StructuredRule(
                    "healthy_volunteers",
                    "Does not accept healthy volunteers",
                    "needs_review",
                    "Participants must have the studied condition; the study team confirms whether yours qualifies.",
                    ("healthy_volunteers",),
                )
            )
        else:
            rules.append(
                StructuredRule(
                    "healthy_volunteers",
                    "Does not accept healthy volunteers",
                    "unknown",
                    "Participants must have the studied condition.",
                    ("healthy_volunteers",),
                )
            )
    return rules


def mentions_any(text: str, terms: list[str]) -> list[str]:
    """Terms (case-insensitive, word-boundary) that appear in ``text``."""
    hits = []
    for t in terms:
        t = t.strip()
        if len(t) < 3:
            continue
        if re.search(rf"(?<!\w){re.escape(t)}", text, re.IGNORECASE):
            hits.append(t)
    return hits
