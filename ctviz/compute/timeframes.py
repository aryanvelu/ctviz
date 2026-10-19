"""Outcome time frame / duration text -> relative day offsets (pure, deterministic).

Conversions: hour=1/24, day=1, week=7, month=30.4375, year=365.25 days. Every parsed point keeps the
character span of the text it came from so callers can cite it verbatim.

Deliberately not converted (callers add an APPROXIMATED note): time frames counted in treatment
cycles ("Day 1 of each 21-day cycle" is cycle-relative and the cycle count is unknown), offsets from
the last dose ("30 days after last dose") and calendar dates ("up to September 2016").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

DAYS_PER_UNIT = {"hour": 1 / 24, "day": 1.0, "week": 7.0, "month": 30.4375, "year": 365.25}
_ALIASES = {"hr": "hour", "h": "hour", "wk": "week", "mo": "month", "yr": "year"}
_NUM = r"\d+(?:\.\d+)?"
_SNUM = r"-?\d+(?:\.\d+)?"  # signed: "Day -7" (screening, before baseline)
_SEP = r"\s*(?:,\s*and|,|\band\b|\bor\b|&)\s*"

Kind = Literal["baseline", "point", "range_start", "range_end", "up_to"]


@dataclass(frozen=True, slots=True)
class TimePoint:
    """A relative day parsed from a time frame string (``day`` may be negative for screening)."""

    day: float
    kind: Kind
    start: int  # char span of the supporting text
    end: int
    unit: str
    value: float

    @property
    def label(self) -> str:
        """Short human label, e.g. 'Baseline', 'Week 68', 'Up to 21 months'."""
        if self.kind == "baseline":
            return "Baseline"
        v = f"{self.value:g}"
        unit = self.unit if abs(self.value) == 1 else f"{self.unit}s"
        if self.kind == "up_to":
            return f"Up to {v} {unit}"
        return f"{self.unit.capitalize()} {v}"


def _unit(raw: str) -> str:
    raw = raw.lower()
    if raw.endswith("s") and len(raw) > 1:
        raw = raw[:-1]
    return _ALIASES.get(raw, raw)


def _days(value: float, unit: str) -> float:
    return round(value * DAYS_PER_UNIT[unit], 2)


_CYCLE = re.compile(r"\bcycles?\b", re.IGNORECASE)
_BASELINE = re.compile(r"\b(baseline|randomi[sz]ation|day\s*0)\b", re.IGNORECASE)
# "Week 0 to Week 68", "Day 1 - 28", "weeks 4 through 12", "Day -7 to Day 28"
_RANGE = re.compile(
    rf"\b(?P<u1>hour|day|week|month|year)s?\s+(?P<a>{_SNUM})\s*(?:to|-|–|through|until)\s*"
    rf"(?:(?P<u2>hour|day|week|month|year)s?\s+)?(?P<b>{_NUM})\b",
    re.IGNORECASE,
)
# "Week 68", "Weeks 5, 7, 11 and 157", "Hour 24", "Day -7"
_UNIT_FIRST = re.compile(rf"\b(?P<u>hour|day|week|month|year)s?\s+(?P<nums>{_SNUM}(?:{_SEP}{_NUM})*)\b", re.IGNORECASE)
# "24 months", "6 and 12 months", "0.5, 1, 2, 4 hours", "~1.6 years", "698h"
_NUM_FIRST = re.compile(
    rf"(?<![\w.])(?P<nums>{_NUM}(?:{_SEP}{_NUM})*)\s*-?\s*(?P<u>hours?|hrs?|h|days?|weeks?|wks?|months?|mo|"
    rf"years?|yrs?)\b",
    re.IGNORECASE,
)
_UP_TO = re.compile(r"\b(up\s+to|within|through|until|by)\s+(?:~|approx(?:imately|\.)?\s+|about\s+)?$", re.IGNORECASE)
_AFTER_LAST_DOSE = re.compile(r"^\s*(?:after|following|post)\s+(?:the\s+)?(?:last|final|end)", re.IGNORECASE)
_DATE_CONTEXT = re.compile(
    r"(january|february|march|april|may|june|july|august|september|october|november|"
    r"december)\s*$",
    re.IGNORECASE,
)


def mentions_cycles(text: str | None) -> bool:
    """True if a time frame is counted in treatment cycles (left unconverted by :func:`parse_timeframe`)."""
    return bool(text and _CYCLE.search(text))


def parse_timeframe(text: str | None) -> list[TimePoint]:
    """Parse a timeFrame string into relative-day points with char spans.

    Recognizes baseline/randomization (day 0), "Week N", unit-first and number-first lists
    ("Weeks 4, 8 and 12", "6 and 12 months", "0.5, 1, 2, 4 hours"), ranges ("Week 0 to Week 68"),
    hours ("Hour 24"), negative screening days ("Day -7") and "up to N" windows. Cycle-based time
    frames, last-dose offsets and calendar dates yield no points.
    """
    if not text or mentions_cycles(text):
        return []
    pts: list[TimePoint] = []
    taken: list[tuple[int, int]] = []

    def free(s: int, e: int) -> bool:
        return all(e <= a or s >= b for a, b in taken)

    def window(start: int) -> bool:
        return bool(_UP_TO.search(text[max(0, start - 30) : start].replace("~", "")))

    for m in _BASELINE.finditer(text):
        pts.append(TimePoint(0.0, "baseline", m.start(), m.end(), "day", 0.0))
        if m.group(1).lower().startswith("day"):
            taken.append((m.start(), m.end()))
    for m in _RANGE.finditer(text):
        if not free(m.start(), m.end()):
            continue
        u1 = _unit(m.group("u1"))
        u2 = _unit(m.group("u2")) if m.group("u2") else u1
        a, b = float(m.group("a")), float(m.group("b"))
        pts.append(TimePoint(_days(a, u1), "range_start", m.start(), m.end("a"), u1, a))
        pts.append(TimePoint(_days(b, u2), "range_end", m.start(), m.end(), u2, b))
        taken.append((m.start(), m.end()))
    for m in _UNIT_FIRST.finditer(text):
        if not free(m.start(), m.end()):
            continue
        unit = _unit(m.group("u"))
        kind: Kind = "up_to" if window(m.start()) else "point"
        for i, nm in enumerate(re.finditer(_SNUM, m.group("nums"))):
            v = float(nm.group())
            s = m.start() if i == 0 else m.start("nums") + nm.start()
            pts.append(TimePoint(_days(v, unit), kind, s, m.start("nums") + nm.end(), unit, v))
        taken.append((m.start(), m.end()))
    for m in _NUM_FIRST.finditer(text):
        if not free(m.start(), m.end()):
            continue
        taken.append((m.start(), m.end()))
        if _AFTER_LAST_DOSE.match(text[m.end() :]) or _DATE_CONTEXT.search(text[: m.start()]):
            continue  # last-dose offsets and calendar dates are not baseline-relative
        unit = _unit(m.group("u"))
        kind = "up_to" if window(m.start()) else "point"
        for nm in re.finditer(_NUM, m.group("nums")):
            v = float(nm.group())
            # each number's span runs to the unit so the excerpt reads e.g. "12 months"
            s = m.start("nums") + nm.start()
            e = m.end() if nm.end() == len(m.group("nums")) else m.start("nums") + nm.end()
            pts.append(TimePoint(_days(v, unit), kind, s, e, unit, v))
    if any(p.kind == "baseline" for p in pts):
        # "Baseline (Day 0) through Week 76": one baseline point, not a duplicate day-0 assessment
        first_base = next(p for p in pts if p.kind == "baseline")
        pts = [p for p in pts if p.day != 0 or p is first_base]
    pts.sort(key=lambda p: (p.start, p.day))
    return pts


# -- treatment durations -----------------------------------------------------------------------

_DURATION = re.compile(
    rf"\b(?:for|over|during)\s+(?:(?P<upto>up\s+to|at\s+least|approximately|about)\s+)?(?P<n>{_NUM})\s+"
    rf"(?P<u>day|week|month|year)s?\b(?!\s+(?:after|following|post|before|prior))"
    rf"|\b(?P<n2>{_NUM})[- ](?P<u2>day|week|month|year)\s+(?:[\w-]+\s+){{0,3}}?"
    rf"(?:period|phase|treatment|course|trial|study)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Duration:
    """A treatment/period duration phrase, e.g. "for 68 weeks" or "16-week dose-escalation period"."""

    days: float
    value: float
    unit: str
    start: int
    end: int
    qualifier: str | None = None


def parse_durations(text: str | None) -> list[Duration]:
    """Find duration phrases ("for 68 weeks", "52-week maintenance period") with char spans.

    Cycle lengths ("21-day cycle", "cycle is 28 days") are not treatment durations and are skipped.
    """
    if not text:
        return []
    out = []
    for m in _DURATION.finditer(text):
        if _CYCLE.search(text[m.start() : m.end() + 12]):
            continue
        if m.group("n"):
            v, unit, q = float(m.group("n")), _unit(m.group("u")), m.group("upto")
        else:
            v, unit, q = float(m.group("n2")), _unit(m.group("u2")), None
        out.append(Duration(_days(v, unit), v, unit, m.start(), m.end(), q.lower() if q else None))
    return out


def day_to_week_label(day: float) -> str:
    """Human label for a relative day (e.g. 476 -> 'Week 68', 3 -> 'Day 3')."""
    if day == 0:
        return "Day 0"
    if day >= 7 and abs(day / 7 - round(day / 7)) < 1e-6:
        return f"Week {round(day / 7)}"
    return f"Day {round(day, 1):g}"
