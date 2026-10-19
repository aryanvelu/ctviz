"""Pure helpers for cohort/landscape handlers: partial dates, time buckets, name normalization.

Nothing here touches provenance; handlers pair these values with ``Study.ref`` citations.
"""

from __future__ import annotations

import datetime as dt
import re
import unicodedata
from collections import Counter
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Literal

Granularity = Literal["year", "quarter", "month"]
Precision = Literal["year", "month", "day"]

_DATE_RE = re.compile(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$")
_TRADEMARKS = re.compile(r"[®™©℠]")
_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")
_PLACEBO = re.compile(r"\bplacebos?\b|\bsham\b|\bdummy\b|\bvehicle\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class PartialDate:
    """A registry date that may be 'YYYY', 'YYYY-MM' or 'YYYY-MM-DD'.

    ``value`` is the first day of the period the registry string names; ``precision`` says how much
    of it was actually given.
    """

    value: dt.date
    precision: Precision

    @property
    def iso(self) -> str:
        return self.value.isoformat()


def parse_partial_date(text: object) -> PartialDate | None:
    """Parse a ClinicalTrials.gov date string; ``None`` for missing or malformed values."""
    if not isinstance(text, str):
        return None
    m = _DATE_RE.match(text.strip())
    if not m:
        return None
    year, month, day = int(m.group(1)), m.group(2), m.group(3)
    try:
        if day is not None and month is not None:
            return PartialDate(dt.date(year, int(month), int(day)), "day")
        if month is not None:
            return PartialDate(dt.date(year, int(month), 1), "month")
        return PartialDate(dt.date(year, 1, 1), "year")
    except ValueError:
        return None


def supports_granularity(date: PartialDate, granularity: Granularity) -> bool:
    """True if the date is precise enough to be placed in a bucket of this granularity."""
    if granularity == "year":
        return True
    return date.precision in ("month", "day")


def bucket_start(date: dt.date, granularity: Granularity) -> dt.date:
    """First day of the year / quarter / month containing ``date``."""
    if granularity == "year":
        return dt.date(date.year, 1, 1)
    if granularity == "quarter":
        return dt.date(date.year, 3 * ((date.month - 1) // 3) + 1, 1)
    return dt.date(date.year, date.month, 1)


def bucket_label(start: dt.date, granularity: Granularity) -> str:
    """Readable bucket label: '2015', '2015-Q2' or '2015-04'."""
    if granularity == "year":
        return str(start.year)
    if granularity == "quarter":
        return f"{start.year}-Q{(start.month - 1) // 3 + 1}"
    return f"{start.year}-{start.month:02d}"


def next_bucket(start: dt.date, granularity: Granularity) -> dt.date:
    """Start of the bucket following ``start``."""
    if granularity == "year":
        return dt.date(start.year + 1, 1, 1)
    step = 3 if granularity == "quarter" else 1
    month = start.month - 1 + step
    return dt.date(start.year + month // 12, month % 12 + 1, 1)


def bucket_range(first: dt.date, last: dt.date, granularity: Granularity) -> list[dt.date]:
    """Every bucket start from ``first`` to ``last`` inclusive (used to zero-fill a series)."""
    out: list[dt.date] = []
    cur = bucket_start(first, granularity)
    end = bucket_start(last, granularity)
    while cur <= end:
        out.append(cur)
        cur = next_bucket(cur, granularity)
    return out


def normalize_name(name: object) -> str:
    """Grouping key for entity names: casefold, drop ®/™, punctuation and extra whitespace.

    Deliberately conservative: no fuzzy matching, so 'Novo Nordisk A/S' and 'Novo Nordisk' stay apart.
    """
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKC", name)
    text = _TRADEMARKS.sub("", text).casefold()
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def country_key(name: object) -> str:
    """Grouping key for a location country: whitespace collapsed and casefolded ('' when blank).

    Only whitespace and case are normalized, so 'United States ' and 'united states' group together
    while distinct spellings are left alone.
    """
    if not isinstance(name, str):
        return ""
    return " ".join(name.split()).casefold()


def is_placebo(name: object) -> bool:
    """True for placebo / sham / vehicle comparators (excluded from drug networks and rankings)."""
    return isinstance(name, str) and bool(_PLACEBO.search(name))


def split_arm_intervention(entry: object) -> tuple[str, str] | None:
    """Split an ``armGroups[].interventionNames`` entry like 'Drug: Nivolumab' into (type, name)."""
    if not isinstance(entry, str) or ":" not in entry:
        return None
    kind, _, name = entry.partition(":")
    name = name.strip()
    return (kind.strip(), name) if name else None


class LabelPicker:
    """Remembers the most frequent raw spelling for each normalized key (display labels)."""

    def __init__(self) -> None:
        self._counts: dict[Hashable, Counter[str]] = {}

    def add(self, key: Hashable, raw: str) -> None:
        self._counts.setdefault(key, Counter())[raw.strip()] += 1

    def label(self, key: Hashable) -> str:
        counts = self._counts.get(key)
        if not counts:
            return str(key)
        best = max(counts.items(), key=lambda kv: (kv[1], -len(kv[0]), kv[0]))
        return best[0]


def top_keys(counts: dict[Hashable, int], n: int) -> list[Hashable]:
    """Keys sorted by count desc (ties alphabetical), first ``n``."""
    return [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[:n]]


def median(values: Iterable[float]) -> float | None:
    """Median of a sequence (None when empty)."""
    vals = sorted(values)
    if not vals:
        return None
    mid = len(vals) // 2
    return float(vals[mid]) if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2


ENROLLMENT_BINS: list[tuple[int, int | None]] = [
    (0, 25),
    (25, 50),
    (50, 100),
    (100, 250),
    (250, 500),
    (500, 1000),
    (1000, 2500),
    (2500, 5000),
    (5000, 10000),
    (10000, None),
]


def enrollment_bin(count: int) -> int:
    """Index into :data:`ENROLLMENT_BINS` for a participant count (bins are [start, end))."""
    for i, (lo, hi) in enumerate(ENROLLMENT_BINS):
        if count >= lo and (hi is None or count < hi):
            return i
    return 0


def enrollment_bin_label(index: int) -> str:
    lo, hi = ENROLLMENT_BINS[index]
    return f"{lo:,}+" if hi is None else f"{lo:,}–{hi - 1:,}"
