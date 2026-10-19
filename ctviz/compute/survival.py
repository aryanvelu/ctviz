"""Approximate survival curves from reported summary statistics (no patient-level data exist).

Reported anchors: S(0)=1, S(median)=0.5 from MEDIAN measures, and landmark rates ("OS rate at 12
months"). Between consecutive anchors the curve is piecewise exponential:
``S(t) = S0 * exp(-lambda (t - t0))`` with ``lambda = ln(S0/S1) / (t1 - t0)``. Nothing is
extrapolated past the last reported anchor. Interpolated points are flagged
(``source="interpolated"``), cite only the outcome measure's title and name the two anchors they lie
between (``between``), so a handler can link them to the anchor datums (``derived_from``) instead of
repeating every anchor's citations on every interpolated point.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ctviz.compute.results import OM_PATH, measure_cells, short_endpoint
from ctviz.core.provenance import FieldRef, Study

METHOD = "piecewise exponential interpolation between reported anchors (median, landmark rates)"
ASSUMPTIONS = [
    "No patient-level time-to-event data are in the registry; only reported medians and landmark rates are used.",
    "S(0) = 1 for every arm.",
    "Constant hazard (exponential decay) between consecutive reported anchors.",
    "No extrapolation beyond the last reported anchor.",
    "Medians and landmark rates from different outcome measures of the same endpoint are combined per arm.",
]

DEFAULT_MAX_INTERPOLATED = 12  # interpolated points per series (between S(0) and the last anchor)
MIN_STEP_MONTHS = 1.0  # never denser than one interpolated point per month
_MEASURE_TITLE = re.compile(r"outcomeMeasures\[\d+\]\.title$")

_TIME_UNITS = {"day": 1 / 30.4375, "week": 7 / 30.4375, "month": 1.0, "year": 12.0}
_TIME_AFTER = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*(day|week|wk|month|mo|year|yr)s?\b", re.I)
_TIME_BEFORE = re.compile(r"\b(day|week|month|year)s?\s*(\d+(?:\.\d+)?)", re.I)
_SURV_TITLE = re.compile(
    r"surviv|\bos\b|\bpfs\b|\befs\b|\bdfs\b|progression[- ]free|event[- ]free|disease[- ]free", re.I
)


def _unit_key(u: str) -> str:
    u = u.lower()
    return {"wk": "week", "mo": "month", "yr": "year"}.get(u, u.rstrip("s"))


def parse_time_months(text: str) -> float | None:
    """'12 months' / 'Month 6' / '2-year' -> months; None if no time found."""
    if not text:
        return None
    m = _TIME_AFTER.search(text)
    if m:
        return float(m.group(1)) * _TIME_UNITS[_unit_key(m.group(2))]
    m = _TIME_BEFORE.search(text)
    if m:
        return float(m.group(2)) * _TIME_UNITS[_unit_key(m.group(1))]
    return None


def unit_to_months(unit: str) -> float | None:
    """Multiplier from a measure's unitOfMeasure ('Months', 'weeks') to months."""
    u = (unit or "").lower()
    for key, mult in _TIME_UNITS.items():
        if key in u:
            return mult
    return None


@dataclass
class Anchor:
    """A reported (or definitional) point on one arm's curve."""

    series: str
    endpoint: str
    group: str
    time: float
    survival: float
    kind: str  # start | median | landmark
    refs: list[FieldRef] = field(default_factory=list)
    ci_low: float | None = None
    ci_high: float | None = None
    time_ci_low: float | None = None
    time_ci_high: float | None = None


@dataclass
class CurvePoint:
    series: str
    endpoint: str
    group: str
    time: float
    survival: float
    source: str  # reported | interpolated
    refs: list[FieldRef]
    kind: str
    ci_low: float | None = None
    ci_high: float | None = None
    time_ci_low: float | None = None
    time_ci_high: float | None = None
    anchor_pos: int | None = None  # reported points: position in the series' anchor chain (0 = S(0) = 1)
    between: tuple[int, int] | None = None  # interpolated points: chain positions of the two anchors used


@dataclass
class SurvivalExtraction:
    anchors: dict[str, list[Anchor]]  # series -> anchors (sorted, monotone)
    not_reached: list[str]  # series whose median was 'not reached' / censored
    dropped: list[str]  # anchors dropped as non-monotone
    skipped: list[str] = field(default_factory=list)  # values not used because their meaning was unclear
    converted: list[str] = field(default_factory=list)  # event rates converted to S = 1 - rate


_NOT_SURVIVAL = re.compile(
    r"follow[- ]?up|duration of (?:treatment|therapy|exposure|response)|treatment duration|exposure|"
    r"time on (?:treatment|study|therapy)|time to (?:response|onset|treatment)|dose intensity",
    re.I,
)
_EVENT_RATE = re.compile(
    r"\b(?:who died|died|deaths?|mortality|who progressed|(?:disease )?progression or death|with an? event|"
    r"event rate)\b",
    re.I,
)
_RATE_WORD = re.compile(r"\brate\b|probab|percent|proportion|%|\bat (?:month|week|year|day)|\balive\b", re.I)
_OPEN_TIME = re.compile(r"up to|until|through|maximum|approximately|from .* to", re.I)
_PCT_UNIT = re.compile(r"percent|%", re.I)
_PROB_UNIT = re.compile(r"probab|proportion|fraction", re.I)


def is_survival_measure(title: str) -> bool:
    """Survival / event-free endpoint (not follow-up, treatment duration or exposure)."""
    t = title or ""
    return bool(_SURV_TITLE.search(t)) and not _NOT_SURVIVAL.search(t)


def is_event_rate_measure(title: str) -> bool:
    """Explicit event rate (e.g. 'Percentage of participants who died'); S = 1 - rate."""
    t = title or ""
    return bool(_EVENT_RATE.search(t)) and not _SURV_TITLE.search(t) and not _NOT_SURVIVAL.search(t)


def is_time_to_event_measure(title: str) -> bool:
    """Measures the survival purpose may use: survival endpoints or explicit event rates."""
    return is_survival_measure(title) or is_event_rate_measure(title)


def _time_expressions(text: str) -> list[float]:
    out = [float(m.group(1)) * _TIME_UNITS[_unit_key(m.group(2))] for m in _TIME_AFTER.finditer(text or "")]
    out += [float(m.group(2)) * _TIME_UNITS[_unit_key(m.group(1))] for m in _TIME_BEFORE.finditer(text or "")]
    return sorted({round(x, 6) for x in out})


def _landmark_time(om: dict, base: str, cell) -> tuple[float | None, str | None]:
    """(months, path of the field it was parsed from). Subgroup cells only use their own titles."""
    if cell.class_title or cell.category_title:
        for text, path in ((cell.class_title, cell.class_path), (cell.category_title, cell.category_path)):
            times = _time_expressions(text or "")
            if len(times) == 1 and path:
                return times[0], path
        return None, None
    times = _time_expressions(str(om.get("title", "")))
    if len(times) == 1:
        return times[0], f"{base}.title"
    tf = str(om.get("timeFrame", ""))
    times = _time_expressions(tf)
    if len(times) == 1 and not _OPEN_TIME.search(tf):
        return times[0], f"{base}.timeFrame"
    return None, None


def extract_anchors(study: Study, indices: list[int]) -> SurvivalExtraction:
    """Collect reported anchors from the selected outcome measures (medians + landmark rates).

    Anything whose meaning is unclear (non-survival measure, unknown scale, ambiguous time, censored
    value) is skipped and listed in ``skipped`` rather than plotted.
    """
    anchors: dict[str, list[Anchor]] = {}
    not_reached: list[str] = []
    skipped: list[str] = []
    converted: list[str] = []
    for idx in indices:
        base = f"{OM_PATH}[{idx}]"
        om = study.get(base)
        if not isinstance(om, dict):
            continue
        title = str(om.get("title", ""))
        endpoint = short_endpoint(title)
        ptype = str(om.get("paramType", "")).upper()
        unit = str(om.get("unitOfMeasure", ""))
        cells = measure_cells(study, idx)
        if ptype == "MEDIAN":
            mult = unit_to_months(unit)
            if not is_survival_measure(title):
                skipped.append(f"measure {idx}: median of a non-survival measure")
                continue
            if mult is None:
                skipped.append(f"measure {idx}: median unit is not a time unit")
                continue
            for c in cells:
                series = f"{endpoint} · {c.group_title}"
                if c.class_title or c.category_title:
                    continue  # subgroup medians (e.g. by biomarker) are not the arm's curve
                if c.value is None or c.value <= 0 or c.qualified:
                    if series not in not_reached:
                        not_reached.append(series)
                    continue
                if any(a.kind == "median" for a in anchors.get(series, [])):
                    skipped.append(f"{series}: additional median in measure {idx} not used")
                    continue
                refs = [
                    r
                    for r in (
                        study.ref(c.value_path, label=f"{series}: median ({unit})"),
                        study.ref(f"{base}.paramType"),
                        study.ref(f"{base}.title"),
                        study.ref(f"{base}.unitOfMeasure"),
                        study.ref(c.group_path) if c.group_path else None,
                    )
                    if r
                ]
                anchors.setdefault(series, []).append(
                    Anchor(
                        series,
                        endpoint,
                        c.group_title,
                        round(c.value * mult, 4),
                        0.5,
                        "median",
                        refs,
                        time_ci_low=c.lower * mult if c.lower is not None and c.lower > 0 else None,
                        time_ci_high=c.upper * mult if c.upper is not None and c.upper > 0 else None,
                    )
                )
        elif ptype == "NUMBER":
            survival_kind = is_survival_measure(title)
            event_kind = not survival_kind and is_event_rate_measure(title)
            if not (survival_kind or event_kind):
                skipped.append(f"measure {idx}: not a survival or event rate")
                continue
            if _PCT_UNIT.search(unit):
                scale = 0.01
            elif _PROB_UNIT.search(unit):
                scale = 1.0
            else:
                skipped.append(f"measure {idx}: unit is neither percent nor probability")
                continue
            values = [c.value for c in cells if c.value is not None]
            if values and ((scale == 0.01 and max(values) <= 1.0) or (scale == 1.0 and max(values) > 1.0)):
                skipped.append(f"measure {idx}: values do not match the stated unit")
                continue
            names_rate = bool(_RATE_WORD.search(title)) or any(
                _RATE_WORD.search(" ".join(x for x in (c.class_title, c.category_title) if x)) for c in cells
            )
            if survival_kind and not names_rate:
                skipped.append(f"measure {idx}: does not name a survival rate")
                continue
            for c in cells:
                if c.value is None:
                    continue
                series = f"{endpoint} · {c.group_title}"
                if c.qualified:
                    skipped.append(f"{series}: inequality-qualified rate in measure {idx}")
                    continue
                t, t_path = _landmark_time(om, base, c)
                if t is None or t <= 0:
                    skipped.append(f"{series}: no unambiguous time point in measure {idx}")
                    continue
                rate = c.value * scale
                surv = rate if survival_kind else 1.0 - rate
                if not 0 < surv <= 1:
                    skipped.append(f"{series}: rate out of range in measure {idx}")
                    continue
                if event_kind:
                    converted.append(f"{series} at {t:g} months")
                refs = [
                    study.ref(c.value_path, label=f"{series}: {'rate' if survival_kind else 'event rate'}"),
                    study.ref(t_path, label="time point") if t_path else None,
                    study.ref(f"{base}.title"),
                    study.ref(f"{base}.unitOfMeasure"),
                    study.ref(c.group_path) if c.group_path else None,
                ]
                lo = c.lower * scale if c.lower is not None else None
                hi = c.upper * scale if c.upper is not None else None
                if event_kind:
                    lo, hi = (1.0 - hi if hi is not None else None), (1.0 - lo if lo is not None else None)
                anchors.setdefault(series, []).append(
                    Anchor(
                        series,
                        endpoint,
                        c.group_title,
                        round(t, 4),
                        surv,
                        "landmark",
                        [r for r in refs if r],
                        ci_low=lo,
                        ci_high=hi,
                    )
                )
    dropped: list[str] = []
    clean: dict[str, list[Anchor]] = {}
    for series, pts in anchors.items():
        pts.sort(key=lambda a: (a.time, -a.survival))
        kept: list[Anchor] = []
        for a in pts:
            if kept and (a.survival > kept[-1].survival or a.time == kept[-1].time):
                dropped.append(f"{series} at {a.time:g} months")
                continue
            kept.append(a)
        if kept:
            clean[series] = kept
    return SurvivalExtraction(
        clean, [s for s in not_reached if s not in clean], dropped, list(dict.fromkeys(skipped)), converted
    )


def reconstruct_survival(
    anchors: dict[str, list[Anchor]],
    start_refs: dict[str, list[FieldRef]],
    step_months: float | None = None,
    max_interpolated: int = DEFAULT_MAX_INTERPOLATED,
) -> list[CurvePoint]:
    """Curve points per series: the reported anchors plus exponential interpolation between them.

    Density: with ``step_months`` the grid is fixed; otherwise each series gets its own grid
    (multiples of a half-month step, at least ``MIN_STEP_MONTHS``) with at most ``max_interpolated``
    interpolated points up to its last anchor. Reported points carry the anchor's citations and its
    chain position (``anchor_pos``); interpolated points carry exactly one citation (the outcome
    measure's title) and the chain positions of the anchors they were derived from (``between``).
    """
    out: list[CurvePoint] = []
    for series, pts in anchors.items():
        if not pts:
            continue
        first = pts[0]
        start = Anchor(series, first.endpoint, first.group, 0.0, 1.0, "start", start_refs.get(series, []))
        chain = [start, *pts]
        step = step_months or _series_step(chain[-1].time, max_interpolated)
        out.append(_point(start, "reported", 0))
        for pos, (a0, a1) in enumerate(zip(chain, chain[1:], strict=False)):
            span = a1.time - a0.time
            if span > 0 and a1.survival > 0 and a0.survival >= a1.survival:
                lam = math.log(a0.survival / a1.survival) / span
                title_ref = measure_title_ref(a1) or measure_title_ref(a0) or single_ref(a1) or single_ref(a0)
                k = math.floor(a0.time / step + 1e-9) + 1
                while k * step < a1.time - 1e-9:
                    t = k * step
                    k += 1
                    if t <= a0.time + 1e-9:
                        continue
                    out.append(
                        CurvePoint(
                            series,
                            a0.endpoint,
                            a0.group,
                            round(t, 3),
                            round(a0.survival * math.exp(-lam * (t - a0.time)), 4),
                            "interpolated",
                            refs=[title_ref] if title_ref else [],
                            kind="interpolated",
                            between=(pos, pos + 1),
                        )
                    )
            out.append(_point(a1, "reported", pos + 1))
    return out


def _series_step(last_time: float, max_interpolated: int) -> float:
    """Grid step (months) giving at most ``max_interpolated`` points before ``last_time``: half-month multiples."""
    raw = last_time / (max(1, max_interpolated) + 1)
    return max(MIN_STEP_MONTHS, math.ceil(raw * 2 - 1e-9) / 2)


def measure_title_ref(a: Anchor) -> FieldRef | None:
    """The outcome measure's title among an anchor's refs (the one citation an interpolated point keeps)."""
    return next((r for r in a.refs if _MEASURE_TITLE.search(r.field_path)), None)


def single_ref(a: Anchor) -> FieldRef | None:
    """The one citation a derived point keeps: the measure title, else the anchor's first ref (its value)."""
    return measure_title_ref(a) or (a.refs[0] if a.refs else None)


def _point(a: Anchor, source: str, pos: int | None = None) -> CurvePoint:
    return CurvePoint(
        a.series,
        a.endpoint,
        a.group,
        a.time,
        round(a.survival, 4),
        source,
        a.refs,
        a.kind,
        ci_low=a.ci_low,
        ci_high=a.ci_high,
        time_ci_low=a.time_ci_low,
        time_ci_high=a.time_ci_high,
        anchor_pos=pos,
    )


def curve_problems(points: list[CurvePoint], tol: float = 1e-9) -> dict[str, str]:
    """Series that are not a valid survival curve -> why (empty dict when every series is valid).

    A valid series starts at time 0 with S = 1, has non-decreasing time and non-increasing S in [0, 1].
    Handlers drop invalid series (with a note) instead of drawing a curve that rises.
    """
    by_series: dict[str, list[CurvePoint]] = {}
    for p in points:
        by_series.setdefault(p.series, []).append(p)
    problems: dict[str, str] = {}
    for series, pts in by_series.items():
        first = pts[0]
        if first.time != 0 or abs(first.survival - 1.0) > tol:
            problems[series] = "does not start at S(0) = 1"
            continue
        for a, b in zip(pts, pts[1:], strict=False):
            if b.time < a.time - tol:
                problems[series] = "time is not ordered"
                break
            if b.survival > a.survival + tol or not 0.0 <= b.survival <= 1.0:
                problems[series] = f"survival rises or leaves [0, 1] at {b.time:g} months"
                break
    return problems
