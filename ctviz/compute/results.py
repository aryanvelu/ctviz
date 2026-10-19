"""Shared parsing for ``resultsSection`` plus outcome-measure tools (catalog, forest rows, bars).

ClinicalTrials.gov serialises most result numbers as strings (``"1306"``, ``"<0.001"``, ``"NA"``).
Everything here parses defensively and keeps the exact JSON path of every number so the handler
can cite the leaf it came from. Nothing in this module calls an LLM.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from ctviz.core.provenance import FieldRef, Study
from ctviz.ctgov.fields import P

OM_PATH = P["outcome_measures"]
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_MISSING_TOKENS = {"", "na", "n/a", "nr", "ne", "nc", "not reached", "not estimable", "not applicable", "-", "--"}


# -- number parsing -------------------------------------------------------------------------------


def parse_num(value: Any) -> float | None:
    """Parse a registry number (str/int/float) to float; ``None`` for NA / not reached / garbage.

    ``"<0.001"`` -> 0.001 (use :func:`parse_qualified` to keep the qualifier); ``"0.62 (0.52-0.74)"``
    -> 0.62 (first number). Thousands separators are removed.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value) if math.isfinite(value) else None
    if not isinstance(value, str):
        return None
    s = _normalize(value).replace(",", "")
    if s.lower() in _MISSING_TOKENS:
        return None
    m = _NUM.search(s)
    if not m:
        return None
    try:
        out = float(m.group(0))
    except ValueError:  # pragma: no cover - regex guarantees a float literal
        return None
    return out if math.isfinite(out) else None


def _normalize(s: str) -> str:
    """Strip and map typographic minus signs (U+2212, U+2012-U+2014 before a digit) to '-'."""
    s = s.strip().replace("\u2212", "-")
    if s[:1] in ("\u2012", "\u2013", "\u2014"):
        s = "-" + s[1:]
    return re.sub(r"^-\s+(?=[\d.])", "-", s)


def parse_exact(value: Any) -> float | None:
    """Like :func:`parse_num` but ``None`` for inequality-qualified values (">30.2" is censored, not exact)."""
    num, qual = parse_qualified(value)
    return None if qual else num


def parse_qualified(value: Any) -> tuple[float | None, str | None]:
    """Parse a value that may carry an inequality (``"<0.0001"``); returns (number, qualifier)."""
    num = parse_num(value)
    if num is None or not isinstance(value, str):
        return num, None
    s = _normalize(value)
    for q in ("<=", ">=", "≤", "≥", "<", ">"):
        if s.startswith(q):
            return num, q
    return num, None


def parse_int(value: Any) -> int | None:
    """Parse a count; ``None`` if absent or not a whole number."""
    num = parse_num(value)
    if num is None or num < 0 or abs(num - round(num)) > 1e-9:
        return None
    return int(round(num))


def has_results_section(study: Study) -> bool:
    """True when the record actually contains a ``resultsSection`` (the flag alone is not enough)."""
    return isinstance(study.raw.get("resultsSection"), dict) and bool(study.raw["resultsSection"])


# -- outcome-measure catalog -----------------------------------------------------------------------


@dataclass(frozen=True)
class MeasureInfo:
    """Compact, number-free description of one outcome measure (what the Analysis agent sees)."""

    index: int
    path: str
    type: str
    title: str
    param_type: str
    unit: str
    time_frame: str
    dispersion: str
    groups: list[tuple[str, str]]  # (group id, title)
    n_analyses: int
    class_titles: list[str]

    def catalog_entry(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "type": self.type,
            "paramType": self.param_type,
            "unit": self.unit,
            "timeFrame": self.time_frame[:120],
            "groups": [t for _, t in self.groups],
            "group_ids": [g for g, _ in self.groups],
            "has_analyses": self.n_analyses > 0,
            "classes": self.class_titles[:8],
        }


def measure_catalog(study: Study) -> list[MeasureInfo]:
    """All outcome measures with posted results, in registry order."""
    out: list[MeasureInfo] = []
    for i, om, path in study.items(OM_PATH):
        if not isinstance(om, dict):
            continue
        groups = [(g.get("id", ""), g.get("title", "")) for g in om.get("groups", []) if isinstance(g, dict)]
        classes = [c.get("title", "") for c in om.get("classes", []) if isinstance(c, dict) and c.get("title")]
        out.append(
            MeasureInfo(
                index=i,
                path=path,
                type=str(om.get("type", "")),
                title=str(om.get("title", "")),
                param_type=str(om.get("paramType", "")),
                unit=str(om.get("unitOfMeasure", "")),
                time_frame=str(om.get("timeFrame", "")),
                dispersion=str(om.get("dispersionType", "")),
                groups=groups,
                n_analyses=len(om.get("analyses", []) or []),
                class_titles=classes,
            )
        )
    return out


@dataclass
class Cell:
    """One measurement cell with the exact paths of its numbers."""

    measure_index: int
    group_id: str
    group_title: str
    group_path: str | None
    class_title: str | None
    category_title: str | None
    value: float | None
    raw_value: Any
    value_path: str
    lower: float | None = None
    lower_path: str | None = None
    upper: float | None = None
    upper_path: str | None = None
    spread: float | None = None
    spread_path: str | None = None
    comment: str | None = None
    class_path: str | None = None
    category_path: str | None = None
    extra_paths: list[str] = field(default_factory=list)

    @property
    def qualified(self) -> bool:
        """True when the value carries an inequality (e.g. '>30.2'): censored / not reached, not exact."""
        return parse_qualified(self.raw_value)[1] is not None

    @property
    def label(self) -> str:
        parts = [p for p in (self.class_title, self.category_title) if p]
        return " · ".join(parts)


def group_titles(om: dict[str, Any], base: str) -> dict[str, tuple[str, str]]:
    """group id -> (title, path of the title leaf)."""
    out: dict[str, tuple[str, str]] = {}
    for gi, g in enumerate(om.get("groups", []) or []):
        if isinstance(g, dict) and g.get("id"):
            out[g["id"]] = (str(g.get("title", g["id"])), f"{base}.groups[{gi}].title")
    return out


def measure_cells(study: Study, index: int) -> list[Cell]:
    """Every measurement cell of outcome measure ``index`` (classes x categories x groups)."""
    base = f"{OM_PATH}[{index}]"
    om = study.get(base)
    if not isinstance(om, dict):
        return []
    titles = group_titles(om, base)
    cells: list[Cell] = []
    for ci, cls in enumerate(om.get("classes", []) or []):
        if not isinstance(cls, dict):
            continue
        cpath = f"{base}.classes[{ci}]"
        for ki, cat in enumerate(cls.get("categories", []) or []):
            if not isinstance(cat, dict):
                continue
            kpath = f"{cpath}.categories[{ki}]"
            for mi, m in enumerate(cat.get("measurements", []) or []):
                if not isinstance(m, dict):
                    continue
                mpath = f"{kpath}.measurements[{mi}]"
                gid = str(m.get("groupId", ""))
                title, gpath = titles.get(gid, (gid, None))
                cells.append(
                    Cell(
                        measure_index=index,
                        group_id=gid,
                        group_title=title,
                        group_path=gpath,
                        class_title=cls.get("title"),
                        category_title=cat.get("title"),
                        value=parse_num(m.get("value")),
                        raw_value=m.get("value"),
                        value_path=f"{mpath}.value",
                        lower=parse_num(m.get("lowerLimit")),
                        lower_path=f"{mpath}.lowerLimit" if "lowerLimit" in m else None,
                        upper=parse_num(m.get("upperLimit")),
                        upper_path=f"{mpath}.upperLimit" if "upperLimit" in m else None,
                        spread=parse_num(m.get("spread")),
                        spread_path=f"{mpath}.spread" if "spread" in m else None,
                        comment=m.get("comment"),
                        class_path=f"{cpath}.title" if cls.get("title") else None,
                        category_path=f"{kpath}.title" if cat.get("title") else None,
                    )
                )
    return cells


# -- analyses (effect estimates) -------------------------------------------------------------------

_RATIO = re.compile(r"ratio|relative risk|\brr\b|\bhr\b|\bor\b", re.I)
_LOG = re.compile(r"\blog\b|\bln\b|logarithm", re.I)
_DIRECTION = re.compile(
    r"\bvs\.?\b|\bversus\b|compared (?:to|with)|relative to|\bover\b|numerator|\b\w+\s*/\s*\w+\b(?<!and/or)", re.I
)
_DIFF = re.compile(r"differen|change|\bmd\b|\blsm\b|least squares", re.I)


def effect_kind(param_type: str) -> str | None:
    """'ratio' / 'difference' / None for an analysis ``paramType`` string."""
    if not param_type:
        return None
    if _LOG.search(param_type):
        return "difference"  # log ratios / log-scale estimates: null 0, linear axis
    if _RATIO.search(param_type):
        return "ratio"
    if _DIFF.search(param_type):
        return "difference"
    return None


@dataclass
class Estimate:
    """One between-group analysis with an effect estimate and CI, plus exact citation paths."""

    measure_index: int
    analysis_index: int
    measure_title: str
    comparison: str
    description: str | None
    param_type: str
    kind: str
    estimate: float
    ci_low: float
    ci_high: float
    ci_pct: float | None
    p_value: float | None
    p_qualifier: str | None
    p_raw: str | None
    method: str | None
    refs: list[FieldRef]
    group_names: tuple[str, ...] = ()  # titles of the compared groups (registry order of ``groupIds``)

    @property
    def null_value(self) -> float:
        return 1.0 if self.kind == "ratio" else 0.0

    @property
    def label(self) -> str:
        lab = f"{self.measure_title}: {self.comparison}"
        return f"{lab} ({self.description})" if self.description else lab

    def as_meta(self) -> dict[str, Any]:
        base = f"{OM_PATH}[{self.measure_index}].analyses[{self.analysis_index}]"
        return {
            "label": self.label,
            "param_type": self.param_type,
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_pct": self.ci_pct,
            "p_value": self.p_raw,
            "method": self.method,
            "field_path": f"{base}.paramValue",
        }


def analyses_for(study: Study, index: int, *, require_ci: bool = True) -> list[Estimate]:
    """Ratio/difference analyses of outcome measure ``index`` that have a parseable estimate (+CI)."""
    base = f"{OM_PATH}[{index}]"
    om = study.get(base)
    if not isinstance(om, dict):
        return []
    titles = group_titles(om, base)
    out: list[Estimate] = []
    for ai, an in enumerate(om.get("analyses", []) or []):
        if not isinstance(an, dict):
            continue
        apath = f"{base}.analyses[{ai}]"
        ptype = str(an.get("paramType", ""))
        kind = effect_kind(ptype)
        # inequality-qualified estimates/limits (">30.2") are censored, not exact: not plotted
        est = parse_exact(an.get("paramValue"))
        lo, hi = parse_exact(an.get("ciLowerLimit")), parse_exact(an.get("ciUpperLimit"))
        if kind is None or est is None or (require_ci and (lo is None or hi is None)):
            continue
        if lo is not None and hi is not None and lo > hi:
            lo, hi = hi, lo
        if kind == "ratio" and (est <= 0 or (lo is not None and lo <= 0)):
            continue  # cannot be shown on a log scale; malformed
        gids = [g for g in an.get("groupIds", []) or [] if isinstance(g, str)]
        names = [titles.get(g, (g, None))[0] for g in gids]
        # Neutral wording: the registry does not say which group is the numerator unless the
        # analysis description states a direction.
        desc = an.get("groupDescription") if isinstance(an.get("groupDescription"), str) else None
        directed = bool(desc and _DIRECTION.search(desc))
        comparison = desc if directed else ("groups: " + ", ".join(names) if names else "groups")
        p_val, p_q = parse_qualified(an.get("pValue"))
        refs: list[FieldRef | None] = [
            study.ref(f"{apath}.paramValue", label="estimate"),
            study.ref(f"{apath}.ciLowerLimit", label="CI lower"),
            study.ref(f"{apath}.ciUpperLimit", label="CI upper"),
            study.ref(f"{apath}.pValue", label="p-value"),
            study.ref(f"{apath}.paramType", label="estimate type"),
            study.ref(f"{base}.title", label="outcome measure"),
        ]
        refs += [study.ref(titles[g][1], label="group") for g in gids if g in titles and titles[g][1]]
        if desc:
            refs.append(study.ref(f"{apath}.groupDescription", label="analysis description"))
        out.append(
            Estimate(
                measure_index=index,
                analysis_index=ai,
                measure_title=str(om.get("title", "")),
                comparison=comparison,
                description=None if directed else desc,
                param_type=ptype,
                kind=kind,
                estimate=est,
                ci_low=lo if lo is not None else est,
                ci_high=hi if hi is not None else est,
                ci_pct=parse_num(an.get("ciPctValue")),
                p_value=p_val,
                p_qualifier=p_q,
                p_raw=an.get("pValue") if isinstance(an.get("pValue"), str) else None,
                method=an.get("statisticalMethod"),
                refs=[r for r in refs if r is not None],
                group_names=tuple(names),
            )
        )
    return out


def forest_estimates(study: Study, indices: list[int], prefer: str | None = "ratio") -> list[Estimate]:
    """Estimates across the selected measures, restricted to one scale.

    ``prefer='ratio'`` keeps ratios when any exist; ``prefer=None`` keeps the kind of the first
    estimate in selection order (and, for differences, only measures sharing its unit).
    """
    ests = [e for i in indices for e in analyses_for(study, i)]
    if not ests:
        return []
    if prefer is not None:
        chosen = [e for e in ests if e.kind == prefer]
        if chosen:
            return chosen
    kind = ests[0].kind
    ests = [e for e in ests if e.kind == kind]
    if kind == "difference":
        units = {i: str(study.get(f"{OM_PATH}[{i}].unitOfMeasure", "")) for i in indices}
        first = units.get(ests[0].measure_index)
        ests = [e for e in ests if units.get(e.measure_index) == first]
    return ests


def short_endpoint(title: str) -> str:
    """'Overall Survival (OS)' -> 'OS'; falls back to a trimmed title."""
    t = title.lower()
    if re.search(r"progression[- ]free|\bpfs\b", t):
        return "PFS"
    if re.search(r"overall survival|\bos\b", t):
        return "OS"
    if re.search(r"event[- ]free|\befs\b", t):
        return "EFS"
    if re.search(r"disease[- ]free|\bdfs\b", t):
        return "DFS"
    return title if len(title) <= 60 else title[:57] + "..."
