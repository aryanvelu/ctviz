"""Adverse-event rates from ``resultsSection.adverseEventsModule`` (pure, deterministic).

Rates are ``numAffected / numAtRisk`` per (term, event group). Every value keeps the JSON paths of
its numerator and denominator so the handler can cite both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ctviz.compute.results import parse_int, parse_num
from ctviz.core.provenance import FieldRef, Study
from ctviz.ctgov.fields import P

AE_PATH = P["adverse_events"]
DLT_CONVENTION = 0.33  # 3+3 design: escalation stops when >= 1/3 of a cohort has a DLT (convention)


@dataclass
class AERate:
    """Rate of one event term (or 'Any serious adverse event') in one event group."""

    term: str
    organ_system: str
    seriousness: str  # serious | other | any_serious
    group_id: str
    group_title: str
    n_affected: int
    n_at_risk: int
    refs: list[FieldRef] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.n_affected / self.n_at_risk if self.n_at_risk else 0.0

    @property
    def key(self) -> tuple[str, str]:
        return (self.term, self.seriousness)


@dataclass
class EventGroup:
    index: int
    id: str
    title: str
    title_path: str


def event_groups(study: Study) -> list[EventGroup]:
    """Adverse-event groups (arms as reported in the AE module)."""
    return [
        EventGroup(i, str(g.get("id", "")), str(g.get("title", g.get("id", ""))), f"{p}.title")
        for i, g, p in study.items(f"{AE_PATH}.eventGroups")
        if isinstance(g, dict)
    ]


def frequency_threshold(study: Study) -> tuple[float | None, FieldRef | None]:
    """Registry 'other (non-serious) AE' reporting threshold as a fraction, with its citation."""
    path = f"{AE_PATH}.frequencyThreshold"
    val = parse_num(study.get(path))
    if val is None:
        return None, None
    return val / 100.0, study.ref(path, label="Frequency threshold for reporting other adverse events (%)")


def ae_rates(study: Study) -> list[AERate]:
    """All per-term, per-group rates (serious + other) plus 'Any serious adverse event' per group."""
    groups = {g.id: g for g in event_groups(study)}
    out: list[AERate] = []
    for g in groups.values():
        gpath = f"{AE_PATH}.eventGroups[{g.index}]"
        num, den = (
            parse_int(study.get(f"{gpath}.seriousNumAffected")),
            parse_int(study.get(f"{gpath}.seriousNumAtRisk")),
        )
        if num is not None and den:
            out.append(
                AERate(
                    term="Any serious adverse event",
                    organ_system="All",
                    seriousness="any_serious",
                    group_id=g.id,
                    group_title=g.title,
                    n_affected=num,
                    n_at_risk=den,
                    refs=[
                        r
                        for r in (
                            study.ref(f"{gpath}.seriousNumAffected", label=f"{g.title}: affected"),
                            study.ref(f"{gpath}.seriousNumAtRisk", label=f"{g.title}: at risk"),
                            study.ref(g.title_path),
                        )
                        if r
                    ],
                )
            )
    for kind, key in (("serious", "seriousEvents"), ("other", "otherEvents")):
        for _, ev, epath in study.items(f"{AE_PATH}.{key}"):
            if not isinstance(ev, dict) or not ev.get("term"):
                continue
            for si, st in enumerate(ev.get("stats", []) or []):
                if not isinstance(st, dict):
                    continue
                spath = f"{epath}.stats[{si}]"
                num, den = parse_int(st.get("numAffected")), parse_int(st.get("numAtRisk"))
                gid = str(st.get("groupId", ""))
                if num is None or not den or gid not in groups:
                    continue
                g = groups[gid]
                out.append(
                    AERate(
                        term=str(ev["term"]),
                        organ_system=str(ev.get("organSystem", "")),
                        seriousness=kind,
                        group_id=gid,
                        group_title=g.title,
                        n_affected=num,
                        n_at_risk=den,
                        refs=[
                            r
                            for r in (
                                study.ref(f"{spath}.numAffected", label=f"{ev['term']} / {g.title}: affected"),
                                study.ref(f"{spath}.numAtRisk", label=f"{g.title}: at risk"),
                                study.ref(f"{epath}.term"),
                                study.ref(f"{epath}.organSystem"),
                                study.ref(g.title_path),
                            )
                            if r
                        ],
                    )
                )
    return out


def top_terms(
    rates: list[AERate], top_n: int = 15, group_ids: set[str] | None = None, min_share_of_largest: float = 0.1
) -> tuple[list[tuple[str, str]], int]:
    """Top (term, seriousness) keys by max rate across groups; returns (keys, number omitted).

    Groups with fewer than ``min_share_of_largest`` x the largest group's at-risk count are ignored
    for ranking (a 2-participant crossover group would otherwise dominate with 100% rates).
    """
    pool = [r for r in rates if r.seriousness != "any_serious" and (group_ids is None or r.group_id in group_ids)]
    if not pool:
        return [], 0
    largest = max(r.n_at_risk for r in pool)
    ranked_pool = [r for r in pool if r.n_at_risk >= min_share_of_largest * largest] or pool
    best: dict[tuple[str, str], float] = {}
    for r in ranked_pool:
        best[r.key] = max(best.get(r.key, 0.0), r.rate)
    keys = sorted(best, key=lambda k: (-best[k], k[0], k[1]))
    return keys[:top_n], max(0, len(keys) - top_n)


def serious_rate_pooled(study: Study) -> AERate | None:
    """Pooled 'any serious AE' across all event groups of one trial (cohort comparisons)."""
    per_group = ae_rates_any_serious(study)
    if not per_group:
        return None
    return AERate(
        term="Any serious adverse event",
        organ_system="All",
        seriousness="any_serious",
        group_id="*",
        group_title="All groups",
        n_affected=sum(r.n_affected for r in per_group),
        n_at_risk=sum(r.n_at_risk for r in per_group),
        refs=[x for r in per_group for x in r.refs],
    )


def ae_rates_any_serious(study: Study) -> list[AERate]:
    """Only the per-group 'Any serious adverse event' rows."""
    return [r for r in ae_rates(study) if r.seriousness == "any_serious"]


# -- dose escalation -------------------------------------------------------------------------------

_DOSE = re.compile(r"(\d+(?:\.\d+)?)\s*(mg/m\^?2|mg/kg|mcg/kg|µg/kg|mg|mcg|µg|g)\b", re.I)
_ESCALATION = re.compile(r"dose[- ]?(escalation|level|cohort|finding)|\bDL\s?\d", re.I)
DLT_TITLE = re.compile(r"dose[- ]limiting|\bDLTs?\b", re.I)


def parse_dose(title: str) -> float | None:
    """First numeric dose in an arm title ('Drug 200 mg Q3W' -> 200.0), else None."""
    m = _DOSE.search(title or "")
    return float(m.group(1)) if m else None


def is_dose_escalation(study: Study) -> bool:
    """Phase 1 trial whose groups are dose levels / escalation cohorts, or that reports DLTs."""
    phases = study.get("phases") or []
    if not any(p in ("PHASE1", "EARLY_PHASE1") for p in phases):
        return False
    titles = [g.title for g in event_groups(study)]
    dosed = sum(1 for t in titles if parse_dose(t) is not None)
    if any(_ESCALATION.search(t) for t in titles) or dosed >= 2:
        return True
    oms = study.get("outcome_measures") or []
    return any(isinstance(om, dict) and DLT_TITLE.search(str(om.get("title", ""))) for om in oms)


def dose_order(titles: list[str]) -> list[int]:
    """Indices of ``titles`` sorted by parsed dose (titles without a dose keep registry order, last)."""
    doses = [parse_dose(t) for t in titles]
    return sorted(range(len(titles)), key=lambda i: (doses[i] is None, doses[i] or 0.0, i))


# -- no results: risk statements -------------------------------------------------------------------

RISK_WORDS = re.compile(
    r"safety|tolerab|side[- ]effect|adverse|toxicit|risk|harm|hypoglyc|bleed|pancreatit|nausea|vomit|"
    r"infection|allerg|death|suicid|contraindicat|warning",
    re.I,
)
_SENT = re.compile(r"[^.!?\n]+[.!?]?")


@dataclass
class RiskStatement:
    source: str
    text: str
    ref: FieldRef


def risk_statements(study: Study, max_desc: int = 6, max_exclusions: int = 12) -> list[RiskStatement]:
    """Verbatim risk-related sentences from the descriptions and the exclusion criteria bullets."""
    out: list[RiskStatement] = []
    for key, source in (("brief_summary", "Brief summary"), ("detailed_description", "Detailed description")):
        text = study.get(key)
        if not isinstance(text, str):
            continue
        for m in _SENT.finditer(text):
            sent = m.group(0)
            if RISK_WORDS.search(sent) and len([s for s in out if s.source != "Exclusion criterion"]) < max_desc:
                lead = len(sent) - len(sent.lstrip())
                ref = study.text_ref(key, m.start() + lead, m.end(), label=source)
                if ref and ref.excerpt.strip():
                    out.append(RiskStatement(source, ref.excerpt.strip(), ref))
    crit = study.get("eligibility_criteria")
    if isinstance(crit, str):
        low = crit.lower()
        start = low.find("exclusion criteria")
        if start >= 0:
            n = 0
            pos = start
            for line in crit[start:].splitlines(keepends=True):
                stripped = line.strip()
                body = stripped.lstrip("*-•0123456789.) ").strip()
                if body and not body.lower().startswith("exclusion criteria") and n < max_exclusions:
                    off = pos + line.find(body)
                    ref = study.text_ref("eligibility_criteria", off, off + len(body), label="Exclusion criterion")
                    if ref:
                        out.append(RiskStatement("Exclusion criterion", ref.excerpt, ref))
                        n += 1
                pos += len(line)
    return out
