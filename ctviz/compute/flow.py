"""Participant flow (``participantFlowModule``) as per-arm, per-period flow tables.

For each arm and period: started -> completed + not completed by reason. When the reported reasons
do not add up to the 'NOT COMPLETED' milestone, the gap is kept as an explicit
``unexplained`` bucket (no citation: the registry does not say why). Conservation holds per
period: ``started == completed + sum(reasons) + unexplained``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ctviz.compute.results import parse_int
from ctviz.core.provenance import FieldRef, Study
from ctviz.ctgov.fields import P

FLOW_PATH = P["participant_flow"]


@dataclass
class Reason:
    label: str
    count: int
    refs: list[FieldRef]


@dataclass
class PeriodFlow:
    period: str
    started: int
    completed: int
    reasons: list[Reason]
    unexplained: int
    started_refs: list[FieldRef]
    completed_refs: list[FieldRef]
    not_completed_reported: int | None = None
    inconsistent: bool = False  # reasons exceed the not-completed count

    @property
    def not_completed(self) -> int:
        return self.started - self.completed


@dataclass
class ArmFlow:
    group_id: str
    title: str
    title_ref: FieldRef | None
    periods: list[PeriodFlow] = field(default_factory=list)

    @property
    def started(self) -> int:
        return self.periods[0].started if self.periods else 0

    @property
    def completed(self) -> int:
        return self.periods[-1].completed if self.periods else 0

    @property
    def dropout_rate(self) -> float | None:
        return (self.started - self.completed) / self.started if self.started else None


def _milestone(period: dict, kind: str) -> tuple[int, dict] | None:
    for mi, m in enumerate(period.get("milestones", []) or []):
        if isinstance(m, dict) and str(m.get("type", "")).strip().upper() == kind:
            return mi, m
    return None


def _achievement(m: dict, gid: str) -> tuple[int, int | None] | None:
    for ai, a in enumerate(m.get("achievements", []) or []):
        if isinstance(a, dict) and a.get("groupId") == gid:
            return ai, parse_int(a.get("numSubjects"))
    return None


def flow_table(study: Study) -> list[ArmFlow]:
    """Per-arm flow across all periods; arms with no STARTED count are skipped."""
    groups = list(study.items(f"{FLOW_PATH}.groups"))
    periods = list(study.items(f"{FLOW_PATH}.periods"))
    arms: list[ArmFlow] = []
    for _, g, gpath in groups:
        if not isinstance(g, dict) or not g.get("id"):
            continue
        gid = g["id"]
        arm = ArmFlow(gid, str(g.get("title", gid)), study.ref(f"{gpath}.title"))
        for _, per, ppath in periods:
            if not isinstance(per, dict):
                continue
            st, co, nc = _milestone(per, "STARTED"), _milestone(per, "COMPLETED"), _milestone(per, "NOT COMPLETED")
            if st is None or co is None:
                continue
            sa, ca = _achievement(st[1], gid), _achievement(co[1], gid)
            if not sa or not ca or sa[1] is None or ca[1] is None:
                continue
            started, completed = sa[1], ca[1]
            s_path = f"{ppath}.milestones[{st[0]}].achievements[{sa[0]}].numSubjects"
            c_path = f"{ppath}.milestones[{co[0]}].achievements[{ca[0]}].numSubjects"
            nc_reported = None
            if nc is not None:
                na = _achievement(nc[1], gid)
                nc_reported = na[1] if na else None
            reasons: list[Reason] = []
            for _di, dw, dpath in study.items(f"{ppath}.dropWithdraws"):
                if not isinstance(dw, dict):
                    continue
                for ri, r in enumerate(dw.get("reasons", []) or []):
                    if isinstance(r, dict) and r.get("groupId") == gid:
                        cnt = parse_int(r.get("numSubjects"))
                        if cnt:
                            refs = [
                                study.ref(f"{dpath}.reasons[{ri}].numSubjects", label=f"{arm.title}"),
                                study.ref(f"{dpath}.type"),
                            ]
                            reasons.append(Reason(str(dw.get("type", "Other")), cnt, [x for x in refs if x]))
            not_completed = max(0, started - completed)
            listed = sum(r.count for r in reasons)
            per_flow = PeriodFlow(
                period=str(per.get("title", "Overall Study")),
                started=started,
                completed=completed,
                reasons=reasons,
                unexplained=max(0, not_completed - listed),
                started_refs=[x for x in (study.ref(s_path, label=f"{arm.title}: started"),) if x],
                completed_refs=[x for x in (study.ref(c_path, label=f"{arm.title}: completed"),) if x],
                not_completed_reported=nc_reported,
                inconsistent=listed > not_completed,
            )
            arm.periods.append(per_flow)
        if arm.periods:
            arms.append(arm)
    return arms


def period_titles(arms: list[ArmFlow]) -> list[str]:
    """Distinct period titles in order."""
    seen: list[str] = []
    for a in arms:
        for p in a.periods:
            if p.period not in seen:
                seen.append(p.period)
    return seen
