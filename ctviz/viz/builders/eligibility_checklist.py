"""eligibility_checklist: one row per criterion.

Row shape: ``{section: structured|inclusion|exclusion|unknown, index, text,
status: met|unmet|unknown|needs_review, reason, evaluated_by: rule|llm|none, relevance?, subheading?,
parent_text?}``. ``section`` is ``unknown`` when the record's criteria have no clear inclusion/exclusion
heading; ``parent_text`` is the list item a nested criterion sits under. ``status`` means "does this criterion block
the person": met = does not block, unmet = blocks. ``meta.summary`` holds counts by status and section;
``meta.diagnostics.profile_used`` lists which profile fields were considered and
``meta.diagnostics.status_semantics`` spells out the status values.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ctviz.api.schemas.viz import Channel, Datum, Encoding, Sort, Visualization, VizMeta, VizType
from ctviz.viz.base import make_viz

STATUSES = ("met", "unmet", "needs_review", "unknown")
SECTIONS = ("structured", "inclusion", "exclusion", "unknown")  # unknown: no/ambiguous heading in the record
EVALUATORS = ("rule", "llm", "none")
STATUS_SEMANTICS = (
    "met = does not block eligibility; unmet = blocks eligibility; unknown = not evaluated; "
    "needs_review = evaluated without confidence, confirm with the study team"
)


def checklist_summary(rows: list[Datum]) -> dict[str, Any]:
    """Deterministic counts by status, section and evaluator."""
    ex = [r.model_extra or {} for r in rows]
    by_status = Counter(e["status"] for e in ex)
    return {
        "total": len(rows),
        "by_status": {s: by_status.get(s, 0) for s in STATUSES},
        "by_section": dict(Counter(e["section"] for e in ex)),
        "by_evaluator": dict(Counter(e["evaluated_by"] for e in ex)),
    }


def eligibility_checklist(
    rows: list[Datum],
    *,
    title: str,
    profile_used: dict[str, Any] | None = None,
    subtitle: str | None = None,
    meta: VizMeta | None = None,
) -> Visualization:
    """Build the checklist; validates row enums and fills ``meta.summary`` and the diagnostics."""
    for r in rows:
        e = r.model_extra or {}
        if (
            e.get("section") not in SECTIONS
            or e.get("status") not in STATUSES
            or e.get("evaluated_by") not in EVALUATORS
        ):
            raise ValueError(f"checklist row {r.id} has invalid section/status/evaluated_by")
    meta = meta or VizMeta()
    meta.summary = checklist_summary(rows)
    meta.diagnostics["profile_used"] = profile_used or {}
    meta.diagnostics["status_semantics"] = STATUS_SEMANTICS
    enc = Encoding(
        label=Channel(field="text", type="nominal", title="Criterion"),
        series=Channel(field="status", type="nominal", title="Status", sort=Sort(values=list(STATUSES))),
        tooltip=["section", "reason", "evaluated_by", "relevance"],
    )
    return make_viz(VizType.ELIGIBILITY_CHECKLIST, title, enc, rows, meta, subtitle)
