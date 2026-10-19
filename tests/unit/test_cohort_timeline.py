"""sponsor_activity and development_progression timelines."""

from __future__ import annotations

from ctviz.api.schemas import Intent, VizType
from tests.unit.cohort_helpers import assert_valid, rows, run

TIMELINE_FIELDS = {"task", "start", "end", "lane", "category", "status", "is_estimated", "point"}


async def test_sponsor_activity_capped(make_ctx, cohort):
    c = cohort("tirzepatide")
    res = await run(Intent.SPONSOR_ACTIVITY, make_ctx(Intent.SPONSOR_ACTIVITY, cohorts=[c]))
    viz = res.visualization
    assert viz.type is VizType.TIMELINE and viz.meta.time_axis.model_dump(exclude_none=True) == {"kind": "calendar"}
    assert (viz.encoding.x.field, viz.encoding.x2.field, viz.encoding.y.field) == ("start", "end", "task")
    rs = rows(viz)
    assert len(rs) == 40 == res.facts["trials_shown_derived"]
    assert all(set(r) >= TIMELINE_FIELDS for r in rs)
    assert any(n.code.value == "TRUNCATED" for n in res.notes)
    # the 40 shown are the most recently started
    starts = sorted((s.get("start_date") or "" for s in c.studies), reverse=True)
    cutoff = starts[39]
    assert all(r["start"] >= cutoff[:7] for r in rs)
    for d in viz.data:
        r = d.model_extra
        assert r["point"] == (r["end"] is None)
        start_cites = [cit for cit in d.citations if cit.field_path.endswith("startDateStruct.date")]
        assert start_cites and r["start"].startswith(start_cites[0].excerpt[:7])
    assert sum(res.facts["trials_by_status_derived"].values()) == 40
    assert_valid(viz, c)


async def test_development_progression(make_ctx, cohort):
    c = cohort("semaglutide", "Semaglutide")
    res = await run(Intent.DEVELOPMENT_PROGRESSION, make_ctx(Intent.DEVELOPMENT_PROGRESSION, cohorts=[c]))
    viz = res.visualization
    assert viz.type is VizType.TIMELINE and viz.meta.time_axis.kind == "calendar"
    rs = rows(viz)
    assert all(set(r) >= TIMELINE_FIELDS for r in rs)
    assert rs[0]["category"] == "milestone" and rs[0]["point"] is True
    assert any(n.code.value == "NOT_IN_REGISTRY" for n in res.notes)
    phase_rows = {r["task"]: r for r in rs[1:]}
    assert "Phase 3" in phase_rows
    for label, r in phase_rows.items():
        f = res.facts["phases"][label]
        assert f["first_start"] == r["start"] and f["trial_count"] == r["trial_count"]
        assert r["end"] is None or r["start"] <= r["end"]
        phases = {
            "Phase 1": "PHASE1",
            "Phase 2": "PHASE2",
            "Phase 3": "PHASE3",
            "Phase 4": "PHASE4",
            "Early Phase 1": "EARLY_PHASE1",
        }[label]
        assert r["trial_count"] == sum(
            1 for s in c.studies if phases in (s.get("phases") or []) and s.get("start_date")
        )
    for d in viz.data[1:]:
        first = d.citations[0]
        assert first.nct_id == d.model_extra["first_nct"] and first.field_path.endswith("startDateStruct.date")
    assert_valid(viz, c)
