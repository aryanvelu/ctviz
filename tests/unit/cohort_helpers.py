"""Shared assertions for the cohort handler tests."""

from __future__ import annotations

from typing import Any

import ctviz.handlers  # noqa: F401  (registers handlers)
from ctviz.api.schemas.viz import GraphData
from ctviz.viz.base import get_handler
from tests.conftest import assert_citations_verified


async def run(intent, ctx):
    return await get_handler(intent)(ctx)


def rows(viz) -> list[dict[str, Any]]:
    return [d.model_extra or {} for d in viz.data]


def raws(*cohorts) -> dict[str, dict]:
    return {s.nct_id: s.raw for c in cohorts for s in c.studies}


def encoded_fields(viz) -> list[str]:
    enc = viz.encoding
    return [c.field for c in (enc.x, enc.x2, enc.y, enc.y2, enc.series, enc.size, enc.label) if c is not None]


def assert_valid(viz, *cohorts, min_checked: int = 1) -> int:
    """Encoding fields present in every row + the citation invariant."""
    if not isinstance(viz.data, GraphData):
        for r in rows(viz):
            for f in encoded_fields(viz):
                assert f in r, f"missing encoded field {f}"
    checked = assert_citations_verified(viz, raws(*cohorts))
    assert checked >= min_checked
    return checked


def data_numbers(viz) -> set[float]:
    data = viz.data.nodes + viz.data.edges if isinstance(viz.data, GraphData) else viz.data
    out: set[float] = set()
    for d in data:
        for v in (d.model_extra or {}).values():
            if isinstance(v, int | float) and not isinstance(v, bool):
                out.add(v)
    return out


def assert_nonzero_cites(viz, count_field: str = "trial_count") -> None:
    """Every non-zero bucket has evidence; every zero bucket has none."""
    for d in viz.data:
        n = (d.model_extra or {})[count_field]
        assert (n > 0) == bool(d.citations), f"{d.id}: count {n} vs {len(d.citations)} citations"
