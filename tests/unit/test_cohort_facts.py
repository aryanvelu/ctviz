"""Facts invariant: every number in HandlerResult.facts is a value in the data, unless a key on its
path ends in ``_derived`` (documented aggregates). The summary template relies on this."""

from __future__ import annotations

from typing import Any

import pytest

from ctviz.api.schemas import Intent, VizType
from tests.unit.cohort_helpers import data_numbers, run

CASES = [
    (Intent.TREND_OVER_TIME, ["semaglutide"], {}),
    (Intent.TREND_OVER_TIME, ["semaglutide", "tirzepatide"], {}),
    (Intent.TREND_OVER_TIME, ["obesity_since2020"], {"series_by": "phase"}),
    (Intent.PHASE_DISTRIBUTION, ["semaglutide", "tirzepatide"], {}),
    (Intent.STATUS_DISTRIBUTION, ["semaglutide"], {}),
    (Intent.STATUS_DISTRIBUTION, ["semaglutide", "tirzepatide"], {}),
    (Intent.SPONSOR_LANDSCAPE, ["obesity_since2020"], {}),
    (Intent.SPONSOR_LANDSCAPE, ["semaglutide", "tirzepatide"], {"group_by": "sponsor_class"}),
    (Intent.SPONSOR_LANDSCAPE, ["pembrolizumab_since2015"], {"series_by": "phase"}),
    (Intent.INTERVENTION_LANDSCAPE, ["melanoma_ph2_3"], {}),
    (Intent.INTERVENTION_LANDSCAPE, ["semaglutide", "tirzepatide"], {"group_by": "intervention_type"}),
    (Intent.PHASE_STATUS_HEATMAP, ["obesity_since2020"], {}),
    (Intent.LANDSCAPE_GAP_HEATMAP, ["semaglutide"], {"group_by": "condition"}),
    (Intent.LANDSCAPE_GAP_HEATMAP, ["melanoma_ph2_3"], {"series_by": "year"}),
    (Intent.GEOGRAPHIC_DISTRIBUTION, ["t2d_recruiting"], {}),
    (Intent.GEOGRAPHIC_DISTRIBUTION, ["semaglutide", "tirzepatide"], {}),
    (Intent.ENROLLMENT_DISTRIBUTION, ["obesity_since2020"], {}),
    (Intent.ENROLLMENT_DISTRIBUTION, ["semaglutide", "tirzepatide"], {}),
    (Intent.ENROLLMENT_DISTRIBUTION, ["semaglutide"], {"preferred_visualization": VizType.SCATTER_PLOT}),
    (Intent.ENTITY_NETWORK, ["obesity_since2020"], {}),
    (Intent.ENTITY_NETWORK, ["melanoma_ph2_3"], {"network_kind": "drug_drug"}),
    (Intent.ENTITY_NETWORK, ["melanoma_ph2_3"], {"network_kind": "site_investigator"}),
    (Intent.SPONSOR_ACTIVITY, ["tirzepatide"], {}),
    (Intent.DEVELOPMENT_PROGRESSION, ["semaglutide"], {}),
    (Intent.DESIGN_METHODOLOGY, ["obesity_since2020"], {}),
    (Intent.DESIGN_METHODOLOGY, ["semaglutide", "tirzepatide"], {}),
]


def _undeclared_numbers(obj: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], float]]:
    if any(p.endswith("_derived") for p in path):
        return []
    if isinstance(obj, dict):
        return [x for k, v in obj.items() for x in _undeclared_numbers(v, (*path, str(k)))]
    if isinstance(obj, list):
        return [x for v in obj for x in _undeclared_numbers(v, path)]
    if isinstance(obj, int | float) and not isinstance(obj, bool):
        return [(path, obj)]
    return []


@pytest.mark.parametrize("intent,names,kw", CASES, ids=lambda v: getattr(v, "value", None) or str(v))
async def test_facts_numbers_are_in_data_or_derived(make_ctx, cohort, intent, names, kw):
    cohorts = [cohort(n, n.capitalize()) for n in names]
    res = await run(intent, make_ctx(intent, cohorts=cohorts, **kw))
    values = data_numbers(res.visualization)
    bad = [(p, v) for p, v in _undeclared_numbers(res.facts) if v not in values]
    assert not bad, f"facts not backed by data and not *_derived: {bad[:5]}"


async def test_design_cards_facts(make_ctx, study):
    s = study("NCT03548935")
    res = await run(Intent.DESIGN_METHODOLOGY, make_ctx(Intent.DESIGN_METHODOLOGY, trials=[s]))
    values = data_numbers(res.visualization)
    assert all(v in values for _, v in _undeclared_numbers(res.facts))
