"""Phase 0 contract tests: schemas, intent registry, provenance, Essie builder, reference handler."""

from __future__ import annotations

import pytest

from ctviz.api.schemas import Intent, VisualizeRequest, VisualizeResponse
from ctviz.api.schemas.plan import CohortFilter
from ctviz.core.intents import load_intents
from ctviz.core.provenance import Dataset, verify_citation
from ctviz.ctgov.requests import SearchSpec
from ctviz.viz.base import get_handler
from tests.conftest import assert_citations_verified, load_study_raw


def test_all_intents_registered_in_yaml():
    assert set(load_intents()) == set(Intent)


@pytest.mark.parametrize(
    "phase,expected",
    [("2", "PHASE2"), ("III", "PHASE3"), ("Phase 1", "PHASE1"), ("early_phase1", "EARLY_PHASE1"), ("NA", "NA")],
)
def test_request_phase_normalization(phase, expected):
    assert VisualizeRequest(query="q", trial_phase=[phase]).trial_phase == [expected]


def test_request_rejects_bad_years_and_nct():
    with pytest.raises(ValueError):
        VisualizeRequest(query="q", start_year=2020, end_year=2010)
    with pytest.raises(ValueError):
        VisualizeRequest(query="q", nct_id="NCT123")


def test_response_schema_exports():
    schema = VisualizeResponse.model_json_schema()
    assert "visualization" in schema["properties"]


def test_text_and_quote_refs_verify(study):
    s = study("NCT03548935")
    raw = load_study_raw("NCT03548935")
    q = s.quote_ref("eligibility_criteria", "Body mass index (BMI) greater than or equal to 30.0 kg/sqm")
    assert q is not None and verify_citation(q.to_citation(), raw)
    assert s.quote_ref("eligibility_criteria", "this sentence does not exist") is None
    assert s.ref("protocolSection.notAField") is None


def test_dataset_caps_citations(study):
    s = study("NCT03548935")
    ds = Dataset(max_citations=2, max_citations_single_trial=2)
    d = ds.emit({"x": 1}, [s.ref("phases"), s.ref("status"), s.ref("enrollment_count")])
    assert len(d.citations) == 2 and d.citations_total == 3 and len(ds.full_refs[d.id]) == 3
    assert d.nct_ids == ["NCT03548935"]


def test_aggregate_datum_lists_every_trial_but_cites_three(cohort):
    studies = cohort("semaglutide").studies[:8]
    ds = Dataset()
    d = ds.emit({"trial_count": 8}, [s.ref("nct_id") for s in studies])
    assert d.nct_ids == sorted(s.nct_id for s in studies) and len(d.nct_ids) == 8
    assert len(d.citations) == 3 and d.citations_total == 8
    assert [c.nct_id for c in d.citations] == d.nct_ids[:3]  # cited trials are the first by id: reproducible


def test_search_spec_from_cohort():
    spec = SearchSpec.from_cohort(
        CohortFilter(label="x", drugs=["semaglutide", "GLP-1 agonist"], phases=["PHASE2", "PHASE3"], start_year=2015)
    )
    p = spec.params()
    assert p["query.intr"] == 'semaglutide OR "GLP-1 agonist"'
    assert "AREA[Phase](PHASE2 OR PHASE3)" in p["filter.advanced"]
    assert "AREA[StartDate]RANGE[2015-01-01,MAX]" in p["filter.advanced"]


async def test_reference_phase_distribution_comparison(make_ctx, cohort):
    import ctviz.handlers  # noqa: F401  (registers handlers)

    a, b = cohort("semaglutide", "Semaglutide"), cohort("tirzepatide", "Tirzepatide")
    ctx = make_ctx(Intent.PHASE_DISTRIBUTION, cohorts=[a, b])
    result = await get_handler(Intent.PHASE_DISTRIBUTION)(ctx)
    viz = result.visualization
    assert viz.type.value == "bar_chart" and viz.encoding.series is not None
    assert sum(r.model_extra["trial_count"] for r in viz.data) == len(a.studies) + len(b.studies)
    raws = {s.nct_id: s.raw for s in a.studies + b.studies}
    assert assert_citations_verified(viz, raws) > 0
