"""Extraction agent with a mocked LLM: chunk-id grounding, quote verification, clamping, fallbacks."""

from __future__ import annotations

import json

import pytest

import ctviz.handlers.trial_text  # noqa: F401  (registers handlers)
from ctviz.agents.extractors import LLMExtraction, LLMVerdict, extract_criteria, extract_criteria_detailed
from ctviz.agents.llm import StubLLM
from ctviz.api.schemas import Intent, VisualizeRequest
from ctviz.api.schemas.request import PatientProfile
from ctviz.core.provenance import verify_citation
from ctviz.obs.trace import trace_scope
from ctviz.viz.base import get_handler
from tests.conftest import assert_citations_verified, load_study_raw

NCT = "NCT03548935"
PROFILE = PatientProfile(age=45, sex="female", conditions=["obesity"], medications=["metformin"])
BMI_QUOTE = "Body mass index (BMI) greater than or equal to 30.0 kg/sqm"


def _chunks(user: str) -> list[dict]:
    return json.loads(user)["chunks"]


def _find(chunks: list[dict], needle: str) -> dict:
    return next(c for c in chunks if needle in c["text"])


class Recorder:
    """StubLLM responder that records calls and builds verdicts from the chunks it was given."""

    def __init__(self, build):
        self.build = build
        self.calls: list[str] = []

    def __call__(self, user: str, schema):
        self.calls.append(user)
        return schema(verdicts=self.build(_chunks(user)))


def _llm(build) -> tuple[StubLLM, Recorder]:
    rec = Recorder(build)
    return StubLLM({"extraction": rec}), rec


def _row(outcome, needle):
    return next(v for v in outcome.verdicts if needle in v.text)


async def test_valid_citation_is_accepted_and_verified(study, settings):
    s = study(NCT)

    def build(chunks):
        bmi = _find(chunks, "Body mass index")
        return [
            LLMVerdict(
                chunk_id=bmi["chunk_id"], criterion=bmi["criterion"], applies="yes", confidence="high", quote=BMI_QUOTE
            )
        ]

    llm, rec = _llm(build)
    out = await extract_criteria_detailed(s, llm=llm, settings=settings, profile=PROFILE, focus_terms=[])
    assert len(rec.calls) == 1 and out.llm_status == "ok" and out.accepted == 1 and not out.rejected
    row = _row(out, "Body mass index")
    # accepted and cited, but the quote never names a profile term ("obesity"): not final (review item 1/7)
    assert row.status == "needs_review" and row.evaluated_by == "llm" and BMI_QUOTE in row.reason
    raw = load_study_raw(NCT)
    assert any(r.excerpt == BMI_QUOTE for r in row.refs)
    assert all(verify_citation(r.to_citation(), raw) for r in row.refs)
    # chunk ids are stable and encode the exact span
    nct, path, span = row.chunk_id.split(":")
    start, end = map(int, span.split("-"))
    assert nct == NCT and raw["protocolSection"]["eligibilityModule"]["eligibilityCriteria"][start:end] == row.text


async def test_hallucinated_chunk_id_is_rejected(study, settings):
    def build(chunks):
        return [
            LLMVerdict(
                chunk_id=f"{NCT}:protocolSection.eligibilityModule.eligibilityCriteria:0-10",
                criterion=1,
                applies="no",
                confidence="high",
                quote="Inclusion",
            )
        ]

    llm, _ = _llm(build)
    out = await extract_criteria_detailed(study(NCT), llm=llm, settings=settings, profile=PROFILE, focus_terms=[])
    assert out.accepted == 0 and out.rejected[0][1] == "chunk_id_not_provided"
    assert all(v.evaluated_by != "llm" for v in out.verdicts)


async def test_quote_not_in_chunk_is_rejected(study, settings):
    def build(chunks):
        hba1c = _find(chunks, "HbA1C")
        bmi = _find(chunks, "Body mass index")
        return [
            # hallucinated text that appears nowhere in the record
            LLMVerdict(
                chunk_id=hba1c["chunk_id"],
                criterion=hba1c["criterion"],
                applies="no",
                confidence="high",
                quote="HbA1c below 5% is required",
            ),
            # real record text, but from a different chunk than the one cited
            LLMVerdict(
                chunk_id=hba1c["chunk_id"],
                criterion=hba1c["criterion"],
                applies="no",
                confidence="high",
                quote=BMI_QUOTE,
            ),
            # empty quote
            LLMVerdict(
                chunk_id=bmi["chunk_id"], criterion=bmi["criterion"], applies="yes", confidence="high", quote="   "
            ),
        ]

    llm, _ = _llm(build)
    out = await extract_criteria_detailed(study(NCT), llm=llm, settings=settings, profile=PROFILE, focus_terms=[])
    assert out.accepted == 0
    assert [r for _, r in out.rejected] == ["quote_not_in_chunk", "quote_not_in_chunk", "quote_too_short"]
    assert _row(out, "HbA1C").status == "unknown" and _row(out, "HbA1C").evaluated_by == "none"


async def test_index_mismatch_invalid_enum_and_duplicates_rejected(study, settings):
    def build(chunks):
        bmi = _find(chunks, "Body mass index")
        good = LLMVerdict(
            chunk_id=bmi["chunk_id"],
            criterion=bmi["criterion"],
            applies="yes",
            confidence="high",
            quote="Body mass index",
        )
        return [
            LLMVerdict(
                chunk_id=bmi["chunk_id"], criterion=999, applies="yes", confidence="high", quote="Body mass index"
            ),
            LLMVerdict.model_construct(
                chunk_id=bmi["chunk_id"],
                criterion=bmi["criterion"],
                applies="definitely",
                confidence="high",
                quote="Body mass index",
            ),
            good,
            good,
        ]

    llm, _ = _llm(build)
    out = await extract_criteria_detailed(study(NCT), llm=llm, settings=settings, profile=PROFILE, focus_terms=[])
    assert [r for _, r in out.rejected] == ["criterion_index_mismatch", "invalid_enum", "duplicate_verdict"]
    assert out.accepted == 1


async def test_low_confidence_and_unclear_clamp_to_needs_review(study, settings):
    def build(chunks):
        bmi = _find(chunks, "Body mass index")
        hba1c = _find(chunks, "HbA1C")
        return [
            LLMVerdict(
                chunk_id=bmi["chunk_id"],
                criterion=bmi["criterion"],
                applies="yes",
                confidence="medium",
                quote="Body mass index",
            ),
            LLMVerdict(
                chunk_id=hba1c["chunk_id"],
                criterion=hba1c["criterion"],
                applies="unclear",
                confidence="high",
                quote="Glycated haemoglobin",
            ),
        ]

    llm, _ = _llm(build)
    out = await extract_criteria_detailed(study(NCT), llm=llm, settings=settings, profile=PROFILE, focus_terms=[])
    assert _row(out, "Body mass index").status == "needs_review"
    assert _row(out, "HbA1C").status == "needs_review"


async def test_exclusion_semantics(study, settings):
    def build(chunks):
        preg = _find(chunks, "pregnant")
        return [
            LLMVerdict(
                chunk_id=preg["chunk_id"],
                criterion=preg["criterion"],
                applies="yes",
                confidence="high",
                quote="Female who is pregnant",
            )
        ]

    llm, _ = _llm(build)
    # grounded: the person asked about pregnancy, so a confident "applies" on the exclusion is final
    out = await extract_criteria_detailed(
        study(NCT), llm=llm, settings=settings, profile=PROFILE, focus_terms=["pregnant"]
    )
    assert _row(out, "pregnant").status == "unmet"  # exclusion criterion that applies blocks eligibility
    # ungrounded: nothing in the profile mentions pregnancy -> the same verdict stays needs_review
    llm, _ = _llm(build)
    out = await extract_criteria_detailed(study(NCT), llm=llm, settings=settings, profile=PROFILE, focus_terms=[])
    assert _row(out, "pregnant").status == "needs_review"


async def test_rules_only_fallback_when_llm_unavailable(study, settings):
    with trace_scope() as tr:
        rows = await extract_criteria(study(NCT), llm=StubLLM(), settings=settings, profile=PROFILE, focus_terms=[])
    structured = [r for r in rows if r.section == "structured"]
    assert {r.evaluated_by for r in structured} == {"rule"}
    assert {r.evaluated_by for r in rows if r.section != "structured"} == {"none"}
    assert next(r for r in structured if r.text.startswith("Age")).status == "met"
    steps = {s.name: s for s in tr.steps}
    assert steps["extract.llm"].status == "degraded" and steps["extract.llm"].detail["result"] == "unavailable"
    assert "metformin" not in json.dumps(steps["extract.llm"].detail)  # profile values never logged


async def test_no_profile_means_no_llm_call(study, settings):
    llm, rec = _llm(lambda chunks: [])
    out = await extract_criteria_detailed(study(NCT), llm=llm, settings=settings, profile=None, focus_terms=[])
    assert rec.calls == [] and not out.llm_called


async def test_medication_mode_only_sends_medication_chunks(study, settings):
    llm, rec = _llm(lambda chunks: [])
    s = study("NCT02578680")
    out = await extract_criteria_detailed(
        s, llm=llm, settings=settings, profile=PROFILE, focus_terms=["aspirin"], mode="medications"
    )
    sent = _chunks(rec.calls[0])
    assert sent and len(sent) == out.chunks_provided
    assert any("aspirin" in c["text"] for c in sent)
    assert all(v.section != "structured" for v in out.verdicts)


# -- through the handlers --------------------------------------------------------------------------


@pytest.fixture
def eligibility_ctx(make_ctx, study):
    def make(intent, nct, build, profile=PROFILE, focus=None):
        s = study(nct)
        ctx = make_ctx(intent, trials=[s], persona="patient", focus_terms=focus or [])
        ctx.request = VisualizeRequest(query="Am I eligible?", patient_profile=profile)
        ctx.llm, rec = _llm(build)
        return ctx, rec

    return make


async def test_eligibility_handler_valid_and_hallucinated(eligibility_ctx):
    def build(chunks):
        bmi = _find(chunks, "Body mass index")
        hba1c = _find(chunks, "HbA1C")
        return [
            LLMVerdict(
                chunk_id=bmi["chunk_id"], criterion=bmi["criterion"], applies="yes", confidence="high", quote=BMI_QUOTE
            ),
            LLMVerdict(
                chunk_id=hba1c["chunk_id"],
                criterion=hba1c["criterion"],
                applies="no",
                confidence="high",
                quote="HbA1c under 5.7% for all participants",
            ),
        ]

    ctx, rec = eligibility_ctx(Intent.ELIGIBILITY_CHECK, NCT, build)
    res = await get_handler(Intent.ELIGIBILITY_CHECK)(ctx)
    viz = res.visualization
    assert viz.type.value == "eligibility_checklist" and len(rec.calls) == 1
    assert assert_citations_verified(viz, {NCT: load_study_raw(NCT)}) >= len(viz.data)
    rows = [d.model_extra for d in viz.data]
    assert sum(r["evaluated_by"] == "llm" for r in rows) == 1
    assert viz.meta.summary["by_evaluator"]["llm"] == 1
    assert viz.meta.diagnostics["extraction"]["rejected"][0]["reason"] == "quote_not_in_chunk"
    assert viz.meta.diagnostics["profile_used"]["age"] == 45
    codes = [n.code.value for n in res.notes]
    assert codes.count("LLM_EVALUATED") == 2  # one for accepted verdicts, one warning for the discarded one


async def test_eligibility_handler_nothing_relevant_found(eligibility_ctx):
    ctx, _ = eligibility_ctx(Intent.ELIGIBILITY_CHECK, NCT, lambda chunks: [])
    res = await get_handler(Intent.ELIGIBILITY_CHECK)(ctx)
    assert any(n.code.value == "NOT_IN_REGISTRY" and "could be matched" in n.message for n in res.notes)
    assert all((d.model_extra or {})["evaluated_by"] != "llm" for d in res.visualization.data)


async def test_medications_handler_nothing_relevant_says_so(eligibility_ctx, study, settings):
    # STEP 1 criteria never name metformin -> explicit "not named" note, no guessing.
    ctx, _ = eligibility_ctx(Intent.CONCOMITANT_MEDICATIONS, NCT, lambda chunks: [])
    res = await get_handler(Intent.CONCOMITANT_MEDICATIONS)(ctx)
    assert any("Not named in the eligibility criteria: metformin" in n.message for n in res.notes)


async def test_medications_handler_all_claims_rejected(eligibility_ctx):
    def build(chunks):
        return [LLMVerdict(chunk_id="made-up", criterion=1, applies="yes", confidence="high", quote="aspirin")]

    ctx, _ = eligibility_ctx(Intent.CONCOMITANT_MEDICATIONS, "NCT02578680", build, focus=["aspirin"])
    res = await get_handler(Intent.CONCOMITANT_MEDICATIONS)(ctx)
    rows = [d.model_extra for d in res.visualization.data]
    assert all(r["evaluated_by"] == "none" for r in rows)
    aspirin = next(r for r in rows if "aspirin" in r["text"])
    assert aspirin["relevance"] == "high" and aspirin["status"] == "unknown"
    assert any(n.code.value == "NOT_IN_REGISTRY" and "could be matched" in n.message for n in res.notes)
    assert res.visualization.meta.diagnostics["extraction"]["rejected"] == [
        {"chunk_id": "made-up", "reason": "chunk_id_not_provided"}
    ]


def test_llm_schema_accepts_bad_enum_so_code_rejects_per_verdict():
    """A bad enum must not invalidate the whole response (StubLLM re-validates); code rejects it alone."""
    out = LLMExtraction.model_validate(
        {"verdicts": [{"chunk_id": "x", "criterion": 1, "applies": "maybe", "confidence": "high", "quote": "q"}]}
    )
    assert out.verdicts[0].applies == "maybe"
