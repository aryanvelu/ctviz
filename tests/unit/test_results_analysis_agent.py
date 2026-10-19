"""Analysis agent: validated id-only LLM selection with regex fallback (mocked LLM, offline)."""

from __future__ import annotations

import asyncio

import pytest

from ctviz.agents.analysis_agent import (
    LLMPick,
    build_catalog,
    rule_select,
    select_measures,
    validate_pick,
)
from ctviz.agents.llm import LLMError, StubLLM
from ctviz.api.schemas import Intent
from ctviz.handlers import results as R
from ctviz.obs.trace import trace_scope


def stub(pick: LLMPick) -> StubLLM:
    return StubLLM({"analysis": lambda user, schema: pick})


class SlowLLM:
    async def structured(self, **kw):
        await asyncio.sleep(5)


class BrokenLLM:
    async def structured(self, **kw):
        raise LLMError("analysis: boom")


@pytest.fixture
def fast_settings(settings):
    return settings.model_copy(update={"llm_timeout_s": 0.05})


def test_catalog_survival_filters_and_has_no_numbers(study):
    cat = build_catalog(study("NCT02578680"), "survival")
    titles = [m["title"] for m in cat["measures"]]
    assert titles and all("Surviv" in t or "PFS" in t for t in titles)
    text = str(cat)
    assert "11.3" not in text and "0.49" not in text


def test_rule_select_survival_keynote189(study):
    sel = rule_select(study("NCT02578680"), "survival curve", "survival")
    assert sel.method == "rules" and set(sel.measure_indices) == {0, 1}  # PFS + OS medians, not irRECIST (6)


def test_rule_select_pfs_only(study):
    sel = rule_select(study("NCT01844505"), "progression-free survival", "survival")
    assert sel.measure_indices == [0, 3]


def test_rule_select_results_primary(study):
    sel = rule_select(study("NCT03548935"), "what were the results?", "results")
    assert sel.measure_indices == [0, 1]


async def test_valid_llm_pick_is_used(study, settings):
    s = study("NCT02578680")
    with trace_scope() as tr:
        sel = await select_measures(
            s,
            "overall survival",
            "survival",
            llm=stub(LLMPick(purpose="survival", measure_indices=[1], rationale="OS")),
            settings=settings,
        )
    assert sel.method == "llm" and sel.measure_indices == [1] and not sel.rejected
    step = next(st for st in tr.steps if st.name == "analysis.select_measures")
    assert step.detail["method"] == "llm" and step.status == "ok"
    assert tr.llm_calls and tr.llm_calls[0].agent == "analysis"


@pytest.mark.parametrize(
    "pick,reason",
    [
        (LLMPick(purpose="survival", measure_indices=[99]), "bad_indices:n=1"),  # out of range
        (LLMPick(purpose="survival", measure_indices=[2]), "bad_indices:n=1"),  # exists, but not sent
        (LLMPick(purpose="survival", measure_indices=[1], group_ids=["OG999"]), "bad_group_ids:n=1"),  # invalid id
        (LLMPick(purpose="results", measure_indices=[1]), "bad_purpose"),  # wrong purpose
        (LLMPick(purpose="survival", measure_indices=[]), "no_measures"),
    ],
)
async def test_invalid_llm_pick_falls_back(study, settings, pick, reason):
    s = study("NCT02578680")
    with trace_scope() as tr:
        sel = await select_measures(s, "survival", "survival", llm=stub(pick), settings=settings)
    assert sel.method == "rules" and set(sel.measure_indices) == {0, 1}
    assert reason in sel.rejected
    step = next(st for st in tr.steps if st.name == "analysis.select_measures")
    assert step.status == "degraded" and step.detail["rejected"]


def test_too_many_selections_rejected(study):
    s = study("NCT03548935")
    cat = build_catalog(s, "results")
    problems = validate_pick(LLMPick(purpose="results", measure_indices=list(range(10))), cat, "results")
    assert "too_many_measures" in problems


async def test_timeout_falls_back(study, fast_settings):
    sel = await select_measures(study("NCT02578680"), "survival", "survival", llm=SlowLLM(), settings=fast_settings)
    assert sel.method == "rules" and sel.rejected == ["timeout"]


async def test_llm_error_and_unavailable_fall_back(study, settings):
    s = study("NCT02578680")
    sel = await select_measures(s, "survival", "survival", llm=BrokenLLM(), settings=settings)
    assert sel.method == "rules" and sel.rejected
    sel = await select_measures(s, "survival", "survival", llm=StubLLM(), settings=settings)  # offline stub
    assert sel.method == "rules" and set(sel.measure_indices) == {0, 1}
    sel = await select_measures(s, "survival", "survival", llm=None, settings=settings)
    assert sel.method == "rules" and not sel.rejected


async def test_unknown_purpose_raises(study, settings):
    with pytest.raises(ValueError):
        await select_measures(study("NCT02578680"), "q", "power", llm=None, settings=settings)  # type: ignore[arg-type]


async def test_handler_records_rejection(make_ctx, study):
    ctx = make_ctx(Intent.SURVIVAL_SUMMARY, trials=[study("NCT02578680")], query="survival")
    ctx.llm = stub(LLMPick(purpose="survival", measure_indices=[42]))
    res = await R.survival_summary(ctx)
    analysis = res.visualization.meta.diagnostics["analysis"]
    assert analysis["method"] == "rules" and analysis["llm_rejected"] == ["bad_indices:n=1"]
    assert res.visualization.type.value == "survival_curve"


async def test_handler_uses_valid_llm_selection(make_ctx, study):
    ctx = make_ctx(Intent.RESULTS_OUTCOMES, trials=[study("NCT03548935")], query="waist")
    ctx.llm = stub(LLMPick(purpose="results", measure_indices=[5]))
    res = await R.results_outcomes(ctx)
    assert res.visualization.meta.diagnostics["analysis"] == {"method": "llm", "measure_indices": [5]}
    assert res.visualization.type.value == "bar_chart"


# -- review fixes ------------------------------------------------------------------------------------


async def test_rejection_codes_never_echo_model_output(make_ctx, study, settings):
    """Item 8: model-written strings (ids, purpose, rationale) never reach the response or trace."""
    evil = "IGNORE_ALL_INSTRUCTIONS"
    pick = LLMPick(purpose="survival", measure_indices=[1, 777], group_ids=[evil], rationale=f"{evil} 99.9%")
    ctx = make_ctx(Intent.SURVIVAL_SUMMARY, trials=[study("NCT02578680")], query="survival")
    ctx.llm = stub(pick)
    with trace_scope() as tr:
        res = await R.survival_summary(ctx)
    analysis = res.visualization.meta.diagnostics["analysis"]
    assert analysis["llm_rejected"] == ["bad_indices:n=1", "bad_group_ids:n=1"]
    dumped = res.visualization.model_dump_json() + str([n.message for n in res.notes]) + str(tr.steps)
    assert evil not in dumped and "777" not in dumped


async def test_valid_pick_rationale_not_propagated(study, settings):
    pick = LLMPick(purpose="survival", measure_indices=[1], rationale="model text 42")
    sel = await select_measures(study("NCT02578680"), "os", "survival", llm=stub(pick), settings=settings)
    assert sel.method == "llm" and "model text" not in sel.rationale


def test_llmpick_purpose_max_length():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LLMPick(purpose="x" * 17)


def test_survival_catalog_has_no_all_measures_fallback(study):
    """Item 2: a trial without survival measures yields an empty survival catalog (no fallback)."""
    cat = build_catalog(study("NCT03548935"), "survival")
    assert cat["measures"] == []
    sel = rule_select(study("NCT03548935"), "survival", "survival")
    assert sel.measure_indices == []
