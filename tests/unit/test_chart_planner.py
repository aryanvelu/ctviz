"""Chart planner agent: validated menu, never overrides an explicit view, falls back on bad picks."""

from __future__ import annotations

from types import SimpleNamespace

from ctviz.agents.chart_planner import ChartPick, choose, menu, profile
from ctviz.api.schemas.plan import CohortFilter, Intent, QueryPlan
from ctviz.api.schemas.viz import VizType


class FakeLLM:
    def __init__(self, pick):
        self.pick, self.calls = pick, 0

    async def structured(self, **kw):
        self.calls += 1
        assert "profile" in kw["user"] and kw["schema"] is ChartPick
        return self.pick


SETTINGS = SimpleNamespace(llm_timeout_s=2.0, model_analysis="gpt-5.4-mini")


def _plan(intent=Intent.PHASE_DISTRIBUTION):
    return QueryPlan(intent=intent, intent_confidence=0.6, cohorts=[CohortFilter(label="Sema", drugs=["semaglutide"])])


def test_profile_counts_categories_only(cohort):
    p = profile([cohort("semaglutide", "Sema")])
    assert p["trials"] > 0 and p["phases"]["distinct"] >= 1 and "studies" not in p


async def test_open_question_lets_the_agent_pick_a_better_view(cohort):
    pick = ChartPick(intent=Intent.TREND_OVER_TIME, visualization=VizType.TIME_SERIES, reason="Starts vary by year.")
    llm = FakeLLM(pick)
    d = await choose(
        _plan(),
        "tell me about semaglutide trials",
        [cohort("semaglutide", "Sema")],
        llm=llm,
        settings=SETTINGS,
        explicit_viz=False,
    )
    assert d.method == "llm" and d.plan.intent is Intent.TREND_OVER_TIME and llm.calls == 1


async def test_question_that_names_its_view_keeps_the_intent_without_an_llm_call(cohort):
    q = "How are semaglutide trials distributed across phases?"
    assert list(menu(_plan(), q, [cohort("semaglutide", "Sema")])["intents"]) == ["phase_distribution"]
    llm = FakeLLM(ChartPick(intent=Intent.ENTITY_NETWORK, visualization=VizType.NETWORK_GRAPH, reason="x"))
    d = await choose(_plan(), q, [cohort("semaglutide", "Sema")], llm=llm, settings=SETTINGS, explicit_viz=False)
    assert d.method == "rules" and d.plan.intent is Intent.PHASE_DISTRIBUTION and llm.calls == 0


async def test_split_only_when_the_question_mentions_it(cohort):
    pick = ChartPick(intent=Intent.TREND_OVER_TIME, visualization=VizType.TIME_SERIES, split="status_group", reason="x")
    d = await choose(
        _plan(),
        "tell me about semaglutide trials",
        [cohort("semaglutide", "Sema")],
        llm=FakeLLM(pick),
        settings=SETTINGS,
        explicit_viz=False,
    )
    assert d.method == "rules" and d.plan.series_by is None  # never split by status unless asked
    d2 = await choose(
        _plan(),
        "tell me about active vs completed semaglutide trials",
        [cohort("semaglutide", "Sema")],
        llm=FakeLLM(pick),
        settings=SETTINGS,
        explicit_viz=False,
    )
    assert d2.method == "llm" and d2.plan.series_by == "status_group"


async def test_incompatible_viz_or_requested_viz_is_respected(cohort):
    q = "tell me about semaglutide trials"
    bad = ChartPick(intent=Intent.PHASE_DISTRIBUTION, visualization=VizType.SANKEY, reason="x")
    d = await choose(
        _plan(), q, [cohort("semaglutide", "Sema")], llm=FakeLLM(bad), settings=SETTINGS, explicit_viz=False
    )
    assert d.method == "rules"
    pie = ChartPick(intent=Intent.PHASE_DISTRIBUTION, visualization=VizType.PIE_CHART, reason="x")
    d = await choose(
        _plan(), q, [cohort("semaglutide", "Sema")], llm=FakeLLM(pie), settings=SETTINGS, explicit_viz=True
    )
    assert d.method == "rules" and "fixed the visualization" in " ".join(d.rejected)


async def test_no_llm_means_no_change(cohort):
    d = await choose(
        _plan(),
        "tell me about semaglutide",
        [cohort("semaglutide", "Sema")],
        llm=None,
        settings=SETTINGS,
        explicit_viz=False,
    )
    assert d.method == "rules" and d.plan == _plan()
