"""ModelRouter task->model mapping and escalation rules; RequestBudget limits."""

from __future__ import annotations

import pytest

from ctviz.agents.router import ModelRouter, is_complex
from ctviz.api.schemas.response import LLMCallInfo
from ctviz.core.budget import BudgetExceeded, RequestBudget, budget_scope, current_budget
from ctviz.obs.pricing import ALLOWED_MODELS
from ctviz.obs.trace import trace_scope


def test_task_models(settings):
    r = ModelRouter(settings)
    assert r.model_for("planner") == settings.model_planner
    assert r.model_for("synthesis") == settings.model_synthesis
    assert r.model_for("extract") == settings.model_extract
    assert r.model_for("analysis") == settings.model_analysis
    assert r.model_for("planner", quality="max") == settings.model_max
    assert r.model_for("planner", complex_query=True) == settings.model_escalation
    assert r.model_for("synthesis", complex_query=True) == settings.model_synthesis
    for task in ("planner", "synthesis", "extract", "analysis", "retrieval", "escalation"):
        assert r.model_for(task) in ALLOWED_MODELS


def test_escalation_rules(settings):
    r = ModelRouter(settings)
    assert r.should_escalate(confidence=0.59)
    assert not r.should_escalate(confidence=0.6)
    assert r.should_escalate(schema_failures=1)
    assert r.should_escalate(verified_quote_ratio=0.4)
    assert not r.should_escalate()
    assert r.escalation_model(settings.model_planner) == settings.model_escalation
    assert r.escalation_model(settings.model_escalation) is None


def test_is_complex():
    assert not is_complex("How many obesity trials per year?")
    assert is_complex("word " * 50)
    assert is_complex("Compare semaglutide vs tirzepatide phase 3 recruiting trials since 2018 between sites?")


def test_budget_tool_cap_and_deadline():
    b = RequestBudget(max_tool_calls=2, deadline_s=60)
    b.charge_tool()
    b.charge_tool()
    with pytest.raises(BudgetExceeded):
        b.charge_tool()
    assert RequestBudget(deadline_s=0).remaining_s() == 0
    assert RequestBudget(deadline_s=0).timeout_for(5) == 0.1
    assert RequestBudget(deadline_s=60).timeout_for(5) == 5


def test_budget_llm_calls_from_trace(settings):
    with trace_scope() as tr, budget_scope(settings) as b:
        assert current_budget() is b and b.can_call_llm()
        for _ in range(settings.max_llm_calls):
            tr.add_llm(LLMCallInfo(agent="x", model="stub"))
        assert not b.can_call_llm()
