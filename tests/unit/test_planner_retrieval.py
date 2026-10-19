"""Retrieval agent tool loop (StubLLM + fake tools): validation, rejection/retry, limits, backoff, logging."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from ctviz.agents.llm import StubLLM
from ctviz.agents.retrieval_agent import (
    AgentStep,
    GetStudyArgs,
    ProbeCountArgs,
    RetrievalTools,
    SearchStudiesArgs,
    StepCohort,
    ToolRunner,
    TransientToolError,
    candidate_rewrites,
    resolve_retrieval,
)
from ctviz.api.schemas import Intent, QueryPlan
from ctviz.api.schemas.plan import CohortFilter
from ctviz.core.budget import budget_scope
from ctviz.obs.trace import trace_scope

PLAN = QueryPlan(intent=Intent.TREND_OVER_TIME, intent_confidence=0.5)
ZERO = CohortFilter(label="Pembro", drugs=["pembrolizumb"], phases=["PHASE3"])


def sc(**kw) -> StepCohort:
    base = dict(
        drugs=[],
        conditions=[],
        sponsors=[],
        countries=[],
        phases=[],
        statuses=[],
        start_year=None,
        end_year=None,
        term=None,
    )
    return StepCohort(**{**base, **kw})


def step(action, cohort=None, nct_id=None, page_size=None, reason="r") -> AgentStep:
    return AgentStep(action=action, reason=reason, cohort=cohort, page_size=page_size, nct_id=nct_id)


class Steps:
    def __init__(self, *steps):
        self.steps = list(steps)
        self.users: list[str] = []

    def __call__(self, user, schema):
        self.users.append(user)
        return self.steps.pop(0)


class FakeTools:
    """Counts by drug name; records every executed call."""

    def __init__(self, counts=None, fail_times=0, sleep_s=0.0):
        self.counts = {"pembrolizumab": 120} if counts is None else counts
        self.executed: list[tuple] = []
        self.fail_times = fail_times
        self.sleep_s = sleep_s

    async def probe(self, cf: CohortFilter) -> int:
        self.executed.append(("probe_count", cf))
        if self.sleep_s:
            await asyncio.sleep(self.sleep_s)
        if self.fail_times:
            self.fail_times -= 1
            raise TransientToolError("429")
        return sum(self.counts.get(d, 0) for d in cf.drugs) or (5 if cf.term else 0)

    async def search(self, cf: CohortFilter, n: int):
        self.executed.append(("search_studies", cf, n))
        return [{"nct_id": "NCT01295827", "title": "KEYNOTE-001"}]

    async def get(self, nct: str):
        self.executed.append(("get_study", nct))
        return {"protocolSection": {"identificationModule": {"nctId": nct, "briefTitle": "A study"}}}

    def tools(self) -> RetrievalTools:
        return RetrievalTools(probe_count=self.probe, search_studies=self.search, get_study=self.get)


async def go(llm, fake, settings, **kw):
    with trace_scope() as tr, budget_scope(settings):
        d = await resolve_retrieval(
            PLAN, "pembro trials per year", ZERO, llm=llm, settings=settings, tools=fake.tools(), backoff_s=0.0, **kw
        )
    return d, tr


# --------------------------------------------------------------------------- arg models
@pytest.mark.parametrize(
    "args",
    [
        {"cohort": {"drugs": ["x"], "phases": ["PHASE9"]}},
        {"cohort": {"drugs": ["x"], "statuses": ["OPEN"]}},
        {"cohort": {"drugs": ["x"], "start_year": 1980}},
        {"cohort": {"drugs": ["x"], "end_year": 2999}},
        {"cohort": {"drugs": ["x"], "start_year": 2020, "end_year": 2010}},
        {"cohort": {"term": "x" * 201}},
        {"cohort": {}},
        {"cohort": {"drugs": [""]}},
        {"cohort": {"drugs": ["x"], "bogus": 1}},
    ],
)
def test_probe_args_rejected(args):
    with pytest.raises(ValidationError):
        ProbeCountArgs.model_validate(args)


def test_search_page_size_limits():
    assert SearchStudiesArgs.model_validate({"cohort": {"drugs": ["x"]}, "page_size": 1000}).page_size == 1000
    with pytest.raises(ValidationError):
        SearchStudiesArgs.model_validate({"cohort": {"drugs": ["x"]}, "page_size": 1001})
    with pytest.raises(ValidationError):
        SearchStudiesArgs.model_validate({"cohort": {"drugs": ["x"]}, "page_size": 0})


@pytest.mark.parametrize(
    ("nct", "ok"),
    [
        ("NCT03548935", True),
        ("nct03548935", True),
        ("NCT0354893", False),
        ("NCT035489350", False),
        ("03548935", False),
        (None, False),
    ],
)
def test_get_study_nct_regex(nct, ok):
    if ok:
        assert GetStudyArgs.model_validate({"nct_id": nct}).nct_id == "NCT03548935"
    else:
        with pytest.raises(ValidationError):
            GetStudyArgs.model_validate({"nct_id": nct})


# --------------------------------------------------------------------------- runner
async def test_runner_rejects_invalid_without_executing(settings):
    fake = FakeTools()
    with trace_scope() as tr, budget_scope(settings):
        runner = ToolRunner(fake.tools())
        rec = await runner.run("get_study", {"nct_id": "NCT123"})
        rec2 = await runner.run("drop_table", {})
    assert not rec.ok and "nct_id" in rec.error and not rec2.ok
    assert fake.executed == [] and runner.calls == [] and len(runner.rejected) == 2
    step_ = tr.steps[0]
    assert step_.name == "tool:get_study" and step_.status == "error" and step_.detail["rejected"]


async def test_runner_logs_validated_inputs_and_summary(settings):
    fake = FakeTools()
    with trace_scope() as tr, budget_scope(settings):
        rec = await ToolRunner(fake.tools()).run("probe_count", {"cohort": {"drugs": ["pembrolizumab"]}})
    assert rec.ok and rec.summary == {"count": 120}
    s = tr.steps[0]
    assert s.name == "tool:probe_count" and s.status == "ok"
    assert s.detail["args"]["cohort"]["drugs"] == ["pembrolizumab"] and s.detail["result"] == {"count": 120}
    assert s.ms >= 0


async def test_runner_retries_transient_with_backoff(settings):
    fake = FakeTools(fail_times=2)
    with trace_scope() as tr, budget_scope(settings):
        rec = await ToolRunner(fake.tools(), retries=2, base_delay_s=0.0).run(
            "probe_count", {"cohort": {"drugs": ["pembrolizumab"]}}
        )
    assert rec.ok and len(fake.executed) == 3 and tr.steps[0].detail["retries"] == 2


async def test_runner_gives_up_after_retries(settings):
    fake = FakeTools(fail_times=5)
    with trace_scope(), budget_scope(settings):
        rec = await ToolRunner(fake.tools(), retries=1, base_delay_s=0.0).run(
            "probe_count", {"cohort": {"drugs": ["pembrolizumab"]}}
        )
    assert not rec.ok and "TransientToolError" in rec.error and len(fake.executed) == 2


async def test_runner_timeout(settings):
    fake = FakeTools(sleep_s=0.5)
    with trace_scope(), budget_scope(settings):
        rec = await ToolRunner(fake.tools(), timeout_s=0.05, retries=0).run(
            "probe_count", {"cohort": {"drugs": ["pembrolizumab"]}}
        )
    assert not rec.ok and "TimeoutError" in rec.error


async def test_runner_hard_call_limit(settings):
    fake = FakeTools()
    with trace_scope(), budget_scope(settings):
        runner = ToolRunner(fake.tools(), max_calls=3)
        for _ in range(4):
            await runner.run("probe_count", {"cohort": {"drugs": ["pembrolizumab"]}})
    assert len(runner.calls) == 3 and len(fake.executed) == 3 and "limit" in runner.rejected[0].error


# --------------------------------------------------------------------------- loop
async def test_llm_loop_probe_then_finish(settings):
    good = sc(drugs=["pembrolizumab"], phases=["PHASE3"])
    llm = StubLLM({"retrieval": Steps(step("probe_count", good), step("finish", good))})
    d, tr = await go(llm, FakeTools(), settings)
    assert d.used_llm and d.count == 120 and d.cohort.drugs == ["pembrolizumab"] and d.cohort.label == "Pembro"
    assert d.tool_calls == 1 and [s.name for s in tr.steps] == ["tool:probe_count"]


async def test_invalid_proposal_is_rejected_and_rules_take_over(settings):
    bad = sc(drugs=["pembrolizumab"], phases=["PHASE 3"])
    steps = Steps(step("probe_count", bad), step("finish", None))
    fake = FakeTools()
    d, tr = await go(StubLLM({"retrieval": steps}), fake, settings)
    assert len(steps.users) == 1  # at most one LLM call
    assert len(d.rejected) == 1 and "phases" in d.rejected[0].error
    assert d.count == 120 and "rule rewrite" in d.rationale
    assert tr.steps[0].detail["rejected"] and all(e[0] == "probe_count" for e in fake.executed)


async def test_alternatives_are_tried_in_order(settings):
    s1 = sc(drugs=["pembrolizumab"], phases=["PHASE3"])
    s2 = sc(drugs=["pembrolizumab"])
    fake = FakeTools(counts={"pembrolizumab": 0})
    steps = Steps(
        AgentStep(action="probe_count", reason="r", cohort=s1, page_size=None, nct_id=None, alternatives=[s2])
    )
    d, _ = await go(StubLLM({"retrieval": steps}), fake, settings)
    assert [e[1].phases for e in fake.executed[:2]] == [["PHASE3"], []]


@pytest.mark.parametrize(
    ("proposal", "why"),
    [
        (sc(drugs=["nivolumab"]), "not aliases"),  # different drug
        (sc(drugs=["pembrolizumab"], conditions=["melanoma"]), "conditions"),  # adds a condition
        (sc(drugs=["pembrolizumab"], phases=["PHASE2"]), "phase"),  # adds a phase filter
        (sc(drugs=["pembrolizumab"], start_year=2001), "start_year"),  # changes a year
        (sc(term="pembrolizumab melanoma"), "term adds"),  # term adds words
        (sc(drugs=["pembrolizumab OR nivolumab"]), "operators"),  # Essie operator
        (sc(term="AREA[Phase]PHASE3"), "operators"),
    ],
)
async def test_unfaithful_or_operator_rewrites_rejected(settings, proposal, why):
    fake = FakeTools(counts={"pembrolizumab": 0, "nivolumab": 50})
    d, _ = await go(StubLLM({"retrieval": Steps(step("probe_count", proposal))}), fake, settings)
    assert d.rejected and why in d.rejected[0].error
    assert all("nivolumab" not in e[1].drugs and "melanoma" not in e[1].conditions for e in fake.executed)


async def test_chosen_rewrite_gets_approximated_note(settings):
    good = sc(drugs=["pembrolizumab"])
    d, _ = await go(StubLLM({"retrieval": Steps(step("probe_count", good))}), FakeTools(), settings)
    (note,) = d.notes
    assert note.code.value == "APPROXIMATED"
    assert "pembrolizumb" in note.message and "pembrolizumab" in note.message and "PHASE3" in note.message


async def test_ambiguity_probes_each_meaning_under_its_kind(settings):
    seen: list[CohortFilter] = []

    async def probe(cf):
        seen.append(cf)
        return 30 if "Merck Sharp & Dohme" in cf.sponsors else 0

    tools = RetrievalTools(probe_count=probe, search_studies=FakeTools().search, get_study=FakeTools().get)
    with trace_scope(), budget_scope(settings):
        d = await resolve_retrieval(
            PLAN,
            "merck trials",
            CohortFilter(label="Merck"),
            llm=None,
            settings=settings,
            tools=tools,
            ambiguous={"merck": ["Merck Sharp & Dohme", "Merck KGaA"]},
        )
    assert [c.sponsors for c in seen] == [["Merck Sharp & Dohme"], ["Merck KGaA"]]
    assert d.cohort.sponsors == ["Merck Sharp & Dohme"] and d.count == 30
    assert d.notes[0].code.value == "APPROXIMATED" and "Merck KGaA" in d.notes[0].message


async def test_ambiguity_with_several_hits_asks(settings):
    async def probe(cf):
        return 10

    tools = RetrievalTools(probe_count=probe, search_studies=FakeTools().search, get_study=FakeTools().get)
    with trace_scope(), budget_scope(settings):
        d = await resolve_retrieval(
            PLAN,
            "AD trials",
            CohortFilter(label="AD"),
            llm=None,
            settings=settings,
            tools=tools,
            ambiguous={"AD": ["Alzheimer's disease", "atopic dermatitis"]},
        )
    assert d.cohort is None and "Alzheimer's disease" in d.clarification and "atopic dermatitis" in d.clarification
    assert d.notes[0].code.value == "ENTITY_AMBIGUOUS"


async def test_non_registry_meaning_is_not_probed(settings):
    seen: list[CohortFilter] = []

    async def probe(cf):
        seen.append(cf)
        return 10

    tools = RetrievalTools(probe_count=probe, search_studies=FakeTools().search, get_study=FakeTools().get)
    with trace_scope(), budget_scope(settings):
        d = await resolve_retrieval(
            PLAN,
            "MS trials",
            CohortFilter(label="MS"),
            llm=None,
            settings=settings,
            tools=tools,
            ambiguous={"MS": ["multiple sclerosis", "mass spectrometry"]},
        )
    assert [c.conditions for c in seen] == [["multiple sclerosis"]]  # "mass spectrometry" has no registry kind
    assert d.cohort.conditions == ["multiple sclerosis"]


async def test_retry_stops_when_deadline_is_too_close(settings):
    fake = FakeTools(fail_times=5)
    with trace_scope(), budget_scope(settings) as b:
        b.deadline_s = 0.05
        rec = await ToolRunner(fake.tools(), retries=3, base_delay_s=1.0).run(
            "probe_count", {"cohort": {"drugs": ["pembrolizumab"]}}
        )
    assert not rec.ok and len(fake.executed) == 1  # no retry: backoff would overrun the deadline


async def test_loop_never_exceeds_three_tool_calls(settings):
    s = sc(drugs=["nothing"])
    steps = Steps(*[step("probe_count", s) for _ in range(5)])
    fake = FakeTools(counts={})
    d, _ = await go(StubLLM({"retrieval": steps}), fake, settings)
    assert d.tool_calls <= 3 and len(fake.executed) <= 3 and len(steps.users) == 1
    assert not any(e[1].drugs == ["nothing"] for e in fake.executed)  # unfaithful proposal never ran


async def test_get_study_collects_nct(settings):
    steps = Steps(step("get_study", nct_id="nct01295827"), step("finish", None))
    d, _ = await go(StubLLM({"retrieval": steps}), FakeTools(), settings)
    assert d.nct_ids == ["NCT01295827"]


async def test_offline_uses_rule_rewrites(settings):
    d, tr = await go(StubLLM(), FakeTools(), settings)
    assert not d.used_llm and d.count == 120 and d.cohort.drugs == ["pembrolizumab"]
    assert all(s.name == "tool:probe_count" for s in tr.steps)


async def test_no_llm_object(settings):
    d, _ = await go(None, FakeTools(), settings)
    assert d.count == 120


def test_candidate_rewrites_order():
    cands = candidate_rewrites(CohortFilter(label="x", drugs=["GLP-1"], phases=["PHASE3"]))
    assert "semaglutide" in cands[0].drugs and cands[0].phases == ["PHASE3"]
    assert cands[1].phases == []
    assert cands[-1].term == "GLP-1"
