"""Systemic guarantees added after manual testing (each test covers a class of questions, not one query).

1. Subjective verdicts ("which drug is safest", "is X safe", "should I take X") -> status=unanswerable, no
   chart, no ClinicalTrials.gov data fetched.
2. Names that match no trial -> status=needs_clarification; the search is never widened to "All trials".
3. Cohorts larger than max_studies -> exact per-bucket registry counts that reconcile to totalCount
   (checked against a brute-force count over a synthetic registry with a mini Essie evaluator).
4. The final validation step catches count/trial-list mismatches.
5. The public JSON keeps the assignment's {visualization: {type,title,encoding,data}, meta} shape.
"""

from __future__ import annotations

import datetime as dt
import re
from types import SimpleNamespace
from typing import Any

import pytest

from ctviz.agents.llm import StubLLM
from ctviz.agents.validation import check_counts
from ctviz.api.public import MAX_NCT_IDS_PER_DATUM, PublicResponse, to_public
from ctviz.api.schemas.plan import CohortFilter, Intent, QueryPlan, Scope, UnverifiedTerm
from ctviz.api.schemas.request import VisualizeRequest
from ctviz.api.schemas.response import NoteCode
from ctviz.api.schemas.viz import Datum, VizType
from ctviz.core.answerability import is_subjective
from ctviz.core.background import drain
from ctviz.core.orchestrator import Orchestrator
from ctviz.ctgov.client import SearchPage
from ctviz.handlers.registry_counts import PHASE_BUCKETS
from ctviz.store.db import Database

# -- 1. subjective questions ----------------------------------------------------------------------
SUBJECTIVE = [
    "Which drug is the safest between semaglutide and canagliflozin for diabetes?",
    "Is pembrolizumab safe?",
    "Is this drug safe",
    "What is the best treatment for psoriasis?",
    "Should I take tirzepatide?",
    "Is metformin a bad drug?",
    "Which works better, semaglutide or liraglutide?",
    "Is tirzepatide better than semaglutide?",
    "Which is the most effective GLP-1 drug?",
    "Which drug is safest?",
    "Is semaglutide safe?",
    "Should I take metformin?",
    "What works best for migraine?",
    "Is pembrolizumab effective in trials?",  # efficacy verdict, same shape as "is X safe"
    "Which drug is best overall?",
    "How safe is semaglutide?",
    "How effective is tirzepatide?",
    "Which drug has the best overall response rate?",  # ranking even though an endpoint name follows
    "Is tirzepatide superior to semaglutide?",
    "What is the best available therapy for myelofibrosis?",
]
ANSWERABLE = [
    "What serious adverse events were reported in completed semaglutide trials?",
    "How many trials for pembrolizumab started each year since 2015?",
    "What are the safety endpoints in NCT03548935?",
    "Which sponsors run the most obesity trials?",
    "Compare phases for trials involving semaglutide vs tirzepatide",
    "What happens if my condition gets worse?",
    "Which countries have the most recruiting trials for type 2 diabetes?",
    "Can I keep taking my current medications?",
    "What are the best-studied drugs for obesity?",
    "What is the overall survival across pembrolizumab trials in NSCLC?",
    "What was the best overall response rate in NCT03548935?",
    "Best objective response in pembrolizumab melanoma trials",
    "Trials comparing a new drug with best supportive care in AML",
    "Best-corrected visual acuity endpoints in aflibercept trials",
    "Which trials reported better outcomes for pembrolizumab?",
    "What are the safety outcomes reported in semaglutide trials?",
    "What are the most common side effects in semaglutide trials?",
    "Progression-free survival hazard ratios across nivolumab trials",
    "Trials of drug X versus best supportive-care in AML",
    "Best  overall   response by arm in NCT03548935",
    "How is worst-case imputation used in semaglutide trials?",
    "Is semaglutide non-inferior to insulin glargine in these trials?",
    "noninferior to metformin trials",
]


@pytest.mark.parametrize("q", SUBJECTIVE)
def test_subjective_detected(q):
    assert is_subjective(q)


@pytest.mark.parametrize("q", ANSWERABLE)
def test_answerable_not_refused(q):
    assert not is_subjective(q)


# -- synthetic registry ----------------------------------------------------------------------------
PHASE_SETS = [
    [],
    ["EARLY_PHASE1"],
    ["PHASE1"],
    ["PHASE1", "PHASE2"],
    ["PHASE2"],
    ["PHASE2", "PHASE3"],
    ["PHASE3"],
    ["PHASE4"],
    ["NA"],
]
STATUSES = ["RECRUITING", "COMPLETED", "TERMINATED", "WITHDRAWN", "UNKNOWN", "AVAILABLE"]


def _study(i: int) -> dict[str, Any]:
    year = None if i % 23 == 0 else 1995 + (i * 7) % 35  # some missing, some before/after the window
    status: dict[str, Any] = {"overallStatus": STATUSES[i % len(STATUSES)]}
    if year is not None:
        status["startDateStruct"] = {"date": f"{year}-0{1 + i % 9}"}
        if year >= 2020 and i % 4 == 0:
            status["startDateStruct"]["type"] = "ESTIMATED"
    return {
        "protocolSection": {
            "identificationModule": {"nctId": f"NCT{90000000 + i:08d}", "briefTitle": f"Synthetic {i}"},
            "statusModule": status,
            "designModule": {"phases": PHASE_SETS[i % len(PHASE_SETS)]} if PHASE_SETS[i % len(PHASE_SETS)] else {},
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": f"Sponsor {i % 5}", "class": ["INDUSTRY", "OTHER", "NIH"][i % 3]}
            },
        }
    }


REGISTRY = [_study(i) for i in range(400)]


def _year(s: dict) -> int | None:
    d = s["protocolSection"]["statusModule"].get("startDateStruct", {}).get("date")
    return int(d[:4]) if d else None


def _phases(s: dict) -> set[str]:
    return set(s["protocolSection"].get("designModule", {}).get("phases", []))


_PHASE_PRED = {
    "EARLY_PHASE1": lambda p: "EARLY_PHASE1" in p,
    "PHASE1": lambda p: "PHASE1" in p and "PHASE2" not in p,
    "PHASE1/PHASE2": lambda p: {"PHASE1", "PHASE2"} <= p,
    "PHASE2": lambda p: "PHASE2" in p and not p & {"PHASE1", "PHASE3"},
    "PHASE2/PHASE3": lambda p: {"PHASE2", "PHASE3"} <= p,
    "PHASE3": lambda p: "PHASE3" in p and "PHASE2" not in p,
    "PHASE4": lambda p: "PHASE4" in p,
    "NA": lambda p: "NA" in p or not p,
}
_CLAUSE_PRED = {clause: (lambda s, k=k: _PHASE_PRED[k](_phases(s))) for k, clause in PHASE_BUCKETS}
_RANGE = re.compile(r"AREA\[StartDate\]RANGE\[(MIN|\d{4})(?:-\d\d-\d\d)?,(MAX|\d{4})(?:-\d\d-\d\d)?\]")


def _estimated(s: dict) -> bool:
    return s["protocolSection"]["statusModule"].get("startDateStruct", {}).get("type") == "ESTIMATED"


def matches(s: dict, clause: str) -> bool:
    """Mini Essie evaluator for the clauses the registry-count handlers issue."""
    if clause in _CLAUSE_PRED:
        return _CLAUSE_PRED[clause](s)
    if " AND " in clause:
        return all(matches(s, c) for c in clause.split(" AND "))
    if clause == "AREA[StartDateType]ESTIMATED":
        return _estimated(s)
    if clause == "AREA[StartDate]MISSING":
        return _year(s) is None
    if m := _RANGE.fullmatch(clause):
        y = _year(s)
        lo = -1 if m.group(1) == "MIN" else int(m.group(1))
        hi = 10**4 if m.group(2) == "MAX" else int(m.group(2))
        return y is not None and lo <= y <= hi
    if m := re.fullmatch(r"AREA\[OverallStatus\]\((.+)\)", clause):
        return s["protocolSection"]["statusModule"]["overallStatus"] in m.group(1).split(" OR ")
    if m := re.fullmatch(r"AREA\[OverallStatus\](\w+)", clause):
        return s["protocolSection"]["statusModule"]["overallStatus"] == m.group(1)
    if m := re.fullmatch(r"AREA\[LeadSponsorClass\](\w+)", clause):
        return s["protocolSection"]["sponsorCollaboratorsModule"]["leadSponsor"]["class"] == m.group(1)
    raise AssertionError(f"unexpected clause {clause}")


class RegistryClient:
    """search_page filters the synthetic registry by the spec's advanced clauses (ANDed)."""

    def __init__(self, db: Database, known: set[str] | None = None):
        self.db = db
        self.base = "https://clinicaltrials.gov/api/v2"
        self.calls: list = []
        self.known = known if known is not None else {"synthetica"}

    def _rows(self, spec) -> list[dict]:
        name = (spec.cond or spec.intr or "").strip('"').lower()
        if name not in self.known:
            return []
        rows = REGISTRY
        if spec.sort and spec.sort.startswith("StartDate:asc"):
            rows = sorted((r for r in rows if _year(r) is not None), key=_year)
        rows = [
            r
            for r in rows
            if not spec.statuses or r["protocolSection"]["statusModule"]["overallStatus"] in spec.statuses
        ]
        return [r for r in rows if all(matches(r, c) for c in spec.advanced)]

    async def search_page(
        self, spec, *, page_size=1000, page_token=None, count_total=True, refresh=False, fields_override=None
    ) -> SearchPage:
        self.calls.append(spec)
        rows = self._rows(spec)
        return SearchPage(
            studies=rows[:page_size], total_count=len(rows), next_page_token=None, url=f"{self.base}/studies?fake=1"
        )

    async def count(self, spec, *, refresh=False) -> int:
        return len(self._rows(spec))

    async def get_studies(self, nct_ids, *, refresh=False) -> list[dict]:
        return []

    async def version(self) -> dict:
        return {"apiVersion": "2.0.test"}


def _planner(plan: QueryPlan):
    async def planner(request, *, settings, llm, db):
        return plan, []

    return planner


@pytest.fixture
async def reg(settings, tmp_path):
    db = Database(tmp_path / "r.db").connect()
    client = RegistryClient(db)
    yield settings.model_copy(update={"max_studies": 50}), db, client
    await drain()
    db.close()


def _orch(reg, plan: QueryPlan) -> Orchestrator:
    settings, db, client = reg
    return Orchestrator(
        settings, db, client, StubLLM(), planner=_planner(plan), runs_dir=None, use_response_cache=False
    )


def _cohort(**kw) -> CohortFilter:
    return CohortFilter(label="Synthetica", conditions=["synthetica"], **kw)


# -- 1b. refusal end to end -----------------------------------------------------------------------
async def test_subjective_plan_returns_unanswerable_without_fetching(reg):
    plan = QueryPlan(
        intent=Intent.RISKS_ADVERSE_EVENTS,
        intent_confidence=0.9,
        answerability="subjective_judgment",
        cohorts=[CohortFilter(label="Semaglutide", drugs=["semaglutide"], conditions=["diabetes"])],
    )
    resp = await _orch(reg, plan).run(VisualizeRequest(query="Which drug is the safest for diabetes?"))
    assert resp.status == "unanswerable" and resp.visualization is None and not resp.answer_summary
    assert resp.clarification.reason == "subjective_question"
    assert any("semaglutide" in s for s in resp.clarification.suggestions)
    assert reg[2].calls == []  # nothing fetched, nothing ranked


# -- 2. unknown names: clarify, never widen -------------------------------------------------------
async def test_unverified_term_with_no_trials_asks_and_never_widens(reg):
    plan = QueryPlan(
        intent=Intent.TREND_OVER_TIME,
        intent_confidence=0.7,
        cohorts=[CohortFilter(label="All trials", start_year=2020)],
        unverified_terms=[UnverifiedTerm(kind="condition", text="Supercalifragilistic")],
    )
    resp = await _orch(reg, plan).run(VisualizeRequest(query="trials for the condition Supercalifragilistic"))
    assert resp.status == "needs_clarification" and resp.visualization is None
    assert resp.clarification.reason == "unrecognized_entity"
    assert resp.clarification.unmatched_terms == ["Supercalifragilistic"]
    assert all(spec.cond for spec in reg[2].calls)  # only the name probe: no all-trials search ever ran


async def test_passthrough_name_with_no_trials_asks(reg):
    plan = QueryPlan(
        intent=Intent.PHASE_DISTRIBUTION,
        intent_confidence=0.9,
        cohorts=[CohortFilter(label="Glorbitis", conditions=["glorbitis"])],
    )
    settings, db, client = reg
    o = Orchestrator(
        settings,
        db,
        client,
        StubLLM(),
        runs_dir=None,
        use_response_cache=False,
        planner=_entities_planner(plan, "glorbitis"),
    )
    resp = await o.run(VisualizeRequest(query="phases of glorbitis trials"))
    assert resp.status == "needs_clarification" and resp.clarification.unmatched_terms == ["glorbitis"]


def _entities_planner(plan: QueryPlan, name: str):
    from ctviz.api.schemas.response import ResolvedEntity

    async def planner(request, *, settings, llm, db):
        return plan, [
            ResolvedEntity(kind="condition", input=name, resolved=name, method="passthrough", search_terms=[name])
        ]

    return planner


async def test_unverified_term_with_trials_is_searched_as_written(reg):
    plan = QueryPlan(
        intent=Intent.PHASE_DISTRIBUTION,
        intent_confidence=0.7,
        cohorts=[CohortFilter(label="All trials")],
        unverified_terms=[UnverifiedTerm(kind="condition", text="synthetica")],
    )
    resp = await _orch(reg, plan).run(VisualizeRequest(query="phases for synthetica"))
    assert resp.status == "ok"
    assert resp.meta.interpretation.cohorts[0].conditions == ["synthetica"]
    assert resp.meta.interpretation.cohorts[0].label == "synthetica"
    assert any(n.code is NoteCode.TERM_SEARCHED_VERBATIM for n in resp.meta.notes)


# -- 3. registry counts reconcile -------------------------------------------------------------------
async def test_trend_registry_counts_match_brute_force_and_reconcile(reg):
    plan = QueryPlan(intent=Intent.TREND_OVER_TIME, intent_confidence=0.9, cohorts=[_cohort()], time_granularity="year")
    resp = await _orch(reg, plan).run(VisualizeRequest(query="synthetica trials per year"))
    assert resp.meta.coverage.counting_method == "registry_counts"
    assert resp.meta.coverage.studies_matched_total == len(REGISTRY)
    this_year = dt.date.today().year
    years = [d.model_extra["year"] for d in resp.visualization.data]
    early_est = sum(1 for s in REGISTRY if _estimated(s) and years[0] <= _year(s) < this_year - 1)
    for d in resp.visualization.data:
        y = d.model_extra["year"]
        assert d.model_extra["trial_count"] == sum(1 for s in REGISTRY if _year(s) == y)
        assert d.query_url and d.nct_ids is None
        planned = d.model_extra.get("estimated_count")
        if y >= this_year - 1:
            assert planned == sum(1 for s in REGISTRY if _year(s) == y and _estimated(s))
        else:
            assert planned == (None if early_est else 0)
    checks = {c.name: c for c in resp.meta.validation.checks}
    assert resp.meta.validation.passed, checks
    assert "= 400" in checks["retrieval_matches_api_total"].detail
    assert "Exact registry counts" in (resp.answer_summary or "")
    assert not any(n.code is NoteCode.SAMPLED for n in resp.meta.notes)


async def test_phase_registry_counts_equal_full_record_semantics(reg):
    plan = QueryPlan(intent=Intent.PHASE_DISTRIBUTION, intent_confidence=0.9, cohorts=[_cohort()])
    resp = await _orch(reg, plan).run(VisualizeRequest(query="synthetica trials by phase"))
    got = {d.model_extra["phase"]: d.model_extra["trial_count"] for d in resp.visualization.data}
    assert sum(got.values()) == len(REGISTRY)  # mutually exclusive buckets cover every trial
    assert got["Phase 1/2"] == sum(1 for s in REGISTRY if _phases(s) == {"PHASE1", "PHASE2"})
    assert got["Not applicable"] == sum(1 for s in REGISTRY if not _phases(s) or _phases(s) == {"NA"})
    assert resp.meta.validation.passed


async def test_status_registry_counts_include_remainder(reg):
    plan = QueryPlan(intent=Intent.STATUS_DISTRIBUTION, intent_confidence=0.9, cohorts=[_cohort()])
    resp = await _orch(reg, plan).run(VisualizeRequest(query="synthetica trials by status"))
    rows = {d.model_extra["status"]: d.model_extra["trial_count"] for d in resp.visualization.data}
    assert sum(rows.values()) == len(REGISTRY)
    assert rows["Other statuses (expanded access)"] == sum(
        1 for s in REGISTRY if s["protocolSection"]["statusModule"]["overallStatus"] == "AVAILABLE"
    )
    assert resp.meta.validation.passed


async def test_small_cohort_lists_every_trial_per_bar(reg):
    settings, db, client = reg
    o = Orchestrator(
        settings.model_copy(update={"max_studies": 1000}),
        db,
        client,
        StubLLM(),
        runs_dir=None,
        planner=_planner(QueryPlan(intent=Intent.PHASE_DISTRIBUTION, intent_confidence=0.9, cohorts=[_cohort()])),
        use_response_cache=False,
    )
    resp = await o.run(VisualizeRequest(query="synthetica trials by phase"))
    assert resp.meta.coverage.counting_method == "all_records"
    for d in resp.visualization.data:
        assert len(d.nct_ids) == d.model_extra["trial_count"] and len(d.citations) <= 3
    assert resp.meta.validation.passed


# -- 4. validation catches mismatches --------------------------------------------------------------
def test_check_counts_flags_count_list_mismatch(reg):
    from ctviz.api.schemas.response import Coverage, Interpretation, ResponseMeta, VisualizeResponse
    from ctviz.api.schemas.viz import Channel, Encoding, Visualization

    viz = Visualization(
        type=VizType.BAR_CHART,
        title="t",
        encoding=Encoding(
            x=Channel(field="phase", type="nominal", title="Phase"),
            y=Channel(field="trial_count", type="quantitative", title="Trials", unit="trials"),
        ),
        data=[Datum(id="d1", nct_ids=["NCT00000001", "NCT00000002"], phase="Phase 1", trial_count=3)],
    )
    resp = VisualizeResponse(
        request_id="x",
        created_at="now",
        visualization=viz,
        meta=ResponseMeta(
            interpretation=Interpretation(
                intent=Intent.PHASE_DISTRIBUTION,
                intent_confidence=1,
                persona="researcher",
                scope=Scope.COHORT,
                parser="rules",
            ),
            coverage=Coverage(studies_matched_total=3, studies_analyzed=3),
        ),
    )
    check = check_counts(resp)
    assert not check.passed and "count 3 but 2 NCT ids" in check.detail


# -- 5. public JSON shape ---------------------------------------------------------------------------
async def test_public_json_matches_assignment_shape(reg):
    plan = QueryPlan(intent=Intent.PHASE_DISTRIBUTION, intent_confidence=0.9, cohorts=[_cohort()])
    resp = await _orch(reg, plan).run(VisualizeRequest(query="synthetica trials by phase"))
    pub = to_public(resp)
    PublicResponse.model_validate(pub)
    assert set(pub) == {"status", "visualization", "answer_summary", "meta"}
    assert set(pub["visualization"]) == {"type", "title", "encoding", "data"}
    assert pub["meta"]["source"] == "clinicaltrials.gov" and pub["meta"]["filters"] == {"conditions": ["synthetica"]}
    assert "debug" not in pub["meta"] and "trace" not in pub["meta"]
    assert pub["meta"]["units"] == {"trial_count": "trials"}  # rendering hints live in meta

    def no_empty(o: Any) -> None:
        if isinstance(o, dict):
            assert all(v not in ([], {}, None) for v in o.values()), o
            for v in o.values():
                no_empty(v)
        elif isinstance(o, list):
            for v in o:
                no_empty(v)

    no_empty(pub)
    assert "debug" in to_public(resp, debug=True)["meta"]


# -- 6. citation detail levels and the one-bucket invariant -----------------------------------------
async def _small_phase(reg, detail: str):
    settings, db, client = reg
    o = Orchestrator(
        settings.model_copy(update={"max_studies": 1000}),
        db,
        client,
        StubLLM(),
        runs_dir=None,
        planner=_planner(QueryPlan(intent=Intent.PHASE_DISTRIBUTION, intent_confidence=0.9, cohorts=[_cohort()])),
        use_response_cache=False,
    )
    resp = await o.run(VisualizeRequest(query="synthetica trials by phase", citation_detail=detail))
    pub = to_public(resp, citation_detail=detail)
    PublicResponse.model_validate(pub)
    return resp, pub


async def test_citation_detail_sample_is_lean_hoisted_and_sorted(reg):
    _, pub = await _small_phase(reg, "sample")
    assert pub["meta"]["citation_policy"]["detail"] == "sample"
    for d in pub["visualization"]["data"]:
        assert d["nct_ids"] == sorted(d["nct_ids"]) and d["nct_ids_total"] == d["trial_count"]
        assert len(d["nct_ids"]) == min(d["trial_count"], MAX_NCT_IDS_PER_DATUM)
        assert "url" not in str(d["citations"])  # built from citation_policy.study_url
        for c in d["citations"]:  # {nct_id, excerpt}; a path only when it differs from the datum's shared one
            assert set(c) <= {"nct_id", "excerpt", "field_path"} and c.get("field_path") != d["field_path"]
        assert [c["nct_id"] for c in d["citations"]] == d["nct_ids"][: len(d["citations"])]
        assert d["citations_total"] == d["trial_count"]
        assert d["excerpts_truncated"] is (d["citations_total"] > len(d["citations"]))


async def test_citation_detail_full_inlines_every_excerpt_and_verifies_them(reg):
    resp, pub = await _small_phase(reg, "full")
    raw = {s["protocolSection"]["identificationModule"]["nctId"]: s for s in REGISTRY}
    from ctviz.core.provenance import verify_citation

    for d in resp.visualization.data:  # the substring/field-value check passes on every inline excerpt
        assert len(d.citations) == d.model_extra["trial_count"]
        assert all(verify_citation(c, raw[c.nct_id]) for c in d.citations)
    for d in pub["visualization"]["data"]:
        assert len(d["citations"]) == d["citations_total"] and d["excerpts_truncated"] is False
    assert resp.meta.validation.passed


async def test_citation_detail_ids_only_drops_excerpts(reg):
    _, pub = await _small_phase(reg, "ids_only")
    for d in pub["visualization"]["data"]:
        assert d["nct_ids"] and "citations" not in d and "field_path" not in d


async def test_each_trial_in_one_bucket_holds_and_catches_duplicates(reg):
    from ctviz.agents.validation import check_exclusive

    resp, _ = await _small_phase(reg, "sample")
    assert check_exclusive(resp).passed and "exactly one bucket" in check_exclusive(resp).detail
    rows = resp.visualization.data
    rows[1].nct_ids = sorted(rows[1].nct_ids + [rows[0].nct_ids[0]])  # one trial in two bars
    bad = check_exclusive(resp)
    assert not bad.passed and rows[0].nct_ids[0] in bad.detail


# -- 7. status filters read as "status today"; splitting by status group ---------------------------
def test_status_groups_and_phrase():
    from ctviz.ctgov.fields import STATUS_GROUPS, status_group, status_phrase

    assert status_group("TERMINATED") == status_group("WITHDRAWN") == "DONE"  # over, so done
    assert status_group("SUSPENDED") == "OTHER" and status_group(None) == "OTHER"
    assert status_phrase(list(STATUS_GROUPS["ACTIVE"])) == "status today: active"
    assert status_phrase(["RECRUITING"]) == "status today: recruiting"
    assert status_phrase([]) is None


async def test_status_filtered_trend_says_status_today(reg):
    settings, db, client = reg
    o = Orchestrator(
        settings.model_copy(update={"max_studies": 1000}),
        db,
        client,
        StubLLM(),
        runs_dir=None,
        planner=_planner(
            QueryPlan(intent=Intent.TREND_OVER_TIME, intent_confidence=0.9, cohorts=[_cohort(statuses=["RECRUITING"])])
        ),
        use_response_cache=False,
    )
    resp = await o.run(VisualizeRequest(query="recruiting synthetica trials per year"))
    assert resp.visualization.title.endswith("(status today: recruiting)")
    assert "status today: recruiting" in resp.answer_summary
    assert sum(d.model_extra["trial_count"] for d in resp.visualization.data) == sum(
        1 for r in REGISTRY if r["protocolSection"]["statusModule"]["overallStatus"] == "RECRUITING" and _year(r)
    )
    assert resp.meta.validation.passed


async def test_trend_split_by_status_group_stacks_every_trial_once(reg):
    settings, db, client = reg
    o = Orchestrator(
        settings.model_copy(update={"max_studies": 1000}),
        db,
        client,
        StubLLM(),
        runs_dir=None,
        planner=_planner(
            QueryPlan(
                intent=Intent.TREND_OVER_TIME, intent_confidence=0.9, cohorts=[_cohort()], series_by="status_group"
            )
        ),
        use_response_cache=False,
    )
    resp = await o.run(VisualizeRequest(query="active vs completed synthetica trials per year"))
    groups = {d.model_extra["status_group"] for d in resp.visualization.data}
    assert (
        groups <= {"Active", "Done (completed, terminated, withdrawn)", "Other (suspended, unknown, expanded access)"}
        and len(groups) == 3
    )
    assert sum(d.model_extra["trial_count"] for d in resp.visualization.data) == sum(1 for r in REGISTRY if _year(r))
    assert resp.meta.validation.passed


# -- 8. knowledge graph on the answer path ----------------------------------------------------------
class FakeGraph:
    """Alias table as graph ingest builds it: an alias -> the raw intervention names it was seen with."""

    async def aliases(self, alias, kind):
        return {
            "mk-3475": [
                "Pembrolizumab (MK-3475)",
                "pembrolizumab (KEYTRUDA)",
                "MK-3475 (pembrolizumab)",
                "Anti-PD-1 regimen",
            ]
        }.get(alias.casefold(), [])

    async def close_names(self, name, kind, n=2):
        return ["Glorbitol"] if name.casefold() == "glorbitolx" else []


async def test_other_name_known_to_the_graph_also_searches_the_entity(reg):
    from ctviz.api.schemas.plan import UnverifiedTerm

    settings, db, client = reg
    client.known |= {"mk-3475", "pembrolizumab"}
    plan = QueryPlan(
        intent=Intent.PHASE_DISTRIBUTION,
        intent_confidence=0.9,
        cohorts=[CohortFilter(label="MK")],
        unverified_terms=[UnverifiedTerm(kind="drug", text="MK-3475", cohort_index=0)],
    )
    o = Orchestrator(
        settings, db, client, StubLLM(), FakeGraph(), planner=_planner(plan), runs_dir=None, use_response_cache=False
    )
    resp = await o.run(VisualizeRequest(query="MK-3475 trials by phase"))
    assert resp.status == "ok"
    kg = [e for e in resp.meta.interpretation.resolved_entities if e.method == "knowledge_graph"]
    assert kg and kg[0].resolved == "pembrolizumab"  # 3 of 4 recorded names are pembrolizumab
    assert any("another name" in a for a in resp.meta.interpretation.assumptions)


async def test_spelling_suggestion_comes_from_names_seen_in_trials(reg):
    from ctviz.api.schemas.plan import UnverifiedTerm

    settings, db, client = reg
    plan = QueryPlan(
        intent=Intent.PHASE_DISTRIBUTION,
        intent_confidence=0.9,
        cohorts=[CohortFilter(label="X")],
        unverified_terms=[UnverifiedTerm(kind="drug", text="glorbitolx", cohort_index=0)],
    )
    o = Orchestrator(
        settings, db, client, StubLLM(), FakeGraph(), planner=_planner(plan), runs_dir=None, use_response_cache=False
    )
    resp = await o.run(VisualizeRequest(query="glorbitolx trials by phase"))
    assert resp.status == "needs_clarification"
    assert "Did you mean 'Glorbitol'?" in resp.clarification.suggestions


async def test_registry_trend_fits_the_call_budget_and_still_reconciles(reg):
    settings, db, client = reg
    plan = QueryPlan(intent=Intent.TREND_OVER_TIME, intent_confidence=0.9, cohorts=[_cohort()], time_granularity="year")
    o = Orchestrator(
        settings.model_copy(update={"max_ctgov_calls_per_request": 20}),
        db,
        client,
        StubLLM(),
        planner=_planner(plan),
        runs_dir=None,
        use_response_cache=False,
    )
    resp = await o.run(VisualizeRequest(query="synthetica trials per year"))
    assert resp.meta.coverage.counting_method == "registry_counts"
    assert len(resp.visualization.data) <= 13  # 20 calls - 7 outside-range/planned buckets
    assert resp.meta.validation.passed  # older years are counted in the "before" bucket, not lost


async def test_registry_status_groups_are_exact_and_cover_every_trial(reg):
    plan = QueryPlan(
        intent=Intent.STATUS_DISTRIBUTION, intent_confidence=0.9, cohorts=[_cohort()], group_by="status_group"
    )
    resp = await _orch(reg, plan).run(VisualizeRequest(query="active vs completed synthetica trials"))
    assert resp.meta.coverage.counting_method == "registry_counts"
    got = {d.model_extra["status_group"]: d.model_extra["trial_count"] for d in resp.visualization.data}
    assert sum(got.values()) == len(REGISTRY)
    done = sum(
        1
        for r in REGISTRY
        if r["protocolSection"]["statusModule"]["overallStatus"] in ("COMPLETED", "TERMINATED", "WITHDRAWN")
    )
    assert got["Done (completed, terminated, withdrawn)"] == done
    assert resp.meta.validation.passed


async def test_failed_planner_falls_back_to_the_rule_plan_not_a_minimal_one(reg):
    settings, db, client = reg

    async def broken(request, *, settings, llm, db):
        raise TimeoutError

    o = Orchestrator(settings, db, client, StubLLM(), planner=broken, runs_dir=None, use_response_cache=False)
    resp = await o.run(
        VisualizeRequest(query="How are synthetica trials distributed across phases?", condition="synthetica")
    )
    assert resp.meta.interpretation.intent is Intent.PHASE_DISTRIBUTION
    assert any("rule-based plan" in a for a in resp.meta.interpretation.assumptions)


async def test_result_records_are_balanced_across_compared_cohorts(reg):
    from ctviz.core.provenance import Study
    from ctviz.viz.base import CohortData

    settings, db, client = reg
    o = Orchestrator(
        settings.model_copy(update={"max_full_records": 4}),
        db,
        client,
        StubLLM(),
        runs_dir=None,
        use_response_cache=False,
    )

    def study(n, enroll):
        return Study(
            {
                "protocolSection": {
                    "identificationModule": {"nctId": f"NCT{n:08d}"},
                    "designModule": {"enrollmentInfo": {"count": enroll}},
                },
                "hasResults": True,
            }
        )

    a = CohortData(label="A", spec=None, studies=[study(i, i) for i in range(1, 10)], total_count=9)
    b = CohortData(label="B", spec=None, studies=[study(i, i) for i in range(100, 103)], total_count=3)
    fetched: list[list[str]] = []

    async def fake_fetch(client, ids, *, refresh=False):
        fetched.append(ids)
        return []

    import ctviz.core.orchestrator as orch_mod

    st = SimpleNamespace(
        ctx=SimpleNamespace(cohorts=[a, b], trials=[]), request=SimpleNamespace(refresh=False), degrade=lambda msg: None
    )
    orig, orch_mod.fetch_trials = orch_mod.fetch_trials, fake_fetch
    try:
        await o._fetch_result_records(st, lambda: 30.0)
    finally:
        orch_mod.fetch_trials = orig
    assert fetched[0] == ["NCT00000009", "NCT00000102", "NCT00000008", "NCT00000101"]  # interleaved, largest first
