"""Shared pytest fixtures. Tests never touch the network or OpenAI."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("CTVIZ_OFFLINE", "1")
# Tests must never write under the project's .cache/ or ctviz.db: every Settings() built during the test
# session (explicitly, via get_settings(), or in module-scoped fixtures) defaults its readable HTTP copy and
# database into a throwaway directory. Explicit kwargs still win; the autouse fixture below narrows these to
# each test's own tmp_path.
_SESSION_TMP = Path(tempfile.mkdtemp(prefix="ctviz-tests-"))
os.environ["CTVIZ_CACHE_DIR"] = str(_SESSION_TMP / "cache")
os.environ["CTVIZ_DB_PATH"] = str(_SESSION_TMP / "session.db")

from ctviz.api.schemas import Intent, QueryPlan, Scope, VisualizeRequest  # noqa: E402
from ctviz.api.schemas.plan import CohortFilter  # noqa: E402
from ctviz.config import Settings  # noqa: E402
from ctviz.core.provenance import Study, verify_citation  # noqa: E402
from ctviz.ctgov.requests import SearchSpec  # noqa: E402
from ctviz.viz.base import CohortData, HandlerContext  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "ctgov"


def load_study_raw(nct_id: str) -> dict:
    return json.loads((FIXTURES / "studies" / f"{nct_id}.json").read_text())


def load_cohort_raw(name: str) -> dict:
    with gzip.open(FIXTURES / "cohorts" / f"{name}.json.gz", "rt") as fh:
        return json.load(fh)


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point every Settings() created inside a test at that test's tmp_path (never the project's .cache/)."""
    monkeypatch.setenv("CTVIZ_CACHE_DIR", str(tmp_path / "ctgov_cache"))
    monkeypatch.setenv("CTVIZ_DB_PATH", str(tmp_path / "ctviz.db"))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(offline=True, db_path=tmp_path / "test.db", cache_dir=tmp_path / "ctgov_cache", _env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def study():
    """``study("NCT03548935")`` -> Study built from a recorded full record."""
    return lambda nct_id: Study(load_study_raw(nct_id))


@pytest.fixture
def cohort():
    """``cohort("semaglutide", label="Semaglutide")`` -> CohortData from a recorded search page."""

    def make(name: str, label: str | None = None) -> CohortData:
        raw = load_cohort_raw(name)
        studies = [Study(s) for s in raw["studies"]]
        return CohortData(
            label=label or name, spec=SearchSpec(), studies=studies, total_count=raw.get("totalCount") or len(studies)
        )

    return make


@pytest.fixture
def make_ctx(settings):
    """Build a HandlerContext for a handler unit test."""

    def make(
        intent: Intent,
        *,
        cohorts=None,
        trials=None,
        scope: Scope | None = None,
        query: str = "test",
        persona: str = "researcher",
        **plan_kwargs,
    ) -> HandlerContext:
        if scope is None:
            scope = (
                Scope.SINGLE_TRIAL if trials else (Scope.COMPARISON if cohorts and len(cohorts) > 1 else Scope.COHORT)
            )
        plan = QueryPlan(
            intent=intent,
            intent_confidence=1.0,
            scope=scope,
            persona=persona,
            nct_ids=[t.nct_id for t in trials or []],
            cohorts=[CohortFilter(label=c.label) for c in cohorts or []],
            **plan_kwargs,
        )
        return HandlerContext(
            request=VisualizeRequest(query=query),
            plan=plan,
            settings=settings,
            client=None,
            db=None,
            llm=None,
            cohorts=cohorts or [],
            trials=trials or [],
        )

    return make


def assert_citations_verified(viz, studies_by_id: dict[str, dict]) -> int:
    """The citation invariant: every excerpt is exactly supported by its raw record. Returns count checked."""
    from ctviz.api.schemas.viz import GraphData

    data = viz.data.nodes + viz.data.edges if isinstance(viz.data, GraphData) else viz.data
    checked = 0
    for datum in data:
        for c in datum.citations:
            if c.kind.value == "derived_count":
                assert c.excerpt.isdigit()
                continue
            assert c.nct_id in studies_by_id, f"{datum.id}: citation to unknown study {c.nct_id}"
            assert verify_citation(c, studies_by_id[c.nct_id]), f"{datum.id}: unverified {c.field_path}"
            checked += 1
    return checked
