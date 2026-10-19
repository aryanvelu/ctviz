"""QueryPlan: the planner agent's structured interpretation of a request.

The planner (LLM or rule-based) must output exactly this shape. It is validated and repaired
deterministically before any retrieval happens.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ctviz.api.schemas.viz import VizType

_NCT = re.compile(r"^NCT\d{8}$")


class Intent(StrEnum):
    # Cohort / landscape
    TREND_OVER_TIME = "trend_over_time"
    PHASE_DISTRIBUTION = "phase_distribution"
    STATUS_DISTRIBUTION = "status_distribution"
    PHASE_STATUS_HEATMAP = "phase_status_heatmap"
    LANDSCAPE_GAP_HEATMAP = "landscape_gap_heatmap"
    SPONSOR_LANDSCAPE = "sponsor_landscape"
    SPONSOR_ACTIVITY = "sponsor_activity"
    GEOGRAPHIC_DISTRIBUTION = "geographic_distribution"
    ENTITY_NETWORK = "entity_network"
    ENROLLMENT_DISTRIBUTION = "enrollment_distribution"
    INTERVENTION_LANDSCAPE = "intervention_landscape"
    DEVELOPMENT_PROGRESSION = "development_progression"
    # Single trial
    SINGLE_TRIAL_OVERVIEW = "single_trial_overview"
    STUDY_RATIONALE = "study_rationale"
    ELIGIBILITY_CHECK = "eligibility_check"
    CONCOMITANT_MEDICATIONS = "concomitant_medications"
    TRIAL_JOURNEY_TIMELINE = "trial_journey_timeline"
    VISIT_BURDEN = "visit_burden"
    RISKS_ADVERSE_EVENTS = "risks_adverse_events"
    RESULTS_OUTCOMES = "results_outcomes"
    SURVIVAL_SUMMARY = "survival_summary"
    DROPOUT_FLOW = "dropout_flow"
    DEMOGRAPHICS = "demographics"
    ENROLLMENT_VS_POWER = "enrollment_vs_power"
    SITES_AND_CONTACTS = "sites_and_contacts"
    SAFETY_PROVISIONS = "safety_provisions"
    STUDY_LIMITATIONS = "study_limitations"
    DESIGN_METHODOLOGY = "design_methodology"
    ENDPOINTS_CATALOG = "endpoints_catalog"
    BIOMARKERS_PKPD = "biomarkers_pkpd"
    NOT_IN_REGISTRY = "not_in_registry_fallback"


class Persona(StrEnum):
    PATIENT = "patient"
    RESEARCHER = "researcher"


class Scope(StrEnum):
    SINGLE_TRIAL = "single_trial"
    COHORT = "cohort"
    COMPARISON = "comparison"


GroupBy = Literal[
    "phase",
    "status",
    "status_group",
    "sponsor",
    "sponsor_class",
    "condition",
    "intervention",
    "intervention_type",
    "country",
    "year",
    "allocation",
    "masking",
    "study_type",
]

NetworkKind = Literal[
    "sponsor_drug",
    "drug_drug",
    "drug_condition",
    "site_investigator",
    "sponsor_collaborator",
]


class CohortFilter(BaseModel):
    """One set of trials. Comparison scope has 2–5 of these; each becomes a chart series."""

    label: str = Field(description="Human label, used as the series name in comparisons.")
    drugs: list[str] = Field(default_factory=list, description="Intervention names (OR within the list).")
    conditions: list[str] = Field(default_factory=list)
    sponsors: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    phases: list[str] = Field(default_factory=list, description="EARLY_PHASE1|PHASE1|PHASE2|PHASE3|PHASE4|NA")
    statuses: list[str] = Field(default_factory=list, description="OverallStatus enum values")
    start_year: int | None = None
    end_year: int | None = None
    term: str | None = Field(default=None, description="Free-text fallback search term")
    interventional_only: bool = False

    @field_validator("phases")
    @classmethod
    def _phases(cls, v: list[str]) -> list[str]:
        """Normalize ('Phase 2', '2', 'II') and drop unknown values (planner output is untrusted)."""
        from ctviz.api.schemas.request import normalize_phase

        return list(dict.fromkeys(p for p in (normalize_phase(x) for x in v) if p))

    @field_validator("statuses")
    @classmethod
    def _statuses(cls, v: list[str]) -> list[str]:
        from ctviz.api.schemas.request import STATUSES

        out = [s.strip().upper().replace(" ", "_").replace(",", "") for s in v]
        return list(dict.fromkeys(s for s in out if s in STATUSES))


Answerability = Literal["answerable", "subjective_judgment"]


class UnverifiedTerm(BaseModel):
    """An entity named in the question that the vocabulary could not ground (checked against the API)."""

    kind: Literal["drug", "condition", "sponsor", "country"]
    text: str = Field(max_length=200)
    cohort_index: int = Field(default=0, ge=0, le=4)


class QueryPlan(BaseModel):
    intent: Intent
    intent_confidence: float = Field(ge=0.0, le=1.0)
    persona: Persona = Persona.RESEARCHER
    scope: Scope = Scope.COHORT
    nct_ids: list[str] = Field(default_factory=list, max_length=5)
    cohorts: list[CohortFilter] = Field(default_factory=list, max_length=5)
    group_by: GroupBy | None = None
    series_by: GroupBy | None = None
    network_kind: NetworkKind | None = None
    time_granularity: Literal["year", "quarter", "month"] | None = None
    top_n: int | None = Field(default=None, ge=1, le=100)
    preferred_visualization: VizType | None = None
    focus_terms: list[str] = Field(
        default_factory=list,
        description="Words the answer should focus on (e.g. a medication, biomarker, side effect, outcome).",
    )
    assumptions: list[str] = Field(default_factory=list)
    clarification: str | None = None
    answerability: Answerability = Field(
        default="answerable",
        description="subjective_judgment: the question asks for a verdict the registry cannot support "
        "(safest / best / most effective / should I take). Code then returns status=unanswerable.",
    )
    unverified_terms: list[UnverifiedTerm] = Field(
        default_factory=list,
        description="Named entities that must be confirmed by a ClinicalTrials.gov count "
        "before any data is fetched; zero matches -> needs_clarification.",
    )
    parser: Literal["llm", "rules", "cache"] = "rules"

    @field_validator("nct_ids")
    @classmethod
    def _valid_ncts(cls, v: list[str]) -> list[str]:
        """Keep only well-formed NCT ids (planner output is untrusted)."""
        out = [x.strip().upper() for x in v]
        return list(dict.fromkeys(x for x in out if _NCT.match(x)))
