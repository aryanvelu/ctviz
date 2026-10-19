"""Request schema for ``POST /v1/visualize``.

Only ``query`` is required. Structured fields, when present, override anything the planner
extracts from the query text.
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints, field_validator, model_validator

from ctviz.api.schemas.plan import GroupBy
from ctviz.api.schemas.viz import VizType

NCT_PATTERN = r"^NCT\d{8}$"
PHASES = ("EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA")
STATUSES = (
    "RECRUITING",
    "NOT_YET_RECRUITING",
    "ENROLLING_BY_INVITATION",
    "ACTIVE_NOT_RECRUITING",
    "SUSPENDED",
    "TERMINATED",
    "COMPLETED",
    "WITHDRAWN",
    "UNKNOWN",
)
_PHASE_ALIASES = {
    "0": "EARLY_PHASE1",
    "EARLY1": "EARLY_PHASE1",
    "EARLYPHASE1": "EARLY_PHASE1",
    "EARLY_PHASE_1": "EARLY_PHASE1",
    "1": "PHASE1",
    "I": "PHASE1",
    "2": "PHASE2",
    "II": "PHASE2",
    "3": "PHASE3",
    "III": "PHASE3",
    "4": "PHASE4",
    "IV": "PHASE4",
    "N/A": "NA",
    "NOT_APPLICABLE": "NA",
}


MAX_ITEM_CHARS = 200  # per string inside a request list (drug names, conditions, phases, statuses, ...)

# A bounded, whitespace-stripped string for list items (a 100k-character "drug name" is rejected with 422).
ListItem = Annotated[str, StringConstraints(strip_whitespace=True, max_length=MAX_ITEM_CHARS)]


def _norm_nct(v: Any) -> Any:
    """' nct03548935 ' -> 'NCT03548935' before pattern validation (non-strings pass through to fail typing)."""
    return v.strip().upper() if isinstance(v, str) else v


NctId = Annotated[str, BeforeValidator(_norm_nct), StringConstraints(pattern=NCT_PATTERN)]


def normalize_phase(raw: str) -> str | None:
    """'Phase 2' / '2' / 'II' / 'phase2' -> 'PHASE2'; None if unrecognized."""
    key = raw.strip().upper().replace(" ", "_").replace("-", "_")
    if key in PHASES:
        return key
    key = key.replace("PHASE_", "PHASE")
    if key in PHASES:
        return key
    return _PHASE_ALIASES.get(key.removeprefix("PHASE"))


class PatientProfile(BaseModel):
    """Optional self-description used for eligibility checks. Never stored."""

    model_config = ConfigDict(extra="forbid")  # a typo ("medication", "conditons") is a 422, not silently ignored

    age: int | None = Field(default=None, ge=0, le=120)
    sex: Literal["female", "male", "other"] | None = None
    conditions: list[ListItem] = Field(default_factory=list, max_length=20)
    medications: list[ListItem] = Field(default_factory=list, max_length=30)
    notes: str | None = Field(default=None, max_length=1000)


class VisualizeRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "query": "How has the number of trials for this drug changed over time?",
                    "drug_name": "Pembrolizumab",
                },
                {
                    "query": "Am I eligible?",
                    "nct_id": "NCT03548935",
                    "patient_profile": {"age": 45, "sex": "female", "conditions": ["obesity"]},
                },
            ]
        },
    )

    query: str = Field(min_length=1, max_length=2000, description="Natural-language question.")
    nct_id: NctId | None = Field(default=None, description="A specific trial (case-insensitive).")
    nct_ids: list[NctId] = Field(default_factory=list, max_length=5, description="Trials to compare.")
    drug_name: str | None = Field(default=None, max_length=200)
    drug_names: list[ListItem] = Field(default_factory=list, max_length=5, description="Drugs to compare.")
    condition: str | None = Field(default=None, max_length=200)
    conditions: list[ListItem] = Field(default_factory=list, max_length=5, description="Conditions to compare.")
    trial_phase: list[ListItem] = Field(
        default_factory=list, max_length=len(PHASES) * 2, description=f"Any of {PHASES}; '2', 'II' also accepted."
    )
    status: list[ListItem] = Field(
        default_factory=list, max_length=len(STATUSES) * 2, description=f"Any of {STATUSES}."
    )
    sponsor: str | None = Field(default=None, max_length=200)
    country: str | None = Field(default=None, max_length=100)
    start_year: int | None = Field(default=None, ge=1990)
    end_year: int | None = Field(default=None, ge=1990)
    persona: Literal["patient", "researcher", "auto"] = "auto"
    patient_profile: PatientProfile | None = None
    preferred_visualization: VizType | None = None
    group_by: GroupBy | None = None
    series_by: GroupBy | None = None
    time_granularity: Literal["year", "quarter", "month"] | None = None
    top_n: int | None = Field(default=None, ge=1, le=100)
    max_studies: int | None = Field(default=None, ge=1, le=25000)
    narrative: bool = Field(default=True, description="Include a short natural-language answer_summary.")
    include_trace: bool = Field(
        default=False,
        description="Debug: add meta.debug (pipeline steps, LLM calls, cost, cache counters, raw API parameters).",
    )
    refresh: bool = Field(default=False, description="Bypass caches and refetch from ClinicalTrials.gov.")
    citation_detail: Literal["ids_only", "sample", "full"] = Field(
        default="sample",
        description="Per datum: ids_only = nct_ids only (drawing); sample = nct_ids + the first "
        "3 excerpts by nct_id; full = nct_ids + every excerpt, all substring-verified.",
    )

    @field_validator("trial_phase")
    @classmethod
    def _norm_phases(cls, v: list[str]) -> list[str]:
        out = []
        for raw in v:
            key = normalize_phase(raw)
            if key is None:
                raise ValueError(f"unknown phase '{raw}'; use one of {PHASES}")
            out.append(key)
        return out

    @field_validator("status")
    @classmethod
    def _norm_status(cls, v: list[str]) -> list[str]:
        out = [s.strip().upper().replace(" ", "_") for s in v]
        bad = [s for s in out if s not in STATUSES]
        if bad:
            raise ValueError(f"unknown status {bad}; use any of {STATUSES}")
        return out

    @model_validator(mode="after")
    def _years(self) -> VisualizeRequest:
        max_year = dt.date.today().year + 5
        for name in ("start_year", "end_year"):
            val = getattr(self, name)
            if val is not None and val > max_year:
                raise ValueError(f"{name} must be <= {max_year}")
        if self.start_year and self.end_year and self.start_year > self.end_year:
            raise ValueError("start_year must be <= end_year")
        return self
