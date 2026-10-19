"""Citation model: the evidence attached to every visualized datum.

A citation points at one value inside one ClinicalTrials.gov study record:

* ``field_value``   — ``excerpt`` is the verbatim JSON leaf at ``field_path`` (strings as-is,
  numbers/booleans as JSON text, lists of scalars as compact JSON).
* ``text_excerpt``  — ``excerpt`` is a verbatim substring ``value[char_start:char_end]`` of the
  string at ``field_path`` (at most 300 characters).
* ``derived_count`` — the datum is a server-side count from a filtered search; ``excerpt`` is the
  ``totalCount`` returned, ``nct_id`` is empty and ``url`` reproduces the search.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

MAX_EXCERPT_CHARS = 300
# Inline citations per datum. An aggregate datum (several trials: a bar, a node) shows 3 example excerpts and
# lists every trial in datum.nct_ids; a single-trial datum (a criterion, an outcome) keeps up to 10 so long
# quoted text stays complete. The full list is always available from the citations endpoint.
MAX_CITATIONS_PER_DATUM = 3
MAX_CITATIONS_SINGLE_TRIAL = 10


class CitationKind(StrEnum):
    FIELD_VALUE = "field_value"
    TEXT_EXCERPT = "text_excerpt"
    DERIVED_COUNT = "derived_count"


class Citation(BaseModel):
    """One piece of evidence from a ClinicalTrials.gov record supporting a datum."""

    model_config = ConfigDict(frozen=True)

    nct_id: str = Field(description="Trial identifier, e.g. NCT03548935. Empty for derived_count.")
    excerpt: str = Field(description="Exact field value or verbatim text excerpt from the API response.")
    field_path: str = Field(
        description="Dot path into the API study JSON, with list indices, e.g. "
        "protocolSection.designModule.phases or ...interventions[0].name. '$.totalCount' for derived_count."
    )
    kind: CitationKind = CitationKind.FIELD_VALUE
    char_start: int | None = Field(default=None, description="Start offset of a text_excerpt within the field.")
    char_end: int | None = Field(default=None, description="End offset (exclusive) of a text_excerpt.")
    url: str = Field(description="Human-viewable source: the study page, or the search URL for derived_count.")
    label: str | None = Field(default=None, description="Optional human context, e.g. 'Nausea / Placebo arm'.")
