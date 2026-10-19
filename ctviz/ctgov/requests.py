"""Translate a CohortFilter into ClinicalTrials.gov ``/studies`` query parameters (Essie syntax)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from urllib.parse import urlencode

from ctviz.api.schemas.plan import CohortFilter
from ctviz.ctgov.fields import FIELD_SETS, FieldSet

SEARCH_UI = "https://clinicaltrials.gov/search"


_ESSIE_SPECIAL = re.compile(r"[\s()\[\]:^~*\\/-]|\b(AND|OR|NOT)\b")


def _or_terms(terms: list[str]) -> str:
    """``a OR "b c"``: quote any term with spaces, operators or Essie syntax so it is a literal phrase."""
    cleaned = [" ".join(t.replace('"', " ").split()) for t in terms if t and t.strip()]
    return " OR ".join(f'"{t}"' if _ESSIE_SPECIAL.search(t) else t for t in cleaned if t)


@dataclass(frozen=True)
class SearchSpec:
    """Immutable description of one /studies search (before pagination)."""

    intr: str | None = None
    cond: str | None = None
    spons: str | None = None
    locn: str | None = None
    term: str | None = None
    ids: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    advanced: tuple[str, ...] = ()  # ANDed Essie clauses
    field_set: FieldSet = FieldSet.LANDSCAPE
    sort: str | None = None
    extra: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @classmethod
    def from_cohort(cls, c: CohortFilter, field_set: FieldSet = FieldSet.LANDSCAPE) -> SearchSpec:
        adv: list[str] = []
        if c.phases:
            adv.append(f"AREA[Phase]({' OR '.join(c.phases)})" if len(c.phases) > 1 else f"AREA[Phase]{c.phases[0]}")
        if c.start_year or c.end_year:
            lo = f"{c.start_year}-01-01" if c.start_year else "MIN"
            hi = f"{c.end_year}-12-31" if c.end_year else "MAX"
            adv.append(f"AREA[StartDate]RANGE[{lo},{hi}]")
        if c.interventional_only:
            adv.append("AREA[StudyType]INTERVENTIONAL")
        return cls(
            intr=_or_terms(c.drugs) or None,
            cond=_or_terms(c.conditions) or None,
            spons=_or_terms(c.sponsors) or None,
            locn=_or_terms(c.countries) or None,
            term=c.term or None,
            statuses=tuple(c.statuses),
            advanced=tuple(adv),
            field_set=field_set,
        )

    def with_clause(self, clause: str) -> SearchSpec:
        return replace(self, advanced=(*self.advanced, clause))

    def without(self, **kwargs: object) -> SearchSpec:
        return replace(self, **kwargs)

    def params(
        self,
        page_size: int = 1000,
        count_total: bool = True,
        page_token: str | None = None,
        fields_override: list[str] | None = None,
    ) -> dict[str, str]:
        p: dict[str, str] = {}
        for key, val in (
            ("query.intr", self.intr),
            ("query.cond", self.cond),
            ("query.spons", self.spons),
            ("query.locn", self.locn),
            ("query.term", self.term),
        ):
            if val:
                p[key] = val
        if self.ids:
            p["filter.ids"] = ",".join(self.ids)
        if self.statuses:
            p["filter.overallStatus"] = ",".join(self.statuses)
        if self.advanced:
            p["filter.advanced"] = " AND ".join(self.advanced)
        fields = fields_override if fields_override is not None else FIELD_SETS[self.field_set]
        if fields:
            p["fields"] = ",".join(fields)
        if self.sort:
            p["sort"] = self.sort
        p["pageSize"] = str(page_size)
        if count_total:
            p["countTotal"] = "true"
        if page_token:
            p["pageToken"] = page_token
        p.update(dict(self.extra))
        return p

    def ui_url(self) -> str:
        """A ClinicalTrials.gov website URL a human can open to see the same cohort."""
        q: dict[str, str] = {}
        for key, val in (
            ("intr", self.intr),
            ("cond", self.cond),
            ("spons", self.spons),
            ("locStr", self.locn),
            ("term", self.term),
        ):
            if val:
                q[key] = val
        if self.advanced:
            q["advanced"] = " AND ".join(self.advanced)
        if self.statuses:
            q["aggFilters"] = "status:" + " ".join(s.lower()[:3] for s in self.statuses)
        return f"{SEARCH_UI}?{urlencode(q)}"

    def describe(self) -> dict[str, str]:
        """Compact, human-readable filter summary for ``meta.filters_applied``."""
        return {
            k: v
            for k, v in self.params(page_size=1, count_total=False, fields_override=[]).items()
            if k not in ("pageSize", "fields")
        }
