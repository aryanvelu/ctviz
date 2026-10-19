"""Analysis agent: picks WHICH outcome measures / groups answer a question (ids only).

One ``llm.structured`` call (agent="analysis") sees a number-free catalog of the trial's outcome
measures (index, title, type, paramType, unit, timeFrame, group titles) and returns indices / group
ids plus a one-line rationale. Every returned argument is validated against the catalog that was
sent (indices exist, group ids exist, purpose matches, at most ``MAX_MEASURES`` selections); any
violation rejects the whole answer and the deterministic regex selector is used instead. All numbers
are computed afterwards by ``ctviz.compute.*`` tools; the LLM never produces numbers or summaries.
"""

from __future__ import annotations

import asyncio
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ctviz.agents.llm import LLM, LLMError, LLMUnavailable
from ctviz.compute.adverse_events import event_groups
from ctviz.compute.flow import FLOW_PATH
from ctviz.compute.results import MeasureInfo, measure_catalog, short_endpoint
from ctviz.compute.survival import is_time_to_event_measure
from ctviz.config import Settings
from ctviz.core.provenance import Study
from ctviz.obs.trace import current_trace

Purpose = Literal["survival", "results", "ae", "flow"]
PURPOSES: frozenset[str] = frozenset({"survival", "results", "ae", "flow"})
MAX_MEASURES = 6
MAX_GROUPS = 20
MAX_CATALOG = 60
PROMPT_FILE = Path(__file__).parent / "prompts" / "analysis.md"


class AnalysisSelection(BaseModel):
    """Validated selection handed to the deterministic tools."""

    purpose: Purpose
    measure_indices: list[int] = Field(default_factory=list)
    group_ids: list[str] = Field(default_factory=list, description="Empty = all groups.")
    method: Literal["llm", "rules"] = "rules"
    rationale: str = Field(default="", description="Fixed text written by this module, never by the model.")
    rejected: list[str] = Field(
        default_factory=list,
        description="Fixed rejection codes (e.g. 'bad_indices:n=2'); never contains model output.",
    )


class LLMPick(BaseModel):
    """What the model must return: identifiers only."""

    purpose: str = Field(max_length=16)
    measure_indices: list[int] = Field(default_factory=list, max_length=50)
    group_ids: list[str] = Field(default_factory=list, max_length=50)
    rationale: str = Field(default="", max_length=300, description="Logged nowhere; kept for the model's benefit.")


SYSTEM_PROMPT = """You select which parts of ONE ClinicalTrials.gov results record answer the user's question.

Input JSON: `purpose` (survival | results | ae | flow), `question`, `measures` (outcome measures with
`index`, `title`, `type`, `paramType`, `unit`, `timeFrame`, `groups`, `group_ids`, `has_analyses`,
`classes`; no numbers are given) and `groups` (group ids and titles, for ae / flow).

Return ONLY identifiers:
- `purpose`: copy the purpose you were given.
- `measure_indices`: indices from `measures` (at most 6). For survival: arm-level overall survival /
  progression-free survival measures (medians and landmark rates such as "rate at 12 months"); avoid
  subgroup measures (e.g. "by PD-L1 expression") unless asked. For results: the measures that answer
  the question; for a generic question, the PRIMARY measures.
- `group_ids`: group ids to include, or an empty list for all groups.
- `rationale`: one short sentence explaining the choice, without numbers.

Never invent indices or ids. Never output numbers other than indices. Never summarize results."""


@lru_cache(maxsize=1)
def _system_prompt() -> str:
    """Prompt override file if present (ctviz/agents/prompts/analysis.md), else the built-in prompt."""
    return PROMPT_FILE.read_text() if PROMPT_FILE.exists() else SYSTEM_PROMPT


# -- catalog ---------------------------------------------------------------------------------------


def _groups_for(study: Study, purpose: str) -> list[tuple[str, str]]:
    if purpose == "ae":
        return [(g.id, g.title) for g in event_groups(study)]
    if purpose == "flow":
        return [
            (g.get("id", ""), g.get("title", ""))
            for _, g, _ in study.items(f"{FLOW_PATH}.groups")
            if isinstance(g, dict) and g.get("id")
        ]
    return []


def build_catalog(study: Study, purpose: str) -> dict:
    """Number-free catalog sent to the LLM (and used to validate its answer)."""
    measures = measure_catalog(study) if purpose in ("survival", "results") else []
    if purpose == "survival":
        measures = [m for m in measures if is_time_to_event_measure(m.title)]  # no all-measures fallback
    return {
        "purpose": purpose,
        "measures": [m.catalog_entry() for m in measures[:MAX_CATALOG]],
        "groups": [{"id": gid, "title": t} for gid, t in _groups_for(study, purpose)],
    }


# -- validation ------------------------------------------------------------------------------------


def validate_pick(pick: LLMPick, catalog: dict, purpose: str) -> list[str]:
    """Fixed rejection codes for an LLM answer vs the catalog it was given (empty list = valid).

    Codes never echo model output: ``bad_purpose``, ``bad_indices:n=<count>``, ``too_many_measures``,
    ``no_measures``, ``bad_group_ids:n=<count>``, ``too_many_groups``.
    """
    problems: list[str] = []
    if pick.purpose not in PURPOSES or pick.purpose != purpose:
        problems.append("bad_purpose")
    valid_idx = {m["index"] for m in catalog["measures"]}
    bad_idx = [i for i in pick.measure_indices if i not in valid_idx]
    if bad_idx:
        problems.append(f"bad_indices:n={len(bad_idx)}")
    if len(set(pick.measure_indices)) > MAX_MEASURES:
        problems.append("too_many_measures")
    if purpose in ("survival", "results") and not pick.measure_indices and valid_idx:
        problems.append("no_measures")
    valid_groups = {g["id"] for g in catalog["groups"]}
    valid_groups |= {gid for m in catalog["measures"] for gid in m["group_ids"]}
    bad_groups = [g for g in pick.group_ids if g not in valid_groups]
    if bad_groups:
        problems.append(f"bad_group_ids:n={len(bad_groups)}")
    if len(set(pick.group_ids)) > MAX_GROUPS:
        problems.append("too_many_groups")
    return problems


# -- rule-based selection --------------------------------------------------------------------------

_SUBGROUP = re.compile(
    r"based on|according to|by .{0,40}(status|expression|level|subgroup)|subgroup|"
    r"\bin participants with\b",
    re.I,
)
_TYPE_RANK = {"PRIMARY": 0, "SECONDARY": 1}
_STOP = {
    "what",
    "were",
    "was",
    "the",
    "results",
    "result",
    "trial",
    "study",
    "final",
    "show",
    "this",
    "that",
    "with",
    "from",
    "about",
    "outcome",
    "outcomes",
    "data",
    "does",
    "did",
    "have",
    "there",
    "which",
    "their",
}


def _rank(m: MeasureInfo) -> tuple[int, int]:
    return (_TYPE_RANK.get(m.type, 2), m.index)


def wanted_endpoints(question: str) -> list[str]:
    q = question.lower()
    out = []
    if re.search(r"progression|\bpfs\b", q):
        out.append("PFS")
    if re.search(r"overall survival|\bos\b|death|die|live longer|mortality", q):
        out.append("OS")
    return out or ["OS", "PFS"]


def rule_select(study: Study, question: str, purpose: str, catalog: dict | None = None) -> AnalysisSelection:
    """Deterministic fallback: regex over outcome titles (survival) / keyword overlap (results)."""
    catalog = catalog or build_catalog(study, purpose)
    if purpose == "survival":
        measures = [m for m in measure_catalog(study) if is_time_to_event_measure(m.title)]
        chosen: list[int] = []
        for ep in wanted_endpoints(question):
            cands = sorted(
                (m for m in measures if short_endpoint(m.title) == ep and not _SUBGROUP.search(m.title)), key=_rank
            )
            for ptype in ("MEDIAN", "NUMBER"):
                pick = next((m for m in cands if m.param_type.upper() == ptype), None)
                if pick is not None and pick.index not in chosen:
                    chosen.append(pick.index)
        if not chosen:  # any survival-like measure with analyses (e.g. HR only)
            chosen = [m.index for m in sorted(measures, key=_rank)[:2]]
        return AnalysisSelection(
            purpose="survival",
            measure_indices=chosen[:MAX_MEASURES],
            method="rules",
            rationale="Arm-level survival measures matched by title.",
        )
    if purpose == "results":
        measures = measure_catalog(study)
        words = {w for w in re.findall(r"[a-z0-9]{4,}", question.lower())} - _STOP
        scored = []
        for m in measures:
            tw = set(re.findall(r"[a-z0-9]{4,}", m.title.lower()))
            overlap = len(words & tw)
            if overlap:
                scored.append((-overlap, _rank(m), m.index))
        if scored:
            chosen = [i for *_, i in sorted(scored)[:3]]
            why = "Measures whose titles match the question."
        else:
            prim = sorted((m for m in measures if m.type == "PRIMARY"), key=_rank) or sorted(measures, key=_rank)
            chosen = [m.index for m in prim[:4]]
            why = "Primary outcome measures (generic question)."
        return AnalysisSelection(purpose="results", measure_indices=chosen, method="rules", rationale=why)
    return AnalysisSelection(
        purpose=purpose,
        group_ids=[],
        method="rules",  # type: ignore[arg-type]
        rationale="All reported groups.",
    )


# -- entry point -----------------------------------------------------------------------------------


async def select_measures(
    study: Study, question: str, purpose: Purpose, *, llm: LLM | None, settings: Settings
) -> AnalysisSelection:
    """Choose measures/groups for ``purpose`` with one validated LLM call; regex fallback otherwise."""
    if purpose not in PURPOSES:
        raise ValueError(f"unknown analysis purpose {purpose!r}")
    tr = current_trace()
    catalog = build_catalog(study, purpose)
    rejected: list[str] = []
    with tr.step(
        "analysis.select_measures",
        tool="select_measures",
        purpose=purpose,
        nct_id=study.nct_id,
        catalog_measures=len(catalog["measures"]),
        catalog_groups=len(catalog["groups"]),
    ) as info:
        if llm is not None and (catalog["measures"] or catalog["groups"]):
            timeout = settings.llm_timeout_s
            user = json.dumps({**catalog, "question": question[:500]}, ensure_ascii=False)
            try:
                pick = await asyncio.wait_for(
                    llm.structured(
                        agent="analysis",
                        model=settings.model_analysis,
                        system=_system_prompt(),
                        user=user,
                        schema=LLMPick,
                        max_output_tokens=300,
                        effort="none",
                        timeout_s=timeout,
                    ),
                    timeout=timeout + 1.0,
                )
                if not isinstance(pick, LLMPick):
                    pick = LLMPick.model_validate(pick if isinstance(pick, dict) else pick.model_dump())
                problems = validate_pick(pick, catalog, purpose)
                if not problems:
                    sel = AnalysisSelection(
                        purpose=purpose,
                        measure_indices=list(dict.fromkeys(pick.measure_indices)),
                        group_ids=list(dict.fromkeys(pick.group_ids)),
                        method="llm",
                        rationale="Validated LLM selection of catalog ids.",
                    )
                    info.update(method="llm", measure_indices=sel.measure_indices, group_ids=sel.group_ids)
                    return sel
                rejected = problems
            except TimeoutError:
                rejected = ["timeout"]
            except LLMUnavailable:
                rejected = ["llm_unavailable"]
            except (LLMError, ValidationError):
                rejected = ["llm_error"]
        sel = rule_select(study, question, purpose, catalog)
        sel.rejected = rejected
        info.update(method="rules", measure_indices=sel.measure_indices, rejected=rejected)
        if rejected:
            info["status"] = "degraded"
        return sel
