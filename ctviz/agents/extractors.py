"""Extraction agent: eligibility / medication criteria evaluated against a patient profile.

Pipeline (at most one LLM call per trial):

1. **Rules** (always): structured fields (age, sex, healthy volunteers) are evaluated deterministically;
   free-text criteria are split with exact char spans and start as ``unknown`` / ``evaluated_by="none"``.
2. **Retrieve chunks**: each candidate free-text criterion becomes a chunk with a stable id
   ``f"{nct_id}:{field_path}:{char_start}-{char_end}"`` (plus its parent text and sub-heading for
   nested items). Only these chunks are sent to the model.
3. **LLM (optional)**: the model returns ``{chunk_id, criterion, applies, confidence, quote}`` per verdict.
4. **Validate every argument, per verdict**: the chunk id must be one provided to this call, the
   criterion number must match that chunk, ``applies``/``confidence`` must be allowed values, and the
   quote must be >= 8 characters, >= 2 words and a verbatim substring of that chunk (then re-verified
   against the raw record via ``Study.text_ref``). A bad verdict is rejected and recorded; the others
   still count.
5. **Clamp**: a verdict becomes a final met/unmet only when ``applies`` is yes/no, ``confidence`` is
   high, the section is known, the chunk is not a nested item, and the quote is *grounded* in the
   person's own data: it names a profile/focus term, or it is about age and the deterministic age rule
   gives the same answer. Everything else is ``needs_review``. A wrong "met" is the worst error, so
   absence-based verdicts ("the profile doesn't mention pregnancy, so not excluded") never become final.

The model never writes reasons, summaries or numbers: row reasons are templated by code.
If the LLM is unavailable (``StubLLM`` raises ``LLMUnavailable``), the rules-only rows are returned.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ctviz.agents.llm import LLM, LLMError
from ctviz.api.schemas.citations import MAX_EXCERPT_CHARS
from ctviz.api.schemas.request import PatientProfile
from ctviz.compute.eligibility import Criterion, mentions_any, split_criteria, structured_rules
from ctviz.compute.text_scan import CATEGORIES
from ctviz.config import Settings
from ctviz.core.provenance import FieldRef, Study, find_normalized
from ctviz.ctgov.fields import P
from ctviz.obs.trace import current_trace

log = logging.getLogger(__name__)
PROMPTS = Path(__file__).parent / "prompts"
AGENT = "extraction"
MAX_CHUNKS = 80
MAX_CHUNK_CHARS = 600
MIN_QUOTE_CHARS = 8
MIN_QUOTE_WORDS = 2
APPLIES = ("yes", "no", "unclear")
CONFIDENCE = ("high", "medium", "low")
_AGE_WORDS = re.compile(r"\bage[ds]?\b|\byears?\s+(?:of\s+age|old)\b", re.IGNORECASE)

Mode = Literal["eligibility", "medications"]
RowSection = Literal["structured", "inclusion", "exclusion", "unknown"]
RowStatus = Literal["met", "unmet", "unknown", "needs_review"]


# -- LLM I/O schema --------------------------------------------------------------------------------


class LLMVerdict(BaseModel):
    """One model verdict. Enum-like fields are plain strings and validated per verdict in code, so a
    single malformed verdict is rejected without discarding the whole response."""

    chunk_id: str = Field(description="Exactly one of the chunk_id values provided.")
    criterion: int | str = Field(description="The criterion number of that chunk.")
    applies: str = Field(description="yes | no | unclear")
    confidence: str = Field(description="high | medium | low")
    quote: str = Field(description="Exact substring of the chunk text the verdict depends on.")


class LLMExtraction(BaseModel):
    verdicts: list[LLMVerdict] = Field(default_factory=list)


# -- outputs ---------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrieved text span offered to the model (one free-text criterion)."""

    chunk_id: str
    number: int  # global criterion number shown to the model (1-based)
    criterion: Criterion
    field_path: str

    @property
    def text(self) -> str:
        return self.criterion.text


@dataclass
class CriterionVerdict:
    """One checklist row with its evidence. ``refs`` cover the whole criterion text."""

    section: RowSection
    index: int
    text: str
    status: RowStatus
    reason: str
    evaluated_by: Literal["rule", "llm", "none"]
    refs: list[FieldRef] = field(default_factory=list)
    relevance: Literal["high"] | None = None
    subheading: str | None = None
    chunk_id: str | None = None
    matched_terms: list[str] = field(default_factory=list)
    parent_text: str | None = None


@dataclass
class ExtractionOutcome:
    """Verdicts plus an audit of what the LLM step did."""

    verdicts: list[CriterionVerdict]
    chunks_provided: int = 0
    llm_called: bool = False
    llm_status: Literal["not_needed", "ok", "unavailable", "error", "invalid_output"] = "not_needed"
    accepted: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (chunk_id, reason)


def chunk_id_for(nct_id: str, field_path: str, start: int, end: int) -> str:
    """Stable chunk identifier."""
    return f"{nct_id}:{field_path}:{start}-{end}"


def span_refs(study: Study, path: str, start: int, end: int, label: str | None = None) -> list[FieldRef]:
    """Consecutive <=300-char text refs covering ``value[start:end]`` exactly (long criteria)."""
    refs, pos = [], start
    while pos < end:
        ref = study.text_ref(path, pos, min(end, pos + MAX_EXCERPT_CHARS), label=label)
        if ref is None:
            break
        refs.append(ref)
        pos = ref.char_end or end
    return refs


def _profile_terms(profile: PatientProfile | None, focus_terms: list[str]) -> list[str]:
    terms = list(focus_terms)
    if profile:
        terms += profile.conditions + profile.medications
    return [t for t in dict.fromkeys(t.strip() for t in terms) if t]


def _profile_has_content(profile: PatientProfile | None) -> bool:
    return bool(
        profile
        and (
            profile.age is not None
            or profile.sex
            or profile.conditions
            or profile.medications
            or (profile.notes and profile.notes.strip())
        )
    )


def is_medication_criterion(c: Criterion, terms: list[str]) -> bool:
    """True if a criterion is about medicines (keyword category) or names a profile/focus medicine."""
    return bool(CATEGORIES["medication"].search(c.text) or mentions_any(c.text, terms))


@dataclass
class _Context:
    """Deterministic facts the clamp step checks a verdict against."""

    terms: list[str]
    age_status: str | None  # status of the structured age rule, if evaluated


def rule_rows(
    study: Study, profile: PatientProfile | None, focus_terms: list[str], mode: Mode
) -> tuple[list[CriterionVerdict], list[Chunk], _Context]:
    """Deterministic rows (structured rules + unevaluated free-text rows) and the chunks for the LLM."""
    rows: list[CriterionVerdict] = []
    terms = _profile_terms(profile, focus_terms)
    rules = structured_rules(
        min_age=study.get("min_age"),
        max_age=study.get("max_age"),
        sex=study.get("sex"),
        healthy_volunteers=study.get("healthy_volunteers"),
        profile_age=profile.age if profile else None,
        profile_sex=profile.sex if profile else None,
        profile_conditions=profile.conditions if profile else None,
    )
    age_rule = next((r for r in rules if r.key == "age"), None)
    ctx = _Context(terms, age_rule.status if age_rule and age_rule.status in ("met", "unmet") else None)
    if mode == "eligibility":
        for i, r in enumerate(rules, start=1):
            refs = [study.ref(k, label=r.text) for k in r.fields]
            rows.append(
                CriterionVerdict(
                    "structured", i, r.text, r.status, r.reason, "rule", refs=[x for x in refs if x is not None]
                )
            )
    path = P["eligibility_criteria"]
    chunks: list[Chunk] = []
    for c in split_criteria(study.get("eligibility_criteria")):
        if mode == "medications" and not is_medication_criterion(c, terms):
            continue
        refs = span_refs(study, path, c.start, c.end, label=f"{c.section.capitalize()} criterion {c.index}")
        hits = mentions_any(c.text, terms)
        if hits:
            reason = f"Mentions {', '.join(hits)} from your question or profile; not evaluated automatically."
        else:
            reason = "Not evaluated: free-text criterion; confirm with the study team."
        cid = chunk_id_for(study.nct_id, path, c.start, c.end)
        rows.append(
            CriterionVerdict(
                c.section,
                c.index,
                c.text,
                "unknown",
                reason,
                "none",
                refs=refs,
                relevance="high" if hits else None,
                subheading=c.subheading,
                chunk_id=cid,
                matched_terms=hits,
                parent_text=c.parent_text,
            )
        )
        if len(chunks) < MAX_CHUNKS:
            chunks.append(Chunk(cid, len(chunks) + 1, c, path))
    return rows, chunks, ctx


def _locate_in_chunk(chunk: Chunk, quote: str) -> tuple[int, int] | None:
    """Absolute span of ``quote`` inside the chunk text (exact, then whitespace-normalized)."""
    q = quote.strip()
    if not q:
        return None
    idx = chunk.text.find(q)
    if idx >= 0:
        return chunk.criterion.start + idx, chunk.criterion.start + idx + len(q)
    span = find_normalized(chunk.text, q)
    return (chunk.criterion.start + span[0], chunk.criterion.start + span[1]) if span else None


def _proposed_status(section: str, applies: str) -> RowStatus | None:
    """met/unmet implied by the verdict, or None when it cannot be decided (unclear / unknown section)."""
    if applies not in ("yes", "no") or section not in ("inclusion", "exclusion"):
        return None
    blocks = (section == "inclusion" and applies == "no") or (section == "exclusion" and applies == "yes")
    return "unmet" if blocks else "met"


def clamp_status(chunk: Chunk, applies: str, confidence: str, quote: str, ctx: _Context) -> tuple[RowStatus, str]:
    """Final status for a validated verdict, plus why it was (not) allowed to be final."""
    proposed = _proposed_status(chunk.criterion.section, applies)
    if proposed is None:
        return "needs_review", "unclear_or_unknown_section"
    if confidence != "high":
        return "needs_review", "not_high_confidence"
    if chunk.criterion.parent_text is not None:
        return "needs_review", "nested_item"  # meaning depends on the parent ("none of the following:")
    if mentions_any(quote, ctx.terms):
        return proposed, "grounded_in_profile_term"
    if _AGE_WORDS.search(quote) and ctx.age_status == proposed:
        return proposed, "backed_by_age_rule"
    return "needs_review", "not_grounded_in_profile"


def apply_llm_verdicts(
    study: Study, rows: list[CriterionVerdict], chunks: list[Chunk], output: LLMExtraction, ctx: _Context | None = None
) -> tuple[int, list[tuple[str, str]]]:
    """Validate model verdicts against the provided chunks and merge accepted ones into ``rows``.

    Returns ``(accepted_count, rejected[(chunk_id, reason)])``.
    """
    ctx = ctx or _Context([], None)
    by_id = {c.chunk_id: c for c in chunks}
    row_by_chunk = {r.chunk_id: r for r in rows if r.chunk_id}
    accepted, rejected, seen = 0, [], set()
    for v in output.verdicts:
        cid = v.chunk_id if isinstance(v.chunk_id, str) else str(v.chunk_id)
        chunk = by_id.get(cid)
        if chunk is None:
            rejected.append((cid, "chunk_id_not_provided"))
            continue
        crit = v.criterion
        if isinstance(crit, str) and crit.strip().isdigit():
            crit = int(crit)
        if not isinstance(crit, int) or not (1 <= crit <= len(chunks)) or crit != chunk.number:
            rejected.append((cid, "criterion_index_mismatch"))
            continue
        applies = v.applies.strip().lower() if isinstance(v.applies, str) else None
        confidence = v.confidence.strip().lower() if isinstance(v.confidence, str) else None
        if applies not in APPLIES or confidence not in CONFIDENCE:
            rejected.append((cid, "invalid_enum"))
            continue
        if cid in seen:
            rejected.append((cid, "duplicate_verdict"))
            continue
        quote = v.quote.strip() if isinstance(v.quote, str) else ""
        if len(quote) < MIN_QUOTE_CHARS or len(quote.split()) < MIN_QUOTE_WORDS:
            rejected.append((cid, "quote_too_short"))
            continue
        span = _locate_in_chunk(chunk, quote)
        if span is None:
            rejected.append((cid, "quote_not_in_chunk"))
            continue
        quote_ref = study.text_ref(chunk.field_path, span[0], span[1], label="Quote used for the evaluation")
        if quote_ref is None or quote_ref.excerpt.strip() == "":
            rejected.append((cid, "quote_unverifiable"))
            continue
        seen.add(cid)
        row = row_by_chunk[cid]
        status, _why = clamp_status(chunk, applies, confidence, quote_ref.excerpt, ctx)
        quoted = quote_ref.excerpt if len(quote_ref.excerpt) <= 120 else quote_ref.excerpt[:117] + "..."
        if status == "needs_review":
            reason = f'Possibly relevant to your profile ("{quoted}"); confirm with the study team.'
        elif status == "met":
            reason = f'Based on your profile this criterion does not appear to exclude you ("{quoted}").'
        else:
            reason = f'Based on your profile this criterion appears to exclude you ("{quoted}").'
        row.status, row.reason, row.evaluated_by = status, reason, "llm"
        row.refs = [*row.refs, quote_ref]
        accepted += 1
    return accepted, rejected


def _load_prompt(mode: Mode) -> str:
    return (PROMPTS / f"extract_{'medications' if mode == 'medications' else 'criteria'}.md").read_text()


def _user_payload(profile: PatientProfile | None, focus_terms: list[str], chunks: list[Chunk]) -> str:
    prof = profile.model_dump(exclude_none=True) if profile else {}
    out = []
    for c in chunks:
        item = {
            "chunk_id": c.chunk_id,
            "criterion": c.number,
            "section": c.criterion.section,
            "text": c.text[:MAX_CHUNK_CHARS],
        }
        if c.criterion.subheading:
            item["subheading"] = c.criterion.subheading
        if c.criterion.parent_text:
            item["parent_text"] = c.criterion.parent_text[:MAX_CHUNK_CHARS]
        out.append(item)
    return json.dumps({"patient_profile": prof, "focus_terms": focus_terms, "chunks": out}, ensure_ascii=False)


async def extract_criteria_detailed(
    study: Study,
    *,
    llm: LLM | None,
    settings: Settings,
    profile: PatientProfile | None,
    focus_terms: list[str],
    mode: Mode = "eligibility",
) -> ExtractionOutcome:
    """Rules first, then (if there is a profile/focus and an LLM) one validated LLM pass."""
    tr = current_trace()
    with tr.step("extract.rules", nct_id=study.nct_id, mode=mode) as info:
        rows, chunks, ctx = rule_rows(study, profile, focus_terms, mode)
        info.update(rows=len(rows), chunks=len(chunks))
    outcome = ExtractionOutcome(rows, chunks_provided=len(chunks))
    if llm is None or not chunks or not (_profile_has_content(profile) or focus_terms):
        return outcome
    timeout = settings.llm_timeout_s
    with tr.step(
        "extract.llm",
        nct_id=study.nct_id,
        mode=mode,
        chunks=len(chunks),
        model=settings.model_extract,
        profile_fields=sorted(profile.model_dump(exclude_none=True, exclude_defaults=True)) if profile else [],
        focus_terms=len(focus_terms),
    ) as info:
        outcome.llm_called = True
        try:
            result = await asyncio.wait_for(
                llm.structured(
                    agent=AGENT,
                    model=settings.model_extract,
                    system=_load_prompt(mode),
                    user=_user_payload(profile, focus_terms, chunks),
                    schema=LLMExtraction,
                    max_output_tokens=1200,
                    timeout_s=timeout,
                ),
                timeout=timeout + 1.0,
            )
            if not isinstance(result, LLMExtraction):
                result = LLMExtraction.model_validate(result)
        except (LLMError, TimeoutError) as exc:  # LLMUnavailable is an LLMError
            outcome.llm_status = "unavailable" if type(exc).__name__ == "LLMUnavailable" else "error"
            info.update(status="degraded", result=outcome.llm_status)
            log.info("extraction fell back to rules for %s: %s", study.nct_id, type(exc).__name__)
            return outcome
        except (ValidationError, ValueError, TypeError):
            outcome.llm_status = "invalid_output"
            info.update(status="degraded", result="invalid_output")
            return outcome
        outcome.accepted, outcome.rejected = apply_llm_verdicts(study, rows, chunks, result, ctx)
        outcome.llm_status = "ok"
        info.update(
            result="ok",
            returned=len(result.verdicts),
            accepted=outcome.accepted,
            rejected=len(outcome.rejected),
            rejected_reasons=sorted({r for _, r in outcome.rejected}),
            final=sum(r.evaluated_by == "llm" and r.status in ("met", "unmet") for r in rows),
        )
    return outcome


async def extract_criteria(
    study: Study,
    *,
    llm: LLM | None,
    settings: Settings,
    profile: PatientProfile | None,
    focus_terms: list[str],
    mode: Mode = "eligibility",
) -> list[CriterionVerdict]:
    """Interface entry point (docs/interfaces.md): checklist verdicts; LLM optional, rules fallback."""
    return (
        await extract_criteria_detailed(
            study, llm=llm, settings=settings, profile=profile, focus_terms=focus_terms, mode=mode
        )
    ).verdicts
