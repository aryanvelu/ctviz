"""Single-trial text intent handlers (overview, rationale, eligibility, medications, journey, visits, sites,
safety provisions, limitations, endpoints, biomarkers, not-in-registry).

Every handler follows the ``phase_distribution`` pattern: read fields through :class:`Study`, emit cited
datums with :class:`Dataset`, build the viz with a builder, and return notes + compact facts. Text values
are verbatim excerpts with char spans; absent fields produce notes, never invented content.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from ctviz.agents.extractors import CriterionVerdict, ExtractionOutcome, extract_criteria_detailed, span_refs
from ctviz.api.schemas.citations import MAX_EXCERPT_CHARS
from ctviz.api.schemas.plan import Intent, Scope
from ctviz.api.schemas.response import Note, NoteCode
from ctviz.api.schemas.viz import Approximation, Channel, Datum, TimeAxis, TimelineLane, Visualization, VizMeta
from ctviz.compute.eligibility import split_criteria
from ctviz.compute.endpoints import CATEGORY_LABELS, CATEGORY_ORDER, classify_endpoint
from ctviz.compute.text_scan import CATEGORIES, Hit, scan_study, split_sentences
from ctviz.compute.timeframes import (
    DAYS_PER_UNIT,
    day_to_week_label,
    mentions_cycles,
    parse_durations,
    parse_timeframe,
)
from ctviz.core.provenance import Dataset, FieldRef, Study
from ctviz.ctgov.fields import PHASE_LABELS, P
from ctviz.viz.base import HandlerContext, HandlerResult, note, register
from ctviz.viz.builders.bar_chart import bar_chart, ordered
from ctviz.viz.builders.eligibility_checklist import eligibility_checklist
from ctviz.viz.builders.geo_map import site_map
from ctviz.viz.builders.kpi_cards import kpi_cards
from ctviz.viz.builders.table import col, table
from ctviz.viz.builders.timeline import lane_timeline

NOT_STATED = "Not stated in the ClinicalTrials.gov record"
OTHER_OUTCOMES = "protocolSection.outcomesModule.otherOutcomes"
PRE_ASSIGNMENT = "resultsSection.participantFlowModule.preAssignmentDetails"
RECRUITMENT = "resultsSection.participantFlowModule.recruitmentDetails"
POINT_OF_CONTACT = "resultsSection.moreInfoModule.pointOfContact"
HAS_DMC = "protocolSection.oversightModule.oversightHasDmc"
FIRST_POST_TYPE = "protocolSection.statusModule.studyFirstPostDateStruct.type"
RESULTS_POST_TYPE = "protocolSection.statusModule.resultsFirstPostDateStruct.type"
MAX_TABLE_ROWS = 60
OUTCOME_KEYS = (("primary_outcomes", "Primary"), ("secondary_outcomes", "Secondary"), (OTHER_OUTCOMES, "Other"))


# -- shared helpers --------------------------------------------------------------------------------


def _trial(ctx: HandlerContext) -> Study | None:
    return ctx.trials[0] if ctx.trials else None


def _trial_required(title: str) -> HandlerResult:
    """A renderable card when the question needs one specific trial but none was resolved."""
    ds = Dataset()
    ds.emit(
        {
            "label": "Trial needed",
            "value": "Ask about a specific trial (for example by its NCT number).",
            "kind": "text",
        },
        [],
    )
    viz = kpi_cards(ds.rows, title=title)
    return HandlerResult(
        visualization=viz,
        full_refs=ds.full_refs,
        studies_analyzed=0,
        notes=[note(NoteCode.TRIAL_REQUIRED, "This question is about one trial; no trial was identified.", "warning")],
    )


def _not_stated_card(ds: Dataset, s: Study, label: str, detail: str | None = None) -> Datum:
    """Explicit 'not in the record' card; cites the record id to show which record was checked."""
    row = {"label": label, "value": NOT_STATED, "kind": "text"}
    if detail:
        row["detail"] = detail
    return ds.emit(row, [s.ref("nct_id", label="Record checked")])


def _kpi_field(
    ds: Dataset,
    s: Study,
    label: str,
    key: str,
    kind: str,
    *,
    value=None,
    detail_key: str | None = None,
    missing: list[str] | None = None,
) -> Datum | None:
    """Emit one kpi card from a field (value defaults to the field itself); record absence in ``missing``."""
    whole = _whole_refs(s, key, label=label)
    if not whole:
        if missing is not None:
            missing.append(label)
        return None
    row = {"label": label, "value": s.get(key) if value is None else value, "kind": kind}
    refs: list[FieldRef | None] = list(whole)
    if detail_key:
        dref = s.ref(detail_key)
        if dref is not None:
            row["detail"] = s.get(detail_key)
            refs.append(dref)
    return ds.emit(row, refs)


def _whole_refs(s: Study, path: str, label: str | None = None) -> list[FieldRef]:
    """Refs that together cover the whole field value (several <=300-char spans for long strings)."""
    value = s.get(path)
    if isinstance(value, str) and len(value) > MAX_EXCERPT_CHARS:
        return span_refs(s, path, 0, len(value), label=label)
    ref = s.ref(path, label=label)
    return [ref] if ref else []


def _text_pieces(s: Study, path: str) -> list[tuple[str, FieldRef]]:
    """A text field as verbatim pieces: the whole value if <= 300 chars, else its sentences."""
    value = s.get(path)
    if not isinstance(value, str) or not value.strip():
        return []
    if len(value) <= 300:
        ref = s.text_ref(path, 0, len(value))
        return [(value, ref)] if ref else []
    out = []
    for a, b in split_sentences(value):
        ref = s.text_ref(path, a, b)
        if ref:
            out.append((ref.excerpt, ref))
    return out


def _field_label(path: str) -> str:
    """Human label for a field path used as a 'source' column."""
    tail = path.split(".")
    labels = {
        "briefSummary": "Brief summary",
        "detailedDescription": "Detailed description",
        "eligibilityCriteria": "Eligibility criteria",
        "facility": "Site listing",
        "preAssignmentDetails": "Participant flow (pre-assignment)",
        "recruitmentDetails": "Participant flow (recruitment)",
        "whyStopped": "Why stopped",
    }
    if tail[-1] == "description":
        if "armGroups" in path:
            return "Arm description"
        if "interventions" in path:
            return "Intervention description"
        return "Outcome description"
    return labels.get(tail[-1], tail[-1])


def _contact_rows(ds: Dataset, s: Study) -> int:
    """Emit kpi rows for central contacts, overall officials and the results point of contact."""
    n = 0
    for _, c, path in s.items("central_contacts"):
        if not isinstance(c, dict):
            continue
        parts, refs = [], []
        for k in ("name", "phone", "email"):
            ref = s.ref(f"{path}.{k}")
            if ref is not None:
                parts.append(str(c[k]))
                refs.append(ref)
        if parts:
            ds.emit({"label": "Central contact", "value": " · ".join(parts), "kind": "text"}, refs)
            n += 1
    if n == 0:
        for _, o, path in s.items("overall_officials"):
            if not isinstance(o, dict):
                continue
            parts, refs = [], []
            for k in ("name", "affiliation"):
                ref = s.ref(f"{path}.{k}")
                if ref is not None:
                    parts.append(str(o[k]))
                    refs.append(ref)
            role_ref = s.ref(f"{path}.role")
            if parts:
                row = {"label": "Study official", "value": ", ".join(parts), "kind": "text"}
                if role_ref:
                    row["detail"] = o.get("role")
                    refs.append(role_ref)
                ds.emit(row, refs)
                n += 1
    poc = s.get(POINT_OF_CONTACT)
    if isinstance(poc, dict):
        parts, refs = [], []
        for k in ("title", "organization", "phone", "email"):
            ref = s.ref(f"{POINT_OF_CONTACT}.{k}")
            if ref is not None:
                parts.append(str(poc[k]))
                refs.append(ref)
        if parts:
            ds.emit({"label": "Results point of contact", "value": " · ".join(parts), "kind": "text"}, refs)
            n += 1
    return n


def _outcomes(s: Study) -> Iterable[tuple[str, str, dict]]:
    """(item_path, outcome_type, outcome dict) for primary, secondary and other outcomes."""
    for key, otype in OUTCOME_KEYS:
        for _, item, path in s.items(key):
            if isinstance(item, dict):
                yield path, otype, item


def _result(
    viz: Visualization, ds: Dataset, *, notes: list[Note] | None = None, facts: dict | None = None, studies: int = 1
) -> HandlerResult:
    return HandlerResult(
        visualization=viz, notes=notes or [], facts=facts or {}, full_refs=ds.full_refs, studies_analyzed=studies
    )


def _title(s: Study) -> str:
    acr = s.get("acronym")
    return f"{s.nct_id}{f' ({acr})' if acr else ''}"


# -- overview --------------------------------------------------------------------------------------


@register(Intent.SINGLE_TRIAL_OVERVIEW)
async def single_trial_overview(ctx: HandlerContext) -> HandlerResult:
    s = _trial(ctx)
    if s is None:
        return _trial_required("Trial overview")
    ds, missing = Dataset(), []
    _kpi_field(ds, s, "Title", "brief_title", "text", missing=missing)
    _kpi_field(ds, s, "Status", "status", "enum", missing=missing)
    phases = s.get("phases") or []
    _kpi_field(
        ds, s, "Phase", "phases", "enum", value=" / ".join(PHASE_LABELS.get(p, p) for p in phases), missing=missing
    )
    _kpi_field(ds, s, "Study type", "study_type", "enum", missing=missing)
    _kpi_field(ds, s, "Conditions", "conditions", "list", missing=missing)
    names, refs = [], []
    for _, it, path in s.items("interventions"):
        ref = s.ref(f"{path}.name")
        if ref is not None:
            names.append(it["name"])
            refs.append(ref)
    if names:
        ds.emit({"label": "Interventions", "value": names, "kind": "list"}, refs)
    else:
        missing.append("Interventions")
    _kpi_field(ds, s, "Lead sponsor", "lead_sponsor", "text", missing=missing)
    _kpi_field(ds, s, "Enrollment", "enrollment_count", "number", detail_key="enrollment_type", missing=missing)
    _kpi_field(ds, s, "Start date", "start_date", "date", detail_key="start_date_type", missing=missing)
    _kpi_field(
        ds,
        s,
        "Primary completion",
        "primary_completion_date",
        "date",
        detail_key="primary_completion_type",
        missing=missing,
    )
    _kpi_field(ds, s, "Study completion", "completion_date", "date", detail_key="completion_type", missing=missing)
    locs = list(s.items("locations"))
    if locs:
        ds.emit({"label": "Sites listed", "value": len(locs), "kind": "number"}, [s.ref("locations")])
        countries: dict[str, FieldRef] = {}
        for _, loc, path in locs:
            if isinstance(loc, dict) and loc.get("country") and loc["country"] not in countries:
                ref = s.ref(f"{path}.country")
                if ref:
                    countries[loc["country"]] = ref
        if countries:
            ds.emit(
                {"label": "Countries", "value": sorted(countries), "kind": "list"},
                [countries[c] for c in sorted(countries)],
            )
    else:
        missing.append("Sites")
    ds.emit(
        {"label": "Results posted", "value": "yes" if s.has_results else "no", "kind": "enum"}, [s.ref("has_results")]
    )
    notes = []
    if missing:
        notes.append(note(NoteCode.NOT_IN_REGISTRY, f"This record does not report: {', '.join(missing)}."))
    if not s.has_results:
        notes.append(note(NoteCode.NO_RESULTS_POSTED, "No results have been posted for this trial."))
    viz = kpi_cards(ds.rows, title=f"Overview of {_title(s)}", subtitle=s.get("brief_title"))
    facts = {"status": s.get("status"), "phases": phases, "enrollment": s.get("enrollment_count"), "sites": len(locs)}
    return _result(viz, ds, notes=notes, facts=facts)


# -- rationale -------------------------------------------------------------------------------------


@register(Intent.STUDY_RATIONALE)
async def study_rationale(ctx: HandlerContext) -> HandlerResult:
    s = _trial(ctx)
    if s is None:
        return _trial_required("Study goal and rationale")
    ds, notes = Dataset(), []
    summary = s.get("brief_summary")
    if isinstance(summary, str) and summary.strip():
        spans = split_sentences(summary)
        a, b = spans[0]
        ref = s.text_ref("brief_summary", a, b, label="First sentence of the brief summary")
        if ref:
            ds.emit(
                {
                    "label": "Main goal",
                    "value": ref.excerpt,
                    "kind": "text",
                    "detail": "First sentence of the registry's brief summary",
                },
                [ref],
            )
    else:
        _not_stated_card(ds, s, "Main goal", "The record has no brief summary.")
    _kpi_field(ds, s, "Primary purpose", "primary_purpose", "enum")
    measures, mrefs = [], []
    for _, o, path in s.items("primary_outcomes"):
        ref = s.ref(f"{path}.measure")
        if ref:
            measures.append(o["measure"])
            mrefs.append(ref)
    if measures:
        ds.emit({"label": "Primary outcome(s)", "value": measures, "kind": "list"}, mrefs)
    first_sentence = (
        (P["brief_summary"], *split_sentences(summary)[0]) if isinstance(summary, str) and summary.strip() else None
    )
    hits = [h for h in scan_study(s, ["rationale"], arms=False) if (h.field_path, h.start, h.end) != first_sentence]
    for h in hits[:6]:
        ref = s.text_ref(h.field_path, h.start, h.end, label=f"Matched '{h.term}'")
        if ref:
            ds.emit(
                {
                    "label": "Why it may work (from the record)",
                    "value": ref.excerpt,
                    "kind": "text",
                    "detail": _field_label(h.field_path),
                },
                [ref],
            )
    if not hits:
        _not_stated_card(
            ds,
            s,
            "Why researchers think it may work",
            "The summary and description do not state a hypothesis or mechanism.",
        )
        notes.append(
            note(
                NoteCode.NOT_IN_REGISTRY,
                "The record does not explain why the treatment is expected to work; the protocol or "
                "published papers may.",
            )
        )
    viz = kpi_cards(ds.rows, title=f"Goal and rationale of {_title(s)}")
    return _result(viz, ds, notes=notes, facts={"rationale_excerpts": len(hits)})


# -- eligibility / medications ---------------------------------------------------------------------


def _profile_used(ctx: HandlerContext) -> dict:
    p = ctx.request.patient_profile
    if p is None:
        return {}
    used = p.model_dump(exclude_none=True, exclude_defaults=True, exclude={"notes"})
    if p.notes:
        used["notes_provided"] = True
    return used


def _emit_verdicts(ds: Dataset, verdicts: list[CriterionVerdict]) -> None:
    for v in verdicts:
        row = {
            "section": v.section,
            "index": v.index,
            "text": v.text,
            "status": v.status,
            "reason": v.reason,
            "evaluated_by": v.evaluated_by,
        }
        if v.relevance:
            row["relevance"] = v.relevance
        if v.subheading:
            row["subheading"] = v.subheading
        if v.parent_text:
            row["parent_text"] = v.parent_text
        ds.emit(row, v.refs)


def _extraction_notes(outcome: ExtractionOutcome, profile_given: bool) -> list[Note]:
    notes = []
    n_llm = sum(v.evaluated_by == "llm" for v in outcome.verdicts)
    if n_llm:
        notes.append(
            note(
                NoteCode.LLM_EVALUATED,
                f"{n_llm} free-text criteria were checked by a language model against your profile, "
                "each backed by a verified verbatim quote; anything not clear-cut is marked needs_review. "
                "Only the study team can confirm eligibility.",
            )
        )
    if outcome.rejected:
        notes.append(
            note(
                NoteCode.LLM_EVALUATED,
                f"{len(outcome.rejected)} model verdicts were discarded because they did not cite a provided "
                "criterion with a verbatim quote of at least two words (see meta.extraction).",
                "warning",
            )
        )
    if any(v.section == "unknown" for v in outcome.verdicts):
        notes.append(
            note(
                NoteCode.NOT_IN_REGISTRY,
                "The record's criteria have no clear Inclusion/Exclusion headings, so free-text criteria "
                "are never marked met or unmet automatically.",
                "warning",
            )
        )
    if outcome.llm_status in ("unavailable", "error", "invalid_output") and profile_given:
        notes.append(
            note(
                NoteCode.DEGRADED,
                "Automatic evaluation of free-text criteria was unavailable; they are listed unevaluated.",
            )
        )
    if outcome.llm_called and outcome.llm_status == "ok" and n_llm == 0:
        notes.append(
            note(
                NoteCode.NOT_IN_REGISTRY,
                "Nothing in the record's criteria could be matched to your profile with a verifiable quote; "
                "the criteria are listed unevaluated.",
            )
        )
    return notes


def _extraction_meta(outcome: ExtractionOutcome) -> dict:
    return {
        "chunks_provided": outcome.chunks_provided,
        "llm_called": outcome.llm_called,
        "llm_status": outcome.llm_status,
        "accepted": outcome.accepted,
        "rejected": [{"chunk_id": c, "reason": r} for c, r in outcome.rejected],
    }


@register(Intent.ELIGIBILITY_CHECK)
async def eligibility_check(ctx: HandlerContext) -> HandlerResult:
    s = _trial(ctx)
    if s is None:
        return _trial_required("Eligibility checklist")
    profile = ctx.request.patient_profile
    outcome = await extract_criteria_detailed(
        s, llm=ctx.llm, settings=ctx.settings, profile=profile, focus_terms=ctx.plan.focus_terms, mode="eligibility"
    )
    ds = Dataset(prefix="c")
    if not outcome.verdicts:
        _not_stated_card(ds, s, "Eligibility criteria")
        return _result(
            kpi_cards(ds.rows, title=f"Eligibility for {_title(s)}"),
            ds,
            notes=[note(NoteCode.NOT_IN_REGISTRY, "The record lists no eligibility criteria.")],
        )
    _emit_verdicts(ds, outcome.verdicts)
    meta = VizMeta(diagnostics={"extraction": _extraction_meta(outcome)})
    viz = eligibility_checklist(
        ds.rows, title=f"Eligibility checklist for {_title(s)}", profile_used=_profile_used(ctx), meta=meta
    )
    notes = _extraction_notes(outcome, profile is not None)
    if profile is None:
        notes.append(
            note(
                NoteCode.NOT_IN_REGISTRY,
                "No patient profile was given, so only the trial's structured age/sex rules can be "
                "checked; add age, sex, conditions and medications to evaluate more.",
            )
        )
    if any(not v.refs for v in outcome.verdicts):
        notes.append(note(NoteCode.TRUNCATED, "Some criteria could not be cited verbatim.", "warning"))
    return _result(viz, ds, notes=notes, facts=viz.meta.summary or {})


@register(Intent.CONCOMITANT_MEDICATIONS)
async def concomitant_medications(ctx: HandlerContext) -> HandlerResult:
    s = _trial(ctx)
    if s is None:
        return _trial_required("Medication rules")
    profile = ctx.request.patient_profile
    outcome = await extract_criteria_detailed(
        s, llm=ctx.llm, settings=ctx.settings, profile=profile, focus_terms=ctx.plan.focus_terms, mode="medications"
    )
    protocol_note = note(
        NoteCode.NOT_IN_REGISTRY,
        "Complete lists of permitted and prohibited medicines are in the study protocol, not the "
        "registry. Never stop a medicine without talking to your doctor and the study team.",
    )
    ds = Dataset(prefix="c")
    if not outcome.verdicts:
        _not_stated_card(ds, s, "Medication rules", "No eligibility criterion mentions medicines or therapies.")
        return _result(kpi_cards(ds.rows, title=f"Medication rules in {_title(s)}"), ds, notes=[protocol_note])
    _emit_verdicts(ds, outcome.verdicts)
    meta = VizMeta(diagnostics={"extraction": _extraction_meta(outcome)})
    viz = eligibility_checklist(
        ds.rows, title=f"Medication-related criteria in {_title(s)}", profile_used=_profile_used(ctx), meta=meta
    )
    notes = [protocol_note, *_extraction_notes(outcome, profile is not None)]
    meds = (profile.medications if profile else []) + list(ctx.plan.focus_terms)
    if meds:
        named = {t.lower() for v in outcome.verdicts for t in v.matched_terms}
        unnamed = [m for m in meds if m.lower() not in named]
        if unnamed:
            notes.append(
                note(
                    NoteCode.NOT_IN_REGISTRY,
                    f"Not named in the eligibility criteria: {', '.join(unnamed)}. The registry cannot "
                    "say whether they are allowed; ask the study team.",
                )
            )
    return _result(viz, ds, notes=notes, facts=viz.meta.summary or {})


# -- journey timeline ------------------------------------------------------------------------------

_MILESTONES = [
    ("Study start", "start_date", "start_date_type"),
    ("First posted", "first_post_date", FIRST_POST_TYPE),
    ("Primary completion", "primary_completion_date", "primary_completion_type"),
    ("Study completion", "completion_date", "completion_type"),
    ("Results first posted", "results_first_post_date", RESULTS_POST_TYPE),
]
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class _Timepoints:
    """Outcome time frames grouped by whole day, plus what could not be placed on the axis."""

    pts: dict[int, list] = field(default_factory=lambda: defaultdict(list))
    refs: dict[int, list[FieldRef]] = field(default_factory=lambda: defaultdict(list))
    cycle_refs: list[FieldRef] = field(default_factory=list)  # cycle-based time frames left unconverted
    negative: list[FieldRef] = field(default_factory=list)  # pre-baseline (screening) days, not shown


def _timepoints(s: Study) -> _Timepoints:
    """Parse every outcome time frame into whole days (>= 0); collect cycle-based and negative ones."""
    tp = _Timepoints()
    for path, _otype, o in _outcomes(s):
        tf = o.get("timeFrame")
        if mentions_cycles(tf):
            ref = s.ref(f"{path}.timeFrame", label="Cycle-based time frame (not converted)")
            if ref:
                tp.cycle_refs.append(ref)
            continue
        for p in parse_timeframe(tf):
            ref = s.text_ref(f"{path}.timeFrame", p.start, p.end, label=p.label)
            if p.day < 0:
                if ref:
                    tp.negative.append(ref)
                continue
            day = round(p.day)
            tp.pts[day].append(p)
            if ref and all(r.key() != ref.key() for r in tp.refs[day]):
                tp.refs[day].append(ref)
    return tp


def _timepoint_notes(tp: _Timepoints) -> list[Note]:
    notes = []
    if tp.cycle_refs:
        notes.append(
            note(
                NoteCode.APPROXIMATED,
                f"{len(tp.cycle_refs)} outcome time frames are counted in treatment cycles and were not "
                "converted to days (cycle-relative timing is ambiguous).",
            )
        )
    if tp.negative:
        notes.append(
            note(
                NoteCode.APPROXIMATED,
                f"{len(tp.negative)} pre-baseline (screening) time points with negative days are not shown.",
            )
        )
    return notes


# Two lanes with different axes: registry milestones on calendar dates (``date``), the participant journey
# in days from baseline (``day``/``day_end``). No channel mixes the two; each lane names its own fields.
JOURNEY_LANES = [
    TimelineLane(name="registry_milestones", axis=TimeAxis(kind="calendar"), start_field="date"),
    TimelineLane(
        name="participant_journey",
        axis=TimeAxis(kind="relative", unit="days", origin="baseline / start of participation"),
        start_field="day",
        end_field="day_end",
        approximated=True,
    ),
]


def _timeline_viz(rows: list[Datum], title: str, meta: VizMeta) -> Visualization:
    return lane_timeline(
        rows,
        title=title,
        lanes=JOURNEY_LANES,
        label=Channel(field="event", type="nominal", title="Event"),
        tooltip=["date_type", "detail"],
        meta=meta,
    )


@register(Intent.TRIAL_JOURNEY_TIMELINE)
async def trial_journey_timeline(ctx: HandlerContext) -> HandlerResult:
    s = _trial(ctx)
    if s is None:
        return _trial_required("Trial timeline")
    ds, notes = Dataset(), []
    partial = False
    for label, key, type_key in _MILESTONES:
        ref = s.ref(key, label=label)
        if ref is None:
            continue
        date = s.get(key)
        row = {"lane": "registry_milestones", "event": label, "date": date}
        refs = [ref]
        tref = s.ref(type_key)
        if tref is not None:
            row["date_type"] = s.get(type_key)
            refs.append(tref)
        partial = partial or not _DATE_RE.match(str(date))
        ds.emit(row, refs)
    registry_ids = [r.id for r in ds.rows]
    # Participant lane: outcome assessment time points + treatment duration phrases.
    tp = _timepoints(s)
    participant_ids = []
    for day in sorted(tp.pts):
        non_window = [p for p in tp.pts[day] if p.kind != "up_to"]
        if day == 0:
            event = "Baseline / start of participation"
        elif non_window:
            event = f"Assessment ({non_window[0].label})"
        else:
            event = f"End of follow-up window ({tp.pts[day][0].label})"
        d = ds.emit(
            {
                "lane": "participant_journey",
                "event": event,
                "day": day,
                "day_end": day,
                "detail": day_to_week_label(day),
            },
            tp.refs[day],
        )
        participant_ids.append(d.id)
    seen_durations: set[int] = set()
    for path, text in _duration_sources(s):
        for dur in parse_durations(text):
            days = round(dur.days)
            if days in seen_durations or days <= 0:
                continue
            seen_durations.add(days)
            ref = s.text_ref(path, dur.start, dur.end, label="Treatment duration phrase")
            if ref is None:
                continue
            d = ds.emit(
                {
                    "lane": "participant_journey",
                    "event": f"Treatment period ({ref.excerpt})",
                    "day": 0,
                    "day_end": days,
                    "detail": f"{_field_label(path)}; {day_to_week_label(days)}",
                },
                [ref],
            )
            participant_ids.append(d.id)
    if not ds.rows:
        _not_stated_card(ds, s, "Trial timeline", "No dates or time frames are recorded.")
        return _result(
            kpi_cards(ds.rows, title=f"Timeline of {_title(s)}"),
            ds,
            notes=[
                note(NoteCode.NOT_IN_REGISTRY, "The record has no milestone dates or outcome time frames."),
                *_timepoint_notes(tp),
            ],
        )
    meta = VizMeta(
        units={"day": "days", "day_end": "days"},
        diagnostics={"partial_dates": "registry_milestones dates may be YYYY-MM (no day)"},
    )
    meta.approximation = Approximation(
        is_approximation=True,
        method="Outcome time frames and treatment duration phrases converted to days from "
        f"baseline (week=7 d, month={DAYS_PER_UNIT['month']} d, year={DAYS_PER_UNIT['year']} d).",
        assumptions=[
            "Applies to the participant_journey lane only; registry_milestones are registry dates.",
            "Assessment time points come from outcome measures, not a visit schedule.",
            "Offsets from the last dose, cycle-based time frames and pre-baseline days are not placed on the axis.",
        ],
    )
    if participant_ids:
        notes.append(
            note(
                NoteCode.APPROXIMATED,
                "The participant journey is approximated from outcome time "
                "frames and treatment durations; the real visit schedule is in "
                "the protocol.",
                affects=participant_ids,
            )
        )
    else:
        notes.append(
            note(
                NoteCode.NOT_IN_REGISTRY,
                "No outcome time frames could be converted to days, so the participant journey is not shown.",
            )
        )
    notes += _timepoint_notes(tp)
    if partial:
        notes.append(
            note(NoteCode.PARTIAL_DATES, "Some registry dates are month-only (YYYY-MM).", affects=registry_ids)
        )
    estimated = [r.id for r in ds.rows if (r.model_extra or {}).get("date_type") == "ESTIMATED"]
    if estimated:
        notes.append(
            note(NoteCode.APPROXIMATED, "Dates typed ESTIMATED are the sponsor's projections.", affects=estimated)
        )
    viz = _timeline_viz(ds.rows, f"Timeline of {_title(s)}", meta)
    facts = {
        "last_assessment_day": max(tp.pts) if tp.pts else None,
        "longest_treatment_days": max(seen_durations) if seen_durations else None,
    }
    return _result(viz, ds, notes=notes, facts=facts)


def _duration_sources(s: Study) -> list[tuple[str, str]]:
    out = []
    for key in ("arm_groups", "interventions"):
        for _, item, path in s.items(key):
            if isinstance(item, dict) and isinstance(item.get("description"), str):
                out.append((f"{path}.description", item["description"]))
    for path in (P["detailed_description"], PRE_ASSIGNMENT):
        v = s.get(path)
        if isinstance(v, str):
            out.append((path, v))
    return out


# -- visit burden ----------------------------------------------------------------------------------


@register(Intent.VISIT_BURDEN)
async def visit_burden(ctx: HandlerContext) -> HandlerResult:
    """Uniform table rows ``{event, kind, detail}``; ``value`` (count) and ``day`` are tooltip cells."""
    s = _trial(ctx)
    if s is None:
        return _trial_required("Visits and hospital stays")
    ds, notes = Dataset(), []
    tp = _timepoints(s)
    visit_days = sorted(d for d, ps in tp.pts.items() if any(p.kind != "up_to" for p in ps))
    all_refs = [r for d in visit_days for r in tp.refs[d]]
    if visit_days:
        ds.emit(
            {
                "event": "Distinct assessment time points (lower bound on visits)",
                "kind": "lower_bound",
                "value": len(visit_days),
                "detail": f"{len(visit_days)} distinct days",
            },
            all_refs,
        )
        for d in visit_days:
            label = next(p.label for p in tp.pts[d] if p.kind != "up_to")
            ds.emit(
                {
                    "event": f"Assessment: {label}",
                    "kind": "assessment_timepoint",
                    "day": d,
                    "detail": day_to_week_label(d),
                },
                tp.refs[d],
            )
    hits = scan_study(s, ["visits"], eligibility=True)
    for h in hits[:MAX_TABLE_ROWS]:
        ref = s.text_ref(h.field_path, h.start, h.end, label=f"Matched '{h.term}'")
        if ref:
            ds.emit({"event": ref.excerpt, "kind": "visit_mention", "detail": _field_label(h.field_path)}, [ref])
    if not ds.rows:
        _not_stated_card(ds, s, "Visits and hospital stays")
        return _result(
            kpi_cards(ds.rows, title=f"Visits in {_title(s)}"),
            ds,
            notes=[
                note(NoteCode.NOT_IN_REGISTRY, "The record states no assessment time points or visit details."),
                *_timepoint_notes(tp),
            ],
        )
    notes.append(
        note(
            NoteCode.APPROXIMATED,
            "The number of visits is a lower bound: distinct assessment time "
            "points from outcome measures (baseline included, follow-up window "
            "ends excluded). Screening and safety visits are usually extra.",
        )
    )
    notes.append(
        note(
            NoteCode.NOT_IN_REGISTRY,
            "The registry does not publish the visit schedule or whether "
            "overnight stays are required; ask the study site.",
        )
    )
    notes += _timepoint_notes(tp)
    if not hits:
        notes.append(
            note(NoteCode.NOT_IN_REGISTRY, "No sentence in the record mentions visits, clinics or hospital stays.")
        )
    meta = VizMeta(units={"day": "days from baseline", "value": "time points"})
    meta.approximation = Approximation(
        is_approximation=True,
        method="distinct whole days parsed from outcome time frames",
        assumptions=["One visit per distinct assessment day."],
    )
    viz = table(
        ds.rows,
        title=f"Visits and assessments in {_title(s)}",
        meta=meta,
        columns=[col("event", "Item"), col("kind", "Kind"), col("detail", "Detail")],
        tooltip=["value", "day"],
    )
    return _result(viz, ds, notes=notes, facts={"assessment_timepoints": len(visit_days), "visit_mentions": len(hits)})


# -- sites and contacts ----------------------------------------------------------------------------


@register(Intent.SITES_AND_CONTACTS)
async def sites_and_contacts(ctx: HandlerContext) -> HandlerResult:
    s = _trial(ctx)
    if s is None:
        return _trial_required("Study sites and contacts")
    doctor_note = note(
        NoteCode.NOT_IN_REGISTRY,
        "The registry does not say whether you can keep seeing your regular doctor during the study. "
        "Most trials let you, but ask the site contact how study care and your usual care fit together.",
    )
    ds = Dataset(prefix="s")
    n_geo = 0
    for i, loc, path in s.items("locations"):
        if not isinstance(loc, dict):
            continue
        row: dict = {"row_type": "site"}
        refs: list[FieldRef | None] = []
        for k in ("facility", "city", "state", "country", "status"):
            ref = s.ref(f"{path}.{k}")
            if ref is not None:
                row[k] = loc[k]
                refs.append(ref)
        geo = loc.get("geoPoint")
        if isinstance(geo, dict) and all(isinstance(geo.get(k), int | float) for k in ("lat", "lon")):
            row["lat"], row["lon"] = geo["lat"], geo["lon"]
            refs.append(s.ref(f"{path}.geoPoint"))
            n_geo += 1
        contacts = loc.get("contacts") or []
        if contacts and isinstance(contacts[0], dict):
            for k in ("name", "phone", "email"):
                ref = s.ref(f"{path}.contacts[0].{k}")
                if ref is not None:
                    row[f"contact_{k}"] = contacts[0][k]
                    refs.append(ref)
        row.setdefault("facility", ", ".join(str(loc[k]) for k in ("city", "country") if loc.get(k)) or f"Site {i + 1}")
        ds.emit(row, refs)
    n_sites = len(ds.rows)
    if n_sites == 0:
        kds = Dataset()
        _not_stated_card(kds, s, "Study sites", "No site locations are listed.")
        n_contacts = _contact_rows(kds, s)
        if n_contacts == 0:
            _not_stated_card(kds, s, "Contacts", "No contacts or officials are listed.")
        return _result(
            kpi_cards(kds.rows, title=f"Sites and contacts for {_title(s)}"),
            kds,
            notes=[note(NoteCode.NOT_IN_REGISTRY, "This record lists no study sites."), doctor_note],
        )
    for _, c, path in s.items("central_contacts"):
        if not isinstance(c, dict):
            continue
        row, refs = {"row_type": "central_contact", "facility": "Central contact"}, []
        for k in ("name", "phone", "email"):
            ref = s.ref(f"{path}.{k}")
            if ref is not None:
                row[f"contact_{k}"] = c[k]
                refs.append(ref)
        if refs:
            ds.emit(row, refs)
    for h in scan_study(s, ["doctor"], arms=False, eligibility=True, locations=True)[:3]:
        ref = s.text_ref(h.field_path, h.start, h.end, label=f"Matched '{h.term}'")
        if ref:
            ds.emit({"row_type": "record_mention", "facility": "Mention in the record", "text": ref.excerpt}, [ref])
    meta = VizMeta(
        stats={"mappable_rows": n_geo},
        diagnostics={
            "row_type_field": "row_type",
            "row_types": "site | central_contact | record_mention (only sites have lat/lon)",
        },
    )
    viz = site_map(
        ds.rows,
        title=f"Study sites for {_title(s)}",
        label=Channel(field="facility", type="nominal", title="Site"),
        meta=meta,
        tooltip=["city", "state", "country", "status", "contact_name", "contact_phone", "contact_email", "text"],
    )
    notes = [doctor_note]
    if n_geo < n_sites:
        notes.append(
            note(
                NoteCode.NOT_IN_REGISTRY,
                f"{n_sites - n_geo} of {n_sites} sites have no coordinates and cannot be placed on the map.",
            )
        )
    countries = sorted({(r.model_extra or {}).get("country") for r in ds.rows if (r.model_extra or {}).get("country")})
    return _result(viz, ds, notes=notes, facts={"sites": n_sites, "countries": len(countries)})


# -- safety provisions -----------------------------------------------------------------------------

_SAFETY_TOPICS = [
    ("Rescue therapy", re.compile(r"rescue", re.I)),
    ("Independent monitoring", re.compile(r"monitor|dsmb|\bdmc\b|committee", re.I)),
    ("If the condition worsens", re.compile(r"worsen|progression", re.I)),
    ("Dose changes", re.compile(r"dose (?:reduction|interruption|modification|delay)", re.I)),
    ("Stopping treatment or the study", re.compile(r"discontinu|withdraw|stop|unacceptable toxicit", re.I)),
    ("Adverse events", re.compile(r"adverse|emergency|hospitali", re.I)),
]


def _safety_topic(h: Hit) -> str:
    for topic, rx in _SAFETY_TOPICS:
        if rx.search(h.term):
            return topic
    return "Safety"


@register(Intent.SAFETY_PROVISIONS)
async def safety_provisions(ctx: HandlerContext) -> HandlerResult:
    s = _trial(ctx)
    if s is None:
        return _trial_required("Safety provisions")
    ds = Dataset()
    dmc = s.get(HAS_DMC)
    if isinstance(dmc, bool):
        ds.emit(
            {
                "topic": "Independent monitoring",
                "source": "Oversight",
                "text": "Has a data monitoring committee" if dmc else "No data monitoring committee reported",
            },
            [s.ref(HAS_DMC)],
        )
    hits = scan_study(s, ["safety"], outcomes=True)
    shown = 0
    seen_text: set[str] = set()
    for h in hits:
        if shown >= MAX_TABLE_ROWS:
            break
        ref = s.text_ref(h.field_path, h.start, h.end, label=f"Matched '{h.term}'")
        if ref is None or ref.excerpt in seen_text:
            continue
        seen_text.add(ref.excerpt)
        ds.emit({"topic": _safety_topic(h), "source": _field_label(h.field_path), "text": ref.excerpt}, [ref])
        shown += 1
    notes = [
        note(
            NoteCode.NOT_IN_REGISTRY,
            "Detailed rules for rescue treatment, withdrawal and what happens if your condition worsens are in "
            "the study protocol and consent form, not the registry. You can leave a study at any time.",
        )
    ]
    if shown == 0:
        ds.emit(
            {"topic": "If the condition worsens", "source": "Record", "text": NOT_STATED},
            [s.ref("nct_id", label="Record checked")],
        )
        notes.append(
            note(
                NoteCode.NOT_IN_REGISTRY,
                "No sentence in the record describes rescue therapy, discontinuation or monitoring.",
            )
        )
    if len(hits) > shown and shown >= MAX_TABLE_ROWS:
        notes.append(note(NoteCode.TRUNCATED, f"Showing the first {MAX_TABLE_ROWS} matching sentences."))
    viz = table(
        ds.rows,
        title=f"Safety provisions in {_title(s)}",
        columns=[col("topic", "Topic"), col("text", "What the record says"), col("source", "Source")],
    )
    return _result(viz, ds, notes=notes, facts={"safety_sentences": shown, "has_dmc": dmc})


# -- limitations -----------------------------------------------------------------------------------


@register(Intent.STUDY_LIMITATIONS)
async def study_limitations(ctx: HandlerContext) -> HandlerResult:
    s = _trial(ctx)
    if s is None:
        return _trial_required("Study limitations")
    ds = Dataset()
    status = s.get("status")
    if status in ("TERMINATED", "WITHDRAWN", "SUSPENDED"):
        ds.emit({"topic": "Overall status", "text": status, "source": "Status"}, [s.ref("status")])
    sources = [
        ("Why the study stopped", P["why_stopped"], "Status"),
        ("Limitations and caveats", P["limitations"], "Results: more information"),
        ("Pre-assignment details", PRE_ASSIGNMENT, "Results: participant flow"),
        ("Recruitment details", RECRUITMENT, "Results: participant flow"),
    ]
    absent = []
    for topic, path, source in sources:
        pieces = _text_pieces(s, path)
        if not pieces:
            absent.append(topic.lower())
        for text, ref in pieces:
            ds.emit({"topic": topic, "text": text, "source": source}, [ref])
    notes = [
        note(
            NoteCode.NOT_IN_REGISTRY,
            "Protocol deviation logs and data-integrity audits are not published "
            "in ClinicalTrials.gov; only the fields shown here are.",
        )
    ]
    if not s.has_results:
        notes.append(
            note(
                NoteCode.NO_RESULTS_POSTED,
                "No results are posted, so limitations and participant-flow details are not available yet.",
            )
        )
    if not ds.rows:
        _not_stated_card(ds, s, "Limitations and caveats", "No limitations, caveats or stop reasons are reported.")
        return _result(kpi_cards(ds.rows, title=f"Limitations of {_title(s)}"), ds, notes=notes)
    if absent:
        notes.append(note(NoteCode.NOT_IN_REGISTRY, f"Not reported in this record: {', '.join(absent)}."))
    viz = table(
        ds.rows,
        title=f"Limitations and caveats of {_title(s)}",
        columns=[col("topic", "Topic"), col("text", "What the record says"), col("source", "Source")],
    )
    return _result(viz, ds, notes=notes, facts={"rows": len(ds.rows), "status": status})


# -- endpoints / biomarkers ------------------------------------------------------------------------


def _endpoint_rows(s: Study, ds: Dataset, keep: set[str] | None) -> int:
    n = 0
    for path, otype, o in _outcomes(s):
        mrefs = _whole_refs(s, f"{path}.measure", label=f"{otype} outcome")
        if not mrefs:
            continue
        cls = classify_endpoint(o.get("measure"), o.get("description"))
        if keep is not None and cls.category not in keep:
            continue
        row = {
            "endpoint": o["measure"],
            "outcome_type": otype,
            "category": CATEGORY_LABELS[cls.category],
            "category_key": cls.category,
            "source": "Outcome measure",
        }
        refs: list[FieldRef | None] = list(mrefs)
        if cls.term:
            row["matched_term"] = cls.term
        trefs = _whole_refs(s, f"{path}.timeFrame")
        if trefs:
            row["time_frame"] = o["timeFrame"]
            refs.extend(trefs)
        ds.emit(row, refs)
        n += 1
    return n


def _cohort_endpoint_bars(ctx: HandlerContext, keep: set[str] | None, title: str) -> HandlerResult:
    buckets: dict[tuple[str, str], list[FieldRef]] = defaultdict(list)
    studies = ctx.studies
    for s in studies:
        for path, otype, o in _outcomes(s):
            cls = classify_endpoint(o.get("measure"), o.get("description"))
            if keep is not None and cls.category not in keep:
                continue
            ref = s.ref(f"{path}.measure")
            if ref:
                buckets[(cls.category, otype)].append(ref)
    ds = Dataset()
    cats = [c for c in CATEGORY_ORDER if any(k[0] == c for k in buckets)]
    for cat in cats:
        for otype in ("Primary", "Secondary", "Other"):
            refs = buckets.get((cat, otype))
            if refs:
                ds.emit({"category": CATEGORY_LABELS[cat], "outcome_type": otype, "endpoint_count": len(refs)}, refs)
    notes = []
    if not ds.rows:
        notes.append(note(NoteCode.NOT_IN_REGISTRY, "No matching outcome measures are registered for these trials."))
        ds.emit({"label": "Endpoints", "value": NOT_STATED, "kind": "text"}, [])
        return HandlerResult(
            visualization=kpi_cards(ds.rows, title=title),
            notes=notes,
            full_refs=ds.full_refs,
            studies_analyzed=len(studies),
        )
    if any(c.sampled for c in ctx.cohorts):
        notes.append(note(NoteCode.SAMPLED, "Counts reflect a sample of the matching trials; see meta.coverage."))
    if any(not s.get("primary_outcomes") for s in studies):
        notes.append(
            note(
                NoteCode.NOT_IN_REGISTRY,
                "Some trials (often fetched with a reduced field set) list no outcome measures.",
            )
        )
    viz = bar_chart(
        ds.rows,
        title=title,
        x=Channel(
            field="category",
            type="nominal",
            title="Endpoint category",
            sort=ordered([CATEGORY_LABELS[c] for c in cats]),
        ),
        y=Channel(field="endpoint_count", type="quantitative", title="Endpoints", unit="endpoints", format=",d"),
        series=Channel(field="outcome_type", type="nominal", title="Outcome type"),
        mode="stacked",
        meta=VizMeta(
            grouping={
                "by": ["category", "outcome_type"],
                "aggregate": "count",
                "classifier": "regex over outcome measure titles",
            }
        ),
    )
    facts = {f"{CATEGORY_LABELS[c]}|{t}": len(v) for (c, t), v in buckets.items()}
    return HandlerResult(
        visualization=viz, notes=notes, facts=facts, full_refs=ds.full_refs, studies_analyzed=len(studies)
    )


# Uniform columns only (every row has them); optional cells are tooltip fields.
_ENDPOINT_COLUMNS = [
    col("endpoint", "Endpoint"),
    col("category", "Category"),
    col("outcome_type", "Type"),
    col("source", "Source"),
]
_ENDPOINT_TOOLTIP = ["time_frame", "matched_term"]


@register(Intent.ENDPOINTS_CATALOG)
async def endpoints_catalog(ctx: HandlerContext) -> HandlerResult:
    s = _trial(ctx)
    if s is None or (ctx.plan.scope is not Scope.SINGLE_TRIAL and len(ctx.trials) != 1):
        if not ctx.studies:
            return _trial_required("Endpoints")
        return _cohort_endpoint_bars(ctx, None, "Endpoint categories across trials")
    ds = Dataset()
    n = _endpoint_rows(s, ds, None)
    if n == 0:
        _not_stated_card(ds, s, "Endpoints")
        return _result(
            kpi_cards(ds.rows, title=f"Endpoints of {_title(s)}"),
            ds,
            notes=[note(NoteCode.NOT_IN_REGISTRY, "The record lists no outcome measures.")],
        )
    meta = VizMeta(grouping={"classifier": "regex over outcome measure titles (description as fallback)"})
    viz = table(
        ds.rows, title=f"Endpoints of {_title(s)}", columns=_ENDPOINT_COLUMNS, tooltip=_ENDPOINT_TOOLTIP, meta=meta
    )
    counts: dict[str, int] = defaultdict(int)
    for r in ds.rows:
        counts[(r.model_extra or {})["category"]] += 1
    return _result(viz, ds, facts=dict(counts))


@register(Intent.BIOMARKERS_PKPD)
async def biomarkers_pkpd(ctx: HandlerContext) -> HandlerResult:
    keep = {"pk", "pd", "biomarker"}
    s = _trial(ctx)
    if s is None or (ctx.plan.scope is not Scope.SINGLE_TRIAL and len(ctx.trials) != 1):
        if not ctx.studies:
            return _trial_required("Biomarkers and PK/PD")
        return _cohort_endpoint_bars(ctx, keep, "Biomarker and PK/PD endpoints across trials")
    ds = Dataset()
    n = _endpoint_rows(s, ds, keep)
    n_crit = 0
    for c in split_criteria(s.get("eligibility_criteria")):
        m = CATEGORIES["biomarker"].search(c.text)
        if not m:
            continue
        crefs = span_refs(
            s, P["eligibility_criteria"], c.start, c.end, label=f"{c.section.capitalize()} criterion {c.index}"
        )
        if crefs:
            ds.emit(
                {
                    "endpoint": c.text,
                    "category": "Biomarker",
                    "category_key": "biomarker",
                    "outcome_type": f"{c.section.capitalize()} criterion",
                    "matched_term": m.group(0),
                    "source": "Eligibility criteria",
                },
                crefs,
            )
            n_crit += 1
    if n + n_crit == 0:
        _not_stated_card(ds, s, "Biomarkers and PK/PD", "No outcome or criterion mentions biomarkers or PK/PD.")
        return _result(
            kpi_cards(ds.rows, title=f"Biomarkers and PK/PD in {_title(s)}"),
            ds,
            notes=[note(NoteCode.NOT_IN_REGISTRY, "No biomarker or PK/PD measures were found in this record.")],
        )
    notes = []
    if n == 0:
        notes.append(
            note(
                NoteCode.NOT_IN_REGISTRY,
                "No outcome measure is a biomarker or PK/PD measure; rows come from the eligibility criteria.",
            )
        )
    viz = table(
        ds.rows,
        title=f"Biomarkers and PK/PD in {_title(s)}",
        columns=_ENDPOINT_COLUMNS,
        tooltip=_ENDPOINT_TOOLTIP,
        meta=VizMeta(grouping={"classifier": "regex over outcome titles and eligibility criteria"}),
    )
    return _result(viz, ds, notes=notes, facts={"outcome_measures": n, "eligibility_mentions": n_crit})


# -- not in registry -------------------------------------------------------------------------------

_TOPICS = [
    ("Costs to participants", re.compile(r"cost|pay\b|paid|out[- ]of[- ]pocket|price|free", re.I)),
    ("Payment or compensation", re.compile(r"compensat|stipend|paid", re.I)),
    ("Travel and parking reimbursement", re.compile(r"travel|parking|reimburs|transport", re.I)),
    ("Insurance coverage", re.compile(r"insur", re.I)),
    (
        "Keeping your regular doctor",
        re.compile(
            r"\b(regular|own|usual|family|primary care) (doctor|physician|gp)\b|"
            r"\bmy (doctor|physician|gp)\b",
            re.I,
        ),
    ),
]
_DOCTOR_EXPLANATION = (
    "ClinicalTrials.gov does not record whether participants keep seeing their own doctor; "
    "the consent form and the study team explain how your usual care continues."
)
_COST_EXPLANATION = (
    "ClinicalTrials.gov has no fields for participant costs, payments, insurance coverage, or "
    "travel/parking reimbursement; this is set out in the consent form."
)


@register(Intent.NOT_IN_REGISTRY)
async def not_in_registry_fallback(ctx: HandlerContext) -> HandlerResult:
    q = ctx.request.query or ""
    topics = [t for t, rx in _TOPICS if rx.search(q)] or ["Costs, reimbursement and insurance"]
    doctor_only = topics == ["Keeping your regular doctor"]
    heading = "Your regular doctor" if doctor_only else "Costs and reimbursement"
    explanation = note(
        NoteCode.NOT_IN_REGISTRY,
        f"{_DOCTOR_EXPLANATION if doctor_only else _COST_EXPLANATION} Ask the study contact listed "
        "here, who can explain it.",
    )
    s = _trial(ctx)
    if s is None:
        res = _trial_required(heading)
        res.notes.insert(0, explanation)
        return res
    ds = Dataset()
    for t in topics:
        _not_stated_card(ds, s, t, "The registry has no field for this.")
    hits = [] if doctor_only else scan_study(s, ["costs"], eligibility=True, outcomes=True, locations=True)
    for h in hits[:10]:
        ref = s.text_ref(h.field_path, h.start, h.end, label=f"Matched '{h.term}'")
        if ref:
            ds.emit(
                {
                    "label": f"Mention in the record ({_field_label(h.field_path)})",
                    "value": ref.excerpt,
                    "kind": "text",
                    "detail": f"Matched '{h.term}'",
                },
                [ref],
            )
    n_contacts = _contact_rows(ds, s)
    notes = [explanation]
    if not hits and not doctor_only:
        notes.append(
            note(
                NoteCode.NOT_IN_REGISTRY,
                "No sentence in this record mentions costs, payment, insurance, travel or parking.",
            )
        )
    if n_contacts == 0:
        notes.append(note(NoteCode.NOT_IN_REGISTRY, "No central contact is listed; check the site list."))
    viz = kpi_cards(ds.rows, title=f"{heading}: {_title(s)}")
    return _result(viz, ds, notes=notes, facts={"cost_mentions": len(hits), "contacts": n_contacts})
