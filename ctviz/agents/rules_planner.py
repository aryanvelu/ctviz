"""Deterministic rule-based planner: VisualizeRequest -> QueryPlan without any LLM.

Used as the offline planner, as the fallback whenever the LLM planner is unavailable or fails, and as
a grounding signal for the LLM planner's validator. Pipeline:

1. NCT ids (structured fields + ``NCT\\d{8}`` in the text) decide single-trial context.
2. Intent = high-precision priority rules, then weighted keyword scoring over the regexes in
   ``data/intents.yaml`` (plus a few supplementary ones), filtered by scope compatibility. Then two
   deterministic overrides: a broad landscape question ("lung cancer trials", "overview of X") gets a
   richer default view (:func:`landscape_default`), and a chart word or ``preferred_visualization``
   ("sankey", "map", "timeline", ...) steers the intent to one that can draw it (:func:`steer_by_chart`).
3. Entities = alias-table scan + phrase patterns ("trials for X", "involving X", "X vs Y",
   "what is X doing", "doing in X"), then structured request fields override them.
4. Filters (years, phases, statuses), grouping, network kind, persona, then scope + cohorts.
5. A question with no entity, no NCT id, no structured field and no clinical-trial vocabulary
   (gibberish, or mostly non-English) gets NO cohorts, so the orchestrator asks which drug/condition/
   sponsor/trial is meant instead of silently analyzing "All trials".
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from functools import lru_cache

from ctviz.api.schemas.plan import CohortFilter, Intent, Persona, QueryPlan, Scope
from ctviz.api.schemas.request import NCT_PATTERN, PHASES, STATUSES, VisualizeRequest
from ctviz.api.schemas.response import ResolvedEntity
from ctviz.api.schemas.viz import VizType
from ctviz.core.answerability import is_subjective
from ctviz.core.entities import EntityKind, EntityResolver, get_resolver
from ctviz.core.intents import IntentSpec, load_intents
from ctviz.ctgov.fields import STATUS_GROUPS

NCT_RE = re.compile(r"\bNCT\d{8}\b", re.I)
_NCT_FULL = re.compile(NCT_PATTERN)
MIN_YEAR = 1990

# Supplementary keywords (the YAML is a frozen contract; these cover phrasings it misses).
EXTRA_KEYWORDS: dict[Intent, list[str]] = {
    Intent.DROPOUT_FLOW: [r"drop(ped|ping)? out", r"how many (patients|participants) (left|quit)"],
    Intent.DEMOGRAPHICS: [r"patient demographic", r"\bethnicit"],
    Intent.DEVELOPMENT_PROGRESSION: [r"advance[ds]? (from|into|to)", r"later phases"],
    Intent.STUDY_RATIONALE: [
        r"why (do|does|would) .*(work|help)",
        r"goal of (this|the) (study|trial)",
        r"\bpoint of (this|the) (study|trial)",
    ],
    Intent.RISKS_ADVERSE_EVENTS: [r"\bside[- ]?[ea]f+e?c?ts?\b"],
    Intent.RESULTS_OUTCOMES: [
        r"did (it|the drug|the treatment|they|this) (\w+ )?work",
        r"\b(meet|met|achieve[ds]?|hit) (its|the) (primary )?(endpoint|goal)",
    ],
    Intent.TRIAL_JOURNEY_TIMELINE: [r"how long (will|does|is)"],
    Intent.VISIT_BURDEN: [r"hospital stays?"],
    Intent.SITES_AND_CONTACTS: [r"\bnear (me|[A-Z]\w+)", r"where is .* recruiting", r"nearest site"],
    Intent.ENDPOINTS_CATALOG: [r"endpoints? (are|is) measured", r"safety, dos(age|ing),? or efficacy"],
    Intent.ENROLLMENT_VS_POWER: [r"statistical(ly)? power"],
    Intent.STUDY_LIMITATIONS: [r"protocol deviation"],
    Intent.INTERVENTION_LANDSCAPE: [
        r"common interventions?",
        r"\b(kinds?|types?|sorts?) of (interventions?|treatments?|drugs?|therap)",
        r"\btested (for|in)\b",
        r"\bpipeline\b",
    ],
    Intent.PHASE_DISTRIBUTION: [r"compare phases", r"phase (breakdown|mix|split)"],
    Intent.STATUS_DISTRIBUTION: [r"how many .* recruiting", r"status breakdown"],
}

# Generic words that alone are weak evidence for an intent.
WEAK_KEYWORDS = {
    r"phases?\b",
    "recruiting",
    "status",
    r"sponsors?\b",
    r"medications?\b",
    r"last\b",
    "where are",
    "missing",
    "efficacy",
    "detect",
    "what if",
    "outcome",
    "power",
    "location",
    "design",
    "mechanism",
    "purpose",
    "contact",
    "most trials",
    "region",
    r"map\b",
}

# Every user-facing string the planners emit comes from here (never from LLM free text).
TEMPLATES: dict[str, str] = {
    "TRIAL_REQUIRED": "Which trial do you mean? Add an NCT ID (e.g. NCT03548935) or a drug/condition.",
    "AMBIGUOUS": "Ambiguous term {terms}: please say which meaning you want.",
    "AMBIGUOUS_MEANINGS": "Ambiguous term: {details}.",
    "UNVERIFIED": (
        "I could not match the {kinds} in your question to the registry vocabulary; please check the "
        "spelling or use the generic/official name."
    ),
    "UNVERIFIED_NOTE": "UNVERIFIED_ENTITY: the {kinds} could not be verified and was not used as a filter.",
    "LOW_CONFIDENCE": "I read this as: {description} If that is not what you meant, please rephrase.",
    "MERGED": "{intent} does not support comparisons; entities were combined into one cohort.",
    "ALL_TRIALS": "No drug, condition or sponsor given; analyzing all registered trials.",
    "UNCLEAR": (
        "I could not find a drug, condition, sponsor or trial ID in the question (questions are read in "
        "English); please name one."
    ),
    "LANDSCAPE": (
        "Broad question: showing {view} as an overview. Ask for a trend, phase, status or country "
        "breakdown for a different view."
    ),
    "PLACE_NOT_FILTERED": (
        "'{place}' is not a country: the registry search filters locations by country only, so "
        "sites are not limited to {place}."
    ),
}
_LANDSCAPE_VIEW = {
    Intent.ENTITY_NETWORK: "a network of sponsors and the drugs they test",
    Intent.SPONSOR_ACTIVITY: "a timeline of the sponsor's trials",
}
_LANDSCAPE_VIEW_DRUG = "a network of the drug and the conditions it is tested in"

_TIMEY = r"per year|by year|each year|every year|over time|\btrend|since \d{4}|per month|annual"
# Phrases removed before keyword scoring (population words that look like intent keywords).
_MASKS = [re.compile(r"\b(cancer )?survivors?\b", re.I)]

# High-precision rules checked before scoring: (pattern, intent, unless-pattern).
_ROW_WORD = (
    r"(?:sponsor(?:s| class(?:es)?| categor(?:y|ies)| types?)?|compan(?:y|ies)|drugs?|interventions?|"
    r"treatments?|therap(?:y|ies)|conditions?|diseases?|indications?|countr(?:y|ies))"
)
_COL_WORD = r"(?:phases?|(?:recruitment |overall )?status(?:es)?|start years?|years?)"
_JOIN = r"\s+(?:by|x|×|vs\.?|versus|against|and|across|per|over)\s+(?:the\s+|each\s+)?"
# "heatmap of sponsors by phase", "drugs × phase matrix", "phase by sponsor heatmap": a row dimension is named.
_HEAT_ROWS = re.compile(
    rf"(?:heat ?maps?|\bmatrix)\s+(?:of|for|showing|with)\s+(?:the\s+)?(?:top\s+\d+\s+)?(?P<r1>{_ROW_WORD})\b|"
    rf"\b(?P<r2>{_ROW_WORD}){_JOIN}(?P<c2>{_COL_WORD})\b|\b(?P<c3>{_COL_WORD}){_JOIN}(?P<r3>{_ROW_WORD})\b",
    re.I,
)
_HEAT = re.compile(r"heat ?maps?|\bmatrix\b", re.I)

# High-precision rules checked before scoring: (pattern, intent, unless-pattern).
PRIORITY_RULES: list[tuple[str, Intent, str | None]] = [
    (r"\bgaps?\b|white ?space|under-?studied", Intent.LANDSCAPE_GAP_HEATMAP, None),
    # a heatmap whose rows are sponsors/drugs/conditions/countries is the gap heatmap (phase x status has no rows)
    (rf"^(?=[\s\S]*(?:heat ?map|\bmatrix\b))[\s\S]*?(?:{_HEAT_ROWS.pattern})", Intent.LANDSCAPE_GAP_HEATMAP, None),
    (r"heat ?map|\bmatrix\b", Intent.PHASE_STATUS_HEATMAP, None),
    (
        r"regular doctor|\bmy (own |usual |family |regular )?(doctor|physician|gp)\b|\bown (doctor|physician)\b|"
        r"primary care (doctor|physician)",
        Intent.NOT_IN_REGISTRY,
        r"where can i|\bsites?\b|near me|\bcontact",
    ),
    (
        r"\bnetwork\b|co-?occur|combination (studies|trials|therap)|\bwork together\b|\bcombined with\b|"
        r"\bcombinations? (of|with)\b|\b(paired|co-?administered|given together) with\b|"
        r"\bwhat (is|are) .{1,60} (combined|paired|co-?administered)\b|"
        r"\b(relationships?|connections?|connect(s|ed)?|links?|linked)\b.{0,40}\b(sponsors?|drugs?|compan(y|ies)|"
        r"conditions?|diseases?|investigators?|collaborators?)\b|"
        r"\b(sponsors?|drugs?|compan(y|ies)|conditions?|investigators?)\b.{0,40}\b(relate|connect|link)",
        Intent.ENTITY_NETWORK,
        _TIMEY,
    ),
    (
        r"\btrial sites\b|\bsites? (in|across)\b|\b(which|what) (sites|hospitals|cities|facilities)\b",
        Intent.GEOGRAPHIC_DISTRIBUTION,
        r"\bNCT\d|\bthis (trial|study)\b|\bcontact",
    ),
    # "where are obesity trials recruiting", "trials near Boston", "recruiting sites in Texas" (a cohort map)
    (
        r"\bwhere\b[^?.]{0,60}\b(recruit\w*|enroll\w*|running|conducted|located|taking place|held|being (run|done|"
        r"tested))\b|\bnear (?!me\b)\w+|\b(recruiting|active|open) (sites?|locations?|hospitals?|cent(er|re)s?)\b|"
        r"\b(sites?|locations?) (near|around)\b",
        Intent.GEOGRAPHIC_DISTRIBUTION,
        r"\bNCT\d|\bthis (trial|study)\b|\bcontact|where can i|near me",
    ),
    (
        r"first[- ]in[- ]human|development history|advance[ds]? from|progress(ed)? from",
        Intent.DEVELOPMENT_PROGRESSION,
        None,
    ),
    (r"\bwhat (is|are) .+ doing\b|\bpipeline\b|\bportfolio\b", Intent.SPONSOR_ACTIVITY, None),  # needs a sponsor
    (r"\bendpoints?\b", Intent.ENDPOINTS_CATALOG, r"\bresults?\b|did .* work|\b(meet|met|achieve[ds]?|hit) (its|the)"),
    (r"statistical(ly)? power|adequately powered|power calculation", Intent.ENROLLMENT_VS_POWER, None),
    (r"drop(ped|ping)? ?outs?\b|attrition", Intent.DROPOUT_FLOW, None),
    (
        r"\b(which|what|top|most|how many)\b[^?.]*\bcountr(y|ies)\b|\b(by|per|each) countr(y|ies)\b|"
        r"\bcountr(y|ies) (with|have|has|run|running)\b",
        Intent.GEOGRAPHIC_DISTRIBUTION,
        _TIMEY,
    ),
    (r"\bsurviv(al|e|ing)\b|kaplan|progression[- ]free", Intent.SURVIVAL_SUMMARY, None),
    (r"out[- ]of[- ]pocket|reimburs|\bcost (me|anything)", Intent.NOT_IN_REGISTRY, None),
]

# Intents whose x-axis is a phase/status dimension: never filter on that dimension from the text.
PHASE_DIM = {
    Intent.PHASE_DISTRIBUTION,
    Intent.PHASE_STATUS_HEATMAP,
    Intent.DEVELOPMENT_PROGRESSION,
    Intent.LANDSCAPE_GAP_HEATMAP,
}
STATUS_DIM = {Intent.STATUS_DISTRIBUTION, Intent.PHASE_STATUS_HEATMAP}

_FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|i'd|my|mine|myself|am i)\b|(?<!tell )(?<!show )(?<!give )\bme\b", re.I)
_TRIAL_HINT = re.compile(r"\b(this|the|that|my) (study|trial)\b", re.I)
_COMPARE_CUE = re.compile(r"\b(vs\.?|versus|compare[ds]?|comparing|comparison|against|relative to|between)\b", re.I)

_GENERIC = {
    "this",
    "that",
    "it",
    "drug",
    "drugs",
    "treatment",
    "treatments",
    "trial",
    "trials",
    "study",
    "studies",
    "patients",
    "patient",
    "people",
    "participants",
    "sponsor",
    "sponsors",
    "country",
    "countries",
    "combination",
    "network",
    "heatmap",
    "phase",
    "phases",
    "status",
    "statuses",
    "recruitment",
    "enrollment",
    "year",
    "years",
    "time",
    "intervention",
    "interventions",
    "type",
    "types",
    "results",
    "design",
    "endpoints",
    "sites",
    "me",
    "you",
    "them",
    "other",
    "different",
    "similar",
    "such",
    "these",
    "those",
    "industry",
    "academic",
    "category",
    "categories",
    "number",
    "count",
    "trend",
    "data",
    "my",
    "our",
    "their",
    "registry",
    "clinical",
    "medical",
    "safety",
    "efficacy",
    "dosage",
    "outcomes",
    "visits",
    "risks",
    "side",
    "effects",
    "cost",
    "biomarkers",
    "researcher",
    "researchers",
    "scientists",
    "doctors",
    "doctor",
    "companies",
    "company",
    "they",
    "investigators",
    "pipeline",
    "portfolio",
    "oncology",
    # verbs / chart words / date words that the phrase patterns used to read as condition names
    "studied",
    "tested",
    "researched",
    "investigated",
    "examined",
    "evaluated",
    "used",
    "start",
    "started",
    "date",
    "dates",
    "scatter",
    "plot",
    "chart",
    "graph",
    "histogram",
    "sankey",
    "map",
    "timeline",
    "pie",
    "curve",
    "diagram",
    "landscape",
    "overview",
    "connect",
    "combined",
    "relationship",
}
_LEAD_STOP = {
    "how",
    "are",
    "is",
    "was",
    "were",
    "the",
    "a",
    "an",
    "all",
    "any",
    "of",
    "for",
    "in",
    "many",
    "much",
    "which",
    "what",
    "show",
    "me",
    "list",
    "recent",
    "current",
    "ongoing",
    "new",
    "active",
    "recruiting",
    "completed",
    "terminated",
    "interventional",
    "observational",
    "clinical",
    "early",
    "late",
    "stage",
    "did",
    "do",
    "does",
    "there",
    "most",
    "top",
    "common",
    "big",
    "large",
    "small",
    "registered",
    "and",
    "or",
    "by",
    "per",
    "each",
    "i",
    "ii",
    "iii",
    "iv",
    "1",
    "2",
    "3",
    "4",
    "compare",
    "comparing",
    "across",
    "between",
    "number",
    "total",
    "involving",
    "with",
    "trials",
    "studies",
    "on",
    "to",
    "currently",
    "still",
    "now",
    "actively",
    "withdrawn",
    "suspended",
    "enrolling",
    "not",
    "yet",
}

# Chart words in the question -> VizType (checked in order; the first match wins, so "heat map" is a heatmap,
# not a geo map). A structured ``preferred_visualization`` always wins over these.
CHART_WORDS: list[tuple[re.Pattern[str], VizType]] = [
    (re.compile(p, re.I), v)
    for p, v in [
        (r"\bkaplan|\bmeier\b|\bsurvival curves?\b|\bkm (curves?|plots?)\b", VizType.SURVIVAL_CURVE),
        (r"\bforest ?plots?\b", VizType.FOREST_PLOT),
        (r"\bsankey\b|\balluvial\b|\bflow (diagram|chart)\b|\bconsort (diagram|chart)\b", VizType.SANKEY),
        (r"\bnetwork\b|\bnode[- ]link\b", VizType.NETWORK_GRAPH),
        (r"\bheat ?maps?\b|\bmatrix\b", VizType.HEATMAP),
        (r"\b(geo ?|world |choropleth )?maps?\b(?! (to|onto)\b)|\bon a map\b", VizType.GEO_MAP),
        (r"\btimelines?\b|\bgantt\b|\bswim ?lanes?\b", VizType.TIMELINE),
        (r"\bhistograms?\b", VizType.HISTOGRAM),
        (r"\bscatter( ?plots?| ?charts?)?\b", VizType.SCATTER_PLOT),
        (r"\bpie( charts?)?\b|\bdonut\b|\bdoughnut\b", VizType.PIE_CHART),
        (r"\bpower curves?\b", VizType.POWER_CURVE),
        (r"\bchecklist\b", VizType.ELIGIBILITY_CHECKLIST),
        (r"\bline (charts?|graphs?|plots?)\b|\btime[- ]series\b", VizType.TIME_SERIES),
        (r"\bbar (charts?|graphs?|plots?)\b", VizType.BAR_CHART),
        (r"\b(as|in) a table\b|\btabular\b", VizType.TABLE),
    ]
]
# Views every intent can fall back to: they never change the intent.
GENERIC_VIZ = {VizType.BAR_CHART, VizType.TABLE, VizType.KPI_CARDS}
# Trial-level charts: with no NCT id, a single-trial intent (which then asks "which trial?") still beats a
# cohort chart the user did not ask for.
_TRIAL_LEVEL_VIZ = {
    VizType.SURVIVAL_CURVE,
    VizType.FOREST_PLOT,
    VizType.POWER_CURVE,
    VizType.ELIGIBILITY_CHECKLIST,
    VizType.THRESHOLD_LINE,
}

# Broad "landscape" wording (no specific view named).
_LANDSCAPE = re.compile(
    r"\blandscape\b|\boverview\b|\btell me (about|everything)\b|\bsummar(y|ize|ise)\b|"
    r"\bexplore\b|\bwhat('s| is) (going on|happening|out there)\b|\binfo(rmation)? (on|about)\b|"
    r"\bresearch (on|in|into)\b|\b(state|scope) of\b",
    re.I,
)
# "How many ..." asks for a count (the trend's totals), not a landscape overview.
_COUNT_Q = re.compile(r"\bhow many\b|\bnumber of\b|\bcount(s|ing)?\b|\btotal\b", re.I)
# Clinical-trial vocabulary: a question with none of it, no entity and no NCT id is not answerable as asked.
_VOCAB = re.compile(
    r"\b(trials?|stud(y|ies)|drugs?|medications?|medicines?|treatments?|therap(y|ies)|vaccines?|"
    r"sponsors?|compan(y|ies)|phases?|recruit\w*|enrol\w*|conditions?|diseases?|disorders?|"
    r"patients?|participants?|placebo|interventions?|registry|clinical|endpoints?|outcomes?|"
    r"results?|sites?|countr(y|ies)|eligib\w*|cancer|oncology|investigators?|dropouts?|adverse|"
    r"side effects?|safety|efficacy|nct)\b",
    re.I,
)
# Function words of other languages the planner is likely to see (Spanish, French, German, Portuguese, Italian).
_FOREIGN = {
    "de",
    "del",
    "la",
    "las",
    "el",
    "los",
    "en",
    "por",
    "para",
    "con",
    "sobre",
    "hay",
    "cuántos",
    "cuantos",
    "cuántas",
    "cuantas",
    "qué",
    "cuál",
    "cual",
    "dónde",
    "donde",
    "ensayos",
    "ensayo",
    "estudios",
    "estudio",
    "clínicos",
    "clinicos",
    "y",
    "una",
    "son",
    "está",
    "están",
    "muestra",
    "muéstrame",
    "año",
    "años",
    "le",
    "les",
    "des",
    "du",
    "sur",
    "pour",
    "avec",
    "combien",
    "essais",
    "essai",
    "cliniques",
    "clinique",
    "quels",
    "quelles",
    "quel",
    "où",
    "est",
    "sont",
    "et",
    "ou",
    "dans",
    "montre",
    "wie",
    "viele",
    "studien",
    "studie",
    "klinische",
    "gibt",
    "und",
    "oder",
    "der",
    "das",
    "mit",
    "für",
    "zu",
    "welche",
    "wo",
    "sind",
    "ist",
    "quantos",
    "ensaios",
    "estudos",
    "onde",
    "são",
    "quanti",
    "studi",
    "sperimentazioni",
    "dove",
    "sono",
    "il",
    "di",
    "per",
}

_TAIL = (
    r"(?=\s*(?:[?.,;!:]|$)|\s+(?:since|from|between|before|after|in|by|per|over|vs\.?|versus|and|or|across|"
    r"trials?|studies|that|which|with|for|during|changed|started|each|is|are|were|have|has|compared|"
    r"recruiting|doing|been)\b)"
)
_ENT = r"(?P<e>[A-Za-z0-9][\w'/\-]*(?:\s+[A-Za-z0-9][\w'/\-]*){0,4}?)"
# (pattern, forced kind or None, strong?)  Strong patterns accept unknown phrases; weak ones only confident guesses.
_PATTERNS: list[tuple[re.Pattern[str], EntityKind | None, bool]] = [
    # "the condition X", "a drug called X": the user labels the kind, so unknown names are kept (and later
    # verified against ClinicalTrials.gov by a count probe; zero matches -> needs_clarification).
    (
        re.compile(r"\b(?:condition|disease|disorder|indication)s?\s+(?:called\s+|named\s+)?" + _ENT + _TAIL, re.I),
        "condition",
        True,
    ),
    (
        re.compile(r"\b(?:drug|medication|medicine|compound)s?\s+(?:called\s+|named\s+)" + _ENT + _TAIL, re.I),
        "drug",
        True,
    ),
    (re.compile(r"\bwhat\s+(?:is|are)\s+" + _ENT + r"\s+doing\b", re.I), "sponsor", True),
    (re.compile(r"\bdoing\s+in\s+" + _ENT + _TAIL, re.I), "condition", True),
    (
        re.compile(r"\b(?:trials?|studies)\s+(?:for|of|on|involving|testing|evaluating|using)\s+" + _ENT + _TAIL, re.I),
        None,
        True,
    ),
    (re.compile(r"\binvolving\s+" + _ENT + _TAIL, re.I), None, True),
    (re.compile(r"\b(?:for|of)\s+" + _ENT + r"\s+(?:trials?|studies)\b", re.I), None, False),
    (re.compile(_ENT + r"\s+(?:trials?|studies)\b", re.I), None, False),
]
_VS = re.compile(r"\s+(?:vs\.?|versus)\s+", re.I)
_VS_LEFT = re.compile(r"([A-Za-z0-9][\w'/\-]*(?:\s+[A-Za-z0-9][\w'/\-]*){0,2})$")
_VS_RIGHT = re.compile(
    r"[A-Za-z0-9][\w'/\-]*(?:\s+(?!(?:trials?|studies|in|for|since|from|by|per|and|or|across|over|"
    r"with|during|on|vs|versus)\b)[A-Za-z0-9][\w'/\-]*){0,2}",
    re.I,
)

_YEAR = r"((?:19|20)\d{2})"
_PHASE_TOKEN = r"(1|2|3|4|iv|iii|ii|i)"
_PHASE_MAP = {
    "1": "PHASE1",
    "i": "PHASE1",
    "2": "PHASE2",
    "ii": "PHASE2",
    "3": "PHASE3",
    "iii": "PHASE3",
    "4": "PHASE4",
    "iv": "PHASE4",
}


@dataclass
class Found:
    """A resolved entity mention (from text or a structured field)."""

    kind: EntityKind
    entity: ResolvedEntity
    start: int = 10**6
    is_class: bool = False

    @property
    def terms(self) -> list[str]:
        return list(self.entity.search_terms) if self.is_class else [self.entity.resolved]


@dataclass
class Extraction:
    """Everything the rule planner pulled from a request; reused by the LLM plan validator."""

    nct_ids: list[str] = field(default_factory=list)
    found: list[Found] = field(default_factory=list)
    ambiguous: dict[str, list[str]] = field(default_factory=dict)
    start_year: int | None = None
    end_year: int | None = None
    phases: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    phases_from_request: bool = False
    statuses_from_request: bool = False
    compare_cue: bool = False
    trial_hint: bool = False
    chart: VizType | None = None  # requested chart: structured field, else a chart word in the text
    chart_from_request: bool = False
    unintelligible: bool = False  # no entity / NCT / structured field / trial vocabulary (or not English)
    heat_dims: tuple[str | None, str | None] = (None, None)  # (rows, columns) named for a heatmap
    rule: str = ""  # which planner rule picked the rule intent (keywords, priority:..., landscape, chart:...)
    places: list[str] = field(default_factory=list)  # cities/states named for a map (never used as countries)


# ---------------------------------------------------------------------------------------- keyword tables
@lru_cache(maxsize=1)
def _keyword_table() -> list[tuple[Intent, re.Pattern[str], float]]:
    table: list[tuple[Intent, re.Pattern[str], float]] = []
    for intent, spec in load_intents().items():
        for kw in [*spec.keywords, *EXTRA_KEYWORDS.get(intent, [])]:
            table.append((intent, re.compile(kw, re.I), 0.4 if kw in WEAK_KEYWORDS else 1.0))
    return table


_PRIORITY = [(re.compile(p, re.I), i, re.compile(u, re.I) if u else None) for p, i, u in PRIORITY_RULES]


def _supports(spec: IntentSpec, scope: Scope) -> bool:
    return scope in spec.scopes


def _mask(text: str) -> str:
    for rx in _MASKS:
        text = rx.sub(" population ", text)
    return text


def score_intents(
    text: str, *, trial_ctx: bool, cohort_ctx: bool, has_sponsor: bool = True
) -> list[tuple[Intent, float]]:
    """Keyword scores per intent (descending), after scope-compatibility filtering/penalties."""
    specs = load_intents()
    text = _mask(text)
    scores: dict[Intent, float] = {}
    for intent, rx, w in _keyword_table():
        if rx.search(text):
            scores[intent] = scores.get(intent, 0.0) + w
    out: list[tuple[Intent, float]] = []
    order = list(Intent)
    for intent, s in scores.items():
        spec = specs[intent]
        if trial_ctx and not _supports(spec, Scope.SINGLE_TRIAL):
            continue
        if intent is Intent.SPONSOR_ACTIVITY and not has_sponsor:
            continue  # "pipeline" without a sponsor is not one sponsor's activity
        if not trial_ctx and cohort_ctx and spec.scopes == [Scope.SINGLE_TRIAL]:
            s *= 0.6
        out.append((intent, s))
    out.sort(key=lambda kv: (-kv[1], order.index(kv[0])))
    return out


def classify_intent(
    text: str, *, trial_ctx: bool, cohort_ctx: bool, has_sponsor: bool = True
) -> tuple[Intent, float, str]:
    """Pick an intent. Returns ``(intent, confidence, how)`` where ``how`` names the rule that fired."""
    specs = load_intents()
    masked = _mask(text)
    for rx, intent, unless in _PRIORITY:
        if rx.search(masked) and not (unless and unless.search(masked)):
            if trial_ctx and not _supports(specs[intent], Scope.SINGLE_TRIAL):
                continue
            if intent is Intent.SPONSOR_ACTIVITY and not has_sponsor:
                continue
            return intent, 0.9, f"priority:{rx.pattern}"
    ranked = score_intents(text, trial_ctx=trial_ctx, cohort_ctx=cohort_ctx, has_sponsor=has_sponsor)
    if not ranked:
        fallback = Intent.SINGLE_TRIAL_OVERVIEW if trial_ctx else Intent.TREND_OVER_TIME
        return fallback, 0.3, "default"
    top, s1 = ranked[0]
    s2 = ranked[1][1] if len(ranked) > 1 else 0.0
    if s1 >= 1.0 and s1 - s2 >= 0.5:
        conf = 0.8
    elif s1 >= 1.0:
        conf = 0.65
    else:
        conf = 0.55
    return top, conf, "keywords"


# ---------------------------------------------------------------------------------------- extraction
def _clean_candidate(raw: str) -> str | None:
    toks = raw.strip(" '\"-").split()
    while toks and toks[0].lower() in _LEAD_STOP:
        toks.pop(0)
    while toks and toks[-1].lower() in {"trial", "trials", "study", "studies", "drug", "drugs", "patients"}:
        toks.pop()
    if not toks or any(t.lower() in _GENERIC for t in toks) or len(toks) > 5:
        return None
    cand = " ".join(toks).strip(" '\"-")
    if len(cand) < 2 or cand.isdigit() or NCT_RE.fullmatch(cand):
        return None
    return cand


def extract_ncts(request: VisualizeRequest) -> list[str]:
    """NCT ids from structured fields first, then the text (deduped, order kept, max 5)."""
    ids = [*([request.nct_id] if request.nct_id else []), *request.nct_ids]
    ids += [m.group(0).upper() for m in NCT_RE.finditer(request.query)]
    return [x for x in dict.fromkeys(i.upper() for i in ids) if _NCT_FULL.match(x)][:5]


def extract_years(text: str, today: dt.date | None = None) -> tuple[int | None, int | None]:
    """``since 2015`` / ``2015-2020`` / ``between 2015 and 2020`` / ``before 2020`` / ``in 2019`` / ``last 5 years``."""
    now = (today or dt.date.today()).year
    lo = hi = None
    rng = re.search(rf"\bbetween\s+{_YEAR}\s+and\s+{_YEAR}\b", text, re.I) or re.search(
        rf"\b{_YEAR}\s*(?:-|–|to|through|until)\s*{_YEAR}\b", text, re.I
    )
    if rng:
        lo, hi = int(rng.group(1)), int(rng.group(2))
    else:
        if m := re.search(rf"\b(since|from|starting(?: in)?|after)\s+{_YEAR}\b", text, re.I):
            lo = int(m.group(2)) + (1 if m.group(1).lower() == "after" else 0)
        if m := re.search(rf"\b(before|prior to|until|through|up to)\s+{_YEAR}\b", text, re.I):
            hi = int(m.group(2)) - (1 if m.group(1).lower() in ("before", "prior to") else 0)
        if lo is None and hi is None and (m := re.search(rf"\bin\s+{_YEAR}\b", text, re.I)):
            lo = hi = int(m.group(1))
        if lo is None and (m := re.search(r"\b(?:last|past)\s+(\d{1,2})\s+years\b", text, re.I)):
            lo = now - int(m.group(1))
    return clamp_years(lo, hi, today)


def clamp_years(lo: int | None, hi: int | None, today: dt.date | None = None) -> tuple[int | None, int | None]:
    """Clamp to 1990..current+5 and swap an inverted range."""
    top = (today or dt.date.today()).year + 5
    lo = None if lo is None else min(max(lo, MIN_YEAR), top)
    hi = None if hi is None else min(max(hi, MIN_YEAR), top)
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    return lo, hi


def extract_phases(text: str) -> list[str]:
    """``phase 3``, ``phase III``, ``phase 2/3``, ``early phase 1`` -> PHASE enum values."""
    out: list[str] = []
    if re.search(r"\bearly phase\s*(1|i)\b", text, re.I):
        out.append("EARLY_PHASE1")
        text = re.sub(r"\bearly phase\s*(1|i)\b", " ", text, flags=re.I)
    rx = rf"\bphases?\s*{_PHASE_TOKEN}[ab]?(?:\s*(?:/|-|and|or|&|,)\s*(?:phase\s*)?{_PHASE_TOKEN}[ab]?)?\b"
    for m in re.finditer(rx, text, re.I):
        for g in m.groups():
            if g:
                out.append(_PHASE_MAP[g.lower()])
    return list(dict.fromkeys(out))


def extract_statuses(text: str) -> list[str]:
    """Status filters phrased in the text (``recruiting trials``, ``completed studies`` ...)."""
    t = text.lower()
    out: list[str] = []
    if "not yet recruiting" in t:
        out.append("NOT_YET_RECRUITING")
    if re.search(r"active,? not recruiting", t):
        out.append("ACTIVE_NOT_RECRUITING")
    if re.search(r"(?<!not yet )(?<!not )\brecruiting\b", t):
        out.append("RECRUITING")
    if re.search(r"\b(active|ongoing|open)\b.{0,20}\b(vs\.?|versus|and|or)\b", t):
        return []  # "active vs completed" is a split by status group, not a filter
    for word, status in (
        ("completed", "COMPLETED"),
        ("terminated", "TERMINATED"),
        ("withdrawn", "WITHDRAWN"),
        ("suspended", "SUSPENDED"),
    ):
        # "completed trials", "trials that (have) completed", "trials that were terminated"
        if re.search(
            rf"\b{word} (\w+ )?(trials|studies)\b|\b(trials|studies) (that|which) (have |had |were |"
            rf"was |are )?(been )?{word}\b",
            t,
        ):
            out.append(status)
    if re.search(
        r"\b(finished|ended|closed|done|over) (\w+ )?(trials|studies)\b|\b(trials|studies) (that|which) "
        r"(have |had |are )?(finished|ended|closed)\b",
        t,
    ):
        out += list(STATUS_GROUPS["DONE"])
    if re.search(r"\b(active|ongoing|open|current) (\w+ )?(trials|studies)\b", t):
        out += list(STATUS_GROUPS["ACTIVE"])
    return [s for s in dict.fromkeys(out) if s in STATUSES]


def detect_chart(text: str) -> VizType | None:
    """The chart type a question names ("show me a sankey of ...", "map of ...", "kaplan meier"), if any."""
    return next((v for rx, v in CHART_WORDS if rx.search(text)), None)


def heatmap_dims(text: str) -> tuple[str | None, str | None]:
    """``(row group_by, column series_by)`` named for a heatmap ("sponsors by phase" -> sponsor, phase)."""
    m = _HEAT_ROWS.search(text)
    if not m:
        return None, None
    row = (m.group("r1") or m.group("r2") or m.group("r3") or "").lower()
    col = (m.group("c2") or m.group("c3") or "").lower()
    if re.match(r"sponsor (class|categor|type)", row):
        row_dim = "sponsor_class"
    elif re.match(r"sponsor|compan", row):
        row_dim = "sponsor"
    elif re.match(r"drug|intervention|treatment|therap", row):
        row_dim = "intervention"
    elif re.match(r"countr", row):
        row_dim = "country"
    else:
        row_dim = "condition"
    if not col:  # "heatmap of sponsors" (column named elsewhere, or the handler's phase default)
        col = next((w for w in ("status", "recruit", "year", "phase") if re.search(rf"\b{w}", text, re.I)), "")
    col_dim = "status" if re.search(r"status|recruit", col) else "year" if "year" in col else "phase"
    return row_dim, col_dim


def is_non_english(text: str) -> bool:
    """Mostly words of another language (common Spanish/French/German/... function words or accented words)."""
    toks = re.findall(r"[^\W\d_]+", text.lower())
    foreign = sum(1 for t in toks if t in _FOREIGN or any(ord(c) > 127 for c in t))
    return foreign >= 2 and foreign >= 0.5 * len(toks)


def _ambiguous_mentions(
    text: str, resolver: EntityResolver, spans: list[tuple[int, int]] | None = None
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    spans = spans or []
    for key, meanings in resolver.ambiguous.items():
        for m in re.finditer(rf"(?<![\w/-]){re.escape(key)}(?![\w/-])", text, re.I):
            surface = m.group(0)
            if len(key) <= 4 and surface != surface.upper():
                continue
            if any(m.start() < e and s < m.end() for s, e in spans):
                continue  # part of a longer known entity ("Merck Sharp & Dohme", "Novo Nordisk")
            out[surface] = meanings
    return out


def extract_entities(text: str, resolver: EntityResolver | None = None) -> tuple[list[Found], dict[str, list[str]]]:
    """Entity mentions in free text: alias-table scan, then phrase patterns for unknown names."""
    resolver = resolver or get_resolver()
    found: list[Found] = []
    spans: list[tuple[int, int]] = []
    for m in resolver.scan(text):
        found.append(Found(kind=m.kind, entity=m.entity, start=m.start, is_class=m.is_class))
        spans.append((m.start, m.end))
    scanned = list(spans)

    def overlaps(a: int, b: int) -> bool:
        return any(a < e and s < b for s, e in spans)

    def add(raw: str, start: int, end: int, forced: EntityKind | None, strong: bool) -> None:
        if overlaps(start, end):
            return
        cand = _clean_candidate(raw)
        if not cand:
            return
        kind, confident = resolver.guess_kind(cand)
        if forced:
            kind = forced
        elif not (confident or strong):
            return
        offset = raw.find(cand)
        found.append(Found(kind=kind, entity=resolver.resolve(cand, kind), start=start + max(offset, 0)))
        spans.append((start, end))

    for rx, forced, strong in _PATTERNS:
        for m in rx.finditer(text):
            add(m.group("e"), m.start("e"), m.end("e"), forced, strong)
    for m in _VS.finditer(text):
        left = _VS_LEFT.search(text[: m.start()])
        right = _VS_RIGHT.match(text, m.end())
        if not left or not right:
            continue
        sides = [(left.group(1), left.start(1), left.end(1)), (right.group(0), right.start(), right.end())]
        kinds = {resolver.guess_kind(c)[0] for c in (_clean_candidate(r) for r, _, _ in sides) if c} or {"drug"}
        forced: EntityKind | None = next(iter(kinds)) if len(kinds) == 1 else None  # type: ignore[assignment]
        for raw, a, b in sides:
            add(raw, a, b, forced, True)
    dedup: dict[tuple[str, str], Found] = {}
    for f in sorted(found, key=lambda f: f.start):
        dedup.setdefault((f.kind, f.entity.resolved.lower()), f)
    return list(dedup.values()), _ambiguous_mentions(text, resolver, scanned)


def _structured(request: VisualizeRequest, resolver: EntityResolver) -> dict[EntityKind, list[Found]]:
    out: dict[EntityKind, list[Found]] = {}
    pairs: list[tuple[EntityKind, list[str]]] = [
        ("drug", [*([request.drug_name] if request.drug_name else []), *request.drug_names]),
        ("condition", [*([request.condition] if request.condition else []), *request.conditions]),
        ("sponsor", [request.sponsor] if request.sponsor else []),
        ("country", [request.country] if request.country else []),
    ]
    for kind, values in pairs:
        vals = list(dict.fromkeys(v.strip() for v in values if v and v.strip()))
        if vals:
            out[kind] = [
                Found(
                    kind=kind,
                    entity=resolver.resolve(v, kind, method="structured_field"),
                    start=-1,
                    is_class=kind == "drug" and resolver.is_drug_class(v),
                )
                for v in vals
            ]
    return out


def extract(
    request: VisualizeRequest, resolver: EntityResolver | None = None, today: dt.date | None = None
) -> Extraction:
    """Run all extractors. Structured request fields replace text mentions of the same kind."""
    resolver = resolver or get_resolver()
    text = request.query
    ex = Extraction(nct_ids=extract_ncts(request))
    found, ex.ambiguous = extract_entities(text, resolver)
    structured = _structured(request, resolver)
    ex.found = [f for f in found if f.kind not in structured] + [f for fs in structured.values() for f in fs]
    ex.found.sort(key=lambda f: f.start)
    lo, hi = extract_years(text, today)
    ex.start_year = request.start_year if request.start_year is not None else lo
    ex.end_year = request.end_year if request.end_year is not None else hi
    ex.phases = list(request.trial_phase) or extract_phases(text)
    ex.statuses = list(request.status) or extract_statuses(text)
    ex.phases_from_request = bool(request.trial_phase)
    ex.statuses_from_request = bool(request.status)
    ex.compare_cue = bool(_COMPARE_CUE.search(text))
    ex.trial_hint = bool(_TRIAL_HINT.search(text))
    ex.chart_from_request = request.preferred_visualization is not None
    ex.chart = request.preferred_visualization or detect_chart(text)
    if _HEAT.search(text) or ex.chart is VizType.HEATMAP:
        ex.heat_dims = heatmap_dims(text)
    structured_filters = bool(
        request.trial_phase or request.status or request.start_year is not None or request.end_year is not None
    )
    if not (ex.found or ex.nct_ids or structured_filters or ex.ambiguous):
        ex.unintelligible = is_non_english(text) or not _VOCAB.search(text)
    return ex


# ---------------------------------------------------------------------------------------- plan details
def detect_group_by(text: str, intent: Intent) -> tuple[str | None, str | None]:
    """``(group_by, series_by)`` from the text plus per-intent defaults."""
    t = text.lower()
    dim: str | None = None
    if re.search(r"sponsor (categor|class|type)|industry (vs\.?|versus|or|and) academ|by sponsor type", t):
        dim = "sponsor_class"
    elif re.search(
        r"\b(active|ongoing|open)\b.{0,20}\b(vs\.?|versus|and|or)\b.{0,20}\b(completed|finished|done|"
        r"closed|ended)\b|status groups?|active (vs\.?|versus) (done|finished)",
        t,
    ):
        dim = "status_group"
    elif re.search(r"intervention types?|types? of interventions?", t):
        dim = "intervention_type"
    elif m := re.search(
        r"\bby (recruitment status|status|phase|country|sponsor|year|condition|allocation|masking|"
        r"study type)\b",
        t,
    ):
        dim = {"recruitment status": "status", "study type": "study_type"}.get(m.group(1), m.group(1))
    elif re.search(r"across phases|per phase", t):
        dim = "phase"
    if intent in (Intent.PHASE_STATUS_HEATMAP, Intent.LANDSCAPE_GAP_HEATMAP, Intent.ENTITY_NETWORK):
        return None, None
    if intent is Intent.TREND_OVER_TIME:
        return None, dim if dim in (
            "phase",
            "status",
            "status_group",
            "sponsor_class",
            "intervention_type",
            "country",
        ) else None
    defaults = {
        Intent.SPONSOR_LANDSCAPE: "sponsor",
        Intent.INTERVENTION_LANDSCAPE: "intervention",
        Intent.GEOGRAPHIC_DISTRIBUTION: "country",
        Intent.PHASE_DISTRIBUTION: "phase",
        Intent.STATUS_DISTRIBUTION: "status",
    }
    if intent is Intent.STATUS_DISTRIBUTION and dim == "status_group":
        return "status_group", None
    if intent is Intent.SPONSOR_LANDSCAPE and dim not in ("sponsor_class", "sponsor"):
        dim = None
    if intent is Intent.INTERVENTION_LANDSCAPE and dim not in ("intervention_type", "intervention"):
        dim = None
    return dim or defaults.get(intent), None


def detect_network_kind(text: str) -> str:
    """Which co-occurrence network the question asks for (default sponsor_drug)."""
    t = text.lower()
    if re.search(r"\bsites?\b|investigator", t):
        return "site_investigator"
    if re.search(r"collaborat", t):
        return "sponsor_collaborator"
    if re.search(r"combin|co-?occur|drug[- ]drug|together|paired with|co-?administ", t) and "sponsor" not in t:
        return "drug_drug"
    if re.search(r"sponsor|compan|industry", t):
        return "sponsor_drug"
    if re.search(r"condition|disease|indication", t):
        return "drug_condition"
    return "sponsor_drug"


def chart_intents(viz: VizType) -> list[Intent]:
    """Intents that can draw ``viz`` (from intents.yaml): those with it as primary first, then as a fallback."""
    specs = load_intents()
    primary = [i for i, s in specs.items() if s.primary is viz]
    return primary + [i for i, s in specs.items() if viz in s.fallbacks and i not in primary]


def steer_by_chart(
    intent: Intent,
    viz: VizType | None,
    text: str,
    *,
    trial_ctx: bool,
    cohort_ctx: bool,
    has_sponsor: bool,
    has_drug: bool,
) -> Intent | None:
    """A different intent that can draw the requested chart, or ``None`` to keep ``intent``.

    Candidates come from :func:`chart_intents` and must fit the scope (an NCT id -> single-trial intents;
    otherwise cohort intents, or single-trial ones only for trial-level charts such as a survival curve);
    sponsor_activity needs a sponsor and development_progression a drug. Ties go to the intent whose
    keywords the question also matches, then to intents.yaml order.
    """
    specs = load_intents()
    if viz is None or viz in GENERIC_VIZ or viz in specs[intent].compatible_viz:
        return None

    def ok(i: Intent, scope: Scope) -> bool:
        if scope not in specs[i].scopes or i is Intent.NOT_IN_REGISTRY:
            return False
        if i is Intent.SPONSOR_ACTIVITY and not has_sponsor:
            return False
        return not (i is Intent.DEVELOPMENT_PROGRESSION and not has_drug)

    cands = chart_intents(viz)
    if trial_ctx:
        pool = [i for i in cands if ok(i, Scope.SINGLE_TRIAL)]
    else:
        pool = [i for i in cands if ok(i, Scope.COHORT)]
        if not pool and (viz in _TRIAL_LEVEL_VIZ or not cohort_ctx):
            pool = [i for i in cands if ok(i, Scope.SINGLE_TRIAL)]
    if not pool:
        return None
    scores = dict(score_intents(text, trial_ctx=trial_ctx, cohort_ctx=cohort_ctx, has_sponsor=has_sponsor))
    return min(pool, key=lambda i: (-scores.get(i, 0.0), pool.index(i)))


def landscape_default(ex: Extraction) -> tuple[Intent, str | None]:
    """Richer default view for a broad question about an entity ("lung cancer trials", "overview of X").

    One sponsor (and no drug) -> that sponsor's activity timeline. Otherwise an entity network: a drug on
    its own -> drug-condition (what it is tested for); anything else -> sponsor-drug (who tests what). The
    network shows several dimensions of the landscape at once and stays readable for large cohorts (edge
    thresholds), unlike a phase or status bar that answers a narrower question nobody asked.
    """
    kinds = {f.kind for f in ex.found}
    if "sponsor" in kinds and "drug" not in kinds:
        return Intent.SPONSOR_ACTIVITY, None
    if "drug" in kinds and "condition" not in kinds and "sponsor" not in kinds:
        return Intent.ENTITY_NETWORK, "drug_condition"
    return Intent.ENTITY_NETWORK, "sponsor_drug"


def _place_mentions(text: str, ex: Extraction, resolver: EntityResolver) -> list[str]:
    """Capitalized places after near/in/around that are not countries or known entities ("Boston", "Texas")."""
    out: list[str] = []
    known = {f.entity.input.lower() for f in ex.found} | {f.entity.resolved.lower() for f in ex.found}
    for m in re.finditer(r"\b(?:near|around|in|within)\s+((?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+){0,2})\b", text):
        place = m.group(1)
        if place.lower() in known or resolver.known(place) or place.lower() in _LEAD_STOP | _GENERIC:
            continue
        out.append(place)
    return list(dict.fromkeys(out))


def detect_granularity(text: str) -> str | None:
    t = text.lower()
    if re.search(r"per month|monthly|each month|by month", t):
        return "month"
    if re.search(r"quarter", t):
        return "quarter"
    if re.search(r"per year|each year|by year|annual|yearly|over time|since \d{4}", t):
        return "year"
    return None


def detect_top_n(text: str) -> int | None:
    if m := re.search(r"\btop\s+(\d{1,3})\b", text, re.I):
        return max(1, min(100, int(m.group(1))))
    return None


def detect_persona(request: VisualizeRequest, intent: Intent) -> Persona:
    """Explicit field > first-person wording > patient-only intent > researcher."""
    if request.persona != "auto":
        return Persona(request.persona)
    if _FIRST_PERSON.search(request.query) or request.patient_profile is not None:
        return Persona.PATIENT
    if load_intents()[intent].personas == [Persona.PATIENT]:
        return Persona.PATIENT
    return Persona.RESEARCHER


def display_name(name: str) -> str:
    """Capitalize a lowercase entity name for labels ("semaglutide" -> "Semaglutide")."""
    return name[:1].upper() + name[1:] if name[:1].islower() else name


def build_cohorts(ex: Extraction, intent: Intent) -> tuple[list[CohortFilter], bool]:
    """Cohort filters for the plan. Returns ``(cohorts, is_comparison)``.

    A comparison is formed when the text has a comparison cue (or structured lists were given) and at
    least two distinct entities of one kind exist (drugs, then conditions, then sponsors).
    """
    by_kind: dict[str, list[Found]] = {}
    for f in ex.found:
        by_kind.setdefault(f.kind, []).append(f)
    structured_multi = any(
        len([f for f in fs if f.entity.method == "structured_field"]) >= 2
        for k, fs in by_kind.items()
        if k in ("drug", "condition")
    )
    split_kind = None
    if ex.compare_cue or structured_multi:
        split_kind = next((k for k in ("drug", "condition", "sponsor") if len(by_kind.get(k, [])) >= 2), None)

    phases = [] if intent in PHASE_DIM and not ex.phases_from_request else [p for p in ex.phases if p in PHASES]
    statuses = [] if intent in STATUS_DIM and not ex.statuses_from_request else ex.statuses

    def base(label: str, exclude: str | None) -> CohortFilter:
        def terms(kind: str) -> list[str]:
            if kind == exclude:
                return []
            return list(dict.fromkeys(t for f in by_kind.get(kind, []) for t in f.terms))

        return CohortFilter(
            label=label,
            drugs=terms("drug"),
            conditions=terms("condition"),
            sponsors=terms("sponsor"),
            countries=terms("country"),
            phases=phases,
            statuses=statuses,
            start_year=ex.start_year,
            end_year=ex.end_year,
            interventional_only=intent in (Intent.ENTITY_NETWORK, Intent.DEVELOPMENT_PROGRESSION),
        )

    if split_kind:
        cohorts = []
        for f in by_kind[split_kind][:5]:
            c = base(display_name(f.entity.resolved), split_kind)
            setattr(c, {"drug": "drugs", "condition": "conditions", "sponsor": "sponsors"}[split_kind], f.terms)
            cohorts.append(c)
        return cohorts, True
    names = [
        display_name(f.entity.resolved) for k in ("sponsor", "drug", "condition", "country") for f in by_kind.get(k, [])
    ]
    label = " · ".join(names[:3]) if names else "All trials"
    return [base(label, None)], False


def _focus_terms(request: VisualizeRequest, ex: Extraction, intent: Intent) -> list[str]:
    terms: list[str] = [m.group(1) for m in re.finditer(r'"([^"]{2,60})"', request.query)]
    prof = request.patient_profile
    if prof and intent in (Intent.CONCOMITANT_MEDICATIONS, Intent.ELIGIBILITY_CHECK):
        terms += prof.medications + prof.conditions
    if intent in (
        Intent.CONCOMITANT_MEDICATIONS,
        Intent.RISKS_ADVERSE_EVENTS,
        Intent.BIOMARKERS_PKPD,
        Intent.RESULTS_OUTCOMES,
        Intent.ELIGIBILITY_CHECK,
    ):
        terms += [f.entity.resolved for f in ex.found if f.kind in ("drug", "condition")]
    if intent is Intent.BIOMARKERS_PKPD:
        terms += [m.group(0) for m in re.finditer(r"\b(hba1c|ctdna|cmax|auc|pk|pd|half-?life)\b", request.query, re.I)]
    return list(dict.fromkeys(t.strip() for t in terms if t.strip()))[:10]


def rule_plan_with_extraction(
    request: VisualizeRequest, resolver: EntityResolver | None = None, today: dt.date | None = None
) -> tuple[QueryPlan, Extraction]:
    """Rule plan plus the raw extraction (used by the LLM plan validator)."""
    resolver = resolver or get_resolver()
    ex = extract(request, resolver, today)
    text = request.query
    specs = load_intents()
    trial_ctx = bool(ex.nct_ids)
    cohort_ctx = bool(ex.found)
    has_sponsor = any(f.kind == "sponsor" for f in ex.found)
    has_drug = any(f.kind == "drug" for f in ex.found)
    intent, conf, how = classify_intent(text, trial_ctx=trial_ctx, cohort_ctx=cohort_ctx, has_sponsor=has_sponsor)
    assumptions: list[str] = []
    clarification: str | None = None
    network_kind: str | None = None

    # Broad question about an entity: a richer overview than a low-confidence trend (never for patients, who
    # ask about joining; never when a specific trial is meant).
    patient = (
        request.persona == "patient"
        or request.patient_profile is not None
        or (request.persona == "auto" and bool(_FIRST_PERSON.search(text)))
    )
    if (
        not trial_ctx
        and cohort_ctx
        and not ex.trial_hint
        and not patient
        and not _COUNT_Q.search(text)
        and (how == "default" or (_LANDSCAPE.search(text) and Scope.COHORT not in specs[intent].scopes))
    ):
        intent, network_kind = landscape_default(ex)
        conf, how = 0.65, "landscape"
        view = _LANDSCAPE_VIEW_DRUG if network_kind == "drug_condition" else _LANDSCAPE_VIEW[intent]
        assumptions.append(TEMPLATES["LANDSCAPE"].format(view=view))
    # A named chart ("sankey of ...", "map of ...", preferred_visualization) steers the intent to one that draws it.
    steered = steer_by_chart(
        intent, ex.chart, text, trial_ctx=trial_ctx, cohort_ctx=cohort_ctx, has_sponsor=has_sponsor, has_drug=has_drug
    )
    if steered is not None:
        assert ex.chart is not None
        intent, conf, how = steered, 0.85, f"chart:{ex.chart.value}"
        network_kind = None
        assumptions = [a for a in assumptions if not a.startswith("Broad question")]
    elif ex.chart is not None and ex.chart not in GENERIC_VIZ and ex.chart in specs[intent].compatible_viz:
        conf = max(conf, 0.8)  # the question names a chart this intent draws ("map of X trials")
    ex.rule = how
    spec = specs[intent]

    cohorts: list[CohortFilter] = []
    if ex.unintelligible:
        # Gibberish / another language with nothing recognizable: no cohort, so the orchestrator asks which
        # drug, condition, sponsor or trial is meant (it never silently analyzes "All trials").
        scope = Scope.COHORT if Scope.COHORT in spec.scopes else spec.scopes[0]
        clarification = TEMPLATES["UNCLEAR"]
    elif trial_ctx:
        scope = Scope.COMPARISON if len(ex.nct_ids) > 1 else Scope.SINGLE_TRIAL
        if Scope.SINGLE_TRIAL not in spec.scopes:  # unreachable via classify_intent; keep plan valid
            scope = Scope.SINGLE_TRIAL
    elif Scope.SINGLE_TRIAL in spec.scopes and (Scope.COHORT not in spec.scopes or not cohort_ctx or ex.trial_hint):
        scope = Scope.SINGLE_TRIAL
        clarification = TEMPLATES["TRIAL_REQUIRED"]
    else:
        cohorts, is_cmp = build_cohorts(ex, intent)
        scope = Scope.COMPARISON if is_cmp else Scope.COHORT
        if is_cmp and Scope.COMPARISON not in spec.scopes:
            merged = cohorts[0].model_copy()
            for c in cohorts[1:]:
                merged.drugs += c.drugs
                merged.conditions += c.conditions
                merged.sponsors += c.sponsors
            merged.label = " + ".join(c.label for c in cohorts)
            cohorts, scope = [merged], Scope.COHORT
            assumptions.append(TEMPLATES["MERGED"].format(intent=intent.value))
        if not ex.found:
            assumptions.append(TEMPLATES["ALL_TRIALS"])
    group_by, series_by = detect_group_by(text, intent)
    if intent is Intent.LANDSCAPE_GAP_HEATMAP:
        group_by, series_by = heatmap_dims(text)
    if intent is Intent.GEOGRAPHIC_DISTRIBUTION and not ex.unintelligible:
        ex.places = _place_mentions(text, ex, resolver)[:2]
        assumptions += [TEMPLATES["PLACE_NOT_FILTERED"].format(place=p) for p in ex.places]
    if ex.ambiguous:
        amb = "; ".join(f"'{k}' could mean {', '.join(v)}" for k, v in ex.ambiguous.items())
        clarification = (clarification + " " if clarification else "") + TEMPLATES["AMBIGUOUS_MEANINGS"].format(
            details=amb
        )
    if conf < 0.6 and not clarification:
        clarification = TEMPLATES["LOW_CONFIDENCE"].format(description=spec.description)
    if intent is Intent.ENTITY_NETWORK:
        network_kind = network_kind or detect_network_kind(text)
    plan = QueryPlan(
        intent=intent,
        intent_confidence=conf,
        persona=detect_persona(request, intent),
        scope=scope,
        nct_ids=ex.nct_ids,
        cohorts=cohorts,
        group_by=request.group_by or group_by,  # type: ignore[arg-type]
        series_by=request.series_by or series_by,  # type: ignore[arg-type]
        network_kind=network_kind,  # type: ignore[arg-type]
        time_granularity=request.time_granularity
        or detect_granularity(text)  # type: ignore[arg-type]
        or ("year" if intent is Intent.TREND_OVER_TIME else None),
        top_n=request.top_n or detect_top_n(text),
        preferred_visualization=preferred_chart(ex, intent),
        focus_terms=_focus_terms(request, ex, intent),
        assumptions=assumptions,
        clarification=clarification,
        answerability="subjective_judgment" if is_subjective(text) else "answerable",
        parser="rules",
    )
    return plan, ex


def preferred_chart(ex: Extraction, intent: Intent) -> VizType | None:
    """The request's ``preferred_visualization``; else a chart word from the text when the intent can draw it
    (or when it names a specific chart the intent cannot draw, so the response says it was not possible)."""
    if ex.chart_from_request or ex.chart is None:
        return ex.chart
    if ex.chart in load_intents()[intent].compatible_viz:
        return ex.chart
    return None if ex.chart in GENERIC_VIZ | {VizType.TIMELINE, VizType.TIME_SERIES} else ex.chart


def rule_plan(request: VisualizeRequest) -> QueryPlan:
    """Plan a request with rules only (pure, sync, no I/O beyond the static alias/intent tables)."""
    return rule_plan_with_extraction(request)[0]


def entities_of(ex: Extraction) -> list[ResolvedEntity]:
    """Resolved entities for ``meta.interpretation.resolved_entities``."""
    out = [f.entity for f in ex.found]
    out += [ResolvedEntity(kind="nct_id", input=n, resolved=n, method="exact", search_terms=[n]) for n in ex.nct_ids]
    return out
