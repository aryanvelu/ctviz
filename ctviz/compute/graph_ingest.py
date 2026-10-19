"""Pure extraction of knowledge-graph facts from one ClinicalTrials.gov study record.

Nothing here touches the database. :func:`extract` turns a :class:`~ctviz.core.provenance.Study` into a
:class:`TrialFacts` bundle that :class:`ctviz.store.graph.GraphStore` persists:

* **mentions** -- ``(entity_type, canonical, canonical_norm, rel, field_path, excerpt)``: one trial -> entity
  edge with evidence. ``excerpt`` is always the verbatim JSON leaf at ``field_path`` (capped at 300 chars),
  so ``verify_citation`` on a field-value citation built from it succeeds.
* **pairs** -- per-trial co-occurrences (``sponsor_drug``, ``drug_drug``, ``drug_condition``,
  ``site_investigator``, ``sponsor_collaborator``) between two entity keys.
* **aliases** -- ``alias_norm -> canonical`` rows harvested from ``interventions[].otherNames`` and MeSH terms.

Entity keys are ``(entity_type, canonical_norm)``, matching ``kg_nodes UNIQUE(type, canonical_norm)``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from itertools import combinations

from ctviz.api.schemas.citations import MAX_EXCERPT_CHARS
from ctviz.core.provenance import Study

DRUG_TYPES = frozenset({"DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT", "GENETIC"})
# armGroups[].interventionNames entries look like "Drug: Nivolumab" (type label + ": " + intervention name).
_ARM_DRUG_PREFIXES = ("Drug: ", "Biological: ", "Combination Product: ", "Genetic: ")
_ARM_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z /-]{1,40}:\s+")

PAIR_TYPES = ("sponsor_drug", "drug_drug", "drug_condition", "site_investigator", "sponsor_collaborator")
NODE_TYPES = ("trial", "drug", "condition", "sponsor", "collaborator", "site", "investigator", "country")
PAIR_NODE_TYPES: dict[str, tuple[str, str]] = {
    "sponsor_drug": ("sponsor", "drug"),
    "drug_drug": ("drug", "drug"),
    "drug_condition": ("drug", "condition"),
    "site_investigator": ("site", "investigator"),
    "sponsor_collaborator": ("sponsor", "collaborator"),
}

_TRADEMARKS = re.compile(r"[®™©℠]")
_UNIT = (
    r"(?:mg/kg|mg/m2|mg/m²|mg/ml|mg/day|mcg/kg|µg/kg|ug/kg|iu/kg|mg|mcg|µg|ug|ng|g|kg|ml|l|iu|units?|u|%"
    r"|mmol|meq)"
)
_DOSE = re.compile(rf"(?<![\w.])\d+(?:[.,]\d+)?(?:\s*(?:-|to|/)\s*\d+(?:[.,]\d+)?)?\s*{_UNIT}(?![\w])", re.I)
_EMPTY_PARENS = re.compile(r"\(\s*[,;/-]*\s*\)|\[\s*\]")
_WS = re.compile(r"\s+")
_EDGE_PUNCT = " \t\n,;:/-–—."
_SKIP_DRUG = re.compile(
    r"placebo|\bsham\b|standard (?:of )?care|\busual care\b|best supportive care|\bsoc\b|no intervention"
    r"|\bvehicle\b|saline control",
    re.I,
)
# Families each pair type is derived from: a trial's pairs of that type are replaced only if all are observed.
PAIR_DEPENDS: dict[str, frozenset[str]] = {
    "sponsor_drug": frozenset({"sponsor", "drug"}),
    "drug_drug": frozenset({"arm_drug"}),
    "drug_condition": frozenset({"drug", "condition"}),
    "site_investigator": frozenset({"site", "investigator", "site_investigator"}),
    "sponsor_collaborator": frozenset({"sponsor", "collaborator"}),
}

# Generic words that do not identify an institution when matching an affiliation to a facility.
_GENERIC_ORG_TOKENS = frozenset(
    {
        "a", "an", "and", "at", "de", "del", "der", "des", "di", "du", "for", "in", "la", "le", "of", "the", "y",
        "university", "universidad", "universite", "universität", "universitat", "hospital", "hospitals",
        "center", "centre", "medical", "medicine", "institute", "institut", "clinic", "clinics", "clinical",
        "college", "school", "health", "healthcare", "research", "foundation", "inc", "llc", "ltd", "gmbh", "co",
        "corp", "corporation", "company", "sa", "ag", "group", "department", "dept", "system", "services",
        "site", "sites", "investigational", "local", "institution", "national", "general", "cancer",
    }
)  # fmt: skip
_ORG_TOKEN = re.compile(r"[^\W_]+")


# -- normalization -------------------------------------------------------------------------------
def norm_text(value: str) -> str:
    """Generic key normalization: NFKC, casefold, trademark symbols removed, whitespace collapsed."""
    value = unicodedata.normalize("NFKC", _TRADEMARKS.sub("", value))
    return _WS.sub(" ", value).strip().casefold()


def clean_drug_name(name: str) -> str:
    """Display form of a drug name: trademark symbols and dosage tokens ("2.4 mg") removed."""
    value = unicodedata.normalize("NFKC", _TRADEMARKS.sub("", name))
    value = _DOSE.sub(" ", value)
    value = _EMPTY_PARENS.sub(" ", value)
    value = _WS.sub(" ", value).strip(_EDGE_PUNCT)
    return value


def normalize_drug(name: str) -> str | None:
    """Canonical key for a drug name, or ``None`` for placebo / standard-of-care / empty names."""
    if not name or _SKIP_DRUG.search(name):
        return None
    cleaned = clean_drug_name(name)
    if not cleaned or not re.search(r"[^\W\d_]", cleaned):
        return None
    return norm_text(cleaned)


def strip_arm_prefix(entry: str) -> tuple[str | None, str]:
    """Split ``"Drug: Nivolumab"`` into ``("Drug", "Nivolumab")``; ``(None, entry)`` if no prefix."""
    m = _ARM_PREFIX_RE.match(entry)
    if not m:
        return None, entry
    return entry[: m.end()].split(":", 1)[0].strip(), entry[m.end() :]


def _cap(excerpt: str) -> str:
    return excerpt[:MAX_EXCERPT_CHARS]


# -- data model ----------------------------------------------------------------------------------
EntityKey = tuple[str, str]  # (entity_type, canonical_norm)


@dataclass(frozen=True, slots=True)
class Mention:
    """One trial -> entity edge with verbatim evidence."""

    entity_type: str
    canonical: str
    canonical_norm: str
    rel: str
    field_path: str
    excerpt: str
    meta: dict | None = None

    @property
    def key(self) -> EntityKey:
        return (self.entity_type, self.canonical_norm)


@dataclass(frozen=True, slots=True)
class Pair:
    """An undirected co-occurrence of two entities within one trial (``a <= b`` by key)."""

    pair_type: str
    a: EntityKey
    b: EntityKey


@dataclass(frozen=True, slots=True)
class Alias:
    alias_norm: str
    kind: str
    canonical: str
    source: str


@dataclass
class TrialFacts:
    """Everything the graph learns from one trial."""

    nct_id: str
    title: str | None
    mentions: list[Mention] = field(default_factory=list)
    pairs: list[Pair] = field(default_factory=list)
    aliases: list[Alias] = field(default_factory=list)
    # Display names for entities that only appear as pair endpoints / arm entries.
    labels: dict[EntityKey, str] = field(default_factory=dict)
    # Mention families (``rel`` values) whose source fields are present in this record. On re-ingest exactly
    # these families are replaced, so a sparse projection cannot erase facts it never saw.
    observed: frozenset[str] = frozenset()


def _pair(pair_type: str, x: EntityKey, y: EntityKey) -> Pair | None:
    if x == y:
        return None
    if pair_type == "drug_drug" and y < x:
        x, y = y, x
    return Pair(pair_type, x, y)


# -- extraction ----------------------------------------------------------------------------------
def extract_mentions(study: Study) -> list[Mention]:
    """All trial -> entity mentions, deduplicated per ``(type, canonical_norm, rel, field_path)``."""
    out: list[Mention] = []
    seen: set[tuple[str, str, str, str]] = set()
    once_per_trial = {"country"}  # a 300-site trial would otherwise yield 300 identical country edges
    seen_once: set[EntityKey] = set()

    def add(m: Mention) -> None:
        k = (m.entity_type, m.canonical_norm, m.rel, m.field_path)
        if k in seen:
            return
        if m.entity_type in once_per_trial:
            if m.key in seen_once:
                return
            seen_once.add(m.key)
        seen.add(k)
        out.append(m)

    for _, iv, path in study.items("interventions"):
        if not isinstance(iv, dict) or iv.get("type") not in DRUG_TYPES:
            continue
        name = iv.get("name")
        if not isinstance(name, str):
            continue
        key = normalize_drug(name)
        if key:
            add(
                Mention(
                    "drug",
                    clean_drug_name(name),
                    key,
                    "drug",
                    f"{path}.name",
                    _cap(name),
                    {"intervention_type": iv.get("type")},
                )
            )

    for _, cond, path in study.items("conditions"):
        if isinstance(cond, str) and cond.strip():
            add(Mention("condition", _WS.sub(" ", cond).strip(), norm_text(cond), "condition", path, _cap(cond)))

    sponsor_path = "protocolSection.sponsorCollaboratorsModule.leadSponsor.name"
    sponsor = study.get(sponsor_path)
    if isinstance(sponsor, str) and sponsor.strip():
        add(
            Mention(
                "sponsor",
                sponsor.strip(),
                norm_text(sponsor),
                "sponsor",
                sponsor_path,
                _cap(sponsor),
                {"class": study.get("lead_sponsor_class")},
            )
        )

    for _, col, path in study.items("collaborators"):
        name = col.get("name") if isinstance(col, dict) else None
        if isinstance(name, str) and name.strip():
            add(
                Mention(
                    "collaborator",
                    name.strip(),
                    norm_text(name),
                    "collaborator",
                    f"{path}.name",
                    _cap(name),
                    {"class": col.get("class")},
                )
            )

    for _, off, path in study.items("overall_officials"):
        name = off.get("name") if isinstance(off, dict) else None
        if isinstance(name, str) and name.strip():
            add(
                Mention(
                    "investigator",
                    name.strip(),
                    norm_text(name),
                    "investigator",
                    f"{path}.name",
                    _cap(name),
                    {"role": off.get("role"), "affiliation": off.get("affiliation")},
                )
            )

    for _, loc, path in study.items("locations"):
        if not isinstance(loc, dict):
            continue
        facility, city, country = loc.get("facility"), loc.get("city"), loc.get("country")
        if isinstance(facility, str) and facility.strip():
            label = ", ".join(x.strip() for x in (facility, city, country) if isinstance(x, str) and x.strip())
            add(
                Mention(
                    "site",
                    label,
                    norm_text(label),
                    "site",
                    f"{path}.facility",
                    _cap(facility),
                    {"city": city, "country": country},
                )
            )
        if isinstance(country, str) and country.strip():
            add(Mention("country", country.strip(), norm_text(country), "country", f"{path}.country", _cap(country)))
        for _, contact, cpath in _iter(loc.get("contacts"), f"{path}.contacts"):
            role = contact.get("role") if isinstance(contact, dict) else None
            name = contact.get("name") if isinstance(contact, dict) else None
            if isinstance(name, str) and name.strip() and isinstance(role, str) and "INVESTIGATOR" in role:
                add(
                    Mention(
                        "investigator",
                        name.strip(),
                        norm_text(name),
                        "site_investigator",
                        f"{cpath}.name",
                        _cap(name),
                        {"role": role},
                    )
                )
    return out


def _iter(items: object, path: str):
    if isinstance(items, list):
        for i, item in enumerate(items):
            yield i, item, f"{path}[{i}]"


def _arm_drug_mentions(study: Study, drug_lookup: dict[str, str]) -> tuple[list[Mention], list[list[EntityKey]]]:
    """Mentions of drugs inside arms (evidence for combinations) and the drug keys per arm."""
    mentions: list[Mention] = []
    arms: list[list[EntityKey]] = []
    for _, arm, apath in study.items("arm_groups"):
        if not isinstance(arm, dict):
            continue
        keys: list[EntityKey] = []
        for _, entry, epath in _iter(arm.get("interventionNames"), f"{apath}.interventionNames"):
            if not isinstance(entry, str) or not entry.startswith(_ARM_DRUG_PREFIXES):
                continue
            _, bare = strip_arm_prefix(entry)
            key = drug_lookup.get(bare) or drug_lookup.get(bare.strip()) or normalize_drug(bare)
            if not key or ("drug", key) in keys:
                continue
            keys.append(("drug", key))
            mentions.append(Mention("drug", clean_drug_name(bare), key, "arm_drug", epath, _cap(entry)))
        arms.append(keys)
    return mentions, arms


def _site_investigator_pairs(study: Study, mentions: list[Mention]) -> list[Pair]:
    """Site-level investigator contacts pair with their own site. An overall official pairs with a site only
    when :func:`affiliation_matches` holds, or the trial has exactly one site and the official is its
    PRINCIPAL_INVESTIGATOR. Never an N x M cross product."""
    pairs: list[Pair] = []
    sites = [m for m in mentions if m.entity_type == "site"]
    if not sites:
        return pairs
    site_by_loc = {m.field_path.rsplit(".", 1)[0]: m for m in sites}
    for m in mentions:
        if m.rel == "site_investigator":
            loc_path = m.field_path.split(".contacts[", 1)[0]
            site = site_by_loc.get(loc_path)
            if site:
                pairs.append(Pair("site_investigator", site.key, m.key))
        elif m.rel == "investigator":
            meta = m.meta or {}
            single_pi = len(sites) == 1 and meta.get("role") == "PRINCIPAL_INVESTIGATOR"
            for site in sites:
                if single_pi or affiliation_matches(meta.get("affiliation") or "", site.excerpt):
                    pairs.append(Pair("site_investigator", site.key, m.key))
    return pairs


def _org_tokens(name: str) -> set[str]:
    return {t for t in _ORG_TOKEN.findall(norm_text(name)) if len(t) > 1 and t not in _GENERIC_ORG_TOKENS}


def affiliation_matches(affiliation: str, facility: str) -> bool:
    """True if an affiliation and a facility name the same institution: equal normalized names, or at least two
    shared meaningful whole tokens (generic words such as "University" or "Hospital" do not count)."""
    a, f = norm_text(affiliation), norm_text(facility)
    if not a or not f:
        return False
    return a == f or len(_org_tokens(a) & _org_tokens(f)) >= 2


def _aliases(study: Study, drug_mentions: list[Mention], conditions: list[Mention]) -> list[Alias]:
    out: list[Alias] = []
    for _, iv, _path in study.items("interventions"):
        if not isinstance(iv, dict) or iv.get("type") not in DRUG_TYPES or not isinstance(iv.get("name"), str):
            continue
        key = normalize_drug(iv["name"])
        if not key:
            continue
        canonical = clean_drug_name(iv["name"])
        for other in iv.get("otherNames") or []:
            if isinstance(other, str):
                alias = normalize_drug(other)
                if alias and alias != key:
                    out.append(Alias(alias, "drug", canonical, "otherNames"))

    def mesh_terms(path: str) -> list[str]:
        meshes = study.get(path) or []
        return [m["term"] for m in meshes if isinstance(m, dict) and isinstance(m.get("term"), str)]

    def link(kind: str, ents: list[Mention], terms: list[str]) -> None:
        uniq = {m.canonical_norm: m for m in ents}
        for term in terms:
            tnorm = norm_text(term)
            tw = set(re.findall(r"\w+", tnorm))
            for enorm in uniq:
                if enorm == tnorm:
                    continue
                ew = set(re.findall(r"\w+", enorm))
                single = len(uniq) == 1 and len(terms) == 1
                if single or (tw and ew and (tw <= ew or ew <= tw)):
                    out.append(Alias(enorm, kind, term, "mesh"))

    link("condition", conditions, mesh_terms("condition_meshes"))
    link("drug", drug_mentions, mesh_terms("intervention_meshes"))
    return out


def extract(study: Study) -> TrialFacts:
    """Extract mentions, co-occurrence pairs and aliases for one trial."""
    mentions = extract_mentions(study)
    drug_lookup: dict[str, str] = {}
    for _, iv, _ in study.items("interventions"):
        if isinstance(iv, dict) and iv.get("type") in DRUG_TYPES and isinstance(iv.get("name"), str):
            key = normalize_drug(iv["name"])
            if key:
                drug_lookup[iv["name"]] = key
    arm_mentions, arms = _arm_drug_mentions(study, drug_lookup)

    by_type: dict[str, list[EntityKey]] = {}
    for m in mentions:
        if m.rel != "site_investigator":
            lst = by_type.setdefault(m.entity_type, [])
            if m.key not in lst:
                lst.append(m.key)
    drugs = by_type.get("drug", [])
    pairs: list[Pair | None] = []
    for s in by_type.get("sponsor", []):
        pairs += [_pair("sponsor_drug", s, d) for d in drugs]
        pairs += [_pair("sponsor_collaborator", s, c) for c in by_type.get("collaborator", [])]
    for d in drugs:
        pairs += [_pair("drug_condition", d, c) for c in by_type.get("condition", [])]
    for keys in arms:
        if len(keys) >= 2:
            pairs += [_pair("drug_drug", x, y) for x, y in combinations(keys, 2)]
    pairs += _site_investigator_pairs(study, mentions)

    unique: dict[Pair, None] = {}
    for p in pairs:
        if p is not None:
            unique.setdefault(p)

    title = study.get("brief_title")
    facts = TrialFacts(
        nct_id=study.nct_id,
        title=title if isinstance(title, str) else None,
        mentions=mentions + arm_mentions,
        pairs=list(unique),
        aliases=_aliases(
            study,
            [m for m in mentions if m.entity_type == "drug"],
            [m for m in mentions if m.entity_type == "condition"],
        ),
    )
    for m in facts.mentions:
        facts.labels.setdefault(m.key, m.canonical)
    facts.observed = observed_families(study)
    return facts


def observed_families(study: Study) -> frozenset[str]:
    """Mention families whose source fields exist in this (possibly projected) record."""
    ps = study.raw.get("protocolSection") or {}
    arms_mod = ps.get("armsInterventionsModule")
    spons = ps.get("sponsorCollaboratorsModule")
    contacts = ps.get("contactsLocationsModule")
    conds = ps.get("conditionsModule")
    locs = [x for x in (study.get("locations") or []) if isinstance(x, dict)]
    out: set[str] = set()
    if isinstance(arms_mod, dict):
        if "interventions" in arms_mod:
            out.add("drug")
        if "armGroups" in arms_mod:
            out.add("arm_drug")
    if isinstance(conds, dict) and "conditions" in conds:
        out.add("condition")
    if isinstance(spons, dict) and "leadSponsor" in spons:
        out |= {"sponsor", "collaborator"}  # once the module is present, no collaborators list means none
    if isinstance(contacts, dict) and "overallOfficials" in contacts:
        out.add("investigator")
    if any("facility" in x for x in locs):
        out.add("site")
    if any("country" in x for x in locs):
        out.add("country")
    if any("contacts" in x for x in locs):
        out.add("site_investigator")
    return frozenset(out)
