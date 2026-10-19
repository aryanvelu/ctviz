"""Entity resolution: map free-text drug / condition / sponsor / country mentions to search terms.

Backed by the seed alias table in ``data/aliases.json`` (brand -> generic, drug-class expansion,
common abbreviations) plus terms learned at runtime from MeSH browse modules of fetched studies.
Everything here is deterministic; no LLM is involved.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from ctviz.api.schemas.response import ResolvedEntity

ALIASES_FILE = Path(__file__).resolve().parents[2] / "data" / "aliases.json"

EntityKind = Literal["drug", "condition", "sponsor", "country", "nct_id"]

_DRUG_SUFFIX = re.compile(
    r"(mab|nib|tide|glutide|tinib|ciclib|parib|lisib|stat|pril|sartan|olol|vir|navir|gliflozin|gliptin|"
    r"azole|cycline|mycin|cillin|platin|taxel|rubicin|zumab|ximab|umab|limus|tecan|dronate|setron|"
    r"azepam|afil|tidine|prazole|semide|thiazide|formin|gliptin|lukast|triptan|grel|xaban|parin)$"
)
_DRUG_CODE = re.compile(r"^[A-Za-z]{1,5}-?\d{3,7}[A-Za-z]?$")  # e.g. LY3298176, MK-3475, BNT162b2-like codes
_SPONSOR_HINT = re.compile(
    r"\b(inc|corp|corporation|ltd|llc|gmbh|ag|s\.?a\.?|plc|pharma|pharmaceuticals?|therapeutics|biotech|"
    r"biosciences|biopharma|university|universit[eé]|hospital|institute|foundation|medical center|clinic|"
    r"laboratories|labs|group|company|health|kgaa|holdings|a/s)(?![\w/])",
    re.I,
)
_CONDITION_HINT = re.compile(
    r"(cancer|carcinoma|tumou?r|neoplasm|disease|syndrome|disorder|infection|failure|itis|oma|emia|aemia|osis|"
    r"pathy|diabetes|injury|pain|deficiency|insufficiency|obesity|sclerosis|dystrophy|atrophy|palsy|leukemia|"
    r"lymphoma|myeloma|hypertension|asthma|apnea|insomnia|depression|anxiety|addiction|dependence|fibrosis)\b",
    re.I,
)


def norm(text: str) -> str:
    """Lowercase, straighten quotes, collapse whitespace."""
    t = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", t).strip().lower()


_norm = norm


def _requires_upper(alias: str) -> bool:
    """Short all-letter aliases (US, ALS, UC) only match when written in upper case."""
    letters = alias.replace(".", "")
    return len(letters) <= 4 and letters.isalpha()


def _upper_ok(surface: str) -> bool:
    """Upper-case surface form, allowing a lower-case plural 's' (``ICIs``, ``TKIs``)."""
    if surface == surface.upper():
        return True
    return len(surface) > 2 and surface[-1] == "s" and surface[:-1] == surface[:-1].upper()


def _safe_correction(key: str, hit: str) -> bool:
    """A spelling correction may only fix letters inside words of the same shape.

    Refused when the token counts differ, or when a differing token contains a digit or is a single
    character (``hepatitis a`` != ``hepatitis c``, ``type 3 diabetes`` != ``type 1 diabetes``).
    """
    a, b = key.split(), hit.split()
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if x != y and (len(x) <= 1 or len(y) <= 1 or any(c.isdigit() for c in x + y)):
            return False
    return True


@dataclass(frozen=True)
class Mention:
    """An entity found in free text by :meth:`EntityResolver.scan`."""

    kind: EntityKind
    surface: str  # text as written
    start: int
    end: int
    entity: ResolvedEntity
    is_class: bool = False


class EntityResolver:
    """Resolve entity mentions using the alias table, drug-class expansion and learned MeSH terms."""

    def __init__(self, table: dict[str, Any] | None = None):
        table = table if table is not None else json.loads(ALIASES_FILE.read_text())
        self.classes: dict[str, list[str]] = {}
        self.class_alias: dict[str, str] = {}
        for name, spec in table.get("drug_classes", {}).items():
            self.classes[name] = list(spec["members"])
            self.class_alias[_norm(name)] = name
            for a in spec.get("aliases", []):
                self.class_alias[_norm(a)] = name
        self.alias: dict[str, dict[str, str]] = {
            "drug": {_norm(k): v for k, v in table.get("drugs", {}).items()},
            "condition": {_norm(k): v for k, v in table.get("conditions", {}).items()},
            "sponsor": {_norm(k): v for k, v in table.get("sponsors", {}).items()},
            "country": {_norm(k): v for k, v in table.get("countries", {}).items()},
        }
        canon: dict[str, set[str]] = {
            "drug": set(table.get("generics", [])) | set(self.alias["drug"].values()),
            "condition": set(table.get("known_conditions", [])) | set(self.alias["condition"].values()),
            "sponsor": set(self.alias["sponsor"].values()),
            "country": set(self.alias["country"].values()),
        }
        self.canonical: dict[str, dict[str, str]] = {k: {_norm(v): v for v in vs} for k, vs in canon.items()}
        self.ambiguous: dict[str, list[str]] = {_norm(k): v for k, v in table.get("ambiguous", {}).items()}
        self.learned: dict[str, dict[str, str]] = {"drug": {}, "condition": {}}
        self._scan_re: re.Pattern[str] | None = None
        self._surface: dict[str, tuple[EntityKind, bool]] = {}

    # ------------------------------------------------------------------ lookups
    def is_drug_class(self, name: str) -> bool:
        """True when ``name`` names a drug class in the alias table (e.g. "GLP-1")."""
        return _norm(name) in self.class_alias

    def expand_drug_class(self, name: str) -> list[str]:
        """Members of a drug class (``"GLP-1"`` -> semaglutide, liraglutide, ...); ``[]`` if not a class."""
        cls = self.class_alias.get(_norm(name))
        return list(self.classes[cls]) if cls else []

    def is_ambiguous(self, text: str) -> list[str]:
        """Candidate meanings when ``text`` is a known ambiguous abbreviation, else ``[]``."""
        return list(self.ambiguous.get(_norm(text), []))

    def aliases_of(self, kind: EntityKind, canonical: str) -> list[str]:
        """Surface forms in the table that map to ``canonical`` (brand names, abbreviations)."""
        table = self.alias.get(kind, {})
        return sorted({k for k, v in table.items() if v.lower() == canonical.lower()})

    def resolve(self, text: str, kind: EntityKind, *, method: str | None = None) -> ResolvedEntity:
        """Resolve one mention of a known ``kind``.

        Order: drug class -> exact canonical -> alias -> learned MeSH term -> close spelling match ->
        passthrough. ``method`` overrides the reported method (e.g. ``"structured_field"``); structured
        fields are never spelling-corrected (the user typed them deliberately).
        """
        raw = text.strip()
        key = _norm(raw)
        if kind == "nct_id":
            return ResolvedEntity(
                kind="nct_id", input=raw, resolved=raw.upper(), method=method or "exact", search_terms=[raw.upper()]
            )
        if kind == "drug" and key in self.class_alias:
            cls = self.class_alias[key]
            return ResolvedEntity(
                kind="drug", input=raw, resolved=cls, method=method or "alias", search_terms=list(self.classes[cls])
            )
        canon = self.canonical.get(kind, {})
        alias = self.alias.get(kind, {})
        if key in canon:
            resolved, how = canon[key], "exact"
        elif key in alias:
            resolved, how = alias[key], "alias"
        elif key in self.learned.get(kind, {}):
            resolved, how = self.learned[kind][key], "mesh"
        else:
            close: list[str] = []
            if len(key) >= 6 and method != "structured_field":
                close = [
                    c
                    for c in difflib.get_close_matches(key, list(canon) + list(alias), n=3, cutoff=0.88)
                    if _safe_correction(key, c)
                ][:1]
            if close:
                hit = close[0]
                resolved, how = (canon[hit] if hit in canon else alias[hit]), "spelling_correction"
            else:
                resolved, how = raw, "passthrough"
        terms = [resolved]
        if kind == "drug":
            terms += [a for a in self.aliases_of("drug", resolved) if not _DRUG_CODE.match(a)][:3]
        return ResolvedEntity(
            kind=kind, input=raw, resolved=resolved, method=method or how, search_terms=list(dict.fromkeys(terms))
        )

    def known(self, text: str) -> EntityKind | None:
        """Kind of ``text`` if it is in the alias table / vocabulary (classes count as drugs)."""
        key = _norm(text)
        if key in self.class_alias:
            return "drug"
        for kind in ("drug", "condition", "sponsor", "country"):
            if key in self.canonical[kind] or key in self.alias[kind] or key in self.learned.get(kind, {}):
                return kind  # type: ignore[return-value]
        return None

    def close_matches(self, text: str, kind: EntityKind, n: int = 2) -> list[str]:
        """Vocabulary names spelled like ``text`` (for "did you mean" suggestions; never applied silently)."""
        import difflib

        pool = {**self.canonical.get(kind, {}), **self.alias.get(kind, {}), **self.learned.get(kind, {})}
        keys = difflib.get_close_matches(_norm(text), list(pool), n=n * 2, cutoff=0.8)
        return list(dict.fromkeys(pool[k] for k in keys))[:n]

    def guess_kind(self, text: str) -> tuple[EntityKind, bool]:
        """Best-effort kind for an unknown phrase. Returns ``(kind, confident)``."""
        k = self.known(text)
        if k:
            return k, True
        t = text.strip()
        words = t.split()
        if _SPONSOR_HINT.search(t):
            return "sponsor", True
        if _CONDITION_HINT.search(t):
            return "condition", True
        if len(words) == 1 and (_DRUG_CODE.match(t) or _DRUG_SUFFIX.search(t.lower())):
            return "drug", True
        return "condition", False

    # ------------------------------------------------------------------ text scanning
    def _build_scan(self) -> None:
        surface: dict[str, tuple[EntityKind, bool]] = {}
        for kind in ("country", "sponsor", "condition", "drug"):
            for s in list(self.canonical[kind]) + list(self.alias[kind]) + list(self.learned.get(kind, {})):
                surface.setdefault(s, (kind, False))  # type: ignore[arg-type]
        for s in self.class_alias:
            surface[s] = ("drug", True)
        for amb in self.ambiguous:
            surface.pop(amb, None)
        self._surface = surface
        alts = sorted(surface, key=len, reverse=True)
        self._scan_re = re.compile(r"(?<![\w-])(" + "|".join(re.escape(a) for a in alts) + r")(?![\w-])")

    def scan(self, text: str) -> list[Mention]:
        """Find every known entity mention in ``text`` (longest match wins, no overlaps)."""
        if self._scan_re is None:
            self._build_scan()
        assert self._scan_re is not None
        # same-length normalisation that keeps offsets aligned with ``text``
        low = text.replace("’", "'").replace("‘", "'").lower()
        out: list[Mention] = []
        for m in self._scan_re.finditer(low):
            key = m.group(1)
            surface = text[m.start() : m.end()]
            if _requires_upper(key) and not _upper_ok(surface):
                continue
            kind, is_class = self._surface[key]
            out.append(
                Mention(
                    kind=kind,
                    surface=surface,
                    start=m.start(),
                    end=m.end(),
                    entity=self.resolve(surface, kind),
                    is_class=is_class,
                )
            )
        return out

    # ------------------------------------------------------------------ growth
    def learn_from_study(self, raw: dict[str, Any]) -> int:
        """Grow the table from a study's MeSH browse modules (condition / intervention synonyms).

        Each listed condition or intervention name that is not yet known becomes an alias of the
        study's first MeSH term of that kind. Returns the number of new aliases.
        """
        derived = raw.get("derivedSection", {}) or {}
        proto = raw.get("protocolSection", {}) or {}
        added = 0
        pairs = [
            (
                "condition",
                proto.get("conditionsModule", {}).get("conditions", []) or [],
                derived.get("conditionBrowseModule", {}).get("meshes", []) or [],
            ),
            (
                "drug",
                [i.get("name", "") for i in proto.get("armsInterventionsModule", {}).get("interventions", []) or []],
                derived.get("interventionBrowseModule", {}).get("meshes", []) or [],
            ),
        ]
        for kind, names, meshes in pairs:
            if len(meshes) != 1 or len(names) != 1:
                continue  # only learn unambiguous 1:1 pairs
            term = meshes[0].get("term")
            name = _norm(names[0])
            if not term or not name or self.known(name) or name in self.ambiguous or len(name) > 80:
                continue
            self.learned[kind][name] = term
            added += 1
        if added:
            self._scan_re = None
        return added


@lru_cache(maxsize=1)
def get_resolver() -> EntityResolver:
    """Process-wide resolver over the seed table."""
    return EntityResolver()
