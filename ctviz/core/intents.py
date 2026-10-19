"""Intent registry: loads ``data/intents.yaml`` and holds the handler for each intent."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ctviz.api.schemas.plan import Intent, Persona, Scope
from ctviz.api.schemas.viz import VizType
from ctviz.ctgov.fields import FieldSet

INTENTS_FILE = Path(__file__).resolve().parents[2] / "data" / "intents.yaml"


class IntentNeeds(BaseModel):
    full_records: bool = False
    results: bool = False
    extraction: bool = False
    analysis: bool = False
    graph: bool = False


class IntentSpec(BaseModel):
    intent: Intent
    description: str
    personas: list[Persona]
    scopes: list[Scope]
    field_set: FieldSet
    primary: VizType
    fallbacks: list[VizType] = Field(default_factory=list)
    needs: IntentNeeds = Field(default_factory=IntentNeeds)
    keywords: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)

    @property
    def compatible_viz(self) -> list[VizType]:
        return [self.primary, *self.fallbacks]


@lru_cache(maxsize=1)
def load_intents() -> dict[Intent, IntentSpec]:
    raw = yaml.safe_load(INTENTS_FILE.read_text())
    specs = {Intent(k): IntentSpec(intent=Intent(k), **v) for k, v in raw.items()}
    missing = set(Intent) - set(specs)
    if missing:
        raise RuntimeError(f"intents.yaml is missing: {sorted(m.value for m in missing)}")
    return specs


def get_intent_spec(intent: Intent) -> IntentSpec:
    return load_intents()[intent]
