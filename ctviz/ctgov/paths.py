"""Dot-path access into ClinicalTrials.gov study JSON.

Paths look like ``protocolSection.armsInterventionsModule.interventions[0].name``. They are the
``field_path`` of citations, so the same function is used to build and to verify excerpts.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")
_MISSING = object()


def parse_path(path: str) -> list[str | int]:
    """Split a dot path into keys (str) and list indices (int)."""
    parts: list[str | int] = []
    for key, idx in _TOKEN.findall(path):
        parts.append(int(idx) if idx else key)
    return parts


def path_get(obj: Any, path: str, default: Any = None) -> Any:
    """Return the value at ``path`` or ``default`` if any segment is missing."""
    cur = obj
    for part in parse_path(path):
        if isinstance(part, int):
            if not isinstance(cur, list) or part >= len(cur):
                return default
            cur = cur[part]
        else:
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
    return cur


def path_exists(obj: Any, path: str) -> bool:
    return path_get(obj, path, _MISSING) is not _MISSING


def iter_list(obj: Any, path: str) -> Iterator[tuple[int, Any, str]]:
    """Yield ``(index, item, item_path)`` for a list at ``path`` (nothing if absent)."""
    items = path_get(obj, path)
    if isinstance(items, list):
        for i, item in enumerate(items):
            yield i, item, f"{path}[{i}]"
