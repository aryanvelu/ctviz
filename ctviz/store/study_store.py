"""Study store helpers: latest raw record per trial (``studies`` table in :mod:`ctviz.store.db`)."""

from __future__ import annotations

from typing import Any

from ctviz.core.background import spawn
from ctviz.ctgov.fields import P
from ctviz.ctgov.paths import path_get


def _records(raws: list[dict[str, Any]], level: str) -> list[tuple[str, str, str | None, Any]]:
    out = []
    for raw in raws:
        nct = path_get(raw, P["nct_id"])
        if isinstance(nct, str):
            out.append((nct, level, path_get(raw, P["last_update_date"]), raw))
    return out


async def save_studies(db: Any, raws: list[dict[str, Any]], level: str) -> None:
    """Upsert raw records (``level`` is 'projected' or 'full'; full is never downgraded)."""
    if db is None or not raws:
        return
    records = _records(raws, level)
    if records:
        await db.put_studies(records)


def save_studies_background(db: Any, raws: list[dict[str, Any]], level: str) -> None:
    """Schedule :func:`save_studies` without waiting for it."""
    if db is not None and raws:
        spawn(save_studies(db, list(raws), level), name="study-store", low_priority=True)


async def load_studies(db: Any, nct_ids: list[str], level: str | None = None) -> dict[str, Any]:
    """Raw records from the store keyed by NCT id (missing ids are absent)."""
    if db is None or not nct_ids:
        return {}
    return await db.get_studies(nct_ids, level)
