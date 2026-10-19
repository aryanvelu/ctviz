"""Cohort snapshots: the complete member list of a cohort search, so a cohort is read once, not per question.

A snapshot is ``{nct_ids, total, taken_at}`` for one search (drug/condition/sponsor/... filters, independent
of sort and paging), stored in the ``cache`` table under namespace ``cohort``. The records themselves live
in the study store (``studies``), which is also what feeds the knowledge graph.

They are written after any complete read and, ahead of time, by ``ctviz warm`` for common conditions with
a higher cap than a live request may read (so even cohorts above ``max_studies`` can be answered from every
record). A snapshot is used only while ClinicalTrials.gov still reports the same ``totalCount`` for the
search (checked on the request's first page, which is fetched anyway) and every member is in the store.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from ctviz.ctgov.requests import SearchSpec

NS = "cohort"
_IGNORED = {"sort", "pageSize", "pageToken", "countTotal", "fields", "format"}


def snapshot_key(spec: SearchSpec) -> str:
    params = {k: v for k, v in spec.params().items() if k not in _IGNORED}
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


async def load_snapshot(db: Any, spec: SearchSpec) -> dict[str, Any] | None:
    if db is None:
        return None
    snap = await db.cache_get(NS, snapshot_key(spec))
    return snap if isinstance(snap, dict) and snap.get("nct_ids") else None


async def save_snapshot(db: Any, spec: SearchSpec, nct_ids: list[str], total: int, ttl_s: float) -> None:
    if db is None or not nct_ids:
        return
    await db.cache_set(
        NS, snapshot_key(spec), {"nct_ids": sorted(set(nct_ids)), "total": total, "taken_at": time.time()}, ttl_s
    )
