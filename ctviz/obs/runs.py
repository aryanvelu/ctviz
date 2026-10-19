"""Per-request run artifacts and step logging.

Every pipeline step goes through :func:`step`, which (a) times it in the response trace, (b) appends a
record ``{tool, inputs (redacted), ms, status, summary}`` to the run, and (c) emits one JSON log line.
ClinicalTrials.gov calls are taken from ``RequestTrace.ctgov_calls_log`` (recorded by the client); their
``cache_file`` points at the single readable copy of each raw response under ``settings.cache_dir``.

At the end of a request :meth:`RunRecorder.flush` writes ``runs/<request_id>/``::

    request.json  plan.json  ctgov_calls.json  handler_result.json  steps.jsonl  response.json
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ctviz.obs.logging import log_step, redact
from ctviz.obs.trace import current_trace


def query_digest(query: str) -> str:
    """Stable, non-reversible stand-in for a question in logs and artifacts."""
    return "sha256:" + hashlib.sha256(query.encode()).hexdigest()[:16]


def scrub(obj: Any, terms: frozenset[str] | set[str]) -> Any:
    """Replace every string containing one of ``terms`` (casefolded) with its :func:`query_digest`."""
    if not terms:
        return obj
    if isinstance(obj, dict):
        return {k: scrub(v, terms) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [scrub(v, terms) for v in obj]
    if isinstance(obj, str) and any(t in obj.casefold() for t in terms):
        return query_digest(obj)
    return obj


@dataclass
class RunRecorder:
    """Collects artifacts for one request; ``directory=None`` keeps them in memory only.

    ``private_terms`` (patient-profile strings, casefolded) are hashed out of every artifact, step record
    and step log line, so a patient's conditions/medications/notes never reach ``runs/`` or the logs.
    """

    request_id: str
    directory: Path | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    ctgov_calls: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    private_terms: frozenset[str] = frozenset()

    def add_step(self, record: dict[str, Any]) -> None:
        self.steps.append(record)

    def scrub(self, value: Any) -> Any:
        return scrub(value, self.private_terms)

    def set(self, name: str, value: Any) -> None:
        """Record an artifact (``request``, ``plan``, ``handler_result``, ``response``)."""
        self.artifacts[name] = self.scrub(redact(value))

    def capture_calls(self) -> None:
        """Snapshot the client's call log for this request (url, cached, ms, status, cache_file)."""
        log = getattr(current_trace(), "ctgov_calls_log", None)
        self.ctgov_calls = [dict(c) for c in log] if isinstance(log, list) else []

    def flush(self) -> Path | None:
        """Write artifacts to disk (synchronous; run it in a thread)."""
        if self.directory is None:
            return None
        d = self.directory
        d.mkdir(parents=True, exist_ok=True)
        for name, value in self.artifacts.items():
            (d / f"{name}.json").write_text(json.dumps(value, indent=1, default=str))
        (d / "ctgov_calls.json").write_text(json.dumps(self.ctgov_calls, indent=1, default=str))
        with (d / "steps.jsonl").open("w") as fh:
            for s in self.steps:
                fh.write(json.dumps(s, default=str, separators=(",", ":")) + "\n")
        return d


_current: contextvars.ContextVar[RunRecorder | None] = contextvars.ContextVar("ctviz_run", default=None)


def current_run() -> RunRecorder | None:
    return _current.get()


@contextlib.contextmanager
def run_scope(rec: RunRecorder) -> Iterator[RunRecorder]:
    token = _current.set(rec)
    try:
        yield rec
    finally:
        _current.reset(token)


@contextlib.contextmanager
def step(name: str, **inputs: Any) -> Iterator[dict[str, Any]]:
    """Time a pipeline step. Set ``info['summary']`` (compact result) and optionally ``info['status']``."""
    tr = current_trace()
    rec = current_run()
    t0 = time.perf_counter()
    with tr.step(name) as info:
        status = "ok"
        try:
            yield info
        except BaseException as exc:
            status = "timeout" if isinstance(exc, TimeoutError) else "error"
            info.setdefault("error", type(exc).__name__)
            raise
        finally:
            status = info.get("status", status)
            record = {
                "tool": name,
                "inputs": redact(inputs),
                "ms": round((time.perf_counter() - t0) * 1000, 1),
                "status": status,
                "summary": info.get("summary"),
                "error": info.get("error"),
            }
            if rec is not None:
                record = rec.scrub(record)
                record["request_id"] = rec.request_id
                rec.add_step(record)
            log_step(**record)
