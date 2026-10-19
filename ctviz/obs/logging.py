"""Structured JSON logging: one line per HTTP request and one per pipeline step.

Only whitelisted, non-sensitive fields are emitted. Request bodies are never logged, and anything
named ``patient_profile`` (or that looks like a credential) is redacted by :func:`redact`.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

REQUEST_LOGGER = "ctviz.request"
STEP_LOGGER = "ctviz.step"
_SENSITIVE_KEYS = {
    "patient_profile",
    "api_key",
    "openai_api_key",
    "authorization",
    "x-api-key",
    "password",
    "token",
    "secret",
}


def redact(obj: Any) -> Any:
    """Deep copy of ``obj`` with sensitive keys replaced by ``'[redacted]'``."""
    if isinstance(obj, dict):
        return {
            k: ("[redacted]" if str(k).lower() in _SENSITIVE_KEYS and v is not None else redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list | tuple):
        return [redact(v) for v in obj]
    return obj


class JsonFormatter(logging.Formatter):
    """Formats a record as one JSON object; structured fields come from ``extra={'fields': {...}}``."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(redact(fields))
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_type"] = record.exc_info[0].__name__
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    root.setLevel(level.upper())
    logging.getLogger("httpx").setLevel(logging.WARNING)  # httpx logs full URLs at INFO


def log_request(**fields: Any) -> None:
    """Emit the single per-request log line."""
    logging.getLogger(REQUEST_LOGGER).info("request", extra={"fields": fields})


def log_step(**fields: Any) -> None:
    """Emit one line per pipeline step / tool call."""
    logging.getLogger(STEP_LOGGER).info("step", extra={"fields": fields})
