"""FastAPI application factory.

``create_app()`` builds Settings / Database / CTGovClient / LLM / GraphStore / Orchestrator in the
lifespan. Tests may pass ``settings`` and a ready ``orchestrator`` to skip the wiring.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import ctviz.handlers  # noqa: F401  (registers intent handlers)
from ctviz.api.middleware import GuardMiddleware
from ctviz.api.routes import router
from ctviz.api.schemas.response import SCHEMA_VERSION, ErrorResponse
from ctviz.config import PROJECT_ROOT, Settings, get_settings
from ctviz.core.background import drain
from ctviz.obs.logging import configure_logging

log = logging.getLogger(__name__)
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def build_services(settings: Settings) -> dict[str, Any]:
    """Construct the long-lived services for one process."""
    from ctviz.agents.llm import build_llm
    from ctviz.core.orchestrator import Orchestrator
    from ctviz.ctgov.client import CTGovClient
    from ctviz.store.db import Database

    db = Database(settings.db_path).connect()
    client = CTGovClient(settings, db)
    llm = build_llm(settings)
    graph = text_index = None
    try:
        from ctviz.store.fts import TextIndex
        from ctviz.store.graph import GraphStore

        graph, text_index = GraphStore(db), TextIndex(db)
    except ImportError:
        log.info("knowledge graph module not available; graph features disabled")
    return {
        "db": db,
        "client": client,
        "llm": llm,
        "graph": graph,
        "text_index": text_index,
        "orchestrator": Orchestrator(settings, db, client, llm, graph, text_index=text_index),
    }


def create_app(settings: Settings | None = None, orchestrator: Any = None) -> FastAPI:
    """App factory (``uvicorn --factory ctviz.app:create_app``)."""

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if getattr(app.state, "settings", None) is None:
            app.state.settings = get_settings()
        configure_logging(app.state.settings.log_level)
        owned: dict[str, Any] = {}
        if getattr(app.state, "orchestrator", None) is None:
            owned = build_services(app.state.settings)
            for k, v in owned.items():
                setattr(app.state, k, v)
        try:
            yield
        finally:
            await drain(timeout_s=5.0)
            if owned:
                await owned["client"].aclose()
                owned["db"].close()

    app = FastAPI(
        title="ctviz — ClinicalTrials.gov query-to-visualization API",
        version=SCHEMA_VERSION,
        lifespan=lifespan,
        description="Natural-language clinical-trial questions → cited visualization specs.",
    )
    app.state.settings = settings
    app.state.orchestrator = orchestrator
    if orchestrator is not None:
        for name in ("db", "client", "llm", "graph", "text_index"):
            setattr(app.state, name, getattr(orchestrator, name, None))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"loc": list(e.get("loc", [])), "msg": e.get("msg", ""), "type": e.get("type", "")} for e in exc.errors()
        ]
        msg = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}" for e in errors)
        body = ErrorResponse(error="validation_error", message=msg[:1000]).model_dump(exclude_none=True)
        return JSONResponse({**body, "detail": errors}, status_code=422)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error (%s)", type(exc).__name__)
        return JSONResponse(
            ErrorResponse(error="internal_error", message="Unexpected server error.").model_dump(exclude_none=True),
            status_code=500,
        )

    app.include_router(router)
    app.add_middleware(GuardMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )
    if FRONTEND_DIST.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    return app
