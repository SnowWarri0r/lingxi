"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from lingxi.web.routes import router
from lingxi.web.session import SessionManager


# Routes exempt from API key auth (read-only, no resource consumption).
_PUBLIC_PATHS = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


def create_app(engine=None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        engine: Optional engine instance to store on app.state.engine.
                When provided, annotation endpoints become functional.

    Security defaults:
    - CORS: empty (no cross-origin) unless `CORS_ORIGINS` env var is set
    - API key: enforced on all non-public routes if `LINGXI_API_KEY` env
      var is set. Without the env var, no auth (single-machine local dev).
    """
    app = FastAPI(
        title="Persona Agent API",
        version="0.1.0",
        description="虚拟人格对话代理 API - 支持 REST 和 WebSocket",
    )

    # CORS — default to empty (same-origin only). The previous "*" with
    # allow_credentials=True is rejected by spec-compliant browsers anyway,
    # but it could still leak via misbehaving clients. Set CORS_ORIGINS
    # explicitly when you need cross-origin access.
    origins_env = os.environ.get("CORS_ORIGINS", "").strip()
    origins = [o.strip() for o in origins_env.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Optional API key — when set, all non-public routes require
    # `X-API-Key` header to match.
    api_key = os.environ.get("LINGXI_API_KEY", "").strip()
    if api_key:
        @app.middleware("http")
        async def _api_key_gate(request: Request, call_next):
            path = request.url.path
            if path in _PUBLIC_PATHS or path.startswith("/docs"):
                return await call_next(request)
            provided = request.headers.get("x-api-key", "")
            if provided != api_key:
                raise HTTPException(status_code=401, detail="invalid or missing API key")
            return await call_next(request)

    # Session manager
    timeout = int(os.environ.get("SESSION_TIMEOUT", "1800"))
    session_manager = SessionManager(session_timeout=timeout)
    app.state.session_manager = session_manager

    # Optional engine (for annotation endpoints)
    app.state.engine = engine

    app.include_router(router)

    @app.on_event("startup")
    async def startup():
        asyncio.create_task(session_manager.cleanup_loop())

    @app.on_event("shutdown")
    async def shutdown():
        await session_manager.shutdown_all()

    return app


app = create_app()
