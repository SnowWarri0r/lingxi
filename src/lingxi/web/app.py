"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lingxi.web.routes import router
from lingxi.web.session import SessionManager


def create_app(engine=None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        engine: Optional engine instance to store on app.state.engine.
                When provided, annotation endpoints become functional.
    """
    app = FastAPI(
        title="Persona Agent API",
        version="0.1.0",
        description="虚拟人格对话代理 API - 支持 REST 和 WebSocket",
    )

    # CORS - allow all origins by default, configurable via env
    origins = os.environ.get("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
