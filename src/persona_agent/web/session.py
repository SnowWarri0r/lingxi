"""Session manager: one ConversationEngine per session."""

from __future__ import annotations

import asyncio
import time
import uuid

from persona_agent.auth.manager import AuthManager
from persona_agent.conversation.engine import ConversationEngine
from persona_agent.app import create_engine


class SessionState:
    """Represents an active chat session."""

    def __init__(self, session_id: str, engine: ConversationEngine):
        self.session_id = session_id
        self.engine = engine
        self.created_at = time.time()
        self.last_active = time.time()
        self.lock = asyncio.Lock()

    def touch(self) -> None:
        self.last_active = time.time()

    @property
    def persona_name(self) -> str:
        return self.engine.persona.name

    @property
    def persona_full_name(self) -> str:
        return self.engine.persona.identity.full_name


class SessionManager:
    """Manages concurrent conversation sessions."""

    def __init__(
        self,
        session_timeout: float = 1800,  # 30 minutes
        auth_manager: AuthManager | None = None,
    ):
        self._sessions: dict[str, SessionState] = {}
        self._session_timeout = session_timeout
        self._auth_manager = auth_manager
        self._cleanup_task: asyncio.Task | None = None

    async def create_session(
        self,
        persona_path: str | None = None,
        config_path: str = "config/default.yaml",
    ) -> SessionState:
        """Create a new session with its own ConversationEngine."""
        session_id = uuid.uuid4().hex[:16]

        engine = await create_engine(
            persona_path=persona_path,
            config_path=config_path,
            auth_manager=self._auth_manager,
        )

        state = SessionState(session_id, engine)
        self._sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> SessionState | None:
        state = self._sessions.get(session_id)
        if state:
            state.touch()
        return state

    async def end_session(self, session_id: str) -> dict:
        state = self._sessions.pop(session_id, None)
        if state is None:
            return {"facts_stored": 0, "episode_id": None}
        return await state.engine.end_session()

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    def list_sessions(self) -> list[dict]:
        return [
            {
                "session_id": s.session_id,
                "persona_name": s.persona_name,
                "created_at": s.created_at,
                "last_active": s.last_active,
            }
            for s in self._sessions.values()
        ]

    async def cleanup_stale(self) -> int:
        """Remove sessions that have been idle too long."""
        now = time.time()
        stale = [
            sid
            for sid, s in self._sessions.items()
            if now - s.last_active > self._session_timeout
        ]
        for sid in stale:
            await self.end_session(sid)
        return len(stale)

    async def cleanup_loop(self, interval: float = 60) -> None:
        """Background task to periodically clean up stale sessions."""
        while True:
            await asyncio.sleep(interval)
            await self.cleanup_stale()

    async def shutdown_all(self) -> None:
        """End all sessions on server shutdown."""
        for sid in list(self._sessions.keys()):
            await self.end_session(sid)
