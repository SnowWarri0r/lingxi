"""FastAPI REST and WebSocket routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect

from persona_agent.web.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MemoryStats,
    MoodResponse,
    PersonaInfo,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionEndRequest,
    SessionEndResponse,
    WSIncoming,
    WSOutgoing,
)
from persona_agent.web.session import SessionManager

router = APIRouter()


def get_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


# --- REST ---


@router.get("/health", response_model=HealthResponse)
async def get_health(mgr: SessionManager = Depends(get_manager)):
    return HealthResponse(status="ok", active_sessions=mgr.active_count)


@router.post("/session/create", response_model=SessionCreateResponse)
async def create_session(
    body: SessionCreateRequest,
    mgr: SessionManager = Depends(get_manager),
):
    state = await mgr.create_session(
        persona_path=body.persona_path,
        config_path=body.config_path,
    )
    return SessionCreateResponse(
        session_id=state.session_id,
        persona_name=state.persona_name,
        persona_full_name=state.persona_full_name,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    mgr: SessionManager = Depends(get_manager),
):
    state = mgr.get_session(body.session_id)
    if not state:
        raise HTTPException(404, f"Session not found: {body.session_id}")

    async with state.lock:
        response = await state.engine.chat(body.message)

    return ChatResponse(response=response, mood=state.engine._current_mood)


@router.get("/persona", response_model=PersonaInfo)
async def get_persona(
    session_id: str = Query(...),
    mgr: SessionManager = Depends(get_manager),
):
    state = mgr.get_session(session_id)
    if not state:
        raise HTTPException(404, f"Session not found: {session_id}")

    p = state.engine.persona
    return PersonaInfo(
        name=p.name,
        full_name=p.identity.full_name,
        age=p.identity.age,
        occupation=p.identity.occupation,
        background=p.identity.background,
        traits=[t.model_dump() for t in p.personality.traits],
        values=p.personality.values,
        speaking_style=p.speaking_style.model_dump(),
        mood=state.engine._current_mood,
    )


@router.get("/memory/stats", response_model=MemoryStats)
async def get_memory_stats(
    session_id: str = Query(...),
    mgr: SessionManager = Depends(get_manager),
):
    state = mgr.get_session(session_id)
    if not state:
        raise HTTPException(404, f"Session not found: {session_id}")

    stats = state.engine.memory.get_stats()
    return MemoryStats(**stats)


@router.get("/mood", response_model=MoodResponse)
async def get_mood(
    session_id: str = Query(...),
    mgr: SessionManager = Depends(get_manager),
):
    state = mgr.get_session(session_id)
    if not state:
        raise HTTPException(404, f"Session not found: {session_id}")

    return MoodResponse(mood=state.engine._current_mood)


@router.post("/session/end", response_model=SessionEndResponse)
async def end_session(
    body: SessionEndRequest,
    mgr: SessionManager = Depends(get_manager),
):
    result = await mgr.end_session(body.session_id)
    return SessionEndResponse(
        facts_stored=result.get("facts_stored", 0),
        episode_id=result.get("episode_id"),
    )


# --- WebSocket ---


@router.websocket("/ws/chat/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming conversation.

    Client sends:  {"type": "message", "content": "你好"}
    Server sends:  {"type": "chunk", "content": "你"}
                   {"type": "chunk", "content": "好"}
                   {"type": "mood", "content": "开心"}
                   {"type": "done", "content": "你好！..."}
    """
    mgr: SessionManager = websocket.app.state.session_manager
    state = mgr.get_session(session_id)

    if not state:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg = WSIncoming.model_validate_json(raw)
            except Exception:
                await websocket.send_text(
                    WSOutgoing(type="error", content="Invalid message format").model_dump_json()
                )
                continue

            if msg.type == "ping":
                await websocket.send_text(WSOutgoing(type="pong").model_dump_json())
                continue

            if msg.type != "message" or not msg.content.strip():
                continue

            state.touch()

            async with state.lock:
                try:
                    async for event in state.engine.chat_stream_events(msg.content):
                        out = WSOutgoing(type=event.type, content=event.content)
                        await websocket.send_text(out.model_dump_json())
                except Exception as e:
                    await websocket.send_text(
                        WSOutgoing(type="error", content=str(e)).model_dump_json()
                    )

    except WebSocketDisconnect:
        pass
