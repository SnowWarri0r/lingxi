"""Pydantic request/response models for the web API."""

from __future__ import annotations

from pydantic import BaseModel


# --- REST ---

class SessionCreateRequest(BaseModel):
    persona_path: str | None = None
    config_path: str = "config/default.yaml"


class SessionCreateResponse(BaseModel):
    session_id: str
    persona_name: str
    persona_full_name: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    mood: str


class SessionEndRequest(BaseModel):
    session_id: str


class SessionEndResponse(BaseModel):
    facts_stored: int
    episode_id: str | None = None


class PersonaInfo(BaseModel):
    name: str
    full_name: str
    age: int | None = None
    occupation: str | None = None
    background: str = ""
    traits: list[dict] = []
    values: list[str] = []
    speaking_style: dict = {}
    mood: str = ""


class MemoryStats(BaseModel):
    short_term_turns: int
    long_term_entries: int
    episodes: int


class MoodResponse(BaseModel):
    mood: str


class HealthResponse(BaseModel):
    status: str
    active_sessions: int


# --- WebSocket ---

class WSIncoming(BaseModel):
    """Message from client via WebSocket."""

    type: str = "message"  # "message" | "ping"
    content: str = ""


class WSOutgoing(BaseModel):
    """Message to client via WebSocket."""

    type: str  # "chunk" | "done" | "mood" | "memory_write" | "plan_update" | "error" | "pong"
    content: str | None = None
