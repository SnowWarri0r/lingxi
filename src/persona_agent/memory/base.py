"""Abstract base classes for memory stores."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"


class MemoryEntry(BaseModel):
    """A single memory entry."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: str
    memory_type: MemoryType
    timestamp: datetime = Field(default_factory=datetime.now)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    metadata: dict = Field(default_factory=dict)
    access_count: int = 0
    last_accessed: datetime | None = None


class EpisodeEntry(BaseModel):
    """A summary of a past conversation session."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = Field(default_factory=datetime.now)
    summary: str
    emotional_tone: str = "neutral"
    key_topics: list[str] = Field(default_factory=list)
    turn_count: int = 0
    embedding: list[float] | None = None


class MemoryStore(ABC):
    """Abstract interface for memory storage backends."""

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> str:
        """Store an entry and return its ID."""

    @abstractmethod
    async def retrieve(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Retrieve entries relevant to the query."""

    @abstractmethod
    async def get_by_id(self, entry_id: str) -> MemoryEntry | None:
        """Get a specific entry by ID."""

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID."""

    @abstractmethod
    async def list_all(self) -> list[MemoryEntry]:
        """List all entries."""

    @abstractmethod
    async def save_to_disk(self, path: str) -> None:
        """Persist the store to disk."""

    @abstractmethod
    async def load_from_disk(self, path: str) -> None:
        """Load the store from disk."""
