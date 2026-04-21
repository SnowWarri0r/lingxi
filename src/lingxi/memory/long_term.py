"""Long-term memory: persistent fact/knowledge store with embedding-based retrieval."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from lingxi.memory.base import MemoryEntry, MemoryStore, MemoryType


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


class LongTermMemory(MemoryStore):
    """Persistent fact store with semantic retrieval via embeddings."""

    def __init__(
        self,
        max_entries: int = 10000,
        importance_threshold: float = 0.3,
        decay_rate: float = 0.01,
    ):
        self.max_entries = max_entries
        self.importance_threshold = importance_threshold
        self.decay_rate = decay_rate
        self._entries: dict[str, MemoryEntry] = {}

    async def store(self, entry: MemoryEntry) -> str:
        entry.memory_type = MemoryType.LONG_TERM
        self._entries[entry.id] = entry

        # Evict low-importance entries if over capacity
        if len(self._entries) > self.max_entries:
            await self._evict()

        return entry.id

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        query_embedding: list[float] | None = None,
    ) -> list[MemoryEntry]:
        """Retrieve entries by semantic similarity. Falls back to keyword matching."""
        if query_embedding:
            return self._retrieve_by_embedding(query_embedding, limit)
        return self._retrieve_by_keyword(query, limit)

    def _retrieve_by_embedding(
        self, query_embedding: list[float], limit: int
    ) -> list[MemoryEntry]:
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self._entries.values():
            if entry.embedding:
                sim = cosine_similarity(query_embedding, entry.embedding)
                # Boost by importance
                score = sim * 0.7 + entry.importance * 0.3
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [entry for _, entry in scored[:limit]]

        # Update access metadata
        now = datetime.now()
        for entry in results:
            entry.access_count += 1
            entry.last_accessed = now

        return results

    def _retrieve_by_keyword(self, query: str, limit: int) -> list[MemoryEntry]:
        query_lower = query.lower()
        scored: list[tuple[float, MemoryEntry]] = []

        for entry in self._entries.values():
            content_lower = entry.content.lower()
            # Simple keyword overlap scoring
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1) * 0.7 + entry.importance * 0.3
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [entry for _, entry in scored[:limit]]

        now = datetime.now()
        for entry in results:
            entry.access_count += 1
            entry.last_accessed = now

        return results

    async def get_by_id(self, entry_id: str) -> MemoryEntry | None:
        return self._entries.get(entry_id)

    async def delete(self, entry_id: str) -> bool:
        if entry_id in self._entries:
            del self._entries[entry_id]
            return True
        return False

    async def list_all(self) -> list[MemoryEntry]:
        return list(self._entries.values())

    async def _evict(self) -> None:
        """Remove lowest-scored entries to stay within capacity."""
        now = datetime.now()
        scored: list[tuple[float, str]] = []

        for eid, entry in self._entries.items():
            # Score based on importance, recency, and access frequency
            days_since_access = 0.0
            if entry.last_accessed:
                days_since_access = (now - entry.last_accessed).total_seconds() / 86400
            recency_penalty = self.decay_rate * days_since_access
            score = entry.importance + (entry.access_count * 0.01) - recency_penalty
            scored.append((score, eid))

        scored.sort(key=lambda x: x[0])

        # Remove bottom 10% or entries below threshold
        to_remove = max(1, len(scored) // 10)
        for _, eid in scored[:to_remove]:
            del self._entries[eid]

    async def save_to_disk(self, path: str) -> None:
        data = [entry.model_dump(mode="json") for entry in self._entries.values()]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    async def load_from_disk(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            entry = MemoryEntry.model_validate(item)
            self._entries[entry.id] = entry
