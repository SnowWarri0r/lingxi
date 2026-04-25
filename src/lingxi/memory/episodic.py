"""Episodic memory: timestamped records of past conversation sessions."""

from __future__ import annotations

import json
from pathlib import Path

from lingxi.memory.base import EpisodeEntry
from lingxi.memory.long_term import cosine_similarity


class EpisodicMemory:
    """Stores summaries of past conversation sessions as episodes."""

    def __init__(self, max_episodes: int = 500):
        self.max_episodes = max_episodes
        self._episodes: list[EpisodeEntry] = []

    async def store_episode(self, episode: EpisodeEntry) -> str:
        """Store a new episode and return its ID."""
        self._episodes.append(episode)

        # Trim oldest episodes if over capacity
        if len(self._episodes) > self.max_episodes:
            self._episodes = self._episodes[-self.max_episodes :]

        return episode.id

    async def retrieve_relevant(
        self,
        query: str,
        limit: int = 3,
        query_embedding: list[float] | None = None,
    ) -> list[EpisodeEntry]:
        """Retrieve episodes relevant to the query."""
        if query_embedding:
            return self._retrieve_by_embedding(query_embedding, limit)
        return self._retrieve_by_keyword(query, limit)

    def _retrieve_by_embedding(
        self, query_embedding: list[float], limit: int
    ) -> list[EpisodeEntry]:
        scored: list[tuple[float, EpisodeEntry]] = []
        for ep in self._episodes:
            if ep.embedding:
                sim = cosine_similarity(query_embedding, ep.embedding)
                scored.append((sim, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

    def _retrieve_by_keyword(self, query: str, limit: int) -> list[EpisodeEntry]:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored: list[tuple[float, EpisodeEntry]] = []

        for ep in self._episodes:
            text = f"{ep.summary} {' '.join(ep.key_topics)}".lower()
            text_words = set(text.split())
            overlap = len(query_words & text_words)
            if overlap > 0:
                scored.append((overlap / max(len(query_words), 1), ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

    async def get_recent(self, limit: int = 5) -> list[EpisodeEntry]:
        """Get the most recent episodes."""
        return self._episodes[-limit:]

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    async def save_to_disk(self, path: str) -> None:
        data = [ep.model_dump(mode="json") for ep in self._episodes]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    async def load_from_disk(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        self._episodes = [EpisodeEntry.model_validate(item) for item in data]
