"""Retrieve biographical LifeEvents relevant to the current conversation.

Small in-memory store: at init time, embed each life_event's content;
at query time, cosine-similarity rank against the query embedding.

Scale: ~30-50 events per persona, so no need for ChromaDB — plain numpy
is fast enough and keeps the setup zero-dependency for this piece.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from lingxi.persona.models import LifeEvent


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class BiographyRetriever:
    """Keeps LifeEvents + their embeddings in memory; retrieves top-k by topic."""

    def __init__(self, events: list[LifeEvent], embedder: Embedder):
        self.events = events
        self.embedder = embedder
        self._embeddings: np.ndarray | None = None

    async def bootstrap(self) -> None:
        """Embed all events. Idempotent — no-op if already embedded."""
        if self._embeddings is not None or not self.events:
            return
        texts = [e.content + " " + " ".join(e.tags) for e in self.events]
        vecs: list[list[float]] = []
        for t in texts:
            v = await self.embedder.embed(t)
            vecs.append(v)
        self._embeddings = np.array(vecs, dtype=np.float32)

    async def retrieve(
        self,
        query: str,
        k: int = 3,
        threshold: float = 0.55,
    ) -> list[LifeEvent]:
        """Return top-k LifeEvents whose similarity > threshold, most relevant first."""
        if self._embeddings is None or not self.events or not query.strip():
            return []
        q = np.array(await self.embedder.embed(query), dtype=np.float32)
        sims = [_cosine(q, row) for row in self._embeddings]
        scored = sorted(
            [(s, e) for s, e in zip(sims, self.events) if s >= threshold],
            key=lambda x: x[0],
            reverse=True,
        )
        return [e for _, e in scored[:k]]
