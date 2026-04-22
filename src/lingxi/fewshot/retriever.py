"""Retrieve FewShotSamples relevant to the current inner_thought/user_msg."""

from __future__ import annotations

from typing import Protocol

from lingxi.fewshot.models import FewShotSample
from lingxi.fewshot.store import FewShotQueryResult, FewShotStore


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


_SOURCE_BOOST = {
    "user_correction": 0.05,
    "positive": 0.02,
    "seed": 0.0,
}
_RECIPIENT_BOOST = 0.1


class FewShotRetriever:
    def __init__(self, store: FewShotStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    async def retrieve(
        self,
        query_text: str,
        recipient_key: str | None = None,
        k: int = 3,
        threshold: float = 0.6,
    ) -> list[FewShotSample]:
        """Return top-k samples by reranked score: similarity + source + recipient boosts.

        Candidates called with 4x k then filtered by threshold and deduplicated.
        """
        embedding = await self.embedder.embed(query_text)
        raw: list[FewShotQueryResult] = await self.store.query(
            query_embedding=embedding,
            k=max(k * 4, 12),
            recipient_key=recipient_key,
        )

        scored: list[tuple[float, float, FewShotSample]] = []
        for r in raw:
            boost = _SOURCE_BOOST.get(r.sample.source, 0.0)
            if recipient_key and r.sample.recipient_key == recipient_key:
                boost += _RECIPIENT_BOOST
            scored.append((r.similarity + boost, r.similarity, r.sample))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Threshold filters on raw similarity (not boosted score)
        filtered = [(s, raw_sim, x) for s, raw_sim, x in scored if raw_sim >= threshold]
        # Dedup by inner_thought near-match (light heuristic)
        seen: set[str] = set()
        out: list[FewShotSample] = []
        for _, _raw_sim, sample in filtered:
            key = (sample.inner_thought or sample.context_summary)[:40]
            if key in seen:
                continue
            seen.add(key)
            out.append(sample)
            if len(out) >= k:
                break
        return out
