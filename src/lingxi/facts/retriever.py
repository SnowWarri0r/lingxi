"""Fact retrieval interface used by both the Orchestrator (catalog,
counts only) and the Renderer (full fetch).

FactQuery is a lightweight dataclass that captures the orchestrator's
intent: "give me up to N facts for subject S of type T, optionally
matching semantic keyword K".
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from lingxi.facts.models import Fact, FactType
from lingxi.facts.store import FactStore


@dataclass
class FactQuery:
    subject: str
    type: FactType | None = None
    since: datetime | None = None
    semantic: str | None = None  # optional FTS keyword
    limit: int = 5


class FactRetriever:
    def __init__(self, store: FactStore):
        self._store = store

    async def fetch(self, query: FactQuery) -> list[Fact]:
        """Return up to query.limit facts ranked by 3D scoring:

        score = 0.5 * recency_decay(hours_old)
              + 0.3 * (importance / 10)
              + 0.2 * fts_rank

        recency_decay(h) = exp(-0.01 * h)

        fts_rank is 0.0 when no semantic query is given; 1.0 for the best
        FTS5 match when one is given (normalized across candidates).

        After returning, last_accessed is stamped on each returned fact.
        """
        candidates = await self._store.query(
            subject=query.subject,
            type=query.type,
            since=query.since,
            limit=query.limit * 8,
        )
        if not candidates:
            return []

        if query.semantic:
            fts_ranks = await self._store.fts_rank(
                query.semantic, [c.id for c in candidates]
            )
        else:
            fts_ranks = {c.id: 0.0 for c in candidates}

        now = datetime.now()
        scored: list[tuple[float, Fact]] = []
        for fact in candidates:
            # Existing rows may have tz-aware ts (legacy data); new writes from
            # this codebase are naive. Normalize to naive for subtraction.
            fact_ts = fact.ts.replace(tzinfo=None) if fact.ts.tzinfo else fact.ts
            hours_old = max(0.0, (now - fact_ts).total_seconds() / 3600)
            recency = math.exp(-0.01 * hours_old)
            importance = (fact.importance if fact.importance is not None else 5) / 10.0
            relevance = fts_ranks.get(fact.id, 0.0)
            score = 0.5 * recency + 0.3 * importance + 0.2 * relevance
            scored.append((score, fact))

        scored.sort(key=lambda x: -x[0])
        top = [f for _, f in scored[: query.limit]]
        if top:
            await self._store.update_last_accessed([f.id for f in top], now)
        return top

    async def fetch_by_id(self, fact_id: str) -> Fact | None:
        """Return a single Fact by ID, or None if not found."""
        return await self._store.get(fact_id)

    async def get_core_block(self, subject: str) -> Fact | None:
        """Current MemGPT core-memory block for subject (or None)."""
        return await self._store.get_core_block(subject)

    async def catalog(self) -> dict[str, int]:
        """Return {bucket: count} for orchestrator's decision input.

        Bucket key format: "<subject>.<type>" — e.g. "aria.event",
        "user:oc_xxx.pattern", "npc:xiaomin.event".
        """
        all_facts = await self._store.query(subject=None, limit=10000)
        counts: dict[str, int] = defaultdict(int)
        for f in all_facts:
            counts[f"{f.subject}.{f.type.value}"] += 1
        return dict(counts)
