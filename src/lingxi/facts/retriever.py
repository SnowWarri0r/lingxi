"""Fact retrieval interface used by both the Orchestrator (catalog,
counts only) and the Renderer (full fetch).

FactQuery is a lightweight dataclass that captures the orchestrator's
intent: "give me up to N facts for subject S of type T, optionally
matching semantic keyword K".
"""

from __future__ import annotations

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
        if query.semantic:
            # FTS5 path: search content, then filter by subject/type in Python.
            # FTS5 index doesn't span structured fields cheaply.
            candidates = await self._store.search_fts(
                query.semantic, limit=query.limit * 4
            )
            filtered = [
                f for f in candidates
                if f.subject == query.subject
                and (query.type is None or f.type == query.type)
                and (query.since is None or f.ts >= query.since)
            ]
            return filtered[: query.limit]

        return await self._store.query(
            subject=query.subject,
            type=query.type,
            since=query.since,
            limit=query.limit,
        )

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
