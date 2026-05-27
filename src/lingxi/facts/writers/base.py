"""Base class for Writers.

Each writer subclass declares the (Source, subject-pattern) pair it owns.
Attempting to write outside that ownership raises ValueError. This is the
mechanism that makes subject isolation a structural invariant — there is
no path for, say, NPCTicker to write a fact with subject=aria.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import ClassVar

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.store import FactStore


class WriterBase:
    ALLOWED_SOURCE: ClassVar[Source]
    SUBJECT_PATTERN: ClassVar[str]  # regex string

    def __init__(self, store: FactStore):
        self._store = store
        self._pattern = re.compile(self.SUBJECT_PATTERN)

    async def write(
        self,
        *,
        subject: str,
        content: str,
        type: FactType,
        ts: datetime,
        confidence: float | None = None,
        tags: list[str] | None = None,
        supersedes: str | None = None,
        expires_at: datetime | None = None,
    ) -> Fact:
        if not self._pattern.match(subject):
            raise ValueError(
                f"{self.__class__.__name__} cannot write subject={subject!r}; "
                f"allowed pattern: {self.SUBJECT_PATTERN}"
            )

        fact = Fact(
            subject=subject,
            content=content,
            source=self.ALLOWED_SOURCE,
            type=type,
            ts=ts,
            confidence=confidence,
            tags=tags or [],
            supersedes=supersedes,
            expires_at=expires_at,
        )
        await self._store.write(fact)
        return fact
