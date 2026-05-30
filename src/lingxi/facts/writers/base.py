"""Base class for Writers.

Each writer subclass declares the (Source, subject-pattern) pair it owns.
Attempting to write outside that ownership raises ValueError. This is the
mechanism that makes subject isolation a structural invariant — there is
no path for, say, NPCTicker to write a fact with subject=aria.

Subclasses may declare either:
  ALLOWED_SOURCE: ClassVar[Source]   — single allowed source (legacy)
  ALLOWED_SOURCES: ClassVar[frozenset[Source]]  — multi-source (e.g. LifeWriter)

If ALLOWED_SOURCES is defined it takes precedence over ALLOWED_SOURCE.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import ClassVar

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.store import FactStore


class WriterBase:
    ALLOWED_SOURCE: ClassVar[Source]
    ALLOWED_SOURCES: ClassVar[frozenset[Source] | None] = None
    SUBJECT_PATTERN: ClassVar[str]  # regex string

    def __init__(
        self,
        store: FactStore,
        *,
        scorer=None,              # ImportanceScorer-like (has async score_one(fact) -> int)
        reflection_trigger=None,  # ReflectionTrigger-like (has async observe(n: int))
    ):
        self._store = store
        self._pattern = re.compile(self.SUBJECT_PATTERN)
        self._scorer = scorer
        self._trigger = reflection_trigger

    def _check_source(self, source: Source) -> None:
        allowed = self.__class__.__dict__.get("ALLOWED_SOURCES") or getattr(
            self.__class__, "ALLOWED_SOURCES", None
        )
        if allowed is not None:
            if source not in allowed:
                raise ValueError(
                    f"{self.__class__.__name__} cannot write source={source!r}; "
                    f"allowed: {allowed}"
                )
        else:
            if source != self.ALLOWED_SOURCE:
                raise ValueError(
                    f"{self.__class__.__name__} cannot write source={source!r}; "
                    f"allowed: {self.ALLOWED_SOURCE}"
                )

    async def write(
        self,
        fact_or_subject=None,
        *,
        subject: str | None = None,
        content: str | None = None,
        type: FactType | None = None,
        source: Source | None = None,
        ts: datetime | None = None,
        confidence: float | None = None,
        tags: list[str] | None = None,
        supersedes: str | None = None,
        expires_at: datetime | None = None,
        importance: int | None = None,
    ) -> Fact:
        """Write a fact.

        Two calling conventions:
          (a) write(fact)  — pass a pre-built Fact object directly
          (b) write(subject=..., content=..., type=..., ts=...)  — build internally

        In both cases subject-pattern and source allow-list are enforced.

        If the fact's importance is None and a scorer was provided, score_one()
        is called to assign importance before writing.  After a successful write,
        if a reflection_trigger was provided AND importance is not None, observe()
        is called with the importance value.
        """
        if isinstance(fact_or_subject, Fact):
            fact = fact_or_subject
        else:
            # Legacy keyword-only calling convention
            if fact_or_subject is not None:
                raise TypeError(
                    "write() first positional arg must be a Fact or omitted"
                )
            if subject is None or content is None or type is None or ts is None:
                raise ValueError("write() requires subject, content, type, ts")
            resolved_source = source or self.ALLOWED_SOURCE
            fact = Fact(
                subject=subject,
                content=content,
                source=resolved_source,
                type=type,
                ts=ts,
                confidence=confidence,
                tags=tags or [],
                supersedes=supersedes,
                expires_at=expires_at,
                importance=importance,
            )

        if not self._pattern.match(fact.subject):
            raise ValueError(
                f"{self.__class__.__name__} cannot write subject={fact.subject!r}; "
                f"allowed pattern: {self.SUBJECT_PATTERN}"
            )
        self._check_source(fact.source)

        # Score importance if missing and scorer is available
        if fact.importance is None and self._scorer is not None:
            fact.importance = await self._scorer.score_one(fact)

        await self._store.write(fact)

        # Notify trigger after successful write (only when importance is known)
        if self._trigger is not None and fact.importance is not None:
            await self._trigger.observe(fact.importance)

        return fact

    async def write_skip_scorer(
        self,
        fact: Fact,
        trigger_observation: bool = True,
    ) -> Fact:
        """Bypass the scorer.

        For callers (planner, reflector) that pre-assign importance and do not
        want it overwritten by the scorer.  If importance is still None at call
        time, a neutral fallback of 5 is assigned.

        trigger_observation=False suppresses the reflection-trigger call even
        when a trigger is configured — useful when the caller handles its own
        batching or wants to suppress redundant signals.
        """
        if not self._pattern.match(fact.subject):
            raise ValueError(
                f"{self.__class__.__name__} cannot write subject={fact.subject!r}; "
                f"allowed pattern: {self.SUBJECT_PATTERN}"
            )
        self._check_source(fact.source)

        if fact.importance is None:
            fact.importance = 5  # neutral fallback

        await self._store.write(fact)

        if self._trigger is not None and trigger_observation:
            await self._trigger.observe(fact.importance)

        return fact
