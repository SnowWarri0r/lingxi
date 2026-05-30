"""Schema for the unified Fact table.

A Fact is the atomic unit of knowledge in the new architecture.
Replaces inner_life events, relational_memory entries, social NPC events,
world briefings, and long-term memory facts in one typed structure.

Subject ownership is the core invariant: each Fact's `subject` identifies
who or what the fact is ABOUT (aria/user:x/npc:y/world). Writers enforce
this — `LifeWriter` only writes subject=aria, `NPCWriter` only writes
subject=npc:*, etc. Subject isolation prevents the cross-contamination
that plagued the old system (e.g. NPC events bleeding into Aria's
self-narrative).
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Source(str, Enum):
    USER_STATED      = "user_stated"
    LIFE_SIMULATED   = "life_simulated"
    NPC_TICKER       = "npc_ticker"
    LLM_INFERRED     = "llm_inferred"
    WORLD_FETCH      = "world_fetch"
    BIOGRAPHY        = "biography"

    @property
    def default_confidence(self) -> float:
        return {
            Source.USER_STATED:    1.0,
            Source.BIOGRAPHY:      1.0,
            Source.WORLD_FETCH:    0.9,
            Source.LIFE_SIMULATED: 0.8,
            Source.NPC_TICKER:     0.8,
            Source.LLM_INFERRED:   0.5,
        }[self]


class FactType(str, Enum):
    EVENT        = "event"
    PATTERN      = "pattern"
    OPINION      = "opinion"
    PLAN         = "plan"
    EMOTION_NOTE = "emotion_note"


_SUBJECT_RE = re.compile(r"^(aria|world|user:[A-Za-z0-9_:-]+|npc:[A-Za-z0-9_-]+)$")


class Fact(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    subject: str
    content: str
    source: Source
    type: FactType
    ts: datetime
    written_at: datetime = Field(default_factory=datetime.now)
    confidence: float | None = None
    importance: int | None = None
    last_accessed: datetime | None = None
    expires_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    supersedes: str | None = None

    @field_validator("subject")
    @classmethod
    def _check_subject(cls, v: str) -> str:
        if not _SUBJECT_RE.match(v):
            raise ValueError(
                f"subject must match aria|world|user:X|npc:X, got {v!r}"
            )
        return v

    def model_post_init(self, _ctx) -> None:
        if self.confidence is None:
            self.confidence = self.source.default_confidence
