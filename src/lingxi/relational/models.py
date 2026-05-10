"""Schemas for per-recipient relational memory (#1 from the agency-set).

We already have:
- biography (Aria's life events) — what HAPPENED to her
- subjective_view (impressions/worries) — how she SEES this person
- recent_episodes (chroma) — past conversation summaries

What's missing — and what distill-skill projects (ex-skill / her-skill /
forge-skill) all build first-class data structures for — is **OUR**
between-us texture: inside jokes, shared places, fight-and-repair patterns,
sweet moments, pet names, daily patterns. Without this, even with rich
episode memory, the agent never feels "我们之间有东西"; it only feels like
"she remembers what happened".

Storage is per-recipient (a different relationship has different texture).
Auto-extracted from recent dialogue + diary by the reflection loop. The
prompt renders these alongside subjective_view so they're always in
context, not retrieval-gated.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class InsideJoke(BaseModel):
    """A phrase / reference only the two of you understand."""

    phrase: str            # "蜘蛛会做梦的"
    origin: str            # one-liner explaining where it came from
    last_used_at: datetime = Field(default_factory=datetime.now)
    use_count: int = 1


class SharedPlace(BaseModel):
    """A location significant to the relationship."""

    name: str              # "楼下那家便利店"
    significance: str      # "他下班路过的，有次他给你带了饭团"
    last_referenced_at: datetime = Field(default_factory=datetime.now)


class FightPattern(BaseModel):
    """A recurring conflict shape and how it usually repairs.

    Captured BEHAVIORALLY (her_pattern + repair), not as a moralized
    "she's wrong / he's wrong". The point is to know the rhythm.
    """

    trigger: str           # "他超过 3 小时不回消息"
    her_pattern: str       # "嘴硬几小时然后说'算了'"
    typical_repair: str    # "他用具体小事示弱：'晚饭一起吃？'"
    last_occurred_at: datetime | None = None


class SweetMoment(BaseModel):
    """A specific moment worth holding onto.

    Not just "we had a nice conversation" — a CONCRETE moment with
    sensory or situational detail that, when referenced months later,
    still rings.
    """

    timestamp: datetime
    content: str           # "他凌晨2点说想看流星雨那次"
    weight: Literal["high", "medium", "low"] = "medium"


class DailyPattern(BaseModel):
    """A regularity in the user's life that Aria has internalized.

    Not "what he does" generically but what's predictable enough that
    Aria can reference it casually ("你今天 11 点下班吗").
    """

    pattern: str           # "他每天 11 点下班"
    confidence: Literal["high", "medium", "low"] = "medium"
    last_confirmed_at: datetime = Field(default_factory=datetime.now)


class RelationalMemory(BaseModel):
    """Root container for one recipient's relationship texture.

    Auto-grown by the reflection loop's extractor; rendered into the
    prompt alongside subjective_view. All fields default to empty so a
    fresh relationship starts blank and accumulates over time.
    """

    recipient_key: str
    inside_jokes: list[InsideJoke] = Field(default_factory=list)
    shared_places: list[SharedPlace] = Field(default_factory=list)
    fight_patterns: list[FightPattern] = Field(default_factory=list)
    sweet_moments: list[SweetMoment] = Field(default_factory=list)
    pet_names: list[str] = Field(default_factory=list)         # "笨蛋", "老李"
    daily_patterns: list[DailyPattern] = Field(default_factory=list)
    # Voice diff (#4): phrases Aria has characteristically used WITH THIS
    # user — different from inside_jokes (which both use as references)
    # and from persona.message_habits.signature_phrases (character-wide,
    # kept empty to avoid forced tics). Per-recipient: her voice grows
    # with the relationship.
    signature_phrases: list[str] = Field(default_factory=list)
    relationship_summary: str = ""    # one-paragraph narrative she tells herself
    last_extracted_at: datetime | None = None  # for incremental extraction

    def is_empty(self) -> bool:
        """Skip prompt rendering when nothing's been accumulated yet."""
        return not (
            self.inside_jokes
            or self.shared_places
            or self.fight_patterns
            or self.sweet_moments
            or self.pet_names
            or self.daily_patterns
            or self.signature_phrases
            or self.relationship_summary
        )
