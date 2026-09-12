"""External-world awareness — daily news briefing.

Aria has internal-life simulation (current_activity, recent_events) and
relational memory (per-recipient texture), but no awareness of the
real world. Result: she can't engage with "最近 X" topics and feels
disconnected — "不像生活在真实世界里".

This module gives her a daily morning briefing fetched from the web,
re-voiced in her own register, surfaced into the inner-state section
of the prompt. She doesn't broadcast it; the model decides whether to
bring something up when relevant.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class NewsItem(BaseModel):
    """One item in today's briefing.

    `voice` is what gets surfaced — not the headline. The fetcher re-writes
    each item in her register ("今早扫到 X" / "看了下 X") so it reads as what
    she absorbed, not a news ticker.

    category is free text: the subjects come from the persona's
    world_interests, so no fixed vocabulary can cover them.
    """

    headline: str             # original headline / topic
    voice: str                # her one-line take, IM-style
    category: str = "其他"
    source: str = ""          # outlet / domain
    url: str = ""             # optional
    fetched_at: datetime = Field(default_factory=datetime.now)


class DailyBriefing(BaseModel):
    """Today's news briefing — short list, not a digest."""

    date: date
    items: list[NewsItem] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)

    def is_empty(self) -> bool:
        return not self.items
