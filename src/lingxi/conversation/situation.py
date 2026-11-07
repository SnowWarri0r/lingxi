"""What is true for one person at one moment, gathered once for every path
that speaks to him.

Chat and openers each fetched their own copy of the same things — his facts,
today's scan, her schedule, the agenda — at their own clock, and each fix
had to land twice. The opener's show countdown read the wall clock while
everything else in that turn read the clock it was given; her promises were
visible to one path and not the other. The paths still render differently
(a reply wants a short floor, an opener a fuller picture); what they read is
gathered here, against one `now`.

Every part is fail-safe on its own: a part that cannot be read is empty, and
the turn goes on with the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time

# His facts are fetched once, at the largest size any reader takes; each
# reader takes the first N of the same ranking: the chat floor CHAT_FACTS,
# the opener's known-facts block OPENER_FACTS, his routine all of them.
HIS_FACTS = 40
OPENER_FACTS = 24
CHAT_FACTS = 18

# Facts about his routine. A reply and an opener both set them above the
# generic clock guess: without them, at 20:23 she was told he was off work
# while he had said he works late. Matched on substrings rather than a
# semantic query because `semantic` is a 0.2-weight FTS boost, not a filter.
SCHEDULE_MARKERS = (
    "下班", "上班", "起床", "睡", "加班", "通勤", "作息",
    "上课", "放学", "午休", "值班", "排班", "工作日", "周末",
)


@dataclass
class Situation:
    key: str
    now: datetime
    shows: str = ""                              # her upcoming shows, rendered
    world: list = field(default_factory=list)    # today's scan
    his_facts: list = field(default_factory=list)  # ranked, best first
    agenda: list = field(default_factory=list)   # open agenda items

    @property
    def schedule(self) -> list[str]:
        """His stated routine, up to three lines."""
        return [f.content for f in self.his_facts
                if any(m in f.content for m in SCHEDULE_MARKERS)][:3]


async def gather(engine, key: str | None, now: datetime) -> Situation:
    from lingxi.facts.models import FactType
    from lingxi.facts.retriever import FactQuery
    from lingxi.persona.prompt_builder import build_upcoming_shows_block

    sit = Situation(key=key or "", now=now)
    sit.shows = build_upcoming_shows_block(engine.persona, now.date()) or ""
    retriever = getattr(engine, "fact_retriever", None)
    if retriever is not None:
        try:
            sit.world = await retriever.fetch(FactQuery(
                subject="world", type=FactType.EVENT,
                since=datetime.combine(now.date(), time.min), limit=5))
        except Exception as e:
            print(f"[situation] world failed (non-fatal): {e!r}", flush=True)
        if key:
            try:
                sit.his_facts = await retriever.fetch(FactQuery(
                    subject=f"user:{key}", type=FactType.PATTERN, limit=HIS_FACTS))
            except Exception as e:
                print(f"[situation] his facts failed (non-fatal): {e!r}", flush=True)
    book = getattr(engine, "agenda", None)
    if book is not None and key:
        try:
            sit.agenda = book.open_items(key, now.date())
        except Exception as e:
            print(f"[situation] agenda failed (non-fatal): {e!r}", flush=True)
    return sit
