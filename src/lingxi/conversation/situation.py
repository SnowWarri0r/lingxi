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

# His facts are fetched once at the larger of the two paths' sizes; the chat
# floor reads the first CHAT_FACTS of the same ranking.
HIS_FACTS = 24
CHAT_FACTS = 18


@dataclass
class Situation:
    key: str
    now: datetime
    shows: str = ""                              # her upcoming shows, rendered
    world: list = field(default_factory=list)    # today's scan
    his_facts: list = field(default_factory=list)  # ranked, best first
    agenda: list = field(default_factory=list)   # open agenda items


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
