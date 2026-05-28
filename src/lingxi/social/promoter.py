"""Promote significant NPC events into Aria's facts + share-intent queue.

Was: write LifeEvent(wants_to_share=True) into inner_state.recent_events.
Now: write Fact(subject="aria", tags=[...]) via LifeWriter + ShareIntentStore.queue().

The two safeguards (per-NPC cooldown, idempotency via NPCEvent.promoted_to_aria)
remain. Cooldown lives in ShareIntentStore now.
"""

from __future__ import annotations

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.writers.life import LifeWriter
from lingxi.proactive.share_intent import ShareIntentStore
from lingxi.social.models import NPC, NPCEvent


DEFAULT_THRESHOLD = 0.6


class SocialPromoter:
    def __init__(
        self,
        life_writer: LifeWriter,
        share_intent_store: ShareIntentStore,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        social_store=None,  # optional, for marking NPCEvent.promoted_to_aria
    ):
        self._life_writer = life_writer
        self._share_intent = share_intent_store
        self._threshold = threshold
        self._social = social_store

    async def maybe_promote(self, npc: NPC, event: NPCEvent) -> bool:
        if event.type != "aria_interaction":
            return False
        if event.significance < self._threshold:
            return False
        if event.promoted_to_aria:
            return False
        if await self._share_intent.is_in_cooldown(npc.id):
            return False

        content = _format_for_aria_pov(npc, event)
        fact = Fact(
            subject="aria",
            content=content,
            source=Source.NPC_TICKER,
            type=FactType.EVENT,
            ts=event.ts,
            tags=[
                f"from_npc:{npc.id}",
                f"significance:{event.significance:.2f}",
            ],
        )
        await self._life_writer.write(fact)
        queued = await self._share_intent.queue(fact.id, npc.id, event.significance)
        if not queued:
            # cooldown race — fact already written, just skip the intent
            return False
        if self._social is not None:
            try:
                await self._social.mark_event_promoted(npc.id, event.ts)
            except Exception as e:
                print(f"[promoter] mark_event_promoted failed: {e}", flush=True)
        print(
            f"[social.promoter] +push {npc.id} sig={event.significance:.2f}: "
            f"{content[:50]}...",
            flush=True,
        )
        return True


def _format_for_aria_pov(npc: NPC, event: NPCEvent) -> str:
    """Render NPC event from Aria's first-person POV."""
    raw = event.content.strip()
    if event.type == "aria_interaction":
        swapped = raw.replace("Aria", "你")
        if swapped == raw:
            return f"{npc.name}{raw}"
        return swapped
    if raw.startswith(npc.name):
        return raw
    return f"{npc.name}{raw}"
