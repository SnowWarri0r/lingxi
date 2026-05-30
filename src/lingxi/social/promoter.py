"""Push significant NPC events into Aria's inner_state.recent_events.

When the scheduler writes an event with significance ≥ threshold,
this hook flips it from background-knowledge (pull, rendered in social
section) to foreground (push, eligible for proactive opener).

Two safeguards against firehose:
- per-NPC cooldown: once a 24h window, max 1 push per NPC
- promoted_to_aria flag on the NPCEvent — idempotent if hook re-fires

The push writes a LifeEvent with wants_to_share=True so the existing
proactive_mode filter (recent_events with wants_to_share) surfaces it
for one outgoing message and then clears.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from lingxi.inner_life.models import LifeEvent
from lingxi.inner_life.store import InnerLifeStore
from lingxi.social.models import NPC, NPCEvent
from lingxi.social.store import SocialStore


DEFAULT_THRESHOLD = 0.6
COOLDOWN = timedelta(hours=24)


class SocialPromoter:
    """Promote NPC events to Aria's recent_events with cooldown + threshold."""

    def __init__(
        self,
        inner_store: InnerLifeStore,
        social_store: SocialStore,
        data_dir: Path | str,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        cooldown: timedelta = COOLDOWN,
    ):
        self._inner = inner_store
        self._social = social_store
        self._threshold = threshold
        self._cooldown = cooldown
        self._cooldown_path = Path(data_dir) / "social" / "recent_promotions.json"
        self._lock = asyncio.Lock()

    async def maybe_promote(self, npc: NPC, event: NPCEvent) -> bool:
        """Called from scheduler's on_event_written hook.

        Returns True if the event was promoted (for logging/tests).
        """
        if event.significance < self._threshold:
            return False
        if event.promoted_to_aria:
            return False
        if await self._is_in_cooldown(npc.id):
            return False

        content = _format_for_inner_state(npc, event)
        life_event = LifeEvent(
            content=content,
            significance=event.significance,
            wants_to_share=True,
        )

        def _mutate(state):
            state.recent_events.insert(0, life_event)
            state.recent_events = state.recent_events[:30]

        await self._inner.update_state(_mutate)
        await self._social.mark_event_promoted(npc.id, event.ts)
        await self._record_promotion(npc.id, datetime.now())
        print(
            f"[social.promoter] +push {npc.id} sig={event.significance:.2f}: "
            f"{content[:50]}...",
            flush=True,
        )
        return True

    async def _is_in_cooldown(self, npc_id: str) -> bool:
        recent = await self._load_recent_promotions()
        last = recent.get(npc_id)
        if last is None:
            return False
        try:
            last_dt = datetime.fromisoformat(last)
        except Exception:
            return False
        return (datetime.now() - last_dt) < self._cooldown

    async def _record_promotion(self, npc_id: str, ts: datetime) -> None:
        async with self._lock:
            recent = await self._load_recent_promotions()
            recent[npc_id] = ts.isoformat()
            # Drop expired entries — keeps file small
            cutoff = datetime.now() - self._cooldown
            recent = {
                k: v for k, v in recent.items()
                if _parse_or_min(v) >= cutoff
            }
            await self._save_recent_promotions(recent)

    async def _load_recent_promotions(self) -> dict[str, str]:
        if not self._cooldown_path.exists():
            return {}
        try:
            data = await asyncio.to_thread(
                lambda: json.loads(self._cooldown_path.read_text(encoding="utf-8"))
            )
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    async def _save_recent_promotions(self, data: dict[str, str]) -> None:
        self._cooldown_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cooldown_path.with_suffix(".tmp")

        def _write():
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.rename(self._cooldown_path)

        await asyncio.to_thread(_write)


def _parse_or_min(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.min


def _format_for_inner_state(npc: NPC, event: NPCEvent) -> str:
    """Render NPC event from Aria's first-person POV.

    - life: prefix with NPC name (it's something happening to them)
    - aria_interaction: substitute "Aria" → "你" so it reads as Aria's memory
    """
    raw = event.content.strip()
    if event.type == "aria_interaction":
        # Generator was instructed to use "Aria" by name; swap to second-person
        swapped = raw.replace("Aria", "你")
        if swapped == raw:
            # Generator didn't use the name explicitly — prefix the NPC name
            return f"{npc.name}{raw}"
        return swapped
    # life event — prefix with NPC name as subject if not already there
    if raw.startswith(npc.name):
        return raw
    return f"{npc.name}{raw}"
