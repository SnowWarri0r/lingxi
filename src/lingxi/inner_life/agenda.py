"""AgendaEngine: things Aria wants to say/ask specific recipients.

Agenda items come from:
- Life events she wants to share ("今天看到流星，想告诉你")
- Follow-ups from previous conversations ("上次他说要考试，结果怎样了")
- Concerns she has ("他熬夜太多，想提醒")
- Ideas/questions she wants to raise

Unlike the old proactive scheduler (which triggered by silence timer),
agenda-driven proactive messaging only fires when there's real substance.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from lingxi.inner_life.models import AgendaItem, AgendaKind, InnerState
from lingxi.inner_life.store import InnerLifeStore

if TYPE_CHECKING:
    from lingxi.conversation.engine import ConversationEngine


AGENDA_GENERATION_PROMPT = """你是 {persona_name}。下面是你刚刚发生的事情，以及你对 {recipient_label} 的了解。

判断这件事里有没有你**真的想跟 {recipient_label} 说的**（不是客套，是你自然会想到的）。

## 你刚发生的事
{event_content}

## 对方的情况（你的主观印象）
{subjective_blurb}

## 你的目标
{goals}

如果你想告诉他/问他/关心他，用 JSON 输出（如果没有就回复 `{{"items": []}}`）：
{{
  "items": [
    {{
      "kind": "share|follow_up|concern|question|invitation",
      "content": "你想说的内容（短，1-2 句）",
      "priority": 0.0-1.0
    }}
  ]
}}

不要强行编。真的没有就返回空数组。"""


class AgendaEngine:
    """Manages Aria's agenda of things to bring up per-recipient."""

    def __init__(self, store: InnerLifeStore):
        self.store = store

    async def generate_from_event(
        self,
        event_content: str,
        recipient_key: str,
        recipient_label: str,
        subjective_blurb: str,
        persona_name: str,
        goals: str,
        llm,
    ) -> list[AgendaItem]:
        """After a life event, decide if it becomes an agenda item for someone."""
        prompt = AGENDA_GENERATION_PROMPT.format(
            persona_name=persona_name,
            recipient_label=recipient_label,
            event_content=event_content,
            subjective_blurb=subjective_blurb,
            goals=goals,
        )

        try:
            result = await llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.7,
            )
        except Exception:
            return []

        match = re.search(r"\{[\s\S]*\}", result.content)
        if not match:
            return []

        try:
            data = json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            return []

        items: list[AgendaItem] = []
        for raw in data.get("items", []):
            try:
                kind_str = str(raw.get("kind", "share")).lower()
                try:
                    kind = AgendaKind(kind_str)
                except ValueError:
                    kind = AgendaKind.SHARE
                content = str(raw.get("content", "")).strip()
                if not content:
                    continue
                priority = float(raw.get("priority", 0.5))
                priority = max(0.0, min(1.0, priority))
                items.append(AgendaItem(
                    recipient_key=recipient_key,
                    kind=kind,
                    content=content,
                    priority=priority,
                    expires_at=datetime.now() + timedelta(days=2),
                    source=f"event",
                ))
            except Exception:
                continue

        if items:
            # Merge into existing agenda
            existing = await self.store.load_agenda(recipient_key)
            existing.extend(items)
            await self.store.save_agenda(recipient_key, existing)

        return items

    async def top_pending(
        self,
        recipient_key: str,
        limit: int = 5,
    ) -> list[AgendaItem]:
        """Return pending (undelivered, not expired) items sorted by priority."""
        items = await self.store.load_agenda(recipient_key)
        now = datetime.now()
        pending = [
            i for i in items
            if not i.delivered
            and (i.expires_at is None or i.expires_at > now)
        ]
        pending.sort(key=lambda i: i.priority, reverse=True)
        return pending[:limit]

    async def mark_delivered(
        self,
        recipient_key: str,
        item_ids: list[str],
    ) -> None:
        items = await self.store.load_agenda(recipient_key)
        now = datetime.now()
        for i in items:
            if i.id in item_ids:
                i.delivered = True
                i.delivered_at = now
        await self.store.save_agenda(recipient_key, items)

    async def prune_expired(self, recipient_key: str) -> int:
        """Remove expired/delivered items older than 7 days."""
        items = await self.store.load_agenda(recipient_key)
        cutoff = datetime.now() - timedelta(days=7)
        kept = [
            i for i in items
            if not (i.delivered and i.delivered_at and i.delivered_at < cutoff)
            and (i.expires_at is None or i.expires_at > datetime.now() - timedelta(days=3))
        ]
        removed = len(items) - len(kept)
        if removed > 0:
            await self.store.save_agenda(recipient_key, kept)
        return removed
