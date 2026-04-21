"""SubjectiveLayer: Aria's subjective view of each specific recipient.

Unlike factual memory ("he is a programmer"), this is Aria's opinions/feelings
about them ("他很认真但老加班, 让我有点担心").

Updated periodically via LLM reflection on recent conversations.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import TYPE_CHECKING

from persona_agent.inner_life.models import (
    Impression,
    SubjectiveView,
)
from persona_agent.inner_life.store import InnerLifeStore

if TYPE_CHECKING:
    from persona_agent.conversation.engine import ConversationEngine


SUBJECTIVE_UPDATE_PROMPT = """你是 {persona_name}。你在反思你和对方最近的互动，更新你对他的**主观印象**。

## 关于对方的事实
{facts}

## 最近对话 ({turn_count} 轮)
{recent_conversation}

## 你之前的印象
{prior_view}

请更新你对他的主观印象 - 不是罗列事实，而是**你私下会怎么想这个人**。
- impressions: 3-5 个形容词/短句，像你跟朋友描述这个人时会用的
- worries: 你在替他担心什么（如果有）
- appreciations: 你欣赏他哪些地方
- current_stance: 你对他的整体态度 (比如 'friendly-distant', 'warm-caring', 'cautious-curious', 'fond', 'protective')
- relationship_feeling: 一段话（50字内），你私下怎么感觉你们俩的关系

保持真实——如果你对他的印象没变，保留旧的；如果对话里有新观察，更新。

只回复 JSON：
{{
  "impressions": ["..."],
  "worries": ["..."],
  "appreciations": ["..."],
  "current_stance": "...",
  "relationship_feeling": "..."
}}"""


class SubjectiveLayer:
    """Reads and updates Aria's subjective view of each recipient."""

    def __init__(self, store: InnerLifeStore):
        self.store = store

    async def get(self, recipient_key: str) -> SubjectiveView:
        return await self.store.load_subjective(recipient_key)

    async def update_from_conversation(
        self,
        recipient_key: str,
        recipient_label: str,
        persona_name: str,
        facts_blurb: str,
        recent_conversation: str,
        turn_count: int,
        llm,
    ) -> SubjectiveView:
        """After a conversation session, LLM updates the subjective view."""
        prior = await self.get(recipient_key)
        prior_blurb = self._render_prior(prior) if prior.impressions else "（还没有具体印象）"

        prompt = SUBJECTIVE_UPDATE_PROMPT.format(
            persona_name=persona_name,
            facts=facts_blurb,
            recent_conversation=recent_conversation,
            turn_count=turn_count,
            prior_view=prior_blurb,
        )

        try:
            result = await llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.6,
            )
        except Exception:
            return prior

        match = re.search(r"\{[\s\S]*\}", result.content)
        if not match:
            return prior

        try:
            data = json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            return prior

        now = datetime.now()

        impressions_raw = data.get("impressions", [])
        impressions: list[Impression] = []
        for s in impressions_raw[:8]:
            if isinstance(s, str) and s.strip():
                impressions.append(Impression(
                    content=s.strip()[:80],
                    confidence=0.6,
                    last_reinforced=now,
                ))

        worries = [str(w).strip()[:100] for w in data.get("worries", [])[:5] if str(w).strip()]
        appreciations = [str(a).strip()[:100] for a in data.get("appreciations", [])[:5] if str(a).strip()]
        stance = str(data.get("current_stance", prior.current_stance))[:50]
        feeling = str(data.get("relationship_feeling", prior.relationship_feeling))[:300]

        updated = SubjectiveView(
            recipient_key=recipient_key,
            impressions=impressions if impressions else prior.impressions,
            worries=worries,
            appreciations=appreciations,
            current_stance=stance or "friendly",
            relationship_feeling=feeling,
            last_updated=now,
        )
        await self.store.save_subjective(updated)
        return updated

    @staticmethod
    def _render_prior(view: SubjectiveView) -> str:
        lines = []
        if view.impressions:
            lines.append("印象：" + "、".join(i.content for i in view.impressions[:5]))
        if view.worries:
            lines.append("担心：" + "；".join(view.worries[:3]))
        if view.appreciations:
            lines.append("欣赏：" + "；".join(view.appreciations[:3]))
        if view.relationship_feeling:
            lines.append("关系：" + view.relationship_feeling)
        return "\n".join(lines)

    @staticmethod
    def render_for_prompt(view: SubjectiveView) -> str:
        """Render a concise block for injection into system prompt."""
        if not view.impressions and not view.relationship_feeling:
            return ""

        lines = ["## 你对这个人的主观感受（私下的想法，不要直接念出来）"]
        if view.relationship_feeling:
            lines.append(view.relationship_feeling)
        if view.impressions:
            lines.append("印象词：" + "、".join(i.content for i in view.impressions[:5]))
        if view.worries:
            lines.append("你心里担心：" + "；".join(view.worries[:3]))
        if view.current_stance:
            lines.append(f"你的态度基调：{view.current_stance}")
        return "\n".join(lines)
