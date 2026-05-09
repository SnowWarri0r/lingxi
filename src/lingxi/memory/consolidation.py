"""Memory consolidation: promotes short-term memories to long-term and generates episodes."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lingxi.providers.base import LLMProvider

from lingxi.memory.base import EpisodeEntry, MemoryEntry, MemoryType
from lingxi.memory.episodic import EpisodicMemory
from lingxi.memory.long_term import LongTermMemory
from lingxi.memory.short_term import ConversationTurn

FACT_EXTRACTION_PROMPT = """从以下对话中提取关键事实和信息。每个事实独立一行。
只提取重要的、值得长期记住的信息（比如用户的偏好、个人信息、重要事件等）。
对每个事实评估重要性（0.0-1.0），格式：[重要性] 事实内容

对话：
{conversation}

提取的事实（如果没有值得记住的，回复"无"）："""

EPISODE_SUMMARY_PROMPT = """请总结以下对话，生成一段简短的回忆摘要。
包括：主要话题、情绪氛围、关键信息。

对话：
{conversation}

请按以下格式回复：
摘要：<一段话总结>
情绪：<对话整体情绪>
话题：<用逗号分隔的关键话题>"""


class MemoryConsolidator:
    """Handles promotion of short-term memories to long-term storage and episode generation."""

    def __init__(
        self,
        long_term: LongTermMemory,
        episodic: EpisodicMemory,
        llm_provider: LLMProvider,
        min_turns: int = 5,
    ):
        self.long_term = long_term
        self.episodic = episodic
        self.llm_provider = llm_provider
        self.min_turns = min_turns

    async def consolidate(
        self,
        turns: list[ConversationTurn],
        embed_fn=None,
        recipient_key: str = "_global",
    ) -> dict:
        """Run full consolidation: extract facts and generate episode summary."""
        if len(turns) < self.min_turns:
            return {"facts_stored": 0, "episode_id": None}

        conversation_text = self._format_conversation(turns)

        facts_stored = await self._extract_and_store_facts(
            conversation_text, embed_fn, recipient_key
        )
        episode_id = await self._generate_episode(
            conversation_text, turns, embed_fn, recipient_key
        )

        return {"facts_stored": facts_stored, "episode_id": episode_id}

    async def _extract_and_store_facts(
        self, conversation_text: str, embed_fn=None, recipient_key: str = "_global"
    ) -> int:
        prompt = FACT_EXTRACTION_PROMPT.format(conversation=conversation_text)
        response = await self.llm_provider.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )

        text = response.content
        if "无" in text and len(text) < 10:
            return 0

        count = 0
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # Parse [importance] fact format
            importance = 0.5
            match = re.match(r"\[?([\d.]+)\]?\s*(.+)", line)
            if match:
                try:
                    importance = float(match.group(1))
                    importance = max(0.0, min(1.0, importance))
                except ValueError:
                    pass
                content = match.group(2)
            else:
                content = line

            entry = MemoryEntry(
                content=content,
                memory_type=MemoryType.LONG_TERM,
                importance=importance,
                metadata={"recipient_key": recipient_key},
            )

            if embed_fn:
                entry.embedding = await embed_fn(content)

            await self.long_term.store(entry)
            count += 1

        return count

    async def _generate_episode(
        self,
        conversation_text: str,
        turns: list[ConversationTurn],
        embed_fn=None,
        recipient_key: str = "_global",
    ) -> str | None:
        prompt = EPISODE_SUMMARY_PROMPT.format(conversation=conversation_text)
        response = await self.llm_provider.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )

        text = response.content
        summary = text
        emotional_tone = "neutral"
        key_topics: list[str] = []

        # Parse structured response
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("摘要：") or line.startswith("摘要:"):
                summary = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("情绪：") or line.startswith("情绪:"):
                emotional_tone = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("话题：") or line.startswith("话题:"):
                topics_str = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                key_topics = [t.strip() for t in topics_str.split("，") if t.strip()]
                if not key_topics:
                    key_topics = [t.strip() for t in topics_str.split(",") if t.strip()]

        episode = EpisodeEntry(
            summary=summary,
            emotional_tone=emotional_tone,
            key_topics=key_topics,
            turn_count=len(turns),
        )

        if embed_fn:
            episode.embedding = await embed_fn(summary)

        # Pass recipient_key if the store supports it
        try:
            return await self.episodic.store_episode(episode, recipient_key=recipient_key)
        except TypeError:
            return await self.episodic.store_episode(episode)

    async def consolidate_day_narrative(
        self,
        narrative: str,
        date_label: str,
        embed_fn=None,
        recipient_key: str = "_global",
    ) -> str | None:
        """Consolidate one day of Aria's life (diary + significant events,
        pre-formatted as `narrative`) into a chroma episode. Returns episode_id.

        This is the bridge between LifeSimulator (which writes daily diary
        to InnerLifeStore) and the memory layer (chroma episodes that get
        retrieved via assemble_context). Without this, diary entries were
        invisible to the conversation engine — Aria couldn't recall what
        she did 3 days ago even when user asked.
        """
        prompt = (
            f"下面是 {date_label} 这一天 Aria 自己的活动记录"
            f"（生活模拟器生成，含日程和当天发生的事）。请压成一段简洁的回忆摘要。"
            f"\n\n记录：\n{narrative}\n\n"
            "请按以下格式回复：\n"
            "摘要：<≤80字，用第三人称描述这一天 Aria 经历了什么，不要照搬原文>\n"
            "情绪：<这一天的整体情绪>\n"
            "话题：<逗号分隔的关键话题>"
        )
        try:
            response = await self.llm_provider.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.4,
            )
        except Exception as e:
            print(f"[consolidate_day] LLM failed for {date_label}: {e}")
            return None

        text = response.content
        summary = text
        emotional_tone = "neutral"
        key_topics: list[str] = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("摘要：") or line.startswith("摘要:"):
                summary = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("情绪：") or line.startswith("情绪:"):
                emotional_tone = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("话题：") or line.startswith("话题:"):
                topics_str = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                key_topics = [t.strip() for t in topics_str.split("，") if t.strip()]
                if not key_topics:
                    key_topics = [t.strip() for t in topics_str.split(",") if t.strip()]

        # Tag the date in summary so it's retrievable as "those days"
        if date_label and date_label not in summary:
            summary = f"[{date_label}] {summary}"

        episode = EpisodeEntry(
            summary=summary,
            emotional_tone=emotional_tone,
            key_topics=key_topics + [f"life:{date_label}"],
            turn_count=0,
        )
        if embed_fn:
            episode.embedding = await embed_fn(summary)

        try:
            return await self.episodic.store_episode(episode, recipient_key=recipient_key)
        except TypeError:
            return await self.episodic.store_episode(episode)

    @staticmethod
    def _format_conversation(turns: list[ConversationTurn]) -> str:
        lines = []
        for turn in turns:
            role_label = "用户" if turn.role == "user" else "助手"
            lines.append(f"{role_label}：{turn.content}")
        return "\n".join(lines)
