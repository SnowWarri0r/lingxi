"""Evaluates relationship level progression based on quantitative + LLM signals.

Hybrid approach:
- Quantitative gates set the MAXIMUM possible level (turns, sessions, days, memory)
- LLM judges actual level within those bounds
- Asymmetric: never regresses (relationships are cumulative)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from lingxi.persona.models import PersonaConfig
from lingxi.temporal.tracker import InteractionRecord

if TYPE_CHECKING:
    from lingxi.memory.manager import MemoryManager
    from lingxi.providers.base import LLMProvider


@dataclass
class LevelGate:
    """Quantitative minimums required before a level can be reached."""

    min_turns: int
    min_sessions: int
    min_days_known: int
    min_memory_facts: int


# Default gates: index = level. Level 1 has no requirements.
DEFAULT_GATES: dict[int, LevelGate] = {
    1: LevelGate(min_turns=0, min_sessions=0, min_days_known=0, min_memory_facts=0),
    2: LevelGate(min_turns=20, min_sessions=3, min_days_known=2, min_memory_facts=3),
    3: LevelGate(min_turns=80, min_sessions=8, min_days_known=7, min_memory_facts=8),
    4: LevelGate(min_turns=200, min_sessions=20, min_days_known=21, min_memory_facts=15),
}


RELATIONSHIP_EVAL_PROMPT = """你是 {persona_name}。请评估你和对方的关系是否应该进阶。

## 当前关系
当前等级：{current_level} - {current_name}（{current_desc}）
可选进阶到：{next_level} - {next_name}（{next_desc}）

## 互动统计
总对话轮数：{total_turns}
对话次数（不同 session）：{session_count}
认识天数：{days_known}

## 你对对方的了解
{memory_facts}

## 最近的对话回忆
{recent_episodes}

## 判断标准
- 对方是否展现了信任（分享个人经历、感受、脆弱面）？
- 你们是否有了共同话题和默契？
- 对话深度如何（深入交流 vs 表面寒暄）？
- 关系发展是否自然，不要太快也不要错过应该的进展

请只回复 JSON：
{{"new_level": <{current_level} 或 {next_level}>, "reason": "简短原因"}}"""


class RelationshipEvaluator:
    """Decides when relationship_level should advance."""

    def __init__(
        self,
        persona: PersonaConfig,
        llm_provider: LLMProvider,
        gates: dict[int, LevelGate] | None = None,
    ):
        self.persona = persona
        self.llm = llm_provider
        self.gates = gates or DEFAULT_GATES
        levels = [il.level for il in persona.relationship.intimacy_levels]
        self._max_level = max(levels) if levels else 4

    def _level_info(self, level: int) -> tuple[str, str]:
        """Returns (name, description) for a level."""
        for il in self.persona.relationship.intimacy_levels:
            if il.level == level:
                return il.name, il.description
        return f"等级{level}", ""

    def compute_max_allowed_level(
        self,
        record: InteractionRecord,
        memory_fact_count: int,
    ) -> int:
        """Pure quantitative gate: highest level this user qualifies for."""
        now = datetime.now()
        days_known = 0
        if record.first_interaction:
            days_known = (now - record.first_interaction).days

        max_level = 1
        for level in range(2, self._max_level + 1):
            gate = self.gates.get(level)
            if gate is None:
                continue
            if (
                record.total_turns >= gate.min_turns
                and record.session_count >= gate.min_sessions
                and days_known >= gate.min_days_known
                and memory_fact_count >= gate.min_memory_facts
            ):
                max_level = level
            else:
                break  # Levels are ordered; can't reach N+1 if can't reach N
        return max_level

    async def evaluate(
        self,
        record: InteractionRecord,
        memory_manager: MemoryManager,
    ) -> int:
        """Hybrid evaluation. Returns the new level (>= current_level)."""
        current_level = record.relationship_level
        memory_stats = memory_manager.get_stats()
        fact_count = memory_stats.get("long_term_entries", 0)

        max_allowed = self.compute_max_allowed_level(record, fact_count)

        # Already at or above the gate ceiling
        if max_allowed <= current_level:
            return current_level

        # Gate allows upgrade. Ask LLM whether it's appropriate.
        next_level = current_level + 1
        new_level = await self._ask_llm(record, memory_manager, current_level, next_level)

        # Clamp: never below current, never above gate
        return max(current_level, min(new_level, max_allowed))

    async def _ask_llm(
        self,
        record: InteractionRecord,
        memory_manager: MemoryManager,
        current_level: int,
        next_level: int,
    ) -> int:
        memory_context = await memory_manager.assemble_context("")

        memory_facts = "\n".join(
            f"- {f.content}" for f in memory_context.long_term_facts[:10]
        ) or "（暂无记忆）"

        recent_episodes = "\n".join(
            f"- [{ep.timestamp}] {ep.summary}"
            for ep in memory_context.relevant_episodes[:3]
        ) or "（暂无回忆）"

        days_known = 0
        if record.first_interaction:
            days_known = (datetime.now() - record.first_interaction).days

        current_name, current_desc = self._level_info(current_level)
        next_name, next_desc = self._level_info(next_level)

        prompt = RELATIONSHIP_EVAL_PROMPT.format(
            persona_name=self.persona.name,
            current_level=current_level,
            current_name=current_name,
            current_desc=current_desc,
            next_level=next_level,
            next_name=next_name,
            next_desc=next_desc,
            total_turns=record.total_turns,
            session_count=record.session_count,
            days_known=days_known,
            memory_facts=memory_facts,
            recent_episodes=recent_episodes,
        )

        try:
            result = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
            )
        except Exception:
            return current_level

        text = result.content.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return current_level

        try:
            data = json.loads(match.group())
            new_level = int(data.get("new_level", current_level))
            return new_level
        except (json.JSONDecodeError, ValueError, TypeError):
            return current_level
