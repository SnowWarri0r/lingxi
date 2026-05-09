"""Background reflection loop: agent rumination and insight generation.

When the agent is idle (no active conversation), it periodically "thinks about"
recent conversations. This mimics how humans process their day in quiet moments -
revisiting conversations, connecting dots, forming new insights.

The insights are stored as high-importance long-term memories so they
surface naturally in future conversations. This makes the agent feel like
it has its own inner life, not just responding reactively.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from lingxi.conversation.engine import ConversationEngine

from lingxi.temporal.tracker import InteractionRecord, InteractionTracker


class ReflectionConfig(BaseModel):
    enabled: bool = True
    check_interval_minutes: int = 30
    # Minimum idle time after last interaction before reflecting (avoid during active chat)
    min_idle_minutes: int = 15
    # Max age of last interaction - don't reflect on very old recipients
    max_age_days: int = 14
    # Cooldown between reflections for the same recipient
    reflection_cooldown_hours: float = 6.0


REFLECTION_PROMPT = """你是 {persona_name}。现在是你独处的时间，你在回想最近和 {recipient_desc} 的对话。

## 关于对方你知道的
{memory_facts}

## 最近的对话回忆
{recent_episodes}

## 最近一次对话以来的内容（如果有）
{recent_turns}

请像人一样回想和思考：
- 有什么让你印象深刻的？
- 你对对方有什么新的理解或想法？
- 有没有想要在下次见面时问候或跟进的事？
- 对方的状态让你有什么感受？

提炼出 1-3 条重要的新见解（不是复述已有事实，而是你自己的思考和理解）。
如果没有值得记下的新东西，回复 "no_insights"。

请用 JSON 回复：
{{"insights": [
  {{"content": "一条新见解或思考", "importance": 0.7}},
  ...
]}}

或者：
{{"insights": []}}"""


BIOGRAPHY_ADDITION_PROMPT = """你是 {persona_name}。你在独处时回想刚刚和 {recipient_desc} 的对话。

## 对话片段
{recent_turns}

在这次聊天里，**你自己**有没有发生过一件可以写进个人记忆的**具体小事**？

**重要区分**：
- ✅ 要写**事件**（发生过的、有时间地点感官细节的）
  例："今晚跟他聊到练字，翻出抽屉里那本13岁的日记，字真的歪歪扭扭，看着看着笑出来"
  例："他说在加班，我下意识想给他冲一杯咖啡的冲动，然后才意识到我们隔着屏幕"
- ❌ 不要写**顿悟 / 总结 / 心理分析**
  反例："我发现自己在累的时候特别容易想起小时候的温暖记忆"（这是治疗笔记不是事件）
  反例："我意识到自己对某些话题特别敏感"（这是分析不是事件）

要求：
- 必须是**具体发生过的**（动作 / 看到的 / 触发那一刻）
- **只记属于你自己的事**，不是关于对方的观察
- 多数时候**没有就没有**，宁缺勿滥
- 用第一人称短句（≤40 字，真人日记腔，不写散文）

请用 JSON 回复：
{{"event": {{"content": "<一句具体发生过的事>", "tags": ["tag1", "tag2"]}}}}

如果没有值得记的具体事件（包括只有"顿悟"没有事件的情况），回复：
{{"event": null}}"""


class ReflectionLoop:
    """Background task that periodically reflects on recent conversations."""

    def __init__(
        self,
        config: ReflectionConfig,
        tracker: InteractionTracker,
        engine: ConversationEngine,
    ):
        self.config = config
        self.tracker = tracker
        self.engine = engine
        self._task: asyncio.Task | None = None
        self._running = False
        # Per-recipient last reflection time
        self._last_reflection: dict[str, datetime] = {}

    async def start(self) -> None:
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        print(f"[reflection] loop started (every {self.config.check_interval_minutes}min)")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        # Initial delay
        await asyncio.sleep(60)
        while self._running:
            try:
                await self._reflect_once()
            except Exception as e:
                print(f"[reflection] error: {e}")
            await asyncio.sleep(self.config.check_interval_minutes * 60)

    async def _reflect_once(self) -> None:
        now = datetime.now()
        for record in self.tracker.all_records():
            try:
                await self._maybe_reflect(record, now)
            except Exception as e:
                print(f"[reflection] per-record error: {e}")

    async def _maybe_reflect(self, record: InteractionRecord, now: datetime) -> None:
        key = f"{record.channel}:{record.recipient_id}"

        # Not yet idle
        idle = now - record.last_interaction
        if idle < timedelta(minutes=self.config.min_idle_minutes):
            return

        # Too old
        if idle > timedelta(days=self.config.max_age_days):
            return

        # Cooldown per recipient
        last = self._last_reflection.get(key)
        if last and (now - last) < timedelta(hours=self.config.reflection_cooldown_hours):
            return

        insights = await self._generate_insights(record)
        if not insights:
            self._last_reflection[key] = now
            return

        # Store insights as long-term memories
        for ins in insights:
            content = ins.get("content", "").strip()
            if not content:
                continue
            importance = float(ins.get("importance", 0.7))
            importance = max(0.0, min(1.0, importance))
            # Tag as reflection so we can distinguish
            try:
                rec_key = f"{record.channel}:{record.recipient_id}"
                await self.engine.memory.add_fact(
                    f"[反思] {content}",
                    importance=importance,
                    tags=["reflection", record.channel, record.recipient_id],
                    recipient_key=rec_key,
                )
            except Exception as e:
                print(f"[reflection] store insight failed: {e}")

        print(f"[reflection] {key}: stored {len(insights)} insights")

        # Also: did Aria herself experience anything worth adding to biography?
        try:
            await self._maybe_add_biography_event(record)
        except Exception as e:
            print(f"[reflection] biography add failed: {e}")

        self._last_reflection[key] = now

    async def _maybe_add_biography_event(self, record: InteractionRecord) -> None:
        """Ask the LLM whether this session produced a biographical moment for Aria herself."""
        if self.engine.biography_retriever is None:
            return
        turns = self.engine.memory.short_term.get_history(last_n=12)
        if len(turns) < 4:
            return  # too little substance to reflect on

        lines = []
        for t in turns:
            role = "对方" if t.role == "user" else "我"
            lines.append(f"{role}：{t.content[:200]}")
        recent_turns = "\n".join(lines)

        recipient_desc = f"{record.channel} 上的对方（已认识 {record.total_turns} 轮）"
        prompt = BIOGRAPHY_ADDITION_PROMPT.format(
            persona_name=self.engine.persona.name,
            recipient_desc=recipient_desc,
            recent_turns=recent_turns,
        )

        try:
            result = await self.engine.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.6,
            )
        except Exception as e:
            print(f"[reflection] biography LLM failed: {e}")
            return

        text = result.content.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return
        event_dict = data.get("event")
        if not event_dict or not isinstance(event_dict, dict):
            return
        content = (event_dict.get("content") or "").strip()
        if not content:
            return

        from lingxi.persona.models import LifeEvent

        # Age stamp: current persona age (biographies lean on age as a signal)
        age = self.engine.persona.identity.age or 28
        tags = event_dict.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        event = LifeEvent(age=age, content=content, tags=[str(t) for t in tags][:5])

        rec_key = f"{record.channel}:{record.recipient_id}"
        await self.engine.add_biography_event(event, recipient_key=rec_key, source="reflection")
        print(f"[biography] grew: {age}岁·{content[:40]}")

    async def _generate_insights(self, record: InteractionRecord) -> list[dict]:
        # Temporarily switch short-term to this recipient's context
        recipient_key = f"{record.channel}:{record.recipient_id}"
        await self.engine.memory.short_term.switch_recipient(recipient_key)

        memory_context = await self.engine.memory.assemble_context("")

        memory_facts = "\n".join(
            f"- {f.content}" for f in memory_context.long_term_facts[:10]
        ) or "（暂无记忆）"

        recent_episodes = "\n".join(
            f"- [{ep.timestamp.strftime('%Y-%m-%d %H:%M')}] {ep.summary}"
            for ep in memory_context.relevant_episodes[:3]
        ) or "（暂无回忆）"

        recent_turns = ""
        turns = self.engine.memory.short_term.get_history(last_n=10)
        if turns:
            lines = []
            for t in turns:
                role = "对方" if t.role == "user" else "我"
                snippet = t.content[:150]
                lines.append(f"{role}：{snippet}")
            recent_turns = "\n".join(lines)

        recipient_desc = f"{record.channel} 上的对方（已认识 {record.total_turns} 轮）"

        prompt = REFLECTION_PROMPT.format(
            persona_name=self.engine.persona.name,
            recipient_desc=recipient_desc,
            memory_facts=memory_facts,
            recent_episodes=recent_episodes,
            recent_turns=recent_turns or "（无）",
        )

        try:
            result = await self.engine.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7,
            )
        except Exception as e:
            print(f"[reflection] LLM call failed: {e}")
            return []

        text = result.content.strip()
        if "no_insights" in text.lower():
            return []

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return []

        try:
            data = json.loads(match.group())
            insights = data.get("insights", [])
            if not isinstance(insights, list):
                return []
            return insights
        except (json.JSONDecodeError, ValueError):
            return []
