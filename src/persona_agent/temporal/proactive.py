"""Background scheduler for proactive messages.

Periodically checks interaction tracker, decides whether Aria should
reach out based on silence duration, relationship level, current time,
and persona/memory context. If yes, pushes via the appropriate channel.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from persona_agent.conversation.engine import ConversationEngine

from persona_agent.channels.outbound import ChannelRegistry
from persona_agent.temporal.formatter import format_datetime_cn, format_timedelta_cn
from persona_agent.temporal.tracker import InteractionRecord, InteractionTracker


class ProactiveConfig(BaseModel):
    """Configuration for proactive message scheduling."""

    enabled: bool = True
    check_interval_minutes: int = 5
    # relationship_level -> silence hours before consider reaching out
    silence_thresholds: dict[int, int] = Field(
        default_factory=lambda: {1: 72, 2: 24, 3: 6, 4: 3}
    )
    cooldown_hours: float = 12.0
    quiet_hours_start: int = 23  # inclusive
    quiet_hours_end: int = 8  # exclusive

    def silence_threshold_for(self, level: int) -> timedelta:
        hours = self.silence_thresholds.get(level, 24)
        return timedelta(hours=hours)


PROACTIVE_DECISION_PROMPT = """你是 {persona_name}。{current_time_cn}

你已经有 {silence_cn} 没有和对方对话了。

## 你对对方的了解
{memory_facts}

## 你们最近的对话回忆
{recent_episodes}

## 你的关系程度
{relationship_desc}

## 你的目标
{goals}

请判断：你现在应该主动发消息给对方吗？

考虑：
- 现在时间是否合适（不要在对方可能休息、工作的时候打扰）
- 根据关系程度，这个沉默时长算正常还是偏长
- 你是否真的有话想说（关心的事、承诺过的事、想分享的事）
- 不要为了主动而主动，自然最重要

只回复 JSON：
{{"should_send": true, "message": "你想发送的消息（自然口语化，符合你的性格和说话风格）"}}
或
{{"should_send": false, "reason": "为什么现在不合适"}}"""


_MESSAGE_STYLES = [
    {
        "name": "话题跟进",
        "desc": "从对方告诉过你的某件具体事直接切入，问进展或问感受。"
                "不要解释为什么问。就一句，像老朋友。",
        "example": "你那个小游戏字体搞定没？",
    },
    {
        "name": "日常播报",
        "desc": "告诉对方你正在做什么、在哪儿、看到什么。不追加比喻、不延伸。",
        "example": "今晚云太多，看不到星星了，有点可惜。",
    },
    {
        "name": "关心",
        "desc": "基于对方之前说过的具体情况（累、忙、生病等）简短关心一句。"
                "不要加'加油'之类的废话。",
        "example": "你之前说周末要加班，今天能好好休息吗？",
    },
    {
        "name": "无聊闲话",
        "desc": "没什么事，就是想说两句。内容可以是一个念头、一个观察，甚至一个问题。"
                "允许残缺句和语气词。",
        "example": "今天吃了泡面。",
    },
    {
        "name": "随口一问",
        "desc": "抛一个具体的小问题给对方，和他最近提过的事相关。"
                "不是'你好吗'，是有答案的具体问题。",
        "example": "你平时早上起得来吗？我在想要不要早起看日出。",
    },
    {
        "name": "小吐槽",
        "desc": "对你日常里的一件小事发牢骚/感慨/抱怨，拉对方一起吐槽。",
        "example": "望远镜镜头又脏了，怎么也擦不干净。烦。",
    },
]


PROACTIVE_FORCE_PROMPT = """你是 {persona_name}。{current_time_cn}

（测试请求，假装你们已经很久没聊了。）

## 关于对方你知道的一些事
{memory_facts}

## 你们最近的对话回忆
{recent_episodes}

## 你最近发过的几条主动消息（避免重复套路！）
{recent_proactive}

## 你的关系程度
{relationship_desc}

---

**这次请用【{style_name}】的风格**：
{style_desc}

自然示例（理解语气，不要照抄）：
"{style_example}"

---

**死规则，一条都不能违反**：

1. 禁止开头用"嗨/你好/好久不见/刚才我/刚看到/刚刚"
2. 禁止结尾用"你最近怎么样/你还好吗/等你回复"
3. 禁止虚构引号里的"金句""一段话""名言"
4. 禁止用"就像/正如/让我想到"把你的职业和对方的生活做比喻（比如别把'程序员 debug' 比作'天文学家看星'）
5. 禁止跟"最近发过的主动消息"中的任何一条结构/比喻/切入点重复
6. 长度 ≤ 60 个汉字。越短越像活人。
7. 像真人随手发一条微信，不是写作。**口语化、不完整句、语气词都OK**。
8. 只引用一件具体的事，不要罗列好几条

只回复 JSON：
{{"should_send": true, "message": "消息内容"}}"""


class ProactiveScheduler:
    """Background loop that decides and sends proactive messages."""

    def __init__(
        self,
        config: ProactiveConfig,
        tracker: InteractionTracker,
        channel_registry: ChannelRegistry,
        engine: ConversationEngine,
    ):
        self.config = config
        self.tracker = tracker
        self.channels = channel_registry
        self.engine = engine
        self._task: asyncio.Task | None = None
        self._running = False
        # Per-recipient recent proactive messages (to avoid repetition)
        self._recent_proactive: dict[str, list[str]] = {}
        self._max_recent_proactive = 5

    async def start(self) -> None:
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        print(f"[proactive] scheduler started (check every {self.config.check_interval_minutes}min)")

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
        # Initial delay so we don't fire right at startup
        await asyncio.sleep(30)
        while self._running:
            try:
                await self._check_all()
            except Exception as e:
                print(f"[proactive] check error: {e}")
            await asyncio.sleep(self.config.check_interval_minutes * 60)

    async def _check_all(self) -> None:
        now = datetime.now()
        if self._is_quiet_hours(now):
            print(f"[proactive] skipped: quiet hours ({now.hour}:00)")
            return

        for record in self.tracker.all_records():
            try:
                result = await self._maybe_reach_out(record, now)
                if result["status"] == "sent":
                    pass  # already logged
                elif result["status"] == "skipped_silence":
                    pass  # too noisy to log every tick
                else:
                    print(f"[proactive] {result}")
            except Exception as e:
                print(f"[proactive] per-record error ({record.channel}:{record.recipient_id}): {e}")

    async def _maybe_reach_out(
        self,
        record: InteractionRecord,
        now: datetime,
        force: bool = False,
    ) -> dict:
        """Check + act. Returns dict with status for debugging.

        force=True bypasses silence threshold and cooldown (for manual test).
        """
        key = f"{record.channel}:{record.recipient_id}"

        # Silence threshold
        silence = now - record.last_interaction
        threshold = self.config.silence_threshold_for(record.relationship_level)
        if not force and silence < threshold:
            return {
                "key": key, "status": "skipped_silence",
                "silence_hours": silence.total_seconds() / 3600,
                "threshold_hours": threshold.total_seconds() / 3600,
            }

        # Cooldown since last proactive
        if not force and record.last_proactive_sent:
            since_last_proactive = now - record.last_proactive_sent
            if since_last_proactive < timedelta(hours=self.config.cooldown_hours):
                return {
                    "key": key, "status": "skipped_cooldown",
                    "cooldown_remaining_hours": (
                        self.config.cooldown_hours - since_last_proactive.total_seconds() / 3600
                    ),
                }

        # Channel must be available
        channel = self.channels.get(record.channel)
        if channel is None:
            return {"key": key, "status": "no_channel"}

        # Ask LLM (force mode bypasses the "should I?" check and just generates)
        decision = await self._ask_llm(record, silence, now, force=force)
        if not decision:
            return {"key": key, "status": "llm_no_decision"}
        if not decision.get("should_send"):
            return {
                "key": key, "status": "llm_declined",
                "reason": decision.get("reason", ""),
            }

        message = decision.get("message", "").strip()
        if not message:
            return {"key": key, "status": "empty_message"}

        print(f"[proactive] sending to {key} → {message[:60]}")

        try:
            await channel.send_message(record.recipient_id, message)
        except Exception as e:
            print(f"[proactive] send failed: {e}")
            return {"key": key, "status": "send_failed", "error": str(e)}

        self.tracker.record_proactive_sent(record.channel, record.recipient_id)
        await self.tracker.save()

        # Remember for anti-repetition
        recent = self._recent_proactive.setdefault(key, [])
        recent.append(message)
        if len(recent) > self._max_recent_proactive:
            del recent[: len(recent) - self._max_recent_proactive]

        return {"key": key, "status": "sent", "message": message}

    async def trigger_manually(self) -> list[dict]:
        """Manual trigger for all recipients (for testing). Bypasses silence + cooldown."""
        now = datetime.now()
        results = []
        for record in self.tracker.all_records():
            result = await self._maybe_reach_out(record, now, force=True)
            results.append(result)
        return results

    async def _ask_llm(
        self,
        record: InteractionRecord,
        silence: timedelta,
        now: datetime,
        force: bool = False,
    ) -> dict | None:
        persona = self.engine.persona
        rec_key = f"{record.channel}:{record.recipient_id}"
        memory_context = await self.engine.memory.assemble_context(
            "", recipient_key=rec_key
        )

        # Randomize the order/selection of facts so LLM doesn't fixate on top-ranked ones
        all_facts = memory_context.long_term_facts[:15]
        if len(all_facts) > 5:
            picked = random.sample(all_facts, 5)
        else:
            picked = all_facts
        memory_facts = "\n".join(
            f"- {f.content}" for f in picked
        ) or "（暂无记忆）"

        recent_episodes = "\n".join(
            f"- [{ep.timestamp}] {ep.summary}"
            for ep in memory_context.relevant_episodes[:3]
        ) or "（暂无回忆）"

        # Recent proactive messages for this recipient (avoid repetition)
        recent_msgs = self._recent_proactive.get(rec_key, [])
        recent_proactive = "\n".join(
            f"- {m}" for m in recent_msgs[-self._max_recent_proactive:]
        ) or "（这是第一条主动消息）"

        relationship_desc = f"关系等级 {record.relationship_level}"
        for il in persona.relationship.intimacy_levels:
            if il.level == record.relationship_level:
                relationship_desc = f"{il.name}（{il.description}）"
                break

        goals_text = "\n".join(
            f"- {g.description}" for g in persona.goals[:5]
        ) or "（暂无明确目标）"

        if force:
            style = random.choice(_MESSAGE_STYLES)
            prompt = PROACTIVE_FORCE_PROMPT.format(
                persona_name=persona.name,
                current_time_cn=f"现在是 {format_datetime_cn(now)}。",
                memory_facts=memory_facts,
                recent_episodes=recent_episodes,
                recent_proactive=recent_proactive,
                relationship_desc=relationship_desc,
                style_name=style["name"],
                style_desc=style["desc"],
                style_example=style["example"],
            )
            print(f"[proactive] force style: {style['name']}")
        else:
            prompt = PROACTIVE_DECISION_PROMPT.format(
                persona_name=persona.name,
                current_time_cn=f"现在是 {format_datetime_cn(now)}。",
                silence_cn=format_timedelta_cn(silence),
                memory_facts=memory_facts,
                recent_episodes=recent_episodes,
                relationship_desc=relationship_desc,
                goals=goals_text,
            )

        try:
            result = await self.engine.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.9,  # higher variance for more varied messages
            )
        except Exception as e:
            print(f"[proactive] LLM call failed: {e}")
            return None

        # Extract JSON
        text = result.content.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None

        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None

    def _is_quiet_hours(self, now: datetime) -> bool:
        hour = now.hour
        start = self.config.quiet_hours_start
        end = self.config.quiet_hours_end
        if start > end:
            # Wraps midnight: e.g., 23-8
            return hour >= start or hour < end
        return start <= hour < end
