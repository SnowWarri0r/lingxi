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
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from lingxi.conversation.engine import ConversationEngine

from lingxi.channels.outbound import ChannelRegistry
from lingxi.fewshot.models import AnnotationTurn
from lingxi.temporal.formatter import format_datetime_cn, format_timedelta_cn
from lingxi.temporal.tracker import InteractionRecord, InteractionTracker


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


PROACTIVE_DECISION_USER = """[这一刻没有对方的新消息——你正在自己一个人，在想要不要主动发一条]

距离上次和他/她聊已经 {silence_cn}。

## 你对他/她的了解
{memory_facts}

## 你们最近聊过什么
{recent_episodes}

## 你最近发过的主动消息（不要重复套路/比喻/切入点）
{recent_proactive}

判断要点：
- 时间是否合适（对方可能在睡/工作，**绝大多数时段不主动**）
- 你**真的**有话想说吗？（关心的具体事、承诺过的事、自己手边发生了什么想分享）
- 为了发而发不如不发

格式仍然按 system prompt 里的 `===META===` 输出：

```
<对外的那一句话，跟 IM 真人发消息一样>
===META===
{{"inner": "<你为什么发这条/这一刻的状态>", "should_send": true}}
```

或者如果不该发：

```

===META===
{{"inner": "<为什么不发>", "should_send": false}}
```

注意 `should_send` 字段。如果对外那条是空的、或 `should_send=false`，系统会跳过发送。"""


# Backwards-compat name (some callers still import the old constant).
PROACTIVE_DECISION_PROMPT = PROACTIVE_DECISION_USER


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
7. 像真人随手发一条聊天消息，不是写作。**口语化、不完整句、语气词都OK**。
8. 只引用一件具体的事，不要罗列好几条

只回复 JSON：
{{"should_send": true, "message": "消息内容"}}"""


# Tokens that signal "I'm continuing/responding to something just said".
# Real openers don't start with these — only replies do.
_RESPONSE_TOKEN_PREFIXES = (
    "嗯", "对", "那", "啊", "哈", "欸", "诶", "哦", "唔", "嗯嗯", "对了",
    "好", "行", "是", "嗯…", "嗯...",
)


def _validate_proactive_opener(message: str) -> str | None:
    """Code-side check that a proactive message looks like an opener, not a journal entry.

    Returns a short reason string when the message should be rejected,
    or None when it passes. Engineered constraints (vs prompt rules):
      1. Doesn't start with a response token (嗯/对/啊…)
      2. Has at least one of: question hook, "你" referencing the user,
         or a direct invitation phrase. Pure interior monologue without
         any user vector reads like Aria journaling at the user.
    """
    if not message:
        return "empty"

    stripped = message.strip()

    # Response-token prefix check — proactive opener shouldn't start with
    # tokens that signal "I'm continuing what was just said".
    for tok in _RESPONSE_TOKEN_PREFIXES:
        if stripped.startswith(tok):
            return f"opens_with_response_token:{tok}"

    # Relational hook check — at least one signal that the message
    # invites engagement instead of being pure self-narration.
    has_question = any(c in stripped for c in "？?")
    has_question_particle = any(p in stripped for p in ("吗", "呢", "哈?", "啥"))
    has_you = "你" in stripped
    has_invitation = any(
        marker in stripped
        for marker in ("一起", "要不要", "陪我", "听我说", "告诉你", "跟你说")
    )
    if not (has_question or has_question_particle or has_you or has_invitation):
        return "no_relational_hook"

    return None


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

        # --- Engineering validation: an opener has shape requirements
        # different from a reply. Reject messages that look like a journal
        # entry posted at the user instead of a conversational opener.
        rejection = _validate_proactive_opener(message)
        if rejection:
            print(f"[proactive] rejected ({rejection}): {message[:80]}")
            return {
                "key": key, "status": "validation_rejected",
                "reason": rejection, "message": message,
            }

        print(f"[proactive] sending to {key} → {message[:60]}")

        # Save AnnotationTurn so user can 👍/👎/✏️ the proactive message.
        # `user_message` is empty since this is unprompted; summarizer falls
        # back to a placeholder when generating context_summary.
        turn_id: str | None = None
        if self.engine.annotation_store is not None:
            try:
                turn_id = str(uuid.uuid4())
                await self.engine.annotation_store.record(AnnotationTurn(
                    turn_id=turn_id,
                    recipient_key=key,
                    user_message="(主动开聊)",
                    inner_thought=decision.get("reason", ""),
                    speech=message,
                ))
            except Exception as e:
                print(f"[proactive] annotation_store.record failed: {e}")
                turn_id = None

        try:
            await channel.send_message(record.recipient_id, message, turn_id=turn_id)
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
            for ep in memory_context.relevant_episodes[:8]
        ) or "（暂无回忆）"

        # Time-bound user state: pull recent USER turns directly. These
        # carry "what's happening in his life right now" (五一假期 / 在加班 /
        # 周末有事 / 感冒了…) which long-term facts can't capture because
        # they're persistent labels, not temporal states.
        user_recent_block = "（暂无最近发言）"
        try:
            recent_turns = await self.engine.memory.short_term.snapshot_for_recipient(rec_key)
            # Last ~8 user-side messages (skip assistant)
            user_msgs = [t for t in recent_turns if t.role == "user"][-8:]
            if user_msgs:
                lines = []
                for t in user_msgs:
                    when = t.timestamp.strftime("%m-%d %H:%M")
                    body = (t.content or "").replace("\n", " ")[:80]
                    lines.append(f"- [{when}] {body}")
                user_recent_block = "\n".join(lines)
        except Exception as e:
            print(f"[proactive] user_recent fetch failed: {e}")

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

        # Reuse the SAME system prompt that reactive uses. This unifies the
        # persona/state/biography/anti-reflex framing across reactive +
        # proactive — fixes the "feels like two different people" gap.
        # Inner state, current activity, mood, biography retrieval all flow
        # through identically.
        try:
            inner_state = None
            if self.engine.inner_life_store is not None:
                inner_state = await self.engine.inner_life_store.load_state()
        except Exception:
            inner_state = None

        try:
            relationship_record = self.engine.interaction_tracker.get_record(
                record.channel, record.recipient_id
            )
            emotion_state = None
            if relationship_record is not None:
                from lingxi.persona.models import EmotionState
                emotion_state = EmotionState(
                    dimensions=dict(relationship_record.emotion_dimensions or {}),
                    narrative_label=relationship_record.emotion_narrative or "",
                )
        except Exception:
            emotion_state = None

        system_prompt = self.engine.prompt_builder.build_system_prompt(
            memory_context=memory_context,
            current_mood=None,
            relationship_level=record.relationship_level,
            current_time=now,
            last_interaction_time=record.last_interaction,
            emotion_state=emotion_state,
            inner_state=inner_state,
            mode="single",
        )

        # Recipient-scoped read path: do NOT mutate the singleton active
        # recipient. A reactive chat for a different user could be awaiting
        # an LLM call right now, and switching the global state would cause
        # its eventual `add_turn("assistant", ...)` to land on the wrong
        # recipient's buffer (cross-tenant data leak). Use the explicit
        # snapshot/compress APIs instead.
        recent_history_msgs: list[dict] = []
        try:
            await self.engine.memory.compress_aged_turns_for(
                rec_key,
                threshold_minutes=self.engine.context_assembler.budget.verbatim_window_minutes,
            )
            _, recent_history_msgs = await self.engine.memory.assemble_history_messages_for(
                rec_key, self.engine.context_assembler
            )
        except Exception as e:
            print(f"[proactive] history fetch failed: {e}")

        if force:
            style = random.choice(_MESSAGE_STYLES)
            user_prompt = (
                f"[这一刻没有对方的新消息——你一个人，在想要不要主动发一条]\n\n"
                f"距离上次聊已经 {format_timedelta_cn(silence)}。\n\n"
                f"## 对方最近发的话（**他此刻的状态/在干啥都在这里**，比长期记忆更重要）\n"
                f"{user_recent_block}\n\n"
                f"## 你最近发过的主动消息（不要重复套路/比喻/切入点）\n{recent_proactive}\n\n"
                f"## 这次试一种语气：【{style['name']}】\n"
                f"{style['desc']}\n"
                f"参考语气（不要照抄）：「{style['example']}」\n\n"
                f"按 system prompt 里的 `===META===` 格式输出。如果**真的**没什么想说，就让对白空、"
                f"`should_send: false`：\n```\n<想说的那一句，IM 短句>\n===META===\n"
                f'{{"should_send": true, "inner": "为什么这一刻想发"}}\n```'
            )
            print(f"[proactive] force style: {style['name']}")
        else:
            user_prompt = (
                f"[这一刻没有对方的新消息——你一个人，在想要不要主动发一条]\n\n"
                f"距离上次聊已经 {format_timedelta_cn(silence)}。\n\n"
                f"## 对方最近发的话（**他此刻的状态/在干啥都在这里**，比长期记忆更重要）\n"
                f"{user_recent_block}\n\n"
                f"## 你最近发过的主动消息（避免重复）\n{recent_proactive}\n\n"
                f"考虑：现在时间合不合适、你**真的**有话说吗（具体事不是闲扯）。"
                f"为了发而发不如不发。\n\n"
                f"按 system prompt 的 `===META===` 格式输出，meta 里加 `should_send`：\n```\n"
                f"<对外那一句，IM 短句>\n===META===\n"
                f'{{"should_send": true, "inner": "为什么发"}}\n```\n\n'
                f"不该发就：\n```\n\n===META===\n"
                f'{{"should_send": false, "inner": "为什么不发"}}\n```'
            )

        # Final messages = recent chat history (so model sees what was
        # actually said today) + the proactive trigger as the latest user msg
        final_messages = recent_history_msgs + [{"role": "user", "content": user_prompt}]

        try:
            result = await self.engine.llm.complete(
                messages=final_messages,
                system=system_prompt,
                max_tokens=600,
                temperature=0.9,
            )
        except Exception as e:
            print(f"[proactive] LLM call failed: {e}")
            return None

        # Parse: speech + ===META=== + json (matching reactive output)
        text = result.content.strip()
        from lingxi.conversation.output_schema import META_DELIMITER
        if META_DELIMITER in text:
            speech_part, _, meta_part = text.partition(META_DELIMITER)
        else:
            # Fallback: maybe model just spat raw JSON (legacy).
            speech_part, meta_part = "", text

        # Clean speech (same regex as reactive)
        from lingxi.conversation.response_cleaner import clean_speech
        message = clean_speech(speech_part.strip())

        # Extract JSON
        match = re.search(r"\{.*\}", meta_part, re.DOTALL)
        if not match:
            return {"should_send": bool(message), "message": message}
        try:
            meta = json.loads(match.group())
        except json.JSONDecodeError:
            return {"should_send": bool(message), "message": message}

        # Honour explicit should_send=false; otherwise infer from speech
        should_send = meta.get("should_send")
        if should_send is None:
            should_send = bool(message)

        return {
            "should_send": bool(should_send),
            "message": message,
            "reason": meta.get("inner", ""),
        }

    def _is_quiet_hours(self, now: datetime) -> bool:
        hour = now.hour
        start = self.config.quiet_hours_start
        end = self.config.quiet_hours_end
        if start > end:
            # Wraps midnight: e.g., 23-8
            return hour >= start or hour < end
        return start <= hour < end
