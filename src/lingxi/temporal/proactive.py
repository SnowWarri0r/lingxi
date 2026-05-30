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
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from lingxi.conversation.engine import ConversationEngine

from lingxi.channels.outbound import ChannelRegistry
from lingxi.facts.models import Fact
from lingxi.facts.retriever import FactRetriever
from lingxi.fewshot.models import AnnotationTurn
from lingxi.proactive.share_intent import ShareIntent, ShareIntentStore
from lingxi.temporal.formatter import format_timedelta_cn
from lingxi.temporal.tracker import InteractionRecord, InteractionTracker


async def find_pending_share(
    intent_store: ShareIntentStore,
    retriever: FactRetriever,
) -> tuple[ShareIntent, Fact] | None:
    """Pick the highest-significance pending share intent whose fact still exists.

    Cleans up stale intents (intent referencing a missing fact) along the way.
    Returns (intent, fact) or None if queue is empty / all intents are stale.
    """
    intents = await intent_store.pending()
    if not intents:
        return None
    intents.sort(key=lambda i: -i.significance)
    for intent in intents:
        fact = await retriever.fetch_by_id(intent.fact_id)
        if fact is not None:
            return (intent, fact)
        await intent_store.consume(intent.fact_id)
    return None


class ProactiveConfig(BaseModel):
    """Configuration for proactive message scheduling."""

    enabled: bool = True
    check_interval_minutes: int = 5
    # relationship_level -> silence hours before consider reaching out
    silence_thresholds: dict[int, int] = Field(
        default_factory=lambda: {1: 72, 2: 24, 3: 6, 4: 3}
    )
    cooldown_hours: float = 12.0
    # Hard cap on proactive messages sent without user reply in between.
    # IM register: a real friend who notices silence might reach out once,
    # maybe twice tops. After that you wait for them. Without this cap, a
    # 24h user silence at cooldown=3h yields 8 proactives — stalkery.
    max_consecutive_proactive: int = 2
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
                "不要解释为什么问，就一句。",
        "example": "你那个小游戏字体搞定没？",
    },
    {
        "name": "日常播报",
        "desc": "告诉对方你正在做什么、在哪儿、看到什么。具体一件事，不追加比喻不延伸。",
        "example": "刚发现冰箱酸奶过期一周了 我居然都没察觉。",
    },
    {
        "name": "关心",
        "desc": "基于对方之前说过的具体事情况简短关心。**问具体的细节**，不要"
                "'今天还好吗 / 你怎么样了'这种空泛 check-in。",
        "example": "奶奶今天吃得下东西吗",
    },
    {
        "name": "无聊闲话",
        "desc": "没什么事，就是想说两句。内容可以是一个念头、一个观察、甚至一个问题。"
                "允许残缺句和语气词。",
        "example": "今天吃了泡面。",
    },
    {
        "name": "随口一问",
        "desc": "抛一个具体的小问题给对方，和他最近提过的事相关。"
                "不是'你好吗'这种宽泛的，是有答案的具体问题。",
        "example": "你那个 IDE 主题用的什么 上次截图看着挺舒服。",
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


def _content_overlap(a: str, b: str) -> int:
    """Largest contiguous run of 4+ chars from `a` appearing in `b`.

    Used to detect whether an episode summary or event was already
    voiced in a recent proactive message — overlap≥4 means likely
    already-spoken; the new message would just repeat it.
    """
    if not a or not b:
        return 0
    best = 0
    n = len(a)
    for i in range(n):
        j = i + 4
        if j > n:
            break
        if a[i:j] not in b:
            continue
        k = j
        while k < n and a[i:k + 1] in b:
            k += 1
        run = k - i
        if run > best:
            best = run
    return best


# Tokens that signal "I'm continuing/responding to something just said".
# Real openers don't start with these — only replies do.
_RESPONSE_TOKEN_PREFIXES = (
    "嗯", "对", "那", "啊", "哈", "欸", "诶", "哦", "唔", "嗯嗯", "对了",
    "好", "行", "是", "嗯…", "嗯...",
)


def _validate_proactive_opener(message: str) -> str | None:
    """Code-side check that a proactive message looks like an opener, not a reply.

    Constraints:
    1. Must NOT start with a response token (嗯/对/啊/哈…) — signals
       "I'm continuing what was just said".
    2. Must NOT be a self-report ("今天就一直在刷手机" / "一天都在写
       东西") without any outward-facing element — that reads as
       answering an unasked "你今天怎么样?" question. Reactive-shape
       in proactive context.
    """
    if not message:
        return "empty"

    stripped = message.strip()

    for tok in _RESPONSE_TOKEN_PREFIXES:
        if stripped.startswith(tok):
            return f"opens_with_response_token:{tok}"

    if _looks_like_self_report_opener(stripped):
        return "self_report_opener"

    if _looks_like_phatic_checkin(stripped):
        return "phatic_checkin"

    return None


_PHATIC_CHECKIN_REGEXES: tuple[str, ...] = (
    # All match the ENTIRE stripped message — only blocks when the
    # whole message is just one of these (no other content).
    r"^你?(现在)?睡了吗[?？]?$",
    r"^你?睡了没[?？]?$",
    r"^你?(现在)?回家了吗[?？]?$",
    r"^你?回家了没[?？]?$",
    r"^你?(现在)?下班了吗[?？]?$",
    r"^你?下班了没[?？]?$",
    r"^你?吃饭了吗[?？]?$",
    r"^你?吃了吗[?？]?$",
    r"^你?到家了吗[?？]?$",
    r"^你?到了吗[?？]?$",
    r"^你?在干嘛(呢)?[?？]?$",
)


def _looks_like_phatic_checkin(message: str) -> bool:
    """True if message is a bare check-in question with no specific hook.

    'X 了吗' style yes/no questions read as filler when sent without
    a concrete reason. Production trace at 23:00: '你现在睡了吗' →
    reads like AI fishing for engagement. Block at validator unless
    the message carries additional content suggesting context.
    """
    if not message:
        return False
    return any(re.match(pat, message) for pat in _PHATIC_CHECKIN_REGEXES)


# Self-report opener patterns. User reported: "今天就一直在刷手机" —
# reads like Aria answering "你今天干了啥". Conservative regex; matches
# only when message has no outward-facing element (你/吗/?/呢).
_SELF_REPORT_OPENER_PREFIXES: tuple[str, ...] = (
    "今天就", "今天一直", "今天没怎么", "今天没",
    "一天都在", "一天都没", "我今天", "我一天",
)


def _looks_like_self_report_opener(message: str) -> bool:
    """True if message reads as a self-report answer to an unasked question."""
    if not message:
        return False
    if not any(message.startswith(p) for p in _SELF_REPORT_OPENER_PREFIXES):
        return False
    # If there IS something outward-facing (a question, "你", emoji-like),
    # treat it as a valid opener that happens to start with self-state
    if any(c in message for c in ("?", "？", "吗", "呢")):
        return False
    if "你" in message:
        return False
    return True


class ProactiveScheduler:
    """Background loop that decides and sends proactive messages."""

    def __init__(
        self,
        config: ProactiveConfig,
        tracker: InteractionTracker,
        channel_registry: ChannelRegistry,
        engine: ConversationEngine,
        data_dir: str | None = None,
        share_intent_store: ShareIntentStore | None = None,
        fact_retriever: FactRetriever | None = None,
    ):
        self.config = config
        self.tracker = tracker
        self.channels = channel_registry
        self.engine = engine
        self.share_intent_store = share_intent_store
        self.fact_retriever = fact_retriever
        self._task: asyncio.Task | None = None
        self._running = False
        # Per-recipient recent proactive messages (to avoid repetition).
        # Persisted to disk — process restart MUST NOT wipe it, otherwise
        # Aria forgets what she just sent and re-pitches the same hook.
        # Production trace (2026-05-19): "昨晚又看了一遍那个电影" sent at
        # 08:04 and again at 11:05 to same recipient because in-memory
        # dict was cleared by intervening service restart.
        self._recent_proactive: dict[str, list[str]] = {}
        self._max_recent_proactive = 10
        self._history_path: Path | None = None
        if data_dir:
            self._history_path = Path(data_dir) / "proactive_history.json"
            self._load_history()

    def _load_history(self) -> None:
        if self._history_path is None or not self._history_path.exists():
            return
        try:
            raw = json.loads(self._history_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._recent_proactive = {
                    k: [str(m) for m in v][-self._max_recent_proactive:]
                    for k, v in raw.items()
                    if isinstance(v, list)
                }
        except (json.JSONDecodeError, OSError) as e:
            print(f"[proactive] history load failed (non-fatal): {e}")

    def _save_history(self) -> None:
        if self._history_path is None:
            return
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            self._history_path.write_text(
                json.dumps(self._recent_proactive, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            print(f"[proactive] history save failed (non-fatal): {e}")

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

        # Consecutive cap: a real friend who notices silence reaches out
        # once, maybe twice — then waits. Spamming 4 proactives during a
        # long user silence reads as needy/AI.
        if (
            not force
            and record.consecutive_proactive_count >= self.config.max_consecutive_proactive
        ):
            return {
                "key": key, "status": "skipped_consecutive_cap",
                "sent_without_reply": record.consecutive_proactive_count,
                "cap": self.config.max_consecutive_proactive,
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

        # Record the proactive message in the recipient's short-term buffer
        # as an assistant turn. Without this, the next user reply has no
        # preceding assistant turn in history, and Aria treats user's "怎么说"
        # as a cold opener (production trace fixed: proactive sent "超市门口
        # 那只橘猫居然记得我", user said "怎么说", Aria replied "怎么了?" —
        # she'd forgotten what she just said).
        try:
            await self.engine.memory.short_term.append_for_recipient(
                key, "assistant", message,
            )
        except Exception as e:
            print(f"[proactive] short_term append failed: {e}", flush=True)

        self.tracker.record_proactive_sent(record.channel, record.recipient_id)
        await self.tracker.save()

        # Remember for anti-repetition (in-memory + persisted to disk so
        # process restart doesn't wipe; see _load_history docstring)
        recent = self._recent_proactive.setdefault(key, [])
        recent.append(message)
        if len(recent) > self._max_recent_proactive:
            del recent[: len(recent) - self._max_recent_proactive]
        self._save_history()

        # Clear wants_to_share on whichever recent event this message
        # actually voiced — content-match by character n-gram overlap.
        # Without this, a 📌想说 event stays marked until TTL decay (2h)
        # and proactive cycles every 5min keep picking it.
        try:
            await self._mark_event_shared(message)
        except Exception as e:
            print(f"[proactive] mark_event_shared failed: {e}")

        return {"key": key, "status": "sent", "message": message}

    async def _mark_event_shared(self, message: str) -> None:
        """Find the queued share intent whose fact content overlaps with `message`,
        consume it from ShareIntentStore.

        Heuristic: sliding char-window overlap (≥4 chars shared). Replaces the
        old recent_events / wants_to_share mutation path.
        """
        if self.share_intent_store is None or self.fact_retriever is None or not message:
            return
        msg = message.strip()
        pending = await self.share_intent_store.pending()
        best_id: str | None = None
        best_score = 0
        for intent in pending:
            fact = await self.fact_retriever.fetch_by_id(intent.fact_id)
            if fact is None:
                continue
            score = _content_overlap(fact.content, msg)
            if score >= 4 and score > best_score:
                best_score = score
                best_id = intent.fact_id
        if best_id:
            await self.share_intent_store.consume(best_id)
            print(f"[proactive] consumed share intent for fact={best_id[:8]}...",
                  flush=True)

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
        rec_key = f"{record.channel}:{record.recipient_id}"
        memory_context = await self.engine.memory.assemble_context(
            "", recipient_key=rec_key
        )

        # NOTE: long-term facts and episodes are still retrieved (via
        # build_system_prompt below — it consumes memory_context). We don't
        # need to format them locally any more; the system-prompt path does
        # that consistently with reactive turns.

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

        # Filter relevant_episodes: drop any episode whose content overlaps
        # with recent proactive messages. Without this, an episode that
        # records "Aria 对蜘蛛做梦感到好奇" keeps getting retrieved into the
        # prompt even after proactive already said it 3 times — the model
        # sees the episode framed as "her past" and uses it as fresh
        # proactive material on the next cycle.
        if memory_context.relevant_episodes and recent_msgs:
            kept = []
            dropped = 0
            for ep in memory_context.relevant_episodes:
                summary = (ep.summary or "")
                # If any recent proactive message has 4+ char overlap
                # with this episode summary, treat it as already-spoken.
                already = any(
                    _content_overlap(summary, m) >= 4
                    for m in recent_msgs[-self._max_recent_proactive:]
                )
                if already:
                    dropped += 1
                else:
                    kept.append(ep)
            if dropped:
                print(f"[proactive] filtered {dropped} already-said episodes from {rec_key}")
                memory_context.relevant_episodes = kept

        # relationship_level / goals were used by the legacy ad-hoc proactive
        # prompt; they're now rendered consistently inside build_system_prompt
        # below (relationship section + goals are part of persona).

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

        # Relational memory for this recipient — same source the reactive
        # turn uses, so proactive references "我们的暗号 / 共同地点 / 甜蜜
        # 瞬间" the same way she would mid-conversation.
        relational_memory = None
        if getattr(self.engine, "relational_store", None) is not None:
            try:
                relational_memory = await self.engine.relational_store.load(rec_key)
            except Exception as e:
                print(f"[relational] proactive load failed: {e}", flush=True)
                relational_memory = None

        # World awareness — same source as reactive
        daily_briefing = None
        if getattr(self.engine, "world_store", None) is not None:
            try:
                daily_briefing = await self.engine.world_store.load_today()
            except Exception as e:
                print(f"[world] proactive load failed: {e}", flush=True)
                daily_briefing = None

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
            emotion_state=emotion_state,
            inner_state=inner_state,
            relational_memory=relational_memory,
            mode="single",
        )

        # Dynamic per-turn state goes through the focus reminder, surfaced
        # via `<system-reminder>` user-channel right before the actual user
        # prompt content. Recency weight + distinct framing (CC pattern).
        # No `last_assistant_question` for proactive — there's nothing to
        # answer in this OPENER turn.
        focus_reminder = self.engine.prompt_builder.build_turn_focus_reminder(
            current_time=now,
            last_interaction_time=record.last_interaction,
            inner_state=inner_state,
            emotion_state=emotion_state,
            current_mood=None,
            daily_briefing=daily_briefing,
            recent_proactive_messages=(
                recent_msgs[-self._max_recent_proactive:] if recent_msgs else None
            ),
            proactive_mode=True,
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

        # Pending share intent — highest-significance NPC event queued for voicing.
        # If one exists, surface its content explicitly so the LLM has a concrete
        # seed rather than inventing a topic. Without this the LLM defaults to
        # generic hooks; with it, it can work the actual fact into a natural opener.
        pending_share_block = ""
        if self.share_intent_store is not None and self.fact_retriever is not None:
            try:
                share_result = await find_pending_share(
                    self.share_intent_store, self.fact_retriever
                )
                if share_result is not None:
                    _intent, _fact = share_result
                    pending_share_block = (
                        f"## 你正想分享的一件事（来自身边的人）\n"
                        f"{_fact.content}\n\n"
                        f"（这是你的内心素材——**不要直接念出来**，把它自然带进你的表达里。）\n"
                    )
                    print(
                        f"[proactive] share intent seed: {_fact.content[:40]}...",
                        flush=True,
                    )
            except Exception as e:
                print(f"[proactive] find_pending_share failed (non-fatal): {e}")

        opener_shape = (
            "## 这条消息是 OPENER（你主动起头），形态约束：\n"
            "- **直接进话题**——不寒暄打招呼（不要『嗨/你好/在吗/下午好』起手）\n"
            "- **不空泛 check-in**——不问『今天怎么样/还好吗/最近还好吗』这种"
            "宽泛问候。要么问具体的事（『奶奶今天吃得下吗』），要么不问只是说一件事，"
            "要么干脆别发。\n"
            "- **跟人发消息那种短句**，不是写信开头\n"
            "- **不要写『我+今天的状态』那种 self-report 句式**——『今天就一直在刷"
            "手机』『今天没怎么干活』『一天都在写东西』——**这是回答没被问的问题**，"
            "读起来像在 reactive 回复。对方没问『你今天怎么样』，**你不需要主动汇报**。"
            "要分享就说**具体一件事**（『刚才路过一只猫蹲在车顶』），不是泛报状态。\n"
            "- 同理不要『我刚 X』『我在 Y』当唯一内容——叙述自己的状态作为开头是 AI 套路。"
            "真人 proactive 是**带着具体的事/想法/问题/情绪**来的，不是来打卡的。\n"
            "- **不要问『X 了吗』当空 check-in 钩子**——『睡了吗 / 回家了吗 / 下班了吗 / 吃饭了吗』"
            "这种 yes/no 问题在没有**具体证据**时就是变形的『你还好吗』，AI 味很重。\n"
            "  - 时段不对的问题尤其奇怪：23:00 之前问『睡了吗』读起来像在催睡（多数成人 23:30 后才睡）；"
            "20:00 问『下班了吗』读起来像在催回家\n"
            "  - 真要问就基于**具体证据**：用户上次说要熬夜赶报告 → 深夜了才合理问『还在搞那个吗』；"
            "用户上次说要出差 → 现在才合理问『落地了吗』。**没具体钩子就不发**。\n"
        )
        if force:
            style = random.choice(_MESSAGE_STYLES)
            user_prompt = (
                f"[这一刻没有对方的新消息——你一个人，在想要不要主动发一条]\n\n"
                f"距离上次聊已经 {format_timedelta_cn(silence)}。\n\n"
                f"## 对方最近发的话（**他此刻的状态/在干啥都在这里**，比长期记忆更重要）\n"
                f"{user_recent_block}\n\n"
                f"## 你最近发过的主动消息（不要重复套路/比喻/切入点）\n{recent_proactive}\n\n"
                f"{pending_share_block}"
                f"{opener_shape}\n"
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
                f"{pending_share_block}"
                f"{opener_shape}\n"
                f"考虑：现在时间合不合适、你**真的**有话说吗（具体事不是闲扯）。"
                f"为了发而发不如不发。\n\n"
                f"按 system prompt 的 `===META===` 格式输出，meta 里加 `should_send`：\n```\n"
                f"<对外那一句，IM 短句>\n===META===\n"
                f'{{"should_send": true, "inner": "为什么发"}}\n```\n\n'
                f"不该发就：\n```\n\n===META===\n"
                f'{{"should_send": false, "inner": "为什么不发"}}\n```'
            )

        # Embed the focus reminder at the start of the proactive prompt
        # (same placement as engine.chat_full — recency weight, distinct
        # `<system-reminder>` framing).
        if focus_reminder:
            user_prompt = f"{focus_reminder}\n\n{user_prompt}"

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

        # Honour explicit should_send=false; otherwise infer from speech.
        # Coerce carefully — LLM occasionally emits string "false" / "true"
        # / "no" / "yes", and bool("false") is True in Python (any non-empty
        # string is truthy), so we'd ignore the false signal and send.
        raw = meta.get("should_send")
        if raw is None:
            should_send = bool(message)
        elif isinstance(raw, bool):
            should_send = raw
        elif isinstance(raw, str):
            should_send = raw.strip().lower() not in ("false", "no", "0", "")
        else:
            should_send = bool(raw)

        return {
            "should_send": should_send,
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
