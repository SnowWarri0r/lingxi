"""Plan executor — replaces the random simulator. Every 30min tick,
finds the plan covering the current hour, generates a concrete
first-person moment, and writes it as an event fact.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.retriever import FactQuery, FactRetriever
from lingxi.facts.writers.life import LifeWriter
from lingxi.planner.daily_planner import DailyPlanner
from lingxi.providers.base import LLMProvider


# {self} = persona self-context (build_self_context). The persona drives the
# flavor — a catgirl logs cat moments, a writer logs a writer's.
_SYSTEM_TMPL = "{self} 你正在做今天计划里的某件事，现在记一条此刻给自己看。"


_MOMENT_PROMPT = """我今天这个时段安排的：{plan_content}（{time_window}）
我刚才这 2 小时经历过：
{recent_events}

现在是 {now_hhmm}，{position}。我记一下此刻在做什么。接着刚才往下走就行，事情走到哪儿就写哪儿。

写一条**现在这一刻**，1-2 句，第一人称当下时态，符合你自己的口吻，直接以动作或观察开头（如『趴窗台晒太阳』）。
- **取景放在正常生活的尺度上**：眼前在做的那件事、周围的动静、身边的人和刚说的一句话、脑子里冒出来的念头、外面的天气声音——像跟人讲"我刚在干嘛"那样的粒度。
- **音量按事情本身来**：一天里绝大多数时刻是平的，平铺直叙记一句就好；真遇上让你激动的事，那一条再放开写。
- 已经过去半小时了，写这半小时里**新发生的**：手上的事推到了下一步、或者换了件事做、或者身边有了别的动静。
"""


# Retry nudge when the fresh moment restates the last one.
_MOVE_ON = (
    "\n\n（刚才那条已经把这件事写过了。半小时过去了，写接下来发生的："
    "手上这件事做完了、或者做到了下一步、或者你已经在做别的了。）"
)


_RESTATEMENT_THRESHOLD = 0.80

_TW_RE = re.compile(r"^(\d{2}):(\d{2})-(\d{2}):(\d{2})$")


def _parse_time_window(tag_value: str) -> tuple[int, int] | None:
    """Window as (start, end) minutes-of-day."""
    m = _TW_RE.match(tag_value)
    if not m:
        return None
    start_h, start_m, end_h, end_m = map(int, m.groups())
    return start_h * 60 + start_m, end_h * 60 + end_m


def describe_position(now_minute: int, start: int, end: int) -> str:
    """Where in this plan step we are, in the words the moment needs.

    Steps run 30 to 210 minutes while this ticks every 30, so a long one has
    to yield six or seven distinct moments. Handed only a clock reading, the
    model restated the start of the block for two hours. Saying which part of
    the step this is turns the same window into a beginning, a middle and an
    end — and keeps the restatement guard from having to throw those ticks
    away, which on a 3.5-hour step would empty most of a morning.
    """
    span = (end - start) % 1440 or 1440
    elapsed = (now_minute - start) % 1440
    left = span - elapsed
    if span <= 45:
        return "这一段就这么点时间"
    if elapsed <= 30:
        return "这一段刚开始"
    if left <= 30:
        return "这一段快结束了，手上的事该收尾了"
    if elapsed >= span / 2:
        return f"这一段过半了，还剩 {left} 分钟"
    return f"这一段走了 {elapsed} 分钟，还有 {left} 分钟"


def _in_window(now_minute: int, start: int, end: int) -> bool:
    """Minute-of-day containment; end <= start means the window crosses
    midnight (e.g. 23:00-01:00)."""
    if start < end:
        return start <= now_minute < end
    return now_minute >= start or now_minute < end


class PlanExecutor:
    def __init__(
        self,
        llm: LLMProvider,
        retriever: FactRetriever,
        life_writer: LifeWriter,
        planner: DailyPlanner | None = None,
        model: str | None = None,
        persona=None,
        embedder=None,
    ):
        self._llm = llm
        self._retriever = retriever
        self._writer = life_writer
        self._planner = planner
        self._model = model
        self._embedder = embedder
        self._replan_requested = False
        from lingxi.persona.self_context import build_self_context
        self._self_ctx = (build_self_context(persona)
                          if persona is not None else "你是 Aria。")

    def request_replan(self) -> None:
        self._replan_requested = True

    async def tick(self) -> None:
        now = datetime.now()

        if self._replan_requested and self._planner is not None:
            try:
                await self._planner.plan_aria()
            finally:
                self._replan_requested = False

        current_plan = await self._find_current_plan(now)
        if current_plan is None:
            return

        # fetch() ranks 0.5*recency + 0.3*(importance/10), and across a
        # two-hour window the recency term moves 0.010 end to end while
        # importance moves 0.27 — so importance decided both which moments
        # came back and what order they were in. Over 194 real ticks the
        # guard's `previous` was not the latest moment on 47%, and the list
        # the model read was out of time order on 57%; three ticks running
        # once wrote 「千砂那段还在耳朵边转 / 懒得动」 while the guard compared
        # each against an older entry. Over-fetch, then take the last three by
        # the clock and hand them over oldest-first, so "刚才经历过" reads
        # forward and ends on the moment this one has to follow.
        pool = await self._retriever.fetch(FactQuery(
            subject="aria", type=FactType.EVENT,
            since=now - timedelta(hours=2), limit=12,
        ))
        recent_events = sorted(pool, key=lambda f: f.ts)[-3:]
        tw = self._tag_value(current_plan, "time_window") or "?"
        window = _parse_time_window(tw)
        position = (
            describe_position(now.hour * 60 + now.minute, *window)
            if window else "时间往前走了一点"
        )
        prompt = _MOMENT_PROMPT.format(
            plan_content=current_plan.content,
            time_window=tw,
            recent_events=self._bullets(recent_events) or "（没什么特别的）",
            now_hhmm=now.strftime("%H:%M"),
            position=position,
        )
        previous = recent_events[-1].content if recent_events else ""
        content = await self._generate(prompt)
        if not content:
            return

        # A plan step spans a couple of hours and this ticks every thirty
        # minutes, so when the step has no internal progression the model
        # restates it. Measured over three days: 「守着锅等它咕嘟」 four times
        # across two hours, the same sentence reworded. Give it one nudge to
        # move on, and if it still restates, write nothing — an hour with one
        # entry is honest; four entries about waiting for porridge is not.
        if await self._restates(content, previous):
            content = await self._generate(prompt + _MOVE_ON) or content
            if await self._restates(content, previous):
                print(f"[executor] still restating, skipped: {content[:34]}",
                      flush=True)
                return

        event = Fact(
            subject="aria",
            content=content,
            source=Source.LIFE_SIMULATED,
            type=FactType.EVENT,
            ts=now,
        )
        await self._writer.write(event)

    async def _generate(self, prompt: str) -> str:
        try:
            kwargs = {"model": self._model} if self._model else {}
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=_SYSTEM_TMPL.format(self=self._self_ctx),
                max_tokens=200,
                temperature=0.8,
                _debug_purpose="plan_executor_moment",
                **kwargs,
            )
            return response.content.strip()
        except Exception as e:
            print(f"[executor] moment gen failed: {e}", flush=True)
            return ""

    async def _restates(self, candidate: str, previous: str) -> bool:
        """Whether this moment just rewords the one before it.

        Calibrated on 100 consecutive pairs from three days of her own
        events. Restatements sit at 0.85 and above — 「走在银杏街上，叶子还
        绿着」 against 「走到银杏街了，叶子还硬邦邦绿着」 at 0.845. Real
        progression sits at 0.62–0.65: same preoccupation, but she has moved.
        The nearest pair that would be wrong to drop is 0.789, where she
        finishes one line and turns to the next, so the gate has room below it.

        Fail-safe: no embedder, or an embedding error, and the moment is
        written. A repeat costs a line; refusing to write costs the hour.
        """
        if not self._embedder or not candidate or not previous:
            return False
        try:
            a = await self._embedder.embed(candidate)
            b = await self._embedder.embed(previous)
        except Exception as e:
            print(f"[executor] restatement check unavailable: {e}", flush=True)
            return False
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0.0 or nb == 0.0:
            return False
        return dot / (na * nb) >= _RESTATEMENT_THRESHOLD

    async def _find_current_plan(self, now: datetime) -> Fact | None:
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        plans = await self._retriever._store.query(
            subject="aria", type=FactType.PLAN, since=today_start, limit=20,
        )
        now_minute = now.hour * 60 + now.minute
        for plan in plans:
            tw_value = self._tag_value(plan, "time_window")
            if not tw_value:
                continue
            window = _parse_time_window(tw_value)
            if window is None:
                continue
            if _in_window(now_minute, *window):
                return plan
        return None

    @staticmethod
    def _tag_value(fact: Fact, key: str) -> str | None:
        for t in fact.tags:
            if t.startswith(f"{key}:"):
                return t[len(key) + 1:]
        return None

    @staticmethod
    def _bullets(facts: list[Fact]) -> str:
        return "\n".join(f"  - {f.content}" for f in facts)
