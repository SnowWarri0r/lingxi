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
最近 2 小时我经历过：
{recent_events}

现在是 {now_hhmm}——我正在做这件事的**某个具体片段**。
写一条**现在这一刻**——具体细节，不抽象（眼前的东西／身体感觉／此刻的念头／一个动作，任一就好）。
1-2 句，第一人称当下时态，符合你自己的口吻。
每条直接以动作或观察开头（如『趴窗台晒太阳』）。
"""


_TW_RE = re.compile(r"^(\d{2}):(\d{2})-(\d{2}):(\d{2})$")


def _parse_time_window(tag_value: str) -> tuple[int, int] | None:
    """Window as (start, end) minutes-of-day."""
    m = _TW_RE.match(tag_value)
    if not m:
        return None
    start_h, start_m, end_h, end_m = map(int, m.groups())
    return start_h * 60 + start_m, end_h * 60 + end_m


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
    ):
        self._llm = llm
        self._retriever = retriever
        self._writer = life_writer
        self._planner = planner
        self._model = model
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

        recent_events = await self._retriever.fetch(FactQuery(
            subject="aria", type=FactType.EVENT,
            since=now - timedelta(hours=2), limit=3,
        ))
        tw = self._tag_value(current_plan, "time_window") or "?"
        prompt = _MOMENT_PROMPT.format(
            plan_content=current_plan.content,
            time_window=tw,
            recent_events=self._bullets(recent_events) or "（没什么特别的）",
            now_hhmm=now.strftime("%H:%M"),
        )
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
            content = response.content.strip()
        except Exception as e:
            print(f"[executor] moment gen failed: {e}", flush=True)
            return

        if not content:
            return

        event = Fact(
            subject="aria",
            content=content,
            source=Source.LIFE_SIMULATED,
            type=FactType.EVENT,
            ts=now,
        )
        await self._writer.write(event)

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
