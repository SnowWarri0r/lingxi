"""LifeSimulator: advances Aria's day in the background.

Flow:
- At "dawn" (first tick after 06:00 local time), generate today's DailyPlan via LLM
- Every hour, check the plan and update current_activity based on time slot
- Occasionally generate a LifeEvent (small life happenings)
- Log diary entries when significant moments happen

The resulting state is AUTHORITATIVE — when the user chats, Aria's response
is based on what she's truly in the middle of.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING

from lingxi.inner_life.models import (
    Activity,
    ActivityKind,
    DailyPlan,
    DiaryEntry,
    InnerState,
    LifeEvent,
)
from lingxi.inner_life.store import InnerLifeStore

if TYPE_CHECKING:
    from lingxi.persona.models import PersonaConfig
    from lingxi.providers.base import LLMProvider


DAILY_PLAN_PROMPT = """你是 {persona_name}。请为今天（{date_str}）写一份简短的日程，像真人给自己排一天那样。

## 关于你
{persona_blurb}

## 你最近几天的状态
{recent_context}

## 现在时间
{current_time_cn}

请生成今天的日程（只覆盖从现在到睡前，不用覆盖凌晨）。
要求：
- 6-8 段"活动"，每段对应一个时段
- 不要太理想化——真人一天会有琐事、走神、临时变化
- 活动要和你的人设自洽（天文学家写作+观星的节奏），但也要有普通人做的事（吃饭、洗碗、看手机）
- 给今天定个"情绪基调"（简短一两个词）
- 想 2-3 件"可能今天会发生的小事件"（编辑联系、朋友消息、突发灵感、东西坏了...）

只回复 JSON：
{{
  "mood_theme": "今天整体的感觉",
  "activities": [
    {{
      "start_hour": 9, "end_hour": 11,
      "kind": "work|routine|meal|rest|hobby|outdoors|social|sleep",
      "name": "简短活动名",
      "description": "具体在做什么，一句话",
      "focus_level": 0.0-1.0,
      "social_openness": 0.0-1.0
    }}
  ],
  "possible_events": [
    "编辑发来消息说第三章要改",
    "望远镜目镜进灰，得擦",
    "楼下传来装修噪音"
  ]
}}"""


HOURLY_MICROEVENT_PROMPT = """你是 {persona_name}，现在正在【{current_activity_name}】({current_activity_desc})。

刚刚这一小时内，你可能经历了一点小事（或没有）。
- 1/3 概率：什么都没发生，就是在做这件事
- 1/3 概率：一个小琐事（猫爪了一下、咖啡凉了、想到了什么）
- 1/3 概率：一个有点情绪起伏的事（编辑消息、朋友一句话、突然的感触）

今日情绪基调：{mood_theme}

只回复 JSON（如果什么都没发生，回复 {{"event": null}}）：
{{
  "event": {{
    "content": "发生了什么（简短一句）",
    "significance": 0.0-1.0,
    "emotional_impact": {{"好奇": 0.1, "焦虑": -0.2}},
    "wants_to_share": true/false
  }}
}}"""


class LifeSimulator:
    """Background life simulator for Aria."""

    def __init__(
        self,
        persona: PersonaConfig,
        llm: LLMProvider,
        store: InnerLifeStore,
        tick_interval_minutes: int = 30,
    ):
        self.persona = persona
        self.llm = llm
        self.store = store
        self.tick_interval_minutes = tick_interval_minutes
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        print(f"[life] simulator started (tick every {self.tick_interval_minutes}min)")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        # Immediate first tick so state is fresh on startup
        await asyncio.sleep(10)
        while self._running:
            try:
                await self.tick()
            except Exception as e:
                print(f"[life] tick error: {e}")
            await asyncio.sleep(self.tick_interval_minutes * 60)

    async def tick(self) -> None:
        """One simulator step: maybe generate plan, update current activity, maybe event."""
        state = await self.store.load_state()
        now = datetime.now()

        # 1. Ensure there's a plan for today
        today = now.date()
        need_plan = (
            state.today_plan is None
            or state.today_plan.date != today
            or (now.hour >= 6 and state.today_plan.generated_at.date() < today)
        )
        if need_plan and now.hour >= 6:
            await self._generate_daily_plan(state, now)

        # 2. Update current activity based on plan
        if state.today_plan:
            self._tick_activity(state, now)

        # 3. Maybe generate a micro-event (only if awake)
        if state.current_activity and state.current_activity.kind != ActivityKind.SLEEP:
            if random.random() < 0.35:  # ~1/3 per tick
                await self._maybe_generate_event(state, now)

        # 4. Drift dynamics: energy/creative_drive/social_need over the day
        self._drift_dynamics(state, now)

        state.last_simulated_at = now
        await self.store.save_state(state)

    # -- internals --

    def _persona_blurb(self) -> str:
        p = self.persona
        parts = [f"你是 {p.identity.full_name}（{p.name}）。"]
        if p.identity.occupation:
            parts.append(f"职业：{p.identity.occupation}。")
        if p.identity.background:
            parts.append(p.identity.background.strip())
        traits = "、".join(
            t.trait for t in p.personality.traits[:4]
        )
        if traits:
            parts.append(f"性格：{traits}。")
        return "\n".join(parts)

    async def _generate_daily_plan(self, state: InnerState, now: datetime) -> None:
        """Ask LLM to outline today."""
        from lingxi.temporal.formatter import format_datetime_cn

        # Recent context: last 2 diary entries
        recent_blurb = ""
        if state.recent_diary:
            recent_blurb = "\n".join(
                f"- [{d.timestamp.strftime('%m-%d %H:%M')}] {d.content[:80]}"
                for d in state.recent_diary[-3:]
            )
        else:
            recent_blurb = "（没有近期记录）"

        prompt = DAILY_PLAN_PROMPT.format(
            persona_name=self.persona.name,
            date_str=now.strftime("%Y-%m-%d"),
            persona_blurb=self._persona_blurb(),
            recent_context=recent_blurb,
            current_time_cn=format_datetime_cn(now),
        )

        try:
            result = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.8,
            )
        except Exception as e:
            print(f"[life] daily plan LLM failed: {e}")
            return

        plan_data = self._extract_json(result.content)
        if not plan_data:
            return

        try:
            activities = []
            for a in plan_data.get("activities", []):
                start_h = int(a.get("start_hour", 9))
                end_h = int(a.get("end_hour", start_h + 1))
                started = datetime.combine(now.date(), time(hour=min(23, max(0, start_h))))
                ended = datetime.combine(now.date(), time(hour=min(23, max(0, end_h))))
                kind_str = str(a.get("kind", "routine")).lower()
                try:
                    kind = ActivityKind(kind_str)
                except ValueError:
                    kind = ActivityKind.ROUTINE
                activities.append(Activity(
                    kind=kind,
                    name=str(a.get("name", ""))[:40],
                    description=str(a.get("description", ""))[:200],
                    started_at=started,
                    ended_at=ended,
                    focus_level=float(a.get("focus_level", 0.5)),
                    social_openness=float(a.get("social_openness", 0.5)),
                ))

            plan = DailyPlan(
                date=now.date(),
                mood_theme=str(plan_data.get("mood_theme", ""))[:50],
                scheduled_activities=activities,
                pending_events=[],
                generated_at=now,
            )
            state.today_plan = plan
            print(f"[life] generated plan for {now.date()}: {len(activities)} activities, theme={plan.mood_theme}")

            # Log daybreak as a diary entry
            diary = DiaryEntry(
                content=f"今天的计划：{plan.mood_theme}。" + "、".join(a.name for a in activities[:5]),
                tags=["daybreak", "plan"],
            )
            state.recent_diary.append(diary)
            await self.store.append_diary(diary)

            # Save possible_events list as pending
            for ev in plan_data.get("possible_events", []):
                pending = LifeEvent(
                    content=str(ev)[:150],
                    significance=0.5,
                    wants_to_share=False,
                )
                plan.pending_events.append(pending)
        except Exception as e:
            print(f"[life] plan parse error: {e}")

    def _tick_activity(self, state: InnerState, now: datetime) -> None:
        """Find the current activity from plan based on time."""
        if not state.today_plan:
            return

        current: Activity | None = None
        for act in state.today_plan.scheduled_activities:
            if act.started_at <= now < (act.ended_at or (act.started_at + timedelta(hours=1))):
                current = act
                break

        if current is None:
            # No planned activity for this slot → free time / idle
            hour = now.hour
            if 0 <= hour < 6:
                current = Activity(
                    kind=ActivityKind.SLEEP,
                    name="睡觉",
                    description="在睡觉",
                    started_at=now,
                    focus_level=1.0,
                    social_openness=0.0,
                )
            else:
                current = Activity(
                    kind=ActivityKind.REST,
                    name="空闲",
                    description="没特别在做什么",
                    started_at=now,
                    focus_level=0.2,
                    social_openness=0.7,
                )

        # Update only if different from current to avoid reset-churn
        if state.current_activity is None or state.current_activity.id != current.id:
            state.current_activity = current

    async def _maybe_generate_event(self, state: InnerState, now: datetime) -> None:
        """LLM: did anything happen in the last hour?"""
        if not state.current_activity:
            return

        theme = state.today_plan.mood_theme if state.today_plan else ""
        prompt = HOURLY_MICROEVENT_PROMPT.format(
            persona_name=self.persona.name,
            current_activity_name=state.current_activity.name,
            current_activity_desc=state.current_activity.description,
            mood_theme=theme or "一般",
        )

        try:
            result = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=1.0,
            )
        except Exception as e:
            print(f"[life] event LLM failed: {e}")
            return

        data = self._extract_json(result.content)
        if not data:
            return

        event_data = data.get("event")
        if not event_data or not isinstance(event_data, dict):
            return

        try:
            event = LifeEvent(
                content=str(event_data.get("content", ""))[:200],
                significance=float(event_data.get("significance", 0.3)),
                emotional_impact=dict(event_data.get("emotional_impact", {})),
                wants_to_share=bool(event_data.get("wants_to_share", False)),
            )
        except Exception:
            return

        if not event.content:
            return

        # Prepend + bound
        state.recent_events.insert(0, event)
        state.recent_events = state.recent_events[:30]

        # Log significant events to diary
        if event.significance >= 0.5:
            diary = DiaryEntry(
                content=event.content,
                tags=["event"],
            )
            state.recent_diary.append(diary)
            state.recent_diary = state.recent_diary[-50:]
            await self.store.append_diary(diary)

        print(f"[life] event: {event.content} (sig={event.significance:.1f})")

    def _drift_dynamics(self, state: InnerState, now: datetime) -> None:
        """Simple physics on energy/creative_drive/social_need over time."""
        hour = now.hour
        # Energy: lowest 22-6 (sleep), rises during day
        if 0 <= hour < 6:
            state.energy = max(0.0, state.energy - 0.02)  # fatigue if should be sleeping
        elif 6 <= hour < 9:
            state.energy = min(1.0, state.energy + 0.08)  # morning boost
        elif 14 <= hour < 17:
            state.energy = max(0.3, state.energy - 0.03)  # afternoon dip
        elif hour >= 22:
            state.energy = max(0.2, state.energy - 0.05)  # evening fade

        # Creative drive: peaks in morning and late night for a writer/astronomer
        if 9 <= hour < 12 or 20 <= hour < 23:
            state.creative_drive = min(1.0, state.creative_drive + 0.05)
        else:
            state.creative_drive = max(0.1, state.creative_drive - 0.02)

        # Social need: builds up if she's alone a long time
        state.social_need = min(1.0, state.social_need + 0.02)

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        text = text.strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            return None
