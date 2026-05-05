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


DAILY_PLAN_PROMPT = """你是 {persona_name}。请为今天（{date_str}，{weekday_cn}）写一份简短的日程，像真人给自己排一天那样。

## 关于你
{persona_blurb}

## 今天的环境锚点（必须考虑）
- 现在时间：{current_time_cn}
- 季节/月份：{season_blurb}
- 昨晚睡眠：{sleep_blurb}
- 工作日 vs 周末：{weekday_kind}

## 你最近几天的状态
{recent_context}

## 你身边重要的人（可能今天会有联系）
{recurring_people_blurb}

请生成今天的日程（只覆盖从现在到睡前）。
要求：
- 6-8 段"活动"，每段对应一个时段
- **每段必须有 `scene` 字段**——具体在哪个物理位置（沙发/书桌前/厨房/便利店/床上/阳台/小区楼下…）
- 不要太理想化——真人一天会有琐事、走神、临时变化
- 大部分时段是**普通人每天都做的事**（吃饭、通勤、洗碗、刷手机、发呆、跑腿、看剧、打电话、冲凉、睡前刷剧…），这类 kind=routine/meal/rest/social 等占多数
- 和职业/爱好直接挂钩的活动**最多 1-2 段**（不是每天都要发生），绝不刻意往人设上靠
- 周末和工作日要有差别。下雨/没睡好/季节都会改变节奏
- 每天的活动要有变化，避免日复一日雷同。参考昨天做了什么，今天换个方向
- 情绪基调要从睡眠+季节+最近发生的事**自然推出**，不是抽象词堆砌
- 想 2-3 件"可能今天会发生的小事件"，贴近普通生活，**禁止重复最近几天已经发生过的**

只回复 JSON：
{{
  "mood_theme": "今天整体的感觉",
  "activities": [
    {{
      "start_hour": 9, "end_hour": 11,
      "kind": "work|routine|meal|rest|hobby|outdoors|social|sleep",
      "name": "简短活动名",
      "description": "具体在做什么，一句话",
      "scene": "沙发 / 书桌前 / 厨房 / 便利店 …",
      "focus_level": 0.0-1.0,
      "social_openness": 0.0-1.0
    }}
  ],
  "possible_events": [
    "楼下便利店出了新布丁",
    "手机突然卡了一下 重启了",
    "朋友发来个挺好笑的视频"
  ]
}}

possible_events 也用第一人称 IM 口语，不要写旁白叙述。"""


HOURLY_MICROEVENT_PROMPT = """你是 {persona_name}，现在正在【{current_activity_name}】({current_activity_desc}){scene_blurb}。

## 此刻的物理感受
- 时间：{current_time_cn}（{weekday_cn}）
- 季节：{season_blurb}
- 体力：{energy_blurb}
- 昨晚睡眠：{sleep_blurb}

## 最近几天**已经发生过的**事（**禁止重复**这些主题/事件）
{recent_events_block}

## 今天 daily plan 里**预设可能发生**的事（你可以挑一件让它在此刻发生，也可以不挑）
{pending_events_block}

## 你身边重要的人（事件偶尔可以涉及他们，但不要每次）
{recurring_people_blurb}

刚刚这一小时内，你可能经历了一点小事（或没有）。
- 1/2 概率：什么都没发生，就是在做这件事
- 1/4 概率：一个**有物理细节**的小琐事（咖啡洒了一点、闻到楼下烧菜的味道、猫爪了一下）
- 1/4 概率：一个有情绪波动的事（朋友一句消息、突然的感触、想起一个人）

今日情绪基调：{mood_theme}

要求：
- 事件必须**具体到物理细节**（在哪、感官、时间），不要写抽象的"想到了X"
- 如果挑了上面的 pending_events 里某条，把那条当起点再加具体细节
- 禁止重复最近事件块里出现过的主题
- 不要每次都"刷到X，想起Y" 的固定句式
- **content 字段必须用第一人称 + IM 口语**写——这是她要发给朋友的话的语气，不是旁白叙述。
  反例（旁白腔，禁止）："洗衣机里翻出了一张揉皱的便条" / "她看到楼下便利店换了新布丁"
  正例（第一人称 IM 口语）："今天洗衣服 翻到几个月前的购物清单 哈哈" / "楼下便利店出新布丁了 看着挺好吃"

只回复 JSON（如果什么都没发生，回复 {{"event": null}}）：
{{
  "event": {{
    "content": "≤30 字，第一人称 IM 口语（不是叙述句），必须有物理细节",
    "significance": 0.0-1.0,
    "emotional_impact": {{"好奇": 0.1, "焦虑": -0.2}},
    "wants_to_share": true/false,
    "from_pending": true/false
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
        today = now.date()

        # 0. Reset daily counters at dawn
        if state.significant_events_reset_date != today:
            state.significant_events_today = 0
            state.significant_events_reset_date = today
            # Also: roll a fresh sleep_quality at dawn (for "today felt rested" effects)
            state.sleep_quality = self._sample_sleep_quality(state)

        # 1. Ensure there's a plan for today
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

        # 3. Maybe generate a micro-event (only if awake) — rate-limited
        if state.current_activity and state.current_activity.kind != ActivityKind.SLEEP:
            if random.random() < 0.35:
                await self._maybe_generate_event(state, now)

        # 4. End-of-day quiet anchor: if nothing significant happened today,
        # log a "boring day" diary line so quiet days are also lived.
        if now.hour == 22 and state.significant_events_today == 0:
            already = any(
                d.timestamp.date() == today and "quiet" in d.tags
                for d in state.recent_diary[-10:]
            )
            if not already:
                quiet = DiaryEntry(
                    content=f"今天好像也没什么事 就这样过了一天",
                    tags=["quiet"],
                )
                state.recent_diary.append(quiet)
                await self.store.append_diary(quiet)

        # 5. Drift dynamics: energy/creative_drive/social_need over the day
        self._drift_dynamics(state, now)

        state.last_simulated_at = now
        await self.store.save_state(state)

    # -- ambient helpers --

    @staticmethod
    def _weekday_cn(now: datetime) -> tuple[str, str]:
        """Return (weekday_short, kind) — kind ∈ {工作日, 周末}."""
        names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        wd = now.weekday()
        return names[wd], "周末" if wd >= 5 else "工作日"

    @staticmethod
    def _season_blurb(now: datetime) -> str:
        m = now.month
        if m in (3, 4, 5):
            return f"{m}月，春末，上海开始转闷热，时不时下雨"
        if m in (6, 7, 8):
            return f"{m}月，夏天，上海闷热潮湿"
        if m in (9, 10, 11):
            return f"{m}月，秋天，上海开始凉下来"
        return f"{m}月，冬天，上海湿冷"

    def _sleep_blurb(self, sleep_quality: float) -> str:
        if sleep_quality >= 0.8:
            return "睡得不错，醒来还算清爽"
        if sleep_quality >= 0.6:
            return "一般，做了点梦，半夜醒过一次"
        if sleep_quality >= 0.4:
            return "睡得不太好，半夜翻来覆去"
        return "几乎没怎么睡，今天会比较钝"

    @staticmethod
    def _energy_blurb(energy: float) -> str:
        if energy >= 0.7:
            return "状态在线"
        if energy >= 0.45:
            return "中等，能干活但不兴奋"
        if energy >= 0.25:
            return "有点疲，想躺"
        return "很累，几乎只想发呆"

    def _sample_sleep_quality(self, state: InnerState) -> float:
        """Roll a sleep quality based on stress + a noise term."""
        base = 0.7
        # Stressful days → worse sleep
        recent_neg = sum(
            1 for ev in state.recent_events[:10]
            if any(v < -0.1 for v in ev.emotional_impact.values())
        )
        base -= 0.05 * min(recent_neg, 4)
        # Random noise
        base += random.uniform(-0.15, 0.15)
        return max(0.1, min(1.0, base))

    def _ambient_block(self, state: InnerState, now: datetime) -> dict[str, str]:
        from lingxi.temporal.formatter import format_datetime_cn
        weekday, weekday_kind = self._weekday_cn(now)
        return {
            "current_time_cn": format_datetime_cn(now),
            "weekday_cn": weekday,
            "weekday_kind": weekday_kind,
            "season_blurb": self._season_blurb(now),
            "sleep_blurb": self._sleep_blurb(state.sleep_quality),
            "energy_blurb": self._energy_blurb(state.energy),
        }

    def _recurring_people_blurb(self) -> str:
        people = self.persona.biography.recurring_people if self.persona.biography else []
        if not people:
            return "（无）"
        lines = []
        for p in people[:5]:
            lines.append(f"- {p.name}：{p.relation}")
        return "\n".join(lines)

    def _recent_events_block(self, state: InnerState) -> str:
        """Last ~5 days of significant events — passed to LLM to prevent re-generation."""
        recent = state.recent_events[:8]
        if not recent:
            return "（最近没有什么事）"
        lines = []
        for ev in recent:
            ago = datetime.now() - ev.timestamp
            if ago.days >= 1:
                stamp = f"{ago.days}天前"
            elif ago.seconds // 3600 >= 1:
                stamp = f"{ago.seconds // 3600}小时前"
            else:
                stamp = "刚才"
            lines.append(f"- [{stamp}] {ev.content[:80]}")
        return "\n".join(lines)

    def _pending_events_block(self, state: InnerState) -> str:
        """Today's plan-time predicted events that haven't fired yet."""
        if not state.today_plan or not state.today_plan.pending_events:
            return "（今天没有预设事件）"
        lines = [f"- {ev.content[:80]}" for ev in state.today_plan.pending_events[:5]]
        return "\n".join(lines)

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
        # Recent context: last 2 diary entries
        recent_blurb = ""
        if state.recent_diary:
            recent_blurb = "\n".join(
                f"- [{d.timestamp.strftime('%m-%d %H:%M')}] {d.content[:80]}"
                for d in state.recent_diary[-3:]
            )
        else:
            recent_blurb = "（没有近期记录）"

        ambient = self._ambient_block(state, now)
        prompt = DAILY_PLAN_PROMPT.format(
            persona_name=self.persona.name,
            date_str=now.strftime("%Y-%m-%d"),
            persona_blurb=self._persona_blurb(),
            recent_context=recent_blurb,
            recurring_people_blurb=self._recurring_people_blurb(),
            **ambient,
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
                    scene=str(a.get("scene", ""))[:30],
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

        # Rate-limit: at most 3 significant events per day. Below sig 0.5 still allowed.
        sig_capped = state.significant_events_today >= 3

        theme = state.today_plan.mood_theme if state.today_plan else ""
        ambient = self._ambient_block(state, now)
        scene = state.current_activity.scene
        scene_blurb = f"，在{scene}" if scene else ""

        prompt = HOURLY_MICROEVENT_PROMPT.format(
            persona_name=self.persona.name,
            current_activity_name=state.current_activity.name,
            current_activity_desc=state.current_activity.description,
            scene_blurb=scene_blurb,
            mood_theme=theme or "一般",
            recent_events_block=self._recent_events_block(state),
            pending_events_block=self._pending_events_block(state),
            recurring_people_blurb=self._recurring_people_blurb(),
            **ambient,
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

        # Cap-respect: if significant slot is full, demote to non-significant
        if sig_capped and event.significance >= 0.5:
            event.significance = 0.3

        # If LLM marked from_pending, consume that pending event
        from_pending = bool(event_data.get("from_pending", False))
        if from_pending and state.today_plan and state.today_plan.pending_events:
            state.today_plan.pending_events.pop(0)

        # Prepend + bound
        state.recent_events.insert(0, event)
        state.recent_events = state.recent_events[:30]

        # Log significant events to diary
        if event.significance >= 0.5:
            state.significant_events_today += 1
            diary = DiaryEntry(
                content=event.content,
                tags=["event"],
            )
            state.recent_diary.append(diary)
            state.recent_diary = state.recent_diary[-50:]
            await self.store.append_diary(diary)

        print(f"[life] event: {event.content} (sig={event.significance:.1f}, pending={from_pending})")

    def _drift_dynamics(self, state: InnerState, now: datetime) -> None:
        """Simple physics on energy/creative_drive/social_need over time."""
        hour = now.hour
        # Sleep quality dampens both peaks AND troughs of energy
        sleep_factor = 0.5 + 0.5 * state.sleep_quality  # 0.55-1.0
        if 0 <= hour < 6:
            state.energy = max(0.0, state.energy - 0.02)
        elif 6 <= hour < 9:
            # Morning boost scales with how well she slept
            state.energy = min(1.0, state.energy + 0.08 * sleep_factor)
        elif 14 <= hour < 17:
            # Afternoon dip is harsher when undersleep
            dip = 0.03 + (1.0 - state.sleep_quality) * 0.04
            state.energy = max(0.2, state.energy - dip)
        elif hour >= 22:
            state.energy = max(0.2, state.energy - 0.05)

        # Creative drive: peaks in morning and late night, but only if energy isn't tanked
        if (9 <= hour < 12 or 20 <= hour < 23) and state.energy > 0.4:
            state.creative_drive = min(1.0, state.creative_drive + 0.05)
        else:
            state.creative_drive = max(0.1, state.creative_drive - 0.02)

        # Social need: builds up if alone, drops when actually chatting (last 1h)
        if state.last_chat_at and (now - state.last_chat_at) < timedelta(hours=1):
            state.social_need = max(0.1, state.social_need - 0.05)
        else:
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
