"""Daily planner — Aria and (later) NPCs plan their own days,
first-person, in the morning.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.retriever import FactQuery, FactRetriever
from lingxi.facts.writers.life import LifeWriter
from lingxi.providers.base import LLMProvider
from lingxi.utils import lenient_json


# {self} = persona self-context — the single source of "who am I". Identity
# stays in the persona YAML: a facts-side biography lookup degrades into a
# recency/importance top-N whenever the FTS keyword misses (retriever treats
# semantic as a 0.2-weight boost, not a filter), which once fed days of
# heartbeat-counting events back in as "identity" and locked plans onto them.
_SYSTEM_TMPL = "{self} 你在想自己今天怎么过。"

_WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# The planner may run as a startup catch-up at any hour, not only at the 7am
# tick — the prompt states the real date/weekday/clock so the plan follows the
# actual day (weekend vs weekday) and covers only the hours still ahead.
#
# The outside-world example names no setting. It used to offer 学校里遇到的事,
# a school written into the planner for every persona — and 唐可可 graduated
# (config, 2026-08-13). Her simulated days kept 上课、课间、进教室、回宿舍、
# 食堂 at 4-8% of events and plans for six weeks after. A/B on the real 09-24
# prompt, eight plans each: 8 school items in 77 lines with the example, 1 in
# 75 without. Where she spends her day comes from the persona, not from here.
_PLAN_PROMPT = """今天是 {date_str}（{weekday}），现在 {now_hhmm}。我想一下今天{scope_phrase}怎么过。

【昨天我反思到的】
{reflections}

【最近一周我注意到的模式】
{patterns}

【我生活里的人】
{people}

{outside}【怎么安排】
- {coverage_line}
- hour 粒度，time_window 形如 "09:00-12:00"，全部排在 {now_hhmm} 之后
- 写**具体**符合**你这个人/你这种日子**的事：落到你心里清楚在做的那件具体行为（如『趴窗台晒太阳』『等他下班』）；今天是{weekday}，按{weekday}该有的节奏排
- 至少 2 条对应到你长期在惦记/在做的事
- **一天里有别人也有外面的世界**：至少 2-3 条牵涉到上面这些人、或发生在门外（碰面、一起做点什么、路上/店里遇到的事、跟人说的一句话）。剩下的可以是你一个人的事。真实的一天是里外都有的。

输出 JSON：
[{{"time_window": "09:00-12:00", "content": "...", "goal": "..."}}, ...]
content 用你自己想事情的语气，第一人称，每条直接以动作或观察开头（如『趴窗台晒太阳』）。
"""


# The one channel by which something she did not think of herself can reach
# her day. Everything else the planner reads — reflections, patterns, people —
# is either her own prior output or static YAML, so the day was derived purely
# from the inside. The briefing is fetched at 06:03 and the plan written at
# 07:00 (55 of 63 days in that order), and nothing joined them: on 09-11 the
# scan said the typhoon's outer band would be worst on the 13th and 14th, and
# both days were planned around a sunny window seat and breakfast on the steps
# outside. Kept adjacent to the instruction that uses it — a requirement
# stated several blocks upstream is a requirement at the wrong address.
_OUTSIDE_TMPL = """【今天外面】
{items}
今天这一天是在这些条件里过的——按今天的天气安排哪些事在室内、哪些在外面；这里面有勾到我的，就排一条去做。

"""

# The scan's own weather sentence is written by a language model and is
# unreliable in exactly the way that matters: measured against the Shanghai
# record, 09-19 read 「放晴了…出门不用带伞」 on the wettest day of the
# fortnight (9.5mm) and 09-08 read 「外面一直在下雨」 on a day with none.
# Open-Meteo is already fetched every 20 minutes for the chat prompt, so the
# true figures go in first and the day is keyed to them.
_WEATHER_LINE = "  - 天气（实测）：{phrase}"


def _end_of_day(now: datetime) -> datetime:
    return now.replace(hour=23, minute=59, second=59, microsecond=0)


def _format_people(persona) -> str:
    """Render the persona's recurring_people so the day can include them."""
    bio = getattr(persona, "biography", None) if persona is not None else None
    people = getattr(bio, "recurring_people", None) or []
    lines = []
    for p in people:
        name = getattr(p, "name", "") or ""
        rel = getattr(p, "relation", "") or ""
        if name:
            lines.append(f"  - {name}：{rel}" if rel else f"  - {name}")
    return "\n".join(lines) or "（暂无）"


class DailyPlanner:
    def __init__(
        self,
        llm: LLMProvider,
        retriever: FactRetriever,
        life_writer: LifeWriter,
        model: str | None = None,
        persona=None,
    ):
        self._llm = llm
        self._retriever = retriever
        self._writer = life_writer
        self._model = model
        from lingxi.persona.self_context import build_self_context
        self._self_ctx = (build_self_context(persona)
                          if persona is not None else "你是 Aria。")
        # The people in her life, straight from the persona YAML. Without them
        # the planner only ever sees her own reflections, so it schedules a
        # fully solipsistic day — for a character defined by her group that is
        # both off-character and leaves her nothing to talk about.
        self._people_block = _format_people(persona)
        self._persona = persona

    async def plan_aria(self) -> list[Fact]:
        now = datetime.now()
        yesterday_start = (now - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        reflections = await self._retriever.fetch(
            FactQuery(subject="aria", type=FactType.PATTERN,
                      since=yesterday_start, limit=5)
        )
        # The two blocks are shown as different horizons but draw on one pool
        # ranked 0.5*recency + 0.3*importance, so yesterday's insights win the
        # week slots too: measured on the live store, all 5 reflection lines
        # reappeared among the 10 pattern lines — 15 lines carrying 10
        # insights, one day's theme at double weight in the prompt that sets
        # the whole next day. Over-fetch by what the first block took, drop
        # those, and the week block reaches a day further back instead.
        week_ago = now - timedelta(days=7)
        seen = {f.id for f in reflections}
        wider = await self._retriever.fetch(
            FactQuery(subject="aria", type=FactType.PATTERN,
                      since=week_ago, limit=10 + len(seen))
        )
        patterns = [f for f in wider if f.id not in seen][:10]

        if now.hour < 9:
            scope_phrase = ""
            coverage_line = "6-10 条今天的安排，覆盖一天不同时段（早、白天、晚上）+ 你的日常习惯"
        else:
            scope_phrase = "接下来"
            coverage_line = "3-8 条从现在到睡前的安排，覆盖剩下的时段 + 你的日常习惯"
        prompt = _PLAN_PROMPT.format(
            date_str=now.strftime("%-m月%-d日"),
            weekday=_WEEKDAYS[now.weekday()],
            now_hhmm=now.strftime("%H:%M"),
            scope_phrase=scope_phrase,
            coverage_line=coverage_line,
            reflections=self._bullets(reflections) or "（昨天没特别的反思）",
            patterns=self._bullets(patterns) or "（最近没新模式）",
            people=self._people_block,
            outside=build_outside_block(await self._todays_world(now),
                                        await self._todays_weather(now)),
        )
        items = await self._call_planner(
            prompt, _SYSTEM_TMPL.format(self=self._self_ctx))
        return await self._write_plan_facts("aria", items)

    async def _todays_weather(self, now: datetime) -> str:
        """Today's real high, low and rainfall for where she lives.

        The chat path reads a cache that a 20-minute loop keeps warm; the
        planner runs once a day and would be at the mercy of whether that loop
        had come round yet, so it asks for itself. Fail-safe: a miss plans the
        day without a weather line, as every day before this one did.
        """
        try:
            from lingxi.temporal.sun import persona_location
            from lingxi.temporal.weather import cached_outlook, refresh
            loc = persona_location(self._persona)
            outlook = cached_outlook(loc, now=now)
            if outlook is None:
                await refresh(loc, now=now)
                outlook = cached_outlook(loc, now=now)
            return outlook.phrase() if outlook else ""
        except Exception as e:
            print(f"[planner] weather lookup failed (non-fatal): {e}", flush=True)
            return ""

    async def _todays_world(self, now: datetime) -> list[Fact]:
        """This morning's scan. A day without one still gets planned."""
        try:
            day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return await self._retriever.fetch(FactQuery(
                subject="world", type=FactType.EVENT,
                since=day_start, limit=5))
        except Exception as e:
            print(f"[planner] world lookup failed (non-fatal): {e}", flush=True)
            return []

    async def _call_planner(self, prompt: str, system: str) -> list[dict]:
        try:
            kwargs = {"model": self._model} if self._model else {}
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=system,
                # 6-10 verbose Chinese plan items with goals land near 2000
                # tokens; a run that brushes the ceiling gets truncated and the
                # salvage parser then recovers only a partial day. Headroom.
                max_tokens=3000,
                temperature=0.5,
                _debug_purpose="daily_planner",
                **kwargs,
            )
            items = _parse_plan_items(response.content)
            if not items:
                print("[planner] no valid plan items parsed", flush=True)
            return items
        except Exception as e:
            print(f"[planner] LLM/parse failed: {e}", flush=True)
        return []

    async def _write_plan_facts(self, subject: str, items: list[dict]) -> list[Fact]:
        if not items:
            return []
        now = datetime.now()
        expires = _end_of_day(now)
        written: list[Fact] = []
        for item in items:
            tags = [f"time_window:{item['time_window']}"]
            if item.get("goal"):
                tags.append(f"goal:{item['goal']}")
            fact = Fact(
                subject=subject,
                content=str(item["content"]).strip(),
                source=Source.LIFE_SIMULATED,
                type=FactType.PLAN,
                ts=now,
                importance=7,
                expires_at=expires,
                tags=tags,
            )
            await self._writer.write_skip_scorer(fact, trigger_observation=False)
            written.append(fact)
        return written

    @staticmethod
    def _bullets(facts: list[Fact]) -> str:
        return "\n".join(f"  - {f.content}" for f in facts)


def build_outside_block(facts: list[Fact], weather_phrase: str = "") -> str:
    """Today's real weather and this morning's scan, or nothing at all.

    On the days neither is in hand — the scan was missing or late on 8 of 63 —
    an empty string keeps the prompt as it was rather than leaving an
    instruction pointing at a heading with nothing under it.
    """
    lines = []
    if weather_phrase.strip():
        lines.append(_WEATHER_LINE.format(phrase=weather_phrase.strip()))
    lines += [f"  - {f.content}" for f in facts if f.content.strip()]
    return _OUTSIDE_TMPL.format(items="\n".join(lines)) if lines else ""


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_objects(text: str) -> list[str]:
    """Scan out top-level {...} chunks (string-aware), tolerant of a truncated
    or comma-slipped array. Each chunk is parsed independently so one bad/cut
    item doesn't sink the whole plan."""
    objs: list[str] = []
    depth = 0
    start: int | None = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(text[start:i + 1])
                start = None
    return objs


def _parse_plan_items(content: str) -> list[dict]:
    """Plan items from an LLM response — strict JSON first, then salvage."""
    text = _strip_fences(content)
    candidates: list[dict] = []
    try:
        # Lenient first: the plan's `content` field is prose, and the model
        # marks emphasis in it with ASCII quotes, which is a syntax error.
        data = lenient_json.loads(text)
        if isinstance(data, list):
            candidates = [d for d in data if isinstance(d, dict)]
    except Exception:
        # Still broken means truncated — salvage whole items, quote-repairing
        # each so one emphasised phrase doesn't drop that slot from the day.
        for chunk in _extract_json_objects(text):
            try:
                d = lenient_json.loads(chunk)
                if isinstance(d, dict):
                    candidates.append(d)
            except Exception:
                pass
    return [
        item for item in candidates
        if "time_window" in item and "content" in item
    ]
