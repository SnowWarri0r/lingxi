"""Daily planner — Aria and (later) NPCs plan their own days,
first-person, in the morning.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.retriever import FactQuery, FactRetriever
from lingxi.facts.writers.life import LifeWriter
from lingxi.providers.base import LLMProvider


# {self} = persona self-context. Plan content is driven by 【我是谁】(biography)
# + the persona, NOT hardcoded work hours / astronomy.
_SYSTEM_TMPL = "{self} 你在早上想一下今天打算怎么过。"

_PLAN_PROMPT = """新的一天。我想一下今天打算怎么过。

【我是谁】
{biography}

【昨天我反思到的】
{reflections}

【最近一周我注意到的模式】
{patterns}

【怎么安排】
- 6-10 条今天的安排，覆盖一天不同时段（早、白天、晚上）+ 你的日常习惯
- hour 粒度，time_window 形如 "09:00-12:00"
- 写**具体**符合**你这个人/你这种日子**的事（不是"工作""休息"这种笼统词——写你心里清楚在做的那件具体的事）
- 至少 2 条对应到你长期在惦记/在做的事

输出 JSON：
[{{"time_window": "07:00-08:00", "content": "...", "goal": "..."}}, ...]
content 用你自己想事情的语气，第一人称，但不要在每条前面写"我"——直接写动作。
"""


def _end_of_day(now: datetime) -> datetime:
    return now.replace(hour=23, minute=59, second=59, microsecond=0)


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

    async def plan_aria(self) -> list[Fact]:
        biography = await self._load_biography_summary()
        now = datetime.now()
        yesterday_start = (now - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        reflections = await self._retriever.fetch(
            FactQuery(subject="aria", type=FactType.PATTERN,
                      since=yesterday_start, limit=5)
        )
        week_ago = now - timedelta(days=7)
        patterns = await self._retriever.fetch(
            FactQuery(subject="aria", type=FactType.PATTERN,
                      since=week_ago, limit=10)
        )

        prompt = _PLAN_PROMPT.format(
            biography=biography,
            reflections=self._bullets(reflections) or "（昨天没特别的反思）",
            patterns=self._bullets(patterns) or "（最近没新模式）",
        )
        items = await self._call_planner(
            prompt, _SYSTEM_TMPL.format(self=self._self_ctx))
        return await self._write_plan_facts("aria", items)

    async def _call_planner(self, prompt: str, system: str) -> list[dict]:
        try:
            kwargs = {"model": self._model} if self._model else {}
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=system,
                max_tokens=800,
                temperature=0.5,
                _debug_purpose="daily_planner",
                **kwargs,
            )
            data = json.loads(_strip_fences(response.content))
            if isinstance(data, list):
                return [
                    item for item in data
                    if isinstance(item, dict)
                    and "time_window" in item and "content" in item
                ]
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

    async def _load_biography_summary(self) -> str:
        bio = await self._retriever.fetch(
            FactQuery(subject="aria", semantic="身份", limit=5)
        )
        if not bio:
            return "（暂无身份摘要）"
        return self._bullets(bio)

    @staticmethod
    def _bullets(facts: list[Fact]) -> str:
        return "\n".join(f"  - {f.content}" for f in facts)


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()
