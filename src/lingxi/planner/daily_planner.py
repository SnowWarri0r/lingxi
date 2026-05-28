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


_ARIA_SYSTEM = "你是 Aria，正在早上安排今天打算做什么。"


_ARIA_PROMPT = """新的一天。我想一下今天打算怎么过。

【我是谁】
{biography}

【昨天我反思到的】
{reflections}

【最近一周我注意到的模式】
{patterns}

【我自己定的规矩】
- 6-10 条今天的安排
- 覆盖白天工作时间（9-12, 14-18）+ 晚上 + 早晚习惯
- hour 粒度，time_window 形如 "09:00-12:00"
- 写**具体**的事（"跑光变曲线第三组分析"而不是"工作"——自己心里知道在做什么）
- 至少 2 条对应到长期在做的事

输出 JSON：
[{{"time_window": "07:00-08:00", "content": "...", "goal": "..."}}, ...]
content 用自己想事情的语气，第一人称，但不要在每条前面写"我"——直接写动作。
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
    ):
        self._llm = llm
        self._retriever = retriever
        self._writer = life_writer
        self._model = model

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

        prompt = _ARIA_PROMPT.format(
            biography=biography,
            reflections=self._bullets(reflections) or "（昨天没特别的反思）",
            patterns=self._bullets(patterns) or "（最近没新模式）",
        )
        items = await self._call_planner(prompt, _ARIA_SYSTEM)
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
