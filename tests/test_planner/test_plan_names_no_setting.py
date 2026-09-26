"""The planner's generic instructions named a school for every persona.

Its outside-world example offered 学校里遇到的事. 唐可可 graduated in the
persona config on 2026-08-13 and her simulated days kept going to class,
the classroom, the dorm and the canteen for six weeks after. On the real
09-24 prompt, eight plans each way: 8 school items in 77 lines with the
example, 1 in 75 without.
"""

import json
from types import SimpleNamespace

import pytest

from lingxi.facts.retriever import FactRetriever
from lingxi.facts.store import FactStore
from lingxi.facts.writers.life import LifeWriter
from lingxi.planner.daily_planner import DailyPlanner

SETTINGS = ("学校", "教室", "上课", "宿舍", "食堂", "公司", "工位", "办公室")


class _LLM:
    def __init__(self):
        self.prompts = []

    async def complete(self, *, messages, system=None, **kw):
        self.prompts.append((system or "") + messages[0]["content"])
        return SimpleNamespace(content=json.dumps([]))


@pytest.mark.asyncio
async def test_the_planner_itself_names_no_setting(tmp_path):
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    llm = _LLM()
    planner = DailyPlanner(llm, FactRetriever(store), LifeWriter(store, scorer=None))

    async def _no_weather(now):
        return ""
    planner._todays_weather = _no_weather

    await planner.plan_aria()

    named = [w for w in SETTINGS if w in llm.prompts[0]]
    assert named == [], f"generic planner text names a setting: {named}"
