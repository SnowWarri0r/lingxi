"""Both paths' assembled prompts, frozen.

Chat and openers each gather what she knows — his facts, her days, the
scan, the schedule, the agenda — and each renders it. Before that gathering
moves to one place, these snapshots pin what each path produces today, so
the move can be shown to change nothing; a change that is meant goes in on
its own, with the snapshot updated in the same commit.

    UPDATE_SNAPSHOTS=1 pytest tests/test_evals/test_prompt_snapshots.py

Everything here is invented; the persona file is the public one.
"""

import json
import os
from pathlib import Path

import pytest
import yaml

from lingxi.evals.case import load_case
from lingxi.evals.runner import build_turn

SNAP = Path(__file__).parent / "snapshots"

CASE = {
    "id": "snap", "symptom": "s", "persona": "config/personas/tangkeke.yaml",
    "recipient": "feishu:oc_snap", "clock": "2026-03-13T11:00:00",
    "detect": {"fail": {"any_of": ["x"]}},
    "facts": [
        {"subject": "user:feishu:oc_snap", "type": "pattern", "source": "user_stated",
         "content": "对方在一家小公司做前端，常加班到八点", "days_ago": 30},
        {"subject": "user:feishu:oc_snap", "type": "pattern", "source": "user_stated",
         "content": "对方养了一只叫团子的橘猫", "days_ago": 20},
        {"subject": "user:feishu:oc_snap", "type": "pattern", "source": "user_stated",
         "content": "对方3月14日要去漫展见阿澪", "days_ago": 5},
        {"subject": "aria", "type": "event", "source": "life_simulated",
         "content": "跟香音在排练室把那段副歌的换气对了三遍", "minutes_ago": 90},
        {"subject": "aria", "type": "event", "source": "life_simulated",
         "content": "给姐姐打电话，用上海话聊了半小时", "minutes_ago": 40},
        {"subject": "world", "type": "event", "source": "world_fetch",
         "content": "某新番今天公布了第二季的定档", "minutes_ago": 120},
    ],
    "history": [
        {"role": "assistant", "content": "今天排练室空调又坏了", "minutes_ago": 1500},
        {"role": "user", "content": "那你们怎么练", "minutes_ago": 1490},
        {"role": "assistant", "content": "开窗硬练 汗都滴地上了", "minutes_ago": 1489},
        {"role": "user", "content": "辛苦辛苦，我明天去漫展", "minutes_ago": 1480},
    ],
    "sent_proactive": [{"text": "今早压腿压到腿后侧发紧", "days_ago": 2}],
    "agenda": [{"kind": "his", "text": "对方去漫展见阿澪", "days_ahead": 1}],
}


class _LLM:
    """The orchestrator's answer, and the story curator's."""

    async def complete(self, **kw):
        purpose = kw.get("_debug_purpose", "")
        if purpose == "story_curation":
            body = {"stories": [
                {"text": "跟香音把副歌的换气对了三遍，第三遍终于对上", "aspect": "练习排练"},
                {"text": "给我姐打电话，上海话一开口就停不下来", "aspect": "家人"}]}
        else:
            body = {"engage_level": 0.6, "register": "light", "fact_queries": [],
                    "topic_anchor": "他说明天去漫展", "thread_summary": "聊排练和漫展"}
        return type("R", (), {"content": json.dumps(body, ensure_ascii=False)})()


def _render(system: str, messages: list[dict]) -> str:
    parts = ["=== SYSTEM ===", system]
    for m in messages:
        c = m["content"]
        if not isinstance(c, str):
            c = "\n".join(b.get("text", "") for b in c if b.get("type") == "text")
        parts += [f"=== {m['role'].upper()} ===", c]
    return "\n".join(parts) + "\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,extra", [
    ("reactive", {"input": "在吗 我要出发了"}),
    ("proactive", {"history": CASE["history"][:3]}),
])
async def test_the_assembled_prompt_is_unchanged(tmp_path, mode, extra):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({**CASE, **extra, "mode": mode}, allow_unicode=True))
    system, messages, _ = await build_turn(load_case(path), llm=_LLM())
    got = _render(system, messages)
    snap = SNAP / f"{mode}.txt"
    if os.environ.get("UPDATE_SNAPSHOTS") == "1" or not snap.exists():
        snap.write_text(got, encoding="utf-8")
    assert got == snap.read_text(encoding="utf-8")
