"""The simulated day must include other people and outward events.

Regression: the planner only ever saw her own reflections, so it scheduled a
fully solipsistic day (solo body-awareness drills end to end). That is both
off-character for a group-defined persona and leaves her with nothing to say
to anyone — every proactive opener became a micro-sensation report.
"""
import inspect

import yaml

from lingxi.planner import daily_planner
from lingxi.planner.daily_planner import _PLAN_PROMPT, _format_people
from lingxi.planner import executor
from lingxi.persona.models import PersonaConfig


def _persona(path="config/personas/tangkeke.yaml"):
    return PersonaConfig(**yaml.safe_load(open(path)))


def test_plan_prompt_asks_for_people_and_outward_items():
    assert "{people}" in _PLAN_PROMPT
    assert "一天里有别人也有外面的世界" in _PLAN_PROMPT


def test_people_block_renders_persona_recurring_people():
    block = _format_people(_persona())
    assert "香音" in block          # her group is on the page
    assert "平安名菫" in block


def test_people_block_degrades_gracefully_without_persona():
    assert _format_people(None) == "（暂无）"

    class NoBio:
        pass
    assert _format_people(NoBio()) == "（暂无）"


def test_planner_passes_people_into_the_prompt():
    src = inspect.getsource(daily_planner.DailyPlanner)
    assert "self._people_block = _format_people(persona)" in src
    assert "people=self._people_block" in src


def test_executor_moment_uses_life_scale_and_ordinary_volume():
    """The zoom-in ratchet ('make it differ from last time') is gone; scale and
    volume anchors take its place."""
    p = executor._MOMENT_PROMPT
    assert "取景放在正常生活的尺度上" in p
    assert "音量按事情本身来" in p
    assert "接着刚才往下走" in p        # continuity kept (anti-repetition)
    assert "让这一刻和刚才不一样" not in p  # the escalation driver is gone
