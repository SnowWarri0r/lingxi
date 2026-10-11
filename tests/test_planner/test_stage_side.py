"""Her own shows are read from the stage, in the scan and in the plan.

The morning scan re-voiced a lottery for a show she performs in as a fan's
「抽到了我直接飞过去」, and the planner acted on it: replayed on a real day,
13 of 44 plan items had her entering the lottery or studying the seating.
With the stage block in both, 0 of 46.
"""

from datetime import date
from types import SimpleNamespace

from lingxi.persona.prompt_builder import build_stage_block
from lingxi.world.fetcher import build_fetch_prompt

SHOW = SimpleNamespace(event="冬季巡演终场", date="2026-12", venue="某体育馆", note="")
PERSONA = SimpleNamespace(
    name="可可", upcoming_shows=[SHOW], world_interests=["偶像圈"],
    lexicon=["【她的站位】有自己团的场，她在台上唱", "【台下】推し：最喜欢的那一个"],
    location=None)
TODAY = date(2026, 10, 1)


def test_the_block_names_her_shows_and_her_side_of_the_stage():
    block = build_stage_block(PERSONA, TODAY)
    assert "冬季巡演终场" in block and "某体育馆" in block
    assert "我是台上的人" in block
    assert "有自己团的场，她在台上唱" in block
    assert "推し" not in block


def test_past_shows_are_left_out():
    assert build_stage_block(PERSONA, date(2027, 2, 1), stance=False) is None


def test_a_persona_with_no_stage_gets_nothing():
    assert build_stage_block(SimpleNamespace(upcoming_shows=[], lexicon=[]), TODAY) is None


def test_the_scan_reads_her_shows_from_the_stage(monkeypatch):
    import lingxi.persona.self_context as sc
    monkeypatch.setattr(sc, "build_self_context", lambda p: "你是可可。")
    prompt = build_fetch_prompt(PERSONA, TODAY)
    assert "她自己要上台的场" in prompt and "冬季巡演终场" in prompt


def test_the_stage_block_comes_after_the_language_rule(monkeypatch):
    """Placed between her self-description and 「用上面那段自我描述所用的语言
    写」, its venue names were read as part of that description, and the
    re-voiced items came out in Japanese (7 of 16)."""
    import lingxi.persona.self_context as sc
    monkeypatch.setattr(sc, "build_self_context", lambda p: "你是可可。")
    prompt = build_fetch_prompt(PERSONA, TODAY)
    assert prompt.index("所用的语言写") < prompt.index("她自己要上台的场")


def test_the_scan_leaves_out_the_stance_line(monkeypatch):
    import lingxi.persona.self_context as sc
    monkeypatch.setattr(sc, "build_self_context", lambda p: "你是可可。")
    assert "有自己团的场" not in build_fetch_prompt(PERSONA, TODAY)


def test_the_planner_prompt_carries_the_block():
    from lingxi.planner import daily_planner
    assert "{stage}{outside}" in daily_planner._PLAN_PROMPT
