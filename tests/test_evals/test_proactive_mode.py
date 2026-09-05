"""Cases can replay the opener path, not only replies.

The repetition people notice is produced there, and until now the harness
could only replay a reply — so the one path with a known defect was the one
path it could not measure.
"""

import pytest
import yaml

from lingxi.evals.case import Case, load_case
from lingxi.evals.runner import build_turn, score_case


def _case(**over) -> dict:
    base = {
        "id": "t", "symptom": "s", "persona": "config/personas/tangkeke.yaml",
        "recipient": "feishu:oc_t", "clock": "2026-08-30T20:00:00",
        "detect": {"fail": {"any_of": ["x"]}},
        "history": [{"role": "user", "content": "今天上班有点累",
                     "minutes_ago": 900}],
    }
    base.update(over)
    return base


class _StubLLM:
    """Stands in for the orchestrator; the opener path never calls it."""

    async def complete(self, **kw):
        import json
        return type("R", (), {"content": json.dumps(
            {"engage_level": 0.5, "register": "light", "fact_queries": [],
             "topic_anchor": "x", "thread_summary": ""}, ensure_ascii=False)})()


# --- schema ------------------------------------------------------------

def test_mode_defaults_to_reactive():
    assert Case.model_validate(_case(input="在吗")).mode == "reactive"


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        Case.model_validate(_case(mode="sideways", input="在吗"))


def test_a_proactive_case_may_not_carry_an_input():
    """There is no user message in this turn; one written here never renders."""
    with pytest.raises(ValueError, match="proactive"):
        Case.model_validate(_case(mode="proactive", input="在吗"))


def test_sent_openers_resolve_against_the_frozen_clock():
    c = Case.model_validate(_case(
        mode="proactive",
        sent_proactive=[{"text": "你那边忙完了吗", "days_ago": 2}]))

    entry = c.sent_proactive_entries()[0]
    assert entry["text"] == "你那边忙完了吗"
    assert entry["ts"].startswith("2026-08-28T20:00")


# --- assembly ----------------------------------------------------------

@pytest.mark.asyncio
async def test_a_proactive_case_assembles_an_opener_turn(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(_case(mode="proactive"), allow_unicode=True))

    _system, messages, _persona = await build_turn(
        load_case(path), llm=_StubLLM())

    assert "这一刻没有对方的新消息" in messages[-1]["content"]


@pytest.mark.asyncio
async def test_openers_already_sent_reach_the_anti_repeat_block(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(_case(
        mode="proactive",
        sent_proactive=[{"text": "你那边最近忙完了吗", "days_ago": 2}],
    ), allow_unicode=True))

    _system, messages, _persona = await build_turn(
        load_case(path), llm=_StubLLM())

    assert "你那边最近忙完了吗" in messages[-1]["content"], \
        "without these the case cannot see repetition at all"


@pytest.mark.asyncio
async def test_the_schedule_reaches_the_opener(tmp_path):
    """Regression: it used to ride the system prompt, which this path reuses.

    Once it moved to the per-turn block, a path that forgot to place it had
    nothing on whether a show had happened.
    """
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(_case(mode="proactive"), allow_unicode=True))

    _system, messages, _persona = await build_turn(
        load_case(path), llm=_StubLLM())

    assert "接下来你要演的场" in messages[-1]["content"]


# --- sampling ----------------------------------------------------------

@pytest.mark.asyncio
async def test_each_sample_gets_its_own_assembly(tmp_path):
    """Production draws a message style at random per send.

    Assembling once would measure a single style and report it as the path's
    behaviour.
    """
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(
        _case(mode="proactive", samples=6), allow_unicode=True))

    seen = []

    async def _sampler(system, messages, n):
        seen.append(messages[-1]["content"])
        return ["回复"] * n

    await score_case(load_case(path), sampler=_sampler, llm=_StubLLM())

    assert len(seen) == 6, "one assembly per sample"
    styles = {s.split("## 这次试一种语气：")[1][:8] for s in seen
              if "## 这次试一种语气：" in s}
    assert len(styles) > 1, "the styles must actually vary across samples"


@pytest.mark.asyncio
async def test_two_runs_of_one_case_draw_the_same_styles(tmp_path):
    """Seeded — otherwise run-to-run style churn is read as a real change."""
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(
        _case(mode="proactive", samples=5), allow_unicode=True))

    async def _run():
        seen = []

        async def _sampler(system, messages, n):
            seen.append(messages[-1]["content"])
            return ["回复"] * n

        await score_case(load_case(path), sampler=_sampler, llm=_StubLLM())
        return [s.split("## 这次试一种语气：")[1][:8] for s in seen
                if "## 这次试一种语气：" in s]

    assert await _run() == await _run()


@pytest.mark.asyncio
async def test_a_reactive_case_still_assembles_once(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(
        _case(input="在吗", samples=4), allow_unicode=True))

    calls = []

    async def _sampler(system, messages, n):
        calls.append(n)
        return ["回复"] * n

    await score_case(load_case(path), sampler=_sampler, llm=_StubLLM())

    assert calls == [4], "reply cases have no per-sample randomness to cover"
