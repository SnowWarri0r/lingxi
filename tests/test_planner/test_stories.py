"""Her days reach him as stories he can follow, each told once."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from lingxi.planner.stories import (
    StoryShelf, format_stories_block, parse_stories)
from lingxi.temporal.proactive import OPENER_ASPECTS, parse_proactive_output

NOW = datetime(2026, 10, 13, 20)
PERSONA = SimpleNamespace(name="可可", upcoming_shows=[], lexicon=[])


def _ev(text, minutes_ago):
    return SimpleNamespace(content=text, ts=NOW - timedelta(minutes=minutes_ago))


class _LLM:
    def __init__(self, stories):
        self.stories, self.calls = stories, []

    async def complete(self, **kw):
        self.calls.append(kw["messages"][0]["content"])
        import json
        return SimpleNamespace(content=json.dumps(
            {"stories": [{"text": t, "aspect": a} for t, a in self.stories]},
            ensure_ascii=False))


@pytest.fixture(autouse=True)
def _self_context(monkeypatch):
    import lingxi.persona.self_context as sc
    monkeypatch.setattr(sc, "build_self_context", lambda p: "你是可可。")


STORIES = [("今天跟香音对词，换我先唱她跟，她第一遍就跟上了", "练习排练"),
           ("蓮ノ空新单今天上线了，副歌那一拍我很喜欢", "圈里新鲜事")]


@pytest.mark.asyncio
async def test_it_curates_from_her_diary_and_the_scan(tmp_path):
    llm = _LLM(STORIES)
    shelf = StoryShelf(tmp_path / "stories.json", llm, PERSONA)
    got = await shelf.refresh([_ev("那俩圈还在盯着我", 30)],
                              [SimpleNamespace(content="新单上线")], NOW)
    assert [s.text for s in got] == [t for t, _ in STORIES]
    assert "那俩圈还在盯着我" in llm.calls[0] and "新单上线" in llm.calls[0]


@pytest.mark.asyncio
async def test_it_is_cached_until_her_diary_moves_on(tmp_path):
    llm = _LLM(STORIES)
    shelf = StoryShelf(tmp_path / "stories.json", llm, PERSONA)
    events = [_ev("a", 30)]
    await shelf.refresh(events, [], NOW)
    await shelf.refresh(events, [], NOW + timedelta(hours=2))
    assert len(llm.calls) == 1
    events.append(SimpleNamespace(content="b", ts=NOW + timedelta(minutes=10)))
    await shelf.refresh(events, [], NOW + timedelta(minutes=20))
    assert len(llm.calls) == 1, "too soon after the last curation"
    await shelf.refresh(events, [], NOW + timedelta(hours=1))
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_a_told_story_is_not_offered_to_that_person_again(tmp_path):
    shelf = StoryShelf(tmp_path / "stories.json", _LLM(STORIES), PERSONA)
    first = (await shelf.refresh([_ev("a", 30)], [], NOW))[0]
    shelf.mark_told("feishu:oc_x", first.id)
    assert first.id not in {s.id for s in shelf.untold("feishu:oc_x")}
    assert first.id in {s.id for s in shelf.untold("feishu:oc_y")}
    # and it survives a restart
    again = StoryShelf(tmp_path / "stories.json", _LLM([]), PERSONA)
    assert first.id not in {s.id for s in again.untold("feishu:oc_x")}


@pytest.mark.asyncio
async def test_a_story_kept_word_for_word_keeps_its_id(tmp_path):
    """So a story told yesterday is still recognised after a refresh."""
    shelf = StoryShelf(tmp_path / "stories.json", _LLM(STORIES), PERSONA)
    a = await shelf.refresh([_ev("a", 30)], [], NOW)
    shelf._shelf.generated_at = (NOW - timedelta(days=1)).isoformat()
    b = await shelf.refresh([_ev("a", 30)], [], NOW)
    assert [s.id for s in a] == [s.id for s in b]


@pytest.mark.asyncio
async def test_no_diary_no_call(tmp_path):
    llm = _LLM(STORIES)
    shelf = StoryShelf(tmp_path / "stories.json", llm, PERSONA)
    assert await shelf.refresh([_ev("old", 60 * 48)], [], NOW) == []
    assert llm.calls == []


def test_unknown_aspects_and_empty_stories_are_dropped():
    got = parse_stories('{"stories": [{"text": "x", "aspect": "乱写"}, '
                        '{"text": "", "aspect": "天气"}]}', OPENER_ASPECTS)
    assert [(s.text, s.aspect) for s in got] == [("x", "")]


def test_the_block_numbers_the_stories_for_her_to_name():
    stories = parse_stories('{"stories": [{"text": "x", "aspect": "天气"}]}',
                            OPENER_ASPECTS)
    block, ids = format_stories_block(stories)
    assert "S1｜天气：x" in block and ids == {"S1": stories[0].id}


def test_the_opener_names_the_story_it_told():
    out = parse_proactive_output('今天跟香音对词了！\n===META===\n'
                                 '{"should_send": true, "story": "s2"}')
    assert out["story"] == "S2"
