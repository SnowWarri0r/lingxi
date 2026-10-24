"""Openers rotate across sides of her life in code, not by request.

Asked to pick another aspect, she still picked practice: 9 of 14 openers
after 10-10 were rehearsal details, with news and teammate stories beside
them on the shelf.
"""

from lingxi.planner.stories import Story
from lingxi.temporal.proactive import rotate_stories

SHELF = [Story("a", "压腿压到腿后侧发紧", "练习排练"),
         Story("b", "新番官宣动画化了", "圈里新鲜事"),
         Story("c", "菫又来抢我可乐", "队友")]


def test_the_aspects_of_the_last_two_openers_are_held_back():
    recent = [{"aspect": "练习排练"}, {"aspect": "队友"}]
    assert [s.id for s in rotate_stories(SHELF, recent)] == ["b"]


def test_an_older_opener_does_not_hold_anything_back():
    recent = [{"aspect": "练习排练"}, {"aspect": "天气"}, {"aspect": "吃喝"}]
    assert [s.id for s in rotate_stories(SHELF, recent)] == ["a", "b", "c"]


def test_when_everything_was_just_used_she_still_has_something():
    recent = [{"aspect": "练习排练"}, {"aspect": "练习排练"}]
    only = [SHELF[0]]
    assert rotate_stories(only, recent) == only


def test_news_is_offered_even_right_after_news():
    """He answered news 4 of 7 and practice 2 of 12 (10-10 to 10-27)."""
    recent = [{"aspect": "圈里新鲜事"}, {"aspect": "练习排练"}]
    assert [s.id for s in rotate_stories(SHELF, recent)] == ["b", "c"]


# --- a story she has already told him leaves the shelf, whatever its id ------

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from lingxi.planner.stories import StoryShelf
from lingxi.temporal.proactive import already_said

NOW = datetime(2026, 3, 10, 16, 0)


class _AxisEmb:
    """Texts sharing a keyword point the same way."""
    KEYS = ("压腿", "官宣", "可乐")

    async def embed_batch(self, texts):
        return [[1.0 if k in t else 0.0 for k in self.KEYS] + [0.1] for t in texts]


@pytest.mark.asyncio
async def test_a_story_retold_in_new_words_is_found():
    said = ["早上压腿压了一个钟头 腿后侧现在还紧"]
    assert [s.id for s in await already_said(SHELF, said, _AxisEmb())] == ["a"]
    assert await already_said(SHELF, [], _AxisEmb()) == []


def _sched(tmp_path, recent, shelf_stories):
    from lingxi.temporal.proactive import ProactiveConfig, ProactiveScheduler
    from lingxi.temporal.tracker import InteractionTracker
    from lingxi.channels.outbound import ChannelRegistry
    shelf = StoryShelf(tmp_path / "stories.json", None, None)
    shelf._shelf.stories = [{"id": s.id, "text": s.text, "aspect": s.aspect}
                            for s in shelf_stories]
    shelf._shelf.generated_at = NOW.isoformat()

    async def _refresh(events, world, now):
        return None

    async def _world():
        return []

    shelf.refresh = _refresh
    engine = SimpleNamespace(_todays_world_facts=_world,
                             memory=SimpleNamespace(embedding_provider=_AxisEmb()))
    sched = ProactiveScheduler(config=ProactiveConfig(),
                               tracker=InteractionTracker(tmp_path),
                               channel_registry=ChannelRegistry(), engine=engine)
    sched._story_shelf = shelf
    sched._recent_proactive = {"k": recent}
    return sched, shelf


@pytest.mark.asyncio
async def test_the_shelf_drops_what_she_already_said_and_rotates(tmp_path):
    recent = [{"text": "早上压腿压了一个钟头", "aspect": "",
               "ts": (NOW - timedelta(hours=5)).isoformat()},
              {"text": "香音今天又迟到了", "aspect": "队友",
               "ts": (NOW - timedelta(hours=1)).isoformat()}]
    sched, shelf = _sched(tmp_path, recent, SHELF)

    block = await sched._stories_block("k", [], NOW)

    assert [s.id for s in sched._offered_stories["k"]] == ["b"]
    assert "官宣" in block and "压腿" not in block
    assert {s.id for s in shelf.untold("k")} == {"b", "c"}


@pytest.mark.asyncio
async def test_an_opener_from_last_week_does_not_hold_a_story_back(tmp_path):
    recent = [{"text": "早上压腿压了一个钟头", "aspect": "练习排练",
               "ts": (NOW - timedelta(days=9)).isoformat()},
              {"text": "别的", "aspect": "", "ts": (NOW - timedelta(days=9)).isoformat()},
              {"text": "别的", "aspect": "", "ts": (NOW - timedelta(days=9)).isoformat()}]
    sched, _ = _sched(tmp_path, recent, SHELF)
    await sched._stories_block("k", [], NOW)
    assert [s.id for s in sched._offered_stories["k"]] == ["a", "b", "c"]
