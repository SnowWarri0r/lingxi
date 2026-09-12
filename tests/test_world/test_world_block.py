"""Today's one outside-the-room thing, pushed into the turn.

The retrieval side was pull-only: the orchestrator was offered world.event in
all 324 logged calls and requested it zero times, through either the
fact_queries path or archival_memory_search. That is what a pull path does
with ambient context — it is by definition not what you need to answer the
question in front of you. The only read the facts ever got was the fetcher's
own "did I already run today?" probe.

So it is pushed, like the weather and the schedule block before it.
"""

from datetime import datetime, timedelta

from lingxi.facts.models import Fact, FactType, Source
from lingxi.persona.prompt_builder import build_world_block


def _fact(content, importance=5, hours_ago=3):
    return Fact(subject="world", content=content, source=Source.WORLD_FETCH,
                type=FactType.EVENT, ts=datetime.now() - timedelta(hours=hours_ago),
                importance=importance)


def test_the_block_carries_the_item():
    block = build_world_block([_fact("上海那边这两天降温了")])

    assert "上海那边这两天降温了" in block


def test_only_one_thing_reaches_her():
    """A digest is a news ticker; one thing is something she noticed."""
    block = build_world_block([
        _fact("第一条", importance=8),
        _fact("第二条", importance=7),
        _fact("第三条", importance=6),
    ])

    assert "第一条" in block
    assert "第二条" not in block and "第三条" not in block


def test_the_one_that_reaches_her_is_the_one_she_cared_about_most():
    block = build_world_block([
        _fact("随便看看的", importance=2),
        _fact("真的在意的", importance=9),
    ])

    assert "真的在意的" in block


def test_nothing_scanned_today_means_no_block():
    assert build_world_block([]) is None
    assert build_world_block(None) is None


def test_the_block_says_it_is_hers_to_mention_or_not():
    """Without this she reads a state line as an instruction to report it."""
    block = build_world_block([_fact("x")])

    assert "不用" in block or "不必" in block
