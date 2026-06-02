from datetime import datetime

from lingxi.temporal.proactive import _format_own_life_block
from lingxi.facts.models import Fact, FactType, Source


def _ev(content: str, ts: datetime) -> Fact:
    return Fact(subject="aria", content=content, type=FactType.EVENT,
                source=Source.LIFE_SIMULATED, ts=ts)


def test_own_life_block_formats_recent_events():
    facts = [
        _ev("今晚架望远镜看了仙女座", datetime(2026, 6, 2, 21, 0)),
        _ev("整理了一下午的观测数据", datetime(2026, 6, 2, 15, 0)),
    ]
    block = _format_own_life_block(facts)
    assert "仙女座" in block
    assert "观测数据" in block
    # frames it as HER life + steers the opener toward it
    assert "你自己" in block
    assert "优先从这里" in block


def test_own_life_block_empty_when_no_facts():
    assert _format_own_life_block([]) == ""
