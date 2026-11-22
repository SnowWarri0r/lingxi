"""The transcript keeps what the 30-turn buffer evicts."""

from datetime import datetime, timedelta

import pytest

from lingxi.memory.short_term import ShortTermMemory
from lingxi.memory.transcript import TranscriptLog


@pytest.mark.asyncio
async def test_turns_the_buffer_evicts_stay_in_the_transcript(tmp_path):
    stm = ShortTermMemory(max_turns=4, data_dir=tmp_path)
    await stm.switch_recipient("feishu:oc_x")
    for i in range(10):
        stm.add_turn("user" if i % 2 == 0 else "assistant", f"第{i}句")

    assert stm.turn_count == 4
    rows = stm.transcript.read("feishu:oc_x")
    assert [r["content"] for r in rows] == [f"第{i}句" for i in range(10)]
    assert rows[0]["role"] == "user" and rows[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_openers_appended_from_outside_are_recorded_too(tmp_path):
    stm = ShortTermMemory(data_dir=tmp_path)
    await stm.switch_recipient("feishu:oc_x")
    stm.add_turn("user", "周末去漫展")
    await stm.append_for_recipient("feishu:oc_x", "assistant", "漫展好玩吗")
    await stm.append_for_recipient("qq:other", "assistant", "团子今天又胖了")

    assert [r["content"] for r in stm.transcript.read("feishu:oc_x")] == [
        "周末去漫展", "漫展好玩吗"]
    assert [r["content"] for r in stm.transcript.read("qq:other")] == ["团子今天又胖了"]


@pytest.mark.asyncio
async def test_a_log_begun_mid_conversation_starts_from_the_buffer(tmp_path):
    first = ShortTermMemory(data_dir=tmp_path)
    first.transcript = None          # a buffer written before the log existed
    await first.switch_recipient("feishu:oc_x")
    first.add_turn("user", "小林说要搬家")
    first.add_turn("assistant", "搬去哪呀")
    await first.persist_current()

    stm = ShortTermMemory(data_dir=tmp_path)
    await stm.switch_recipient("feishu:oc_x")
    stm.add_turn("user", "搬到隔壁区")
    stm.add_turn("assistant", "那还挺近")

    assert [r["content"] for r in stm.transcript.read("feishu:oc_x")] == [
        "小林说要搬家", "搬去哪呀", "搬到隔壁区", "那还挺近"]


def test_read_filters_by_time_across_months(tmp_path):
    log = TranscriptLog(tmp_path)

    class T:
        def __init__(self, ts, content):
            self.timestamp, self.role, self.content = ts, "user", content

    base = datetime(2026, 10, 31, 23, 0)
    for h, text in ((0, "十月底"), (2, "十一月初"), (26, "第二天")):
        log.append("feishu:oc_x", T(base + timedelta(hours=h), text))

    assert len(list((tmp_path / "transcripts" / "feishu:oc_x").glob("*.jsonl"))) == 2
    got = log.read("feishu:oc_x", since=datetime(2026, 11, 1),
                   until=datetime(2026, 11, 2))
    assert [r["content"] for r in got] == ["十一月初"]


def test_a_write_that_fails_does_not_raise(tmp_path):
    (tmp_path / "transcripts").write_text("not a directory")
    log = TranscriptLog(tmp_path)

    class T:
        timestamp, role, content = datetime(2026, 11, 1), "user", "x"

    log.append("feishu:oc_x", T())   # prints, does not raise
    assert log.read("feishu:oc_x") == []
