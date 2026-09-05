"""The annotation queue: worst first, and never spend attention twice.

The voice-anchor pool holds 43 generic seeds and 2 approved lines while 198
recorded turns sit unannotated. Nothing tunable substitutes for having
anchors, so the bottleneck is attention and this is what aims it.
"""

import json
from datetime import datetime

import pytest

from lingxi.fewshot.queue import rank_turns, surface_markers
from lingxi.fewshot.queue_cli import load_unannotated


class _Turn:
    def __init__(self, turn_id, speech, user_message=""):
        self.turn_id = turn_id
        self.speech = speech
        self.user_message = user_message


class _Provider:
    """Answers with a canned array; records the prompts it was given."""

    def __init__(self, payload):
        self._payload = payload
        self.prompts = []

    async def complete(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])
        body = self._payload
        text = body(len(self.prompts) - 1) if callable(body) else body
        return type("R", (), {"content": text})()


class _Boom:
    async def complete(self, **kwargs):
        raise RuntimeError("ranker unavailable")


TURNS = [_Turn("a", "辛苦了～早点休息呀"), _Turn("b", "诶 那挺好"),
         _Turn("c", "其实这就是生活的意义呀！！！")]


# --- ranking -----------------------------------------------------------

@pytest.mark.asyncio
async def test_the_worst_turn_comes_first():
    p = _Provider('[{"i":0,"s":7,"why":"客套"},'
                  '{"i":1,"s":1,"why":"像真人"},'
                  '{"i":2,"s":9,"why":"升华"}]')

    ranked = await rank_turns(TURNS, p)

    assert [r.turn_id for r in ranked] == ["c", "a", "b"]


@pytest.mark.asyncio
async def test_the_reason_travels_with_the_turn():
    p = _Provider('[{"i":0,"s":7,"why":"客套"},{"i":1,"s":1,"why":"像真人"}]')

    ranked = await rank_turns(TURNS[:2], p)

    assert ranked[0].why == "客套"


@pytest.mark.asyncio
async def test_turns_the_model_skipped_are_dropped_not_defaulted():
    """A default score sorts into the middle and eats the attention."""
    p = _Provider('[{"i":0,"s":7,"why":"客套"}]')

    ranked = await rank_turns(TURNS, p)

    assert [r.turn_id for r in ranked] == ["a"]


@pytest.mark.asyncio
async def test_scoring_happens_in_batches():
    p = _Provider(lambda n: json.dumps(
        [{"i": i, "s": 5, "why": "x"} for i in range(2)]))

    await rank_turns(TURNS, p, batch=2)

    assert len(p.prompts) == 2


@pytest.mark.asyncio
async def test_a_failed_batch_does_not_lose_the_others():
    calls = {"n": 0}

    class _Flaky:
        async def complete(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("timeout")
            return type("R", (), {"content": '[{"i":0,"s":8,"why":"客套"}]'})()

    ranked = await rank_turns(TURNS, _Flaky(), batch=2)

    assert [r.turn_id for r in ranked] == ["c"]


@pytest.mark.asyncio
async def test_an_unusable_reply_scores_nothing_rather_than_raising():
    assert await rank_turns(TURNS, _Provider("抱歉，我无法完成")) == []


@pytest.mark.asyncio
async def test_a_dead_ranker_is_not_a_crash():
    assert await rank_turns(TURNS, _Boom()) == []


@pytest.mark.asyncio
async def test_blank_turns_are_not_sent_for_scoring():
    p = _Provider('[{"i":1,"s":4,"why":"x"}]')

    await rank_turns([_Turn("a", "   "), _Turn("b", "诶 那挺好")], p)

    assert "0. " not in p.prompts[0]


# --- surface markers ---------------------------------------------------

def test_markers_flag_the_measured_tells():
    assert "客套" in surface_markers("辛苦了 早点休息")
    assert "波浪" in surface_markers("好呀～")
    assert "升华" in surface_markers("其实这就是生活呀")


def test_a_plain_line_carries_no_markers():
    assert surface_markers("诶 那挺好") == []


# --- the queue ---------------------------------------------------------

def test_already_annotated_turns_are_not_shown_again(tmp_path):
    """`annotation` defaults to the STRING "none" — truth-testing it skips
    every unannotated turn, i.e. the entire queue."""
    for tid, ann in (("a", "none"), ("b", "positive")):
        (tmp_path / f"{tid}.json").write_text(json.dumps({
            "turn_id": tid, "recipient_key": "k", "user_message": "",
            "inner_thought": "", "speech": "话",
            "created_at": datetime(2026, 9, 3).isoformat(),
            "annotation": ann, "correction": None,
        }), encoding="utf-8")

    assert [t.turn_id for t in load_unannotated(tmp_path)] == ["a"]


def test_an_unreadable_file_does_not_stop_the_queue(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "ok.json").write_text(json.dumps({
        "turn_id": "ok", "recipient_key": "k", "user_message": "",
        "inner_thought": "", "speech": "话",
        "created_at": datetime(2026, 9, 3).isoformat(),
        "annotation": "none", "correction": None,
    }), encoding="utf-8")

    assert [t.turn_id for t in load_unannotated(tmp_path)] == ["ok"]


def test_newest_turns_are_offered_first(tmp_path):
    for tid, day in (("old", 1), ("new", 3)):
        (tmp_path / f"{tid}.json").write_text(json.dumps({
            "turn_id": tid, "recipient_key": "k", "user_message": "",
            "inner_thought": "", "speech": "话",
            "created_at": datetime(2026, 9, day).isoformat(),
            "annotation": "none", "correction": None,
        }), encoding="utf-8")

    assert [t.turn_id for t in load_unannotated(tmp_path)] == ["new", "old"]
