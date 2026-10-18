"""The re-ask rode in the second bubble, where the whole-message check can't see it.

The one thing he had told her in a week was asked about again in six sent
openers over six days, each behind a fresh first bubble that diluted the
whole-message similarity to 0.44-0.56, under the 0.62 guard. Bubble to
bubble the re-asks sat at 0.653-0.760 against 0.645 for the nearest
non-repeat. Replayed over 40 real openers, 3 lost a bubble — all three the
re-ask — and each kept bubble stood as an opener on its own.
"""

import pytest

from lingxi.temporal.proactive import _drop_repeated_bubbles

ASKED = "你那两天在家拼的是什么乐高 能一坐一下午那种的？"
FRESH = "练习室空调又坏了 今天只能开窗跳"
REASK = "你上回在家拼的那个乐高 拼完没"
OTHER = "名古屋那场还剩三十五天"

_V = {ASKED: [1.0, 0.0, 0.0], REASK: [0.8, 0.6, 0.0],   # cos 0.80
      FRESH: [0.0, 1.0, 0.0], OTHER: [0.0, 0.0, 1.0]}


class _Emb:
    def __init__(self):
        self.batches = 0

    async def embed(self, t):
        return _V[t]

    async def embed_batch(self, ts):
        self.batches += 1
        return [_V[t] for t in ts]


@pytest.mark.asyncio
async def test_a_repeat_in_the_second_bubble_is_dropped():
    left, dropped = await _drop_repeated_bubbles(f"{FRESH}\n\n{REASK}", [ASKED], _Emb())

    assert dropped == [REASK] and left == FRESH


@pytest.mark.asyncio
async def test_fresh_bubbles_are_all_kept():
    msg = f"{FRESH}\n\n{OTHER}"
    left, dropped = await _drop_repeated_bubbles(msg, [ASKED], _Emb())

    assert dropped == [] and left == msg


@pytest.mark.asyncio
async def test_an_opener_that_is_only_a_repeat_leaves_nothing():
    left, dropped = await _drop_repeated_bubbles(REASK, [ASKED], _Emb())

    assert left == "" and dropped == [REASK]


@pytest.mark.asyncio
async def test_a_repeat_hidden_in_an_earlier_openers_second_bubble_is_found():
    left, _ = await _drop_repeated_bubbles(REASK, [f"{OTHER}\n\n{ASKED}"], _Emb())

    assert left == ""


@pytest.mark.asyncio
async def test_it_abstains_without_an_embedder():
    msg = f"{FRESH}\n\n{REASK}"
    assert await _drop_repeated_bubbles(msg, [ASKED], None) == (msg, [])


@pytest.mark.asyncio
async def test_vectors_are_fetched_in_one_call():
    emb = _Emb()
    await _drop_repeated_bubbles(f"{FRESH}\n\n{REASK}", [ASKED, OTHER], emb)

    assert emb.batches == 1


# --- a repeated first bubble sinks the opener -----------------------------------

from datetime import datetime, timedelta  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from lingxi.channels.outbound import ChannelRegistry, OutboundChannel  # noqa: E402
from lingxi.temporal.proactive import ProactiveConfig, ProactiveScheduler  # noqa: E402
from lingxi.temporal.tracker import InteractionTracker  # noqa: E402

FOLLOW = "副歌进那个拍 你听出来没有嘛"
_V[FOLLOW] = [0.0, 0.6, 0.8]


class _Ch(OutboundChannel):
    def __init__(self):
        self.sent = []

    @property
    def channel_name(self):
        return "feishu"

    async def send_message(self, recipient_id, text, turn_id=None):
        self.sent.append(text)

    async def send_sticker(self, recipient_id, file_path):
        pass


@pytest.mark.asyncio
async def test_a_follow_up_is_not_sent_without_the_bubble_it_hangs_off(tmp_path):
    """2026-10-13: the lead bubble repeated the morning's opener and was
    dropped; 「副歌进那个拍 你听出来没有嘛」 went out alone, about no song."""
    tracker = InteractionTracker(tmp_path)
    rec = tracker.record_interaction("feishu", "oc_x")
    rec.last_interaction = datetime.now() - timedelta(days=2)
    ch = _Ch()
    reg = ChannelRegistry()
    reg.register(ch)

    async def _append(*a):
        pass

    engine = SimpleNamespace(
        annotation_store=None, _pending_stickers={},
        memory=SimpleNamespace(embedding_provider=_Emb(),
                               short_term=SimpleNamespace(append_for_recipient=_append)))
    sched = ProactiveScheduler(config=ProactiveConfig(), tracker=tracker,
                               channel_registry=reg, engine=engine)
    sched._recent_proactive["feishu:oc_x"] = [{"text": ASKED, "ts": None}]

    async def _ask_llm(record, silence, now, force=False):
        return {"should_send": True, "message": f"{REASK}\n\n{FOLLOW}",
                "sticker": "", "reason": ""}

    sched._ask_llm = _ask_llm
    result = await sched._maybe_reach_out(rec, datetime.now(), force=True)

    assert result["reason"] == "lead_bubble_repeated"
    assert ch.sent == []


def test_openers_are_read_as_him_before_they_go():
    from lingxi.facts.models import Fact, FactType, Source
    from lingxi.temporal.proactive import _format_own_life_block
    block = _format_own_life_block([Fact(
        subject="aria", content="词本上写了三个字", type=FactType.EVENT,
        source=Source.LIFE_SIMULATED, ts=datetime(2026, 10, 13, 21))])
    assert "替他读一遍" in block


def test_her_moments_are_written_to_be_retold():
    from lingxi.planner.executor import _SYSTEM_TMPL
    assert "跟朋友讲" in _SYSTEM_TMPL and "给自己看" not in _SYSTEM_TMPL


@pytest.mark.asyncio
async def test_a_sent_opener_marks_its_story_told(tmp_path):
    tracker = InteractionTracker(tmp_path)
    rec = tracker.record_interaction("feishu", "oc_x")
    rec.last_interaction = datetime.now() - timedelta(days=2)
    reg = ChannelRegistry()
    reg.register(_Ch())

    async def _append(*a):
        pass

    engine = SimpleNamespace(
        annotation_store=None, _pending_stickers={},
        memory=SimpleNamespace(embedding_provider=None,
                               short_term=SimpleNamespace(append_for_recipient=_append)))
    sched = ProactiveScheduler(config=ProactiveConfig(), tracker=tracker,
                               channel_registry=reg, engine=engine)
    told = []
    sched._story_shelf = SimpleNamespace(mark_told=lambda k, sid: told.append((k, sid)))
    sched._story_ids = {"feishu:oc_x": {"S1": "abc"}}

    async def _ask_llm(record, silence, now, force=False):
        return {"should_send": True, "message": FRESH, "sticker": "",
                "reason": "", "story": "S1"}

    sched._ask_llm = _ask_llm
    result = await sched._maybe_reach_out(rec, datetime.now(), force=True)
    assert result["status"] == "sent"
    assert told == [("feishu:oc_x", "abc")]
