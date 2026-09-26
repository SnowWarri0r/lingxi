"""A message that did not arrive was being counted as one she sent.

Feishu answers a failed send with HTTP 200 and the failure in the body. The
card path checked the body and raised; the plain-text fallback behind it
never read the response at all, so whenever the card failed, the fallback
decided the outcome and always reported success.

It surfaced through the one step that did check: a sticker, sent after an
opener on 2026-09-22 20:27, came back 230002 — the bot is not in that chat.
That chat had one 24-minute conversation on 08-18 and nothing since, and by
then held 62 proactive messages, every one counted as delivered, appended to
her memory as something she said, and set to be quoted back as the number
of times he had left her unanswered.

Making the failure visible is not enough by itself. A failed send leaves
last_proactive_sent where it was, so the recipient is eligible again on the
next five-minute tick: compose, refuse, compose, refuse.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from lingxi.channels.outbound import (
    ChannelRegistry,
    OutboundChannel,
    RecipientUnreachable,
)
from lingxi.temporal.proactive import ProactiveConfig, ProactiveScheduler
from lingxi.temporal.tracker import InteractionTracker


OPENER = "练习室的空调又坏了 今天只能开窗跳"


class _Gone(OutboundChannel):
    """What Feishu is for a chat the bot has been removed from."""

    def __init__(self):
        self.attempts = 0
        self.stickers = 0

    @property
    def channel_name(self):
        return "feishu"

    async def send_message(self, recipient_id, text, turn_id=None):
        self.attempts += 1
        raise RecipientUnreachable("send text: 230002 Bot/User can NOT be out of the chat.")

    async def send_sticker(self, recipient_id, file_path):
        self.stickers += 1


class _Flaky(_Gone):
    """An ordinary failure — the next attempt might well work."""

    async def send_message(self, recipient_id, text, turn_id=None):
        self.attempts += 1
        raise RuntimeError("send text failed: {'code': 230020}")


class _Working(_Gone):
    async def send_message(self, recipient_id, text, turn_id=None):
        self.attempts += 1


def _scheduler(tmp_path, channel):
    tracker = InteractionTracker(tmp_path)
    rec = tracker.record_interaction("feishu", "oc_gone")
    rec.last_interaction = datetime.now() - timedelta(days=36)
    rec.relationship_level = 1

    appended = []

    async def _append(key, role, text):
        appended.append(text)

    engine = SimpleNamespace(
        annotation_store=None,
        memory=SimpleNamespace(
            embedding_provider=None,
            short_term=SimpleNamespace(append_for_recipient=_append)),
        _pending_stickers={},
    )
    reg = ChannelRegistry()
    reg.register(channel)
    sched = ProactiveScheduler(
        config=ProactiveConfig(enabled=True), tracker=tracker,
        channel_registry=reg, engine=engine)

    composed = {"n": 0}

    async def _ask_llm(record, silence, now, force=False):
        composed["n"] += 1
        return {"should_send": True, "message": OPENER, "sticker": "", "reason": ""}

    sched._ask_llm = _ask_llm
    return sched, tracker, composed, appended


def _rec(tracker):
    return tracker._records["feishu:oc_gone"]


@pytest.mark.asyncio
async def test_an_undelivered_message_is_not_counted_as_sent(tmp_path):
    sched, tracker, _, _ = _scheduler(tmp_path, _Gone())
    before = _rec(tracker).consecutive_proactive_count

    result = await sched._maybe_reach_out(_rec(tracker), datetime.now())

    assert result["status"] == "unreachable"
    assert _rec(tracker).consecutive_proactive_count == before


@pytest.mark.asyncio
async def test_she_does_not_remember_saying_what_never_arrived(tmp_path):
    """The short-term buffer is what the next reply reads as her last turn."""
    sched, tracker, _, appended = _scheduler(tmp_path, _Gone())

    await sched._maybe_reach_out(_rec(tracker), datetime.now())

    assert appended == []


@pytest.mark.asyncio
async def test_no_sticker_follows_a_message_that_did_not_go(tmp_path):
    ch = _Gone()
    sched, tracker, _, _ = _scheduler(tmp_path, ch)

    await sched._maybe_reach_out(_rec(tracker), datetime.now())

    assert ch.stickers == 0


@pytest.mark.asyncio
async def test_the_next_tick_composes_nothing(tmp_path):
    """The loop this closes: eligible again five minutes later, forever."""
    ch = _Gone()
    sched, tracker, composed, _ = _scheduler(tmp_path, ch)

    await sched._maybe_reach_out(_rec(tracker), datetime.now())
    later = datetime.now() + timedelta(minutes=5)
    result = await sched._maybe_reach_out(_rec(tracker), later)

    assert result["status"] == "skipped_unreachable"
    assert composed["n"] == 1 and ch.attempts == 1


@pytest.mark.asyncio
async def test_it_stays_stopped_a_week_later(tmp_path):
    ch = _Gone()
    sched, tracker, composed, _ = _scheduler(tmp_path, ch)

    await sched._maybe_reach_out(_rec(tracker), datetime.now())
    for days in (1, 3, 7):
        await sched._maybe_reach_out(_rec(tracker),
                                     datetime.now() + timedelta(days=days))

    assert composed["n"] == 1


@pytest.mark.asyncio
async def test_hearing_from_them_is_what_reopens_it(tmp_path):
    sched, tracker, _, _ = _scheduler(tmp_path, _Gone())
    await sched._maybe_reach_out(_rec(tracker), datetime.now())
    assert _rec(tracker).unreachable_since is not None

    tracker.record_interaction("feishu", "oc_gone")

    assert _rec(tracker).unreachable_since is None


@pytest.mark.asyncio
async def test_the_state_survives_a_restart(tmp_path):
    """Otherwise every restart buys the dead chat another message."""
    sched, tracker, _, _ = _scheduler(tmp_path, _Gone())
    await sched._maybe_reach_out(_rec(tracker), datetime.now())

    reloaded = InteractionTracker(tmp_path)
    await reloaded.load()

    assert reloaded._records["feishu:oc_gone"].unreachable_since is not None


@pytest.mark.asyncio
async def test_a_manual_trigger_still_tries(tmp_path):
    """How you find out whether re-adding the bot worked."""
    ch = _Gone()
    sched, tracker, _, _ = _scheduler(tmp_path, ch)
    await sched._maybe_reach_out(_rec(tracker), datetime.now())

    await sched._maybe_reach_out(_rec(tracker), datetime.now(), force=True)

    assert ch.attempts == 2


class TestAnOrdinaryFailureIsNotAVerdict:
    @pytest.mark.asyncio
    async def test_a_transient_failure_does_not_mark_them_gone(self, tmp_path):
        sched, tracker, _, _ = _scheduler(tmp_path, _Flaky())

        result = await sched._maybe_reach_out(_rec(tracker), datetime.now())

        assert result["status"] == "send_failed"
        assert _rec(tracker).unreachable_since is None

    @pytest.mark.asyncio
    async def test_it_is_not_counted_either(self, tmp_path):
        sched, tracker, _, _ = _scheduler(tmp_path, _Flaky())
        before = _rec(tracker).consecutive_proactive_count

        await sched._maybe_reach_out(_rec(tracker), datetime.now())

        assert _rec(tracker).consecutive_proactive_count == before


@pytest.mark.asyncio
async def test_a_delivered_message_is_still_counted(tmp_path):
    ch = _Working()
    sched, tracker, _, appended = _scheduler(tmp_path, ch)
    before = _rec(tracker).consecutive_proactive_count

    result = await sched._maybe_reach_out(_rec(tracker), datetime.now())

    assert result["status"] == "sent"
    assert _rec(tracker).consecutive_proactive_count == before + 1
    assert appended == [OPENER]


class TestWhatSheSaysWhenTheyComeBack:
    """The unanswered count is read on their return, and quoted to them."""

    async def _prompt(self, tmp_path, monkeypatch, *, unreachable):
        from pathlib import Path

        from lingxi.brain import orchestrator as orch_mod
        from lingxi.brain import renderer as rend_mod
        from lingxi.brain.models import OrchestrationDecision
        from lingxi.conversation.engine import ConversationEngine
        from lingxi.facts.retriever import FactRetriever
        from lingxi.facts.store import FactStore
        from lingxi.memory.manager import MemoryManager
        from lingxi.persona.models import Identity, PersonaConfig

        async def _decide(*a, **k):
            return OrchestrationDecision(register="light", engage_level=0.5,
                                         fact_queries=[], topic_anchor="")

        async def _render(*a, **k):
            return ""

        monkeypatch.setattr(orch_mod, "decide", _decide)
        monkeypatch.setattr(rend_mod, "render_dynamic_blocks", _render)

        store = FactStore(Path(tmp_path) / "facts.db")
        await store.init()
        tracker = InteractionTracker(Path(tmp_path) / "t")
        rec = tracker.record_interaction("feishu", "oc_gone")
        rec.last_interaction = datetime.now() - timedelta(days=36)
        rec.consecutive_proactive_count = 62
        rec.unreachable_since = (datetime.now() - timedelta(days=1)
                                 if unreachable else None)

        class _LLM:
            async def complete(self, **kw): ...

        eng = ConversationEngine(
            persona=PersonaConfig(name="A", identity=Identity(full_name="A")),
            llm_provider=_LLM(),
            memory_manager=MemoryManager(data_dir=str(Path(tmp_path) / "m")),
            fact_retriever=FactRetriever(store),
        )
        eng.interaction_tracker = tracker
        system, msgs = await eng._prepare_turn_v2("在吗", None, "feishu", "oc_gone")
        return system + "\n".join(str(m.get("content", "")) for m in msgs)

    @pytest.mark.asyncio
    async def test_she_claims_no_count_the_channel_could_not_vouch_for(
            self, tmp_path, monkeypatch):
        prompt = await self._prompt(tmp_path, monkeypatch, unreachable=True)

        assert "62" not in prompt and "你叫过他几次" not in prompt

    @pytest.mark.asyncio
    async def test_a_count_of_delivered_messages_is_still_said(
            self, tmp_path, monkeypatch):
        prompt = await self._prompt(tmp_path, monkeypatch, unreachable=False)

        assert "你叫过他几次" in prompt and "62" in prompt
