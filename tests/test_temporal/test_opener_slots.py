"""Openers go out when he reads them, and silence slows her down.

Two weeks of production openers on a fixed clock: 0 of 14 answered at 08:00,
3 of 14 around 14:00, 2 of 12 in the evening. And past the cap a flat 6h
re-engage wait kept her at three a day through six days of silence, 18
unanswered in a row. The record now keeps whether each opener was answered;
slots he never answers in are skipped, and past the cap she sends at most one
a day.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from lingxi.channels.outbound import ChannelRegistry, OutboundChannel
from lingxi.temporal.proactive import ProactiveConfig, ProactiveScheduler, dead_slot
from lingxi.temporal.tracker import (
    ANSWER_WINDOW,
    InteractionTracker,
    ProactiveOutcome,
)

# As config/default.yaml runs it in production.
CFG = ProactiveConfig(
    silence_thresholds={1: 5, 2: 4, 3: 2, 4: 1}, cooldown_hours=5,
    reengage_after_hours=6, reengage_backoff=1.0)
DAY0 = datetime(2026, 9, 1)


def _history(morning=(0, 14), midday=(3, 14), evening=(2, 12)):
    """(answered, sent) per slot, one opener a day per slot."""
    out = []
    for hour, (answered, sent) in ((8, morning), (14, midday), (20, evening)):
        for i in range(sent):
            out.append(ProactiveOutcome(
                sent_at=DAY0 + timedelta(days=i, hours=hour),
                answered=i < answered))
    return out


NOW = DAY0 + timedelta(days=20)


def _at(hour):
    return NOW.replace(hour=hour)


def test_the_slot_he_never_answers_is_skipped():
    assert dead_slot(_history(), _at(8), CFG)


def test_the_slots_he_does_answer_stay_open():
    assert not dead_slot(_history(), _at(14), CFG)
    assert not dead_slot(_history(), _at(20), CFG)


def test_a_slot_is_not_judged_on_a_handful_of_openers():
    assert not dead_slot(_history(morning=(0, 5)), _at(8), CFG)


def test_a_fresh_record_skips_nothing():
    assert not dead_slot([], _at(8), CFG)


def test_an_opener_still_inside_the_answer_window_is_not_a_miss():
    """Eight morning openers, the last only an hour old: seven are resolved."""
    hist = _history(morning=(0, 7))
    hist.append(ProactiveOutcome(sent_at=_at(8) - timedelta(hours=1)))
    assert not dead_slot(hist, _at(8), CFG)


def test_when_he_answers_nothing_no_slot_is_singled_out():
    hist = _history(morning=(0, 14), midday=(0, 14), evening=(0, 12))
    assert not any(dead_slot(hist, _at(h), CFG) for h in (8, 14, 20))


# --- the tracker records the outcome ------------------------------------

def test_a_reply_inside_the_window_marks_the_opener_answered(tmp_path):
    t = InteractionTracker(tmp_path)
    t.record_interaction("feishu", "oc_x")
    t.record_proactive_sent("feishu", "oc_x")
    t.record_interaction("feishu", "oc_x")
    assert t._records["feishu:oc_x"].proactive_outcomes[-1].answered


def test_a_reply_after_the_window_does_not(tmp_path):
    t = InteractionTracker(tmp_path)
    t.record_interaction("feishu", "oc_x")
    t.record_proactive_sent("feishu", "oc_x")
    rec = t._records["feishu:oc_x"]
    rec.proactive_outcomes[-1].sent_at -= ANSWER_WINDOW + timedelta(minutes=1)
    t.record_interaction("feishu", "oc_x")
    assert not rec.proactive_outcomes[-1].answered


def test_old_outcomes_age_out(tmp_path):
    """Which is what lets a dropped slot be tried again."""
    t = InteractionTracker(tmp_path)
    t.record_interaction("feishu", "oc_x")
    for _ in range(80):
        t.record_proactive_sent("feishu", "oc_x")
    assert len(t._records["feishu:oc_x"].proactive_outcomes) == 60


@pytest.mark.asyncio
async def test_outcomes_survive_a_restart(tmp_path):
    t = InteractionTracker(tmp_path)
    t.record_interaction("feishu", "oc_x")
    t.record_proactive_sent("feishu", "oc_x")
    t.record_interaction("feishu", "oc_x")
    await t.save()
    t2 = InteractionTracker(tmp_path)
    await t2.load()
    assert t2._records["feishu:oc_x"].proactive_outcomes[-1].answered


# --- the scheduler applies both ------------------------------------------

class _Working(OutboundChannel):
    @property
    def channel_name(self):
        return "feishu"

    async def send_message(self, recipient_id, text, turn_id=None):
        pass

    async def send_sticker(self, recipient_id, file_path):
        pass


def _scheduler(tmp_path):
    tracker = InteractionTracker(tmp_path)
    rec = tracker.record_interaction("feishu", "oc_x")
    rec.relationship_level = 2
    reg = ChannelRegistry()
    reg.register(_Working())
    sched = ProactiveScheduler(
        config=CFG, tracker=tracker, channel_registry=reg,
        engine=SimpleNamespace())
    composed = []

    async def _ask_llm(record, silence, now, force=False):
        composed.append(now)
        return None

    sched._ask_llm = _ask_llm
    return sched, rec, composed


@pytest.mark.asyncio
async def test_she_composes_nothing_in_a_dead_slot(tmp_path):
    sched, rec, composed = _scheduler(tmp_path)
    rec.last_interaction = _at(8) - timedelta(days=2)
    rec.proactive_outcomes = _history()

    result = await sched._maybe_reach_out(rec, _at(8))

    assert result["status"] == "skipped_dead_slot"
    assert composed == []


@pytest.mark.asyncio
async def test_past_the_cap_one_a_day(tmp_path):
    sched, rec, composed = _scheduler(tmp_path)
    rec.last_interaction = _at(9) - timedelta(days=3)
    rec.consecutive_proactive_count = CFG.max_consecutive_proactive + 1
    rec.last_proactive_sent = _at(9)

    later_today = await sched._maybe_reach_out(rec, _at(21))
    assert later_today["status"] == "skipped_consecutive_cap"

    tomorrow = await sched._maybe_reach_out(rec, _at(21) + timedelta(hours=12))
    assert tomorrow["status"] != "skipped_consecutive_cap"
    assert len(composed) == 1


@pytest.mark.asyncio
async def test_below_the_cap_a_second_opener_the_same_day_still_goes(tmp_path):
    """One a day is the cost of being ignored, not the ordinary pace."""
    sched, rec, composed = _scheduler(tmp_path)
    rec.last_interaction = _at(9) - timedelta(days=1)
    rec.consecutive_proactive_count = 1
    rec.last_proactive_sent = _at(9)

    await sched._maybe_reach_out(rec, _at(21))

    assert len(composed) == 1
