"""Her context with someone is reset once her voice has drifted for days.

A line in the prompt barely moves a voice held down by her own flat history
('!' 1 → 2 of 33 on the September turns); setting that history aside is what
worked on 2026-10-10. The user chose for it to happen automatically.
"""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from lingxi.memory.short_term import ShortTermMemory
from lingxi.persona.voice_monitor import DRIFT_DAYS, note_day
from lingxi.temporal.proactive import ProactiveConfig, ProactiveScheduler
from lingxi.temporal.tracker import InteractionTracker

KEY = "feishu:oc_x"
BAND = {"exclaim_min": 0.15, "shouting_max": 0.25, "self_name": "可可"}
DAY0 = datetime(2026, 11, 2, 12, 0)


def test_days_must_be_consecutive():
    days = note_day([], "2026-11-01", True)
    days = note_day(days, "2026-11-02", True)
    assert days == ["2026-11-01", "2026-11-02"]
    assert note_day(days, "2026-11-04", True) == ["2026-11-04"]


def test_a_day_back_in_band_ends_the_run():
    assert note_day(["2026-11-01", "2026-11-02"], "2026-11-03", False) == []


def test_the_same_day_counts_once():
    assert note_day(["2026-11-01"], "2026-11-01", True) == ["2026-11-01"]


async def _setup(tmp_path, flat=True):
    st = ShortTermMemory(data_dir=tmp_path)
    await st.switch_recipient(KEY)
    for i in range(10):
        st.add_turn("user", "嗯")
        st.add_turn("assistant", "今天练了一下午 腿酸" if flat else "可可今天练完啦！")
    await st.persist_current()
    tracker = InteractionTracker(tmp_path)
    rec = tracker.record_interaction("feishu", "oc_x")
    sched = ProactiveScheduler(
        config=ProactiveConfig(), tracker=tracker, channel_registry=None,
        engine=SimpleNamespace(persona=SimpleNamespace(voice_baseline=BAND),
                               memory=SimpleNamespace(short_term=st)),
        data_dir=str(tmp_path))
    sched._remember_proactive(KEY, "今天下雨了")
    return st, rec, sched


@pytest.mark.asyncio
async def test_three_flat_days_reset_the_context(tmp_path):
    st, rec, sched = await _setup(tmp_path)
    for d in range(DRIFT_DAYS - 1):
        assert not await sched._check_voice(rec, DAY0 + timedelta(days=d))
    assert await sched._check_voice(rec, DAY0 + timedelta(days=DRIFT_DAYS - 1))

    assert st.get_history() == []
    assert list(tmp_path.glob("short_term/*.bak-*-voicereset"))
    assert sched._recent_proactive.get(KEY) is None
    side = json.loads(next(tmp_path.glob("proactive_history.*voicereset.json")).read_text())
    assert side[KEY][0]["text"] == "今天下雨了"
    assert rec.last_voice_reset is not None and rec.voice_drift_days == []


@pytest.mark.asyncio
async def test_one_flat_day_does_nothing(tmp_path):
    st, rec, sched = await _setup(tmp_path)
    assert not await sched._check_voice(rec, DAY0)
    assert len(st.get_history()) == 20


@pytest.mark.asyncio
async def test_in_band_never_resets(tmp_path):
    st, rec, sched = await _setup(tmp_path, flat=False)
    for d in range(DRIFT_DAYS + 2):
        assert not await sched._check_voice(rec, DAY0 + timedelta(days=d))


@pytest.mark.asyncio
async def test_not_again_within_the_cooldown(tmp_path):
    st, rec, sched = await _setup(tmp_path)
    rec.last_voice_reset = DAY0 - timedelta(days=1)
    for d in range(DRIFT_DAYS + 1):
        assert not await sched._check_voice(rec, DAY0 + timedelta(days=d))
