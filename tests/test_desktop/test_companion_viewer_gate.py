"""The desktop pet speaks only while the window is open.

Measured over eight days with no window running: 187 of 217 generated lines —
86% of everything she said — were written for a closed window. The window
polls /pet/state every ~3s, so that request is the presence signal.
"""

import pytest

from lingxi.desktop.activity_sensor import ActivitySignal
from lingxi.desktop.companion import PetCompanion


class _Engine:
    persona = None


def _companion(monkeypatch, sig_kind="awaiting_user"):
    """A companion whose sensor and generator are both under our control."""
    comp = PetCompanion(_Engine(), min_gap_secs=0.0)
    comp.generated = []

    async def _generate(situation):
        comp.generated.append(situation)
        return f"line {len(comp.generated)}"

    comp._generate = _generate
    return comp


def _signals(monkeypatch, kinds):
    """Feed detect_activity a fixed sequence, one per tick."""
    seq = iter(kinds)

    def _detect(now=None):
        kind = next(seq)
        return ActivitySignal(kind, "写代码", 0.0, "")

    monkeypatch.setattr("lingxi.desktop.companion.detect_activity", _detect)


# A stretch of work followed by a pause — the transition that makes her speak.
BUSY_THEN_PAUSE = ["tool_running", "awaiting_user"]


async def _run_ticks(comp, n, *, start=1000.0, step=100.0, monkeypatch=None):
    times = [start + i * step for i in range(n)]
    it = iter(times)
    monkeypatch.setattr("lingxi.desktop.companion.time.time", lambda: next(it))
    for _ in range(n):
        await comp._tick()
    return times


@pytest.mark.asyncio
async def test_a_window_that_never_opened_costs_nothing(monkeypatch):
    comp = _companion(monkeypatch)
    _signals(monkeypatch, BUSY_THEN_PAUSE)
    await _run_ticks(comp, 2, monkeypatch=monkeypatch)

    assert comp.generated == [], "no viewer has ever polled"


@pytest.mark.asyncio
async def test_she_speaks_while_the_window_is_polling(monkeypatch):
    comp = _companion(monkeypatch)
    _signals(monkeypatch, BUSY_THEN_PAUSE)
    comp.mark_polled(now=1100.0 - 3.0)  # a poll 3s before the second tick
    await _run_ticks(comp, 2, monkeypatch=monkeypatch)

    assert len(comp.generated) == 1


@pytest.mark.asyncio
async def test_she_goes_quiet_once_the_polls_stop(monkeypatch):
    """The window was open, then closed — the last poll ages out."""
    comp = _companion(monkeypatch)
    _signals(monkeypatch, BUSY_THEN_PAUSE)
    comp.mark_polled(now=1000.0 - 60.0)  # long stale by the second tick
    await _run_ticks(comp, 2, monkeypatch=monkeypatch)

    assert comp.generated == []


def test_the_presence_window_is_generous_about_a_missed_poll():
    comp = PetCompanion(_Engine())
    comp.mark_polled(now=1000.0)

    assert comp.is_watched(now=1000.0 + 10.0) is True, "3 missed polls is a hiccup"
    assert comp.is_watched(now=1000.0 + 29.0) is True
    assert comp.is_watched(now=1000.0 + 31.0) is False


def test_never_polled_is_not_watched():
    assert PetCompanion(_Engine()).is_watched(now=1000.0) is False


@pytest.mark.asyncio
async def test_sensing_continues_while_nobody_watches(monkeypatch):
    """Cheap, local, and it keeps the first line after opening accurate."""
    comp = _companion(monkeypatch)
    _signals(monkeypatch, ["tool_running", "awaiting_user"])
    await _run_ticks(comp, 2, monkeypatch=monkeypatch)

    assert comp._sig.kind == "awaiting_user", "the sensor kept reading"
