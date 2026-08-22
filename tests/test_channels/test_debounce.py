"""Burst batching for consecutive messages."""

import asyncio

import pytest

from lingxi.channels.debounce import MessageDebouncer

W = 0.05          # quiet window used throughout; keeps the suite fast
SETTLE = W * 4    # comfortably longer than one window


def _recorder():
    calls: list[tuple[str, list]] = []

    async def flush(key, items):
        calls.append((key, list(items)))

    return calls, flush


@pytest.mark.asyncio
async def test_a_burst_becomes_one_flush():
    """Three messages typed in quick succession are answered once, together."""
    calls, flush = _recorder()
    d = MessageDebouncer(flush, window=W, max_wait=10)

    for msg in ("在吗", "有个事想问你", "就是那个"):
        await d.add("chat", msg)
        await asyncio.sleep(W / 4)
    await asyncio.sleep(SETTLE)

    assert calls == [("chat", ["在吗", "有个事想问你", "就是那个"])]


@pytest.mark.asyncio
async def test_a_lone_message_still_flushes():
    calls, flush = _recorder()
    d = MessageDebouncer(flush, window=W, max_wait=10)

    await d.add("chat", "在吗")
    await asyncio.sleep(SETTLE)

    assert calls == [("chat", ["在吗"])]


@pytest.mark.asyncio
async def test_messages_past_the_window_are_separate_turns():
    """A reply-worthy pause means a new thought, not a continuation."""
    calls, flush = _recorder()
    d = MessageDebouncer(flush, window=W, max_wait=10)

    await d.add("chat", "第一句")
    await asyncio.sleep(SETTLE)
    await d.add("chat", "第二句")
    await asyncio.sleep(SETTLE)

    assert calls == [("chat", ["第一句"]), ("chat", ["第二句"])]


@pytest.mark.asyncio
async def test_conversations_batch_independently():
    calls, flush = _recorder()
    d = MessageDebouncer(flush, window=W, max_wait=10)

    await d.add("a", "a1")
    await d.add("b", "b1")
    await d.add("a", "a2")
    await asyncio.sleep(SETTLE)

    assert sorted(calls) == [("a", ["a1", "a2"]), ("b", ["b1"])]


@pytest.mark.asyncio
async def test_steady_typing_still_gets_answered():
    """max_wait exists so the person typing hardest isn't starved.

    Without it, a message arriving every window forever resets the timer and
    the batch never flushes — the most engaged user gets total silence.
    """
    calls, flush = _recorder()
    d = MessageDebouncer(flush, window=W, max_wait=W * 3)

    async def keep_typing():
        for i in range(12):
            await d.add("chat", f"m{i}")
            await asyncio.sleep(W / 2)

    await keep_typing()
    await asyncio.sleep(SETTLE)

    assert calls, "max_wait never fired — a steady typist would get no reply"
    assert calls[0][0] == "chat"
    # It fired mid-burst rather than waiting for the typing to stop.
    assert len(calls[0][1]) < 12


@pytest.mark.asyncio
async def test_window_zero_disables_batching():
    """Keeps the old one-message-one-turn behaviour reachable by config."""
    calls, flush = _recorder()
    d = MessageDebouncer(flush, window=0, max_wait=10)

    await d.add("chat", "一")
    await d.add("chat", "二")

    assert calls == [("chat", ["一"]), ("chat", ["二"])]


@pytest.mark.asyncio
async def test_flush_now_bypasses_the_window():
    calls, flush = _recorder()
    d = MessageDebouncer(flush, window=10, max_wait=60)

    await d.add("chat", "/status")
    await d.flush_now("chat")

    assert calls == [("chat", ["/status"])]


@pytest.mark.asyncio
async def test_flush_now_on_an_empty_key_does_nothing():
    calls, flush = _recorder()
    d = MessageDebouncer(flush, window=W, max_wait=10)

    await d.flush_now("never-used")

    assert calls == []


@pytest.mark.asyncio
async def test_a_slow_flush_does_not_drop_the_next_burst():
    """The next batch must survive a flush that is still running.

    A real flush is a full LLM turn lasting seconds, far longer than the
    window, so this is the normal case rather than an edge case.
    """
    calls: list[tuple[str, list]] = []

    async def slow_flush(key, items):
        await asyncio.sleep(W * 3)
        calls.append((key, list(items)))

    d = MessageDebouncer(slow_flush, window=W, max_wait=10)

    await d.add("chat", "第一批")
    await asyncio.sleep(SETTLE)          # first flush is now in flight
    await d.add("chat", "第二批")
    await asyncio.sleep(W * 8)

    assert calls == [("chat", ["第一批"]), ("chat", ["第二批"])]


@pytest.mark.asyncio
async def test_cancel_all_stops_pending_timers():
    calls, flush = _recorder()
    d = MessageDebouncer(flush, window=W, max_wait=10)

    await d.add("chat", "在吗")
    d.cancel_all()
    await asyncio.sleep(SETTLE)

    assert calls == []
