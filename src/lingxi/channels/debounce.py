"""Batch messages that arrive in a burst into one turn.

People type the way they think: 「在吗」「有个事想问你」「就是那个…」 is one
thought sent as three messages. Answering each one separately is the tell that
something mechanical is on the other end — and it costs three full turns.

This batches per conversation: each new message restarts a short quiet-window
timer, and the batch is flushed once the window passes without a new message.

`max_wait` bounds the other direction. Without it, someone typing steadily
resets the timer forever and never gets an answer at all — the timer would be
starved by exactly the person trying hardest to talk. Once the oldest message
in a batch reaches `max_wait`, it flushes regardless of what is still arriving.

This is batching, not interruption. A turn already in flight runs to
completion; messages that arrive during it accumulate here and are answered
together afterwards, which is why the caller still needs to serialise turns.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable


class MessageDebouncer:
    """Per-key burst batching with a quiet window and a hard ceiling.

    `flush(key, items)` is awaited with the accumulated items, oldest first.
    Set `window=0` to disable batching entirely (every message flushes on its
    own), which keeps the behaviour available without a separate code path.
    """

    def __init__(
        self,
        flush: Callable[[str, list[Any]], Awaitable[None]],
        *,
        window: float = 1.5,
        max_wait: float = 8.0,
    ):
        self._flush = flush
        self._window = window
        self._max_wait = max_wait
        self._pending: dict[str, list[Any]] = {}
        self._timers: dict[str, asyncio.Task] = {}
        self._first_seen: dict[str, float] = {}

    async def add(self, key: str, item: Any) -> None:
        """Queue `item` under `key` and (re)start its quiet window."""
        if self._window <= 0:
            await self._flush(key, [item])
            return

        loop = asyncio.get_running_loop()
        self._pending.setdefault(key, []).append(item)
        self._first_seen.setdefault(key, loop.time())

        self._cancel_timer(key)
        self._timers[key] = asyncio.create_task(self._wait_then_flush(key))

    async def flush_now(self, key: str) -> None:
        """Flush `key` immediately, if anything is pending."""
        self._cancel_timer(key)
        await self._drain(key)

    def cancel_all(self) -> None:
        """Drop every pending timer. Pending items are left queued."""
        for key in list(self._timers):
            self._cancel_timer(key)

    def _cancel_timer(self, key: str) -> None:
        task = self._timers.pop(key, None)
        if task is not None and not task.done():
            task.cancel()

    async def _wait_then_flush(self, key: str) -> None:
        try:
            loop = asyncio.get_running_loop()
            first = self._first_seen.get(key, loop.time())
            # Never wait past max_wait measured from the batch's first message,
            # however many resets happen in between.
            remaining = (first + self._max_wait) - loop.time()
            await asyncio.sleep(min(self._window, max(0.0, remaining)))
        except asyncio.CancelledError:
            return
        self._timers.pop(key, None)
        await self._drain(key)

    async def _drain(self, key: str) -> None:
        items = self._pending.pop(key, [])
        self._first_seen.pop(key, None)
        if items:
            await self._flush(key, items)
