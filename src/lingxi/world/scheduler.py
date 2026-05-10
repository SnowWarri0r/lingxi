"""Daily briefing scheduler — fetch each morning if not yet present."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta

from lingxi.world.fetcher import fetch_daily_briefing
from lingxi.world.store import WorldStore


class WorldScheduler:
    """Background task that ensures today's briefing exists.

    Runs a check loop every `check_interval_minutes`. If today's
    briefing is missing AND the local time is past the configured
    `morning_after_hour` (default 6 AM), fetch it. After that, sleeps
    until the next check.

    Idempotent: if the file already exists for today, no fetch happens.
    """

    def __init__(
        self,
        api_key: str,
        store: WorldStore,
        *,
        morning_after_hour: int = 6,
        check_interval_minutes: int = 30,
        model: str = "claude-sonnet-4-5",
        empty_retry_hours: float = 4.0,
    ):
        self._api_key = api_key
        self._store = store
        self._morning_after_hour = morning_after_hour
        self._check_interval = check_interval_minutes * 60
        self._model = model
        # If today's briefing exists but is empty (API error, parse fail),
        # we'd block retry permanently. Allow retry when the empty briefing
        # is older than this. Genuinely thin news days get retried a few
        # times then sit until tomorrow's date rollover.
        self._empty_retry = timedelta(hours=empty_retry_hours)
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        print(
            f"[world] scheduler started "
            f"(check every {self._check_interval // 60}min, "
            f"fetch after {self._morning_after_hour}:00)",
            flush=True,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        # Initial delay so the rest of bootstrap settles
        await asyncio.sleep(20)
        while self._running:
            try:
                await self._maybe_fetch_today()
            except Exception as e:
                print(f"[world] scheduler tick error: {e}", flush=True)
            await asyncio.sleep(self._check_interval)

    async def _maybe_fetch_today(self) -> None:
        today = date.today()
        now = datetime.now()
        if now.hour < self._morning_after_hour:
            return

        existing = await self._store.load_today()
        if existing is not None:
            # Already have a briefing for today.
            if existing.items:
                # Got real items — done.
                return
            if (now - existing.generated_at) < self._empty_retry:
                # Empty + recent: probably transient error, give it more
                # time before retrying (not retried in <4h).
                return
            # Empty + stale: try again. Common case: 429 rate limit
            # hit at 8am, retried successfully at noon.
            print(
                f"[world] empty briefing for {today} stale "
                f"({(now - existing.generated_at).total_seconds() / 3600:.1f}h), retrying...",
                flush=True,
            )
        else:
            print(f"[world] no briefing for {today}, fetching...", flush=True)

        briefing = await fetch_daily_briefing(
            self._api_key, target_date=today, model=self._model,
        )
        await self._store.save(briefing)

    async def trigger_now(self) -> None:
        """Manual trigger (for /world refresh-style commands or tests)."""
        today = date.today()
        briefing = await fetch_daily_briefing(
            self._api_key, target_date=today, model=self._model,
        )
        await self._store.save(briefing)
