"""Daily briefing scheduler — fetch each morning if not yet present.

world/store.py was deleted in P7. Idempotency is now enforced by
querying the facts table: if any world.EVENT facts exist since
today_start, we already fetched today and skip.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta

from lingxi.world.fetcher import fetch_daily_briefing


class WorldScheduler:
    """Background task that ensures today's briefing exists.

    Runs a check loop every `check_interval_minutes`. If today's
    briefing is missing AND the local time is past the configured
    `morning_after_hour` (default 6 AM), fetch it. After that, sleeps
    until the next check.

    Idempotent: queries the facts store (subject="world") to check
    whether a briefing was already written today before fetching.
    """

    def __init__(
        self,
        llm,
        persona=None,
        *,
        morning_after_hour: int = 6,
        check_interval_minutes: int = 30,
        empty_retry_hours: float = 4.0,
        world_writer=None,
        fact_retriever=None,
    ):
        self._llm = llm
        self._persona = persona
        self._morning_after_hour = morning_after_hour
        self._check_interval = check_interval_minutes * 60
        self._empty_retry = timedelta(hours=empty_retry_hours)
        self._world_writer = world_writer
        self._fact_retriever = fact_retriever
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._task is not None:
            return
        # A persona who follows nothing has no morning to fetch. Checking here
        # rather than inside the loop keeps the whole task from existing: the
        # searches are the single most expensive call in the app (4.33M input
        # tokens over 77 fetches), so a persona that would only discard them
        # should never make one.
        if not (getattr(self._persona, "world_interests", None) or []):
            print("[world] persona follows nothing in particular — "
                  "scheduler not started", flush=True)
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

    async def _already_fetched_today(self) -> bool:
        """Return True if world facts for today already exist in the facts table."""
        if self._fact_retriever is None:
            return False
        try:
            from lingxi.facts.models import FactType as _FactType
            from lingxi.facts.retriever import FactQuery
            today_start = datetime.combine(date.today(), datetime.min.time())
            hits = await self._fact_retriever.fetch(
                FactQuery(subject="world", type=_FactType.EVENT, since=today_start, limit=1)
            )
            return len(hits) > 0
        except Exception as e:
            print(f"[world] facts check failed: {e}", flush=True)
            return False

    async def _maybe_fetch_today(self) -> None:
        today = date.today()
        now = datetime.now()
        if now.hour < self._morning_after_hour:
            return

        if await self._already_fetched_today():
            return

        print(f"[world] no briefing for {today}, fetching...", flush=True)

        briefing = await fetch_daily_briefing(
            self._llm, self._persona, target_date=today,
        )
        await self._write(briefing)

    async def _write(self, briefing) -> None:
        """Store each item as a world EVENT fact."""
        if self._world_writer is None or not briefing.items:
            return
        try:
            from lingxi.facts.models import FactType as _FactType
            from datetime import datetime as _dt, timedelta as _td
            for item in briefing.items:
                content = (item.voice or item.headline or "").strip()
                if not content:
                    continue
                await self._world_writer.write(
                    subject="world",
                    content=content,
                    type=_FactType.EVENT,
                    ts=_dt.combine(briefing.date, _dt.min.time()),
                    tags=[item.category],
                    expires_at=_dt.now() + _td(days=2),
                )
        except Exception as e:
            print(f"[world] facts write failed: {e}", flush=True)

    async def trigger_now(self) -> None:
        """Manual trigger (for /world refresh-style commands or tests)."""
        briefing = await fetch_daily_briefing(
            self._llm, self._persona, target_date=date.today(),
        )
        await self._write(briefing)
