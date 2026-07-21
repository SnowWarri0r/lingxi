"""Background weather refresher — keeps the in-process cache warm so the
(synchronous) prompt path can read current weather without blocking.

Fetches once on start, then every `interval_minutes`. Fully fail-safe: a
failed fetch is logged and skipped, leaving the last good reading in place.
"""

from __future__ import annotations

import asyncio

from lingxi.temporal.sun import Location
from lingxi.temporal import weather


class WeatherScheduler:
    def __init__(self, location: Location, interval_minutes: float = 20.0):
        self._loc = location
        self._interval = interval_minutes * 60.0

    async def start(self) -> None:
        w = await weather.refresh(self._loc)
        if w is not None:
            print(f"[weather] {self._loc.name or ''} {w.phrase()}", flush=True)
        print(f"[weather] scheduler started (refresh every "
              f"{int(self._interval / 60)}min)", flush=True)
        while True:
            await asyncio.sleep(self._interval)
            try:
                await weather.refresh(self._loc)
            except Exception as e:
                print(f"[weather] refresh loop error: {e}", flush=True)
