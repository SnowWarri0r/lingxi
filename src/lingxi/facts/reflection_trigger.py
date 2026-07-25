"""Triggers reflection when importance accumulates past threshold OR
time-since-last-reflection exceeds max_interval. Hybrid policy from spec.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol


class _ReflectorLike(Protocol):
    async def reflect(self) -> None: ...


class ReflectionTrigger:
    def __init__(
        self,
        reflector: _ReflectorLike,
        threshold: int = 150,
        # 12h floor ≈ 1-2 reflections/day. At 2h the life-sim's 30-min event
        # ticks tripped the elapsed branch every cycle — 5 patterns every 2h
        # flooded the store and the recurring theme snowballed into the
        # planner's inputs.
        max_interval_seconds: float = 43200.0,
    ):
        self._reflector = reflector
        self._threshold = threshold
        self._max_interval = max_interval_seconds
        self._accum = 0
        self._last_fire = time.monotonic()
        self._lock = asyncio.Lock()

    async def observe(self, importance: int) -> None:
        async with self._lock:
            self._accum += importance
            elapsed = time.monotonic() - self._last_fire
            if self._accum >= self._threshold or elapsed >= self._max_interval:
                self._accum = 0
                self._last_fire = time.monotonic()
                asyncio.create_task(self._safe_reflect())

    async def _safe_reflect(self) -> None:
        try:
            await self._reflector.reflect()
        except Exception as e:
            print(f"[reflection_trigger] reflect failed: {e}", flush=True)
