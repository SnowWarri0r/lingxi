"""Cron daemon for NPC event generation.

Wakes every check_interval, decides whether the current wall-clock falls
on a tick hour (default 8/10/12/14/16/18/20/22). On a tick:
- For each NPC: compute_tick_probability → roll dice → maybe generate
- Persist any new events to the store
- (P3 will add promoter call to push significance≥0.6 events)
- (P4 will add arc_advancer call after events written)

Idempotency: last_tick.json records the last tick datetime. The same
clock-tick hour won't fire twice in one day even if the daemon
restarts mid-tick.
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timedelta

from lingxi.providers.base import LLMProvider
from lingxi.social.arc_advancer import advance_npc_arcs
from lingxi.social.event_generator import (
    compute_tick_probability,
    generate_events,
    should_tick,
)
from lingxi.social.models import SocialGraph
from lingxi.social.store import SocialStore


# Daytime tick hours (every 2h from 8 to 22 inclusive)
DEFAULT_TICK_HOURS = (8, 10, 12, 14, 16, 18, 20, 22)


class SocialScheduler:
    def __init__(
        self,
        llm: LLMProvider,
        graph: SocialGraph,
        store: SocialStore,
        *,
        tick_hours: tuple[int, ...] = DEFAULT_TICK_HOURS,
        check_interval_minutes: int = 10,
        model: str | None = None,
        rng: random.Random | None = None,
        on_event_written=None,
        npc_writer=None,
    ):
        self._llm = llm
        self._graph = graph
        self._store = store
        self._tick_hours = tick_hours
        self._check_interval = check_interval_minutes * 60
        self._model = model
        self._rng = rng or random.Random()
        # P3 hook — promoter passes a callable that takes (npc, event)
        # and pushes significance≥threshold to Aria.recent_events
        self._on_event_written = on_event_written
        self._npc_writer = npc_writer
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        print(
            f"[social] scheduler started "
            f"(tick hours {self._tick_hours}, check every {self._check_interval // 60}min)",
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
        await asyncio.sleep(30)
        while self._running:
            try:
                await self._maybe_tick()
            except Exception as e:
                print(f"[social] tick error: {e}", flush=True)
            await asyncio.sleep(self._check_interval)

    async def _maybe_tick(self) -> None:
        now = datetime.now()
        if now.hour not in self._tick_hours:
            return

        last = await self._store.load_last_tick()
        if last is not None:
            # Don't fire if we've already ticked within this hour today
            same_hour_today = (
                last.date() == now.date() and last.hour == now.hour
            )
            if same_hour_today:
                return
            # Also dedup if we just ticked very recently (within 30 min)
            if (now - last) < timedelta(minutes=30):
                return

        print(f"[social] tick at {now.strftime('%H:%M')}", flush=True)
        await self._run_tick(now)
        await self._store.save_last_tick(now)

    async def _run_tick(self, now: datetime) -> None:
        for npc in self._graph.npcs:
            try:
                await self._tick_one_npc(npc, now)
            except Exception as e:
                print(f"[social] tick {npc.id} failed: {e}", flush=True)

    async def _tick_one_npc(self, npc, now: datetime) -> None:
        state = await self._store.load_state(npc.id)
        prob = compute_tick_probability(npc, state, now=now)
        if not should_tick(prob, rng=self._rng):
            return

        events = await generate_events(
            self._llm, npc, state, now=now, model=self._model,
        )
        bumped_arcs: set[str] = set()
        for ev in events:
            await self._store.append_event(ev)
            if self._npc_writer is not None:
                try:
                    from lingxi.facts.models import FactType as _FactType
                    _tags = [ev.type]
                    if ev.arc_id:
                        _tags.append(ev.arc_id)
                    await self._npc_writer.write(
                        subject=f"npc:{npc.id}",
                        content=ev.content,
                        type=_FactType.EVENT,
                        ts=ev.ts,
                        confidence=ev.significance,
                        tags=_tags,
                    )
                except Exception as e:
                    print(f"[social] facts write failed: {e}", flush=True)
            print(
                f"[social] {npc.id} +event sig={ev.significance:.2f} "
                f"type={ev.type} arc={ev.arc_id or '-'} content={ev.content[:30]}...",
                flush=True,
            )
            if ev.arc_id:
                await self._bump_arc_count(npc.id, ev.arc_id)
                bumped_arcs.add(ev.arc_id)
            if self._on_event_written is not None:
                try:
                    await self._on_event_written(npc, ev)
                except Exception as e:
                    print(f"[social] event hook failed: {e}", flush=True)

        # Arc advancement: only check arcs that just got a new event (or
        # arcs at force-resolve threshold — caught inside maybe_advance).
        # Keeps LLM calls bounded: at most 1 advancement check per event.
        if bumped_arcs:
            try:
                await advance_npc_arcs(
                    self._llm, npc, self._store,
                    now=now, model=self._model,
                )
            except Exception as e:
                print(f"[social] arc advance failed for {npc.id}: {e}", flush=True)

    async def _bump_arc_count(self, npc_id: str, arc_id: str) -> None:
        state = await self._store.load_state(npc_id)
        changed = False
        for arc in state.arcs:
            if arc.id == arc_id:
                arc.event_count += 1
                changed = True
                break
        if changed:
            await self._store.save_arcs(npc_id, state.arcs)

    async def trigger_now(self) -> None:
        """Manual trigger for tests / debug — bypasses tick-hour gate."""
        now = datetime.now()
        await self._run_tick(now)
        await self._store.save_last_tick(now)
