"""File-based store for NPC runtime state.

Layout:
    data/social/npcs/{npc_id}/arcs.json     # list of NPCArc as JSON
    data/social/npcs/{npc_id}/events.jsonl  # one NPCEvent per line, append-only
    data/social/last_tick.json              # cron coordination

Two write patterns:
- events: append-only jsonl, cheap atomic-append (no rewrite)
- arcs: small JSON, atomic temp+rename rewrite

Reads cache nothing — file size stays small (events trimmed to last 30 days
on read, arcs typically <10 per NPC).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

from lingxi.social.models import NPCArc, NPCEvent, NPCState


class SocialStore:
    """Per-persona NPC state store. All NPCs share one root directory."""

    def __init__(self, data_dir: Path | str):
        self._root = Path(data_dir) / "social" / "npcs"
        self._tick_path = Path(data_dir) / "social" / "last_tick.json"
        self._lock = asyncio.Lock()        # serializes arc writes per process

    def _npc_dir(self, npc_id: str) -> Path:
        return self._root / npc_id

    def _arcs_path(self, npc_id: str) -> Path:
        return self._npc_dir(npc_id) / "arcs.json"

    def _events_path(self, npc_id: str) -> Path:
        return self._npc_dir(npc_id) / "events.jsonl"

    async def load_state(
        self, npc_id: str, *, events_since: datetime | None = None
    ) -> NPCState:
        """Load arcs + recent events for one NPC.

        events_since defaults to 30 days ago — trims event_log read to keep
        memory bounded. Older events stay on disk; they just don't enter
        the in-memory state.
        """
        cutoff = events_since or (datetime.now() - timedelta(days=30))

        arcs = await self._load_arcs(npc_id)
        events = await self._load_events(npc_id, cutoff)
        last_at = max((e.ts for e in events), default=None)

        return NPCState(
            npc_id=npc_id, arcs=arcs, recent_events=events, last_event_at=last_at
        )

    async def _load_arcs(self, npc_id: str) -> list[NPCArc]:
        path = self._arcs_path(npc_id)
        if not path.exists():
            return []
        try:
            data = await asyncio.to_thread(
                lambda: json.loads(path.read_text(encoding="utf-8"))
            )
            return [NPCArc.model_validate(a) for a in data]
        except Exception:
            return []

    async def _load_events(
        self, npc_id: str, cutoff: datetime
    ) -> list[NPCEvent]:
        path = self._events_path(npc_id)
        if not path.exists():
            return []

        def _read() -> list[NPCEvent]:
            out: list[NPCEvent] = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        ev = NPCEvent.model_validate(data)
                        if ev.ts >= cutoff:
                            out.append(ev)
                    except Exception:
                        continue
            return out

        events = await asyncio.to_thread(_read)
        events.sort(key=lambda e: e.ts)
        return events

    async def append_event(self, event: NPCEvent) -> None:
        """Append one event to jsonl. Atomic on POSIX (single-line writes)."""
        path = self._events_path(event.npc_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _append():
            line = event.model_dump_json() + "\n"
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)

        await asyncio.to_thread(_append)

    async def save_arcs(self, npc_id: str, arcs: list[NPCArc]) -> None:
        """Replace arcs.json wholesale (atomic via temp+rename)."""
        path = self._arcs_path(npc_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            tmp = path.with_suffix(".tmp")

            def _write():
                payload = [a.model_dump(mode="json") for a in arcs]
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
                tmp.rename(path)

            await asyncio.to_thread(_write)

    async def mark_event_promoted(
        self, npc_id: str, event_ts: datetime
    ) -> None:
        """Flip promoted_to_aria=True on a stored event.

        events.jsonl is append-only, so we rewrite the whole file under
        the lock. Cheap because read trims to 30 days. Matched by
        (npc_id, ts) — ts has microsecond precision so collisions are
        not a concern in practice.
        """
        path = self._events_path(npc_id)
        if not path.exists():
            return

        async with self._lock:

            def _rewrite():
                lines_out: list[str] = []
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if data.get("ts") == event_ts.isoformat() or (
                                data.get("npc_id") == npc_id
                                and data.get("ts", "").startswith(
                                    event_ts.isoformat()[:19]
                                )
                            ):
                                data["promoted_to_aria"] = True
                            lines_out.append(json.dumps(data, ensure_ascii=False))
                        except Exception:
                            lines_out.append(line)
                tmp = path.with_suffix(".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines_out) + "\n")
                tmp.rename(path)

            await asyncio.to_thread(_rewrite)

    async def load_last_tick(self) -> datetime | None:
        if not self._tick_path.exists():
            return None
        try:
            data = await asyncio.to_thread(
                lambda: json.loads(self._tick_path.read_text(encoding="utf-8"))
            )
            return datetime.fromisoformat(data["ts"])
        except Exception:
            return None

    async def save_last_tick(self, ts: datetime) -> None:
        self._tick_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._tick_path.with_suffix(".tmp")

        def _write():
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"ts": ts.isoformat()}, f)
            tmp.rename(self._tick_path)

        await asyncio.to_thread(_write)
