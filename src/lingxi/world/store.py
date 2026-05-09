"""File-based per-day briefing store.

Layout: data/world/news/YYYY-MM-DD.json — one file per day. The
scheduler writes once per day; chat path reads on every turn (cheap,
small file).
"""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path

from lingxi.world.models import DailyBriefing


class WorldStore:
    def __init__(self, data_dir: Path | str):
        self._root = Path(data_dir) / "world" / "news"
        self._lock = asyncio.Lock()

    def _path_for(self, d: date) -> Path:
        return self._root / f"{d.isoformat()}.json"

    async def _atomic_write(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")

        def _write():
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            tmp.rename(path)

        await asyncio.to_thread(_write)

    async def load_for(self, d: date) -> DailyBriefing | None:
        path = self._path_for(d)
        if not path.exists():
            return None
        try:
            data = await asyncio.to_thread(
                lambda: json.loads(path.read_text(encoding="utf-8"))
            )
            return DailyBriefing.model_validate(data)
        except Exception:
            return None

    async def load_today(self) -> DailyBriefing | None:
        return await self.load_for(date.today())

    async def save(self, briefing: DailyBriefing) -> None:
        async with self._lock:
            await self._atomic_write(
                self._path_for(briefing.date),
                briefing.model_dump(mode="json"),
            )

    async def has_briefing_for(self, d: date) -> bool:
        return self._path_for(d).exists()
