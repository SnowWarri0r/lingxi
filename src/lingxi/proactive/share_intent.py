"""Transient share-intent queue for proactive opener.

Separated from Fact storage because Facts are immutable observations
while share intent is a mutable decision-with-lifecycle (queued →
voiced → consumed). Cooldown lives here too — natural pairing.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class ShareIntent:
    fact_id: str
    source_npc: str
    significance: float
    queued_at: datetime


class ShareIntentStore:
    def __init__(self, data_dir: Path | str, cooldown_hours: int = 24):
        self._path = Path(data_dir) / "proactive" / "share_intents.json"
        self._cooldown = timedelta(hours=cooldown_hours)
        self._lock = asyncio.Lock()

    async def queue(self, fact_id: str, source_npc: str, significance: float) -> bool:
        """Add intent. Returns False if source_npc is in cooldown."""
        async with self._lock:
            data = await self._load()
            cooldown_at_iso = data.get("cooldown_at", {}).get(source_npc)
            if cooldown_at_iso:
                try:
                    last = datetime.fromisoformat(cooldown_at_iso)
                    if datetime.now() - last < self._cooldown:
                        return False
                except ValueError:
                    pass
            now = datetime.now()
            data.setdefault("pending", []).append({
                "fact_id": fact_id,
                "source_npc": source_npc,
                "significance": significance,
                "queued_at": now.isoformat(),
            })
            data.setdefault("cooldown_at", {})[source_npc] = now.isoformat()
            await self._save(data)
            return True

    async def pending(self) -> list[ShareIntent]:
        data = await self._load()
        items = data.get("pending", [])
        result = []
        for item in items:
            try:
                result.append(ShareIntent(
                    fact_id=item["fact_id"],
                    source_npc=item["source_npc"],
                    significance=float(item["significance"]),
                    queued_at=datetime.fromisoformat(item["queued_at"]),
                ))
            except (KeyError, ValueError):
                continue
        return result

    async def consume(self, fact_id: str) -> None:
        async with self._lock:
            data = await self._load()
            data["pending"] = [
                p for p in data.get("pending", [])
                if p.get("fact_id") != fact_id
            ]
            await self._save(data)

    async def is_in_cooldown(self, source_npc: str) -> bool:
        data = await self._load()
        cooldown_at_iso = data.get("cooldown_at", {}).get(source_npc)
        if not cooldown_at_iso:
            return False
        try:
            last = datetime.fromisoformat(cooldown_at_iso)
        except ValueError:
            return False
        return datetime.now() - last < self._cooldown

    async def _load(self) -> dict:
        if not self._path.exists():
            return {"pending": [], "cooldown_at": {}}
        try:
            text = await asyncio.to_thread(self._path.read_text, encoding="utf-8")
            data = json.loads(text)
            return data if isinstance(data, dict) else {"pending": [], "cooldown_at": {}}
        except Exception:
            return {"pending": [], "cooldown_at": {}}

    async def _save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")

        def _write():
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(self._path)

        await asyncio.to_thread(_write)
