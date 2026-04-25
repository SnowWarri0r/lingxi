"""Runtime-growing biography: events Aria accumulates through lived conversations.

Seeded biography (in persona YAML) is static. Over time reflection_loop
decides whether a recent session generated a "biographical moment" for
Aria herself — if so, a LifeEvent gets appended here. It's then merged
into the retriever at startup and incrementally embedded on append.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from lingxi.persona.models import LifeEvent


class BiographyAddendaEntry(BaseModel):
    """A self-accumulated LifeEvent with origin metadata."""

    event: LifeEvent
    source: str = "reflection"  # reflection | manual | migrated
    created_at: datetime = Field(default_factory=datetime.now)
    # Which session/recipient triggered this
    recipient_key: str | None = None


class BiographyAddendaStore:
    """JSON-backed append-only store for self-added life events."""

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "biography_addenda.json"

    async def load(self) -> list[BiographyAddendaEntry]:
        if not self.path.exists():
            return []

        def _read():
            return json.loads(self.path.read_text(encoding="utf-8"))

        try:
            data = await asyncio.to_thread(_read)
        except Exception as e:
            print(f"[biography] addenda load failed: {e}")
            return []

        entries: list[BiographyAddendaEntry] = []
        for item in data if isinstance(data, list) else []:
            try:
                entries.append(BiographyAddendaEntry.model_validate(item))
            except Exception:
                continue
        return entries

    async def append(self, entry: BiographyAddendaEntry) -> None:
        existing = await self.load()
        existing.append(entry)

        def _write():
            serialized = [e.model_dump(mode="json") for e in existing]
            self.path.write_text(
                json.dumps(serialized, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        await asyncio.to_thread(_write)
