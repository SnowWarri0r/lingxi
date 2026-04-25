"""Persistence for inner life state.

Two tables:
- inner_state.json — single document, the current state
- subjective/<recipient_key>.json — per-recipient subjective views
- agenda/<recipient_key>.json — per-recipient agenda items
- diary/<YYYY-MM-DD>.json — daily diary archive
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timedelta
from pathlib import Path

from persona_agent.inner_life.models import (
    AgendaItem,
    DailyPlan,
    DiaryEntry,
    InnerState,
    LifeEvent,
    SubjectiveView,
)


class InnerLifeStore:
    """File-based storage for inner life. Simple, atomic writes."""

    def __init__(self, data_dir: Path | str):
        self._root = Path(data_dir) / "inner_life"
        self._state_path = self._root / "state.json"
        self._subj_dir = self._root / "subjective"
        self._agenda_dir = self._root / "agenda"
        self._diary_dir = self._root / "diary"
        self._lock = asyncio.Lock()

    def _safe_key(self, key: str) -> str:
        return "".join(c if c.isalnum() or c in "-_:" else "_" for c in key)

    async def _atomic_write(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")

        def _write():
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            tmp.rename(path)

        await asyncio.to_thread(_write)

    # --- Inner state ---

    async def load_state(self) -> InnerState:
        if not self._state_path.exists():
            return InnerState()
        try:
            data = await asyncio.to_thread(
                lambda: json.loads(self._state_path.read_text(encoding="utf-8"))
            )
            return InnerState.model_validate(data)
        except (json.JSONDecodeError, OSError, Exception):
            return InnerState()

    async def save_state(self, state: InnerState) -> None:
        async with self._lock:
            await self._atomic_write(self._state_path, state.model_dump(mode="json"))

    # --- Subjective views ---

    async def load_subjective(self, recipient_key: str) -> SubjectiveView:
        path = self._subj_dir / f"{self._safe_key(recipient_key)}.json"
        if not path.exists():
            return SubjectiveView(recipient_key=recipient_key)
        try:
            data = await asyncio.to_thread(
                lambda: json.loads(path.read_text(encoding="utf-8"))
            )
            return SubjectiveView.model_validate(data)
        except Exception:
            return SubjectiveView(recipient_key=recipient_key)

    async def save_subjective(self, view: SubjectiveView) -> None:
        path = self._subj_dir / f"{self._safe_key(view.recipient_key)}.json"
        await self._atomic_write(path, view.model_dump(mode="json"))

    # --- Agenda ---

    async def load_agenda(self, recipient_key: str) -> list[AgendaItem]:
        path = self._agenda_dir / f"{self._safe_key(recipient_key)}.json"
        if not path.exists():
            return []
        try:
            data = await asyncio.to_thread(
                lambda: json.loads(path.read_text(encoding="utf-8"))
            )
            items = data.get("items", [])
            return [AgendaItem.model_validate(i) for i in items]
        except Exception:
            return []

    async def save_agenda(self, recipient_key: str, items: list[AgendaItem]) -> None:
        path = self._agenda_dir / f"{self._safe_key(recipient_key)}.json"
        await self._atomic_write(
            path,
            {"items": [i.model_dump(mode="json") for i in items]},
        )

    async def list_agenda_recipients(self) -> list[str]:
        """Return recipient keys that have agenda items."""
        if not self._agenda_dir.exists():
            return []
        results = []
        for p in self._agenda_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                items = data.get("items", [])
                if items:
                    # Read recipient_key from first item
                    results.append(items[0].get("recipient_key", p.stem))
            except Exception:
                continue
        return results

    # --- Diary archive ---

    async def append_diary(self, entry: DiaryEntry) -> None:
        """Append a diary entry to today's file (one file per day)."""
        day = entry.timestamp.date().isoformat()
        path = self._diary_dir / f"{day}.json"

        existing: list[dict] = []
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                existing = data.get("entries", [])
            except Exception:
                pass

        existing.append(entry.model_dump(mode="json"))
        await self._atomic_write(path, {"date": day, "entries": existing})

    async def load_diary_since(self, since: date) -> list[DiaryEntry]:
        """Load diary entries from dates >= since."""
        if not self._diary_dir.exists():
            return []
        results: list[DiaryEntry] = []
        for p in sorted(self._diary_dir.glob("*.json")):
            try:
                d = date.fromisoformat(p.stem)
            except ValueError:
                continue
            if d < since:
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for e in data.get("entries", []):
                    try:
                        results.append(DiaryEntry.model_validate(e))
                    except Exception:
                        continue
            except Exception:
                continue
        return results
