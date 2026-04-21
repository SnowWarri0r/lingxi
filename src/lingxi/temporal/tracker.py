"""Tracks per-recipient interaction timestamps for time-awareness."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field


class InteractionRecord(BaseModel):
    recipient_id: str
    channel: str  # "feishu", "web", "cli"
    last_interaction: datetime = Field(default_factory=datetime.now)
    last_proactive_sent: datetime | None = None
    relationship_level: int = 1
    total_turns: int = 0
    session_count: int = 0
    first_interaction: datetime | None = None
    last_level_evaluation: datetime | None = None
    # Persisted emotion state: {dimension_name: intensity}
    emotion_dimensions: dict[str, float] = Field(default_factory=dict)
    emotion_last_decay: datetime | None = None
    emotion_narrative: str = ""


class InteractionTracker:
    """Persists per-recipient interaction timestamps. Survives restarts.

    File format:
    {
      "records": {
        "feishu:oc_xxx": { ...InteractionRecord... },
        "web:sess_abc": { ... }
      }
    }
    """

    def __init__(self, data_dir: Path | str):
        self._path = Path(data_dir) / "interactions.json"
        self._records: dict[str, InteractionRecord] = {}
        self._lock = asyncio.Lock()
        self._loaded = False

    @staticmethod
    def _key(channel: str, recipient_id: str) -> str:
        return f"{channel}:{recipient_id}"

    async def load(self) -> None:
        if not self._path.exists():
            self._loaded = True
            return

        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            for key, rdata in data.get("records", {}).items():
                try:
                    self._records[key] = InteractionRecord.model_validate(rdata)
                except Exception:
                    continue
        except (json.JSONDecodeError, OSError):
            pass

        self._loaded = True

    async def save(self) -> None:
        async with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "records": {
                    key: rec.model_dump(mode="json")
                    for key, rec in self._records.items()
                }
            }
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            tmp.rename(self._path)

    def record_interaction(self, channel: str, recipient_id: str) -> InteractionRecord:
        """Called on every user message. Updates last_interaction to now."""
        key = self._key(channel, recipient_id)
        now = datetime.now()

        if key in self._records:
            rec = self._records[key]
            rec.last_interaction = now
            rec.total_turns += 1
        else:
            rec = InteractionRecord(
                recipient_id=recipient_id,
                channel=channel,
                last_interaction=now,
                first_interaction=now,
                total_turns=1,
            )
            self._records[key] = rec

        return rec

    def record_session_end(self, channel: str, recipient_id: str) -> None:
        """Called when a session ends. Increments session_count."""
        key = self._key(channel, recipient_id)
        if key in self._records:
            self._records[key].session_count += 1

    def record_proactive_sent(self, channel: str, recipient_id: str) -> None:
        key = self._key(channel, recipient_id)
        if key in self._records:
            self._records[key].last_proactive_sent = datetime.now()

    def update_relationship_level(
        self, channel: str, recipient_id: str, new_level: int
    ) -> None:
        key = self._key(channel, recipient_id)
        if key in self._records:
            self._records[key].relationship_level = new_level
            self._records[key].last_level_evaluation = datetime.now()

    def get_record(
        self, channel: str, recipient_id: str
    ) -> InteractionRecord | None:
        return self._records.get(self._key(channel, recipient_id))

    def get_silence_duration(
        self, channel: str, recipient_id: str
    ) -> timedelta | None:
        rec = self.get_record(channel, recipient_id)
        if rec is None:
            return None
        return datetime.now() - rec.last_interaction

    def all_records(self) -> list[InteractionRecord]:
        return list(self._records.values())
