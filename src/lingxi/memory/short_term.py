"""Short-term memory: sliding window conversation buffer with persistence."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """A single turn in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict = Field(default_factory=dict)


class ShortTermMemory:
    """Bounded sliding-window buffer of recent conversation turns.

    Supports optional per-recipient persistence to disk. When a recipient
    context is active, the buffer belongs to that recipient and is restored
    from disk on load.
    """

    def __init__(
        self,
        max_turns: int = 30,
        data_dir: Path | str | None = None,
    ):
        self.max_turns = max_turns
        self._buffer: deque[ConversationTurn] = deque(maxlen=max_turns)
        self._data_dir = Path(data_dir) if data_dir else None
        self._current_recipient: str | None = None
        self._lock = asyncio.Lock()

    def _path_for(self, recipient_key: str) -> Path | None:
        if self._data_dir is None:
            return None
        safe = "".join(c if c.isalnum() or c in "-_:" else "_" for c in recipient_key)
        return self._data_dir / "short_term" / f"{safe}.json"

    async def switch_recipient(self, recipient_key: str | None) -> None:
        """Switch active recipient context. Saves current + loads target."""
        async with self._lock:
            if self._current_recipient == recipient_key:
                return

            # Save current buffer
            if self._current_recipient is not None:
                await self._save_to_disk(self._current_recipient)

            # Load target
            self._buffer.clear()
            if recipient_key is not None:
                await self._load_from_disk(recipient_key)

            self._current_recipient = recipient_key

    async def _save_to_disk(self, recipient_key: str) -> None:
        path = self._path_for(recipient_key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "recipient": recipient_key,
            "turns": [t.model_dump(mode="json") for t in self._buffer],
        }
        tmp = path.with_suffix(".tmp")

        def _write():
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            tmp.rename(path)

        await asyncio.to_thread(_write)

    async def _load_from_disk(self, recipient_key: str) -> None:
        path = self._path_for(recipient_key)
        if path is None or not path.exists():
            return

        def _read():
            with open(path, encoding="utf-8") as f:
                return json.load(f)

        try:
            data = await asyncio.to_thread(_read)
            for t in data.get("turns", []):
                try:
                    self._buffer.append(ConversationTurn.model_validate(t))
                except Exception:
                    continue
        except (json.JSONDecodeError, OSError):
            pass

    async def persist_current(self) -> None:
        """Save current buffer if a recipient context is active."""
        if self._current_recipient is not None:
            async with self._lock:
                await self._save_to_disk(self._current_recipient)

    def add_turn(self, role: str, content: str, **metadata: object) -> ConversationTurn:
        """Add a conversation turn to the buffer."""
        turn = ConversationTurn(role=role, content=content, metadata=metadata)
        self._buffer.append(turn)
        return turn

    def get_history(self, last_n: int | None = None) -> list[ConversationTurn]:
        turns = list(self._buffer)
        if last_n is not None:
            turns = turns[-last_n:]
        return turns

    def get_messages(self, last_n: int | None = None) -> list[dict]:
        turns = self.get_history(last_n)
        return [{"role": t.role, "content": t.content} for t in turns]

    def clear(self) -> list[ConversationTurn]:
        turns = list(self._buffer)
        self._buffer.clear()
        return turns

    @property
    def turn_count(self) -> int:
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0
