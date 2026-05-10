"""Short-term memory: sliding window conversation buffer with persistence."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """A single turn in the conversation.

    `summary` is filled lazily by the mid-term compactor when a turn ages
    past the verbatim window. Once summarized, `content` is preserved for
    audit but the model sees `summary` instead via ContextAssembler.
    """

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict = Field(default_factory=dict)
    # Mid-term compressed form. None = not yet aged into mid-term.
    summary: str | None = None


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

    # ---- Read-only snapshots for OTHER recipients without touching active state ----
    async def snapshot_for_recipient(self, recipient_key: str) -> list[ConversationTurn]:
        """Load turns for `recipient_key` from disk WITHOUT switching active state.

        Use this for read-only access (e.g., proactive scheduler peeking at
        a recipient's history while reactive may be in-flight for another).
        Mutating global `_current_recipient` here would let a concurrent
        reactive turn append its assistant reply to the wrong buffer.
        """
        path = self._path_for(recipient_key)
        if path is None or not path.exists():
            return []

        def _read():
            with open(path, encoding="utf-8") as f:
                return json.load(f)

        try:
            data = await asyncio.to_thread(_read)
        except (json.JSONDecodeError, OSError):
            return []

        out: list[ConversationTurn] = []
        for t in data.get("turns", []):
            try:
                out.append(ConversationTurn.model_validate(t))
            except Exception:
                continue
        return out

    async def write_for_recipient(
        self, recipient_key: str, turns: list[ConversationTurn]
    ) -> None:
        """Persist `turns` for `recipient_key` WITHOUT changing active state.

        Used by mid-term compaction for a non-active recipient. Goes through
        the same atomic temp-rename write as the normal save path, but does
        NOT touch `_buffer` or `_current_recipient`.
        """
        path = self._path_for(recipient_key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "recipient": recipient_key,
            "turns": [t.model_dump(mode="json") for t in turns],
        }
        tmp = path.with_suffix(".tmp")

        def _write():
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            tmp.rename(path)

        # Lock so a concurrent active-recipient save (matching key) doesn't
        # race with this write. The lock is shared with all other I/O on
        # the singleton instance — coarse but correct.
        async with self._lock:
            await asyncio.to_thread(_write)

    async def apply_summaries_atomic(
        self, recipient_key: str, summary_map: dict
    ) -> int:
        """Read latest file, apply summaries by identity, write back — all
        under the singleton lock so concurrent appends/saves can't get lost.

        `summary_map` keys are (timestamp_iso, role, content_prefix_60) →
        summary string. We only set summary on turns that match a key AND
        currently have summary=None. Returns count merged.

        Also: turns appended to the file between snapshot and now are
        preserved (we read the LATEST file inside the lock).
        """
        path = self._path_for(recipient_key)
        if path is None or not summary_map:
            return 0

        def _read_latest() -> list[ConversationTurn]:
            if not path.exists():
                return []
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                return []
            out: list[ConversationTurn] = []
            for t in data.get("turns", []):
                try:
                    out.append(ConversationTurn.model_validate(t))
                except Exception:
                    continue
            return out

        def _atomic_write(turns: list[ConversationTurn]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "recipient": recipient_key,
                "turns": [t.model_dump(mode="json") for t in turns],
            }
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            tmp.rename(path)

        merged = 0
        async with self._lock:
            latest = await asyncio.to_thread(_read_latest)
            if not latest:
                return 0
            for turn in latest:
                if turn.summary is not None:
                    continue
                key = (
                    turn.timestamp.isoformat(),
                    turn.role,
                    (turn.content or "")[:60],
                )
                if key in summary_map:
                    turn.summary = summary_map[key]
                    merged += 1
            if merged > 0:
                await asyncio.to_thread(_atomic_write, latest)

            # If the active recipient matches, also patch the in-memory
            # buffer so the next get_history sees the summaries without a
            # disk reload.
            if merged > 0 and self._current_recipient == recipient_key:
                for turn in self._buffer:
                    if turn.summary is not None:
                        continue
                    key = (
                        turn.timestamp.isoformat(),
                        turn.role,
                        (turn.content or "")[:60],
                    )
                    if key in summary_map:
                        turn.summary = summary_map[key]

        return merged

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0
