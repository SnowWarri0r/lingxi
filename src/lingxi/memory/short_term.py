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


# How many of her messages in an unanswered run survive eviction: the last,
# which is what he replies to when he comes back, and the one before it.
_KEEP_OF_RUN = 2


def trim_to_cap(turns: list["ConversationTurn"], cap: int) -> list["ConversationTurn"]:
    """Bring `turns` down to `cap`, spending her unanswered openers first.

    Evicting by age alone let her own monologue decide what she forgot about
    him. Proactive openers arrive about three a day whether he answers or not,
    so on 2026-09-23 his buffer held 30 turns across six days — 24 hers, 6
    his — and everything he had said before 09-17 was gone. Nothing else held
    it: episode summaries were retired, thread_summary lives in memory and
    dies on restart, and no fact about him had been written since 08-28.

    So a run of consecutive assistant turns — messages she sent with no reply
    in between — gives up its older members before anything else goes. The
    last few of each run stay, so the message he eventually answers is still
    there to be answered. With no such turn left, the oldest turn goes, which
    is the old behaviour. Below the cap nothing is dropped at all.
    """
    turns = list(turns)
    if cap <= 0 or len(turns) <= cap:
        return turns
    while len(turns) > cap:
        victim = None
        i = 0
        while i < len(turns) and victim is None:
            if turns[i].role != "assistant":
                i += 1
                continue
            j = i
            while j < len(turns) and turns[j].role == "assistant":
                j += 1
            if j - i > _KEEP_OF_RUN:
                victim = i
            i = j
        turns.pop(0 if victim is None else victim)
    return turns


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
        # No maxlen: a deque drops from the left, which is by age alone.
        self._buffer: deque[ConversationTurn] = deque()
        self._data_dir = Path(data_dir) if data_dir else None
        self._current_recipient: str | None = None
        self._lock = asyncio.Lock()

    def _enforce_cap(self) -> None:
        if self.max_turns and len(self._buffer) > self.max_turns:
            self._buffer = deque(trim_to_cap(list(self._buffer), self.max_turns))

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
            self._enforce_cap()
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
        self._enforce_cap()
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

    async def append_for_recipient(
        self,
        recipient_key: str,
        role: str,
        content: str,
        **metadata: object,
    ) -> None:
        """Atomically append a turn to `recipient_key`'s buffer without
        switching the singleton active recipient.

        Used by the proactive scheduler to record its own outgoing
        message in the recipient's short-term history so the very NEXT
        chat turn sees the proactive message as conversational context.
        Without this the user's reply to a proactive arrives with no
        preceding assistant turn in scope, so the model treats it as a
        cold opener (production trace: proactive 'super cat 记得我', user
        '怎么说' → Aria '怎么了?' because she didn't 'see' her own
        proactive in history).
        """
        async with self._lock:
            new_turn = ConversationTurn(
                role=role, content=content, metadata=dict(metadata)
            )
            # If this IS the active recipient, mutate the buffer and save
            if self._current_recipient == recipient_key:
                self._buffer.append(new_turn)
                self._enforce_cap()
                await self._save_to_disk(recipient_key)
                return

            # Non-active: read existing file, append, write back atomically
            path = self._path_for(recipient_key)
            if path is None:
                return
            existing: list[ConversationTurn] = []
            if path.exists():
                try:
                    data = await asyncio.to_thread(
                        lambda: json.loads(path.read_text(encoding="utf-8"))
                    )
                    for t_dict in data.get("turns", []):
                        try:
                            existing.append(ConversationTurn.model_validate(t_dict))
                        except Exception:
                            continue
                except (json.JSONDecodeError, OSError):
                    pass
            existing.append(new_turn)
            if self.max_turns:
                existing = trim_to_cap(existing, self.max_turns)
            data = {
                "recipient": recipient_key,
                "turns": [t.model_dump(mode="json") for t in existing],
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")

            def _write():
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                tmp.rename(path)

            await asyncio.to_thread(_write)

    def is_empty(self) -> bool:
        return len(self._buffer) == 0
