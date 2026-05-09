"""Per-recipient file-based store for RelationalMemory.

Same pattern as inner_life/store.py: async lock for serialization,
atomic temp+rename writes, and an update_memory(key, mutator) for
read-modify-write under one lock to prevent the load→mutate→save
race that bit the inner_life path (Codex P2 fix).

Layout:
  data/memory/relational/<safe_recipient_key>.json
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from lingxi.relational.models import RelationalMemory


class RelationalMemoryStore:
    def __init__(self, data_dir: Path | str):
        self._root = Path(data_dir) / "relational"
        self._lock = asyncio.Lock()

    @staticmethod
    def _safe_key(key: str) -> str:
        return "".join(c if c.isalnum() or c in "-_:" else "_" for c in key)

    def _path(self, recipient_key: str) -> Path:
        return self._root / f"{self._safe_key(recipient_key)}.json"

    async def _atomic_write(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")

        def _write():
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            tmp.rename(path)

        await asyncio.to_thread(_write)

    async def load(self, recipient_key: str) -> RelationalMemory:
        path = self._path(recipient_key)
        if not path.exists():
            return RelationalMemory(recipient_key=recipient_key)
        try:
            data = await asyncio.to_thread(
                lambda: json.loads(path.read_text(encoding="utf-8"))
            )
            return RelationalMemory.model_validate(data)
        except Exception:
            # Corrupted file falls through to fresh blank — better than
            # crashing the chat path
            return RelationalMemory(recipient_key=recipient_key)

    async def save(self, memory: RelationalMemory) -> None:
        async with self._lock:
            await self._atomic_write(
                self._path(memory.recipient_key),
                memory.model_dump(mode="json"),
            )

    async def update_memory(self, recipient_key: str, mutator) -> RelationalMemory:
        """Atomic read-modify-write under the lock.

        mutator(memory) is called inside the lock; whatever it does to
        the memory object is what gets persisted. This is what we use
        from the reflection extractor — load fresh, merge in newly-
        extracted entries, write — without races against any concurrent
        save.
        """
        async with self._lock:
            path = self._path(recipient_key)
            if not path.exists():
                memory = RelationalMemory(recipient_key=recipient_key)
            else:
                try:
                    data = await asyncio.to_thread(
                        lambda: json.loads(path.read_text(encoding="utf-8"))
                    )
                    memory = RelationalMemory.model_validate(data)
                except Exception:
                    memory = RelationalMemory(recipient_key=recipient_key)
            mutator(memory)
            await self._atomic_write(path, memory.model_dump(mode="json"))
            return memory

    async def list_recipients(self) -> list[str]:
        """Return recipient keys that have any relational memory file."""
        if not self._root.exists():
            return []
        result = []
        for p in self._root.glob("*.json"):
            try:
                data = await asyncio.to_thread(
                    lambda p=p: json.loads(p.read_text(encoding="utf-8"))
                )
                key = data.get("recipient_key")
                if key:
                    result.append(key)
            except Exception:
                continue
        return result
