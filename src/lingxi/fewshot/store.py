"""Persistence layer for fewshot: AnnotationStore (per-turn JSON).

FewShotStore (ChromaDB pool) will be added in Task 7.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from lingxi.fewshot.models import AnnotationKind, AnnotationTurn


class AnnotationStore:
    """Persist AnnotationTurn records under data_dir/turns/<turn_id>.json.

    Cleanup policy:
      - Unannotated turns older than 30 days are removed.
      - Annotated turns older than 7 days past annotation can be removed
        (the upgraded FewShotSample supersedes them).
    """

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.turns_dir = self.data_dir / "turns"
        self.turns_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, turn_id: str) -> Path:
        # Defend against path traversal
        safe = turn_id.replace("/", "_").replace("..", "_")
        return self.turns_dir / f"{safe}.json"

    async def record(self, turn: AnnotationTurn) -> None:
        def _write():
            path = self._path(turn.turn_id)
            path.write_text(turn.model_dump_json(indent=2), encoding="utf-8")
        await asyncio.to_thread(_write)

    async def get_turn(self, turn_id: str) -> AnnotationTurn | None:
        path = self._path(turn_id)
        if not path.exists():
            return None

        def _read() -> dict[str, Any]:
            return json.loads(path.read_text(encoding="utf-8"))

        data = await asyncio.to_thread(_read)
        return AnnotationTurn.model_validate(data)

    async def update_annotation(
        self,
        turn_id: str,
        kind: AnnotationKind,
        correction: str | None = None,
    ) -> AnnotationTurn:
        turn = await self.get_turn(turn_id)
        if turn is None:
            raise KeyError(turn_id)
        turn.annotation = kind
        if correction is not None:
            turn.correction = correction
        await self.record(turn)
        return turn

    async def cleanup(
        self,
        max_age_unannotated_days: int = 30,
        max_age_annotated_days: int = 7,
    ) -> int:
        """Remove stale turn files. Returns count deleted."""
        cutoff_unannot = (datetime.now() - timedelta(days=max_age_unannotated_days)).timestamp()
        cutoff_annot = (datetime.now() - timedelta(days=max_age_annotated_days)).timestamp()

        def _scan() -> int:
            count = 0
            for path in self.turns_dir.glob("*.json"):
                try:
                    mtime = path.stat().st_mtime
                    data = json.loads(path.read_text(encoding="utf-8"))
                    kind = data.get("annotation", "none")
                    if kind == "none" and mtime < cutoff_unannot:
                        path.unlink()
                        count += 1
                    elif kind != "none" and mtime < cutoff_annot:
                        path.unlink()
                        count += 1
                except Exception:
                    continue
            return count

        return await asyncio.to_thread(_scan)
