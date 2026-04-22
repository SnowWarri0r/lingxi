"""Persistence layer for fewshot: AnnotationStore (per-turn JSON).

FewShotStore (ChromaDB pool) added in Task 7.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from lingxi.fewshot.models import AnnotationKind, AnnotationTurn, FewShotSample


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


@dataclass
class FewShotQueryResult:
    sample: FewShotSample
    similarity: float


class FewShotStore:
    """ChromaDB-backed pool of FewShotSamples, with a JSONL backup for disaster recovery.

    Collection name is dim-suffixed (fewshot_pool_d<N>) to avoid dim-mismatch
    errors when embeddings change.
    """

    def __init__(self, data_dir: Path | str, embedding_dim: int):
        self.data_dir = Path(data_dir)
        self.chroma_dir = self.data_dir / "chroma"
        self.backup_path = self.data_dir / "fewshot" / "samples.jsonl"
        self.embedding_dim = embedding_dim
        self.collection_name = f"fewshot_pool_d{embedding_dim}"
        self._client: Any = None
        self._collection: Any = None
        self._lock = asyncio.Lock()

    async def init(self) -> None:
        async with self._lock:
            if self._collection is not None:
                return
            await asyncio.to_thread(self._init_sync)

    def _init_sync(self) -> None:
        import chromadb
        from chromadb.config import Settings

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.backup_path.parent.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add(self, sample: FewShotSample, embedding: list[float]) -> None:
        await self.init()
        if len(embedding) != self.embedding_dim:
            raise ValueError(
                f"embedding dim {len(embedding)} != store dim {self.embedding_dim}"
            )

        meta = {
            "source": sample.source,
            # ChromaDB metadata disallows None; use "" as sentinel
            "recipient_key": sample.recipient_key or "",
            "context_summary": sample.context_summary,
            "tags": ",".join(sample.tags),
            "original_speech": sample.original_speech or "",
            "corrected_speech": sample.corrected_speech,
            "inner_thought": sample.inner_thought,
            "created_at": sample.created_at.isoformat(),
        }

        def _add():
            self._collection.add(
                ids=[sample.id],
                documents=[sample.inner_thought or sample.context_summary],
                embeddings=[embedding],
                metadatas=[meta],
            )

        await asyncio.to_thread(_add)
        await self._append_backup(sample)

    async def _append_backup(self, sample: FewShotSample) -> None:
        def _append():
            with self.backup_path.open("a", encoding="utf-8") as fh:
                fh.write(sample.model_dump_json() + "\n")
        await asyncio.to_thread(_append)

    async def query(
        self,
        query_embedding: list[float],
        k: int = 6,
        recipient_key: str | None = None,
    ) -> list[FewShotQueryResult]:
        """Return the top-k samples by cosine similarity, filtered by recipient.

        If recipient_key is given, returns samples where metadata.recipient_key
        matches OR is empty (i.e. global seeds). Cross-user samples are excluded.
        """
        await self.init()

        where_clause: dict[str, Any] | None = None
        if recipient_key is not None:
            where_clause = {
                "$or": [
                    {"recipient_key": recipient_key},
                    {"recipient_key": ""},
                ]
            }

        def _query() -> dict[str, Any]:
            return self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause,
                include=["metadatas", "documents", "distances"],
            )

        result = await asyncio.to_thread(_query)
        ids_list = (result.get("ids") or [[]])[0]
        metas_list = (result.get("metadatas") or [[]])[0]
        dists_list = (result.get("distances") or [[]])[0]

        out: list[FewShotQueryResult] = []
        for sample_id, meta, dist in zip(ids_list, metas_list, dists_list):
            similarity = max(0.0, 1.0 - float(dist))
            sample = FewShotSample(
                id=sample_id,
                inner_thought=meta.get("inner_thought", ""),
                corrected_speech=meta.get("corrected_speech", ""),
                context_summary=meta.get("context_summary", ""),
                original_speech=meta.get("original_speech") or None,
                tags=[t for t in meta.get("tags", "").split(",") if t],
                recipient_key=(meta.get("recipient_key") or None),
                source=meta.get("source", "seed"),
                created_at=datetime.fromisoformat(
                    meta.get("created_at") or datetime.now().isoformat()
                ),
            )
            out.append(FewShotQueryResult(sample=sample, similarity=similarity))
        return out

    async def count(self) -> int:
        await self.init()

        def _count():
            return self._collection.count()

        return await asyncio.to_thread(_count)
