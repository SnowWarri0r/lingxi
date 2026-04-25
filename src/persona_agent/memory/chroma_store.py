"""ChromaDB-backed long-term memory store.

Uses ChromaDB's PersistentClient for automatic disk persistence and
HNSW indexing for efficient similarity search. Scales to millions of
entries with sub-linear retrieval time.

Auto-migrates from legacy JSON store on first load.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from persona_agent.memory.base import MemoryEntry, MemoryStore, MemoryType


def _entry_to_metadata(entry: MemoryEntry) -> dict[str, Any]:
    """Serialize MemoryEntry fields as Chroma metadata (must be scalar).

    recipient_key (stored under extra.recipient_key) is promoted to top-level
    so Chroma can filter on it efficiently.
    """
    recipient_key = (entry.metadata or {}).get("recipient_key", "") if entry.metadata else ""
    return {
        "memory_type": entry.memory_type.value,
        "timestamp": entry.timestamp.isoformat(),
        "importance": float(entry.importance),
        "tags": json.dumps(entry.tags, ensure_ascii=False) if entry.tags else "",
        "extra": json.dumps(entry.metadata, ensure_ascii=False) if entry.metadata else "",
        "access_count": entry.access_count,
        "last_accessed": entry.last_accessed.isoformat() if entry.last_accessed else "",
        "recipient_key": recipient_key or "_global",
    }


def _metadata_to_entry(entry_id: str, content: str, meta: dict[str, Any], embedding: list[float] | None) -> MemoryEntry:
    """Reconstruct a MemoryEntry from Chroma metadata."""
    tags_raw = meta.get("tags", "")
    tags = json.loads(tags_raw) if tags_raw else []
    extra_raw = meta.get("extra", "")
    extra = json.loads(extra_raw) if extra_raw else {}

    last_accessed = None
    last_accessed_str = meta.get("last_accessed", "")
    if last_accessed_str:
        try:
            last_accessed = datetime.fromisoformat(last_accessed_str)
        except ValueError:
            pass

    return MemoryEntry(
        id=entry_id,
        content=content,
        memory_type=MemoryType(meta.get("memory_type", "long_term")),
        timestamp=datetime.fromisoformat(meta["timestamp"]) if meta.get("timestamp") else datetime.now(),
        importance=float(meta.get("importance", 0.5)),
        tags=tags,
        metadata=extra,
        access_count=int(meta.get("access_count", 0)),
        last_accessed=last_accessed,
        embedding=embedding,
    )


class ChromaMemoryStore(MemoryStore):
    """Long-term memory backed by ChromaDB.

    Collection name includes embedding dimension so different embedding
    models use separate collections automatically. Switching models
    won't cause dimension mismatch errors.
    """

    DEFAULT_COLLECTION_NAME = "persona_long_term_memory"

    def __init__(
        self,
        db_path: str | Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        max_entries: int = 100000,
        importance_threshold: float = 0.3,
        embedding_dim: int | None = None,
    ):
        self.db_path = Path(db_path)
        self.base_collection_name = collection_name
        self.collection_name = (
            f"{collection_name}_d{embedding_dim}" if embedding_dim else collection_name
        )
        self.max_entries = max_entries
        self.importance_threshold = importance_threshold
        self.embedding_dim = embedding_dim
        self._client: Any = None
        self._collection: Any = None
        self._init_lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self._collection is not None:
            return
        async with self._init_lock:
            if self._collection is not None:
                return
            await asyncio.to_thread(self._init_sync)

    def _init_sync(self) -> None:
        import chromadb
        from chromadb.config import Settings

        self.db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=False),
        )
        # Note: we provide our own embeddings, so no embedding_function needed
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    DUPLICATE_THRESHOLD = 0.92

    async def store(self, entry: MemoryEntry, dedup: bool = True) -> str:
        """Store an entry. If dedup=True and a near-duplicate exists,
        merges importance/access into the existing entry instead.
        """
        await self._ensure_loaded()
        entry.memory_type = MemoryType.LONG_TERM

        # Dedup check (only if we have an embedding to compare)
        if dedup and entry.embedding:
            existing_id = await self._find_duplicate(entry)
            if existing_id is not None:
                await self._merge_into(existing_id, entry)
                return existing_id

        def _add():
            self._collection.add(
                ids=[entry.id],
                documents=[entry.content],
                embeddings=[entry.embedding] if entry.embedding else None,
                metadatas=[_entry_to_metadata(entry)],
            )

        await asyncio.to_thread(_add)
        return entry.id

    async def _find_duplicate(self, entry: MemoryEntry) -> str | None:
        """Look for a near-duplicate within the same recipient scope."""
        recipient_key = (entry.metadata or {}).get("recipient_key", "_global")

        def _query():
            return self._collection.query(
                query_embeddings=[entry.embedding],
                n_results=1,
                where={"recipient_key": recipient_key},
                include=["distances"],
            )

        try:
            result = await asyncio.to_thread(_query)
        except Exception:
            return None

        ids_list = result.get("ids") or [[]]
        dists_list = result.get("distances") or [[]]
        if not ids_list or not ids_list[0]:
            return None

        similarity = max(0.0, 1.0 - dists_list[0][0])
        if similarity >= self.DUPLICATE_THRESHOLD:
            return ids_list[0][0]
        return None

    async def _merge_into(self, existing_id: str, new_entry: MemoryEntry) -> None:
        """Bump importance/recency on existing entry instead of duplicating."""
        existing = await self.get_by_id(existing_id)
        if existing is None:
            return

        # Take max importance, increment access_count, refresh timestamp
        existing.importance = max(existing.importance, new_entry.importance)
        existing.access_count += 1
        existing.last_accessed = datetime.now()
        existing.timestamp = datetime.now()

        # Merge tags
        merged_tags = list(set(existing.tags + new_entry.tags))
        existing.tags = merged_tags

        def _update():
            self._collection.update(
                ids=[existing_id],
                metadatas=[_entry_to_metadata(existing)],
            )

        await asyncio.to_thread(_update)

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        query_embedding: list[float] | None = None,
        recipient_key: str | None = None,
    ) -> list[MemoryEntry]:
        """Retrieve memories.

        recipient_key: If set, only return memories scoped to this recipient
            OR global memories (recipient_key='_global'). If None, returns all.
        """
        await self._ensure_loaded()

        if query_embedding:
            return await self._retrieve_by_embedding(query_embedding, limit, recipient_key)
        return await self._retrieve_by_keyword(query, limit, recipient_key)

    def _build_where(self, recipient_key: str | None) -> dict | None:
        if recipient_key is None:
            return None
        # Include recipient-scoped + global memories
        return {"recipient_key": {"$in": [recipient_key, "_global"]}}

    async def _retrieve_by_embedding(
        self, query_embedding: list[float], limit: int, recipient_key: str | None = None
    ) -> list[MemoryEntry]:
        where = self._build_where(recipient_key)

        def _query():
            kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": min(limit * 3, 50),
                "include": ["documents", "metadatas", "distances", "embeddings"],
            }
            if where:
                kwargs["where"] = where
            return self._collection.query(**kwargs)

        result = await asyncio.to_thread(_query)
        return self._process_query_result(result, limit)

    async def _retrieve_by_keyword(
        self, query: str, limit: int, recipient_key: str | None = None
    ) -> list[MemoryEntry]:
        """Fallback keyword search via Chroma's document search."""
        where = self._build_where(recipient_key)

        def _get_all():
            kwargs = {"include": ["documents", "metadatas", "embeddings"]}
            if where:
                kwargs["where"] = where
            return self._collection.get(**kwargs)

        result = await asyncio.to_thread(_get_all)
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        embs = result.get("embeddings")

        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored: list[tuple[float, MemoryEntry]] = []

        for i, doc in enumerate(docs):
            text_words = set(doc.lower().split())
            overlap = len(query_words & text_words)
            if overlap == 0:
                continue
            meta = metas[i] if i < len(metas) else {}
            emb = None
            if embs is not None and i < len(embs) and embs[i] is not None:
                emb = list(embs[i])
            entry = _metadata_to_entry(ids[i], doc, meta, emb)
            score = (overlap / max(len(query_words), 1)) * 0.7 + entry.importance * 0.3
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [e for _, e in scored[:limit]]
        await self._update_access(results)
        return results

    def _process_query_result(self, result: dict, limit: int) -> list[MemoryEntry]:
        ids_list = result.get("ids") or [[]]
        docs_list = result.get("documents") or [[]]
        metas_list = result.get("metadatas") or [[]]
        dists_list = result.get("distances") or [[]]
        embs_list = result.get("embeddings")

        if not ids_list or not ids_list[0]:
            return []

        ids = ids_list[0]
        docs = docs_list[0]
        metas = metas_list[0]
        dists = dists_list[0]
        embs = embs_list[0] if embs_list is not None else None

        scored: list[tuple[float, MemoryEntry]] = []
        for i, entry_id in enumerate(ids):
            doc = docs[i]
            meta = metas[i]
            distance = dists[i]  # cosine distance in [0, 2]
            similarity = max(0.0, 1.0 - distance)
            emb = None
            if embs is not None and i < len(embs) and embs[i] is not None:
                emb = list(embs[i])

            entry = _metadata_to_entry(entry_id, doc, meta, emb)
            # Blend similarity with importance
            score = similarity * 0.7 + entry.importance * 0.3
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [e for _, e in scored[:limit]]

        # Update access stats asynchronously (fire and forget via asyncio.to_thread)
        asyncio.create_task(self._update_access(results))
        return results

    async def _update_access(self, entries: list[MemoryEntry]) -> None:
        if not entries:
            return
        now = datetime.now()

        def _update():
            for entry in entries:
                entry.access_count += 1
                entry.last_accessed = now
                try:
                    self._collection.update(
                        ids=[entry.id],
                        metadatas=[_entry_to_metadata(entry)],
                    )
                except Exception:
                    pass  # non-critical

        await asyncio.to_thread(_update)

    async def get_by_id(self, entry_id: str) -> MemoryEntry | None:
        await self._ensure_loaded()

        def _get():
            return self._collection.get(
                ids=[entry_id],
                include=["documents", "metadatas", "embeddings"],
            )

        result = await asyncio.to_thread(_get)
        ids = result.get("ids") or []
        if not ids:
            return None

        embs = result.get("embeddings")
        emb = None
        if embs is not None and len(embs) > 0 and embs[0] is not None:
            emb = list(embs[0])
        return _metadata_to_entry(
            ids[0], result["documents"][0], result["metadatas"][0], emb
        )

    async def delete(self, entry_id: str) -> bool:
        await self._ensure_loaded()

        def _delete():
            existing = self._collection.get(ids=[entry_id])
            if not existing.get("ids"):
                return False
            self._collection.delete(ids=[entry_id])
            return True

        return await asyncio.to_thread(_delete)

    async def list_all(self) -> list[MemoryEntry]:
        await self._ensure_loaded()

        def _list():
            return self._collection.get(
                include=["documents", "metadatas", "embeddings"],
            )

        result = await asyncio.to_thread(_list)
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        embs = result.get("embeddings")

        out: list[MemoryEntry] = []
        for i, entry_id in enumerate(ids):
            emb = None
            if embs is not None and i < len(embs) and embs[i] is not None:
                emb = list(embs[i])
            out.append(_metadata_to_entry(entry_id, docs[i], metas[i], emb))
        return out

    @property
    def _entries(self) -> dict[str, Any]:
        """Backward-compat shim for MemoryManager.get_stats()."""
        # Lazy, cheap count
        if self._collection is None:
            return {}

        def _count():
            try:
                return self._collection.count()
            except Exception:
                return 0

        count = _count()
        # Return a dict-like object with __len__
        return {f"id_{i}": None for i in range(count)}

    async def count(self) -> int:
        await self._ensure_loaded()
        return await asyncio.to_thread(self._collection.count)

    async def save_to_disk(self, path: str) -> None:
        """Chroma auto-persists. No-op but keep for interface compat."""
        await self._ensure_loaded()
        # Attempt migration from legacy JSON file if present
        legacy_path = Path(path)
        if legacy_path.exists() and legacy_path.suffix == ".json":
            pass  # already migrated on load

    async def load_from_disk(self, path: str) -> None:
        """Load + auto-migrate from legacy JSON if present."""
        await self._ensure_loaded()
        legacy_path = Path(path)

        # Migrate from legacy long_term.json if Chroma collection is empty
        current_count = await self.count()
        if current_count == 0 and legacy_path.exists() and legacy_path.suffix == ".json":
            await self._migrate_from_json(legacy_path)

    async def _migrate_from_json(self, json_path: Path) -> None:
        """One-time migration from the old JSON format."""
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        if not isinstance(data, list) or not data:
            return

        print(f"[chroma] migrating {len(data)} entries from {json_path}")
        count = 0
        for item in data:
            try:
                entry = MemoryEntry.model_validate(item)
                await self.store(entry)
                count += 1
            except Exception:
                continue

        # Rename old file so migration doesn't re-run
        try:
            backup = json_path.with_suffix(".json.migrated")
            json_path.rename(backup)
            print(f"[chroma] migration complete: {count} entries → {backup} backed up")
        except OSError:
            pass
