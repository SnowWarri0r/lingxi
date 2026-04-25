"""ChromaDB-backed episodic memory store.

Separate collection from long-term facts - episodes are session summaries
with different semantics (timestamp-ordered, emotional tone, key topics).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from persona_agent.memory.base import EpisodeEntry


def _episode_to_metadata(ep: EpisodeEntry, recipient_key: str = "_global") -> dict[str, Any]:
    return {
        "timestamp": ep.timestamp.isoformat(),
        "emotional_tone": ep.emotional_tone or "neutral",
        "key_topics": json.dumps(ep.key_topics, ensure_ascii=False) if ep.key_topics else "",
        "turn_count": int(ep.turn_count),
        "recipient_key": recipient_key,
    }


def _metadata_to_episode(
    ep_id: str,
    summary: str,
    meta: dict[str, Any],
    embedding: list[float] | None,
) -> EpisodeEntry:
    topics_raw = meta.get("key_topics", "")
    topics = json.loads(topics_raw) if topics_raw else []
    return EpisodeEntry(
        id=ep_id,
        timestamp=datetime.fromisoformat(meta["timestamp"]) if meta.get("timestamp") else datetime.now(),
        summary=summary,
        emotional_tone=meta.get("emotional_tone", "neutral"),
        key_topics=topics,
        turn_count=int(meta.get("turn_count", 0)),
        embedding=embedding,
    )


class ChromaEpisodicMemory:
    """Episodic memory backed by ChromaDB (HNSW semantic search)."""

    DEFAULT_COLLECTION_NAME = "persona_episodic_memory"

    def __init__(
        self,
        db_path: str | Path,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        max_episodes: int = 5000,
        embedding_dim: int | None = None,
    ):
        self.db_path = Path(db_path)
        self.collection_name = (
            f"{collection_name}_d{embedding_dim}" if embedding_dim else collection_name
        )
        self.max_episodes = max_episodes
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
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def store_episode(
        self, episode: EpisodeEntry, recipient_key: str = "_global"
    ) -> str:
        await self._ensure_loaded()

        def _add():
            self._collection.add(
                ids=[episode.id],
                documents=[episode.summary],
                embeddings=[episode.embedding] if episode.embedding else None,
                metadatas=[_episode_to_metadata(episode, recipient_key)],
            )

        await asyncio.to_thread(_add)
        return episode.id

    async def retrieve_relevant(
        self,
        query: str,
        limit: int = 3,
        query_embedding: list[float] | None = None,
        recipient_key: str | None = None,
    ) -> list[EpisodeEntry]:
        await self._ensure_loaded()

        if query_embedding:
            return await self._retrieve_by_embedding(query_embedding, limit, recipient_key)
        return await self._retrieve_by_keyword(query, limit, recipient_key)

    def _build_where(self, recipient_key: str | None) -> dict | None:
        if recipient_key is None:
            return None
        return {"recipient_key": {"$in": [recipient_key, "_global"]}}

    async def _retrieve_by_embedding(
        self, query_embedding: list[float], limit: int, recipient_key: str | None = None
    ) -> list[EpisodeEntry]:
        where = self._build_where(recipient_key)

        def _query():
            kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": min(limit, 20),
                "include": ["documents", "metadatas", "distances", "embeddings"],
            }
            if where:
                kwargs["where"] = where
            return self._collection.query(**kwargs)

        result = await asyncio.to_thread(_query)
        ids_list = result.get("ids") or [[]]
        if not ids_list or not ids_list[0]:
            return []

        ids = ids_list[0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        embs_list = result.get("embeddings")
        embs = embs_list[0] if embs_list is not None else None

        out: list[EpisodeEntry] = []
        for i, ep_id in enumerate(ids):
            emb = None
            if embs is not None and i < len(embs) and embs[i] is not None:
                emb = list(embs[i])
            out.append(_metadata_to_episode(ep_id, docs[i], metas[i], emb))
        return out

    async def _retrieve_by_keyword(
        self, query: str, limit: int, recipient_key: str | None = None
    ) -> list[EpisodeEntry]:
        where = self._build_where(recipient_key)

        def _get_all():
            kwargs = {"include": ["documents", "metadatas"]}
            if where:
                kwargs["where"] = where
            return self._collection.get(**kwargs)

        result = await asyncio.to_thread(_get_all)
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []

        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored: list[tuple[float, EpisodeEntry]] = []
        for i, doc in enumerate(docs):
            text = doc.lower()
            text_words = set(text.split())
            overlap = len(query_words & text_words)
            if overlap == 0:
                continue
            meta = metas[i] if i < len(metas) else {}
            ep = _metadata_to_episode(ids[i], doc, meta, None)
            scored.append((overlap / max(len(query_words), 1), ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

    async def get_recent(self, limit: int = 5) -> list[EpisodeEntry]:
        """Get the most recent episodes (sorted by timestamp desc)."""
        await self._ensure_loaded()

        def _get_all():
            return self._collection.get(include=["documents", "metadatas"])

        result = await asyncio.to_thread(_get_all)
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []

        episodes = [
            _metadata_to_episode(ids[i], docs[i], metas[i], None)
            for i in range(len(ids))
        ]
        episodes.sort(key=lambda e: e.timestamp, reverse=True)
        return episodes[:limit]

    @property
    def episode_count(self) -> int:
        if self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    async def save_to_disk(self, path: str) -> None:
        """Chroma auto-persists. Migration check only."""
        await self._ensure_loaded()

    async def load_from_disk(self, path: str) -> None:
        """Load + auto-migrate from legacy JSON if present."""
        await self._ensure_loaded()
        legacy = Path(path)
        current = await asyncio.to_thread(self._collection.count)
        if current == 0 and legacy.exists() and legacy.suffix == ".json":
            await self._migrate_from_json(legacy)

    async def _migrate_from_json(self, json_path: Path) -> None:
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(data, list) or not data:
            return

        print(f"[chroma-episodic] migrating {len(data)} episodes from {json_path}")
        count = 0
        for item in data:
            try:
                ep = EpisodeEntry.model_validate(item)
                await self.store_episode(ep)
                count += 1
            except Exception:
                continue
        try:
            backup = json_path.with_suffix(".json.migrated")
            json_path.rename(backup)
            print(f"[chroma-episodic] migrated {count} → {backup}")
        except OSError:
            pass
