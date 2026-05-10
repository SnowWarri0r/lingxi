"""Memory manager: coordinates all memory stores and provides unified access."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lingxi.providers.base import LLMProvider
    from lingxi.providers.embedding import EmbeddingProvider

from lingxi.memory.base import EpisodeEntry, MemoryEntry, MemoryStore, MemoryType
from lingxi.memory.consolidation import MemoryConsolidator
from lingxi.memory.entity_graph import EntityExtractor, EntityGraph
from lingxi.memory.episodic import EpisodicMemory
from lingxi.memory.long_term import LongTermMemory
from lingxi.memory.short_term import ConversationTurn, ShortTermMemory


@dataclass
class MemoryContext:
    """Assembled memory context for prompt building."""

    short_term_turns: list[ConversationTurn] = field(default_factory=list)
    long_term_facts: list[MemoryEntry] = field(default_factory=list)
    relevant_episodes: list[EpisodeEntry] = field(default_factory=list)


class MemoryManager:
    """Central coordinator for all memory operations."""

    def __init__(
        self,
        data_dir: str = "./data/memory",
        max_short_term_turns: int = 30,
        max_long_term_entries: int = 10000,
        max_episodes: int = 500,
        retrieval_top_k: int = 10,
        importance_threshold: float = 0.3,
        long_term_backend: str = "chroma",  # "chroma" or "json"
        embedding_dim: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.retrieval_top_k = retrieval_top_k

        self.short_term = ShortTermMemory(
            max_turns=max_short_term_turns,
            data_dir=self.data_dir,
        )

        # Select long-term backend
        self.long_term: MemoryStore
        if long_term_backend == "chroma":
            from lingxi.memory.chroma_store import ChromaMemoryStore

            self.long_term = ChromaMemoryStore(
                db_path=self.data_dir / "chroma",
                max_entries=max_long_term_entries,
                importance_threshold=importance_threshold,
                embedding_dim=embedding_dim,
            )
        else:
            self.long_term = LongTermMemory(
                max_entries=max_long_term_entries,
                importance_threshold=importance_threshold,
            )

        # Episodic memory: Chroma if long-term is Chroma, else JSON
        if long_term_backend == "chroma":
            from lingxi.memory.chroma_episodic import ChromaEpisodicMemory

            self.episodic = ChromaEpisodicMemory(
                db_path=self.data_dir / "chroma",
                max_episodes=max_episodes,
                embedding_dim=embedding_dim,
            )
        else:
            self.episodic = EpisodicMemory(max_episodes=max_episodes)

        self._consolidator: MemoryConsolidator | None = None
        self._embed_fn = None
        self.embedding_provider: EmbeddingProvider | None = None

        # Entity graph (sidecar)
        self.entity_graph = EntityGraph(self.data_dir)
        self._entity_extractor: EntityExtractor | None = None

    def set_llm_provider(self, provider: LLMProvider) -> None:
        """Set the LLM provider for consolidation + entity extraction."""
        self._llm_provider = provider
        self._consolidator = MemoryConsolidator(
            long_term=self.long_term,
            episodic=self.episodic,
            llm_provider=provider,
        )
        self._entity_extractor = EntityExtractor(provider)

    async def compress_aged_turns(self, threshold_minutes: int = 30) -> int:
        """Mid-term layer: compress turns older than threshold into one-line summaries.

        Looks at the active recipient's short-term buffer for turns that are
        (a) older than `threshold_minutes` and (b) not yet summarized. Batches
        them through a single LLM call and stores the summary on each turn.

        Returns count of turns newly summarized.
        """
        from datetime import datetime, timedelta
        llm = getattr(self, "_llm_provider", None)
        if llm is None:
            return 0

        cutoff = datetime.now() - timedelta(minutes=threshold_minutes)
        turns = self.short_term.get_history()
        pending = [t for t in turns if t.summary is None and t.timestamp < cutoff]
        if not pending:
            return 0

        # Build batch prompt
        lines = []
        for i, t in enumerate(pending, 1):
            who = "对方" if t.role == "user" else "我"
            stamp = t.timestamp.strftime("%H:%M")
            content = (t.content or "").replace("\n", " ")[:300]
            lines.append(f"[{i}] {stamp} {who}: {content}")
        block = "\n".join(lines)
        prompt = (
            "下面是聊天记录里的若干 turn。请把每条压成 ≤25 字的中性摘要，"
            "保留具体话题和谁说的，去掉客套和冗余。"
            "返回 JSON 数组，元素格式 {\"i\": 序号, \"s\": \"摘要\"}：\n\n"
            + block
        )
        try:
            result = await llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3,
            )
        except Exception as e:
            print(f"[mid-term] compress LLM failed: {e}")
            return 0

        import json as _json
        import re as _re
        text = (result.content or "").strip()
        # Find JSON array
        m = _re.search(r"\[[\s\S]*\]", text)
        if not m:
            return 0
        try:
            data = _json.loads(m.group())
        except Exception:
            return 0

        count = 0
        for entry in data:
            try:
                idx = int(entry.get("i", 0)) - 1
                summary = str(entry.get("s", "")).strip()
                if 0 <= idx < len(pending) and summary:
                    pending[idx].summary = summary[:80]
                    count += 1
            except Exception:
                continue

        if count > 0:
            try:
                await self.short_term.persist_current()
            except Exception:
                pass
            print(f"[mid-term] compressed {count} aged turns")
        return count

    async def compress_aged_turns_for(
        self, recipient_key: str, threshold_minutes: int = 30
    ) -> int:
        """Recipient-scoped variant: compress aged turns for `recipient_key`
        without mutating `short_term._current_recipient`.

        Safe to call from a background scheduler while a different reactive
        turn is in-flight — the snapshot/write path bypasses the singleton
        active buffer.
        """
        from datetime import datetime, timedelta
        llm = getattr(self, "_llm_provider", None)
        if llm is None:
            return 0

        turns = await self.short_term.snapshot_for_recipient(recipient_key)
        if not turns:
            return 0

        cutoff = datetime.now() - timedelta(minutes=threshold_minutes)
        pending_idx = [i for i, t in enumerate(turns) if t.summary is None and t.timestamp < cutoff]
        if not pending_idx:
            return 0

        lines = []
        for k, idx in enumerate(pending_idx, 1):
            t = turns[idx]
            who = "对方" if t.role == "user" else "我"
            stamp = t.timestamp.strftime("%H:%M")
            content = (t.content or "").replace("\n", " ")[:300]
            lines.append(f"[{k}] {stamp} {who}: {content}")
        prompt = (
            "下面是聊天记录里的若干 turn。请把每条压成 ≤25 字的中性摘要，"
            "保留具体话题和谁说的，去掉客套和冗余。"
            "返回 JSON 数组，元素格式 {\"i\": 序号, \"s\": \"摘要\"}：\n\n"
            + "\n".join(lines)
        )
        try:
            result = await llm.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3,
            )
        except Exception as e:
            print(f"[mid-term] compress LLM failed for {recipient_key}: {e}")
            return 0

        import json as _json
        import re as _re
        text = (result.content or "").strip()
        m = _re.search(r"\[[\s\S]*\]", text)
        if not m:
            return 0
        try:
            data = _json.loads(m.group())
        except Exception:
            return 0

        count = 0
        for entry in data:
            try:
                k = int(entry.get("i", 0)) - 1
                summary = str(entry.get("s", "")).strip()
                if 0 <= k < len(pending_idx) and summary:
                    turns[pending_idx[k]].summary = summary[:80]
                    count += 1
            except Exception:
                continue

        if count > 0:
            try:
                # Build identity-keyed summary map and apply atomically.
                # apply_summaries_atomic re-reads the latest file inside a
                # lock and only patches turns whose summary is still None,
                # preserving any turns appended during our LLM await.
                summary_map = {}
                for orig in turns:
                    if not orig.summary:
                        continue
                    summary_map[(
                        orig.timestamp.isoformat(),
                        orig.role,
                        (orig.content or "")[:60],
                    )] = orig.summary
                merged = await self.short_term.apply_summaries_atomic(
                    recipient_key, summary_map
                )
                if merged > 0:
                    print(f"[mid-term] {recipient_key}: merged {merged}/{count} summaries")
                else:
                    print(
                        f"[mid-term] {recipient_key}: {count} summaries stale "
                        f"(file moved on), dropping rather than clobbering"
                    )
            except Exception as e:
                print(f"[mid-term] persist failed for {recipient_key}: {e}")
        return count

    async def assemble_history_messages_for(
        self, recipient_key: str, assembler
    ) -> tuple[list, list[dict]]:
        """Read-only assembly: snapshot turns for `recipient_key` and run
        them through the given ContextAssembler, returning (turns, messages).

        Does NOT switch the singleton active recipient — safe for background
        callers (proactive/reflection) racing with reactive chat turns.
        """
        turns = await self.short_term.snapshot_for_recipient(recipient_key)
        from lingxi.memory.manager import MemoryContext
        mc = MemoryContext(short_term_turns=turns)
        messages = assembler.assemble_messages(mc)
        return turns, messages

    def set_embed_fn(self, embed_fn) -> None:
        """Set the embedding function for semantic retrieval (raw callable)."""
        self._embed_fn = embed_fn

    def set_embedding_provider(self, provider: EmbeddingProvider | None) -> None:
        """Set a typed EmbeddingProvider for semantic retrieval.

        Exposes the provider as self.embedding_provider so callers like
        the annotation pipeline can reuse it without duplicating probe/init.

        NOTE: for Chroma backend, pass embedding_dim to MemoryManager
        constructor so the collection name is correct BEFORE load.
        """
        self.embedding_provider: EmbeddingProvider | None = provider
        if provider is None:
            self._embed_fn = None
            return
        self._embed_fn = provider.embed

    def add_turn(self, role: str, content: str, **metadata) -> ConversationTurn:
        """Add a conversation turn to short-term memory."""
        return self.short_term.add_turn(role, content, **metadata)

    async def add_fact(
        self,
        content: str,
        importance: float = 0.5,
        tags: list[str] | None = None,
        recipient_key: str | None = None,
        extract_entities: bool = True,
    ) -> str:
        """Directly add a fact to long-term memory."""
        meta = {"recipient_key": recipient_key or "_global"}
        entry = MemoryEntry(
            content=content,
            memory_type=MemoryType.LONG_TERM,
            importance=importance,
            tags=tags or [],
            metadata=meta,
        )
        if self._embed_fn:
            entry.embedding = await self._embed_fn(content)

        fact_id = await self.long_term.store(entry)

        # Extract and link entities (best-effort, async)
        if extract_entities and self._entity_extractor:
            try:
                await self.entity_graph.load()
                entities = await self._entity_extractor.extract(content)
                for e in entities:
                    self.entity_graph.link(e["name"], e["type"], fact_id)
                if entities:
                    await self.entity_graph.save()
            except Exception:
                pass

        return fact_id

    async def assemble_context(
        self,
        query: str,
        short_term_limit: int | None = None,
        long_term_limit: int | None = None,
        episode_limit: int = 3,
        context_aware: bool = True,
        context_turns: int = 4,
        recipient_key: str | None = None,
    ) -> MemoryContext:
        """Assemble a complete memory context for prompt building.

        Retrieves relevant memories from all tiers based on the query.

        Args:
            query: The user's current message.
            context_aware: If True, enrich the retrieval query with recent
                conversation context (e.g., pronouns like "it", "that" become
                meaningful when the last few turns are part of the query).
            context_turns: Number of recent turns to include in retrieval query.
        """
        # Short-term: always include recent turns
        short_term_turns = self.short_term.get_history(last_n=short_term_limit)

        # Build enriched retrieval query
        if context_aware and short_term_turns:
            # Take last N turns (excluding the very latest which is the current user message)
            recent = short_term_turns[-context_turns:] if len(short_term_turns) > 1 else []
            context_parts: list[str] = []
            for t in recent:
                snippet = t.content.strip()
                # Truncate very long turns
                if len(snippet) > 200:
                    snippet = snippet[:200]
                context_parts.append(snippet)
            # Put current query at the end (highest weight after BGE concatenation)
            context_parts.append(query)
            retrieval_query = " ".join(context_parts)
        else:
            retrieval_query = query

        # Get query embedding if available
        query_embedding = None
        if self._embed_fn and retrieval_query.strip():
            try:
                query_embedding = await self._embed_fn(retrieval_query)
            except Exception:
                query_embedding = None

        # Long-term: semantic/keyword retrieval (recipient-scoped if provided)
        long_term_limit = long_term_limit or self.retrieval_top_k
        long_term_kwargs = {
            "query": retrieval_query,
            "limit": long_term_limit,
            "query_embedding": query_embedding,
        }
        if recipient_key is not None:
            long_term_kwargs["recipient_key"] = recipient_key

        try:
            long_term_facts = await self.long_term.retrieve(**long_term_kwargs)
        except TypeError:
            # Fallback for stores that don't support recipient_key
            long_term_kwargs.pop("recipient_key", None)
            long_term_facts = await self.long_term.retrieve(**long_term_kwargs)

        # Episodic: relevant past conversations (recipient-scoped if provided)
        episodic_kwargs = {
            "query": retrieval_query,
            "limit": episode_limit,
            "query_embedding": query_embedding,
        }
        if recipient_key is not None:
            episodic_kwargs["recipient_key"] = recipient_key

        try:
            relevant_episodes = await self.episodic.retrieve_relevant(**episodic_kwargs)
        except TypeError:
            episodic_kwargs.pop("recipient_key", None)
            relevant_episodes = await self.episodic.retrieve_relevant(**episodic_kwargs)

        # Entity-boosted retrieval: add facts about entities mentioned in query
        try:
            await self.entity_graph.load()
            mentioned = self.entity_graph.find_in_text(retrieval_query)
            extra_fact_ids: set[str] = set()
            for ent in mentioned:
                extra_fact_ids.update(ent.fact_ids)
            # Remove already-included
            already = {f.id for f in long_term_facts}
            extra_fact_ids -= already

            # Fetch and append (limit to avoid bloat)
            for fid in list(extra_fact_ids)[:5]:
                try:
                    extra = await self.long_term.get_by_id(fid)
                    if extra:
                        # Skip if recipient mismatch
                        if recipient_key:
                            extra_recipient = (extra.metadata or {}).get(
                                "recipient_key", "_global"
                            )
                            if extra_recipient not in (recipient_key, "_global"):
                                continue
                        long_term_facts.append(extra)
                except Exception:
                    continue
        except Exception:
            pass

        return MemoryContext(
            short_term_turns=short_term_turns,
            long_term_facts=long_term_facts,
            relevant_episodes=relevant_episodes,
        )

    async def consolidate_session(self, recipient_key: str = "_global") -> dict:
        """Consolidate current short-term memory into long-term and episodic stores."""
        if not self._consolidator:
            return {"facts_stored": 0, "episode_id": None}

        turns = self.short_term.get_history()
        result = await self._consolidator.consolidate(
            turns, embed_fn=self._embed_fn, recipient_key=recipient_key
        )
        return result

    async def save(self) -> None:
        """Persist all memory stores to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        await self.long_term.save_to_disk(str(self.data_dir / "long_term.json"))
        await self.episodic.save_to_disk(str(self.data_dir / "episodic.json"))

    async def load(self) -> None:
        """Load all memory stores from disk."""
        await self.long_term.load_from_disk(str(self.data_dir / "long_term.json"))
        await self.episodic.load_from_disk(str(self.data_dir / "episodic.json"))

    def get_stats(self) -> dict:
        """Get memory system statistics (sync, uses cached counts)."""
        # Long-term count depends on backend
        from lingxi.memory.chroma_store import ChromaMemoryStore

        if isinstance(self.long_term, ChromaMemoryStore):
            # Chroma collection may not be loaded yet; return cached count
            try:
                count = self.long_term._collection.count() if self.long_term._collection else 0
            except Exception:
                count = 0
        else:
            count = len(self.long_term._entries)

        # Episodic count: property works for both JSON and Chroma
        ep_count = self.episodic.episode_count

        return {
            "short_term_turns": self.short_term.turn_count,
            "long_term_entries": count,
            "episodes": ep_count,
        }
