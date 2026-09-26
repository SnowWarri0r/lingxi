"""Working-memory manager: the short-term conversation buffer.

Trimmed down when the Chroma long-term/episodic/entity-graph stack was
retired in favour of facts.db as the single source of truth. What remains
is the WORKING memory — the rolling per-recipient conversation buffer
(short_term) plus the mid-term compression helpers that summarise aged
turns. Long-term facts now live in facts.db and reach the prompt via the
brain Orchestrator → Renderer path, not through this manager.

`assemble_context` therefore returns ONLY short-term turns; long_term_facts
and relevant_episodes stay empty (kept on MemoryContext purely so the
existing ContextAssembler signature is unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lingxi.providers.embedding import EmbeddingProvider

from lingxi.memory.short_term import ConversationTurn, ShortTermMemory


@dataclass
class MemoryContext:
    """Assembled memory context for prompt building (short-term only now)."""

    short_term_turns: list[ConversationTurn] = field(default_factory=list)
    long_term_facts: list = field(default_factory=list)      # always empty (facts.db owns this)
    relevant_episodes: list = field(default_factory=list)     # always empty


class MemoryManager:
    """Coordinator for the short-term working-memory buffer."""

    def __init__(
        self,
        data_dir: str = "./data/memory",
        max_short_term_turns: int = 30,
        retrieval_top_k: int = 10,
        **_ignored,
    ):
        # **_ignored swallows legacy kwargs (long_term_backend, embedding_dim,
        # max_long_term_entries, …) so existing callers don't break.
        self.data_dir = Path(data_dir)
        self.retrieval_top_k = retrieval_top_k
        self.short_term = ShortTermMemory(
            max_turns=max_short_term_turns,
            data_dir=self.data_dir,
        )
        self._embed_fn = None
        self.embedding_provider: EmbeddingProvider | None = None

    async def assemble_history_messages_for(
        self, recipient_key: str, assembler
    ) -> tuple[list, list[dict]]:
        """Read-only assembly: snapshot turns for `recipient_key` and run
        them through the given ContextAssembler, returning (turns, messages).

        Does NOT switch the singleton active recipient — safe for background
        callers (proactive) racing with reactive chat turns.
        """
        turns = await self.short_term.snapshot_for_recipient(recipient_key)
        mc = MemoryContext(short_term_turns=turns)
        messages = assembler.assemble_messages(mc)
        return turns, messages

    def set_embed_fn(self, embed_fn) -> None:
        """Set the embedding function (kept for biography retriever bootstrap)."""
        self._embed_fn = embed_fn

    def set_embedding_provider(self, provider: EmbeddingProvider | None) -> None:
        """Set a typed EmbeddingProvider.

        Exposed as self.embedding_provider so callers like the annotation
        pipeline and biography bootstrap can reuse it.
        """
        self.embedding_provider = provider
        self._embed_fn = provider.embed if provider is not None else None

    def add_turn(self, role: str, content: str, **metadata) -> ConversationTurn:
        """Add a conversation turn to short-term memory."""
        return self.short_term.add_turn(role, content, **metadata)

    async def assemble_context(
        self,
        query: str,
        short_term_limit: int | None = None,
        recipient_key: str | None = None,
        **_ignored,
    ) -> MemoryContext:
        """Assemble conversation context — short-term turns only.

        Long-term recall now flows through facts.db (Orchestrator → Renderer),
        so this returns just the recent dialog turns. **_ignored swallows the
        old retrieval kwargs (long_term_limit, episode_limit, context_aware…).
        """
        short_term_turns = self.short_term.get_history(last_n=short_term_limit)
        return MemoryContext(short_term_turns=short_term_turns)

    async def consolidate_session(self, recipient_key: str = "_global") -> dict:
        """No-op: session consolidation into Chroma long-term was retired.

        Facts are written turn-by-turn through the facts writers now, so there
        is no end-of-session batch consolidation step.
        """
        return {"facts_stored": 0, "episode_id": None}

    async def save(self) -> None:
        """Persist working memory. short_term auto-persists per recipient on
        each turn, so this only ensures the data dir exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            await self.short_term.persist_current()
        except Exception:
            pass

    async def load(self) -> None:
        """No-op: short_term loads lazily per recipient via switch_recipient()."""
        return None

    def get_stats(self) -> dict:
        """Working-memory statistics."""
        return {"short_term_turns": self.short_term.turn_count}
