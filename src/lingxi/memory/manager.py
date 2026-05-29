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
    from lingxi.providers.base import LLMProvider
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
        self._llm_provider: LLMProvider | None = None

    def set_llm_provider(self, provider: LLMProvider) -> None:
        """Set the LLM provider used by the mid-term compression helpers."""
        self._llm_provider = provider

    async def compress_aged_turns(self, threshold_minutes: int = 30) -> int:
        """Mid-term layer: compress turns older than threshold into one-line summaries.

        Looks at the active recipient's short-term buffer for turns that are
        (a) older than `threshold_minutes` and (b) not yet summarized. Batches
        them through a single LLM call and stores the summary on each turn.
        """
        from datetime import datetime, timedelta
        llm = self._llm_provider
        if llm is None:
            return 0

        cutoff = datetime.now() - timedelta(minutes=threshold_minutes)
        turns = self.short_term.get_history()
        pending = [t for t in turns if t.summary is None and t.timestamp < cutoff]
        if not pending:
            return 0

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
        llm = self._llm_provider
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
            except Exception as e:
                print(f"[mid-term] persist failed for {recipient_key}: {e}")
        return count

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
