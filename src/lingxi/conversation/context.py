"""Context assembly with priority-based token budget management.

Priority order (highest first):
1. Persona system prompt (never truncated)
2. Last few turns of conversation (always preserved)
3. Most relevant long-term memories
4. Most relevant episodic memories
5. Older conversation history (filled until budget exhausted)

Each section has its own budget; if total would exceed max_context_tokens,
sections are truncated from the bottom.
"""

from __future__ import annotations

from dataclasses import dataclass

from lingxi.memory.manager import MemoryContext


@dataclass
class TokenBudget:
    """Per-section token budgets. Total ≤ max_context_tokens."""

    max_context_tokens: int = 100000
    persona_budget: int = 4000
    recent_turns_min: int = 6  # at least last N turns always included
    recent_turns_budget: int = 6000
    history_budget: int = 8000
    memory_budget: int = 4000  # used by prompt builder, not here
    # === Layered memory windows (progressive forgetting) ===
    # L1 verbatim window: turns younger than this stay full-text.
    verbatim_window_minutes: int = 30
    # L2 mid-term window: turns 30min-X get compressed to one-line summaries.
    # Beyond this, drop entirely from messages (rely on L3 episodes / L4 facts).
    session_window_minutes: int = 720  # 12 hours


def estimate_tokens(text: str) -> int:
    """Rough token count: 1.5 tokens per Chinese char + 1 token per ~4 English chars."""
    if not text:
        return 0
    chinese = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other = len(text) - chinese
    return int(chinese * 1.5 + other * 0.3)


class ContextAssembler:
    """Assembles the message list with priority-based token budgets."""

    def __init__(
        self,
        budget: TokenBudget | None = None,
        max_context_tokens: int = 100000,
        history_token_budget: int = 8000,
    ):
        # Backward compat: accept old kwargs
        self.budget = budget or TokenBudget(
            max_context_tokens=max_context_tokens,
            history_budget=history_token_budget,
        )

    def assemble_messages(self, memory_context: MemoryContext) -> list[dict]:
        """Build messages list with layered memory.

        L1 (≤verbatim_window_minutes): full content
        L2 (verbatim..session_window): summary if available, else fall back to content
        beyond session_window: dropped entirely (L3 episodes/L4 facts handle these)
        """
        turns = memory_context.short_term_turns
        if not turns:
            return []

        from datetime import datetime, timedelta
        now = datetime.now()
        l2_cutoff = now - timedelta(minutes=self.budget.verbatim_window_minutes)
        session_cutoff = now - timedelta(minutes=self.budget.session_window_minutes)

        # ALWAYS preserve the last N turns regardless of age — losing them
        # silently breaks "user comes back next day and replies to what
        # Aria said yesterday" (the model sees only the new message and
        # has zero context for what's being referenced). The session-window
        # filter only applies to the older "fill" turns beyond this anchor.
        guaranteed_count = min(self.budget.recent_turns_min, len(turns))
        guaranteed = turns[-guaranteed_count:]
        guaranteed_tokens = sum(estimate_tokens(t.content) for t in guaranteed)

        # Older candidates are anything before the guaranteed window AND
        # within the session window — beyond session window we rely on
        # episode summaries (rendered in the system prompt's memory block).
        older_all = turns[:-guaranteed_count] if guaranteed_count < len(turns) else []
        in_session_older = [t for t in older_all if t.timestamp >= session_cutoff]
        dropped_by_session = len(older_all) - len(in_session_older)

        # Remaining budget for older turns (filled from newest backward)
        remaining_budget = self.budget.history_budget - guaranteed_tokens

        included_older: list = []
        used = 0
        # Walk from most-recent older turn backward to the oldest
        for turn in reversed(in_session_older):
            cost = estimate_tokens(turn.content)
            if used + cost > remaining_budget:
                break
            included_older.insert(0, turn)
            used += cost

        # Prepend a summary marker if we dropped turns (by session window or
        # by budget). Session-window drops are silent otherwise — model has
        # no signal that prior context exists.
        dropped_by_budget = len(in_session_older) - len(included_older)
        dropped = dropped_by_session + dropped_by_budget
        result_messages: list[dict] = []
        if dropped > 0:
            result_messages.append({
                "role": "user",
                "content": f"[省略了 {dropped} 轮较早的对话以节省上下文]",
            })

        for turn in included_older + guaranteed:
            # Always render verbatim content. Older third-person summary
            # rendering ("[HH:MM 摘要] 我询问对方身体...") was poisoning
            # the chat history — model read its own past as third-person
            # action narration ("我询问 / 我追问") and copied that as a
            # behavioral pattern, generating MORE asking-about-X turns.
            # Long-term coverage lives in episode summaries (rendered in
            # the system prompt's memory block), not in chat history.
            # turn.summary is kept on disk for analytics but never shown.
            result_messages.append({"role": turn.role, "content": turn.content})

        return result_messages

    def estimate_messages_tokens(self, messages: list[dict]) -> int:
        """Estimate total tokens in a message list."""
        return sum(estimate_tokens(m.get("content", "")) for m in messages)
