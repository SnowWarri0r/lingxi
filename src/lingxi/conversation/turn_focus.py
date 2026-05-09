"""Detect the most recent assistant question in conversation history.

Used by engine.chat_full + temporal.proactive to feed
PromptBuilder.build_turn_focus_reminder, which assembles all dynamic
per-turn state (time, current activity, recent events, emotion, mode,
news, recent proactive messages, last question) into a `<system-
reminder>` user message embedded right before the user's actual reply.

Pattern lifted from Claude Code (src/utils/api.ts:449 prependUserContext
+ attachments.ts surfacer). System prompt holds stable persona + rules;
this channel holds turn-relevant dynamic state.
"""

from __future__ import annotations

import re

from lingxi.memory.short_term import ConversationTurn


# A trailing question signal — character set covers Chinese + English
# question forms. Heuristic: scan the LAST clause of each bubble for these.
_QUESTION_CHARS = ("?", "？", "吗", "呢")
_INTERROGATIVES = ("怎么", "几点", "哪里", "什么", "为什么", "啥", "哪")


def _looks_like_question(text: str) -> bool:
    """True if this clause appears to be a question.

    Strict ending check first (most reliable), then a short-tail interrogative
    fallback for cases like "你今天吃了啥" where 啥 isn't at the very end
    after punctuation/emoji.
    """
    text = text.strip().rstrip("。…!！。 ")
    if not text:
        return False
    if text.endswith(_QUESTION_CHARS):
        return True
    tail = text[-12:]
    return any(w in tail for w in _INTERROGATIVES)


def detect_last_assistant_question(
    history: list[ConversationTurn],
) -> str | None:
    """Find the most recent assistant question in conversation history.

    Skips trailing user turns (the new message we're about to respond to),
    grabs the last assistant turn, splits on \\n\\n bubble boundaries, and
    returns the FIRST bubble that looks like a question (most likely the
    main one — questions tend to come early).

    Returns None when:
    - History is empty
    - No assistant turn exists
    - The most recent assistant turn doesn't contain a question
    """
    if not history:
        return None

    # Walk back, skip trailing user turns (current message + any retries)
    last_assistant: ConversationTurn | None = None
    for turn in reversed(history):
        if turn.role == "user":
            continue
        if turn.role == "assistant":
            last_assistant = turn
            break
        # ignore other roles
        return None

    if last_assistant is None:
        return None

    content = (last_assistant.content or "").strip()
    if not content:
        return None

    # Split on bubble boundaries; many of Aria's turns are multi-bubble.
    # The substantive question usually appears in the first non-trivial bubble.
    bubbles = [b.strip() for b in re.split(r"\n\s*\n", content) if b.strip()]
    for bubble in bubbles:
        if _looks_like_question(bubble):
            # Cap length so it doesn't bloat the reminder
            return bubble[:200]
    return None
