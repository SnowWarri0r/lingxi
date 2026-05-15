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


def detect_last_assistant_turn(
    history: list[ConversationTurn],
) -> tuple[str, bool] | None:
    """Return (content, is_question) for the most recent assistant turn.

    Skips trailing user turns (the current message we're about to reply to)
    to find the assistant utterance the user is most likely responding to.
    Used to ALWAYS surface "what you just said" — not just questions —
    so short / ambiguous user replies ("给我吃" / "我也是" / "懂") get
    read in the context of Aria's last utterance, not as cold openers.

    Returns None when:
    - history empty
    - no assistant turn found
    - assistant turn has empty content
    """
    if not history:
        return None

    last_assistant: ConversationTurn | None = None
    for turn in reversed(history):
        if turn.role == "user":
            continue
        if turn.role == "assistant":
            last_assistant = turn
            break
        return None

    if last_assistant is None:
        return None

    content = (last_assistant.content or "").strip()
    if not content:
        return None

    # Multi-bubble: pick the FIRST substantial bubble (usually the topic
    # anchor; later bubbles are continuations / acknowledgments)
    bubbles = [b.strip() for b in re.split(r"\n\s*\n", content) if b.strip()]
    chosen = bubbles[0] if bubbles else content
    is_question = _looks_like_question(chosen)
    return (chosen[:200], is_question)


# Aria-directed accusations / challenges. Each entry is a substring that
# strongly indicates the user is calling Aria out (cold reply / not
# caring / says something hurtful). Conservative: better to miss a
# borderline case than fluster on every venting message.
_CONFRONTATION_PATTERNS: tuple[str, ...] = (
    # Style criticism (almost always Aria-targeted)
    "好敷衍", "敷衍", "好冷漠", "冷漠",
    # 你-prefix accusations
    "你都不", "你怎么不", "你为什么不", "你这样", "你这个人",
    "你竟然", "你居然", "你也不",
    # Relationship / care questioning
    "什么关系", "我算什么",
    "不在乎我", "不安慰", "都不在乎",
    # "I'm hurt" markers
    "失望", "伤心了", "我生气了",
)


def detect_confrontation(user_text: str) -> bool:
    """True if user_text contains Aria-directed accusation/challenge signals.

    These trigger a fluster bump on Aria's emotion (慌乱/心虚) so the
    next reply lands destabilized instead of composed. Real humans
    scramble when caught; the lack of scrambling is itself an AI tell
    (the user's exact diagnosis: "感觉没有手忙脚乱也没有极力解释").
    """
    if not user_text:
        return False
    text = user_text.strip()
    return any(p in text for p in _CONFRONTATION_PATTERNS)
