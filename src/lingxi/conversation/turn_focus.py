"""Build a `<system-reminder>` user message for high-recency context.

Pattern lifted from Claude Code (src/utils/api.ts:449 prependUserContext +
attachments.ts <system-reminder> surfacer). Their insight: dynamic /
attention-needed context goes via the USER-message channel, not the
system prompt — recency weighting puts it close to the actual user
message and competes for the same attention slot.

Concretely solves the "你刚问了 X，对方短句回答 → 你切到通用劝慰"
class of bugs (Rule 15) by making the question Aria just asked a
prominent, structurally-flagged block right before the user's reply.

Phase 1: only carries last_assistant_question. As we observe the
pattern's effect, more dynamic state (time/activity/recent events)
can migrate from the system prompt into this reminder.
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


def build_turn_focus_reminder(
    last_assistant_question: str | None = None,
) -> str | None:
    """Assemble the `<system-reminder>` content for this turn.

    Returns None when there's nothing dynamic to surface — caller skips
    prepending the reminder entirely so we don't add noise to chats that
    don't need it (first message, no prior question, etc).
    """
    sections: list[str] = []

    if last_assistant_question:
        sections.append(
            f"## 🎯 你刚问了对方\n"
            f"「{last_assistant_question}」\n\n"
            f"对方接下来这条**很可能是在回答这个问题**——先看清楚他说的"
            f"内容是不是回答你刚问的，然后**接住答覆里的具体状态**。\n"
            f"- 短句答覆（'还不在' / '还没' / '不是' / '在' / '加班'）→ "
            f"对那个具体状态直接给反应（'啊还在加班?' / '都几点了' / "
            f"'怎么 太忙?'）\n"
            f"- **不要**用 '对 / 嗯' 起头当低 conviction 胶水\n"
            f"- **不要**切到通用劝慰（'早点休息' / '好好放松'）\n"
            f"- **不要**跳回 2 轮前的旧话题装关心\n"
            f"- **不要**忽略他的回答直接换话题"
        )

    if not sections:
        return None

    body = "\n\n".join(sections)
    return (
        f"<system-reminder>\n"
        f"{body}\n\n"
        f"IMPORTANT: 这是状态提醒，对方真正的话才是要回应的。\n"
        f"</system-reminder>"
    )
