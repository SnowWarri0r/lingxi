"""Silence-to-emotion mapping (#3 time weight).

When the user breaks a long silence, Aria should land on a slightly-
shifted emotional state — not the same baseline she was at last turn.
The label "距离上次对话: 3天" by itself is dead text; the time-gap
needs to actually move her internal state to feel real.

Implementation: at turn start, compute the silence duration since last
interaction, map to a delta dict, apply via EmotionState.apply_deltas
(volatility-blended one-shot). Decay handles fade across subsequent
turns within the same conversation, so this only kicks in at the
"first message after silence" moment.

Conservative buckets — short gaps don't register, only multi-hour
silences shift state.
"""

from __future__ import annotations

from datetime import timedelta


def compute_silence_emotion_deltas(silence: timedelta) -> dict[str, float]:
    """Map a silence duration to emotion-state deltas at reconnect.

    Buckets reflect Aria's contemplative / 富有同理心 character — silence
    leans her toward 想念/期待/惆怅 rather than anger/resentment. A
    different persona's table would tilt differently.

    Returns an empty dict when no shift is warranted (still in active
    chat flow), so the caller can skip apply_deltas entirely.
    """
    # Active flow — no effect
    if silence < timedelta(hours=2):
        return {}

    # Half a day: slight anticipation forming
    if silence < timedelta(hours=12):
        return {"期待": 0.2}

    # Half-day to two days: started thinking about user
    if silence < timedelta(days=2):
        return {"想念": 0.3, "期待": 0.2}

    # 2-7 days: distance settling in alongside the missing
    if silence < timedelta(days=7):
        return {"想念": 0.5, "失落": 0.2}

    # > 7 days: heavy — distance + mild hurt + still hopeful
    return {"想念": 0.6, "孤独": 0.3, "失落": 0.3}
