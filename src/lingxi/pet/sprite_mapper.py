"""Map current inner state → sprite filename.

Single source of truth for "what does Aria look like right now". The pet
process polls /pet/state and feeds the response here; the FastAPI endpoint
also imports this so server-side decisions (e.g. logging which sprite is
"on") stay consistent.

Priority (most specific wins — engagement mode dominates because it's how
output style is gated everywhere else in the system):

  flustered > withdrawn > curt          (engagement_mode)
  > heavy > provoked                    (emotion family — negative)
  > happy > tired                       (emotion family — energy)
  > eating > focused > sleepy           (activity)
  > sleepy at night (23-06)             (hour fallback)
  > idle_default                        (final fallback)

`chatting` is not picked here — the window layer flips to it for ~5s
right after a proactive message is sent (using a chatting_until timestamp),
then falls back to whatever pick_sprite returns.
"""

from __future__ import annotations

from typing import Literal


SpriteName = Literal[
    "idle_default",
    "happy",
    "tired",
    "heavy",
    "flustered",
    "provoked",
    "focused",
    "sleepy",
    "eating",
    "chatting",
    "withdrawn",
    "curt",
]


# Allowed emotion family tags coming from the agent's EmotionState.
# Mirror of persona.models.EmotionState.*_DIMS family names.
EmotionFamily = Literal[
    "FLUSTERED", "HEAVY", "PROVOKED", "HIGH_ENERGY", "LOW_ENERGY", "NEUTRAL"
]


def pick_sprite(
    *,
    engagement_mode: str | None = None,
    emotion_family: str | None = None,
    activity_kind: str | None = None,
    hour: int | None = None,
) -> SpriteName:
    """Pick a sprite name from current state.

    All inputs are optional — missing fields just fall through to less
    specific buckets, ending at idle_default.

    Args:
        engagement_mode: one of "flustered" / "withdrawn" / "curt" / "full"
        emotion_family: one of EmotionFamily values (case-insensitive ok)
        activity_kind: one of ActivityKind enum values ("meal" / "work" / "sleep" / ...)
        hour: 0-23 local hour; used as a sleepy fallback for late hours
    """
    em = (engagement_mode or "").lower()
    if em == "flustered":
        return "flustered"
    if em == "withdrawn":
        return "withdrawn"
    if em == "curt":
        return "curt"

    fam = (emotion_family or "").upper()
    if fam == "HEAVY":
        return "heavy"
    if fam == "PROVOKED":
        return "provoked"
    if fam == "HIGH_ENERGY":
        return "happy"
    if fam == "LOW_ENERGY":
        return "tired"

    act = (activity_kind or "").lower()
    if act == "meal":
        return "eating"
    if act == "work":
        return "focused"
    if act == "sleep":
        return "sleepy"

    if hour is not None and (hour >= 23 or hour < 6):
        return "sleepy"

    return "idle_default"
