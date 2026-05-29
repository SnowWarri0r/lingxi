"""Engagement mode — the per-turn "agency" lever.

Moved here from inner_life (which was retired). Engagement is derived
purely from Aria's current EMOTION; it no longer depends on a numeric
inner-state energy scalar (that scalar was dropped when the inner_life
simulator was decommissioned in favour of facts-driven life state).

Existence of this enum is the "agency" lever — without it, Aria is
structurally helpful (every turn must be a good response). With it,
"短一句就停" / "只回个嗯" / "不接对方的话题" are first-class legal
outputs in their own modes, not embarrassing fallbacks.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lingxi.persona.models import EmotionState


class EngagementMode(str, Enum):
    """How present Aria is in this turn.

    full       — default. Engages normally per persona/state.
    curt       — mild irritation. Replies short, no follow-up question,
                 no warmth markers. "嗯" or one-line is fine.
    withdrawn  — heavy emotion (悲伤/孤独/压抑) ≥ 0.5. Sentence-level
                 silence is allowed; one-emoji or "嗯..." is enough.
                 Don't echo user's topic, don't ask follow-up.
    flustered  — just got called out / accused / said something cold the
                 user pushed back on. HIGH engagement, HIGH stress —
                 sentence fragments, repeated chars, over-explanation,
                 self-criticism. The "she was caught and now she's
                 scrambling" register that real humans land in.
    """

    FULL = "full"
    CURT = "curt"
    WITHDRAWN = "withdrawn"
    FLUSTERED = "flustered"


def derive_engagement_mode(emotion: "EmotionState | None") -> EngagementMode:
    """Pick an engagement mode from current emotion.

    Per-turn — represents NOW, not a sticky state machine. Recomputed
    every prompt build so a heavy-emotion turn naturally retreats and a
    recovered turn springs back. Order of checks matters: FLUSTERED
    overrides everything (the acute "just got caught" dominates output
    style); HEAVY (withdraw) next; PROVOKED (curt) last.
    """
    if emotion is None:
        return EngagementMode.FULL

    # Local import to avoid a runtime cycle.
    from lingxi.persona.models import EmotionState

    # Flustered FIRST — the most acute reaction.
    flustered = max(
        (v for n, v in emotion.dimensions.items() if n in EmotionState.FLUSTERED_DIMS),
        default=0.0,
    )
    if flustered >= 0.4:
        return EngagementMode.FLUSTERED

    # Heavy — withdraw (inward, less talking — not outward snapping).
    heavy = max(
        (v for n, v in emotion.dimensions.items() if n in EmotionState.HEAVY_DIMS),
        default=0.0,
    )
    if heavy >= 0.5:
        return EngagementMode.WITHDRAWN

    provoked = max(
        (v for n, v in emotion.dimensions.items() if n in EmotionState.PROVOKED_DIMS),
        default=0.0,
    )
    if provoked >= 0.4:
        return EngagementMode.CURT

    return EngagementMode.FULL
