"""Pydantic models for the few-shot pool and annotation system."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

FewShotSource = Literal["seed", "user_correction", "positive"]
AnnotationKind = Literal["none", "positive", "negative"]


class FewShotSample(BaseModel):
    """One retrieval sample: an inner-thought / speech pair.

    - seed: hand-written baseline, recipient_key typically None (global)
    - user_correction: user-supplied fix after marking a turn 'not like me'
    - positive: confirmed-good turn (thumbs-up)
    """

    id: str
    inner_thought: str
    original_speech: str | None = None
    corrected_speech: str
    context_summary: str
    tags: list[str] = Field(default_factory=list)
    recipient_key: str | None = None
    source: FewShotSource = "seed"
    created_at: datetime = Field(default_factory=datetime.now)


class AnnotationTurn(BaseModel):
    """Per-turn state kept so the user can annotate after the fact."""

    turn_id: str
    recipient_key: str
    user_message: str
    inner_thought: str
    speech: str
    created_at: datetime = Field(default_factory=datetime.now)
    annotation: AnnotationKind = "none"
    correction: str | None = None
