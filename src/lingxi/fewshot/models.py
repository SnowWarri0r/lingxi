from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal


class FewShotSample(BaseModel):
    id: str
    inner_thought: str
    original_speech: str | None = None
    corrected_speech: str
    context_summary: str
    tags: list[str] = Field(default_factory=list)
    recipient_key: str | None = None
    source: Literal["seed", "user_correction", "positive"] = "seed"
    created_at: datetime = Field(default_factory=datetime.now)
