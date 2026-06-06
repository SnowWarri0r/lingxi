"""Schema for one sticker in the library."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class Sticker(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str
    source_url: str = ""
    content_hash: str
    caption: str = ""
    emotion: str = ""
    tags: list[str] = Field(default_factory=list)
    when_to_use: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
