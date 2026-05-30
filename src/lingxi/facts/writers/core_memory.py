from lingxi.facts.models import Source
from lingxi.facts.writers.base import WriterBase


class CoreMemoryWriter(WriterBase):
    """MemGPT core-memory blocks. subject is aria (persona block) or
    user:<recipient_key> (human block). Edits supersede the prior block."""
    ALLOWED_SOURCE = Source.LLM_INFERRED
    SUBJECT_PATTERN = r"^(aria|user:[A-Za-z0-9_:-]+)$"
