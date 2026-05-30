from lingxi.facts.models import Source
from lingxi.facts.writers.base import WriterBase


class WorldWriter(WriterBase):
    """WORLD_FETCH events (subject=world)."""
    ALLOWED_SOURCE = Source.WORLD_FETCH
    SUBJECT_PATTERN = r"^world$"
