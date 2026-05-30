from lingxi.facts.models import Source
from lingxi.facts.writers.base import WriterBase


class LifeWriter(WriterBase):
    """LIFE_SIMULATED events about Aria (subject=aria)."""
    ALLOWED_SOURCE = Source.LIFE_SIMULATED
    SUBJECT_PATTERN = r"^aria$"
