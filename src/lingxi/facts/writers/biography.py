from lingxi.facts.models import Source
from lingxi.facts.writers.base import WriterBase


class BiographyLoader(WriterBase):
    """BIOGRAPHY one-shot import of Aria's backstory (subject=aria, ts=past)."""
    ALLOWED_SOURCE = Source.BIOGRAPHY
    SUBJECT_PATTERN = r"^aria$"
