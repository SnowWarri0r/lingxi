from lingxi.facts.models import Source
from lingxi.facts.writers.base import WriterBase


class LifeWriter(WriterBase):
    """Facts about Aria (subject=aria).

    Primary source is LIFE_SIMULATED (life simulator ticks).
    NPC_TICKER is also allowed: when an NPC event is significant enough
    to promote, SocialPromoter writes it as an Aria-side observation
    (subject=aria, source=NPC_TICKER) via this writer.
    """
    ALLOWED_SOURCE = Source.LIFE_SIMULATED  # default for keyword-only write()
    ALLOWED_SOURCES = frozenset({Source.LIFE_SIMULATED, Source.NPC_TICKER})
    SUBJECT_PATTERN = r"^aria$"
