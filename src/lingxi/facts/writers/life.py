from lingxi.facts.models import Source
from lingxi.facts.writers.base import WriterBase


class LifeWriter(WriterBase):
    """Facts about Aria (subject=aria).

    Primary source is LIFE_SIMULATED (life simulator ticks). NPC_TICKER stays
    an allowed source for backward compatibility with historical facts, though
    the NPC/social subsystem that produced them has been decommissioned.
    """
    ALLOWED_SOURCE = Source.LIFE_SIMULATED  # default for keyword-only write()
    ALLOWED_SOURCES = frozenset({Source.LIFE_SIMULATED, Source.NPC_TICKER})
    SUBJECT_PATTERN = r"^aria$"
