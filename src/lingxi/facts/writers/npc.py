from lingxi.facts.models import Source
from lingxi.facts.writers.base import WriterBase


class NPCWriter(WriterBase):
    """NPC_TICKER events about specific NPCs (subject=npc:<id>)."""
    ALLOWED_SOURCE = Source.NPC_TICKER
    SUBJECT_PATTERN = r"^npc:[A-Za-z0-9_-]+$"
