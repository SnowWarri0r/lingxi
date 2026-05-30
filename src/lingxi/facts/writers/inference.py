from lingxi.facts.models import Source
from lingxi.facts.writers.base import WriterBase


class InferenceWriter(WriterBase):
    """LLM_INFERRED facts from reflection cycle.

    Can write about Aria (her own patterns) or a user (inferred about them).
    Lower default confidence (0.5) reflects the uncertainty.
    """
    ALLOWED_SOURCE = Source.LLM_INFERRED
    SUBJECT_PATTERN = r"^(aria|user:[A-Za-z0-9_:-]+)$"
