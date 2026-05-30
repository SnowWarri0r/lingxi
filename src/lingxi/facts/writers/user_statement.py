from lingxi.facts.models import Source
from lingxi.facts.writers.base import WriterBase


class UserStatementWriter(WriterBase):
    """USER_STATED facts captured from chat (subject=user:<key>)."""
    ALLOWED_SOURCE = Source.USER_STATED
    SUBJECT_PATTERN = r"^user:[A-Za-z0-9_:-]+$"
