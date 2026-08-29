"""Per-persona data layout.

Each persona owns a namespace under data/personas/<slug>/ holding ALL its
state — facts.db, short_term/, proactive_history, fewshot/. Switching
PERSONA_PATH switches the whole memory with it; personas never share state.

Stickers are the exception and live outside that namespace: they are
captioned reaction images with nothing persona-specific in them, and every
persona draws from the same pool.

MEMORY_DATA_DIR, if set, overrides the derived root (back-compat / tests).
"""

from __future__ import annotations

import os


def persona_data_root(persona) -> str:
    """The data directory for this persona's state."""
    override = os.environ.get("MEMORY_DATA_DIR")
    if override:
        return override
    return os.path.join("data", "personas", persona.slug)


def stickers_root() -> str:
    """The shared sticker pool, the same for every persona.

    Anchored absolutely rather than derived from a persona's data dir. It
    used to be written as `data_dir.parent / "stickers"`, which resolved to
    data/stickers/ back when data_dir was data/memory — then per-persona
    namespacing moved data_dir to data/personas/<slug>/ and the same
    expression started resolving to data/personas/stickers/. That directory
    was created empty on startup and every sticker lookup quietly returned
    nothing, so she simply stopped sending them.

    LINGXI_STICKERS_DIR overrides, for tests and for pointing a deployment at
    a different pool.
    """
    return os.environ.get("LINGXI_STICKERS_DIR") or os.path.join("data", "stickers")
