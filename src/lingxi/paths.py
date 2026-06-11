"""Per-persona data layout.

Each persona owns a namespace under data/personas/<slug>/ holding ALL its
state — facts.db, short_term/, proactive_history, fewshot/. Switching
PERSONA_PATH switches the whole memory with it; personas never share state.

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
