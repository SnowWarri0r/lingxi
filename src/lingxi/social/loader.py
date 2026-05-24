"""Load social_graph.yaml + optionally seed initial arcs on first run."""

from __future__ import annotations

from pathlib import Path

import yaml

from lingxi.social.models import NPC, NPCArc, SocialGraph
from lingxi.social.store import SocialStore


def load_social_graph(path: str | Path) -> SocialGraph:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Social graph file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return SocialGraph.model_validate(data)


async def seed_initial_arcs(graph: SocialGraph, store: SocialStore) -> int:
    """For each NPC, if no arcs.json exists yet, write the initial_arcs.

    Idempotent — only writes when the file is missing. Run once at
    bootstrap. Returns number of NPCs seeded.
    """
    seeded = 0
    for npc in graph.npcs:
        path = store._arcs_path(npc.id)
        if path.exists():
            continue
        if not npc.initial_arcs:
            # Still create empty arcs.json so we don't re-attempt every boot
            await store.save_arcs(npc.id, [])
            continue
        await store.save_arcs(npc.id, list(npc.initial_arcs))
        seeded += 1
    return seeded
