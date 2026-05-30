"""Tests for social.loader."""

from pathlib import Path

import pytest

from lingxi.social.loader import load_social_graph, seed_initial_arcs
from lingxi.social.store import SocialStore


# The shipped config currently ships ZERO NPCs (removed 2026-05-28 — the
# old roster contradicted the freelance-astronomer persona). The full
# roster is preserved in the .bak file so the loader/seed MECHANICS still
# have something to exercise and NPCs can be restored later.
ROSTER_YAML = "config/personas/aria_social_graph.yaml.bak"


class TestLoadSocialGraph:
    def test_shipped_yaml_is_empty(self):
        # By design: NPCs disabled in the shipped config.
        graph = load_social_graph("config/personas/aria_social_graph.yaml")
        assert graph.npcs == []

    def test_load_roster_yaml(self):
        graph = load_social_graph(ROSTER_YAML)
        assert len(graph.npcs) == 6
        ids = {n.id for n in graph.npcs}
        assert ids == {"xiaomin", "prof_zhao", "lin_jie", "echo", "mom", "tom"}

    def test_every_npc_has_initial_arc(self):
        graph = load_social_graph(ROSTER_YAML)
        for npc in graph.npcs:
            assert len(npc.initial_arcs) >= 1, f"{npc.id} has no initial arcs"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_social_graph(tmp_path / "missing.yaml")


class TestSeedInitialArcs:
    @pytest.mark.asyncio
    async def test_seed_writes_arcs_first_time(self, tmp_path):
        graph = load_social_graph(ROSTER_YAML)
        store = SocialStore(tmp_path)
        seeded = await seed_initial_arcs(graph, store)
        assert seeded == 6
        # Verify the file actually got written
        state = await store.load_state("xiaomin")
        assert len(state.arcs) == 1
        assert state.arcs[0].id == "thesis_pressure"

    @pytest.mark.asyncio
    async def test_seed_idempotent(self, tmp_path):
        graph = load_social_graph(ROSTER_YAML)
        store = SocialStore(tmp_path)
        first = await seed_initial_arcs(graph, store)
        second = await seed_initial_arcs(graph, store)
        assert first == 6
        assert second == 0  # already seeded, nothing to do
