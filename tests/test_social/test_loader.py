"""Tests for social.loader."""

from pathlib import Path

import pytest

from lingxi.social.loader import load_social_graph, seed_initial_arcs
from lingxi.social.store import SocialStore


class TestLoadSocialGraph:
    def test_load_real_aria_yaml(self):
        path = Path("config/personas/aria_social_graph.yaml")
        graph = load_social_graph(path)
        assert len(graph.npcs) == 6
        ids = {n.id for n in graph.npcs}
        assert ids == {"xiaomin", "prof_zhao", "lin_jie", "echo", "mom", "tom"}

    def test_every_npc_has_initial_arc(self):
        graph = load_social_graph("config/personas/aria_social_graph.yaml")
        for npc in graph.npcs:
            assert len(npc.initial_arcs) >= 1, f"{npc.id} has no initial arcs"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_social_graph(tmp_path / "missing.yaml")


class TestSeedInitialArcs:
    @pytest.mark.asyncio
    async def test_seed_writes_arcs_first_time(self, tmp_path):
        graph = load_social_graph("config/personas/aria_social_graph.yaml")
        store = SocialStore(tmp_path)
        seeded = await seed_initial_arcs(graph, store)
        assert seeded == 6
        # Verify the file actually got written
        state = await store.load_state("xiaomin")
        assert len(state.arcs) == 1
        assert state.arcs[0].id == "thesis_pressure"

    @pytest.mark.asyncio
    async def test_seed_idempotent(self, tmp_path):
        graph = load_social_graph("config/personas/aria_social_graph.yaml")
        store = SocialStore(tmp_path)
        first = await seed_initial_arcs(graph, store)
        second = await seed_initial_arcs(graph, store)
        assert first == 6
        assert second == 0  # already seeded, nothing to do
