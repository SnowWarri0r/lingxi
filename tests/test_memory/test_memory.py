"""Tests for the memory system."""

import pytest

from persona_agent.memory.base import MemoryEntry, MemoryType
from persona_agent.memory.short_term import ShortTermMemory
from persona_agent.memory.long_term import LongTermMemory
from persona_agent.memory.episodic import EpisodicMemory
from persona_agent.memory.base import EpisodeEntry


class TestShortTermMemory:
    def test_add_and_retrieve(self):
        stm = ShortTermMemory(max_turns=5)
        stm.add_turn("user", "Hello")
        stm.add_turn("assistant", "Hi there!")

        history = stm.get_history()
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].content == "Hi there!"

    def test_sliding_window(self):
        stm = ShortTermMemory(max_turns=3)
        for i in range(5):
            stm.add_turn("user", f"Message {i}")

        history = stm.get_history()
        assert len(history) == 3
        assert history[0].content == "Message 2"

    def test_get_messages(self):
        stm = ShortTermMemory()
        stm.add_turn("user", "Hello")
        stm.add_turn("assistant", "Hi")

        messages = stm.get_messages()
        assert messages == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

    def test_clear(self):
        stm = ShortTermMemory()
        stm.add_turn("user", "Hello")
        cleared = stm.clear()
        assert len(cleared) == 1
        assert stm.is_empty


class TestLongTermMemory:
    @pytest.mark.asyncio
    async def test_store_and_retrieve_by_keyword(self):
        ltm = LongTermMemory()
        entry = MemoryEntry(
            content="User likes astronomy and stargazing",
            memory_type=MemoryType.LONG_TERM,
            importance=0.8,
        )
        await ltm.store(entry)

        results = await ltm.retrieve("astronomy", limit=5)
        assert len(results) == 1
        assert "astronomy" in results[0].content

    @pytest.mark.asyncio
    async def test_importance_affects_ranking(self):
        ltm = LongTermMemory()
        await ltm.store(MemoryEntry(
            content="topic alpha beta",
            memory_type=MemoryType.LONG_TERM,
            importance=0.3,
        ))
        await ltm.store(MemoryEntry(
            content="topic alpha gamma",
            memory_type=MemoryType.LONG_TERM,
            importance=0.9,
        ))

        results = await ltm.retrieve("topic alpha", limit=2)
        assert len(results) == 2
        # Higher importance should rank first
        assert results[0].importance > results[1].importance

    @pytest.mark.asyncio
    async def test_save_and_load(self, tmp_path):
        ltm = LongTermMemory()
        await ltm.store(MemoryEntry(
            content="Persistent fact",
            memory_type=MemoryType.LONG_TERM,
            importance=0.7,
        ))

        path = str(tmp_path / "ltm.json")
        await ltm.save_to_disk(path)

        ltm2 = LongTermMemory()
        await ltm2.load_from_disk(path)
        entries = await ltm2.list_all()
        assert len(entries) == 1
        assert entries[0].content == "Persistent fact"


class TestEpisodicMemory:
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        em = EpisodicMemory()
        episode = EpisodeEntry(
            summary="Discussed astronomy and favorite constellations",
            emotional_tone="warm",
            key_topics=["astronomy", "constellations"],
            turn_count=10,
        )
        await em.store_episode(episode)

        results = await em.retrieve_relevant("astronomy")
        assert len(results) == 1
        assert "astronomy" in results[0].summary

    @pytest.mark.asyncio
    async def test_recent_episodes(self):
        em = EpisodicMemory()
        for i in range(3):
            await em.store_episode(EpisodeEntry(
                summary=f"Episode {i}",
                turn_count=5,
            ))

        recent = await em.get_recent(2)
        assert len(recent) == 2
        assert recent[-1].summary == "Episode 2"
