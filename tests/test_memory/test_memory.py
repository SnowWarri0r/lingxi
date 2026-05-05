"""Tests for the memory system."""

import pytest

from lingxi.memory.base import MemoryEntry, MemoryType
from lingxi.memory.short_term import ShortTermMemory
from lingxi.memory.long_term import LongTermMemory
from lingxi.memory.episodic import EpisodicMemory
from lingxi.memory.base import EpisodeEntry


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


class TestShortTermCrossRecipientIsolation:
    """Regression: snapshot_for_recipient must NOT switch active recipient.

    Scenario: reactive chat for user A is in-flight (buffer active = A,
    A's user msg already added, awaiting LLM). Concurrently, proactive
    scheduler reads B's history. After proactive returns, reactive
    completes and adds A's assistant reply. The reply must land in A's
    file, not B's.
    """

    @pytest.mark.asyncio
    async def test_snapshot_does_not_swap_active_recipient(self, tmp_path):
        stm = ShortTermMemory(max_turns=10, data_dir=tmp_path)

        # Reactive: switch to A, add user turn
        await stm.switch_recipient("A")
        stm.add_turn("user", "你今天吃啥")
        await stm.persist_current()

        # Pre-seed B's file with one turn so it exists on disk
        await stm.switch_recipient("B")
        stm.add_turn("user", "B 之前说过的话")
        await stm.persist_current()

        # Reactive resumes: switch back to A (simulating mid-turn await)
        await stm.switch_recipient("A")
        assert stm._current_recipient == "A"

        # Proactive does a snapshot read of B WITHOUT switching active
        b_turns = await stm.snapshot_for_recipient("B")
        assert len(b_turns) == 1
        assert b_turns[0].content == "B 之前说过的话"

        # Critical: active recipient must still be A
        assert stm._current_recipient == "A", (
            "snapshot_for_recipient leaked B into active state — "
            "next add_turn would land on B"
        )

        # Reactive completes: add A's assistant reply
        stm.add_turn("assistant", "我也不知道吃啥")
        await stm.persist_current()

        # Verify isolation: A has both turns, B still has only its one
        a_after = await stm.snapshot_for_recipient("A")
        b_after = await stm.snapshot_for_recipient("B")
        a_contents = [t.content for t in a_after]
        b_contents = [t.content for t in b_after]
        assert "我也不知道吃啥" in a_contents
        assert "我也不知道吃啥" not in b_contents
        assert b_contents == ["B 之前说过的话"]

    @pytest.mark.asyncio
    async def test_apply_summaries_preserves_appended_turns(self, tmp_path):
        """Race regression: a turn appended to the file between snapshot
        and write_for_recipient must NOT be erased.

        Models the real scenario: compress_aged_turns_for snapshots, awaits
        an LLM, and writes back. Meanwhile, a reactive turn appends a new
        entry to the file. The atomic apply_summaries_atomic must merge
        summaries into the latest file without clobbering the new turn.
        """
        from datetime import datetime, timedelta

        stm = ShortTermMemory(max_turns=20, data_dir=tmp_path)

        # Seed file with two old turns (no summaries yet)
        from lingxi.memory.short_term import ConversationTurn
        old_t1 = ConversationTurn(
            role="user", content="老话题 A",
            timestamp=datetime.now() - timedelta(hours=1),
        )
        old_t2 = ConversationTurn(
            role="assistant", content="老话题 A 的回应",
            timestamp=datetime.now() - timedelta(hours=1, seconds=-10),
        )
        await stm.write_for_recipient("X", [old_t1, old_t2])

        # Simulate compaction snapshot (this is what compress_aged_turns_for does)
        snapshot = await stm.snapshot_for_recipient("X")
        assert len(snapshot) == 2

        # While "LLM is awaiting" — a reactive turn writes a NEW turn to
        # the same recipient's file by going through the active path.
        from lingxi.memory.short_term import ConversationTurn as CT
        new_turn = CT(role="user", content="刚发的新消息", timestamp=datetime.now())
        appended = list(snapshot) + [new_turn]
        await stm.write_for_recipient("X", appended)

        # Compaction "finishes": apply summaries by identity. Should
        # patch summaries on old_t1 / old_t2 in the LATEST file (with the
        # new turn) and NOT clobber the new turn.
        summary_map = {
            (old_t1.timestamp.isoformat(), old_t1.role, old_t1.content[:60]): "老话题 A 摘要",
            (old_t2.timestamp.isoformat(), old_t2.role, old_t2.content[:60]): "老话题 A 回应摘要",
        }
        merged = await stm.apply_summaries_atomic("X", summary_map)
        assert merged == 2

        # Final state: 3 turns, 2 with summaries, new turn intact
        final = await stm.snapshot_for_recipient("X")
        assert len(final) == 3, f"new turn was erased! got {[t.content for t in final]}"
        contents = [t.content for t in final]
        assert "刚发的新消息" in contents
        # Old turns now have summaries
        old_turns = [t for t in final if t.summary is not None]
        assert len(old_turns) == 2

    @pytest.mark.asyncio
    async def test_write_for_recipient_does_not_swap(self, tmp_path):
        stm = ShortTermMemory(max_turns=10, data_dir=tmp_path)

        # Active = A
        await stm.switch_recipient("A")
        stm.add_turn("user", "hi from A")

        # Build a turns list for B and persist via the explicit API
        from lingxi.memory.short_term import ConversationTurn
        b_turns = [
            ConversationTurn(role="user", content="hi from B"),
            ConversationTurn(role="assistant", content="hello B"),
        ]
        await stm.write_for_recipient("B", b_turns)

        # Active recipient must remain A; A's buffer must still hold its turn
        assert stm._current_recipient == "A"
        assert any(t.content == "hi from A" for t in stm.get_history())

        # B's file should now have its 2 turns
        b_loaded = await stm.snapshot_for_recipient("B")
        assert [t.content for t in b_loaded] == ["hi from B", "hello B"]
