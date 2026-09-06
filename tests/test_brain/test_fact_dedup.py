"""Not recording the same fact twice."""

import pytest

from lingxi.brain.orchestrator import build_orchestrator_prompt, StateDigest


def _prompt(known=None):
    return build_orchestrator_prompt(
        "在吗", StateDigest(activity="", mood="", last_lived=[]), {},
        known_facts=known)


def test_known_facts_are_shown_not_just_counted():
    """The catalog carries counts only, so 'don't write it twice' was an
    instruction the model had no way to follow."""
    out = _prompt(["对方说下个月会去邻市的漫展见阿澪"])
    assert "对方说下个月会去邻市的漫展见阿澪" in out
    assert "已经记住的关于对方的事" in out


def test_no_known_facts_says_so_rather_than_leaving_a_hole():
    out = _prompt([])
    assert "还没记过什么" in out


def test_the_rule_names_the_rename_case():
    """阿澪 and Mio are one spelling apart; the duplicate that started this was
    exactly that substitution."""
    out = _prompt(["x"])
    assert "不同叫法也算同一件事" in out


def test_it_asks_for_a_merged_fact_not_an_increment():
    out = _prompt(["x"])
    assert "把新旧合成完整的" in out


class _Emb:
    """Deterministic stand-in: identical text is identical, else orthogonal-ish."""

    def __init__(self, table):
        self._t = table

    async def embed(self, text):
        return self._t.get(text, [0.0, 0.0, 1.0])


@pytest.mark.asyncio
async def test_a_reworded_duplicate_is_folded_in_keeping_the_fuller_one(tmp_path):
    from datetime import datetime

    from lingxi.conversation.engine import ConversationEngine
    from lingxi.facts.models import Fact, FactType, Source
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.store import FactStore
    from lingxi.facts.writers.user_statement import UserStatementWriter
    from lingxi.memory.manager import MemoryManager
    from lingxi.persona.models import Identity, PersonaConfig

    store = FactStore(tmp_path / "f.db")
    await store.init()
    SHORT = "对方说下个月会去邻市的漫展见阿澪"
    LONG = "对方说下个月会去邻市的漫展见 Mio，还要递手写信"
    await store.write(Fact(subject="user:feishu:x", content=SHORT,
                           source=Source.USER_STATED, type=FactType.PATTERN,
                           ts=datetime.now(), importance=5))

    eng = ConversationEngine(
        persona=PersonaConfig(name="A", identity=Identity(full_name="A")),
        llm_provider=object(),
        memory_manager=MemoryManager(data_dir=str(tmp_path / "m")),
        fact_retriever=FactRetriever(store),
        user_statement_writer=UserStatementWriter(store),
    )
    eng._current_recipient_key = "feishu:x"
    # Same vector for both wordings — the rename case.
    eng.memory.embedding_provider = _Emb({SHORT: [1.0, 0.0, 0.0],
                                          LONG: [1.0, 0.0, 0.0]})

    await eng._write_one_user_fact("user:feishu:x", LONG)

    live = await store.query(subject="user:feishu:x", type=FactType.PATTERN, limit=10)
    contents = [f.content for f in live]
    assert LONG in contents, "the fuller wording must survive"
    assert SHORT not in contents, "the shorter duplicate should be superseded"


@pytest.mark.asyncio
async def test_a_genuinely_new_fact_is_still_written(tmp_path):
    from datetime import datetime

    from lingxi.conversation.engine import ConversationEngine
    from lingxi.facts.models import Fact, FactType, Source
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.store import FactStore
    from lingxi.facts.writers.user_statement import UserStatementWriter
    from lingxi.memory.manager import MemoryManager
    from lingxi.persona.models import Identity, PersonaConfig

    store = FactStore(tmp_path / "f.db")
    await store.init()
    OLD, NEW = "对方本周六要去邻市", "对方本周日从邻市返回"
    await store.write(Fact(subject="user:feishu:x", content=OLD,
                           source=Source.USER_STATED, type=FactType.PATTERN,
                           ts=datetime.now(), importance=5))
    eng = ConversationEngine(
        persona=PersonaConfig(name="A", identity=Identity(full_name="A")),
        llm_provider=object(),
        memory_manager=MemoryManager(data_dir=str(tmp_path / "m")),
        fact_retriever=FactRetriever(store),
        user_statement_writer=UserStatementWriter(store),
    )
    eng._current_recipient_key = "feishu:x"
    eng.memory.embedding_provider = _Emb({OLD: [1.0, 0.0, 0.0],
                                          NEW: [0.0, 1.0, 0.0]})

    await eng._write_one_user_fact("user:feishu:x", NEW)

    contents = [f.content for f in await store.query(
        subject="user:feishu:x", type=FactType.PATTERN, limit=10)]
    assert OLD in contents and NEW in contents, "different facts both survive"


@pytest.mark.asyncio
async def test_without_an_embedder_the_write_still_happens(tmp_path):
    """Losing a fact is worse than keeping a duplicate, so the check fails open."""
    from lingxi.conversation.engine import ConversationEngine
    from lingxi.facts.models import FactType
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.store import FactStore
    from lingxi.facts.writers.user_statement import UserStatementWriter
    from lingxi.memory.manager import MemoryManager
    from lingxi.persona.models import Identity, PersonaConfig

    store = FactStore(tmp_path / "f.db")
    await store.init()
    eng = ConversationEngine(
        persona=PersonaConfig(name="A", identity=Identity(full_name="A")),
        llm_provider=object(),
        memory_manager=MemoryManager(data_dir=str(tmp_path / "m")),
        fact_retriever=FactRetriever(store),
        user_statement_writer=UserStatementWriter(store),
    )
    eng._current_recipient_key = "feishu:x"
    eng.memory.embedding_provider = None

    await eng._write_one_user_fact("user:feishu:x", "对方养了一只猫")
    contents = [f.content for f in await store.query(
        subject="user:feishu:x", type=FactType.PATTERN, limit=10)]
    assert "对方养了一只猫" in contents
