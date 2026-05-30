from datetime import datetime, timedelta

import pytest

from lingxi.brain.models import OrchestrationDecision, OrchestratorFactQuery
from lingxi.brain.renderer import render_dynamic_blocks
from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.retriever import FactRetriever
from lingxi.facts.store import FactStore
from lingxi.persona.loader import load_persona
from lingxi.persona.prompt_builder import build_persona_block


def test_persona_block_extracts_static_sections():
    persona = load_persona("config/personas/example_persona.yaml")
    block = build_persona_block(persona)
    # Persona name should appear (either via 'Aria' or persona.name)
    assert "Aria" in block or persona.name in block
    # The "how to talk" rules section should be in the format preamble
    assert "## 怎么说话" in block
    # The output format marker should appear
    assert "===META===" in block


@pytest.fixture
async def retriever(tmp_path):
    s = FactStore(tmp_path / "f.db")
    await s.init()
    now = datetime.now()
    # Aria lived
    await s.write(Fact(
        subject="aria", content="今早煮泡面",
        source=Source.LIFE_SIMULATED, type=FactType.EVENT, ts=now,
    ))
    # User pattern
    await s.write(Fact(
        subject="user:u1", content="工作 11-21",
        source=Source.USER_STATED, type=FactType.PATTERN, ts=now,
    ))
    # NPC event
    await s.write(Fact(
        subject="npc:xiaomin", content="小敏改 paper",
        source=Source.NPC_TICKER, type=FactType.EVENT, ts=now,
    ))
    return FactRetriever(s)


@pytest.mark.asyncio
async def test_renders_only_queried_facts(retriever):
    decision = OrchestrationDecision(
        engage_level=0.6, register="warm",
        fact_queries=[OrchestratorFactQuery(category="aria.event", limit=5)],
        topic_anchor="", skip=[],
    )
    out = await render_dynamic_blocks(
        retriever, decision, recipient_key="u1",
    )
    # Aria's event present
    assert "今早煮泡面" in out
    # NPC + user data not pulled (not queried)
    assert "小敏" not in out
    assert "工作 11-21" not in out


@pytest.mark.asyncio
async def test_subject_isolation_per_block(retriever):
    """NPC facts render in '身边的事' block, user facts in '你和他' block,
    aria facts in '你此刻' block — they must not cross."""
    decision = OrchestrationDecision(
        engage_level=0.7, register="warm",
        fact_queries=[
            OrchestratorFactQuery(category="aria.event", limit=5),
            OrchestratorFactQuery(category="user:u1.pattern", limit=5),
            OrchestratorFactQuery(category="npc:xiaomin.event", limit=5),
        ],
        topic_anchor="", skip=[],
    )
    out = await render_dynamic_blocks(retriever, decision, recipient_key="u1")
    # Find positions of each block's header
    h_self  = out.find("【你此刻】")
    h_them  = out.find("【你和他】")
    h_world = out.find("【身边的事】")
    assert h_self >= 0 and h_them >= 0 and h_world >= 0
    # Check each fact appears AFTER its block header and BEFORE next block
    aria_pos    = out.find("今早煮泡面")
    user_pos    = out.find("工作 11-21")
    npc_pos     = out.find("小敏改 paper")
    assert h_self <= aria_pos < h_them
    assert h_them <= user_pos < h_world
    assert h_world <= npc_pos


@pytest.mark.asyncio
async def test_register_renders_into_prompt(retriever):
    decision = OrchestrationDecision(
        engage_level=0.3, register="curt",
        fact_queries=[OrchestratorFactQuery(category="aria.event", limit=2)],
        topic_anchor="", skip=[],
    )
    out = await render_dynamic_blocks(retriever, decision, recipient_key="u1")
    # The register hint should be visible to the model
    assert "curt" in out.lower() or "短" in out


@pytest.mark.asyncio
async def test_skip_omits_category(retriever):
    decision = OrchestrationDecision(
        engage_level=0.6, register="warm",
        fact_queries=[
            OrchestratorFactQuery(category="aria.event", limit=5),
            OrchestratorFactQuery(category="npc:xiaomin.event", limit=5),
        ],
        topic_anchor="",
        skip=["npc:xiaomin.event"],  # explicitly skipped
    )
    out = await render_dynamic_blocks(retriever, decision, recipient_key="u1")
    assert "小敏" not in out


@pytest.mark.asyncio
async def test_topic_anchor_surfaced(retriever):
    decision = OrchestrationDecision(
        engage_level=0.6, register="warm",
        fact_queries=[OrchestratorFactQuery(category="aria.event", limit=2)],
        topic_anchor="对方在 push back 我对他工作时间的判断",
        skip=[],
    )
    out = await render_dynamic_blocks(retriever, decision, recipient_key="u1")
    assert "push back" in out or "工作时间" in out
