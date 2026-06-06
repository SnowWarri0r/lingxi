from lingxi.brain.memory_tools import MEMORY_TOOLS, TOOL_NAMES


def test_tools_defined():
    assert TOOL_NAMES == {
        "archival_memory_search", "archival_memory_insert",
        "core_memory_append", "core_memory_replace", "conversation_search",
        "search_stickers", "send_sticker",
    }


def test_each_tool_has_valid_schema():
    for t in MEMORY_TOOLS:
        assert "name" in t and "description" in t
        assert t["input_schema"]["type"] == "object"
        assert "properties" in t["input_schema"]


import pytest
from pathlib import Path
from datetime import datetime

from lingxi.facts.store import FactStore
from lingxi.facts.retriever import FactRetriever
from lingxi.facts.models import Fact, FactType, Source
from lingxi.brain.models import OrchestrationDecision
from lingxi.brain.renderer import render_dynamic_blocks


@pytest.mark.asyncio
async def test_core_memory_block_rendered(tmp_path):
    s = FactStore(Path(tmp_path) / "facts.db")
    await s.init()
    await s.write(Fact(subject="aria", content="我是自由天文学家",
                       source=Source.LLM_INFERRED, type=FactType.CORE,
                       ts=datetime(2026, 5, 1, 9, 0)))
    await s.write(Fact(subject="user:feishu:x", content="他熬夜",
                       source=Source.LLM_INFERRED, type=FactType.CORE,
                       ts=datetime(2026, 5, 1, 9, 0)))
    r = FactRetriever(s)
    decision = OrchestrationDecision(
        engage_level=0.6, register="warm", fact_queries=[], skip=[],
        topic_anchor="",
    )
    out = await render_dynamic_blocks(r, decision, recipient_key="feishu:x")
    assert "核心记忆" in out
    assert "我是自由天文学家" in out
    assert "他熬夜" in out
