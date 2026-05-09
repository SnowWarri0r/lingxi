"""Tests for BiographySelector — LLM-driven bio event selection."""

import json
from unittest.mock import AsyncMock

import pytest

from lingxi.persona.biography_selector import BiographySelector
from lingxi.persona.models import LifeEvent
from lingxi.providers.base import CompletionResult


def _events():
    return [
        LifeEvent(age=18, content="奶奶去世那天我在大学", tags=["丧亲"]),
        LifeEvent(age=25, content="第一次自己装望远镜", tags=["天文", "成长"]),
        LifeEvent(age=27, content="跟小林吵架后冷战一周", tags=["朋友"]),
        LifeEvent(age=28, content="搬到上海第一晚 失眠", tags=["搬家"]),
    ]


def _fake_llm(payload: str | None) -> AsyncMock:
    """Return an LLMProvider stub whose complete() returns the payload."""
    llm = AsyncMock()
    if payload is None:
        llm.complete.side_effect = RuntimeError("network down")
    else:
        llm.complete.return_value = CompletionResult(content=payload)
    return llm


@pytest.mark.asyncio
async def test_empty_events_returns_empty():
    sel = BiographySelector(events=[], llm=_fake_llm('{"selected": []}'))
    assert await sel.select("hello") == []


@pytest.mark.asyncio
async def test_empty_query_returns_empty():
    sel = BiographySelector(events=_events(), llm=_fake_llm('{"selected": [0]}'))
    assert await sel.select("") == []
    assert await sel.select("   ") == []


@pytest.mark.asyncio
async def test_confrontation_short_circuits_no_llm_call():
    """Confrontation should skip the LLM call entirely (latency saving)."""
    llm = _fake_llm('{"selected": [0, 1]}')
    sel = BiographySelector(events=_events(), llm=llm)
    result = await sel.select("好敷衍", is_confrontation=True)
    assert result == []
    llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_normal_selection_returns_picked_events():
    sel = BiographySelector(
        events=_events(),
        llm=_fake_llm('{"selected": [1]}'),
    )
    picked = await sel.select("我刚买了望远镜")
    assert len(picked) == 1
    assert picked[0].age == 25
    assert "望远镜" in picked[0].content


@pytest.mark.asyncio
async def test_multiple_picks_respected_in_order():
    sel = BiographySelector(
        events=_events(),
        llm=_fake_llm('{"selected": [3, 0]}'),
    )
    picked = await sel.select("最近搬家好累")
    assert len(picked) == 2
    assert picked[0].age == 28  # 搬家
    assert picked[1].age == 18


@pytest.mark.asyncio
async def test_max_events_caps_selection():
    # LLM returns 3 but max=2 caps it
    sel = BiographySelector(
        events=_events(),
        llm=_fake_llm('{"selected": [0, 1, 2]}'),
    )
    picked = await sel.select("test", max_events=2)
    assert len(picked) == 2


@pytest.mark.asyncio
async def test_invalid_id_dropped():
    sel = BiographySelector(
        events=_events(),
        llm=_fake_llm('{"selected": [99, 0, "garbage"]}'),
    )
    picked = await sel.select("test")
    assert len(picked) == 1
    assert picked[0].age == 18


@pytest.mark.asyncio
async def test_garbage_response_returns_empty():
    sel = BiographySelector(
        events=_events(),
        llm=_fake_llm("not json at all"),
    )
    assert await sel.select("test") == []


@pytest.mark.asyncio
async def test_markdown_fenced_response_parsed():
    sel = BiographySelector(
        events=_events(),
        llm=_fake_llm('```json\n{"selected": [1]}\n```'),
    )
    picked = await sel.select("test")
    assert len(picked) == 1
    assert picked[0].age == 25


@pytest.mark.asyncio
async def test_llm_failure_returns_empty():
    sel = BiographySelector(events=_events(), llm=_fake_llm(None))
    assert await sel.select("test") == []


@pytest.mark.asyncio
async def test_append_event_grows_list():
    sel = BiographySelector(events=_events(), llm=_fake_llm('{"selected": []}'))
    new_event = LifeEvent(age=29, content="第一次跑马拉松", tags=["运动"])
    sel.append(new_event)
    assert len(sel.events) == 5
    assert sel.events[-1].content == "第一次跑马拉松"


@pytest.mark.asyncio
async def test_tone_hints_passed_to_prompt():
    """Verify the LLM call gets is_heavy / recent_emotion in user message."""
    llm = AsyncMock()
    llm.complete.return_value = CompletionResult(content='{"selected": []}')
    sel = BiographySelector(events=_events(), llm=llm)
    await sel.select("test", is_heavy=True, recent_emotion="悲伤")
    call_kwargs = llm.complete.call_args.kwargs
    user_msg = call_kwargs["messages"][0]["content"]
    assert "用户在谈沉重话题" in user_msg
    assert "悲伤" in user_msg


class TestManifestRendering:
    def test_manifest_format(self):
        sel = BiographySelector(events=_events(), llm=AsyncMock())
        m = sel._build_manifest()
        # Each event gets [id] N岁·content[#tags]
        assert "[0]" in m
        assert "[1]" in m
        assert "18岁·" in m
        assert "#丧亲" in m
        assert "25岁·" in m

    def test_manifest_truncates_long_content(self):
        long_event = LifeEvent(age=30, content="x" * 200, tags=[])
        sel = BiographySelector(events=[long_event], llm=AsyncMock())
        m = sel._build_manifest()
        assert "…" in m
        # Capped under ~80 chars total
        assert len(m) < 100
