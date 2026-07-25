import pytest
from types import SimpleNamespace

from lingxi.brain.retrieval import web_lookup


class FakeLLM:
    def __init__(self, content="", raise_exc=False):
        self.content = content
        self.raise_exc = raise_exc
        self.tools_seen = None
    async def complete(self, *, messages, tools=None, **kw):
        self.tools_seen = tools
        if self.raise_exc:
            raise RuntimeError("network down")
        return SimpleNamespace(content=self.content)


@pytest.mark.asyncio
async def test_empty_query_short_circuits():
    llm = FakeLLM(content="should not be used")
    assert await web_lookup(llm, "   ") == ""
    assert llm.tools_seen is None  # never called


@pytest.mark.asyncio
async def test_returns_grounding_text_on_success():
    llm = FakeLLM(content="- Liella! 东京预选拿第二\n来源：love-live.fandom.com")
    out = await web_lookup(llm, "Superstar 第一季结局")
    assert "东京预选" in out
    # web_search tool was attached
    assert llm.tools_seen and llm.tools_seen[0]["name"] == "web_search"


@pytest.mark.asyncio
async def test_failsafe_on_exception():
    llm = FakeLLM(raise_exc=True)
    assert await web_lookup(llm, "anything") == ""


@pytest.mark.asyncio
async def test_allowed_domains_passed_through():
    llm = FakeLLM(content="x")
    await web_lookup(llm, "q", allowed_domains=["love-live.fandom.com"])
    assert llm.tools_seen[0]["allowed_domains"] == ["love-live.fandom.com"]
