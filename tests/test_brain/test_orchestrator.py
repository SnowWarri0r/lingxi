import json

import pytest

from lingxi.brain.models import OrchestrationDecision
from lingxi.brain.orchestrator import (
    StateDigest,
    build_orchestrator_prompt,
    decide,
)


class FakeLLMResponse:
    def __init__(self, content): self.content = content


class FakeLLM:
    def __init__(self, response_text=""):
        self.response_text = response_text
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return FakeLLMResponse(self.response_text)


@pytest.mark.asyncio
async def test_decide_returns_parsed_decision():
    payload = json.dumps({
        "engage_level": 0.7, "register": "curious",
        "fact_queries": [{"category": "aria.event", "limit": 2}],
        "topic_anchor": "x", "skip": [],
    })
    llm = FakeLLM(payload)
    digest = StateDigest(activity="刷手机", mood="平静", last_lived=["看了云"])
    catalog = {"aria.event": 5, "user:u1.pattern": 3}

    d = await decide(llm, "你今天忙吗", digest, catalog)
    assert d.engage_level == 0.7
    assert d.register == "curious"


@pytest.mark.asyncio
async def test_decide_falls_back_on_garbled_json():
    llm = FakeLLM("not json at all")
    digest = StateDigest(activity="", mood="", last_lived=[])
    d = await decide(llm, "你好", digest, {})
    # Falls back to default
    assert d.register == "warm"
    assert 0.5 <= d.engage_level <= 0.7


@pytest.mark.asyncio
async def test_decide_falls_back_on_llm_exception():
    class ExceptionLLM:
        async def complete(self, **kwargs):
            raise RuntimeError("boom")
    digest = StateDigest(activity="", mood="", last_lived=[])
    d = await decide(ExceptionLLM(), "x", digest, {})
    assert d.register == "warm"


def test_prompt_includes_user_input_and_digest_and_catalog():
    digest = StateDigest(
        activity="在写代码", mood="专注",
        last_lived=["跟外婆通了电话"],
    )
    catalog = {"aria.event": 5, "user:u1.pattern": 12}
    prompt = build_orchestrator_prompt("怎么了", digest, catalog)
    assert "怎么了" in prompt
    assert "在写代码" in prompt
    assert "外婆" in prompt
    assert "aria.event" in prompt
    assert "12" in prompt or "user:u1.pattern" in prompt


def test_prompt_specifies_strict_json_output():
    prompt = build_orchestrator_prompt(
        "x", StateDigest("", "", []), {}
    )
    assert "JSON" in prompt
    assert "register" in prompt
    assert "fact_queries" in prompt
