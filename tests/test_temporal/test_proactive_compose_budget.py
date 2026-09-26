"""One opener in five came back empty, and was logged as her choosing silence.

The responder reasons before it writes (deepseek: thinking on, low effort),
and reasoning is billed against max_tokens. Proactive gave it 600; replies
from the same model stream with the provider default of 4096. On the live
chat 24 of 119 composes spent all 600 thinking and returned no text, and an
empty reply parses as should_send=False — so each was recorded as
llm_declined, as though she had looked at the moment and let it pass.
"""

from types import SimpleNamespace

import pytest

from lingxi.channels.outbound import ChannelRegistry
from lingxi.temporal.proactive import ProactiveConfig, ProactiveScheduler
from lingxi.temporal.tracker import InteractionTracker


class _Responder:
    def __init__(self, content):
        self.content = content
        self.kwargs = {}

    async def complete(self, **kw):
        self.kwargs = kw
        return SimpleNamespace(content=self.content, usage={"output_tokens": 600})


def _scheduler(tmp_path, responder):
    engine = SimpleNamespace(
        _get_responder_llm=lambda: responder,
        _responder_is_external=lambda: False,
    )
    sched = ProactiveScheduler(config=ProactiveConfig(), tracker=InteractionTracker(tmp_path),
                               channel_registry=ChannelRegistry(), engine=engine)

    async def _prompt(*a, **k):
        return "system", [{"role": "user", "content": "要不要发"}]

    sched.build_proactive_prompt = _prompt
    return sched


@pytest.mark.asyncio
async def test_an_empty_compose_is_not_a_decision(tmp_path):
    sched = _scheduler(tmp_path, _Responder(""))

    assert await sched._ask_llm(None, None, None) is None


@pytest.mark.asyncio
async def test_a_real_decline_is_still_a_decline(tmp_path):
    sched = _scheduler(tmp_path, _Responder('\n===META===\n{"should_send": false, "inner": "他在开会"}'))

    decision = await sched._ask_llm(None, None, None)

    assert decision is not None and decision["should_send"] is False


@pytest.mark.asyncio
async def test_a_written_opener_comes_through(tmp_path):
    sched = _scheduler(tmp_path, _Responder("练习室空调又坏了 今天只能开窗跳"))

    decision = await sched._ask_llm(None, None, None)

    assert decision["should_send"] is True and "开窗" in decision["message"]


@pytest.mark.asyncio
async def test_the_compose_has_room_to_reason_and_still_write(tmp_path):
    responder = _Responder("练习室空调又坏了")
    await _scheduler(tmp_path, responder)._ask_llm(None, None, None)

    assert responder.kwargs["max_tokens"] >= 2000
