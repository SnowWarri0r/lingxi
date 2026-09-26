"""The write that puts the reply into the card was unchecked twice over.

A reply is delivered by posting an empty streaming card into the chat and
then writing text into it. The final write — the finished first bubble — sat
in `except Exception: pass`, and the write itself never read Feishu's answer,
so it could not have raised even if something were listening. When it fails,
the card keeps the last streamed frame: a half sentence, or the 💭 thinking
placeholder, with the reply recorded as delivered.

The error branch already had a static-card rescue for a failed write, keyed
on the write raising. It never fired, because the write never raised.

No occurrence has been observed — the path logged nothing, which is the
point. The same shape, on the proactive path, was hiding 62 undelivered
messages to one chat.
"""

import pytest

from lingxi.channels import feishu as feishu_mod
from lingxi.conversation.engine import StreamEvent


REPLY = "那家店的包子今天换馅了 还挺好吃"


class _Resp:
    def __init__(self, payload):
        self._p = payload
        self.status_code = 200

    def json(self):
        return self._p


class _CardKit:
    """Fake CardKit: streaming writes succeed unless told the final one fails."""

    def __init__(self, fail_final_times=0):
        self.writes = []
        self._fail_left = fail_final_times

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, **kw):
        if url.endswith("/cardkit/v1/cards"):
            return _Resp({"code": 0, "data": {"card_id": "card_1"}})
        return _Resp({"code": 0, "data": {"message_id": "om_1"}})

    async def put(self, url, json=None, **kw):
        text = json["content"]
        if text == REPLY and self._fail_left > 0:
            self._fail_left -= 1
            return _Resp({"code": 300309, "msg": "streaming card update throttled"})
        self.writes.append(text)
        return _Resp({"code": 0})

    async def patch(self, url, **kw):
        return _Resp({"code": 0})


class _Engine:
    async def chat_stream_events(self, text, **kw):
        yield StreamEvent(type="thinking", content="想起早上那家包子铺")
        yield StreamEvent(type="chunk", content="那家店的包子")
        yield StreamEvent(type="chunk", content="今天换馅了")
        # The last frame stops short of the reply, which is exactly when a
        # failed final write leaves a half sentence on screen.
        yield StreamEvent(type="done", content=REPLY)


def _bot(monkeypatch, kit):
    bot = feishu_mod.FeishuBot.__new__(feishu_mod.FeishuBot)

    class _TM:
        def headers(self):
            return {"Authorization": "Bearer t"}
    bot.token_mgr = _TM()
    bot.engine = _Engine()
    bot._update_interval = 0.0
    statics = []

    async def _static(chat_id, text):
        statics.append(text)
        return "card_static"

    async def _noop(*a, **k):
        return None

    bot._send_static_card_async = _static
    bot._append_to_card_id = _noop
    monkeypatch.setattr(feishu_mod.httpx, "AsyncClient", lambda *a, **k: kit)

    async def _nosleep(_):
        return None
    monkeypatch.setattr(feishu_mod.asyncio, "sleep", _nosleep)
    return bot, statics


@pytest.mark.asyncio
async def test_a_healthy_turn_puts_the_reply_in_the_card(monkeypatch):
    kit = _CardKit()
    bot, statics = _bot(monkeypatch, kit)

    await bot._stream_reply("oc_1", "在吗")

    assert kit.writes[-1] == REPLY
    assert statics == []


@pytest.mark.asyncio
async def test_one_refused_write_is_retried_into_the_card(monkeypatch):
    """A throttled write right behind the last frame is the likeliest failure."""
    kit = _CardKit(fail_final_times=1)
    bot, statics = _bot(monkeypatch, kit)

    await bot._stream_reply("oc_1", "在吗")

    assert kit.writes[-1] == REPLY
    assert statics == []


@pytest.mark.asyncio
async def test_a_write_that_never_lands_is_sent_as_its_own_message(monkeypatch):
    kit = _CardKit(fail_final_times=2)
    bot, statics = _bot(monkeypatch, kit)

    await bot._stream_reply("oc_1", "在吗")

    assert kit.writes[-1] == "那家店的包子今天换馅了"   # what the card was left showing
    assert statics == [REPLY]


class TestTheWriteReportsItsOutcome:
    @pytest.mark.asyncio
    async def test_a_refused_update_raises(self):
        class _Http:
            async def put(self, *a, **k):
                return _Resp({"code": 300309, "msg": "throttled"})

        card = feishu_mod.StreamingCardSender(
            type("TM", (), {"headers": lambda self: {}})(), _Http())
        card._card_id = "card_1"

        with pytest.raises(RuntimeError):
            await card.update_content("x")

    @pytest.mark.asyncio
    async def test_an_accepted_update_returns(self):
        class _Http:
            async def put(self, *a, **k):
                return _Resp({"code": 0})

        card = feishu_mod.StreamingCardSender(
            type("TM", (), {"headers": lambda self: {}})(), _Http())
        card._card_id = "card_1"

        await card.update_content("x")
