"""Feishu reports a failed send in the body, and the text path never looked.

HTTP 200 comes back either way; the outcome is `code`. The card path read it
and raised. The plain-text path — the fallback behind every proactive card —
discarded the response, so a failure there returned exactly like a success.
It surfaced only because the sticker path checks: 230002 on a chat the bot
had been removed from, which by then held 62 messages counted as sent.
"""

import pytest

from lingxi.channels import feishu as feishu_mod
from lingxi.channels.outbound import RecipientUnreachable

GONE = {"code": 230002, "msg": "Bot/User can NOT be out of the chat."}
RATE = {"code": 230020, "msg": "This operation triggers the frequency limit."}
OK = {"code": 0, "data": {"message_id": "om_1"}}


class _Resp:
    def __init__(self, payload):
        self._p = payload
        self.status_code = 200

    def json(self):
        return self._p


class _Client:
    def __init__(self, payload):
        self._payload = payload
        self.posts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, **kw):
        self.posts.append(url)
        return _Resp(self._payload)


def _bot():
    bot = feishu_mod.FeishuBot.__new__(feishu_mod.FeishuBot)

    class _TM:
        def headers(self):
            return {"Authorization": "Bearer t"}
    bot.token_mgr = _TM()
    return bot


class TestTheBodyIsTheVerdict:
    def test_code_zero_is_delivered(self):
        feishu_mod._raise_for_send(OK, "send")

    def test_bot_removed_from_the_chat_is_unreachable(self):
        with pytest.raises(RecipientUnreachable):
            feishu_mod._raise_for_send(GONE, "send")

    @pytest.mark.parametrize("code", [230013, 230029, 232009, 230035])
    def test_the_other_recipient_gone_codes_are_unreachable(self, code):
        with pytest.raises(RecipientUnreachable):
            feishu_mod._raise_for_send({"code": code, "msg": "x"}, "send")

    def test_a_rate_limit_is_a_failure_not_a_verdict(self):
        with pytest.raises(RuntimeError) as e:
            feishu_mod._raise_for_send(RATE, "send")
        assert not isinstance(e.value, RecipientUnreachable)

    def test_bot_ability_off_does_not_condemn_one_recipient(self):
        """True of every recipient at once — a config error, loud not local."""
        with pytest.raises(RuntimeError) as e:
            feishu_mod._raise_for_send({"code": 230006, "msg": "x"}, "send")
        assert not isinstance(e.value, RecipientUnreachable)


class TestTheTextPath:
    @pytest.mark.asyncio
    async def test_a_refused_text_raises(self, monkeypatch):
        monkeypatch.setattr(feishu_mod.httpx, "AsyncClient",
                            lambda *a, **k: _Client(GONE))

        with pytest.raises(RecipientUnreachable):
            await _bot()._send_text_async("oc_gone", "在吗")

    @pytest.mark.asyncio
    async def test_a_delivered_text_returns(self, monkeypatch):
        monkeypatch.setattr(feishu_mod.httpx, "AsyncClient",
                            lambda *a, **k: _Client(OK))

        await _bot()._send_text_async("oc_1", "在吗")


class TestTheProactiveSend:
    @pytest.mark.asyncio
    async def test_an_unreachable_card_is_not_retried_as_text(self, monkeypatch):
        """Same chat, same answer — the fallback would only add a request."""
        bot = _bot()
        texts = []

        async def _card(chat_id, text, turn_id=None):
            raise RecipientUnreachable("send card: 230002")

        async def _text(chat_id, text):
            texts.append(text)

        bot._send_proactive_card = _card
        bot._send_text_async = _text

        with pytest.raises(RecipientUnreachable):
            await bot.send_message("oc_gone", "在吗")
        assert texts == []

    @pytest.mark.asyncio
    async def test_a_broken_card_still_falls_back_to_text(self):
        bot = _bot()
        texts = []

        async def _card(chat_id, text, turn_id=None):
            raise RuntimeError("cardid is invalid")

        async def _text(chat_id, text):
            texts.append(text)

        bot._send_proactive_card = _card
        bot._send_text_async = _text

        await bot.send_message("oc_1", "在吗")
        assert texts == ["在吗"]

    @pytest.mark.asyncio
    async def test_when_both_fail_the_caller_hears_about_it(self):
        """The case that was silent: card fails, and so does the text."""
        bot = _bot()

        async def _card(chat_id, text, turn_id=None):
            raise RuntimeError("cardid is invalid")

        async def _text(chat_id, text):
            raise RecipientUnreachable("send text: 230002")

        bot._send_proactive_card = _card
        bot._send_text_async = _text

        with pytest.raises(RecipientUnreachable):
            await bot.send_message("oc_gone", "在吗")


class TestACardThatArrivedIsNotSentAgain:
    """finish() only turns off streaming mode; the text is already there."""

    @pytest.mark.asyncio
    async def test_a_failed_finish_does_not_trigger_the_text_fallback(self, monkeypatch):
        class _Kit:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, **kw):
                if url.endswith("/cardkit/v1/cards"):
                    return _Resp({"code": 0, "data": {"card_id": "card_1"}})
                return _Resp(OK)

            async def put(self, url, **kw):
                return _Resp({"code": 0})

            async def patch(self, url, **kw):
                return _Resp({"code": 300309, "msg": "settings update failed"})

        monkeypatch.setattr(feishu_mod.httpx, "AsyncClient", lambda *a, **k: _Kit())
        bot = _bot()
        texts = []

        async def _text(chat_id, text):
            texts.append(text)

        bot._send_text_async = _text

        await bot.send_message("oc_1", "练习室的空调又坏了")

        assert texts == []
