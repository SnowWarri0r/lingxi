"""Reaching the same person over QQ without her meeting a stranger.

Everything she knows about someone is keyed by "channel:id". A QQ account is
linked to the key that already holds him, replies go back over QQ, and the
next opener goes wherever he last wrote from.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from lingxi.channels.identity import IdentityMap
from lingxi.channels.outbound import (ChannelRegistry, OutboundChannel,
                                      RecipientUnreachable)
from lingxi.channels.qq import QQBot, check
from lingxi.temporal.proactive import ProactiveConfig, ProactiveScheduler
from lingxi.temporal.tracker import InteractionTracker

HOME = "feishu:oc_home"


class _Engine:
    def __init__(self, tracker, speech="诶 你来啦！\n\n可可刚练完"):
        self.interaction_tracker = tracker
        self.speech = speech
        self.calls = []

    async def chat_stream_events(self, text, images=None, channel=None,
                                 recipient_id=None):
        self.calls.append((text, channel, recipient_id))
        yield SimpleNamespace(type="chunk", content=self.speech)
        yield SimpleNamespace(type="done", content=self.speech)


CODE = "pair-7f3a"


def _bot(tmp_path, engine, link_home=HOME):
    bot = QQBot(engine, "app", "secret",
                identities=IdentityMap(tmp_path / "identities.json"),
                link_home=link_home, link_code=CODE)
    sent = []

    async def _api(method, path, body=None):
        sent.append((method, path, body))
        return {"id": "m", "file_info": "fi"}

    bot._api = _api
    return bot, sent


def _event(text="在吗", msg_id="MSG1", openid="OPENID"):
    return {"id": msg_id, "author": {"user_openid": openid}, "content": text}


@pytest.mark.asyncio
async def test_he_is_the_person_she_already_knows(tmp_path):
    tracker = InteractionTracker(tmp_path)
    tracker.record_interaction("feishu", "oc_home")
    engine = _Engine(tracker)
    bot, _ = _bot(tmp_path, engine)

    await bot._on_message(_event(CODE, msg_id="pair"))
    await bot._on_message(_event())

    assert engine.calls == [("在吗", "feishu", "oc_home")]
    assert IdentityMap(tmp_path / "identities.json").resolve("qq", "OPENID") == \
        ("feishu", "oc_home")


@pytest.mark.asyncio
async def test_each_bubble_is_a_reply_to_his_message(tmp_path):
    tracker = InteractionTracker(tmp_path)
    tracker.record_interaction("feishu", "oc_home")
    bot, sent = _bot(tmp_path, _Engine(tracker))

    await bot._on_message(_event(CODE, msg_id="pair"))
    sent.clear()
    await bot._on_message(_event())

    bodies = [b for _, path, b in sent if path.endswith("/messages")]
    assert [b["content"] for b in bodies] == ["诶 你来啦！", "可可刚练完"]
    assert [(b["msg_id"], b["msg_seq"]) for b in bodies] == [("MSG1", 1), ("MSG1", 2)]


@pytest.mark.asyncio
async def test_a_message_delivered_twice_is_answered_once(tmp_path):
    tracker = InteractionTracker(tmp_path)
    tracker.record_interaction("feishu", "oc_home")
    engine = _Engine(tracker)
    bot, _ = _bot(tmp_path, engine)

    await bot._on_message(_event())
    await bot._on_message(_event())

    assert len(engine.calls) == 1


@pytest.mark.asyncio
async def test_after_he_writes_on_qq_her_openers_go_there(tmp_path):
    tracker = InteractionTracker(tmp_path)
    tracker.record_interaction("feishu", "oc_home")
    bot, _ = _bot(tmp_path, _Engine(tracker))

    await bot._on_message(_event(CODE, msg_id="pair"))
    await bot._on_message(_event())

    assert tracker.get_record("feishu", "oc_home").delivery() == ("qq", "OPENID")
    tracker.set_route("feishu", "oc_home", "feishu", "oc_home")
    assert tracker.get_record("feishu", "oc_home").delivery() == ("feishu", "oc_home")


@pytest.mark.asyncio
async def test_an_account_that_writes_first_is_not_given_his_memory(tmp_path):
    """Only the pairing code links; being first proves nothing."""
    tracker = InteractionTracker(tmp_path)
    engine = _Engine(tracker)
    bot, _ = _bot(tmp_path, engine)
    await bot._on_message(_event("你好", openid="SOMEONE", msg_id="a"))
    await bot._on_message(_event("pair-wrong", openid="SOMEONE", msg_id="b"))

    ids = IdentityMap(tmp_path / "identities.json")
    assert ids.resolve("qq", "SOMEONE") == ("qq", "SOMEONE")
    assert all(c[1:] == ("qq", "SOMEONE") for c in engine.calls)


@pytest.mark.asyncio
async def test_the_pairing_code_is_not_shown_to_her(tmp_path):
    tracker = InteractionTracker(tmp_path)
    engine = _Engine(tracker)
    bot, _ = _bot(tmp_path, engine)
    await bot._on_message(_event(CODE, msg_id="pair"))
    assert engine.calls == []


def test_switched_off_openers_is_unreachable_not_an_error():
    with pytest.raises(RecipientUnreachable):
        check({"code": 40054013, "message": "user refused"}, "send")


# --- the opener follows him --------------------------------------------

class _Chan(OutboundChannel):
    def __init__(self, name, refuse=False):
        self.name, self.refuse, self.sent = name, refuse, []

    @property
    def channel_name(self):
        return self.name

    async def send_message(self, recipient_id, text, turn_id=None):
        if self.refuse:
            raise RecipientUnreachable("40054013")
        self.sent.append((recipient_id, text))


def _sched(tmp_path, qq):
    tracker = InteractionTracker(tmp_path)
    rec = tracker.record_interaction("feishu", "oc_home")
    rec.last_interaction = datetime.now() - timedelta(days=2)
    tracker.set_route("feishu", "oc_home", "qq", "OPENID")
    reg = ChannelRegistry()
    feishu = _Chan("feishu")
    reg.register(feishu)
    reg.register(qq)

    async def _append(*a):
        pass

    engine = SimpleNamespace(
        annotation_store=None, _pending_stickers={},
        memory=SimpleNamespace(embedding_provider=None,
                               short_term=SimpleNamespace(append_for_recipient=_append)))
    sched = ProactiveScheduler(config=ProactiveConfig(), tracker=tracker,
                               channel_registry=reg, engine=engine)

    async def _ask(record, silence, now, force=False):
        return {"should_send": True, "message": "可可刚练完！", "sticker": "", "reason": ""}

    sched._ask_llm = _ask
    return sched, rec, feishu


@pytest.mark.asyncio
async def test_the_opener_goes_where_he_last_wrote(tmp_path):
    qq = _Chan("qq")
    sched, rec, feishu = _sched(tmp_path, qq)

    result = await sched._maybe_reach_out(rec, datetime.now(), force=True)

    assert result["status"] == "sent"
    assert qq.sent == [("OPENID", "可可刚练完！")] and feishu.sent == []


@pytest.mark.asyncio
async def test_refused_on_qq_falls_back_without_silencing_him(tmp_path):
    sched, rec, _ = _sched(tmp_path, _Chan("qq", refuse=True))

    result = await sched._maybe_reach_out(rec, datetime.now(), force=True)

    assert result["status"] == "route_dropped"
    assert rec.unreachable_since is None
    assert rec.delivery() == ("feishu", "oc_home")


@pytest.mark.asyncio
async def test_the_code_links_once_and_then_nothing_else(tmp_path):
    """A code that leaks after he has paired cannot hand his memory over."""
    tracker = InteractionTracker(tmp_path)
    engine = _Engine(tracker)
    bot, _ = _bot(tmp_path, engine)
    await bot._on_message(_event(CODE, openid="HIM", msg_id="a"))
    await bot._on_message(_event(CODE, openid="SOMEONE", msg_id="b"))

    ids = IdentityMap(tmp_path / "identities.json")
    assert ids.resolve("qq", "HIM") == ("feishu", "oc_home")
    assert ids.resolve("qq", "SOMEONE") == ("qq", "SOMEONE")


@pytest.mark.asyncio
async def test_any_failure_on_qq_falls_back_instead_of_retrying_there(tmp_path):
    """Only a refusal used to fall back; an IP-whitelist or quota error left
    her composing a new opener every five minutes for a channel that kept
    refusing, and nothing reached Feishu."""
    class _Broken(_Chan):
        async def send_message(self, recipient_id, text, turn_id=None):
            raise RuntimeError("11298 ip not in whitelist")

    sched, rec, _ = _sched(tmp_path, _Broken("qq"))

    result = await sched._maybe_reach_out(rec, datetime.now(), force=True)

    assert result["status"] == "route_dropped"
    assert rec.delivery() == ("feishu", "oc_home")


@pytest.mark.asyncio
async def test_an_opener_on_qq_is_one_message(tmp_path):
    bot, sent = _bot(tmp_path, _Engine(InteractionTracker(tmp_path)))
    await bot.send_message("OPENID", "第一句\n\n第二句")
    bodies = [b for _, path, b in sent if path.endswith("/messages")]
    assert [b["content"] for b in bodies] == ["第一句\n第二句"]
    assert "msg_id" not in bodies[0]


@pytest.mark.asyncio
async def test_a_network_blip_on_qq_does_not_move_him(tmp_path):
    """A timeout is retried on the next tick; only repeated blips fall back."""
    import httpx

    class _Flaky(_Chan):
        async def send_message(self, recipient_id, text, turn_id=None):
            raise httpx.ReadTimeout("timed out")

    sched, rec, _ = _sched(tmp_path, _Flaky("qq"))
    first = await sched._maybe_reach_out(rec, datetime.now(), force=True)
    assert first["status"] == "send_failed"
    assert rec.delivery() == ("qq", "OPENID")
    await sched._maybe_reach_out(rec, datetime.now(), force=True)
    third = await sched._maybe_reach_out(rec, datetime.now(), force=True)
    assert third["status"] == "route_dropped"
