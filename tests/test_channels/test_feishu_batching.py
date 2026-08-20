"""How a batch of inbound messages becomes one Feishu turn."""

import asyncio

import pytest

from lingxi.channels.feishu import FeishuBot


def _bot():
    """A FeishuBot with only the attributes batching touches.

    The real constructor wants credentials and builds a token manager; none of
    that is involved in turning a batch into a turn.
    """
    bot = object.__new__(FeishuBot)
    bot._chat_locks = {}
    bot._handled = []

    async def _fake_handle(chat_id, text, image_keys, msg_id):
        bot._handled.append((chat_id, text, list(image_keys), msg_id))

    bot._handle_reply_safe = _fake_handle
    return bot


@pytest.mark.asyncio
async def test_a_batch_becomes_exactly_one_turn():
    bot = _bot()
    await bot._flush_batch("chat", [
        ("在吗", [], "m1"),
        ("有个事想问你", [], "m2"),
        ("就是那个排练的事", [], "m3"),
    ])

    assert len(bot._handled) == 1, "a burst must not produce several turns"


@pytest.mark.asyncio
async def test_lines_are_joined_the_way_they_were_sent():
    """Newlines, not spaces — the separate lines carry the sender's phrasing."""
    bot = _bot()
    await bot._flush_batch("chat", [
        ("在吗", [], "m1"),
        ("有个事想问你", [], "m2"),
    ])

    _chat, text, _imgs, _mid = bot._handled[0]
    assert text == "在吗\n有个事想问你"


@pytest.mark.asyncio
async def test_images_across_the_batch_all_arrive():
    bot = _bot()
    await bot._flush_batch("chat", [
        ("看这个", ["k1"], "m1"),
        ("", ["k2", "k3"], "m2"),
    ])

    _chat, text, imgs, _mid = bot._handled[0]
    assert imgs == ["k1", "k2", "k3"]
    assert text == "看这个", "an image-only message contributes no empty line"


@pytest.mark.asyncio
async def test_the_turn_is_attributed_to_the_last_message():
    bot = _bot()
    await bot._flush_batch("chat", [("一", [], "m1"), ("二", [], "m2")])

    assert bot._handled[0][3] == "m2"


@pytest.mark.asyncio
async def test_an_image_only_batch_still_produces_a_turn():
    bot = _bot()
    await bot._flush_batch("chat", [("", ["k1"], "m1")])

    assert len(bot._handled) == 1
    assert bot._handled[0][1] == ""


@pytest.mark.asyncio
async def test_batches_for_one_chat_do_not_overlap():
    """The card is created inside the turn, so two turns must not interleave.

    The engine serialises turns per recipient, but only after the channel has
    already created and sent the streaming card — without this lock a second
    batch drops an empty card into the chat that sits idle until the first
    turn finishes.
    """
    bot = _bot()
    active = 0
    overlapped = False

    async def _slow_handle(chat_id, text, image_keys, msg_id):
        nonlocal active, overlapped
        active += 1
        if active > 1:
            overlapped = True
        await asyncio.sleep(0.02)
        active -= 1

    bot._handle_reply_safe = _slow_handle

    await asyncio.gather(
        bot._flush_batch("chat", [("一", [], "m1")]),
        bot._flush_batch("chat", [("二", [], "m2")]),
    )

    assert not overlapped


@pytest.mark.asyncio
async def test_different_chats_are_not_serialised_against_each_other():
    bot = _bot()
    order = []

    async def _handle(chat_id, text, image_keys, msg_id):
        order.append(f"{chat_id}-start")
        await asyncio.sleep(0.02)
        order.append(f"{chat_id}-end")

    bot._handle_reply_safe = _handle

    await asyncio.gather(
        bot._flush_batch("a", [("x", [], "m1")]),
        bot._flush_batch("b", [("y", [], "m2")]),
    )

    # Interleaved rather than one-after-the-other.
    assert order.index("b-start") < order.index("a-end")
