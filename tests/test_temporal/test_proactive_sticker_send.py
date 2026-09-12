"""A mood she tagged on an opener has to leave the process as an image.

The reply path resolves `#表情 X` and emits the file; the opener path parsed
the tag off the text and dropped it. Across 119 logged openers she asked for
a sticker 13 times and sent zero, on the path that carries nearly all her
outgoing messages.
"""

import pytest

from lingxi.channels.outbound import OutboundChannel


class _Channel(OutboundChannel):
    def __init__(self):
        self.messages = []
        self.stickers = []

    @property
    def channel_name(self):
        return "test"

    async def send_message(self, recipient_id, text, turn_id=None):
        self.messages.append(text)

    async def send_sticker(self, recipient_id, file_path):
        self.stickers.append(file_path)


class _Silent(OutboundChannel):
    """A channel that cannot send images — the default must be a no-op."""

    def __init__(self):
        self.messages = []

    @property
    def channel_name(self):
        return "silent"

    async def send_message(self, recipient_id, text, turn_id=None):
        self.messages.append(text)


@pytest.mark.asyncio
async def test_a_channel_without_image_support_just_skips():
    ch = _Silent()
    await ch.send_message("r", "hi")
    assert await ch.send_sticker("r", "/tmp/x.png") is None
    assert ch.messages == ["hi"]


@pytest.mark.asyncio
async def test_a_capable_channel_receives_the_file():
    ch = _Channel()
    await ch.send_sticker("r", "/stickers/happy.png")
    assert ch.stickers == ["/stickers/happy.png"]


class TestParseFeedsTheSend:
    """The dict the send path reads must carry the intent."""

    def test_the_key_the_sender_looks_up_is_present(self):
        from lingxi.temporal.proactive import parse_proactive_output

        out = parse_proactive_output("场刊翻到我们那页了\n#表情 开心")

        assert out.get("sticker") == "开心"
        assert out.get("should_send") is True
        assert "#表情" not in out.get("message", "")

    def test_an_untagged_opener_asks_for_nothing(self):
        from lingxi.temporal.proactive import parse_proactive_output

        assert parse_proactive_output("刚练完")["sticker"] == ""
