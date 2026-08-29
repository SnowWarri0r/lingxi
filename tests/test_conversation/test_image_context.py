"""Images become text before they reach a text-only responder.

deepseek-v4-flash answers `400 This model does not support image`, which
emptied the entire turn — every photo and every Feishu sticker got the
empty-reply fallback instead of a reply.
"""

import pytest

from lingxi.conversation.image_context import describe_images


class _Result:
    def __init__(self, content):
        self.content = content


class _Provider:
    """Records what it was asked, answers with a canned description."""

    def __init__(self, content="一只猫顶着锅盖，图上写着「我不听」"):
        self._content = content
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return _Result(self._content)


class _Boom:
    async def complete(self, **kwargs):
        raise RuntimeError("vision endpoint down")


def _img(data="AAAA", media_type="image/png"):
    return {"data": data, "media_type": media_type}


@pytest.mark.asyncio
async def test_the_description_comes_back_as_a_plain_line():
    out = await describe_images(_Provider(), [_img()])
    assert out == "一只猫顶着锅盖，图上写着「我不听」"


@pytest.mark.asyncio
async def test_the_image_is_sent_as_a_base64_block():
    p = _Provider()
    await describe_images(p, [_img(data="Zm9v", media_type="image/jpeg")])

    content = p.calls[0]["messages"][0]["content"]
    image_block = content[0]
    assert image_block["type"] == "image"
    assert image_block["source"]["data"] == "Zm9v"
    assert image_block["source"]["media_type"] == "image/jpeg"
    assert content[-1]["type"] == "text", "the instruction follows the image"


@pytest.mark.asyncio
async def test_several_images_ride_one_call_and_join_into_one_line():
    p = _Provider("第一张：海边的照片\n第二张：一杯咖啡")
    out = await describe_images(p, [_img(), _img()])

    assert len(p.calls) == 1
    blocks = [b for b in p.calls[0]["messages"][0]["content"] if b["type"] == "image"]
    assert len(blocks) == 2
    assert out == "第一张：海边的照片；第二张：一杯咖啡"
    assert "\n" not in out, "the description is one inline marker"


@pytest.mark.asyncio
async def test_a_failed_look_is_not_a_failed_turn():
    assert await describe_images(_Boom(), [_img()]) == ""


@pytest.mark.asyncio
async def test_an_empty_reply_yields_no_description():
    assert await describe_images(_Provider(""), [_img()]) == ""


@pytest.mark.asyncio
async def test_no_images_costs_no_call():
    p = _Provider()
    assert await describe_images(p, []) == ""
    assert p.calls == []


@pytest.mark.asyncio
async def test_a_runaway_description_is_capped():
    p = _Provider("好" * 500)
    assert len(await describe_images(p, [_img()])) <= 120
