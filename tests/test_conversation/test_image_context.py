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
async def test_a_provider_answering_with_nothing_at_all_costs_no_turn():
    """Not every provider returns a result object; none may cost the turn."""

    class _ReturnsNone:
        async def complete(self, **kwargs):
            return None

    assert await describe_images(_ReturnsNone(), [_img()]) == ""


@pytest.mark.asyncio
async def test_no_images_costs_no_call():
    p = _Provider()
    assert await describe_images(p, []) == ""
    assert p.calls == []


@pytest.mark.asyncio
async def test_a_runaway_description_is_capped():
    p = _Provider("好" * 5000)
    assert len(await describe_images(p, [_img()])) <= 800


@pytest.mark.asyncio
async def test_the_text_on_a_schedule_reaches_her_whole():
    """A schedule summarised to one line lost every date and city on it."""
    rows = ["一张活动日程表", "3月2日 预售开启", "3月14日 见面会 杭州",
            "4月4日 见面会 南京", "4月18日 演唱会 厦门", "4月19日 见面会 厦门",
            "5月1日 快闪店 杭州"]
    out = await describe_images(_Provider("\n".join(rows)), [_img()])
    assert all(r in out for r in rows)


@pytest.mark.asyncio
async def test_she_is_asked_to_copy_the_text_out():
    p = _Provider()
    await describe_images(p, [_img()])
    instruction = p.calls[0]["messages"][0]["content"][-1]["text"]
    assert "照原样抄" in instruction and "日期" in instruction
