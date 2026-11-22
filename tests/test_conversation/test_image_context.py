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


# ── Knowing her own face ──────────────────────────────────────────────────

from lingxi.conversation.image_context import Likeness  # noqa: E402


class _Judge:
    """Describes, and answers the likeness question with `verdict`."""

    def __init__(self, verdict="是", desc="一个小玩偶揣在演出服里"):
        self.verdict, self.desc, self.calls = verdict, desc, []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("_debug_purpose") == "image_likeness":
            if isinstance(self.verdict, Exception):
                raise self.verdict
            return _Result(self.verdict)
        return _Result(self.desc)


_ME = Likeness("阿澪", "UkVG", "image/png")


@pytest.mark.asyncio
async def test_a_picture_of_her_says_it_is_her():
    out = await describe_images(_Judge("是"), [_img()], _ME)
    assert out.startswith("一个小玩偶揣在演出服里")
    assert "阿澪" in out


@pytest.mark.asyncio
async def test_a_picture_of_someone_else_is_described_as_before():
    assert await describe_images(_Judge("否"), [_img()], _ME) == "一个小玩偶揣在演出服里"


@pytest.mark.asyncio
async def test_a_failed_likeness_check_leaves_the_description_alone():
    j = _Judge(RuntimeError("down"))
    assert await describe_images(j, [_img()], _ME) == "一个小玩偶揣在演出服里"


@pytest.mark.asyncio
async def test_the_check_sees_her_picture_first_then_his():
    j = _Judge("是")
    await describe_images(j, [_img(data="SElT")], _ME)
    ask = next(c for c in j.calls if c["_debug_purpose"] == "image_likeness")
    imgs = [b["source"]["data"] for b in ask["messages"][0]["content"] if b["type"] == "image"]
    assert imgs == ["UkVG", "SElT"]
    describe = next(c for c in j.calls if c["_debug_purpose"] == "image_describe")
    assert "UkVG" not in str(describe), "her picture stays out of the description"


@pytest.mark.asyncio
async def test_without_a_likeness_there_is_one_call():
    j = _Judge("是")
    assert await describe_images(j, [_img()]) == "一个小玩偶揣在演出服里"
    assert len(j.calls) == 1


def test_the_likeness_is_read_from_the_data_directory(tmp_path):
    assert Likeness.load(tmp_path, "阿澪") is None
    (tmp_path / "likeness.jpg").write_bytes(b"REF")
    me = Likeness.load(tmp_path, "阿澪")
    assert me.data == "UkVG" and me.media_type == "image/jpeg" and me.name == "阿澪"
