import base64
import pytest
from pathlib import Path

from lingxi.stickers.captioner import caption_image


class _FakeProvider:
    def __init__(self, text):
        self._text = text
        self.last_messages = None

    async def complete(self, messages, system=None, max_tokens=1024,
                       temperature=0.7, **kwargs):
        from lingxi.providers.base import CompletionResult
        self.last_messages = messages
        return CompletionResult(content=self._text)


def _write_png(tmp_path) -> Path:
    # 1x1 PNG (valid header so base64 round-trips)
    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    p = Path(tmp_path) / "x.png"
    p.write_bytes(raw)
    return p


@pytest.mark.asyncio
async def test_caption_image_parses_json(tmp_path):
    prov = _FakeProvider(
        '这是表情 {"caption":"笑哭","emotion":"好笑",'
        '"tags":["笑哭","捂脸"],"when_to_use":"觉得好笑时"} 完毕')
    img = _write_png(tmp_path)
    result = await caption_image(prov, img)
    assert result["caption"] == "笑哭"
    assert result["emotion"] == "好笑"
    assert result["tags"] == ["笑哭", "捂脸"]
    assert result["when_to_use"] == "觉得好笑时"
    # image must be attached as a base64 block
    blocks = prov.last_messages[-1]["content"]
    assert any(b["type"] == "image" for b in blocks)


@pytest.mark.asyncio
async def test_caption_image_bad_json_is_safe(tmp_path):
    prov = _FakeProvider("抱歉我看不清这张图")
    img = _write_png(tmp_path)
    result = await caption_image(prov, img)
    assert result["caption"] == ""
    assert result["tags"] == []


@pytest.mark.asyncio
async def test_caption_image_strips_code_fence(tmp_path):
    prov = _FakeProvider(
        '```json\n{"caption":"摸鱼","emotion":"懒","tags":["摸鱼"],'
        '"when_to_use":"上班划水"}\n```')
    img = _write_png(tmp_path)
    result = await caption_image(prov, img)
    assert result["caption"] == "摸鱼"
    assert result["tags"] == ["摸鱼"]


@pytest.mark.asyncio
async def test_caption_image_json_list_is_safe(tmp_path):
    # A bare JSON array parses fine but is not a dict → must return empty,
    # not raise AttributeError.
    prov = _FakeProvider('[{"caption":"x"}]')
    img = _write_png(tmp_path)
    result = await caption_image(prov, img)
    assert result["caption"] == ""
    assert result["tags"] == []


@pytest.mark.asyncio
async def test_caption_image_trailing_braces_ok(tmp_path):
    # Prose with a flat JSON object followed by an empty {} must still parse
    # the real object, not greedily span to the trailing braces.
    prov = _FakeProvider(
        '这是 {"caption":"无语","emotion":"无语","tags":["无语"],'
        '"when_to_use":"离谱"} 备注 {}')
    img = _write_png(tmp_path)
    result = await caption_image(prov, img)
    assert result["caption"] == "无语"
