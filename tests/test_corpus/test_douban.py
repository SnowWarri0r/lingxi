import pytest
from lingxi.fewshot.corpus.douban import fetch_topic, parse_topic

_HTML = '''<html><head><title>各位infj是否有很强的倾诉欲</title></head><body>
<div class="topic-richtext">本人最近有很强的倾诉欲 持续内耗中</div>
<div class="reply-content"><div class="markdown"><p>有 到了嘴边咽回去</p></div></div>
<div class="reply-content"><div class="markdown"><p>有 到了嘴边咽回去</p></div></div>
<div class="reply-content"><div class="markdown"><p>压抑到一定程度，倾诉欲就特别强。</p></div></div>
</body></html>'''


def test_parse_topic_extracts_title_op_replies():
    title, op, replies = parse_topic(_HTML)
    assert title == "各位infj是否有很强的倾诉欲"
    assert "倾诉欲" in op
    assert replies == ["有 到了嘴边咽回去", "压抑到一定程度，倾诉欲就特别强。"]


@pytest.mark.asyncio
async def test_fetch_topic_returns_html():
    async def fake_fetch(url: str) -> tuple[int, str, str]:
        assert "group/topic/123" in url
        return 200, url, _HTML
    html = await fetch_topic("123", fetch=fake_fetch)
    assert "倾诉欲" in html


@pytest.mark.asyncio
async def test_fetch_topic_blocked_returns_none():
    async def fake_fetch(url: str) -> tuple[int, str, str]:
        return 200, "https://sec.douban.com/c?r=xyz", "challenge"
    assert await fetch_topic("123", fetch=fake_fetch) is None


@pytest.mark.asyncio
async def test_fetch_topic_non_200_returns_none():
    async def fake_fetch(url: str) -> tuple[int, str, str]:
        return 404, url, ""
    assert await fetch_topic("404", fetch=fake_fetch) is None
