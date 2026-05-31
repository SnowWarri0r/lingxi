import hashlib
import pytest
from pathlib import Path

from lingxi.stickers.crawler import download_images


@pytest.mark.asyncio
async def test_download_writes_files_and_dedupes(tmp_path):
    # Two URLs return identical bytes → only one file kept (hash dedup).
    payload = {
        "http://x/a.png": b"AAAA",
        "http://x/b.png": b"AAAA",   # same bytes as a.png
        "http://x/c.png": b"BBBB",
    }

    async def fake_fetch(url: str) -> bytes:
        return payload[url]

    out = Path(tmp_path) / "img"
    results = await download_images(
        list(payload.keys()), out_dir=out, fetch=fake_fetch, delay=0.0)

    # 3 URLs, 2 distinct hashes → 2 stored
    assert len(results) == 2
    hashes = {r["content_hash"] for r in results}
    assert len(hashes) == 2
    for r in results:
        assert Path(r["file_path"]).exists()
        expected = hashlib.sha256(
            Path(r["file_path"]).read_bytes()).hexdigest()
        assert r["content_hash"] == expected


@pytest.mark.asyncio
async def test_download_skips_fetch_failures(tmp_path):
    async def fake_fetch(url: str) -> bytes:
        if url == "http://x/bad.png":
            raise RuntimeError("boom")
        return b"OK"

    out = Path(tmp_path) / "img"
    results = await download_images(
        ["http://x/good.png", "http://x/bad.png"],
        out_dir=out, fetch=fake_fetch, delay=0.0, retries=0)
    assert len(results) == 1
    assert results[0]["source_url"] == "http://x/good.png"


@pytest.mark.asyncio
async def test_download_retries_then_succeeds(tmp_path):
    calls = {"n": 0}

    async def flaky_fetch(url: str) -> bytes:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return b"GOODBYTES"

    out = Path(tmp_path) / "img"
    results = await download_images(
        ["http://x/r.png"], out_dir=out, fetch=flaky_fetch,
        delay=0.0, retries=1)
    assert len(results) == 1
    assert calls["n"] == 2  # failed once, retried, succeeded


@pytest.mark.asyncio
async def test_download_webp_magic_detected(tmp_path):
    # RIFF....WEBP header → .webp extension even when URL has no suffix.
    webp_bytes = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"rest"

    async def fetch(url: str) -> bytes:
        return webp_bytes

    out = Path(tmp_path) / "img"
    results = await download_images(
        ["http://x/no-ext"], out_dir=out, fetch=fetch, delay=0.0)
    assert results[0]["file_path"].endswith(".webp")
