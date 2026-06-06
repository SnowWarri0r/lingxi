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
