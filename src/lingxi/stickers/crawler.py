"""Download sticker images, dedup by content hash.

Deliberately site-agnostic: download_images takes a list of image URLs and
persists distinct bytes to disk. The site-specific step (keyword → image URLs)
lives in the offline build script, so swapping/adding a source or falling back
to a hand-seeded URL list never touches this module.

HTTP fetch is an injectable async callable (default httpx) so tests run without
network.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Awaitable, Callable


async def _httpx_fetch(url: str) -> bytes:
    import httpx
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


_EXT_BY_MAGIC = [
    (b"\x89PNG", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF8", ".gif"),
    (b"RIFF", ".webp"),  # RIFF....WEBP
]


def _guess_ext(data: bytes, url: str) -> str:
    for magic, ext in _EXT_BY_MAGIC:
        if data.startswith(magic):
            return ext
    for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        if url.lower().endswith(ext):
            return ext
    return ".png"


async def download_images(
    urls: list[str],
    *,
    out_dir: str | Path,
    fetch: Callable[[str], Awaitable[bytes]] = _httpx_fetch,
    delay: float = 1.0,
    retries: int = 1,
) -> list[dict]:
    """Download each URL, dedup by sha256, write distinct images to out_dir.

    Returns a list of {file_path, source_url, content_hash} for newly stored
    images. Failed fetches are skipped (logged), the sleep between requests is
    a politeness rate-limit.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    results: list[dict] = []

    for i, url in enumerate(urls):
        if i > 0 and delay > 0:
            await asyncio.sleep(delay)
        data: bytes | None = None
        for attempt in range(retries + 1):
            try:
                data = await fetch(url)
                break
            except Exception as e:  # noqa: BLE001 — skip & continue is the policy
                if attempt >= retries:
                    print(f"[crawler] fetch failed {url}: {e}")
                    data = None
                else:
                    await asyncio.sleep(delay)
        if not data:
            continue
        h = hashlib.sha256(data).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        path = out / f"{h}{_guess_ext(data, url)}"
        path.write_bytes(data)
        results.append({
            "file_path": str(path), "source_url": url, "content_hash": h})

    return results
