"""Offline one-shot: build the sticker library.

Pipeline: collect image URLs -> download+dedup (crawler) -> caption (vision LLM)
-> store (StickerStore). Run manually, NOT part of the serving loop.

URL collection scrapes doutula.com (斗图啦) search pages for emotion stickers
by keyword. Any URLs placed in SEED_URLS are also included, so you can hand-seed
specific stickers without scraping. Personal-use / low-volume only.

Usage:
    .venv/bin/python tools/crawl_stickers.py
"""

import asyncio
import os
import re
import urllib.parse
from pathlib import Path

import httpx

from lingxi.stickers.crawler import download_images
from lingxi.stickers.captioner import caption_image
from lingxi.stickers.store import StickerStore
from lingxi.stickers.models import Sticker


# Optional hand-seeded direct image URLs, merged with the scraped ones.
SEED_URLS: list[str] = [
]

# Emotion/reaction keywords. doutula's search is single-page per keyword
# (~70 distinct stickers each; the &page param is ignored), so volume comes
# from breadth of keywords, not paging.
STICKER_KEYWORDS: list[str] = [
    "开心", "无语", "委屈", "生气", "害怕",
    "惊讶", "得意", "期待", "哭", "尴尬",
    "笑哭", "摸鱼", "好奇", "困", "谢谢",
]

DATA_DIR = os.environ.get("MEMORY_DATA_DIR", "./data/memory")
IMG_DIR = Path(DATA_DIR).parent / "stickers" / "img"
DB_PATH = Path(DATA_DIR).parent / "stickers" / "stickers.db"

_DOUTULA_SEARCH = "https://www.doutula.com/search?keyword={kw}"
_DOUTULA_CDN = "img.doutupk.com"  # primary CDN; OSS backup is byte-identical
_DATA_ORIGINAL = re.compile(r'data-original="([^"]+)"')
_SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://www.doutula.com/",
}


async def _scrape_doutula(
    keywords: list[str], per_keyword: int = 40, delay: float = 1.5,
) -> list[str]:
    """Scrape doutula.com search pages for emotion sticker image URLs.

    Extracts the lazy-load `data-original` image URLs (primary CDN only — the
    OSS backup is byte-identical and would dedup anyway). verify=False because
    doutula serves an expired TLS cert; we only read public HTML.
    """
    seen_basename: set[str] = set()
    out: list[str] = []
    async with httpx.AsyncClient(
        timeout=25, follow_redirects=True, verify=False, headers=_SCRAPE_HEADERS
    ) as client:
        for i, kw in enumerate(keywords):
            if i > 0 and delay > 0:
                await asyncio.sleep(delay)
            url = _DOUTULA_SEARCH.format(kw=urllib.parse.quote(kw))
            try:
                resp = await client.get(url)
                resp.raise_for_status()
            except Exception as e:  # noqa: BLE001 — skip a bad keyword, keep going
                print(f"[collect] keyword {kw!r} failed: {e}")
                continue
            count = 0
            for img_url in _DATA_ORIGINAL.findall(resp.text):
                if not img_url or img_url.endswith("null") or "loader.gif" in img_url:
                    continue
                if _DOUTULA_CDN not in img_url:  # skip OSS backup (identical bytes)
                    continue
                basename = img_url.rsplit("/", 1)[-1]
                if basename in seen_basename:
                    continue
                seen_basename.add(basename)
                out.append(img_url)
                count += 1
                if per_keyword and count >= per_keyword:
                    break
            print(f"[collect] {kw}: +{count} (total {len(out)})")
    return out


async def collect_urls() -> list[str]:
    """Return image URLs to download: hand-seeded SEED_URLS + scraped doutula."""
    urls = list(SEED_URLS)
    urls.extend(await _scrape_doutula(STICKER_KEYWORDS))
    seen: set[str] = set()
    deduped: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


async def main() -> None:
    store = StickerStore(DB_PATH)
    await store.init()

    urls = await collect_urls()
    if not urls:
        print("No URLs to crawl. Populate SEED_URLS or collect_urls(), "
              "or hand-place images in IMG_DIR and adapt this script.")
        return

    print(f"Downloading {len(urls)} candidate images -> {IMG_DIR}")
    downloaded = await download_images(urls, out_dir=IMG_DIR, delay=1.0)
    print(f"{len(downloaded)} distinct images stored.")

    provider = await _build_provider()
    added = 0
    failed = 0
    for item in downloaded:
        try:
            tags = await caption_image(provider, item["file_path"])
            ok = await store.add(Sticker(
                file_path=item["file_path"],
                source_url=item["source_url"],
                content_hash=item["content_hash"],
                caption=tags["caption"],
                emotion=tags["emotion"],
                tags=tags["tags"],
                when_to_use=tags["when_to_use"],
            ))
        except Exception as e:
            failed += 1
            print(f"  [!] caption/store failed for {item['file_path']}: {e}")
            continue
        if ok:
            added += 1
        print(f"  [{'+' if ok else 'dup'}] {tags['caption']!r} {tags['tags']}")
    print(f"Done. {added} new stickers stored, {failed} failed, in {DB_PATH}")


async def _build_provider():
    """Build the vision-capable LLM provider the same way the app does.

    Mirrors `lingxi.app.create_engine`: load config, register the provider
    defaults, build the AuthManager (reusing app.py's helper so OAuth / profile
    wiring stays in one place), then resolve the provider via the registry.
    The returned provider exposes `.complete(...)`, which is what
    `caption_image` calls.
    """
    from lingxi.app import _build_auth_manager
    from lingxi.auth.models import AuthMethod
    from lingxi.providers.registry import ProviderRegistry
    from lingxi.utils.config import load_config, get_nested

    config = load_config(os.environ.get("LINGXI_CONFIG", "config/default.yaml"))

    ProviderRegistry.register_defaults()
    provider_name = get_nested(config, "llm", "provider", default="claude")
    model = get_nested(config, "llm", "model", default="claude-sonnet-4-20250514")
    auth_method = AuthMethod(
        get_nested(config, "llm", "auth_method", default="oauth_pkce")
    )

    auth_manager = _build_auth_manager(config)
    return await ProviderRegistry.create_llm_with_auth(
        provider_name,
        auth_manager=auth_manager,
        auth_method=auth_method,
        model=model,
    )


if __name__ == "__main__":
    asyncio.run(main())
