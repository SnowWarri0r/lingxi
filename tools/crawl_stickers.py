"""Offline one-shot: build the sticker library.

Pipeline: collect image URLs -> download+dedup (crawler) -> caption (vision LLM)
-> store (StickerStore). Run manually, NOT part of the serving loop.

The site-specific URL collection (keyword -> image URLs) is intentionally a
small, swappable function below. SP1 ships with a SEED_URLS fallback so the
library can be built without committing to a specific site's scraping rules;
fill `collect_urls` with a real source when one is validated.

Usage:
    .venv/bin/python tools/crawl_stickers.py
"""

import asyncio
import os
from pathlib import Path

from lingxi.stickers.crawler import download_images
from lingxi.stickers.captioner import caption_image
from lingxi.stickers.store import StickerStore
from lingxi.stickers.models import Sticker


# Replace with real source scraping once a site is validated. Until then, drop
# direct image URLs (or hand-place files and skip crawling) here.
SEED_URLS: list[str] = [
]

DATA_DIR = os.environ.get("MEMORY_DATA_DIR", "./data/memory")
IMG_DIR = Path(DATA_DIR).parent / "stickers" / "img"
DB_PATH = Path(DATA_DIR).parent / "stickers" / "stickers.db"


async def collect_urls() -> list[str]:
    """Site-specific URL collection. Returns image URLs to download.

    SP1 default: SEED_URLS. Fill this with real scraping when a target site is
    chosen and verified."""
    return SEED_URLS


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
    for item in downloaded:
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
        if ok:
            added += 1
        print(f"  [{'+' if ok else 'dup'}] {tags['caption']!r} {tags['tags']}")
    print(f"Done. {added} new stickers captioned & stored in {DB_PATH}")


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
