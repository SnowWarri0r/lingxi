"""Re-caption every sticker in the library with the current captioner prompt
and rebuild the store, so improved tagging propagates to existing stickers.

Reads the existing DB for (id, file_path, content_hash, source_url), re-captions
each image via the vision provider, and writes a FRESH DB (so store.add() keeps
the FTS index correct). On a caption failure/empty it KEEPS the old metadata so
a flaky image never blanks a sticker. Output goes to <db>.rebuilt — the caller
verifies, then swaps it in.

Usage: .venv/bin/python tools/recaption_stickers.py [--concurrency N]
"""

import asyncio
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from lingxi.providers.claude import ClaudeProvider
from lingxi.stickers.captioner import caption_image
from lingxi.stickers.models import Sticker
from lingxi.stickers.store import StickerStore

DB = "data/stickers/stickers.db"
OUT = "data/stickers/stickers.db.rebuilt"


def _oauth_token() -> str:
    out = subprocess.run(
        ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
        capture_output=True, text=True, timeout=10).stdout.strip()
    d = json.loads(out)
    for k in ("claudeAiOauth", "oauthAccount"):
        if k in d:
            d = d[k]
            break
    return d.get("accessToken") or d.get("access_token")


async def main():
    concurrency = 8
    if "--concurrency" in sys.argv:
        concurrency = int(sys.argv[sys.argv.index("--concurrency") + 1])

    rows = sqlite3.connect(DB).execute(
        "SELECT id, file_path, content_hash, source_url, caption, emotion, "
        "tags_json, when_to_use FROM stickers").fetchall()
    print(f"[recaption] {len(rows)} stickers, concurrency={concurrency}", flush=True)

    llm = ClaudeProvider(api_key=_oauth_token(), model="claude-sonnet-4-20250514")

    Path(OUT).unlink(missing_ok=True)
    store = StickerStore(OUT)
    await store.init()

    sem = asyncio.Semaphore(concurrency)
    done = {"n": 0, "recap": 0, "kept": 0}

    async def one(row):
        sid, path, chash, surl, ocap, oemo, otags_json, owhen = row
        async with sem:
            tags = await caption_image(llm, path)
        # caption_image returns {} fields on failure → keep old metadata
        cap = (tags.get("caption") or "").strip() or ocap
        emo = (tags.get("emotion") or "").strip() or oemo
        tag_list = tags.get("tags") or json.loads(otags_json or "[]")
        when = (tags.get("when_to_use") or "").strip() or owhen
        recapped = bool((tags.get("caption") or "").strip())
        await store.add(Sticker(
            id=sid, file_path=path, content_hash=chash, source_url=surl or "",
            caption=cap, emotion=emo, tags=tag_list, when_to_use=when))
        done["n"] += 1
        done["recap" if recapped else "kept"] += 1
        if done["n"] % 20 == 0:
            print(f"[recaption] {done['n']}/{len(rows)} "
                  f"(recap={done['recap']} kept-old={done['kept']})", flush=True)

    await asyncio.gather(*(one(r) for r in rows))
    print(f"[recaption] DONE → {OUT} "
          f"(recap={done['recap']} kept-old={done['kept']})", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
