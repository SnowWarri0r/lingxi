"""Precompute a semantic embedding per sticker so the engine can match a
free-form mood query by meaning (not FTS keyword overlap). Embeds each
sticker's emotional/usage text and stores the vector in the sidecar table.

Run after (re)captioning. Usage: .venv/bin/python tools/embed_stickers.py
"""

import asyncio
import json
import os
import sqlite3

from lingxi.providers.embedding import DoubaoEmbeddingProvider
from lingxi.stickers.store import StickerStore

DB = "data/stickers/stickers.db"


def _emb_text(emotion: str, tags_json: str, when_to_use: str) -> str:
    tags = " ".join(json.loads(tags_json or "[]"))
    # Lead with the emotion (doubled) — the query is a mood word, so the
    # emotion carries most of the matching signal; tags/usage add nuance.
    return f"{emotion} {emotion} {tags} {when_to_use}".strip()


async def main():
    rows = sqlite3.connect(DB).execute(
        "SELECT id, emotion, tags_json, when_to_use FROM stickers").fetchall()
    print(f"[embed] {len(rows)} stickers", flush=True)

    emb = DoubaoEmbeddingProvider(model=os.environ["EMBEDDING_MODEL"])
    store = StickerStore(DB)
    await store.init()

    sem = asyncio.Semaphore(8)
    n = {"done": 0}

    async def one(sid, emotion, tags_json, when):
        async with sem:
            vec = await emb.embed(_emb_text(emotion, tags_json, when))
        await store.set_embedding(sid, vec)
        n["done"] += 1
        if n["done"] % 40 == 0:
            print(f"[embed] {n['done']}/{len(rows)}", flush=True)

    await asyncio.gather(*(one(*r) for r in rows))
    print(f"[embed] DONE — {n['done']} vectors written", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
