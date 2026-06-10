"""SQLite-backed sticker library with FTS5 (trigram) search.

Mirrors the structure of facts/store.py: WAL mode, schema applied
unconditionally on init() via IF NOT EXISTS, sqlite calls dispatched to a
thread pool. Simpler than FactStore — no supersede/importance, just a
content_hash unique key for idempotent crawl re-runs.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from lingxi.stickers.models import Sticker


_SCHEMA = """
CREATE TABLE IF NOT EXISTS stickers (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    source_url TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL UNIQUE,
    caption TEXT NOT NULL DEFAULT '',
    emotion TEXT NOT NULL DEFAULT '',
    tags_json TEXT NOT NULL DEFAULT '[]',
    when_to_use TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS stickers_fts
    USING fts5(caption, tags, emotion, when_to_use,
               tokenize='trigram');
"""


def _row_to_sticker(row: sqlite3.Row) -> Sticker:
    return Sticker(
        id=row["id"],
        file_path=row["file_path"],
        source_url=row["source_url"],
        content_hash=row["content_hash"],
        caption=row["caption"],
        emotion=row["emotion"],
        tags=json.loads(row["tags_json"]),
        when_to_use=row["when_to_use"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


class StickerStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = asyncio.Lock()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self._path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        return c

    async def init(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        def _setup():
            c = self._conn()
            c.executescript(_SCHEMA)
            # Semantic-search vectors (one per sticker). Kept in a sidecar table
            # so re-captioning / FTS rebuilds don't disturb it.
            c.execute(
                "CREATE TABLE IF NOT EXISTS sticker_vecs ("
                "sticker_id TEXT PRIMARY KEY, vec TEXT NOT NULL)")
            c.commit()
            c.close()

        await asyncio.to_thread(_setup)

    async def set_embedding(self, sticker_id: str, vec: list[float]) -> None:
        def _w():
            c = self._conn()
            c.execute(
                "INSERT OR REPLACE INTO sticker_vecs(sticker_id, vec) VALUES (?, ?)",
                (sticker_id, json.dumps(vec)))
            c.commit()
            c.close()
        async with self._lock:
            await asyncio.to_thread(_w)

    async def has_vectors(self) -> bool:
        def _r():
            c = self._conn()
            try:
                n = c.execute("SELECT COUNT(*) FROM sticker_vecs").fetchone()[0]
            except sqlite3.OperationalError:
                n = 0
            c.close()
            return n
        return (await asyncio.to_thread(_r)) > 0

    async def search_semantic(self, query_vec: list[float], k: int = 5) -> list[Sticker]:
        """Cosine-similarity match over precomputed sticker vectors. Robust to
        free-form mood queries where FTS keyword overlap fails ('想抱抱' has no
        shared token with a 撒娇 sticker, but is semantically close)."""
        def _r():
            c = self._conn()
            rows = c.execute(
                "SELECT s.*, v.vec AS vec FROM stickers s "
                "JOIN sticker_vecs v ON v.sticker_id = s.id").fetchall()
            c.close()
            return rows
        rows = await asyncio.to_thread(_r)
        if not rows:
            return []
        import math
        qn = math.sqrt(sum(x * x for x in query_vec)) or 1.0

        def score(vec_json: str) -> float:
            v = json.loads(vec_json)
            dot = sum(a * b for a, b in zip(query_vec, v))
            vn = math.sqrt(sum(b * b for b in v)) or 1.0
            return dot / (qn * vn)

        ranked = sorted(rows, key=lambda r: score(r["vec"]), reverse=True)
        return [_row_to_sticker(r) for r in ranked[:k]]

    async def add(self, sticker: Sticker) -> bool:
        """Insert a sticker. Returns False (skips) if content_hash already
        present, so crawl re-runs are idempotent."""
        def _write() -> bool:
            c = self._conn()
            exists = c.execute(
                "SELECT 1 FROM stickers WHERE content_hash = ?",
                (sticker.content_hash,),
            ).fetchone()
            if exists:
                c.close()
                return False
            c.execute(
                """INSERT INTO stickers
                   (id, file_path, source_url, content_hash, caption,
                    emotion, tags_json, when_to_use, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    sticker.id, sticker.file_path, sticker.source_url,
                    sticker.content_hash, sticker.caption, sticker.emotion,
                    json.dumps(sticker.tags, ensure_ascii=False),
                    sticker.when_to_use, sticker.created_at.isoformat(),
                ),
            )
            rowid = c.execute(
                "SELECT rowid FROM stickers WHERE id = ?", (sticker.id,)
            ).fetchone()[0]
            c.execute(
                "INSERT INTO stickers_fts(rowid, caption, tags, emotion, when_to_use) "
                "VALUES (?, ?, ?, ?, ?)",
                (rowid, sticker.caption, " ".join(sticker.tags),
                 sticker.emotion, sticker.when_to_use),
            )
            c.commit()
            c.close()
            return True

        async with self._lock:
            return await asyncio.to_thread(_write)

    async def get(self, sticker_id: str) -> Sticker | None:
        def _read():
            c = self._conn()
            row = c.execute(
                "SELECT * FROM stickers WHERE id = ?", (sticker_id,)
            ).fetchone()
            c.close()
            return row

        row = await asyncio.to_thread(_read)
        return _row_to_sticker(row) if row else None

    async def search(self, query: str, k: int = 5) -> list[Sticker]:
        """Find stickers by freeform query.

        Two passes: (1) a precise pass — FTS5 phrase MATCH for >=3-char queries,
        LIKE for shorter ones; (2) if that finds nothing, a recall pass that ORs
        the query's 2-char sliding windows via LIKE. CJK has no word boundaries,
        so the bigram windows approximate segmentation: "摸鱼累了" then still
        finds a sticker tagged "摸鱼". A roughly-relevant sticker beats none for
        a picker that randomly selects among the results.
        """
        query = (query or "").strip()
        if not query:
            return []
        hits = await self._search_precise(query, k)
        if hits:
            return hits
        return await self._search_windows(query, k)

    async def _search_precise(self, query: str, k: int) -> list[Sticker]:
        """Precise pass: LIKE for <3-char queries, FTS5 phrase MATCH otherwise."""
        if len(query) < 3:
            pattern = f"%{query}%"
            sql = (
                "SELECT * FROM stickers "
                "WHERE caption LIKE ? OR tags_json LIKE ? "
                "OR emotion LIKE ? OR when_to_use LIKE ? "
                "ORDER BY created_at DESC LIMIT ?"
            )

            def _read_like():
                c = self._conn()
                rows = c.execute(
                    sql, (pattern, pattern, pattern, pattern, k)
                ).fetchall()
                c.close()
                return rows

            rows = await asyncio.to_thread(_read_like)
        else:
            sql = (
                "SELECT s.* FROM stickers s "
                "JOIN stickers_fts fts ON s.rowid = fts.rowid "
                "WHERE stickers_fts MATCH ? "
                "ORDER BY bm25(stickers_fts) LIMIT ?"
            )

            # Escape embedded double-quotes (double them) then wrap as a phrase
            # literal so arbitrary user text is treated as a literal token
            # sequence — FTS5 MATCH otherwise chokes on operators/punctuation.
            safe_query = query.replace('"', '""')

            def _read():
                c = self._conn()
                try:
                    rows = c.execute(sql, (f'"{safe_query}"', k)).fetchall()
                except sqlite3.OperationalError:
                    rows = []
                c.close()
                return rows

            rows = await asyncio.to_thread(_read)
        return [_row_to_sticker(r) for r in rows]

    async def _search_windows(self, query: str, k: int) -> list[Sticker]:
        """Recall pass: OR the 2-char sliding windows of the query via LIKE."""
        if len(query) >= 2:
            windows: list[str] = []
            for i in range(len(query) - 1):
                w = query[i:i + 2]
                if w not in windows:
                    windows.append(w)
        else:
            windows = [query]
        if not windows:
            return []
        clauses: list[str] = []
        params: list = []
        for w in windows:
            p = f"%{w}%"
            clauses.append(
                "(caption LIKE ? OR tags_json LIKE ? "
                "OR emotion LIKE ? OR when_to_use LIKE ?)")
            params.extend([p, p, p, p])
        sql = (
            "SELECT * FROM stickers WHERE " + " OR ".join(clauses) +
            " ORDER BY created_at DESC LIMIT ?"
        )
        params.append(k)

        def _read():
            c = self._conn()
            rows = c.execute(sql, params).fetchall()
            c.close()
            return rows

        rows = await asyncio.to_thread(_read)
        return [_row_to_sticker(r) for r in rows]
