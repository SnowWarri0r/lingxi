"""SQLite-backed Fact store with FTS5 for content search.

Single-file database. WAL mode so multiple processes can read while
one writer commits. Schema migrations are stored in `_apply_schema()`
and run unconditionally on init() — they use IF NOT EXISTS / CREATE
TABLE so re-running is harmless.

All public methods are async because the rest of the codebase is async;
the actual sqlite calls run in a default thread pool via asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from lingxi.facts.models import Fact, FactType, Source


_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    type TEXT NOT NULL,
    ts TEXT NOT NULL,
    written_at TEXT NOT NULL,
    confidence REAL NOT NULL,
    expires_at TEXT,
    tags_json TEXT NOT NULL DEFAULT '[]',
    supersedes TEXT,
    importance INTEGER,
    last_accessed TEXT
);

CREATE INDEX IF NOT EXISTS idx_facts_subject_ts
    ON facts (subject, ts DESC);
CREATE INDEX IF NOT EXISTS idx_facts_source ON facts (source);
CREATE INDEX IF NOT EXISTS idx_facts_type ON facts (type);
CREATE INDEX IF NOT EXISTS idx_facts_expires ON facts (expires_at);
CREATE INDEX IF NOT EXISTS idx_facts_supersedes ON facts (supersedes);
CREATE INDEX IF NOT EXISTS idx_facts_importance
    ON facts (importance) WHERE importance IS NOT NULL;

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content='facts', content_rowid='rowid',
               tokenize='trigram');
"""


def _row_to_fact(row: sqlite3.Row) -> Fact:
    keys = row.keys()
    return Fact(
        id=row["id"],
        subject=row["subject"],
        content=row["content"],
        source=Source(row["source"]),
        type=FactType(row["type"]),
        ts=datetime.fromisoformat(row["ts"]),
        written_at=datetime.fromisoformat(row["written_at"]),
        confidence=row["confidence"],
        expires_at=(
            datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None
        ),
        tags=json.loads(row["tags_json"]),
        supersedes=row["supersedes"],
        importance=row["importance"] if "importance" in keys else None,
        last_accessed=(
            datetime.fromisoformat(row["last_accessed"])
            if "last_accessed" in keys and row["last_accessed"] else None
        ),
    )


class FactStore:
    def __init__(self, db_path: Path | str):
        self._path = Path(db_path)
        self._lock = asyncio.Lock()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self._path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        return c

    async def init(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

        def _setup():
            c = self._conn()
            c.executescript(_SCHEMA)
            # Schema migration: add importance + last_accessed if missing (existing DBs)
            cols = {row[1] for row in c.execute("PRAGMA table_info(facts)")}
            if "importance" not in cols:
                c.execute("ALTER TABLE facts ADD COLUMN importance INTEGER")
            if "last_accessed" not in cols:
                c.execute("ALTER TABLE facts ADD COLUMN last_accessed TEXT")
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_importance "
                "ON facts (importance) WHERE importance IS NOT NULL"
            )
            c.commit()
            c.close()

        await asyncio.to_thread(_setup)

    async def write(self, fact: Fact) -> None:
        def _write():
            c = self._conn()
            c.execute(
                """INSERT INTO facts
                   (id, subject, content, source, type, ts, written_at,
                    confidence, expires_at, tags_json, supersedes,
                    importance, last_accessed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fact.id, fact.subject, fact.content,
                    fact.source.value, fact.type.value,
                    fact.ts.isoformat(), fact.written_at.isoformat(),
                    fact.confidence,
                    fact.expires_at.isoformat() if fact.expires_at else None,
                    json.dumps(fact.tags, ensure_ascii=False),
                    fact.supersedes,
                    fact.importance,
                    fact.last_accessed.isoformat() if fact.last_accessed else None,
                ),
            )
            # Mirror into FTS5
            rowid = c.execute(
                "SELECT rowid FROM facts WHERE id = ?", (fact.id,)
            ).fetchone()[0]
            c.execute(
                "INSERT INTO facts_fts(rowid, content, tags) VALUES (?, ?, ?)",
                (rowid, fact.content, " ".join(fact.tags)),
            )
            c.commit()
            c.close()

        async with self._lock:
            await asyncio.to_thread(_write)

    async def get(self, fact_id: str) -> Fact | None:
        def _read():
            c = self._conn()
            row = c.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
            c.close()
            return row

        row = await asyncio.to_thread(_read)
        return _row_to_fact(row) if row else None

    async def query(
        self,
        *,
        subject: str | None = None,
        type: FactType | None = None,
        since: datetime | None = None,
        limit: int = 100,
        exclude_superseded: bool = True,
        include_expired: bool = False,
    ) -> list[Fact]:
        clauses: list[str] = ["1=1"]
        params: list = []
        if subject:
            clauses.append("subject = ?")
            params.append(subject)
        if type:
            clauses.append("type = ?")
            params.append(type.value)
        if since:
            clauses.append("ts >= ?")
            params.append(since.isoformat())
        if not include_expired:
            clauses.append("(expires_at IS NULL OR expires_at > ?)")
            params.append(datetime.now().isoformat())
        if exclude_superseded:
            clauses.append(
                "id NOT IN (SELECT supersedes FROM facts WHERE supersedes IS NOT NULL)"
            )

        sql = (
            f"SELECT * FROM facts WHERE {' AND '.join(clauses)} "
            f"ORDER BY ts DESC LIMIT ?"
        )
        params.append(limit)

        def _read():
            c = self._conn()
            rows = c.execute(sql, params).fetchall()
            c.close()
            return rows

        rows = await asyncio.to_thread(_read)
        return [_row_to_fact(r) for r in rows]

    async def search_fts(self, query: str, limit: int = 20) -> list[Fact]:
        # FTS5 trigram tokenizer requires a minimum token length of 3 characters.
        # Queries shorter than 3 chars (common with 2-char CJK terms) fall back to
        # a LIKE scan so that short but meaningful search terms still work.
        if len(query) < 3:
            sql = (
                "SELECT * FROM facts "
                "WHERE content LIKE ? OR tags_json LIKE ? "
                "ORDER BY ts DESC LIMIT ?"
            )
            pattern = f"%{query}%"

            def _read_like():
                c = self._conn()
                rows = c.execute(sql, (pattern, pattern, limit)).fetchall()
                c.close()
                return rows

            rows = await asyncio.to_thread(_read_like)
        else:
            sql = (
                "SELECT f.* FROM facts f "
                "JOIN facts_fts fts ON f.rowid = fts.rowid "
                "WHERE facts_fts MATCH ? "
                "ORDER BY rank LIMIT ?"
            )

            def _read():
                c = self._conn()
                rows = c.execute(sql, (query, limit)).fetchall()
                c.close()
                return rows

            rows = await asyncio.to_thread(_read)

        return [_row_to_fact(r) for r in rows]

    async def fts_rank(self, query: str, ids: list[str]) -> dict[str, float]:
        """Return {id: normalized_rank in [0,1]} for ids matching FTS query.

        Missing ids get 0.0. Higher = more relevant.
        bm25() in FTS5 returns negative scores; more negative = better match.
        We invert the sign then min-max normalize to [0, 1].
        """
        if not ids:
            return {}

        def _read():
            c = self._conn()
            placeholders = ",".join("?" * len(ids))
            sql = (
                "SELECT facts.id, bm25(facts_fts) AS rank "
                "FROM facts_fts JOIN facts ON facts_fts.rowid = facts.rowid "
                f"WHERE facts_fts MATCH ? AND facts.id IN ({placeholders})"
            )
            rows = c.execute(sql, (query, *ids)).fetchall()
            c.close()
            return rows

        try:
            rows = await asyncio.to_thread(_read)
        except sqlite3.OperationalError:
            # FTS5 syntax error (e.g. user query contained special chars)
            return {fid: 0.0 for fid in ids}

        if not rows:
            return {fid: 0.0 for fid in ids}

        # bm25 returns negative scores; smaller (more negative) = better match.
        # Invert sign, then min-max normalize to [0, 1].
        raw = {row["id"]: -row["rank"] for row in rows}
        result = {fid: 0.0 for fid in ids}
        if len(raw) == 1:
            result.update({k: 1.0 for k in raw})
            return result
        lo, hi = min(raw.values()), max(raw.values())
        span = hi - lo if hi > lo else 1.0
        for k, v in raw.items():
            result[k] = (v - lo) / span
        return result

    async def update_last_accessed(self, ids: list[str], ts: datetime) -> None:
        """Set last_accessed = ts for all given fact ids."""
        if not ids:
            return

        def _write():
            c = self._conn()
            placeholders = ",".join("?" * len(ids))
            sql = f"UPDATE facts SET last_accessed = ? WHERE id IN ({placeholders})"
            c.execute(sql, (ts.isoformat(), *ids))
            c.commit()
            c.close()

        async with self._lock:
            await asyncio.to_thread(_write)

    async def count_by_subject(self) -> dict[str, int]:
        def _read():
            c = self._conn()
            rows = c.execute(
                "SELECT subject, COUNT(*) AS n FROM facts GROUP BY subject"
            ).fetchall()
            c.close()
            return rows

        rows = await asyncio.to_thread(_read)
        return {r["subject"]: r["n"] for r in rows}
