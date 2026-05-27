# Facts Architecture Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 5 independent stores + 11+ prompt sections with one unified Fact table + Sonnet-based orchestrator + 3-section dynamic renderer. Branch-based; old code deleted at end.

**Architecture:** SQLite-backed Fact table (subject/source/type/confidence/supersedes) → 6 Writers with strict subject ownership → pre-turn Sonnet Orchestrator decides register + fact queries → Renderer emits 3 dynamic blocks. See spec: `docs/superpowers/specs/2026-05-27-facts-arch-refactor-design.md`.

**Tech Stack:** Python 3.12, SQLite (stdlib) + FTS5, Anthropic Sonnet 4 API, Pydantic, pytest-asyncio.

---

## File Structure

**New (create):**

```
src/lingxi/facts/
├── __init__.py
├── models.py              # Source, FactType, Fact (pydantic)
├── store.py               # SQLite store with FTS5, async wrapper
├── retriever.py           # FactQuery + FactRetriever
└── writers/
    ├── __init__.py
    ├── base.py            # WriterBase with source/subject enforcement
    ├── life.py            # LifeWriter
    ├── npc.py             # NPCWriter
    ├── user_statement.py  # UserStatementWriter
    ├── inference.py       # InferenceWriter
    ├── world.py           # WorldWriter
    └── biography.py       # BiographyLoader

src/lingxi/brain/
├── __init__.py
├── models.py              # OrchestrationDecision, FactQuery
├── orchestrator.py        # Sonnet pre-turn decision call
└── renderer.py            # 3 dynamic blocks + persona static block

tools/
└── migrate_to_facts.py    # one-shot migration of all existing stores

tests/test_facts/
├── __init__.py
├── test_models.py
├── test_store.py
├── test_retriever.py
└── test_writers/
    ├── __init__.py
    ├── test_base.py
    ├── test_life.py
    ├── test_npc.py
    ├── test_user_statement.py
    ├── test_inference.py
    ├── test_world.py
    └── test_biography.py

tests/test_brain/
├── __init__.py
├── test_orchestrator.py
└── test_renderer.py
```

**Modify:**
- `src/lingxi/conversation/engine.py` — replace `_prepare_turn` with new orchestrator+renderer call
- `src/lingxi/persona/prompt_builder.py` — extract `_build_identity_section` / `_build_personality_section` / `_build_speaking_style_section` / `_build_message_habits_section` to a `build_persona_block()` function, delete the rest in P7
- `src/lingxi/inner_life/simulator.py` — `_record_event` calls `LifeWriter.write()`
- `src/lingxi/social/scheduler.py` — `_tick_one_npc` calls `NPCWriter.write()`
- `src/lingxi/relational/extractor.py` — extraction results go to `InferenceWriter.write()` instead of `RelationalMemoryStore`
- `src/lingxi/world/scheduler.py` — `_maybe_fetch_today` writes via `WorldWriter`
- `src/lingxi/app.py` — bootstrap Facts store + writers, wire to engine

**Delete (P7):**
- `src/lingxi/inner_life/store.py` (move `current_activity`/`today_plan` to a tiny `inner_life/state.py`)
- `src/lingxi/relational/store.py`
- `src/lingxi/social/store.py` (keep `social/loader.py` for yaml + arcs.json — arcs are config-derived state)
- `src/lingxi/social/promoter.py`
- `src/lingxi/world/store.py`

---

## Setup: Worktree

- [ ] **Step 0.1: Create isolated worktree**

If executing this plan, first create the worktree per the using-git-worktrees skill:

```bash
git worktree add ../agent-facts-refactor refactor/facts-arch
cd ../agent-facts-refactor
```

All subsequent commands run from the worktree directory.

---

## P0 — Data Skeleton (~1 day)

### Task 0.1: Fact models

**Files:**
- Create: `src/lingxi/facts/__init__.py` (empty)
- Create: `src/lingxi/facts/models.py`
- Test: `tests/test_facts/test_models.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_facts/__init__.py` (empty) and:

```python
# tests/test_facts/test_models.py
from datetime import datetime, timedelta

import pytest

from lingxi.facts.models import Fact, FactType, Source


class TestSource:
    def test_default_confidence_per_source(self):
        assert Source.USER_STATED.default_confidence == 1.0
        assert Source.LIFE_SIMULATED.default_confidence == 0.8
        assert Source.NPC_TICKER.default_confidence == 0.8
        assert Source.LLM_INFERRED.default_confidence == 0.5
        assert Source.WORLD_FETCH.default_confidence == 0.9
        assert Source.BIOGRAPHY.default_confidence == 1.0


class TestFact:
    def test_basic_construction(self):
        f = Fact(
            subject="aria",
            content="今早煮了泡面",
            source=Source.LIFE_SIMULATED,
            type=FactType.EVENT,
            ts=datetime(2026, 5, 27, 8, 0),
        )
        assert f.id  # auto-generated uuid
        assert f.confidence == 0.8  # default for LIFE_SIMULATED
        assert f.expires_at is None
        assert f.supersedes is None
        assert f.tags == []
        assert isinstance(f.written_at, datetime)

    def test_explicit_confidence_overrides_default(self):
        f = Fact(
            subject="user:oc_x",
            content="工作时间 11-21",
            source=Source.USER_STATED,
            type=FactType.PATTERN,
            ts=datetime.now(),
            confidence=0.95,
        )
        assert f.confidence == 0.95

    def test_subject_format_validation(self):
        # subject must be "aria" | "user:..." | "npc:..." | "world"
        with pytest.raises(ValueError):
            Fact(
                subject="random",
                content="x",
                source=Source.USER_STATED,
                type=FactType.EVENT,
                ts=datetime.now(),
            )

    def test_supersedes_link(self):
        old = Fact(
            subject="aria", content="老版本", source=Source.LLM_INFERRED,
            type=FactType.PATTERN, ts=datetime.now(),
        )
        new = Fact(
            subject="aria", content="新版本", source=Source.USER_STATED,
            type=FactType.PATTERN, ts=datetime.now(),
            supersedes=old.id,
        )
        assert new.supersedes == old.id
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_facts/test_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'lingxi.facts'`

- [ ] **Step 3: Write implementation**

Create `src/lingxi/facts/__init__.py` as empty file, then:

```python
# src/lingxi/facts/models.py
"""Schema for the unified Fact table.

A Fact is the atomic unit of knowledge in the new architecture.
Replaces inner_life events, relational_memory entries, social NPC events,
world briefings, and long-term memory facts in one typed structure.

Subject ownership is the core invariant: each Fact's `subject` identifies
who or what the fact is ABOUT (aria/user:x/npc:y/world). Writers enforce
this — `LifeWriter` only writes subject=aria, `NPCWriter` only writes
subject=npc:*, etc. Subject isolation prevents the cross-contamination
that plagued the old system (e.g. NPC events bleeding into Aria's
self-narrative).
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Source(str, Enum):
    USER_STATED      = "user_stated"
    LIFE_SIMULATED   = "life_simulated"
    NPC_TICKER       = "npc_ticker"
    LLM_INFERRED     = "llm_inferred"
    WORLD_FETCH      = "world_fetch"
    BIOGRAPHY        = "biography"

    @property
    def default_confidence(self) -> float:
        return {
            Source.USER_STATED:    1.0,
            Source.BIOGRAPHY:      1.0,
            Source.WORLD_FETCH:    0.9,
            Source.LIFE_SIMULATED: 0.8,
            Source.NPC_TICKER:     0.8,
            Source.LLM_INFERRED:   0.5,
        }[self]


class FactType(str, Enum):
    EVENT        = "event"
    PATTERN      = "pattern"
    OPINION      = "opinion"
    PLAN         = "plan"
    EMOTION_NOTE = "emotion_note"


_SUBJECT_RE = re.compile(r"^(aria|world|user:[A-Za-z0-9_:-]+|npc:[A-Za-z0-9_-]+)$")


class Fact(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    subject: str
    content: str
    source: Source
    type: FactType
    ts: datetime
    written_at: datetime = Field(default_factory=datetime.now)
    confidence: float | None = None
    expires_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    supersedes: str | None = None

    @field_validator("subject")
    @classmethod
    def _check_subject(cls, v: str) -> str:
        if not _SUBJECT_RE.match(v):
            raise ValueError(
                f"subject must match aria|world|user:X|npc:X, got {v!r}"
            )
        return v

    def model_post_init(self, _ctx) -> None:
        if self.confidence is None:
            self.confidence = self.source.default_confidence
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_facts/test_models.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lingxi/facts/__init__.py src/lingxi/facts/models.py tests/test_facts/
git commit -m "facts: Fact model with Source/FactType enums + subject validation"
```

---

### Task 0.2: SQLite Store

**Files:**
- Create: `src/lingxi/facts/store.py`
- Test: `tests/test_facts/test_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_facts/test_store.py
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.store import FactStore


@pytest.fixture
async def store(tmp_path):
    s = FactStore(tmp_path / "facts.db")
    await s.init()
    return s


def make_fact(**overrides) -> Fact:
    defaults = dict(
        subject="aria",
        content="测试事件",
        source=Source.LIFE_SIMULATED,
        type=FactType.EVENT,
        ts=datetime.now(),
    )
    defaults.update(overrides)
    return Fact(**defaults)


@pytest.mark.asyncio
async def test_init_creates_schema(tmp_path):
    s = FactStore(tmp_path / "facts.db")
    await s.init()
    assert (tmp_path / "facts.db").exists()


@pytest.mark.asyncio
async def test_write_and_read_by_id(store):
    f = make_fact()
    await store.write(f)
    loaded = await store.get(f.id)
    assert loaded is not None
    assert loaded.content == "测试事件"
    assert loaded.subject == "aria"


@pytest.mark.asyncio
async def test_query_by_subject(store):
    await store.write(make_fact(subject="aria", content="aria 事件"))
    await store.write(make_fact(subject="user:u1", content="user 事件"))
    await store.write(make_fact(subject="npc:xiaomin", content="npc 事件"))

    arias = await store.query(subject="aria")
    assert len(arias) == 1
    assert arias[0].content == "aria 事件"


@pytest.mark.asyncio
async def test_query_by_subject_and_type(store):
    await store.write(make_fact(subject="aria", type=FactType.EVENT, content="事件"))
    await store.write(make_fact(subject="aria", type=FactType.PATTERN, content="规律"))

    events = await store.query(subject="aria", type=FactType.EVENT)
    assert len(events) == 1
    assert events[0].type == FactType.EVENT


@pytest.mark.asyncio
async def test_query_orders_by_ts_desc(store):
    base = datetime(2026, 5, 27, 10, 0)
    await store.write(make_fact(ts=base, content="老"))
    await store.write(make_fact(ts=base + timedelta(hours=1), content="新"))

    results = await store.query(subject="aria")
    assert results[0].content == "新"
    assert results[1].content == "老"


@pytest.mark.asyncio
async def test_query_respects_limit(store):
    for i in range(10):
        await store.write(make_fact(content=f"e{i}"))
    results = await store.query(subject="aria", limit=3)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_supersedes_chain_excludes_old(store):
    """Fact superseded by another should not appear in default queries."""
    old = make_fact(content="老说法")
    await store.write(old)
    new = make_fact(content="新说法", supersedes=old.id)
    await store.write(new)

    results = await store.query(subject="aria", exclude_superseded=True)
    contents = [f.content for f in results]
    assert "新说法" in contents
    assert "老说法" not in contents


@pytest.mark.asyncio
async def test_supersedes_chain_includable(store):
    old = make_fact(content="老说法")
    await store.write(old)
    await store.write(make_fact(content="新说法", supersedes=old.id))

    results = await store.query(subject="aria", exclude_superseded=False)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_expired_facts_filtered(store):
    past = make_fact(content="过期", expires_at=datetime.now() - timedelta(hours=1))
    await store.write(past)
    fresh = make_fact(content="新鲜", expires_at=datetime.now() + timedelta(hours=1))
    await store.write(fresh)

    results = await store.query(subject="aria")
    contents = [f.content for f in results]
    assert "新鲜" in contents
    assert "过期" not in contents


@pytest.mark.asyncio
async def test_fts_search(store):
    await store.write(make_fact(content="今天和外婆通电话", tags=["family", "call"]))
    await store.write(make_fact(content="改 paper 改到崩溃", tags=["work"]))

    family_hits = await store.search_fts("外婆")
    assert len(family_hits) == 1
    assert "外婆" in family_hits[0].content

    work_hits = await store.search_fts("paper")
    assert len(work_hits) == 1


@pytest.mark.asyncio
async def test_count_by_subject(store):
    await store.write(make_fact(subject="aria"))
    await store.write(make_fact(subject="aria"))
    await store.write(make_fact(subject="user:u1"))
    counts = await store.count_by_subject()
    assert counts["aria"] == 2
    assert counts["user:u1"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_facts/test_store.py -v
```

Expected: `ModuleNotFoundError: No module named 'lingxi.facts.store'`

- [ ] **Step 3: Write implementation**

```python
# src/lingxi/facts/store.py
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
    supersedes TEXT
);

CREATE INDEX IF NOT EXISTS idx_facts_subject_ts
    ON facts (subject, ts DESC);
CREATE INDEX IF NOT EXISTS idx_facts_source ON facts (source);
CREATE INDEX IF NOT EXISTS idx_facts_type ON facts (type);
CREATE INDEX IF NOT EXISTS idx_facts_expires ON facts (expires_at);
CREATE INDEX IF NOT EXISTS idx_facts_supersedes ON facts (supersedes);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content='facts', content_rowid='rowid',
               tokenize='unicode61');
"""


def _row_to_fact(row: sqlite3.Row) -> Fact:
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
            c.commit()
            c.close()

        await asyncio.to_thread(_setup)

    async def write(self, fact: Fact) -> None:
        def _write():
            c = self._conn()
            c.execute(
                """INSERT INTO facts
                   (id, subject, content, source, type, ts, written_at,
                    confidence, expires_at, tags_json, supersedes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fact.id, fact.subject, fact.content,
                    fact.source.value, fact.type.value,
                    fact.ts.isoformat(), fact.written_at.isoformat(),
                    fact.confidence,
                    fact.expires_at.isoformat() if fact.expires_at else None,
                    json.dumps(fact.tags, ensure_ascii=False),
                    fact.supersedes,
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_facts/test_store.py -v
```

Expected: 10 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lingxi/facts/store.py tests/test_facts/test_store.py
git commit -m "facts: SQLite store with FTS5, supersedes chain, expiry filter"
```

---

### Task 0.3: FactRetriever (query catalog + fetch)

**Files:**
- Create: `src/lingxi/facts/retriever.py`
- Test: `tests/test_facts/test_retriever.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_facts/test_retriever.py
from datetime import datetime, timedelta

import pytest

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.retriever import FactQuery, FactRetriever
from lingxi.facts.store import FactStore


@pytest.fixture
async def populated(tmp_path):
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    now = datetime.now()
    # aria lived events
    for i in range(3):
        await store.write(Fact(
            subject="aria", content=f"aria 事件 {i}",
            source=Source.LIFE_SIMULATED, type=FactType.EVENT,
            ts=now - timedelta(hours=i),
        ))
    # user patterns
    await store.write(Fact(
        subject="user:u1", content="工作 11-21",
        source=Source.USER_STATED, type=FactType.PATTERN,
        ts=now,
    ))
    # npc events
    await store.write(Fact(
        subject="npc:xiaomin", content="小敏改 paper",
        source=Source.NPC_TICKER, type=FactType.EVENT,
        ts=now,
    ))
    return FactRetriever(store)


@pytest.mark.asyncio
async def test_fetch_by_subject(populated):
    facts = await populated.fetch(FactQuery(subject="aria", limit=5))
    assert len(facts) == 3
    assert all(f.subject == "aria" for f in facts)


@pytest.mark.asyncio
async def test_fetch_by_subject_and_type(populated):
    facts = await populated.fetch(
        FactQuery(subject="aria", type=FactType.EVENT, limit=2)
    )
    assert len(facts) == 2


@pytest.mark.asyncio
async def test_fetch_with_semantic_query(populated):
    facts = await populated.fetch(
        FactQuery(subject="aria", semantic="事件", limit=5)
    )
    # FTS5 match on "事件"
    assert len(facts) >= 1
    assert all("事件" in f.content for f in facts)


@pytest.mark.asyncio
async def test_catalog_returns_counts_per_bucket(populated):
    """Catalog gives orchestrator a count summary without content,
    so it can decide which buckets to query without blowing context."""
    catalog = await populated.catalog()
    assert catalog["aria.event"] == 3
    assert catalog["user:u1.pattern"] == 1
    assert catalog["npc:xiaomin.event"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_facts/test_retriever.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write implementation**

```python
# src/lingxi/facts/retriever.py
"""Fact retrieval interface used by both the Orchestrator (catalog,
counts only) and the Renderer (full fetch).

FactQuery is a lightweight dataclass that captures the orchestrator's
intent: "give me up to N facts for subject S of type T, optionally
matching semantic keyword K".
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from lingxi.facts.models import Fact, FactType
from lingxi.facts.store import FactStore


@dataclass
class FactQuery:
    subject: str
    type: FactType | None = None
    since: datetime | None = None
    semantic: str | None = None  # optional FTS keyword
    limit: int = 5


class FactRetriever:
    def __init__(self, store: FactStore):
        self._store = store

    async def fetch(self, query: FactQuery) -> list[Fact]:
        if query.semantic:
            # FTS5 path: search content, then filter by subject/type in Python.
            # FTS5 index doesn't span structured fields cheaply.
            candidates = await self._store.search_fts(
                query.semantic, limit=query.limit * 4
            )
            filtered = [
                f for f in candidates
                if f.subject == query.subject
                and (query.type is None or f.type == query.type)
                and (query.since is None or f.ts >= query.since)
            ]
            return filtered[: query.limit]

        return await self._store.query(
            subject=query.subject,
            type=query.type,
            since=query.since,
            limit=query.limit,
        )

    async def catalog(self) -> dict[str, int]:
        """Return {bucket: count} for orchestrator's decision input.

        Bucket key format: "<subject>.<type>" — e.g. "aria.event",
        "user:oc_xxx.pattern", "npc:xiaomin.event".
        """
        all_facts = await self._store.query(subject=None, limit=10000)
        counts: dict[str, int] = defaultdict(int)
        for f in all_facts:
            counts[f"{f.subject}.{f.type.value}"] += 1
        return dict(counts)
```

Note: `FactStore.query` currently requires subject. Update to make subject optional:

Edit `src/lingxi/facts/store.py`, change the `query` method signature:

```python
    async def query(
        self,
        *,
        subject: str | None = None,
```

(Already this way per Task 0.2 implementation — confirm.)

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_facts/test_retriever.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lingxi/facts/retriever.py tests/test_facts/test_retriever.py
git commit -m "facts: FactQuery + FactRetriever with catalog and FTS path"
```

---

## P1 — Writers + Migration (~1.5 days)

### Task 1.1: WriterBase with subject enforcement

**Files:**
- Create: `src/lingxi/facts/writers/__init__.py` (empty)
- Create: `src/lingxi/facts/writers/base.py`
- Test: `tests/test_facts/test_writers/__init__.py` (empty)
- Test: `tests/test_facts/test_writers/test_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_facts/test_writers/test_base.py
from datetime import datetime

import pytest

from lingxi.facts.models import FactType, Source
from lingxi.facts.store import FactStore
from lingxi.facts.writers.base import WriterBase


class FakeWriter(WriterBase):
    ALLOWED_SOURCE = Source.LIFE_SIMULATED
    SUBJECT_PATTERN = r"^aria$"


@pytest.fixture
async def store(tmp_path):
    s = FactStore(tmp_path / "f.db")
    await s.init()
    return s


@pytest.mark.asyncio
async def test_writer_accepts_matching_subject_and_source(store):
    w = FakeWriter(store)
    f = await w.write(
        subject="aria", content="x",
        type=FactType.EVENT, ts=datetime.now(),
    )
    assert f.source == Source.LIFE_SIMULATED
    assert f.subject == "aria"


@pytest.mark.asyncio
async def test_writer_rejects_wrong_subject(store):
    w = FakeWriter(store)
    with pytest.raises(ValueError, match="subject"):
        await w.write(
            subject="user:u1", content="x",
            type=FactType.EVENT, ts=datetime.now(),
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_facts/test_writers/test_base.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write implementation**

```python
# src/lingxi/facts/writers/__init__.py
"""Writers — strict source/subject ownership for each background subsystem."""
```

```python
# src/lingxi/facts/writers/base.py
"""Base class for Writers.

Each writer subclass declares the (Source, subject-pattern) pair it owns.
Attempting to write outside that ownership raises ValueError. This is the
mechanism that makes subject isolation a structural invariant — there is
no path for, say, NPCTicker to write a fact with subject=aria.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import ClassVar

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.store import FactStore


class WriterBase:
    ALLOWED_SOURCE: ClassVar[Source]
    SUBJECT_PATTERN: ClassVar[str]  # regex string

    def __init__(self, store: FactStore):
        self._store = store
        self._pattern = re.compile(self.SUBJECT_PATTERN)

    async def write(
        self,
        *,
        subject: str,
        content: str,
        type: FactType,
        ts: datetime,
        confidence: float | None = None,
        tags: list[str] | None = None,
        supersedes: str | None = None,
        expires_at: datetime | None = None,
    ) -> Fact:
        if not self._pattern.match(subject):
            raise ValueError(
                f"{self.__class__.__name__} cannot write subject={subject!r}; "
                f"allowed pattern: {self.SUBJECT_PATTERN}"
            )

        fact = Fact(
            subject=subject,
            content=content,
            source=self.ALLOWED_SOURCE,
            type=type,
            ts=ts,
            confidence=confidence,
            tags=tags or [],
            supersedes=supersedes,
            expires_at=expires_at,
        )
        await self._store.write(fact)
        return fact
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_facts/test_writers/test_base.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lingxi/facts/writers/
git commit -m "facts: WriterBase with subject-pattern enforcement"
```

---

### Task 1.2: LifeWriter, NPCWriter, UserStatementWriter, InferenceWriter, WorldWriter, BiographyLoader

**Files:**
- Create: `src/lingxi/facts/writers/{life,npc,user_statement,inference,world,biography}.py`
- Test: `tests/test_facts/test_writers/test_{life,npc,user_statement,inference,world,biography}.py`

- [ ] **Step 1: Write the 6 writers** (all share the same pattern — declare ALLOWED_SOURCE + SUBJECT_PATTERN)

```python
# src/lingxi/facts/writers/life.py
from lingxi.facts.models import Source
from lingxi.facts.writers.base import WriterBase


class LifeWriter(WriterBase):
    """LIFE_SIMULATED events about Aria (subject=aria)."""
    ALLOWED_SOURCE = Source.LIFE_SIMULATED
    SUBJECT_PATTERN = r"^aria$"
```

```python
# src/lingxi/facts/writers/npc.py
from lingxi.facts.models import Source
from lingxi.facts.writers.base import WriterBase


class NPCWriter(WriterBase):
    """NPC_TICKER events about specific NPCs (subject=npc:<id>)."""
    ALLOWED_SOURCE = Source.NPC_TICKER
    SUBJECT_PATTERN = r"^npc:[A-Za-z0-9_-]+$"
```

```python
# src/lingxi/facts/writers/user_statement.py
from lingxi.facts.models import Source
from lingxi.facts.writers.base import WriterBase


class UserStatementWriter(WriterBase):
    """USER_STATED facts captured from chat (subject=user:<key>)."""
    ALLOWED_SOURCE = Source.USER_STATED
    SUBJECT_PATTERN = r"^user:[A-Za-z0-9_:-]+$"
```

```python
# src/lingxi/facts/writers/inference.py
from lingxi.facts.models import Source
from lingxi.facts.writers.base import WriterBase


class InferenceWriter(WriterBase):
    """LLM_INFERRED facts from reflection cycle.

    Can write about Aria (her own patterns) or a user (inferred about them).
    Lower default confidence (0.5) reflects the uncertainty.
    """
    ALLOWED_SOURCE = Source.LLM_INFERRED
    SUBJECT_PATTERN = r"^(aria|user:[A-Za-z0-9_:-]+)$"
```

```python
# src/lingxi/facts/writers/world.py
from lingxi.facts.models import Source
from lingxi.facts.writers.base import WriterBase


class WorldWriter(WriterBase):
    """WORLD_FETCH events (subject=world)."""
    ALLOWED_SOURCE = Source.WORLD_FETCH
    SUBJECT_PATTERN = r"^world$"
```

```python
# src/lingxi/facts/writers/biography.py
from lingxi.facts.models import Source
from lingxi.facts.writers.base import WriterBase


class BiographyLoader(WriterBase):
    """BIOGRAPHY one-shot import of Aria's backstory (subject=aria, ts=past)."""
    ALLOWED_SOURCE = Source.BIOGRAPHY
    SUBJECT_PATTERN = r"^aria$"
```

- [ ] **Step 2: Write parametrized test for all 6**

```python
# tests/test_facts/test_writers/test_concrete_writers.py
from datetime import datetime

import pytest

from lingxi.facts.models import FactType, Source
from lingxi.facts.store import FactStore
from lingxi.facts.writers.biography import BiographyLoader
from lingxi.facts.writers.inference import InferenceWriter
from lingxi.facts.writers.life import LifeWriter
from lingxi.facts.writers.npc import NPCWriter
from lingxi.facts.writers.user_statement import UserStatementWriter
from lingxi.facts.writers.world import WorldWriter


@pytest.fixture
async def store(tmp_path):
    s = FactStore(tmp_path / "f.db")
    await s.init()
    return s


@pytest.mark.parametrize("writer_cls,source,good,bad", [
    (LifeWriter,            Source.LIFE_SIMULATED, "aria",          "user:u1"),
    (NPCWriter,             Source.NPC_TICKER,     "npc:xiaomin",   "aria"),
    (UserStatementWriter,   Source.USER_STATED,    "user:oc_x",     "npc:xiaomin"),
    (InferenceWriter,       Source.LLM_INFERRED,   "user:oc_x",     "world"),
    (InferenceWriter,       Source.LLM_INFERRED,   "aria",          "world"),
    (WorldWriter,           Source.WORLD_FETCH,    "world",         "aria"),
    (BiographyLoader,       Source.BIOGRAPHY,      "aria",          "user:u1"),
])
@pytest.mark.asyncio
async def test_writer_accepts_allowed_and_rejects_disallowed(
    store, writer_cls, source, good, bad,
):
    w = writer_cls(store)
    # Allowed subject succeeds
    f = await w.write(
        subject=good, content="x",
        type=FactType.EVENT, ts=datetime.now(),
    )
    assert f.source == source
    # Disallowed subject raises
    with pytest.raises(ValueError):
        await w.write(
            subject=bad, content="y",
            type=FactType.EVENT, ts=datetime.now(),
        )
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/test_facts/test_writers/ -v
```

Expected: 9 PASS (2 from test_base + 7 parametrized).

- [ ] **Step 4: Commit**

```bash
git add src/lingxi/facts/writers/ tests/test_facts/test_writers/
git commit -m "facts: 6 writers with strict subject/source ownership"
```

---

### Task 1.3: Migration script

**Files:**
- Create: `tools/migrate_to_facts.py`

- [ ] **Step 1: Write the migration script**

```python
# tools/migrate_to_facts.py
"""One-shot migration of existing stores into the unified Fact table.

Reads from:
- data/memory/inner_life/state.json     → LifeWriter (Aria's recent_events / diary)
- data/memory/relational/*.json         → split across UserStatementWriter +
                                           InferenceWriter by source-of-truth
- data/memory/social/npcs/*/events.jsonl → NPCWriter
- data/memory/world/news/*.json         → WorldWriter
- config/personas/*/biography.yaml      → BiographyLoader (one-shot)

Idempotent: skips facts already present (by a deterministic hash of
subject+content+ts). Dry-run mode prints what would be inserted without
writing.

Usage:
    uv run python tools/migrate_to_facts.py --dry-run
    uv run python tools/migrate_to_facts.py            # actually migrate
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lingxi.facts.models import FactType
from lingxi.facts.store import FactStore
from lingxi.facts.writers.biography import BiographyLoader
from lingxi.facts.writers.inference import InferenceWriter
from lingxi.facts.writers.life import LifeWriter
from lingxi.facts.writers.npc import NPCWriter
from lingxi.facts.writers.user_statement import UserStatementWriter
from lingxi.facts.writers.world import WorldWriter


DATA_DIR = Path("data/memory")
FACTS_DB = Path("data/facts.db")


def _hash(subject: str, content: str, ts: str) -> str:
    return hashlib.sha256(f"{subject}|{content}|{ts}".encode()).hexdigest()[:16]


async def migrate_inner_life(writer: LifeWriter, dry: bool) -> int:
    path = DATA_DIR / "inner_life" / "state.json"
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    for ev in data.get("recent_events", []):
        content = (ev.get("content") or "").strip()
        if not content:
            continue
        ts = datetime.fromisoformat(ev["timestamp"])
        if dry:
            print(f"  LIFE: [aria] {ts.date()} {content[:50]}")
        else:
            await writer.write(
                subject="aria", content=content,
                type=FactType.EVENT, ts=ts,
            )
        count += 1
    for d in data.get("recent_diary", []):
        content = (d.get("content") or "").strip()
        if not content:
            continue
        ts = datetime.fromisoformat(d["timestamp"])
        if dry:
            print(f"  LIFE/DIARY: [aria] {ts.date()} {content[:50]}")
        else:
            await writer.write(
                subject="aria", content=content,
                type=FactType.EMOTION_NOTE, ts=ts,
                tags=d.get("tags") or [],
            )
        count += 1
    return count


async def migrate_relational(
    us_writer: UserStatementWriter,
    inf_writer: InferenceWriter,
    dry: bool,
) -> int:
    relational_dir = DATA_DIR / "relational"
    if not relational_dir.exists():
        return 0
    count = 0
    for f in relational_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        recipient_key = data.get("recipient_key", f.stem)
        subject = f"user:{recipient_key.split(':', 1)[-1]}" if ":" in recipient_key else f"user:{recipient_key}"

        # daily_patterns → InferenceWriter (PATTERN, low confidence)
        for d in data.get("daily_patterns", []):
            content = (d.get("pattern") or "").strip()
            if not content:
                continue
            ts = datetime.fromisoformat(d.get("last_confirmed_at") or datetime.now().isoformat())
            if dry:
                print(f"  REL/PATTERN: [{subject}] {content[:50]}")
            else:
                await inf_writer.write(
                    subject=subject, content=content,
                    type=FactType.PATTERN, ts=ts,
                )
            count += 1

        # sweet_moments → InferenceWriter (OPINION, medium confidence)
        for m in data.get("sweet_moments", []):
            content = (m.get("content") or "").strip()
            if not content:
                continue
            ts = datetime.fromisoformat(m.get("timestamp") or datetime.now().isoformat())
            if dry:
                print(f"  REL/MOMENT: [{subject}] {content[:50]}")
            else:
                await inf_writer.write(
                    subject=subject, content=content,
                    type=FactType.OPINION, ts=ts,
                    tags=["sweet_moment", m.get("weight", "medium")],
                )
            count += 1

        # inside_jokes → InferenceWriter (PATTERN, tag inside_joke)
        for j in data.get("inside_jokes", []):
            content = (j.get("phrase") or "").strip()
            if not content:
                continue
            ts = datetime.fromisoformat(j.get("last_used_at") or datetime.now().isoformat())
            if dry:
                print(f"  REL/JOKE: [{subject}] {content[:50]}")
            else:
                await inf_writer.write(
                    subject=subject, content=f"{content} —— {j.get('origin','')}",
                    type=FactType.PATTERN, ts=ts,
                    tags=["inside_joke"],
                )
            count += 1
    return count


async def migrate_social(writer: NPCWriter, dry: bool) -> int:
    npcs_dir = DATA_DIR / "social" / "npcs"
    if not npcs_dir.exists():
        return 0
    count = 0
    for npc_dir in npcs_dir.iterdir():
        if not npc_dir.is_dir():
            continue
        events_file = npc_dir / "events.jsonl"
        if not events_file.exists():
            continue
        npc_id = npc_dir.name
        with open(events_file, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                content = (ev.get("content") or "").strip()
                if not content:
                    continue
                ts = datetime.fromisoformat(ev["ts"])
                subject = f"npc:{npc_id}"
                if dry:
                    print(f"  NPC: [{subject}] {ts.date()} {content[:50]}")
                else:
                    await writer.write(
                        subject=subject, content=content,
                        type=FactType.EVENT, ts=ts,
                        tags=[ev.get("type", "life")] + ([ev.get("arc_id")] if ev.get("arc_id") else []),
                    )
                count += 1
    return count


async def migrate_world(writer: WorldWriter, dry: bool) -> int:
    world_dir = DATA_DIR.parent / "data" / "world" / "news"
    # also check legacy location data/memory/world
    candidates = [world_dir, DATA_DIR / "world" / "news"]
    count = 0
    for d in candidates:
        if not d.exists():
            continue
        for f in d.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            ts = datetime.fromisoformat(data.get("generated_at") or datetime.now().isoformat())
            for item in data.get("items", []):
                content = (item.get("aria_voice") or item.get("headline") or "").strip()
                if not content:
                    continue
                if dry:
                    print(f"  WORLD: [{ts.date()}] {content[:50]}")
                else:
                    await writer.write(
                        subject="world", content=content,
                        type=FactType.EVENT, ts=ts,
                        tags=[item.get("category", "general")],
                    )
                count += 1
    return count


async def migrate_biography(writer: BiographyLoader, dry: bool) -> int:
    """Read persona yaml's biography.life_events into BIOGRAPHY facts."""
    import yaml
    candidates = list(Path("config/personas").glob("*.yaml"))
    count = 0
    for f in candidates:
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        bio = (data.get("biography") or {})
        for ev in bio.get("life_events", []):
            content = (ev.get("content") or "").strip()
            if not content:
                continue
            # Synthesize ts from age if available — anchor on a 1998-01-01 birth.
            age = ev.get("age")
            ts = datetime(1998 + (age or 0), 1, 1) if age else datetime(2000, 1, 1)
            if dry:
                print(f"  BIO: [{age}岁] {content[:50]}")
            else:
                await writer.write(
                    subject="aria", content=content,
                    type=FactType.EVENT, ts=ts,
                    tags=ev.get("tags") or [],
                )
            count += 1
    return count


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", default=str(FACTS_DB))
    args = ap.parse_args()

    store = FactStore(args.db)
    await store.init()

    life = LifeWriter(store)
    npc = NPCWriter(store)
    us = UserStatementWriter(store)
    inf = InferenceWriter(store)
    world = WorldWriter(store)
    bio = BiographyLoader(store)

    print(f"{'[DRY] ' if args.dry_run else ''}Migrating to {args.db}")
    totals = {}
    totals["life"] = await migrate_inner_life(life, args.dry_run)
    totals["relational"] = await migrate_relational(us, inf, args.dry_run)
    totals["social"] = await migrate_social(npc, args.dry_run)
    totals["world"] = await migrate_world(world, args.dry_run)
    totals["biography"] = await migrate_biography(bio, args.dry_run)

    print("\n=== summary ===")
    for k, v in totals.items():
        print(f"  {k}: {v}")
    print(f"  total: {sum(totals.values())}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Dry-run the migration**

```bash
uv run python tools/migrate_to_facts.py --dry-run 2>&1 | tail -30
```

Expected output: counts per source, no error.

- [ ] **Step 3: Actually migrate**

```bash
uv run python tools/migrate_to_facts.py
ls -la data/facts.db
```

Expected: `data/facts.db` created, size > 0.

- [ ] **Step 4: Spot-check the migrated data**

```bash
uv run python -c "
import asyncio
from lingxi.facts.store import FactStore
async def main():
    s = FactStore('data/facts.db')
    counts = await s.count_by_subject()
    for k, v in sorted(counts.items()):
        print(f'  {k}: {v}')
asyncio.run(main())
"
```

Expected: counts for `aria`, `user:*`, `npc:*`, `world` keys.

- [ ] **Step 5: Commit**

```bash
git add tools/migrate_to_facts.py
git commit -m "tools: one-shot migration from old stores to facts table"
```

---

## P2 — Orchestrator (~1.5 days)

### Task 2.1: OrchestrationDecision model + prompt template

**Files:**
- Create: `src/lingxi/brain/__init__.py` (empty)
- Create: `src/lingxi/brain/models.py`
- Test: `tests/test_brain/__init__.py` (empty)
- Test: `tests/test_brain/test_models.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_brain/test_models.py
import pytest

from lingxi.brain.models import OrchestrationDecision, OrchestratorFactQuery


def test_default_decision_is_warm_mid_engagement():
    d = OrchestrationDecision.default()
    assert d.register == "warm"
    assert 0.5 <= d.engage_level <= 0.7
    assert len(d.fact_queries) >= 1  # must surface SOMETHING


def test_decision_from_json_basic():
    raw = {
        "engage_level": 0.7,
        "register": "warm",
        "fact_queries": [
            {"category": "aria.event", "limit": 3}
        ],
        "topic_anchor": "聊到了工作时间",
        "skip": ["world.event"],
    }
    d = OrchestrationDecision.from_dict(raw)
    assert d.engage_level == 0.7
    assert d.register == "warm"
    assert len(d.fact_queries) == 1
    assert d.fact_queries[0].category == "aria.event"
    assert "world.event" in d.skip


def test_decision_handles_unknown_register_gracefully():
    raw = {
        "engage_level": 0.5, "register": "weirdo",
        "fact_queries": [], "topic_anchor": "", "skip": [],
    }
    d = OrchestrationDecision.from_dict(raw)
    assert d.register == "warm"  # fallback


def test_decision_clamps_engage_level():
    raw = {
        "engage_level": 1.5, "register": "warm",
        "fact_queries": [], "topic_anchor": "", "skip": [],
    }
    d = OrchestrationDecision.from_dict(raw)
    assert d.engage_level == 1.0
```

- [ ] **Step 2: Run test to fail**

```bash
uv run pytest tests/test_brain/test_models.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implementation**

```python
# src/lingxi/brain/__init__.py
"""Brain layer — pre-turn orchestration + per-turn rendering."""
```

```python
# src/lingxi/brain/models.py
"""Structured outputs from the Orchestrator's pre-turn decision call."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_REGISTERS = {"warm", "curt", "curious", "withdrawn", "flustered"}


@dataclass
class OrchestratorFactQuery:
    category: str           # "subject.type" e.g. "user:oc_x.pattern"
    limit: int = 5
    semantic: str | None = None  # FTS keyword


@dataclass
class OrchestrationDecision:
    engage_level: float                 # 0-1 (clamped)
    register: str                       # one of VALID_REGISTERS (clamped)
    fact_queries: list[OrchestratorFactQuery]
    topic_anchor: str
    skip: list[str]                     # category names to skip rendering

    @classmethod
    def default(cls) -> "OrchestrationDecision":
        return cls(
            engage_level=0.6,
            register="warm",
            fact_queries=[
                OrchestratorFactQuery(category="aria.event", limit=3),
            ],
            topic_anchor="",
            skip=[],
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "OrchestrationDecision":
        register = raw.get("register", "warm")
        if register not in VALID_REGISTERS:
            register = "warm"

        engage = float(raw.get("engage_level", 0.6))
        engage = max(0.0, min(1.0, engage))

        queries_raw = raw.get("fact_queries") or []
        queries: list[OrchestratorFactQuery] = []
        for q in queries_raw:
            if not isinstance(q, dict):
                continue
            cat = q.get("category")
            if not cat:
                continue
            queries.append(OrchestratorFactQuery(
                category=str(cat),
                limit=int(q.get("limit", 5)),
                semantic=q.get("semantic"),
            ))

        return cls(
            engage_level=engage,
            register=register,
            fact_queries=queries,
            topic_anchor=str(raw.get("topic_anchor", "")),
            skip=[str(s) for s in raw.get("skip", [])],
        )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_brain/test_models.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lingxi/brain/ tests/test_brain/
git commit -m "brain: OrchestrationDecision model with default + from_dict + clamping"
```

---

### Task 2.2: Orchestrator (Sonnet call)

**Files:**
- Create: `src/lingxi/brain/orchestrator.py`
- Test: `tests/test_brain/test_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brain/test_orchestrator.py
import json

import pytest

from lingxi.brain.models import OrchestrationDecision
from lingxi.brain.orchestrator import (
    StateDigest,
    build_orchestrator_prompt,
    decide,
)


class FakeLLMResponse:
    def __init__(self, content): self.content = content


class FakeLLM:
    def __init__(self, response_text=""):
        self.response_text = response_text
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return FakeLLMResponse(self.response_text)


@pytest.mark.asyncio
async def test_decide_returns_parsed_decision():
    payload = json.dumps({
        "engage_level": 0.7, "register": "curious",
        "fact_queries": [{"category": "aria.event", "limit": 2}],
        "topic_anchor": "x", "skip": [],
    })
    llm = FakeLLM(payload)
    digest = StateDigest(activity="刷手机", mood="平静", last_lived=["看了云"])
    catalog = {"aria.event": 5, "user:u1.pattern": 3}

    d = await decide(llm, "你今天忙吗", digest, catalog)
    assert d.engage_level == 0.7
    assert d.register == "curious"


@pytest.mark.asyncio
async def test_decide_falls_back_on_garbled_json():
    llm = FakeLLM("not json at all")
    digest = StateDigest(activity="", mood="", last_lived=[])
    d = await decide(llm, "你好", digest, {})
    # Falls back to default
    assert d.register == "warm"
    assert 0.5 <= d.engage_level <= 0.7


@pytest.mark.asyncio
async def test_decide_falls_back_on_llm_exception():
    class ExceptionLLM:
        async def complete(self, **kwargs):
            raise RuntimeError("boom")
    digest = StateDigest(activity="", mood="", last_lived=[])
    d = await decide(ExceptionLLM(), "x", digest, {})
    assert d.register == "warm"


def test_prompt_includes_user_input_and_digest_and_catalog():
    digest = StateDigest(
        activity="在写代码", mood="专注",
        last_lived=["跟外婆通了电话"],
    )
    catalog = {"aria.event": 5, "user:u1.pattern": 12}
    prompt = build_orchestrator_prompt("怎么了", digest, catalog)
    assert "怎么了" in prompt
    assert "在写代码" in prompt
    assert "外婆" in prompt
    assert "aria.event" in prompt
    assert "12" in prompt or "user:u1.pattern" in prompt


def test_prompt_specifies_strict_json_output():
    prompt = build_orchestrator_prompt(
        "x", StateDigest("", "", []), {}
    )
    assert "JSON" in prompt
    assert "register" in prompt
    assert "fact_queries" in prompt
```

- [ ] **Step 2: Run test to fail**

```bash
uv run pytest tests/test_brain/test_orchestrator.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implementation**

```python
# src/lingxi/brain/orchestrator.py
"""Pre-turn Sonnet call that decides response shape + which facts to surface.

The orchestrator is the only place that "thinks about thinking" — it
reads the user's latest message + a tiny digest of Aria's state + a
catalog of available facts (counts only), and outputs:

- engage_level (0-1)
- register (warm/curt/curious/withdrawn/flustered)
- fact_queries (which buckets to pull from for rendering)
- topic_anchor (one-line summary of what the user is really asking)
- skip (categories to omit from rendering)

Without this, the renderer would dump everything every turn (current
state). With this, the renderer is focused and the prompt is leaner.

Fallback policy: any failure (LLM error, parse error, bad JSON) returns
OrchestrationDecision.default() — never raises into chat path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from lingxi.brain.models import OrchestrationDecision
from lingxi.providers.base import LLMProvider


@dataclass
class StateDigest:
    activity: str
    mood: str
    last_lived: list[str]


_PROMPT = """你在替 Aria 做一个对话调度决策——看用户刚发的话 + Aria 此刻的状态 + 可用的事实目录，决定：

1. engage_level（0-1）：这一轮想投入多少
2. register：warm | curt | curious | withdrawn | flustered（这一轮的语气底色）
3. fact_queries：需要 renderer 拉哪些事实进 prompt（不要拉的就别列）
4. topic_anchor：一句话概括对方话题落点
5. skip：明显不该提的事实类别

【用户刚发的话】
{user_input}

【Aria 此刻】
- 在做什么：{activity}
- 心情：{mood}
- 最近发生过：{last_lived}

【可用事实目录】（仅 count，不含内容）
{catalog}

输出严格 JSON（直接 {{ 开头）：
{{
  "engage_level": 0.0,
  "register": "warm",
  "fact_queries": [
    {{"category": "aria.event", "limit": 3}},
    {{"category": "user:oc_xxx.pattern", "limit": 2, "semantic": "工作时间"}}
  ],
  "topic_anchor": "...",
  "skip": ["world.event"]
}}
"""


def build_orchestrator_prompt(
    user_input: str,
    digest: StateDigest,
    catalog: dict[str, int],
) -> str:
    last_lived = "；".join(digest.last_lived) if digest.last_lived else "（暂无）"
    catalog_str = "\n".join(f"  {k}: {v}" for k, v in sorted(catalog.items())) or "（空）"
    return _PROMPT.format(
        user_input=user_input.strip()[:300],
        activity=digest.activity or "（未指定）",
        mood=digest.mood or "（未指定）",
        last_lived=last_lived,
        catalog=catalog_str,
    )


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


async def decide(
    llm: LLMProvider,
    user_input: str,
    digest: StateDigest,
    catalog: dict[str, int],
    *,
    model: str | None = None,
) -> OrchestrationDecision:
    prompt = build_orchestrator_prompt(user_input, digest, catalog)
    try:
        kwargs = {"model": model} if model else {}
        response = await llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system="你是 Aria 的对话调度器，专门做结构化决策，输出严格 JSON。",
            max_tokens=400,
            temperature=0.3,
            _debug_purpose="orchestrator",
            **kwargs,
        )
        text = _strip_json_fences(response.content)
        data = json.loads(text)
    except Exception as e:
        print(f"[orchestrator] failed, using default: {e}", flush=True)
        return OrchestrationDecision.default()

    if not isinstance(data, dict):
        return OrchestrationDecision.default()
    return OrchestrationDecision.from_dict(data)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_brain/test_orchestrator.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: Smoke test with real Sonnet**

```bash
uv run python -c "
import asyncio
from lingxi.brain.orchestrator import decide, StateDigest
from lingxi.providers.claude import ClaudeProvider
from lingxi.app import _build_auth_manager
from lingxi.providers.registry import ProviderRegistry
from lingxi.utils.config import load_config
from lingxi.auth.manager import AuthMethod

async def main():
    config = load_config('config/default.yaml')
    ProviderRegistry.register_defaults()
    auth_manager = _build_auth_manager(config)
    llm = await ProviderRegistry.create_llm_with_auth(
        'claude', auth_manager=auth_manager,
        auth_method=AuthMethod('oauth_pkce'),
        model='claude-sonnet-4-20250514',
    )
    digest = StateDigest(activity='整理数据', mood='平静', last_lived=['跟外婆通话'])
    catalog = {'aria.event': 8, 'user:u1.pattern': 12, 'npc:xiaomin.event': 5}
    d = await decide(llm, '我正常上班时间是 11 到 9', digest, catalog)
    print(d)

asyncio.run(main())
"
```

Expected: prints `OrchestrationDecision(...)` with non-default values.

- [ ] **Step 6: Commit**

```bash
git add src/lingxi/brain/orchestrator.py tests/test_brain/test_orchestrator.py
git commit -m "brain: Orchestrator (Sonnet pre-turn decision) with safe fallback"
```

---

## P3 — Renderer (~2 days)

### Task 3.1: Persona block extraction

**Files:**
- Modify: `src/lingxi/persona/prompt_builder.py` — extract `build_persona_block()` standalone
- Test: `tests/test_brain/test_renderer.py` (will start filling in)

- [ ] **Step 1: Locate the existing persona sections in `prompt_builder.py`**

```bash
grep -n "_build_identity_section\|_build_personality_section\|_build_speaking_style_section\|_build_message_habits_section" src/lingxi/persona/prompt_builder.py
```

Expected: 4 method definitions.

- [ ] **Step 2: Add a module-level `build_persona_block` function**

In `src/lingxi/persona/prompt_builder.py`, at the very bottom (after the class definition), add:

```python
def build_persona_block(persona: PersonaConfig) -> str:
    """Standalone persona section: identity + personality + style + habits.

    This is the static, cache-friendly portion of the system prompt. Used
    by the new brain.renderer as the persona prefix; the dynamic facts
    blocks are appended after.
    """
    pb = PromptBuilder(persona)
    sections = [
        pb._build_format_preamble(),
        pb._build_identity_section(),
        pb._build_personality_section(),
        pb._build_speaking_style_section(),
    ]
    habits = pb._build_message_habits_section()
    if habits:
        sections.append(habits)
    return "\n\n".join(sections)
```

- [ ] **Step 3: Test it stands alone**

```python
# Add to tests/test_brain/test_renderer.py (new file):
from lingxi.persona.loader import load_persona
from lingxi.persona.prompt_builder import build_persona_block


def test_persona_block_extracts_static_sections():
    persona = load_persona("config/personas/example_persona.yaml")
    block = build_persona_block(persona)
    assert "Aria" in block or persona.name in block
    assert "## 怎么说话" in block
    assert "===META===" in block
```

```bash
uv run pytest tests/test_brain/test_renderer.py::test_persona_block_extracts_static_sections -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/lingxi/persona/prompt_builder.py tests/test_brain/test_renderer.py
git commit -m "brain: extract build_persona_block as standalone function"
```

---

### Task 3.2: Renderer — 3 dynamic blocks

**Files:**
- Create: `src/lingxi/brain/renderer.py`
- Test: `tests/test_brain/test_renderer.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_brain/test_renderer.py`:

```python
from datetime import datetime, timedelta

import pytest

from lingxi.brain.models import OrchestrationDecision, OrchestratorFactQuery
from lingxi.brain.renderer import render_dynamic_blocks
from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.retriever import FactRetriever
from lingxi.facts.store import FactStore


@pytest.fixture
async def retriever(tmp_path):
    s = FactStore(tmp_path / "f.db")
    await s.init()
    now = datetime.now()
    # Aria lived
    await s.write(Fact(
        subject="aria", content="今早煮泡面",
        source=Source.LIFE_SIMULATED, type=FactType.EVENT, ts=now,
    ))
    # User pattern
    await s.write(Fact(
        subject="user:u1", content="工作 11-21",
        source=Source.USER_STATED, type=FactType.PATTERN, ts=now,
    ))
    # NPC event
    await s.write(Fact(
        subject="npc:xiaomin", content="小敏改 paper",
        source=Source.NPC_TICKER, type=FactType.EVENT, ts=now,
    ))
    return FactRetriever(s)


@pytest.mark.asyncio
async def test_renders_only_queried_facts(retriever):
    decision = OrchestrationDecision(
        engage_level=0.6, register="warm",
        fact_queries=[OrchestratorFactQuery(category="aria.event", limit=5)],
        topic_anchor="", skip=[],
    )
    out = await render_dynamic_blocks(
        retriever, decision, recipient_key="u1",
    )
    # Aria's event present
    assert "今早煮泡面" in out
    # NPC + user data not pulled (not queried)
    assert "小敏" not in out
    assert "工作 11-21" not in out


@pytest.mark.asyncio
async def test_subject_isolation_per_block(retriever):
    """NPC facts render in '身边的事' block, user facts in '你和他' block,
    aria facts in '你此刻' block — they must not cross."""
    decision = OrchestrationDecision(
        engage_level=0.7, register="warm",
        fact_queries=[
            OrchestratorFactQuery(category="aria.event", limit=5),
            OrchestratorFactQuery(category="user:u1.pattern", limit=5),
            OrchestratorFactQuery(category="npc:xiaomin.event", limit=5),
        ],
        topic_anchor="", skip=[],
    )
    out = await render_dynamic_blocks(retriever, decision, recipient_key="u1")
    # Find positions of each block's header
    h_self  = out.find("【你此刻】")
    h_them  = out.find("【你和他】")
    h_world = out.find("【身边的事】")
    assert h_self >= 0 and h_them >= 0 and h_world >= 0
    # Check each fact appears AFTER its block header and BEFORE next block
    aria_pos    = out.find("今早煮泡面")
    user_pos    = out.find("工作 11-21")
    npc_pos     = out.find("小敏改 paper")
    assert h_self <= aria_pos < h_them
    assert h_them <= user_pos < h_world
    assert h_world <= npc_pos


@pytest.mark.asyncio
async def test_register_renders_into_prompt(retriever):
    decision = OrchestrationDecision(
        engage_level=0.3, register="curt",
        fact_queries=[OrchestratorFactQuery(category="aria.event", limit=2)],
        topic_anchor="", skip=[],
    )
    out = await render_dynamic_blocks(retriever, decision, recipient_key="u1")
    # The register hint should be visible to the model
    assert "curt" in out.lower() or "短" in out


@pytest.mark.asyncio
async def test_skip_omits_category(retriever):
    decision = OrchestrationDecision(
        engage_level=0.6, register="warm",
        fact_queries=[
            OrchestratorFactQuery(category="aria.event", limit=5),
            OrchestratorFactQuery(category="npc:xiaomin.event", limit=5),
        ],
        topic_anchor="",
        skip=["npc:xiaomin.event"],  # explicitly skipped
    )
    out = await render_dynamic_blocks(retriever, decision, recipient_key="u1")
    assert "小敏" not in out


@pytest.mark.asyncio
async def test_topic_anchor_surfaced(retriever):
    decision = OrchestrationDecision(
        engage_level=0.6, register="warm",
        fact_queries=[OrchestratorFactQuery(category="aria.event", limit=2)],
        topic_anchor="对方在 push back 我对他工作时间的判断",
        skip=[],
    )
    out = await render_dynamic_blocks(retriever, decision, recipient_key="u1")
    assert "push back" in out or "工作时间" in out
```

- [ ] **Step 2: Run tests to fail**

```bash
uv run pytest tests/test_brain/test_renderer.py -v
```

Expected: `ImportError` for `render_dynamic_blocks`.

- [ ] **Step 3: Implementation**

```python
# src/lingxi/brain/renderer.py
"""Render the 3 dynamic prompt blocks from orchestrator decision + fact retriever.

3 blocks (in fixed order):
- 【你此刻】 — aria.* facts + register hint + topic anchor
- 【你和他】 — user:<recipient_key>.* facts
- 【身边的事】 — npc:*.* + world.* facts

Subject isolation is a structural invariant: facts go into the block
that matches their subject prefix. There is no rendering path that
could place a user:* fact into 【你此刻】.

`skip` removes entire categories from rendering (overrides any matching
fact_query). Empty blocks are dropped.

Token budget: caller composes [persona_block, dynamic_blocks]; this
function returns just the dynamic_blocks string. Caller can check
length and re-render with reduced queries if needed.
"""

from __future__ import annotations

from lingxi.brain.models import OrchestrationDecision, OrchestratorFactQuery
from lingxi.facts.models import Fact, FactType
from lingxi.facts.retriever import FactQuery, FactRetriever


_REGISTER_HINT = {
    "warm":      "状态：温暖好奇——自然投入，听到具体的细节会反应。",
    "curt":      "状态：curt（不太想多聊）——短一句够，按当下状态写多长。",
    "curious":   "状态：curious（被勾起兴趣）——会接着问一个具体细节。",
    "withdrawn": "状态：withdrawn（心里压着事）——沉默是一等选项，一两个字回也对。",
    "flustered": "状态：flustered（被戳到了）——节奏乱、句子可以碎，每片段是通顺的。",
}


def _parse_category(cat: str) -> tuple[str, FactType | None]:
    """'aria.event' → ('aria', FactType.EVENT). '<subject>.<type>'."""
    if "." not in cat:
        return cat, None
    subject, type_str = cat.rsplit(".", 1)
    try:
        return subject, FactType(type_str)
    except ValueError:
        return subject, None


def _block_for(subject: str, recipient_key: str) -> str:
    """Return which of the 3 dynamic blocks this subject belongs in."""
    if subject == "aria":
        return "self"
    if subject == f"user:{recipient_key}":
        return "them"
    return "world"  # npc:* and world and other users


async def render_dynamic_blocks(
    retriever: FactRetriever,
    decision: OrchestrationDecision,
    *,
    recipient_key: str,
) -> str:
    """Produce the 3-section dynamic prompt for this turn."""
    blocks: dict[str, list[str]] = {"self": [], "them": [], "world": []}

    for q in decision.fact_queries:
        if q.category in decision.skip:
            continue
        subject, ftype = _parse_category(q.category)
        facts = await retriever.fetch(FactQuery(
            subject=subject, type=ftype, semantic=q.semantic, limit=q.limit,
        ))
        if not facts:
            continue
        target = _block_for(subject, recipient_key)
        for f in facts:
            ts_label = f.ts.strftime("%m-%d %H:%M")
            blocks[target].append(f"- [{ts_label}] {f.content}")

    sections: list[str] = []

    # 【你此刻】
    self_lines: list[str] = []
    self_lines.append(_REGISTER_HINT.get(decision.register, _REGISTER_HINT["warm"]))
    if decision.topic_anchor:
        self_lines.append(f"对方话题落点：{decision.topic_anchor}")
    if blocks["self"]:
        self_lines.append("\n你最近的事：")
        self_lines.extend(blocks["self"])
    sections.append("【你此刻】\n" + "\n".join(self_lines))

    # 【你和他】
    if blocks["them"]:
        sections.append(
            "【你和他】（你过去注意到的——对方这轮明说的事实优先）\n"
            + "\n".join(blocks["them"])
        )

    # 【身边的事】
    if blocks["world"]:
        sections.append(
            "【身边的事】（背景知识，话题撞上时自然带）\n"
            + "\n".join(blocks["world"])
        )

    return "\n\n".join(sections)
```

- [ ] **Step 4: Run tests to pass**

```bash
uv run pytest tests/test_brain/test_renderer.py -v
```

Expected: 5 new + 1 from Task 3.1 = 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lingxi/brain/renderer.py tests/test_brain/test_renderer.py
git commit -m "brain: 3-block renderer with strict subject isolation"
```

---

## P4 — Wire into chat path (~1 day)

### Task 4.1: Engine `_prepare_turn` swap

**Files:**
- Modify: `src/lingxi/conversation/engine.py`
- Modify: `src/lingxi/app.py` (bootstrap Facts store + retriever)

- [ ] **Step 1: Add Facts store bootstrap in app.py**

In `src/lingxi/app.py`, after the existing store initializations (after `world_store = WorldStore(...)` line), add:

```python
    # Facts store + retriever (new unified data layer)
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    facts_store = FactStore(Path(data_dir).parent / "facts.db")
    await facts_store.init()
    fact_retriever = FactRetriever(facts_store)
```

Then add `fact_retriever=fact_retriever` to the `ConversationEngine(...)` constructor call.

- [ ] **Step 2: Add fact_retriever field to ConversationEngine**

In `src/lingxi/conversation/engine.py`, in the `__init__` signature, add `fact_retriever=None` parameter, and store it:

```python
        fact_retriever=None,
    ):
        ...
        self.fact_retriever = fact_retriever
```

- [ ] **Step 3: Add new `_prepare_turn_v2` method**

In the same engine class, add:

```python
    async def _prepare_turn_v2(
        self,
        user_input: str,
        images: list[dict] | None,
        channel: str | None,
        recipient_id: str | None,
    ) -> tuple[str, list[dict]]:
        """New chat-prep path: orchestrator + renderer.

        Replaces _prepare_turn when self.fact_retriever is wired. Old
        path remains as fallback so we can A/B-compare.
        """
        from lingxi.brain.orchestrator import StateDigest, decide
        from lingxi.brain.renderer import render_dynamic_blocks
        from lingxi.persona.prompt_builder import build_persona_block

        recipient_key = f"{channel}:{recipient_id}" if recipient_id else "_anon"

        # 1. Build state digest from current inner_life snapshot
        inner_state = (
            await self.inner_life_store.load_state()
            if self.inner_life_store else None
        )
        digest = StateDigest(
            activity=(
                inner_state.current_activity.description
                if inner_state and inner_state.current_activity else ""
            ),
            mood="，".join(
                f"{k}({v:.1f})"
                for k, v in sorted(
                    (inner_state.axis_modulation or {}).items()
                )[:3]
            ) if inner_state else "",
            last_lived=[
                e.content[:50] for e in (
                    inner_state.recent_events[:3] if inner_state else []
                )
            ],
        )

        # 2. Build catalog
        catalog = await self.fact_retriever.catalog()

        # 3. Orchestrator decides
        decision = await decide(self.llm, user_input, digest, catalog)
        print(
            f"[brain] orch decision: register={decision.register} "
            f"engage={decision.engage_level:.1f} "
            f"queries={len(decision.fact_queries)} "
            f"anchor={decision.topic_anchor[:30]!r}",
            flush=True,
        )

        # 4. Render
        persona_block = build_persona_block(self.persona)
        dynamic_block = await render_dynamic_blocks(
            self.fact_retriever, decision, recipient_key=recipient_key.split(":", 1)[-1],
        )
        system_prompt = persona_block + "\n\n" + dynamic_block

        # 5. Messages still come from context_assembler (existing path)
        memory_context = await self.memory.assemble_context(
            user_input=user_input, recipient_key=recipient_key,
        )
        messages = self.context_assembler.assemble_messages(memory_context)

        # Append user input as last message
        messages.append({"role": "user", "content": user_input})

        return system_prompt, messages
```

- [ ] **Step 4: Switch chat method to use v2 if fact_retriever exists**

Find the `stream_response` method (around line 1028). Locate the `system_prompt, messages = await self._prepare_turn(...)` line. Replace with:

```python
        if self.fact_retriever is not None:
            system_prompt, messages = await self._prepare_turn_v2(
                user_input, images, channel, recipient_id
            )
        else:
            system_prompt, messages = await self._prepare_turn(
                user_input, images, channel, recipient_id
            )
```

Do the same for the other `_prepare_turn` call sites (search `_prepare_turn(`).

- [ ] **Step 5: Smoke test chat**

```bash
uv run pytest -q 2>&1 | tail -3
```

Expected: all tests still pass (~520).

Then start feishu and send one test message in IM:

```bash
LINGXI_DEBUG_LLM=1 nohup uv run lingxi-feishu > /tmp/feishu.log 2>&1 &
sleep 5
grep -c "scheduler started" /tmp/feishu.log
```

After chatting, check inspector:

```bash
uv run python tools/inspect_llm.py --purpose orchestrator tail 1
uv run python tools/inspect_llm.py --purpose chat_stream_split tail 1 | head -50
```

Expected: orchestrator output appears; chat_stream_split shows new 3-block structure.

- [ ] **Step 6: Commit**

```bash
git add src/lingxi/app.py src/lingxi/conversation/engine.py
git commit -m "engine: chat path v2 (orchestrator + renderer) wired in alongside v1"
```

---

## P5 — Background tasks → new writers (~1 day)

### Task 5.1: LifeWriter wired into life simulator

**Files:**
- Modify: `src/lingxi/inner_life/simulator.py`
- Modify: `src/lingxi/app.py` (init LifeWriter, pass to simulator)

- [ ] **Step 1: Update LifeSimulator constructor**

In `src/lingxi/inner_life/simulator.py`, in `LifeSimulator.__init__`, add `life_writer=None` parameter and store it:

```python
        life_writer=None,
    ):
        ...
        self.life_writer = life_writer
```

- [ ] **Step 2: Add write-to-facts in `_maybe_generate_event`**

In the same file, find the line `state.recent_events.insert(0, event)` (around line 690 — was line 700 before my recent edits). Just before that line, add:

```python
        # New facts-arch: also write to unified Fact store
        if self.life_writer is not None:
            try:
                from lingxi.facts.models import FactType
                await self.life_writer.write(
                    subject="aria",
                    content=event.content,
                    type=FactType.EVENT,
                    ts=datetime.now(),
                    confidence=event.significance,
                    tags=["lived"],
                )
            except Exception as e:
                print(f"[life] facts write failed: {e}", flush=True)
```

(Keep the old `state.recent_events.insert(...)` for now — both old + new paths run in parallel, P7 cleans up.)

- [ ] **Step 3: Wire in app.py bootstrap**

In `app.py`, after creating `fact_retriever`, add:

```python
    from lingxi.facts.writers.life import LifeWriter
    life_writer = LifeWriter(facts_store)
```

Then in the LifeSimulator creation in `channels/feishu.py` (search `LifeSimulator(`), pass `life_writer=engine.life_writer` — for which you need to add `self.life_writer = life_writer` and the constructor param in engine.py too.

(Realistic shortcut: stash on the engine instance to keep wiring simple.)

In `conversation/engine.py` `__init__`:
```python
        life_writer=None,
    ):
        ...
        self.life_writer = life_writer
```

In `app.py`:
```python
    engine = ConversationEngine(
        ...
        fact_retriever=fact_retriever,
        life_writer=life_writer,
    )
```

In `channels/feishu.py` where LifeSimulator is constructed:
```python
            self._life_simulator = LifeSimulator(
                persona=self.engine.persona,
                llm=self.engine.llm,
                store=self.engine.inner_life_store,
                tick_interval_minutes=30,
                memory=self.engine.memory,
                life_writer=self.engine.life_writer,
            )
```

- [ ] **Step 4: Test the wire**

```bash
ps aux | grep lingxi-feishu | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
sleep 2
LINGXI_DEBUG_LLM=1 nohup uv run lingxi-feishu > /tmp/feishu.log 2>&1 &
sleep 12  # wait for life simulator first tick
grep "\[life\] event" /tmp/feishu.log | head -2
```

Then verify a fact was written:

```bash
uv run python -c "
import asyncio
from lingxi.facts.store import FactStore
async def main():
    s = FactStore('data/facts.db')
    facts = await s.query(subject='aria', limit=3)
    for f in facts[:3]:
        print(f'{f.source} {f.ts} {f.content[:60]}')
asyncio.run(main())
"
```

Expected: at least one LIFE_SIMULATED fact appears.

- [ ] **Step 5: Commit**

```bash
git add src/lingxi/inner_life/simulator.py src/lingxi/conversation/engine.py src/lingxi/app.py src/lingxi/channels/feishu.py
git commit -m "facts: life simulator writes via LifeWriter (dual-write with old store)"
```

---

### Task 5.2: NPCWriter wired into social scheduler

**Files:**
- Modify: `src/lingxi/social/scheduler.py`
- Modify: `src/lingxi/app.py`
- Modify: `src/lingxi/channels/feishu.py`

- [ ] **Step 1: Add npc_writer param to SocialScheduler**

In `src/lingxi/social/scheduler.py` `SocialScheduler.__init__`, add:

```python
        npc_writer=None,
    ):
        ...
        self._npc_writer = npc_writer
```

- [ ] **Step 2: Dual-write events in `_tick_one_npc`**

In `_tick_one_npc`, after `await self._store.append_event(ev)`, add:

```python
            if self._npc_writer is not None:
                try:
                    from lingxi.facts.models import FactType
                    tags = [ev.type]
                    if ev.arc_id:
                        tags.append(ev.arc_id)
                    await self._npc_writer.write(
                        subject=f"npc:{npc.id}",
                        content=ev.content,
                        type=FactType.EVENT,
                        ts=ev.ts,
                        confidence=ev.significance,
                        tags=tags,
                    )
                except Exception as e:
                    print(f"[social] facts write failed: {e}", flush=True)
```

- [ ] **Step 3: Wire in feishu.py**

```python
from lingxi.facts.writers.npc import NPCWriter
...
            self._social_scheduler = SocialScheduler(
                llm=self.engine.llm,
                graph=self.engine.social_graph,
                store=self.engine.social_store,
                on_event_written=promoter_hook,
                npc_writer=NPCWriter(facts_store),  # if accessible
            )
```

(Easier: pass facts_store to engine, construct NPCWriter inside scheduler init in feishu.py.)

- [ ] **Step 4: Smoke test**

```bash
ps aux | grep lingxi-feishu | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
sleep 2
LINGXI_DEBUG_LLM=1 nohup uv run lingxi-feishu > /tmp/feishu.log 2>&1 &
# trigger a manual social tick:
sleep 5
uv run python tools/social_tick.py xiaomin --force --no-push 2>&1 | tail -5
# check facts
uv run python -c "
import asyncio
from lingxi.facts.store import FactStore
async def main():
    s = FactStore('data/facts.db')
    facts = await s.query(subject='npc:xiaomin', limit=3)
    for f in facts:
        print(f.content[:60])
asyncio.run(main())
"
```

Expected: NPC facts appear with subject `npc:xiaomin`.

- [ ] **Step 5: Commit**

```bash
git add src/lingxi/social/scheduler.py src/lingxi/channels/feishu.py
git commit -m "facts: social scheduler dual-writes NPC events via NPCWriter"
```

---

### Task 5.3: InferenceWriter for reflection cycle

**Files:**
- Modify: `src/lingxi/relational/extractor.py` (or the reflection loop that calls it)

- [ ] **Step 1: Find where extractor results are persisted**

```bash
grep -n "extract_relational_deltas\|merge_deltas_into_memory\|update_memory" src/lingxi/temporal/reflection.py
```

- [ ] **Step 2: After the merge call, also write a few facts**

In `src/lingxi/temporal/reflection.py`, where deltas are merged into RelationalMemory, after the merge, add a dual-write to InferenceWriter for the key categories (daily_patterns, sweet_moments, signature_phrases):

```python
                if engine.inference_writer is not None and added > 0:
                    from lingxi.facts.models import FactType
                    from datetime import datetime
                    for d in deltas.get("daily_patterns", []):
                        try:
                            await engine.inference_writer.write(
                                subject=f"user:{recipient_key.split(':',1)[-1]}",
                                content=d.get("pattern", ""),
                                type=FactType.PATTERN,
                                ts=datetime.now(),
                                tags=["reflection"],
                            )
                        except Exception as e:
                            print(f"[reflection] facts write failed: {e}", flush=True)
                    # similar for sweet_moments → OPINION, inside_jokes → PATTERN+tag
```

(Add similar blocks for sweet_moments and inside_jokes per spec.)

- [ ] **Step 3: Bootstrap InferenceWriter in app.py**

```python
    from lingxi.facts.writers.inference import InferenceWriter
    inference_writer = InferenceWriter(facts_store)
    ...
    engine = ConversationEngine(
        ...
        inference_writer=inference_writer,
    )
```

Add `inference_writer=None` param to engine init.

- [ ] **Step 4: Test by triggering reflection manually**

```bash
# After a chat session, trigger reflection
uv run python -c "
import asyncio
from lingxi.app import create_engine
async def main():
    engine = await create_engine()
    # ... call reflection_loop.run_once() if exposed, else wait for cron
asyncio.run(main())
"
```

(If no manual hook, wait for the 30min cron in feishu and verify via facts query.)

```bash
uv run python -c "
import asyncio
from lingxi.facts.store import FactStore
from lingxi.facts.models import Source
async def main():
    s = FactStore('data/facts.db')
    all_facts = await s.query(limit=200)
    inferred = [f for f in all_facts if f.source == Source.LLM_INFERRED]
    print(f'{len(inferred)} inferred facts')
    for f in inferred[:3]:
        print(f.content[:60])
asyncio.run(main())
"
```

- [ ] **Step 5: Commit**

```bash
git add src/lingxi/temporal/reflection.py src/lingxi/conversation/engine.py src/lingxi/app.py
git commit -m "facts: reflection cycle dual-writes inferred patterns via InferenceWriter"
```

---

### Task 5.4: WorldWriter for daily news

**Files:**
- Modify: `src/lingxi/world/scheduler.py`
- Modify: `src/lingxi/channels/feishu.py`

- [ ] **Step 1: Pass world_writer to WorldScheduler**

In `src/lingxi/world/scheduler.py` `WorldScheduler.__init__`, add `world_writer=None` param.

In `_maybe_fetch_today` after `await self._store.save(briefing)`, add:

```python
        if self._world_writer is not None and briefing.items:
            from lingxi.facts.models import FactType
            from datetime import datetime
            for item in briefing.items:
                content = (item.aria_voice or item.headline or "").strip()
                if not content:
                    continue
                try:
                    await self._world_writer.write(
                        subject="world",
                        content=content,
                        type=FactType.EVENT,
                        ts=datetime.combine(briefing.date, datetime.min.time()),
                        tags=[item.category],
                        expires_at=datetime.now() + timedelta(days=2),
                    )
                except Exception as e:
                    print(f"[world] facts write failed: {e}", flush=True)
```

- [ ] **Step 2: Wire in feishu.py**

```python
from lingxi.facts.writers.world import WorldWriter
...
            self._world_scheduler = WorldScheduler(
                api_key=api_key_for_world,
                store=self.engine.world_store,
                world_writer=WorldWriter(facts_store),
            )
```

- [ ] **Step 3: Trigger manually + verify**

```bash
uv run python -c "
import asyncio
from lingxi.world.scheduler import WorldScheduler
# ... call .trigger_now() if exposed
"
```

Or just wait for the next morning fetch and verify:

```bash
uv run python -c "
import asyncio
from lingxi.facts.store import FactStore
async def main():
    s = FactStore('data/facts.db')
    facts = await s.query(subject='world', limit=5)
    for f in facts:
        print(f.content[:60])
asyncio.run(main())
"
```

- [ ] **Step 4: Commit**

```bash
git add src/lingxi/world/scheduler.py src/lingxi/channels/feishu.py
git commit -m "facts: world scheduler dual-writes news items via WorldWriter"
```

---

## P6 — End-to-end verification (~1 day)

### Task 6.1: 24h soak + 5-criteria check

- [ ] **Step 1: Restart feishu with full pipeline**

```bash
ps aux | grep lingxi-feishu | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
sleep 2
LINGXI_DEBUG_LLM=1 nohup uv run lingxi-feishu > /tmp/feishu.log 2>&1 &
sleep 10
grep "scheduler started\|connected" /tmp/feishu.log | head
```

- [ ] **Step 2: Have a 10-turn conversation (or wait for natural usage)**

Chat with Aria for ~10 turns covering different topics (work, mood, NPCs).

- [ ] **Step 3: Run 5-criteria check**

```bash
# Check 1: Dynamic section count <= 4
uv run python tools/inspect_llm.py --purpose chat_stream_split tail 1 | grep -c "^【"

# Check 2: Subject isolation (no NPC names in aria context)
uv run python tools/inspect_llm.py --purpose chat_stream_split tail 1 | grep -A50 "【你此刻】" | grep -E "Echo|小敏|赵老师|Tom|Lin姐|妈妈" | head

# Check 3: No contradictory facts (supersedes chain)
uv run python -c "
import asyncio
from lingxi.facts.store import FactStore
from collections import Counter
async def main():
    s = FactStore('data/facts.db')
    facts = await s.query(subject='user:oc_c394e90ef07527af9e8d186645e87df1', limit=100)
    contents = [f.content[:30] for f in facts]
    dups = [c for c, n in Counter(contents).items() if n > 1]
    print(f'Dup-ish patterns: {len(dups)}')
    for d in dups[:5]: print(f' - {d}')
asyncio.run(main())
"

# Check 4: Prompt length
uv run python tools/inspect_llm.py --purpose chat_stream_split tail 1 | wc -c

# Check 5: Eyeball 20 turns for AI tells
uv run python tools/inspect_llm.py --purpose chat_stream_split tail 20
```

Expected (success):
1. Dynamic section count ≤ 4
2. Subject isolation grep returns nothing
3. No exact-dup content in same subject
4. Char count of prompt < 70% of pre-refactor baseline (rough proxy)
5. AI tells noticeably reduced (manual review)

- [ ] **Step 4: Document findings + commit a verification log**

```bash
mkdir -p docs/superpowers/verification
cat > docs/superpowers/verification/2026-05-XX-facts-arch-soak.md <<EOF
# P6 Soak Test Results

## 5-Criteria Check
- [x/✗] Dynamic section ≤ 4: <actual count>
- [x/✗] Subject isolation: <grep findings>
- [x/✗] No contradictions: <dup count>
- [x/✗] Prompt length ≤ 70%: <X tokens vs Y baseline>
- [x/✗] AI tells reduced: <subjective notes>

## Sample chat transcripts
<paste 2-3 representative turns>
EOF
git add docs/superpowers/verification/
git commit -m "verify: P6 soak test results"
```

If any criterion fails, file a follow-up task and iterate before P7.

---

## P7 — Cleanup (~0.5 day)

### Task 7.1: Delete old stores + old PromptBuilder dynamic sections

- [ ] **Step 1: Tag pre-cleanup**

```bash
git tag pre-cleanup-facts-arch
```

- [ ] **Step 2: Delete files**

```bash
git rm src/lingxi/inner_life/store.py  # but FIRST move current_activity/today_plan to inner_life/state.py
```

For `inner_life/store.py`: it contains BOTH the recent_events/diary stuff (replaced by Facts) AND `current_activity`/`today_plan` (kept as small state). Refactor:

Create `src/lingxi/inner_life/state.py`:

```python
"""Tiny persistent store for current_activity and today_plan only.
The rest of inner_life data lives in the Facts table."""

# ... extract just the relevant parts from inner_life/store.py
```

Then update all imports across the codebase from `lingxi.inner_life.store` to `lingxi.inner_life.state`.

```bash
grep -rn "from lingxi.inner_life.store" src/ tests/ | wc -l
# update each
```

Now delete:
```bash
git rm src/lingxi/relational/store.py
git rm src/lingxi/social/promoter.py
git rm src/lingxi/world/store.py  # only items list — current_activity-style state stays in state.py
```

- [ ] **Step 3: Trim PromptBuilder**

In `src/lingxi/persona/prompt_builder.py`, delete the now-unused dynamic methods:
- `_build_inner_state_section`
- `_build_world_section`
- `_build_engagement_section`
- `_build_decision_axes_section`
- `_build_relational_section`
- `_build_relationship_section`
- `_build_biography_section`
- `_build_agenda_section`
- `_build_plan_section`
- `_build_memory_section`
- `build_system_prompt` (the whole method — replaced by build_persona_block + renderer)
- `build_turn_focus_reminder` (replaced by renderer's dynamic blocks)

Keep:
- `_build_format_preamble`
- `_build_identity_section`
- `_build_personality_section`
- `_build_speaking_style_section`
- `_build_message_habits_section`
- `build_persona_block`

- [ ] **Step 4: Remove dual-write paths**

In simulator.py, social/scheduler.py, reflection.py, world/scheduler.py: delete the `state.recent_events.insert(...)` and other old-store writes. Keep only the writer calls.

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest -q 2>&1 | tail -5
```

Expected: all tests pass. Failures are expected due to old-store imports — update tests to use Facts.

- [ ] **Step 6: Delete migration script**

```bash
git rm tools/migrate_to_facts.py  # one-shot, gone after use
git rm tools/dedup_recent_events.py  # also obsolete (was for old store)
```

- [ ] **Step 7: Update CLAUDE.md / README**

In `README.md` and any architecture docs, replace mentions of inner_life store / relational store / social store / world store with: "All facts live in `data/facts.db` (SQLite). Bootstrap via `lingxi.facts`."

- [ ] **Step 8: Final commit**

```bash
git add -A
git commit -m "cleanup: delete old stores + dual-write paths + obsolete migration tools"
git tag post-cleanup-facts-arch
```

- [ ] **Step 9: Merge to main**

```bash
git checkout main
git merge refactor/facts-arch --no-ff -m "Merge Facts architecture refactor"
```

---

## Done Criteria

- [ ] All 5 P6 criteria pass
- [ ] `grep -rn "inner_life.store\|relational.store\|social.store\|world.store" src/ tests/` returns 0 hits
- [ ] `data/facts.db` exists and `count_by_subject()` shows reasonable distribution
- [ ] Feishu runs ≥24h without crash
- [ ] CLAUDE.md / README updated
