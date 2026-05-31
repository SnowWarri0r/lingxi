# 表情包发送 SP1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 Aria 在飞书对话里像人一样按情绪发表情包——离线爬一批中文表情包 + vision 自动打标入库,运行时 Aria 通过 `send_sticker` 工具自己选发,飞书出站发出。

**Architecture:** 新模块 `src/lingxi/stickers/`(models / store / captioner / crawler),沿用 facts.db 的 SQLite+FTS5(trigram)检索模式。运行时 `send_sticker` 加进现有 MemGPT `MEMORY_TOOLS`,engine 的 tool-loop 已就绪;dispatch 搜库选一张存到 `self._pending_sticker`,turn 末 `StreamEvent("sticker", path)` 经飞书 `_send_image`(上传拿 image_key → 发 msg_type=image)发出。离线建库走一次性脚本 `tools/crawl_stickers.py`。

**Tech Stack:** Python 3.12 async, SQLite + FTS5(trigram), httpx, pydantic v2, pytest。

> **环境关键:** 所有测试必须用 `.venv/bin/python -m pytest`(Python 3.12)。系统 `python3` 是 3.9,会在 `X | None` 语法上报错并缺依赖。工作目录是 worktree `/Users/lovart/agent-facts-refactor`(分支 `refactor/facts-arch`),editable install,改源码即时生效。

---

## File Structure

| 文件 | 职责 |
|------|------|
| `src/lingxi/stickers/__init__.py` | 包标记(空) |
| `src/lingxi/stickers/models.py` | `Sticker` pydantic 模型 |
| `src/lingxi/stickers/store.py` | `StickerStore`:SQLite + FTS5,`add` / `search` / `get` |
| `src/lingxi/stickers/captioner.py` | vision LLM 打标 → `{caption, emotion, tags, when_to_use}` |
| `src/lingxi/stickers/crawler.py` | 爬图下载 + content_hash 去重(fetch 可注入) |
| `tools/crawl_stickers.py` | 离线一次性建库脚本(crawl → caption → store) |
| `src/lingxi/brain/memory_tools.py`(改) | 加 `send_sticker` 工具 schema |
| `src/lingxi/conversation/engine.py`(改) | `sticker_store` 参数 + `_pending_sticker` + dispatch 分支 + turn 末 emit |
| `src/lingxi/channels/feishu.py`(改) | 消费 `sticker` 事件 + `_send_image` |
| `src/lingxi/app.py`(改) | 构造 `StickerStore` 并注入 engine |
| `tests/test_stickers/...` | 各单元测试 |

---

### Task 1: Sticker 模型 + StickerStore(SQLite + FTS5)

**Files:**
- Create: `src/lingxi/stickers/__init__.py`
- Create: `src/lingxi/stickers/models.py`
- Create: `src/lingxi/stickers/store.py`
- Test: `tests/test_stickers/__init__.py`
- Test: `tests/test_stickers/test_store.py`

参考实现:`src/lingxi/facts/store.py`(FTS5 trigram + <3字 LIKE 回落 + `asyncio.to_thread` + WAL)。StickerStore 是它的精简版——无 supersede、无 importance,主键 `id`,`content_hash` 唯一去重。

- [ ] **Step 1: 创建包标记文件**

Create `src/lingxi/stickers/__init__.py`(空文件):

```python
```

Create `tests/test_stickers/__init__.py`(空文件):

```python
```

- [ ] **Step 2: 写 Sticker 模型**

Create `src/lingxi/stickers/models.py`:

```python
"""Schema for one sticker in the library."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class Sticker(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str
    source_url: str = ""
    content_hash: str
    caption: str = ""
    emotion: str = ""
    tags: list[str] = Field(default_factory=list)
    when_to_use: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
```

- [ ] **Step 3: 写 StickerStore(失败测试先行）**

Create `tests/test_stickers/test_store.py`:

```python
import pytest
from pathlib import Path

from lingxi.stickers.store import StickerStore
from lingxi.stickers.models import Sticker


async def _store(tmp_path) -> StickerStore:
    s = StickerStore(Path(tmp_path) / "stickers.db")
    await s.init()
    return s


@pytest.mark.asyncio
async def test_add_then_search_fts(tmp_path):
    s = await _store(tmp_path)
    await s.add(Sticker(
        file_path="/img/a.png", content_hash="h1",
        caption="无语翻白眼", emotion="无语",
        tags=["翻白眼", "无语"], when_to_use="对方说了离谱的话"))
    hits = await s.search("翻白眼", k=5)
    assert len(hits) == 1
    assert hits[0].caption == "无语翻白眼"


@pytest.mark.asyncio
async def test_search_short_query_like_fallback(tmp_path):
    # 2-char CJK query is below FTS5 trigram's 3-char minimum → LIKE fallback
    s = await _store(tmp_path)
    await s.add(Sticker(
        file_path="/img/b.png", content_hash="h2",
        caption="笑哭", emotion="好笑", tags=["笑哭"], when_to_use="觉得好笑"))
    hits = await s.search("笑哭", k=5)
    assert any(h.caption == "笑哭" for h in hits)


@pytest.mark.asyncio
async def test_add_dedupes_on_content_hash(tmp_path):
    s = await _store(tmp_path)
    first = await s.add(Sticker(
        file_path="/img/c.png", content_hash="dup", caption="A"))
    second = await s.add(Sticker(
        file_path="/img/c2.png", content_hash="dup", caption="B"))
    assert first is True       # inserted
    assert second is False     # skipped (same hash)
    hits = await s.search("A", k=5)
    assert len(hits) == 1
    assert hits[0].caption == "A"


@pytest.mark.asyncio
async def test_get_by_id(tmp_path):
    s = await _store(tmp_path)
    st = Sticker(file_path="/img/d.png", content_hash="h4", caption="比心")
    await s.add(st)
    got = await s.get(st.id)
    assert got is not None and got.caption == "比心"
```

- [ ] **Step 4: 跑测试确认失败**

Run: `cd /Users/lovart/agent-facts-refactor && .venv/bin/python -m pytest tests/test_stickers/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lingxi.stickers.store'`

- [ ] **Step 5: 实现 StickerStore**

Create `src/lingxi/stickers/store.py`:

```python
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
               content='stickers', content_rowid='rowid',
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
            c.commit()
            c.close()

        await asyncio.to_thread(_setup)

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
        """FTS5 MATCH over caption/tags/emotion/when_to_use. Queries shorter
        than 3 chars fall back to a LIKE scan (trigram needs 3-char tokens)."""
        query = (query or "").strip()
        if not query:
            return []
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

            def _read():
                c = self._conn()
                # FTS5 MATCH chokes on bare punctuation/special chars; wrap the
                # query as a quoted phrase so arbitrary user text is treated as
                # a literal token sequence.
                rows = c.execute(sql, (f'"{query}"', k)).fetchall()
                c.close()
                return rows

            rows = await asyncio.to_thread(_read)
        return [_row_to_sticker(r) for r in rows]
```

- [ ] **Step 6: 跑测试确认通过**

Run: `cd /Users/lovart/agent-facts-refactor && .venv/bin/python -m pytest tests/test_stickers/test_store.py -v`
Expected: PASS(4 passed)

- [ ] **Step 7: 提交**

```bash
cd /Users/lovart/agent-facts-refactor
git add src/lingxi/stickers/__init__.py src/lingxi/stickers/models.py src/lingxi/stickers/store.py tests/test_stickers/
git commit -m "feat(stickers): Sticker model + StickerStore (SQLite+FTS5)"
```

---

### Task 2: captioner — vision LLM 打标

**Files:**
- Create: `src/lingxi/stickers/captioner.py`
- Test: `tests/test_stickers/test_captioner.py`

provider 接口见 `src/lingxi/providers/claude.py:227` — `complete(messages, system, max_tokens, temperature, ...) -> CompletionResult`,返回值 `.content` 是文本。图片以 multimodal block 传入 `messages`,格式同 `engine._build_user_message`(`{"type":"image","source":{"type":"base64","media_type":...,"data":...}}`)。

- [ ] **Step 1: 写失败测试**

Create `tests/test_stickers/test_captioner.py`:

```python
import base64
import pytest
from pathlib import Path

from lingxi.stickers.captioner import caption_image


class _FakeProvider:
    def __init__(self, text):
        self._text = text
        self.last_messages = None

    async def complete(self, messages, system=None, max_tokens=1024,
                       temperature=0.7, **kwargs):
        from lingxi.providers.base import CompletionResult
        self.last_messages = messages
        return CompletionResult(content=self._text)


def _write_png(tmp_path) -> Path:
    # 1x1 PNG (valid header so base64 round-trips)
    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
        "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    p = Path(tmp_path) / "x.png"
    p.write_bytes(raw)
    return p


@pytest.mark.asyncio
async def test_caption_image_parses_json(tmp_path):
    prov = _FakeProvider(
        '这是表情 {"caption":"笑哭","emotion":"好笑",'
        '"tags":["笑哭","捂脸"],"when_to_use":"觉得好笑时"} 完毕')
    img = _write_png(tmp_path)
    result = await caption_image(prov, img)
    assert result["caption"] == "笑哭"
    assert result["emotion"] == "好笑"
    assert result["tags"] == ["笑哭", "捂脸"]
    assert result["when_to_use"] == "觉得好笑时"
    # image must be attached as a base64 block
    blocks = prov.last_messages[-1]["content"]
    assert any(b["type"] == "image" for b in blocks)


@pytest.mark.asyncio
async def test_caption_image_bad_json_is_safe(tmp_path):
    prov = _FakeProvider("抱歉我看不清这张图")
    img = _write_png(tmp_path)
    result = await caption_image(prov, img)
    assert result["caption"] == ""
    assert result["tags"] == []
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/lovart/agent-facts-refactor && .venv/bin/python -m pytest tests/test_stickers/test_captioner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lingxi.stickers.captioner'`

- [ ] **Step 3: 实现 captioner**

Create `src/lingxi/stickers/captioner.py`:

```python
"""Vision LLM tagging for stickers.

caption_image reads one image, asks the provider to describe it as a sticker,
and parses a JSON tag blob out of the reply. Parsing is defensive — a reply
that isn't valid JSON yields empty fields rather than raising, so a single bad
image can't abort a batch crawl.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path


_PROMPT = (
    "这是一张聊天表情包。用 JSON 描述它,方便以后按情绪检索。"
    "只输出 JSON,字段:"
    '{"caption":"≤12字概括画面/文字","emotion":"一个情绪词",'
    '"tags":["3-6个检索关键词"],"when_to_use":"一句话说什么场合发"}'
)

_MEDIA_BY_SUFFIX = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}


def _media_type(path: Path) -> str:
    return _MEDIA_BY_SUFFIX.get(path.suffix.lower(), "image/png")


def _parse(text: str) -> dict:
    """Extract the first {...} block and coerce to the expected shape."""
    empty = {"caption": "", "emotion": "", "tags": [], "when_to_use": ""}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return empty
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return empty
    tags = data.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    return {
        "caption": str(data.get("caption", "") or ""),
        "emotion": str(data.get("emotion", "") or ""),
        "tags": [str(t) for t in tags],
        "when_to_use": str(data.get("when_to_use", "") or ""),
    }


async def caption_image(provider, image_path: str | Path) -> dict:
    """Return {caption, emotion, tags, when_to_use} for one sticker image."""
    path = Path(image_path)
    data_b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "source": {
                "type": "base64",
                "media_type": _media_type(path),
                "data": data_b64,
            }},
            {"type": "text", "text": _PROMPT},
        ],
    }]
    result = await provider.complete(
        messages=messages, max_tokens=512, temperature=0.3,
        _debug_purpose="sticker_caption")
    return _parse(result.content)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /Users/lovart/agent-facts-refactor && .venv/bin/python -m pytest tests/test_stickers/test_captioner.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
cd /Users/lovart/agent-facts-refactor
git add src/lingxi/stickers/captioner.py tests/test_stickers/test_captioner.py
git commit -m "feat(stickers): vision LLM captioner"
```

---

### Task 3: crawler — 下载 + content_hash 去重

**Files:**
- Create: `src/lingxi/stickers/crawler.py`
- Test: `tests/test_stickers/test_crawler.py`

**设计要点(来自 spec §3):** crawler 只负责"取 URL 列表 → 下载字节 → sha256 → 存盘",不打标。HTTP fetch 走可注入 callable(默认 httpx),测试传假 fetcher。**站点解析(关键词→图片 URL)留作实现期实地确定**——本任务只实现并测试"给定一组图片 URL,下载去重存盘"这一稳定核心,站点专属的 URL 抽取在离线脚本(Task 7)里按当时选定的站点填。这样 SP1 不被某个站点的反爬难度卡死(spec §3 风险回落)。

- [ ] **Step 1: 写失败测试**

Create `tests/test_stickers/test_crawler.py`:

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/lovart/agent-facts-refactor && .venv/bin/python -m pytest tests/test_stickers/test_crawler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lingxi.stickers.crawler'`

- [ ] **Step 3: 实现 crawler**

Create `src/lingxi/stickers/crawler.py`:

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /Users/lovart/agent-facts-refactor && .venv/bin/python -m pytest tests/test_stickers/test_crawler.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
cd /Users/lovart/agent-facts-refactor
git add src/lingxi/stickers/crawler.py tests/test_stickers/test_crawler.py
git commit -m "feat(stickers): site-agnostic image crawler with hash dedup"
```

---

### Task 4: send_sticker 工具 + engine dispatch

**Files:**
- Modify: `src/lingxi/brain/memory_tools.py`(在 `MEMORY_TOOLS` 末尾追加)
- Modify: `src/lingxi/conversation/engine.py`(`__init__` 加 `sticker_store` + `_pending_sticker`;`_dispatch_memory_tool` 加分支;`_prepare_turn_v2` 重置)
- Test: `tests/test_conversation/test_tool_loop.py`(追加 dispatch 测试)

dispatch 现有结构见 `engine.py:203-276`,`__init__` 见 `engine.py:112-159`,`_prepare_turn_v2` 见 `engine.py:339-362`(`recipient_key` 在 355 算出,`self._current_recipient_key` 在 360 设置)。

- [ ] **Step 1: 给 MEMORY_TOOLS 加 send_sticker**

Modify `src/lingxi/brain/memory_tools.py` — 在 `conversation_search` 工具(`MEMORY_TOOLS` 列表最后一项)之后、列表收尾 `]` 之前,追加:

```python
    {
        "name": "send_sticker",
        "description": (
            "发一张表情包配合你这条消息的情绪。query 用你自己的话描述想发的表情"
            "(如 '无语'、'笑哭'、'摸鱼累了'、'好奇')。偶尔发、贴当下情绪才发,别每句都甩。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
```

`TOOL_NAMES = {t["name"] for t in MEMORY_TOOLS}` 在列表下方,自动包含新工具,无需改动。

- [ ] **Step 2: 写失败测试**

Modify `tests/test_conversation/test_tool_loop.py` — 在文件末尾追加。注意 `_engine` helper(文件顶部已定义)目前不传 `sticker_store`,新测试需要它,所以新增一个带 store 的局部构造。追加:

```python
@pytest.mark.asyncio
async def test_dispatch_send_sticker_sets_pending(tmp_path):
    from lingxi.stickers.store import StickerStore
    from lingxi.stickers.models import Sticker
    eng, _ = await _engine(tmp_path)
    sstore = StickerStore(Path(tmp_path) / "stickers.db")
    await sstore.init()
    await sstore.add(Sticker(
        file_path="/img/wuyu.png", content_hash="h1",
        caption="无语翻白眼", emotion="无语", tags=["无语", "翻白眼"]))
    eng.sticker_store = sstore

    out = await eng._dispatch_memory_tool(
        "send_sticker", {"query": "无语"}, "feishu:x")
    assert "选好了" in out
    assert eng._pending_sticker == "/img/wuyu.png"


@pytest.mark.asyncio
async def test_dispatch_send_sticker_once_per_turn(tmp_path):
    from lingxi.stickers.store import StickerStore
    from lingxi.stickers.models import Sticker
    eng, _ = await _engine(tmp_path)
    sstore = StickerStore(Path(tmp_path) / "stickers.db")
    await sstore.init()
    await sstore.add(Sticker(
        file_path="/img/a.png", content_hash="h1", caption="无语", tags=["无语"]))
    eng.sticker_store = sstore

    first = await eng._dispatch_memory_tool("send_sticker", {"query": "无语"}, "feishu:x")
    second = await eng._dispatch_memory_tool("send_sticker", {"query": "无语"}, "feishu:x")
    assert "选好了" in first
    assert "已经发过" in second


@pytest.mark.asyncio
async def test_dispatch_send_sticker_no_store(tmp_path):
    eng, _ = await _engine(tmp_path)
    # sticker_store defaults to None
    out = await eng._dispatch_memory_tool(
        "send_sticker", {"query": "无语"}, "feishu:x")
    assert "未启用" in out
    assert eng._pending_sticker is None


@pytest.mark.asyncio
async def test_dispatch_send_sticker_no_hit(tmp_path):
    from lingxi.stickers.store import StickerStore
    eng, _ = await _engine(tmp_path)
    sstore = StickerStore(Path(tmp_path) / "stickers.db")
    await sstore.init()
    eng.sticker_store = sstore
    out = await eng._dispatch_memory_tool(
        "send_sticker", {"query": "无语"}, "feishu:x")
    assert "没找到" in out
    assert eng._pending_sticker is None
```

- [ ] **Step 3: 跑测试确认失败**

Run: `cd /Users/lovart/agent-facts-refactor && .venv/bin/python -m pytest tests/test_conversation/test_tool_loop.py::test_dispatch_send_sticker_sets_pending -v`
Expected: FAIL — `AttributeError: 'ConversationEngine' object has no attribute 'sticker_store'`(或 `_pending_sticker`)

- [ ] **Step 4: engine `__init__` 加 sticker_store 参数 + _pending_sticker**

Modify `src/lingxi/conversation/engine.py`:

(a) `__init__` 签名 — 在 `plan_executor=None,`(line 133)之后加一个参数:

```python
        plan_executor=None,
        sticker_store=None,
```

(b) `__init__` body — 在 `self.plan_executor = plan_executor`(line 147)之后加:

```python
        self.plan_executor = plan_executor
        self.sticker_store = sticker_store
```

(c) `__init__` body — 在 `self._last_response_text: str = ""`(line 159)之后加:

```python
        self._last_response_text: str = ""
        # Sticker chosen by the agent this turn (1/turn cap). Reset at the
        # start of every turn in _prepare_turn_v2; emitted at turn end.
        self._pending_sticker: str | None = None
```

(d) 文件顶部 import 区(line 5 附近,`import uuid` 之后)加:

```python
import random
import uuid
```

- [ ] **Step 5: dispatch 加 send_sticker 分支**

Modify `src/lingxi/conversation/engine.py` — 在 `_dispatch_memory_tool` 的 `if name == "conversation_search":` 分支(line 263)之前插入:

```python
            if name == "send_sticker":
                if self._pending_sticker is not None:
                    return "本轮已经发过一张表情了"
                if self.sticker_store is None:
                    return "（表情库未启用）"
                query = args.get("query", "")
                hits = await self.sticker_store.search(query, k=5)
                if not hits:
                    return f"没找到合适的表情（{query}）"
                chosen = random.choice(hits)
                self._pending_sticker = chosen.file_path
                return f"选好了:{chosen.caption}（会发出去）"

```

- [ ] **Step 6: _prepare_turn_v2 每轮重置 _pending_sticker**

Modify `src/lingxi/conversation/engine.py` — 在 `_prepare_turn_v2` 里 `self._current_recipient_key = recipient_key`(line 360)之后加:

```python
        self._current_recipient_key = recipient_key
        # Fresh turn: clear any sticker the previous turn selected.
        self._pending_sticker = None
```

- [ ] **Step 7: 跑测试确认通过**

Run: `cd /Users/lovart/agent-facts-refactor && .venv/bin/python -m pytest tests/test_conversation/test_tool_loop.py -v`
Expected: PASS(原有 + 4 个新测试全过)

- [ ] **Step 8: 提交**

```bash
cd /Users/lovart/agent-facts-refactor
git add src/lingxi/brain/memory_tools.py src/lingxi/conversation/engine.py tests/test_conversation/test_tool_loop.py
git commit -m "feat(stickers): send_sticker tool + engine dispatch (1/turn cap)"
```

---

### Task 5: turn 末 emit StreamEvent("sticker")

**Files:**
- Modify: `src/lingxi/conversation/engine.py`(`StreamEvent` 文档串;`_chat_stream_events_locked` emit;`StreamEvent("done")` 旁)
- Test: `tests/test_conversation/test_tool_loop.py`(追加流式 emit 测试)

生产路径是飞书的 `chat_stream_events`(走 `_chat_stream_events_locked`,line 818)。本任务只在该流式路径 emit sticker 事件——这是飞书唯一消费的路径。`_chat_full_locked` / `_chat_stream_locked` 非生产路径,不接 sticker(YAGNI)。

- [ ] **Step 1: 写失败测试**

Modify `tests/test_conversation/test_tool_loop.py` — 末尾追加。复用文件里已有的 `_ScriptedLLM` / `_toolcall` / `_final`(顶部已定义):

```python
@pytest.mark.asyncio
async def test_stream_events_emits_sticker(tmp_path):
    from lingxi.stickers.store import StickerStore
    from lingxi.stickers.models import Sticker
    eng, _ = await _engine(tmp_path)
    sstore = StickerStore(Path(tmp_path) / "stickers.db")
    await sstore.init()
    await sstore.add(Sticker(
        file_path="/img/wuyu.png", content_hash="h1",
        caption="无语", emotion="无语", tags=["无语"]))
    eng.sticker_store = sstore

    async def _fake_prep(ui, im, ch, rid):
        eng._current_recipient_key = "feishu:x"
        eng._pending_sticker = None
        return "SYS", [{"role": "user", "content": ui}]
    eng._prepare_turn_v2 = _fake_prep

    eng.llm = _ScriptedLLM([
        _toolcall("send_sticker", {"query": "无语"}),
        _final("哈哈对啊"),
    ])
    events = [e async for e in eng.chat_stream_events(
        "hi", channel="feishu", recipient_id="x")]
    stickers = [e for e in events if e.type == "sticker"]
    assert len(stickers) == 1
    assert stickers[0].content == "/img/wuyu.png"
    # cleared after emit
    assert eng._pending_sticker is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/lovart/agent-facts-refactor && .venv/bin/python -m pytest tests/test_conversation/test_tool_loop.py::test_stream_events_emits_sticker -v`
Expected: FAIL — 断言 `len(stickers) == 1` 失败(没有 sticker 事件被 emit)

- [ ] **Step 3: 在 _chat_stream_events_locked emit sticker**

Modify `src/lingxi/conversation/engine.py` — 在 `_chat_stream_events_locked` 末尾 `yield StreamEvent("done", output.speech)`(line 951)之前插入:

```python
        # Emit the sticker the agent chose this turn (if any), before `done`
        # so the channel can send it right after the speech bubble. Clear so a
        # later turn can't re-emit a stale path.
        if self._pending_sticker:
            yield StreamEvent("sticker", self._pending_sticker)
            self._pending_sticker = None

        yield StreamEvent("done", output.speech)
```

- [ ] **Step 4: 更新 StreamEvent 文档串**

Modify `src/lingxi/conversation/engine.py` — `StreamEvent` 类的 docstring(line 37)从:

```python
    type: str  # "chunk", "mood", "memory_write", "plan_update", "done"
```

改为:

```python
    type: str  # "chunk", "mood", "memory_write", "plan_update", "sticker", "done"
```

并在 `chat_stream_events` 的 docstring(line 802-809)的 `StreamEvent("done", speech)` 行之前加一行:

```python
            StreamEvent("sticker", path)       - sticker file to send (at end)
            StreamEvent("done", speech)        - final clean speech
```

- [ ] **Step 5: 跑测试确认通过**

Run: `cd /Users/lovart/agent-facts-refactor && .venv/bin/python -m pytest tests/test_conversation/test_tool_loop.py -v`
Expected: PASS(全过)

- [ ] **Step 6: 提交**

```bash
cd /Users/lovart/agent-facts-refactor
git add src/lingxi/conversation/engine.py tests/test_conversation/test_tool_loop.py
git commit -m "feat(stickers): emit StreamEvent('sticker') at turn end"
```

---

### Task 6: 飞书出站发图

**Files:**
- Modify: `src/lingxi/channels/feishu.py`(`_stream_reply` 消费 sticker 事件;新增 `_send_image`)
- Test: `tests/test_channels/test_feishu_send_image.py`

飞书 auth/token 见 `feishu.py:141-167`(`self.token_mgr.headers()` 返回带 `Authorization: Bearer ...` 的 dict)。事件消费在 `_stream_reply`(line 1187-1222)。下载图反向参考 `_download_image`(line 841)。

**飞书发图两步:** ① `POST {FEISHU_BASE}/im/v1/images`,multipart 表单 `image_type=message` + `image=<bytes>` → 返回 `data.image_key`;② `POST {FEISHU_BASE}/im/v1/messages?receive_id_type=chat_id`,body `{receive_id, msg_type:"image", content: json({image_key})}`。

- [ ] **Step 1: 写失败测试**

Create `tests/test_channels/test_feishu_send_image.py`:

```python
import json
import pytest
from pathlib import Path


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def json(self):
        return self._payload


class _FakeClient:
    """Records POSTs; returns image_key on upload, ok on send."""
    def __init__(self):
        self.posts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if url.endswith("/im/v1/images"):
            return _FakeResp({"code": 0, "data": {"image_key": "img_xyz"}})
        return _FakeResp({"code": 0, "data": {"message_id": "om_1"}})


@pytest.mark.asyncio
async def test_send_image_uploads_then_sends(tmp_path, monkeypatch):
    from lingxi.channels import feishu as feishu_mod

    img = Path(tmp_path) / "s.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nFAKE")

    # Build an agent instance without running __init__ (avoids full bootstrap);
    # inject only what _send_image touches.
    agent = feishu_mod.FeishuAgent.__new__(feishu_mod.FeishuAgent)

    class _TM:
        def headers(self):
            return {"Authorization": "Bearer t", "Content-Type": "application/json"}
    agent.token_mgr = _TM()

    fake = _FakeClient()
    monkeypatch.setattr(feishu_mod.httpx, "AsyncClient", lambda *a, **k: fake)

    await agent._send_image("chat_1", str(img))

    upload = next(p for p in fake.posts if p[0].endswith("/im/v1/images"))
    send = next(p for p in fake.posts if "/im/v1/messages" in p[0])
    # send body carries the image_key returned by upload
    body = send[1]["json"]
    assert body["msg_type"] == "image"
    assert json.loads(body["content"])["image_key"] == "img_xyz"
    assert body["receive_id"] == "chat_1"
```

> 实现期校验:确认 `FeishuAgent` 的类名与 import 路径(`from lingxi.channels import feishu`,类 `FeishuAgent`)。若类名不同,按实际改测试与下面的引用——但不要改变 `_send_image` 的签名与行为。

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /Users/lovart/agent-facts-refactor && .venv/bin/python -m pytest tests/test_channels/test_feishu_send_image.py -v`
Expected: FAIL — `AttributeError: ... has no attribute '_send_image'`

- [ ] **Step 3: 实现 _send_image**

Modify `src/lingxi/channels/feishu.py` — 在 `_send_text_async`(line 865)旁(同一个类里)新增方法:

```python
    async def _send_image(self, chat_id: str, file_path: str) -> None:
        """Upload a local image to Feishu and send it as an image message.

        Two calls: POST /im/v1/images (multipart) → image_key, then
        POST /im/v1/messages with msg_type=image. Failures are logged, not
        raised, so a sticker problem never breaks the turn.
        """
        from pathlib import Path as _Path
        try:
            data = _Path(file_path).read_bytes()
        except Exception as e:
            print(f"[sticker] read failed {file_path}: {e}", flush=True)
            return

        # Upload: multipart, no JSON Content-Type (httpx sets the boundary).
        headers = self.token_mgr.headers()
        headers = {k: v for k, v in headers.items()
                   if k.lower() != "content-type"}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                up = await client.post(
                    f"{FEISHU_BASE}/im/v1/images",
                    headers=headers,
                    data={"image_type": "message"},
                    files={"image": (_Path(file_path).name, data)},
                )
                up_data = up.json()
                if up_data.get("code") != 0:
                    print(f"[sticker] upload failed: {up_data}", flush=True)
                    return
                image_key = up_data["data"]["image_key"]

                send = await client.post(
                    f"{FEISHU_BASE}/im/v1/messages?receive_id_type=chat_id",
                    headers=self.token_mgr.headers(),
                    json={
                        "receive_id": chat_id,
                        "msg_type": "image",
                        "content": json.dumps({"image_key": image_key}),
                    },
                )
                send_data = send.json()
                if send_data.get("code") != 0:
                    print(f"[sticker] send failed: {send_data}", flush=True)
        except Exception as e:
            print(f"[sticker] _send_image error: {e}", flush=True)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /Users/lovart/agent-facts-refactor && .venv/bin/python -m pytest tests/test_channels/test_feishu_send_image.py -v`
Expected: PASS(1 passed)

> 若 `tests/test_channels/__init__.py` 不存在导致采集失败,创建空文件:`touch tests/test_channels/__init__.py`(实现期视报错决定)。

- [ ] **Step 5: 在 _stream_reply 消费 sticker 事件**

Modify `src/lingxi/channels/feishu.py`:

(a) 在 `_stream_reply` 的事件循环里(line 1218 `elif event.type == "turn_id":` 之前)加一个分支,把 sticker 路径收集到一个局部变量。先在 `turn_id: str | None = None`(line 1183)之后加初始化:

```python
            turn_id: str | None = None
            pending_sticker: str | None = None
```

再在 `elif event.type == "turn_id":`(line 1218)之前插入:

```python
                    elif event.type == "sticker":
                        pending_sticker = event.content

```

(b) 在 `card.finish()` 之后、extras 发送之后(line 1290 `append buttons` 那段之后,方法返回前)发图。在 `_stream_reply` 末尾(line 1290 的 `except` 块之后)加:

```python
            # Send the sticker as a separate image message after the text.
            if pending_sticker:
                try:
                    await self._send_image(chat_id, pending_sticker)
                except Exception as e:
                    print(f"[feishu] sticker send failed: {e}", flush=True)
```

- [ ] **Step 6: 跑全套渠道 + 引擎测试确认没回归**

Run: `cd /Users/lovart/agent-facts-refactor && .venv/bin/python -m pytest tests/test_channels/ tests/test_conversation/ tests/test_stickers/ -v`
Expected: PASS(全过)

- [ ] **Step 7: 提交**

```bash
cd /Users/lovart/agent-facts-refactor
git add src/lingxi/channels/feishu.py tests/test_channels/test_feishu_send_image.py
git commit -m "feat(stickers): Feishu outbound image send + consume sticker event"
```

---

### Task 7: app.py 接线 + 离线建库脚本

**Files:**
- Modify: `src/lingxi/app.py`(构造 `StickerStore` + 注入 engine)
- Create: `tools/crawl_stickers.py`
- Test: `tests/test_stickers/test_app_wiring.py`

engine 构造在 `app.py:278-296`,facts_store 构造在 `app.py:228`。

- [ ] **Step 1: 写 wiring 测试**

Create `tests/test_stickers/test_app_wiring.py`:

```python
import pytest
from pathlib import Path

from lingxi.stickers.store import StickerStore
from lingxi.conversation.engine import ConversationEngine
from lingxi.memory.manager import MemoryManager
from lingxi.persona.models import PersonaConfig, Identity


@pytest.mark.asyncio
async def test_engine_accepts_sticker_store(tmp_path):
    sstore = StickerStore(Path(tmp_path) / "stickers.db")
    await sstore.init()

    class _LLM:
        async def complete(self, **kw): ...

    eng = ConversationEngine(
        persona=PersonaConfig(name="Aria", identity=Identity(full_name="Aria")),
        llm_provider=_LLM(),
        memory_manager=MemoryManager(data_dir=str(Path(tmp_path) / "mem")),
        sticker_store=sstore,
    )
    assert eng.sticker_store is sstore
    assert eng._pending_sticker is None
```

- [ ] **Step 2: 跑测试确认通过(Task 4 已加 sticker_store 参数,此测试应直接过)**

Run: `cd /Users/lovart/agent-facts-refactor && .venv/bin/python -m pytest tests/test_stickers/test_app_wiring.py -v`
Expected: PASS(1 passed)— 验证 engine 接口已就绪。若 FAIL,说明 Task 4 的 `__init__` 参数未落地,回 Task 4 修。

- [ ] **Step 3: app.py 构造并注入 StickerStore**

Modify `src/lingxi/app.py`:

(a) 在 `facts_store = FactStore(Path(data_dir).parent / "facts.db")`(line 228)那段附近,facts_store init 之后加:

```python
    from lingxi.stickers.store import StickerStore
    sticker_store = StickerStore(Path(data_dir).parent / "stickers" / "stickers.db")
    await sticker_store.init()
```

(b) 在 `ConversationEngine(...)` 构造调用(line 278-296)的 `plan_executor=plan_executor,`(line 295)之后加:

```python
        plan_executor=plan_executor,
        sticker_store=sticker_store,
```

- [ ] **Step 4: 写离线建库脚本**

Create `tools/crawl_stickers.py`:

```python
"""Offline one-shot: build the sticker library.

Pipeline: collect image URLs → download+dedup (crawler) → caption (vision LLM)
→ store (StickerStore). Run manually, NOT part of the serving loop.

The site-specific URL collection (keyword → image URLs) is intentionally a
small, swappable function below. SP1 ships with a SEED_URLS fallback so the
library can be built without committing to a specific site's scraping rules;
fill `collect_urls` with a real source when one is validated (spec §3).

Usage:
    .venv/bin/python tools/crawl_stickers.py
"""

import asyncio
import os
from pathlib import Path

from lingxi.stickers.crawler import download_images
from lingxi.stickers.captioner import caption_image
from lingxi.stickers.store import StickerStore, Sticker
from lingxi.providers.claude import ClaudeProvider


# Replace with real source scraping once a site is validated. Until then,
# drop direct image URLs (or hand-place files and skip crawling) here.
SEED_URLS: list[str] = [
]

DATA_DIR = os.environ.get("MEMORY_DATA_DIR", "./data/memory")
IMG_DIR = Path(DATA_DIR).parent / "stickers" / "img"
DB_PATH = Path(DATA_DIR).parent / "stickers" / "stickers.db"


async def collect_urls() -> list[str]:
    """Site-specific URL collection. Returns image URLs to download.

    SP1 default: SEED_URLS. Implementer fills this with real scraping when a
    target site is chosen and verified (spec §3 risk fallback)."""
    return SEED_URLS


async def main() -> None:
    store = StickerStore(DB_PATH)
    await store.init()

    urls = await collect_urls()
    if not urls:
        print("No URLs to crawl. Populate SEED_URLS or collect_urls(), "
              "or hand-place images in IMG_DIR and adapt this script.")
        return

    print(f"Downloading {len(urls)} candidate images → {IMG_DIR}")
    downloaded = await download_images(urls, out_dir=IMG_DIR, delay=1.0)
    print(f"{len(downloaded)} distinct images stored.")

    provider = ClaudeProvider()
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


if __name__ == "__main__":
    asyncio.run(main())
```

> 实现期校验两处 import:① `ClaudeProvider` 的真实类名/构造参数(看 `src/lingxi/providers/claude.py` 顶部 class 定义与 app.py 怎么构造 provider);② `from lingxi.stickers.store import StickerStore, Sticker` 是否需要拆成两行(`Sticker` 在 `lingxi.stickers.models`)。脚本不在测试覆盖内,按实际 import 调整,保持管线逻辑不变。

- [ ] **Step 5: 跑全套测试 + import 烟雾测试**

Run: `cd /Users/lovart/agent-facts-refactor && .venv/bin/python -m pytest tests/test_stickers/ tests/test_conversation/test_tool_loop.py tests/test_channels/test_feishu_send_image.py -v && .venv/bin/python -c "import lingxi.app; import tools.crawl_stickers" 2>&1 | tail -5`
Expected: 测试全 PASS;import 烟雾测试无 ImportError(若 `tools` 非包,用 `.venv/bin/python tools/crawl_stickers.py` 在 SEED_URLS 为空时打印 "No URLs to crawl" 即算通过)。

- [ ] **Step 6: 提交**

```bash
cd /Users/lovart/agent-facts-refactor
git add src/lingxi/app.py tools/crawl_stickers.py tests/test_stickers/test_app_wiring.py
git commit -m "feat(stickers): wire StickerStore into app + offline build script"
```

---

## Self-Review

**1. Spec coverage** (对照 `docs/superpowers/specs/2026-05-29-sticker-sending-design.md`):

- §1 模块与数据流 → Task 1-3(模块)+ Task 4-6(运行时数据流)✅
- §2 StickerStore(SQLite+FTS5,add/search/get,hash 去重,<3字 LIKE 回落)→ Task 1 ✅
- §3 crawler(可注入 fetch、限速、去重、风险回落到种子图)→ Task 3 + Task 7 脚本的 SEED_URLS 回落 ✅
- §4 captioner(vision JSON、解析容错)→ Task 2 ✅
- §5 send_sticker 工具 + dispatch → Task 4 ✅
- §6 turn 末发图 + 飞书出站 → Task 5(emit)+ Task 6(_send_image + 消费)✅
- §7 app.py 接线 → Task 7 ✅
- §8 频率/品味(1/turn 硬上限 + 工具描述软引导 + top-K 随机)→ Task 4 dispatch ✅
- §9 测试(store/captioner/crawler/dispatch/流式/_send_image)→ 各 Task 测试 ✅
- §10 版权/ToS → 文档级,Task 7 脚本注释里标注来源/限速,无代码动作 ✅
- §11 out of scope(自制表情、GIF、reaction、反馈学习、跨渠道)→ 未触碰 ✅

§6 提到"`_chat_full_locked`/`_chat_stream_locked` 也兜底加 sticker_path 字段"——本计划按 YAGNI 收窄到生产路径(`_chat_stream_events_locked`,飞书唯一消费路径),非流式路径不接。这是有意的范围收窄,已在 Task 5 说明。

**2. Placeholder scan:** 无 TBD/TODO/"实现细节略"。两处"实现期校验"(Task 6 类名、Task 7 import/站点)是对真实代码符号的核对指令,非占位——核心逻辑代码均完整给出。crawler 的站点 URL 抽取按 spec §3 明确留给离线脚本,核心下载/去重已完整实现并测试。

**3. Type consistency:**
- `Sticker` 字段(file_path/content_hash/caption/emotion/tags/when_to_use)在 models / store / captioner 返回 dict / dispatch / 脚本间一致 ✅
- `StickerStore.add() -> bool`(True=插入/False=去重),`.search(query, k) -> list[Sticker]`,`.get(id) -> Sticker | None` 在所有调用点签名一致 ✅
- `caption_image(provider, path) -> dict{caption,emotion,tags,when_to_use}` 与脚本消费一致 ✅
- `download_images(urls, *, out_dir, fetch, delay, retries) -> list[dict{file_path,source_url,content_hash}]` 与脚本消费一致 ✅
- engine `self._pending_sticker`、`self.sticker_store`、`StreamEvent("sticker", path)` 在 dispatch / emit / 飞书消费间一致 ✅
- 飞书 `_send_image(chat_id, file_path)` 在定义与 `_stream_reply` 调用点一致 ✅
