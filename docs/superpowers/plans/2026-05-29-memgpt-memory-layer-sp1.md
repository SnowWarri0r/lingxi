# MemGPT 记忆层 SP1 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在纯 GA 回路上叠加 MemGPT 混合能力——orchestrator 预热 + agent 在 turn 内用 function call 主动检索 / 自编辑核心记忆。

**Architecture:** Anthropic 原生 tool use。provider 支持 `tools`/`tool_use`；engine 跑多步 agentic loop（最多 5 步，到顶强制出文本）；5 个记忆工具按 `recipient_key` 把 scope 映射成 subject 落 facts.db；核心记忆块用 `FactType.CORE` + supersede 当版本链。

**Tech Stack:** Python 3.12 async, facts.db(SQLite+FTS5), pydantic, pytest。测试用 `.venv/bin/python -m pytest`。

**Spec:** `docs/superpowers/specs/2026-05-29-memgpt-memory-layer-design.md`

**约定:** 所有 `pytest` 命令前缀 `.venv/bin/python -m`，在 worktree 根 `/Users/lovart/agent-facts-refactor` 运行。

---

## File Structure

- Modify `src/lingxi/facts/models.py` — 加 `FactType.CORE`
- Modify `src/lingxi/facts/store.py` — 加 `get_core_block(subject)`
- Modify `src/lingxi/facts/retriever.py` — 加 `get_core_block(subject)` 透传
- Create `src/lingxi/facts/writers/core_memory.py` — `CoreMemoryWriter`
- Modify `src/lingxi/providers/base.py` — `CompletionResult` 加 `tool_calls` / `raw_content_blocks`
- Modify `src/lingxi/providers/claude.py` — `_build_body` + `complete` 支持 tools / 解析 tool_use
- Create `src/lingxi/brain/memory_tools.py` — `MEMORY_TOOLS` schema 常量
- Modify `src/lingxi/conversation/engine.py` — `_dispatch_memory_tool` + `_generate_with_tools` + 接入两个生成分支
- Modify `src/lingxi/brain/renderer.py` — 渲染【核心记忆】段
- Modify `src/lingxi/app.py` — 构造 `CoreMemoryWriter` 注入 engine
- Tests: `tests/test_facts/test_core_memory.py`, `tests/test_providers/test_tool_use.py`, `tests/test_brain/test_memory_tools.py`, `tests/test_conversation/test_tool_loop.py`

---

### Task 1: FactType.CORE + store.get_core_block

**Files:**
- Modify: `src/lingxi/facts/models.py`
- Modify: `src/lingxi/facts/store.py`
- Test: `tests/test_facts/test_core_memory.py`

- [ ] **Step 1: Add CORE to FactType**

In `src/lingxi/facts/models.py`, the `FactType` enum currently ends with `EMOTION_NOTE = "emotion_note"`. Add:
```python
    CORE         = "core"          # MemGPT core-memory block (one current per subject, supersede chain)
```

- [ ] **Step 2: Write failing test for get_core_block**

Create `tests/test_facts/test_core_memory.py`:
```python
import pytest
from datetime import datetime
from pathlib import Path

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.store import FactStore


async def _store(tmp_path) -> FactStore:
    s = FactStore(Path(tmp_path) / "facts.db")
    await s.init()
    return s


@pytest.mark.asyncio
async def test_get_core_block_none_when_empty(tmp_path):
    s = await _store(tmp_path)
    assert await s.get_core_block("aria") is None


@pytest.mark.asyncio
async def test_get_core_block_returns_latest_unsuperseded(tmp_path):
    s = await _store(tmp_path)
    f1 = Fact(subject="aria", content="v1", source=Source.LLM_INFERRED,
              type=FactType.CORE, ts=datetime(2026, 5, 1, 9, 0))
    await s.write(f1)
    f2 = Fact(subject="aria", content="v2", source=Source.LLM_INFERRED,
              type=FactType.CORE, ts=datetime(2026, 5, 1, 10, 0), supersedes=f1.id)
    await s.write(f2)
    block = await s.get_core_block("aria")
    assert block is not None
    assert block.content == "v2"


@pytest.mark.asyncio
async def test_get_core_block_scoped_by_subject(tmp_path):
    s = await _store(tmp_path)
    await s.write(Fact(subject="aria", content="A", source=Source.LLM_INFERRED,
                       type=FactType.CORE, ts=datetime(2026, 5, 1, 9, 0)))
    await s.write(Fact(subject="user:feishu:x", content="U", source=Source.LLM_INFERRED,
                       type=FactType.CORE, ts=datetime(2026, 5, 1, 9, 0)))
    a = await s.get_core_block("aria")
    u = await s.get_core_block("user:feishu:x")
    assert a.content == "A"
    assert u.content == "U"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_facts/test_core_memory.py -v`
Expected: FAIL with `AttributeError: 'FactStore' object has no attribute 'get_core_block'`

- [ ] **Step 4: Implement get_core_block**

In `src/lingxi/facts/store.py`, add this method to `FactStore` (after `get`):
```python
    async def get_core_block(self, subject: str) -> Fact | None:
        """Return the current CORE-type fact for subject (latest, not superseded)."""
        def _read():
            c = self._conn()
            row = c.execute(
                "SELECT * FROM facts WHERE subject = ? AND type = ? "
                "AND id NOT IN (SELECT supersedes FROM facts WHERE supersedes IS NOT NULL) "
                "ORDER BY ts DESC LIMIT 1",
                (subject, FactType.CORE.value),
            ).fetchone()
            c.close()
            return row

        row = await asyncio.to_thread(_read)
        return _row_to_fact(row) if row else None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_facts/test_core_memory.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/lingxi/facts/models.py src/lingxi/facts/store.py tests/test_facts/test_core_memory.py
git commit -m "feat(facts): FactType.CORE + store.get_core_block"
```

---

### Task 2: retriever.get_core_block passthrough

**Files:**
- Modify: `src/lingxi/facts/retriever.py`
- Test: `tests/test_facts/test_core_memory.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_facts/test_core_memory.py`:
```python
@pytest.mark.asyncio
async def test_retriever_get_core_block(tmp_path):
    from lingxi.facts.retriever import FactRetriever
    s = await _store(tmp_path)
    await s.write(Fact(subject="aria", content="hello", source=Source.LLM_INFERRED,
                       type=FactType.CORE, ts=datetime(2026, 5, 1, 9, 0)))
    r = FactRetriever(s)
    block = await r.get_core_block("aria")
    assert block is not None and block.content == "hello"
    assert await r.get_core_block("aria-missing") is None
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_facts/test_core_memory.py::test_retriever_get_core_block -v`
Expected: FAIL with `AttributeError: 'FactRetriever' object has no attribute 'get_core_block'`

- [ ] **Step 3: Implement**

In `src/lingxi/facts/retriever.py`, add to `FactRetriever` (after `fetch_by_id`):
```python
    async def get_core_block(self, subject: str) -> Fact | None:
        """Current MemGPT core-memory block for subject (or None)."""
        return await self._store.get_core_block(subject)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_facts/test_core_memory.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lingxi/facts/retriever.py tests/test_facts/test_core_memory.py
git commit -m "feat(facts): retriever.get_core_block passthrough"
```

---

### Task 3: CoreMemoryWriter

**Files:**
- Create: `src/lingxi/facts/writers/core_memory.py`
- Test: `tests/test_facts/test_core_memory.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_facts/test_core_memory.py`:
```python
@pytest.mark.asyncio
async def test_core_writer_allows_aria_and_user(tmp_path):
    from lingxi.facts.writers.core_memory import CoreMemoryWriter
    s = await _store(tmp_path)
    w = CoreMemoryWriter(s)
    await w.write(subject="aria", content="self note", type=FactType.CORE,
                  source=Source.LLM_INFERRED, ts=datetime(2026, 5, 1, 9, 0))
    await w.write(subject="user:feishu:x", content="about him", type=FactType.CORE,
                  source=Source.LLM_INFERRED, ts=datetime(2026, 5, 1, 9, 0))
    assert (await s.get_core_block("aria")).content == "self note"
    assert (await s.get_core_block("user:feishu:x")).content == "about him"


@pytest.mark.asyncio
async def test_core_writer_rejects_foreign_subject(tmp_path):
    from lingxi.facts.writers.core_memory import CoreMemoryWriter
    s = await _store(tmp_path)
    w = CoreMemoryWriter(s)
    with pytest.raises(ValueError):
        await w.write(subject="npc:bob", content="x", type=FactType.CORE,
                      source=Source.LLM_INFERRED, ts=datetime(2026, 5, 1, 9, 0))
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_facts/test_core_memory.py::test_core_writer_allows_aria_and_user -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lingxi.facts.writers.core_memory'`

- [ ] **Step 3: Implement CoreMemoryWriter**

Create `src/lingxi/facts/writers/core_memory.py`:
```python
from lingxi.facts.models import Source
from lingxi.facts.writers.base import WriterBase


class CoreMemoryWriter(WriterBase):
    """MemGPT core-memory blocks. subject is aria (persona block) or
    user:<recipient_key> (human block). Edits supersede the prior block."""
    ALLOWED_SOURCE = Source.LLM_INFERRED
    SUBJECT_PATTERN = r"^(aria|user:[A-Za-z0-9_:-]+)$"
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_facts/test_core_memory.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lingxi/facts/writers/core_memory.py tests/test_facts/test_core_memory.py
git commit -m "feat(facts): CoreMemoryWriter (aria + user:* subjects)"
```

---

### Task 4: Provider tool-use support

**Files:**
- Modify: `src/lingxi/providers/base.py`
- Modify: `src/lingxi/providers/claude.py`
- Test: `tests/test_providers/test_tool_use.py`

- [ ] **Step 1: Extend CompletionResult**

In `src/lingxi/providers/base.py`, change the `CompletionResult` dataclass to:
```python
@dataclass
class CompletionResult:
    """Result from an LLM completion call."""

    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)
    finish_reason: str = ""
    tool_calls: list = field(default_factory=list)        # [{"id","name","input"}]
    raw_content_blocks: list = field(default_factory=list)  # original API content blocks
```

- [ ] **Step 2: Write failing test for body + parse**

Create `tests/test_providers/test_tool_use.py`:
```python
from lingxi.providers.claude import ClaudeProvider


def _provider():
    # api-key mode; we only test pure-function _build_body + _parse_content
    return ClaudeProvider(api_key="sk-test", model="claude-x")


def test_build_body_includes_tools():
    p = _provider()
    tools = [{"name": "t", "description": "d", "input_schema": {"type": "object", "properties": {}}}]
    body = p._build_body([{"role": "user", "content": "hi"}], None, 100, 0.7,
                         tools=tools, tool_choice={"type": "auto"})
    assert body["tools"] == tools
    assert body["tool_choice"] == {"type": "auto"}


def test_build_body_omits_tools_when_none():
    p = _provider()
    body = p._build_body([{"role": "user", "content": "hi"}], None, 100, 0.7)
    assert "tools" not in body
    assert "tool_choice" not in body


def test_parse_content_extracts_tool_use():
    p = _provider()
    blocks = [
        {"type": "text", "text": "thinking"},
        {"type": "tool_use", "id": "tu_1", "name": "archival_memory_search",
         "input": {"query": "stars"}},
    ]
    text, tool_calls = p._parse_content(blocks)
    assert text == "thinking"
    assert tool_calls == [{"id": "tu_1", "name": "archival_memory_search",
                           "input": {"query": "stars"}}]
```

- [ ] **Step 3: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_providers/test_tool_use.py -v`
Expected: FAIL — `_build_body() got an unexpected keyword argument 'tools'` and `_parse_content` missing.

- [ ] **Step 4: Implement in claude.py**

In `src/lingxi/providers/claude.py`, change `_build_body` signature and add tool fields:
```python
    def _build_body(
        self,
        messages: list[dict],
        system: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float | None = None,
        prefill: str = "",
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
    ) -> dict:
```
Right before `return body`, insert:
```python
        if tools:
            body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
```
Add a static parse helper (after `_build_body`):
```python
    @staticmethod
    def _parse_content(blocks: list[dict]) -> tuple[str, list[dict]]:
        """Split API content blocks into (joined_text, tool_calls)."""
        text = ""
        tool_calls: list[dict] = []
        for block in blocks:
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": block.get("input", {}),
                })
        return text, tool_calls
```
Then in `complete()`: thread `tools`/`tool_choice` through. Change the `_build_body(...)` call to:
```python
        body = self._build_body(
            messages, system, max_tokens, temperature, top_p, prefill,
            tools=kwargs.get("tools"), tool_choice=kwargs.get("tool_choice"),
        )
```
Replace the text-extraction block (the `content = ""` / `for block in data.get("content", [])` loop) with:
```python
        raw_blocks = data.get("content", [])
        content, tool_calls = self._parse_content(raw_blocks)
```
And update the returned `CompletionResult(...)` to include:
```python
        return CompletionResult(
            content=content,
            model=data.get("model", self.model),
            usage=usage,
            finish_reason=data.get("stop_reason", ""),
            tool_calls=tool_calls,
            raw_content_blocks=raw_blocks,
        )
```
(Keep the existing prefill-prepend + debug log lines unchanged; prefill only applies when no tools.)

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_providers/test_tool_use.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/lingxi/providers/base.py src/lingxi/providers/claude.py tests/test_providers/test_tool_use.py
git commit -m "feat(providers): Anthropic tool-use support (tools + tool_use parsing)"
```

---

### Task 5: MEMORY_TOOLS schema definitions

**Files:**
- Create: `src/lingxi/brain/memory_tools.py`
- Test: `tests/test_brain/test_memory_tools.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_brain/test_memory_tools.py`:
```python
from lingxi.brain.memory_tools import MEMORY_TOOLS, TOOL_NAMES


def test_five_tools_defined():
    assert TOOL_NAMES == {
        "archival_memory_search", "archival_memory_insert",
        "core_memory_append", "core_memory_replace", "conversation_search",
    }


def test_each_tool_has_valid_schema():
    for t in MEMORY_TOOLS:
        assert "name" in t and "description" in t
        assert t["input_schema"]["type"] == "object"
        assert "properties" in t["input_schema"]
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_brain/test_memory_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lingxi.brain.memory_tools'`

- [ ] **Step 3: Implement**

Create `src/lingxi/brain/memory_tools.py`:
```python
"""Anthropic tool schemas for MemGPT-style agent memory management.

The agent calls these mid-turn; engine._dispatch_memory_tool executes them,
mapping `scope` to a concrete subject by the current recipient_key so the
subject-ownership invariant holds (the agent cannot target arbitrary subjects).
"""

MEMORY_TOOLS = [
    {
        "name": "archival_memory_search",
        "description": "搜索你的长期记忆（facts.db）。当前上下文里没有、但你需要的细节，用这个查。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词/问题"},
                "scope": {"type": "string", "enum": ["self", "user", "world"],
                          "description": "self=你自己 user=当前对话对象 world=身边的人和世界"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "archival_memory_insert",
        "description": "把一条值得长期记住的事实写进长期记忆。只在确实重要时用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "scope": {"type": "string", "enum": ["self", "user"],
                          "description": "self=关于你自己 user=关于当前对话对象"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "core_memory_append",
        "description": "往常驻核心记忆块追加一行。persona=你的自我小结，human=你对当前对象的长期印象。",
        "input_schema": {
            "type": "object",
            "properties": {
                "block": {"type": "string", "enum": ["persona", "human"]},
                "content": {"type": "string"},
            },
            "required": ["block", "content"],
        },
    },
    {
        "name": "core_memory_replace",
        "description": "替换核心记忆块里的一段文字（用于更新/纠正/精简）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "block": {"type": "string", "enum": ["persona", "human"]},
                "old": {"type": "string", "description": "要替换掉的原文（必须是块里现有的子串）"},
                "new": {"type": "string"},
            },
            "required": ["block", "old", "new"],
        },
    },
    {
        "name": "conversation_search",
        "description": "搜索你和当前对象最近的对话记录。",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]

TOOL_NAMES = {t["name"] for t in MEMORY_TOOLS}
CORE_BLOCK_MAX_CHARS = 1500
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_brain/test_memory_tools.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lingxi/brain/memory_tools.py tests/test_brain/test_memory_tools.py
git commit -m "feat(brain): MEMORY_TOOLS schema definitions"
```

---

### Task 6: engine._dispatch_memory_tool

**Files:**
- Modify: `src/lingxi/conversation/engine.py`
- Test: `tests/test_conversation/test_tool_loop.py`

This routes a tool call to facts ops, scoped by `recipient_key`. Returns a string for the tool_result.

- [ ] **Step 1: Write failing test**

Create `tests/test_conversation/test_tool_loop.py`:
```python
import pytest
from pathlib import Path
from datetime import datetime

from lingxi.conversation.engine import ConversationEngine
from lingxi.memory.manager import MemoryManager
from lingxi.facts.store import FactStore
from lingxi.facts.retriever import FactRetriever
from lingxi.facts.writers.core_memory import CoreMemoryWriter
from lingxi.facts.writers.user_statement import UserStatementWriter
from lingxi.facts.writers.inference import InferenceWriter
from lingxi.facts.models import Fact, FactType, Source
from lingxi.persona.models import PersonaConfig, Identity


async def _engine(tmp_path):
    store = FactStore(Path(tmp_path) / "facts.db")
    await store.init()
    retr = FactRetriever(store)
    persona = PersonaConfig(name="Aria", identity=Identity(full_name="Aria"))

    class _LLM:  # not used by dispatch tests
        async def complete(self, **kw): ...

    eng = ConversationEngine(
        persona=persona, llm_provider=_LLM(),
        memory_manager=MemoryManager(data_dir=str(Path(tmp_path) / "mem")),
        fact_retriever=retr,
        inference_writer=InferenceWriter(store),
        user_statement_writer=UserStatementWriter(store),
        core_memory_writer=CoreMemoryWriter(store),
    )
    eng._current_recipient_key = "feishu:x"
    return eng, store


@pytest.mark.asyncio
async def test_dispatch_core_append_then_render(tmp_path):
    eng, store = await _engine(tmp_path)
    out = await eng._dispatch_memory_tool(
        "core_memory_append", {"block": "human", "content": "他在做天文"}, "feishu:x")
    assert "ok" in out.lower()
    block = await store.get_core_block("user:feishu:x")
    assert "他在做天文" in block.content


@pytest.mark.asyncio
async def test_dispatch_core_append_size_cap(tmp_path):
    eng, store = await _engine(tmp_path)
    big = "x" * 1600
    out = await eng._dispatch_memory_tool(
        "core_memory_append", {"block": "persona", "content": big}, "feishu:x")
    assert "full" in out.lower()
    # nothing persisted over the cap
    assert await store.get_core_block("aria") is None


@pytest.mark.asyncio
async def test_dispatch_insert_scopes_subject(tmp_path):
    eng, store = await _engine(tmp_path)
    await eng._dispatch_memory_tool(
        "archival_memory_insert", {"content": "喜欢美式", "scope": "user"}, "feishu:x")
    facts = await store.query(subject="user:feishu:x")
    assert any("美式" in f.content for f in facts)


@pytest.mark.asyncio
async def test_dispatch_search_returns_facts(tmp_path):
    eng, store = await _engine(tmp_path)
    await store.write(Fact(subject="aria", content="今天看了仙女座",
                           source=Source.LLM_INFERRED, type=FactType.EVENT,
                           ts=datetime(2026, 5, 1, 9, 0), importance=6))
    out = await eng._dispatch_memory_tool(
        "archival_memory_search", {"query": "仙女座", "scope": "self"}, "feishu:x")
    assert "仙女座" in out
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_conversation/test_tool_loop.py -v`
Expected: FAIL — `__init__() got an unexpected keyword argument 'core_memory_writer'` (then `_dispatch_memory_tool` missing once arg added).

- [ ] **Step 3: Add core_memory_writer to engine __init__**

In `src/lingxi/conversation/engine.py` `__init__`, add param `core_memory_writer=None,` (next to `user_statement_writer=None,`) and assignment `self.core_memory_writer = core_memory_writer` (next to `self.user_statement_writer = ...`).

- [ ] **Step 4: Implement _dispatch_memory_tool**

In `src/lingxi/conversation/engine.py`, add this method (place it right before `_prepare_turn_v2`):
```python
    async def _dispatch_memory_tool(self, name: str, args: dict, recipient_key: str) -> str:
        """Execute one MemGPT memory tool, scoped by recipient_key. Returns a
        string for the tool_result. Errors are returned (not raised) so the
        agent can recover."""
        from datetime import datetime
        from lingxi.brain.memory_tools import CORE_BLOCK_MAX_CHARS
        from lingxi.facts.models import Fact, FactType, Source
        from lingxi.facts.retriever import FactQuery

        def _subject_for(scope: str) -> str:
            if scope == "self":
                return "aria"
            if scope == "world":
                return "world"
            return f"user:{recipient_key}"

        try:
            if name == "archival_memory_search":
                scope = args.get("scope", "user")
                subject = _subject_for(scope)
                facts = await self.fact_retriever.fetch(FactQuery(
                    subject=subject, semantic=args.get("query"), limit=5))
                if not facts:
                    return "（没找到相关记忆）"
                return "\n".join(
                    f"- [{f.ts.strftime('%m-%d')}] {f.content}" for f in facts)

            if name == "archival_memory_insert":
                scope = args.get("scope", "user")
                subject = _subject_for(scope)
                writer = self.inference_writer if scope == "self" else self.user_statement_writer
                if writer is None:
                    return "（写入未启用）"
                await writer.write(
                    subject=subject, content=args["content"], type=FactType.PATTERN,
                    source=writer.ALLOWED_SOURCE, ts=datetime.now(),
                    importance=args.get("importance"))
                return "inserted"

            if name in ("core_memory_append", "core_memory_replace"):
                if self.core_memory_writer is None:
                    return "（核心记忆未启用）"
                block = args.get("block", "human")
                subject = "aria" if block == "persona" else f"user:{recipient_key}"
                current = await self.fact_retriever.get_core_block(subject)
                cur_text = current.content if current else ""
                if name == "core_memory_append":
                    new_text = (cur_text + "\n" + args["content"]).strip()
                else:
                    if args["old"] not in cur_text:
                        return "substring not found in core block"
                    new_text = cur_text.replace(args["old"], args["new"])
                if len(new_text) > CORE_BLOCK_MAX_CHARS:
                    return "core memory full, use core_memory_replace to condense"
                await self.core_memory_writer.write(
                    subject=subject, content=new_text, type=FactType.CORE,
                    source=Source.LLM_INFERRED, ts=datetime.now(),
                    supersedes=current.id if current else None)
                return "ok"

            if name == "conversation_search":
                turns = await self.memory.short_term.snapshot_for_recipient(recipient_key)
                q = args.get("query", "")
                hits = [t for t in turns if q in (t.content or "")][:8]
                if not hits:
                    return "（最近对话里没找到）"
                return "\n".join(
                    f"- [{t.timestamp.strftime('%m-%d %H:%M')}] "
                    f"{'对方' if t.role == 'user' else '我'}: {t.content[:80]}"
                    for t in hits)

            return f"unknown tool: {name}"
        except Exception as e:
            return f"tool error: {e}"
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_conversation/test_tool_loop.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add src/lingxi/conversation/engine.py tests/test_conversation/test_tool_loop.py
git commit -m "feat(engine): _dispatch_memory_tool (scope->subject, ownership-safe)"
```

---

### Task 7: engine._generate_with_tools agentic loop + wire into chat_full

**Files:**
- Modify: `src/lingxi/conversation/engine.py`
- Test: `tests/test_conversation/test_tool_loop.py`

- [ ] **Step 1: Write failing test (loop + runaway cap)**

Append to `tests/test_conversation/test_tool_loop.py`:
```python
class _ScriptedLLM:
    """Emits a queued sequence of CompletionResults."""
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    async def complete(self, **kwargs):
        self.calls += 1
        self._last_tool_choice = kwargs.get("tool_choice")
        return self._results.pop(0) if self._results else _final("forced")


def _toolcall(name, args, tid="t1"):
    from lingxi.providers.base import CompletionResult
    return CompletionResult(
        content="", finish_reason="tool_use",
        tool_calls=[{"id": tid, "name": name, "input": args}],
        raw_content_blocks=[{"type": "tool_use", "id": tid, "name": name, "input": args}],
    )


def _final(text):
    from lingxi.providers.base import CompletionResult
    return CompletionResult(content=text, finish_reason="end_turn")


@pytest.mark.asyncio
async def test_generate_with_tools_runs_tool_then_final(tmp_path):
    eng, store = await _engine(tmp_path)
    eng.llm = _ScriptedLLM([
        _toolcall("core_memory_append", {"block": "human", "content": "记一笔"}),
        _final("好的"),
    ])
    text = await eng._generate_with_tools(
        "SYS", [{"role": "user", "content": "hi"}], recipient_key="feishu:x")
    assert text == "好的"
    assert (await store.get_core_block("user:feishu:x")).content.endswith("记一笔")


@pytest.mark.asyncio
async def test_generate_with_tools_runaway_cap(tmp_path):
    eng, store = await _engine(tmp_path)
    # always tool_use → must stop at cap and force a text reply
    eng.llm = _ScriptedLLM([_toolcall("conversation_search", {"query": "x"}) for _ in range(20)])
    text = await eng._generate_with_tools(
        "SYS", [{"role": "user", "content": "hi"}], recipient_key="feishu:x")
    assert eng.llm.calls <= 6  # MAX_TOOL_ITERS(5) + 1 forced
    assert eng.llm._last_tool_choice == {"type": "none"}
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_conversation/test_tool_loop.py -k generate_with_tools -v`
Expected: FAIL — `_generate_with_tools` missing.

- [ ] **Step 3: Implement _generate_with_tools**

In `src/lingxi/conversation/engine.py`, add (right after `_dispatch_memory_tool`):
```python
    async def _generate_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        *,
        recipient_key: str,
        prefill: str = "",
        purpose: str = "chat_full",
    ) -> str:
        """Agentic generation loop: the model may call memory tools mid-turn.
        Returns the final user-facing text. Caps iterations to avoid runaway."""
        from lingxi.brain.memory_tools import MEMORY_TOOLS
        MAX_TOOL_ITERS = 5
        msgs = list(messages)
        iters = 0
        while True:
            tool_choice = {"type": "auto"} if iters < MAX_TOOL_ITERS else {"type": "none"}
            result = await self.llm.complete(
                messages=msgs,
                system=system_prompt,
                temperature=self.persona.sampling.temperature,
                top_p=self.persona.sampling.top_p,
                prefill=prefill if iters == 0 else "",
                tools=MEMORY_TOOLS,
                tool_choice=tool_choice,
                _debug_purpose=purpose,
            )
            if result.finish_reason != "tool_use" or not result.tool_calls:
                return result.content
            msgs.append({"role": "assistant", "content": result.raw_content_blocks})
            tool_results = []
            for call in result.tool_calls:
                out = await self._dispatch_memory_tool(
                    call["name"], call.get("input", {}), recipient_key)
                tool_results.append({
                    "type": "tool_result", "tool_use_id": call["id"], "content": out})
            msgs.append({"role": "user", "content": tool_results})
            iters += 1
```

- [ ] **Step 4: Wire into _chat_full_locked (both branches)**

In `src/lingxi/conversation/engine.py`, in `_chat_full_locked`, replace the generation block:
```python
        if self.persona.compression.enabled:
            # Two-call: think (Sonnet) then compress (Haiku)
            think_raw = await self._run_think(system_prompt, messages)
            output = self._process_response(think_raw)
            inner_thought = output.inner_thought or output.speech
            speech = await self._run_compress(inner_thought, user_input)
            output.speech = speech
            output.inner_thought = inner_thought
        else:
            prefill = pick_prefill(self.persona.style)
            result = await self.llm.complete(
                messages=messages,
                system=system_prompt,
                temperature=self.persona.sampling.temperature,
                top_p=self.persona.sampling.top_p,
                prefill=prefill,
                _debug_purpose="chat_full",
            )
            output = self._process_response(result.content)
```
with:
```python
        rkey = self._current_recipient_key or "_anon"
        if self.persona.compression.enabled:
            # Two-call: think (with memory tools) then compress (Haiku, no tools)
            think_raw = await self._generate_with_tools(
                system_prompt, messages, recipient_key=rkey, purpose="think")
            output = self._process_response(think_raw)
            inner_thought = output.inner_thought or output.speech
            speech = await self._run_compress(inner_thought, user_input)
            output.speech = speech
            output.inner_thought = inner_thought
        else:
            think_raw = await self._generate_with_tools(
                system_prompt, messages, recipient_key=rkey,
                prefill=pick_prefill(self.persona.style), purpose="chat_full")
            output = self._process_response(think_raw)
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_conversation/test_tool_loop.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: Run full conversation suite (no regression)**

Run: `.venv/bin/python -m pytest tests/test_conversation -v`
Expected: PASS (all)

- [ ] **Step 7: Commit**

```bash
git add src/lingxi/conversation/engine.py tests/test_conversation/test_tool_loop.py
git commit -m "feat(engine): agentic tool loop wired into chat_full (both branches)"
```

---

### Task 8: Renderer 【核心记忆】 section

**Files:**
- Modify: `src/lingxi/brain/renderer.py`
- Test: `tests/test_brain/test_memory_tools.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_brain/test_memory_tools.py`:
```python
import pytest
from pathlib import Path
from datetime import datetime

from lingxi.facts.store import FactStore
from lingxi.facts.retriever import FactRetriever
from lingxi.facts.models import Fact, FactType, Source
from lingxi.brain.models import OrchestrationDecision
from lingxi.brain.renderer import render_dynamic_blocks


@pytest.mark.asyncio
async def test_core_memory_block_rendered(tmp_path):
    s = FactStore(Path(tmp_path) / "facts.db")
    await s.init()
    await s.write(Fact(subject="aria", content="我是自由天文学家",
                       source=Source.LLM_INFERRED, type=FactType.CORE,
                       ts=datetime(2026, 5, 1, 9, 0)))
    await s.write(Fact(subject="user:feishu:x", content="他熬夜",
                       source=Source.LLM_INFERRED, type=FactType.CORE,
                       ts=datetime(2026, 5, 1, 9, 0)))
    r = FactRetriever(s)
    decision = OrchestrationDecision(register="warm", fact_queries=[], skip=[],
                                     topic_anchor="")
    out = await render_dynamic_blocks(r, decision, recipient_key="feishu:x")
    assert "核心记忆" in out
    assert "我是自由天文学家" in out
    assert "他熬夜" in out
```
(If `OrchestrationDecision` requires more fields, construct it the same way the existing renderer tests do — check `tests/test_brain/` for the canonical constructor.)

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest tests/test_brain/test_memory_tools.py::test_core_memory_block_rendered -v`
Expected: FAIL — assertion error, "核心记忆" not in output.

- [ ] **Step 3: Implement in renderer**

In `src/lingxi/brain/renderer.py`, inside `render_dynamic_blocks`, after the `sections: list[str] = []` line and BEFORE the 【你此刻】 block, insert core-memory rendering:
```python
    # 【核心记忆】 — always-present, agent-curated blocks (MemGPT main context).
    core_lines: list[str] = []
    persona_block = await retriever.get_core_block("aria")
    if persona_block and persona_block.content.strip():
        core_lines.append("你自己：\n" + persona_block.content.strip())
    human_block = await retriever.get_core_block(f"user:{recipient_key}")
    if human_block and human_block.content.strip():
        core_lines.append("关于对方：\n" + human_block.content.strip())
    if core_lines:
        sections.append("【核心记忆】（你长期记着的，自己维护的）\n" + "\n\n".join(core_lines))
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_brain/test_memory_tools.py -v`
Expected: PASS

- [ ] **Step 5: Run brain suite (no regression)**

Run: `.venv/bin/python -m pytest tests/test_brain -v`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add src/lingxi/brain/renderer.py tests/test_brain/test_memory_tools.py
git commit -m "feat(renderer): render 【核心记忆】 core-memory blocks"
```

---

### Task 9: app.py wiring

**Files:**
- Modify: `src/lingxi/app.py`
- Test: import smoke

- [ ] **Step 1: Construct + inject CoreMemoryWriter**

In `src/lingxi/app.py`, where the other writers are built (near `user_statement_writer = UserStatementWriter(...)`), add:
```python
    from lingxi.facts.writers.core_memory import CoreMemoryWriter
    core_memory_writer = CoreMemoryWriter(facts_store, scorer=importance_scorer)
```
In the `ConversationEngine(...)` constructor call, add the kwarg next to `user_statement_writer=user_statement_writer,`:
```python
        core_memory_writer=core_memory_writer,
```

- [ ] **Step 2: Import smoke + full suite**

Run: `.venv/bin/python -c "import lingxi.app, lingxi.conversation.engine, lingxi.brain.renderer, lingxi.providers.claude; print('import OK')"`
Expected: `import OK`

Run: `.venv/bin/python -m pytest -q`
Expected: all pass

- [ ] **Step 3: Commit**

```bash
git add src/lingxi/app.py
git commit -m "feat(app): wire CoreMemoryWriter into engine"
```

---

## Notes for the implementer

- **Scope→subject mapping is the security boundary.** Never let a tool write a subject the agent named directly; always derive it from `recipient_key` + `scope`/`block`. The tests in Task 6 lock this in.
- **The hybrid stays cheap:** most turns the orchestrator pre-warm is enough → the model returns `end_turn` with zero tool calls → exactly one LLM call (same cost as today). Tools only add round-trips when the agent decides it needs them.
- **Out of scope (SP2):** context-pressure management (token accounting + recursive summarize/flush), full-history recall persistence (conversation_search currently only sees the short_term window), proactive-path tools, retiring the writer-from-output path.
- After all tasks: run `.venv/bin/python -m pytest -q` and the import smoke once more, then restart the bot to observe tool calls in `data/debug/llm_requests/`.
