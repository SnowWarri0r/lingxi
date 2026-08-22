# 失败案例库 + 回放评测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把被指出的坏输出固化成可重放、可打分、不随时间腐坏的案例，作为一切改进的适应度信号。

**Architecture:** 案例是 YAML 数据（`evals/cases/`），跑它的代码在 `src/lingxi/evals/`。每个案例冻结时钟 + facts + 对话历史，回放时灌进临时 `FactStore`，走**真实的** `_prepare_turn_v2`（真 orchestrator、真 renderer、真组装）重新组装 prompt，再对 responder 并发采样 N 次，用确定性判定器打分。

**Tech Stack:** Python 3.12、pydantic、PyYAML（已是依赖）、pytest + pytest-asyncio、现有 `FactStore`/`FactRetriever`/`ConversationEngine`。

Spec: `docs/superpowers/specs/2026-08-20-eval-harness-design.md`

## Global Constraints

- **判定必须确定性**：本期不引入 LLM judge。
- **不进 `tests/`**：评测要联网、要花钱、结果随机，`pytest` 必须保持确定性、离线、免费。评测代码的**单测**可以进 `tests/test_evals/`，评测本身不行。
- **线上行为零变化**：所有时钟注入参数默认 `None` → `datetime.now()`。
- **天气必须 stub**（联网 + 外部可变量）；**日出日落不 stub**（纯离线计算，值得被测）。
- **`pass` 指标不参与 verdict**，仅作观测。
- **判定器宁可漏报不可误报**：误报会让人不信任分数。
- 提交信息用英文，代码注释用英文，文档与 CLI 输出用中文——遵循仓库现状。
- 每个任务结束时 `.venv/bin/python -m pytest -q` 必须全绿。起始基线 **557 passed, 3 skipped**。
  各任务写的是**本任务新增的用例数**，不是累计总数——累计数会被任何其他新增测试打乱，
  按增量核对更稳。

---

## File Structure

| 文件 | 职责 |
|---|---|
| `src/lingxi/evals/__init__.py` | 包标记 |
| `src/lingxi/evals/case.py` | `Case` 模型、YAML 加载、schema 校验、相对时间解析 |
| `src/lingxi/evals/detectors.py` | 判定器注册表，纯函数，无 IO |
| `src/lingxi/evals/runner.py` | 建临时 store/engine、冻结时钟、组装、前提校验、采样、打分 |
| `src/lingxi/evals/capture.py` | 从线上状态冻出 case 骨架 |
| `src/lingxi/evals/cli.py` | `lingxi-eval` 入口、表格输出、baseline 读写 |
| `evals/cases/*.yaml` | 案例数据 |
| `evals/baseline.json` | 上次记录的分数 |
| `tests/test_evals/` | 判定器、加载器、时钟冻结的单测（离线） |

时钟注入改动的既有文件：`facts/store.py`、`facts/retriever.py`、`conversation/context.py`、`brain/renderer.py`、`conversation/engine.py`。

---

## Task 1: 时钟注入

整套东西的地基。如果同一个案例两次组装出的 prompt 不同，后面所有分数都没有意义。

**设计要点**：`FactStore` / `FactRetriever` 收一个 `clock` 可调用对象（每个 eval 构造一次），**不是**给每个 `fetch()` 加参数——engine 里有十几处 `fact_retriever.fetch(...)`，逐个穿参会把改动摊得到处都是。组装路径（context / renderer / engine）用显式 `now` 参数，因为它天然是每轮一个值。

**Files:**
- Modify: `src/lingxi/facts/store.py:80-89`（`__init__`）、`:196`
- Modify: `src/lingxi/facts/retriever.py:30-31`（`__init__`）、`:63`
- Modify: `src/lingxi/conversation/context.py:79`、`:91`
- Modify: `src/lingxi/brain/renderer.py:80-87`、`:137`
- Modify: `src/lingxi/conversation/engine.py:424-430`、`:713`，以及其中调用 `assemble_messages` / `render_dynamic_blocks` 的两处
- Test: `tests/test_facts/test_clock_injection.py`、`tests/test_conversation/test_clock_injection.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `FactStore(db_path, *, clock: Callable[[], datetime] | None = None)`
  - `FactRetriever(store, *, clock: Callable[[], datetime] | None = None)`
  - `ContextAssembler.assemble_messages(memory_context, *, now: datetime | None = None)`
  - `render_dynamic_blocks(retriever, decision, *, recipient_key, persona=None, acquaintance=None, now: datetime | None = None)`
  - `ConversationEngine._prepare_turn_v2(user_input, images, channel, recipient_id, *, now: datetime | None = None)`

- [ ] **Step 1: 写失败测试——retriever 的 recency 打分吃注入时钟**

Create `tests/test_facts/test_clock_injection.py`:

```python
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.retriever import FactQuery, FactRetriever
from lingxi.facts.store import FactStore


FROZEN = datetime(2026, 8, 19, 20, 20, 54)


async def _store_with_two_facts(tmp_path: Path) -> FactStore:
    store = FactStore(tmp_path / "f.db")
    await store.init()
    # Same importance; only age separates them.
    await store.write(Fact(
        subject="aria", content="两小时前的事", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=FROZEN - timedelta(hours=2), importance=5,
    ))
    await store.write(Fact(
        subject="aria", content="三十天前的事", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=FROZEN - timedelta(days=30), importance=5,
    ))
    return store


@pytest.mark.asyncio
async def test_recency_ranking_uses_injected_clock(tmp_path):
    """Frozen at FROZEN, the 2-hour-old fact must outrank the 30-day-old one.

    Without injection the retriever scores against the real clock, so both
    facts age together as the case sits on disk and the ranking they encode
    silently drifts.
    """
    store = await _store_with_two_facts(tmp_path)
    retr = FactRetriever(store, clock=lambda: FROZEN)
    out = await retr.fetch(FactQuery(subject="aria", type=FactType.EVENT, limit=2))
    assert out[0].content == "两小时前的事"


@pytest.mark.asyncio
async def test_injected_clock_is_stable_across_calls(tmp_path):
    """Two fetches at the same frozen clock return the same order."""
    store = await _store_with_two_facts(tmp_path)
    retr = FactRetriever(store, clock=lambda: FROZEN)
    first = [f.content for f in await retr.fetch(
        FactQuery(subject="aria", type=FactType.EVENT, limit=2))]
    second = [f.content for f in await retr.fetch(
        FactQuery(subject="aria", type=FactType.EVENT, limit=2))]
    assert first == second


@pytest.mark.asyncio
async def test_default_clock_is_wall_clock(tmp_path):
    """Production path unchanged: no clock argument means datetime.now()."""
    store = FactStore(tmp_path / "f.db")
    await store.init()
    now = datetime.now()
    await store.write(Fact(
        subject="aria", content="刚刚", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=now, importance=5,
    ))
    retr = FactRetriever(store)
    out = await retr.fetch(FactQuery(subject="aria", type=FactType.EVENT, limit=1))
    assert out[0].content == "刚刚"


@pytest.mark.asyncio
async def test_store_expiry_uses_injected_clock(tmp_path):
    """A fact that expired before the frozen clock is filtered out."""
    store = FactStore(tmp_path / "f.db", clock=lambda: FROZEN)
    await store.init()
    await store.write(Fact(
        subject="aria", content="已过期", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=FROZEN - timedelta(days=2), importance=5,
        expires_at=FROZEN - timedelta(days=1),
    ))
    await store.write(Fact(
        subject="aria", content="没过期", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=FROZEN - timedelta(days=2), importance=5,
        expires_at=FROZEN + timedelta(days=1),
    ))
    rows = await store.query(subject="aria", limit=10)
    assert [r.content for r in rows] == ["没过期"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_facts/test_clock_injection.py -v`
Expected: FAIL — `TypeError: FactRetriever.__init__() got an unexpected keyword argument 'clock'`

- [ ] **Step 3: 给 store 和 retriever 加 clock**

`src/lingxi/facts/store.py`——修改 `__init__`（当前 80-83 行）：

```python
    def __init__(self, db_path: Path | str, *,
                 clock: Callable[[], datetime] | None = None):
        self._path = Path(db_path)
        self._lock = asyncio.Lock()
        # Injectable so an eval case can freeze expiry evaluation. Production
        # passes nothing and gets the wall clock.
        self._clock = clock or datetime.now
```

文件顶部补 import：

```python
from typing import Callable
```

修改第 196 行：

```python
            params.append(self._clock().isoformat())
```

`src/lingxi/facts/retriever.py`——修改 `__init__`（当前 30-31 行）：

```python
    def __init__(self, store: FactStore, *,
                 clock: Callable[[], datetime] | None = None):
        self._store = store
        # Recency scoring is exp(-0.01 * hours_old). Left on the wall clock,
        # a frozen eval case would see its facts age one week per week and
        # silently change which facts reach the prompt.
        self._clock = clock or datetime.now
```

文件顶部补 `from typing import Callable`（若尚未导入）。

修改第 63 行：

```python
        now = self._clock()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_facts/test_clock_injection.py -v`
Expected: 4 passed

- [ ] **Step 5: 写失败测试——组装路径两次产出逐字节相同**

Create `tests/test_conversation/test_clock_injection.py`:

```python
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from lingxi.conversation.engine import ConversationEngine
from lingxi.facts.retriever import FactRetriever
from lingxi.facts.store import FactStore
from lingxi.memory.manager import MemoryManager
from lingxi.persona.models import Identity, PersonaConfig


FROZEN = datetime(2026, 8, 19, 20, 20, 54)


async def _engine(tmp_path):
    store = FactStore(Path(tmp_path) / "facts.db", clock=lambda: FROZEN)
    await store.init()
    retr = FactRetriever(store, clock=lambda: FROZEN)
    persona = PersonaConfig(name="Aria", identity=Identity(full_name="Aria"))

    class _LLM:
        async def complete(self, **kw): ...

    return ConversationEngine(
        persona=persona, llm_provider=_LLM(),
        memory_manager=MemoryManager(data_dir=str(Path(tmp_path) / "mem")),
        fact_retriever=retr,
    )


@pytest.mark.asyncio
async def test_frozen_clock_makes_assembly_byte_identical(tmp_path, monkeypatch):
    """Same case, same clock, two assemblies — identical system and messages.

    This is the foundation of the whole harness: if two runs of one case do
    not produce the same prompt, every score downstream is meaningless.
    """
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain import renderer as rend_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(
            register="light", engage_level=0.5, fact_queries=[], skip=[],
            topic_anchor="anchor", user_state="还在公司")

    async def _fake_render(*a, **k):
        return "【你此刻】固定内容"

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    monkeypatch.setattr(rend_mod, "render_dynamic_blocks", _fake_render)

    eng = await _engine(tmp_path)
    eng.memory.add_turn("user", "想下班了")
    eng.memory.short_term.get_history()[-1].timestamp = FROZEN - timedelta(minutes=2)

    sys_a, msgs_a = await eng._prepare_turn_v2(
        "在干嘛", None, "feishu", "x", now=FROZEN)
    sys_b, msgs_b = await eng._prepare_turn_v2(
        "在干嘛", None, "feishu", "x", now=FROZEN)

    assert sys_a == sys_b
    assert msgs_a[-1]["content"] == msgs_b[-1]["content"]


@pytest.mark.asyncio
async def test_frozen_clock_reaches_the_time_block(tmp_path, monkeypatch):
    """The reminder states the frozen time, not the wall clock."""
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain import renderer as rend_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(
            register="light", engage_level=0.5, fact_queries=[], skip=[],
            topic_anchor="")

    async def _fake_render(*a, **k):
        return ""

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    monkeypatch.setattr(rend_mod, "render_dynamic_blocks", _fake_render)

    eng = await _engine(tmp_path)
    _sys, msgs = await eng._prepare_turn_v2(
        "在干嘛", None, "feishu", "x", now=FROZEN)
    text = msgs[-1]["content"]
    assert "2026-08-19 20:20" in text


@pytest.mark.asyncio
async def test_date_dividers_use_injected_clock(tmp_path):
    """A turn from the frozen day reads 今天, regardless of the wall clock."""
    from lingxi.conversation.context import ContextAssembler
    from lingxi.memory.manager import MemoryContext
    from lingxi.memory.short_term import ConversationTurn

    turns = [
        ConversationTurn(role="user", content="早",
                         timestamp=FROZEN - timedelta(hours=3)),
        ConversationTurn(role="assistant", content="早呀",
                         timestamp=FROZEN - timedelta(hours=3)),
    ]
    ctx = MemoryContext(short_term_turns=turns)
    out = ContextAssembler().assemble_messages(ctx, now=FROZEN)
    divider = next(m for m in out if m["content"].startswith("[——"))
    assert "今天" in divider["content"]
```

若 `MemoryContext` 的构造参数名与此不符，以 `src/lingxi/memory/manager.py` 中的定义为准，只改这一行。

- [ ] **Step 6: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_conversation/test_clock_injection.py -v`
Expected: FAIL — `_prepare_turn_v2() got an unexpected keyword argument 'now'`

- [ ] **Step 7: 给组装路径加 now 参数**

`src/lingxi/conversation/context.py`——修改 `assemble_messages`（第 79 行）：

```python
    def assemble_messages(self, memory_context: MemoryContext, *,
                          now: datetime | None = None) -> list[dict]:
```

修改第 91 行：

```python
        now = now or datetime.now()
```

注意该方法内部第 90 行有 `from datetime import date, datetime, timedelta` 的局部导入，会遮蔽参数名以外的模块引用——把这行局部导入删掉，改在文件顶部统一导入 `date, datetime, timedelta`（顶部当前已导入 `date, datetime`，补 `timedelta`）。

`src/lingxi/brain/renderer.py`——修改 `render_dynamic_blocks` 签名（第 80-87 行），在 `acquaintance` 后追加：

```python
    now: datetime | None = None,
```

修改第 137 行：

```python
        days = max(0, ((now or datetime.now()).date() - first.date()).days)
```

`src/lingxi/conversation/engine.py`——修改 `_prepare_turn_v2` 签名（第 424-430 行）：

```python
    async def _prepare_turn_v2(
        self,
        user_input: str,
        images: list[dict] | None,
        channel: str | None,
        recipient_id: str | None,
        *,
        now: datetime | None = None,
    ) -> tuple[str, list[dict]]:
```

在方法体开头（`self._pending_stickers[recipient_key] = None` 之前）插入：

```python
        # One clock for the whole turn. Defaults to the wall clock, so
        # production behaviour is unchanged; an eval case pins it so the
        # assembled prompt is reproducible.
        now = now or datetime.now()
```

把该方法内两处调用改为传 `now`：

```python
        messages = self.context_assembler.assemble_messages(
            memory_context, now=now)
```

```python
        dynamic_block = await render_dynamic_blocks(
            self.fact_retriever, decision, recipient_key=recipient_key,
            persona=self.persona,
            acquaintance=getattr(self, "_acquaintance", None),
            now=now,
        )
```

`_build_focus_reminder` 增加 `now` 参数并透传（当前第 713 行 `current_time=datetime.now()`）：

```python
    def _build_focus_reminder(
        self,
        last_interaction_time: datetime | None,
        state_blocks: list[str] | None = None,
        user_schedule: list[str] | None = None,
        user_state: str = "",
        now: datetime | None = None,
    ) -> str | None:
```

```python
            current_time=now or datetime.now(),
```

调用处补 `now=now`：

```python
        focus = self._build_focus_reminder(
            last_interaction_time, state_blocks=state_blocks,
            user_schedule=await self._user_schedule_facts(recipient_key),
            user_state=decision.user_state, now=now)
```

另有一处 `if self.fact_retriever is None:` 的降级分支同样调用 `assemble_messages`，一并补 `now=now`。

- [ ] **Step 8: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_conversation/test_clock_injection.py tests/test_facts/test_clock_injection.py -v`
Expected: 7 passed

- [ ] **Step 9: 全量回归**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m ruff check src/lingxi/`
Expected: 全绿，较基线 **新增 7 条**（本任务的 4 + 3）；ruff 错误数不超过改动前的 28（此前存量）

- [ ] **Step 10: 提交**

```bash
git add src/lingxi/facts/store.py src/lingxi/facts/retriever.py \
        src/lingxi/conversation/context.py src/lingxi/brain/renderer.py \
        src/lingxi/conversation/engine.py \
        tests/test_facts/test_clock_injection.py \
        tests/test_conversation/test_clock_injection.py
git commit -m "feat(clock): make the assembly path's clock injectable

Groundwork for replayable eval cases: a case that freezes only the prompt
clock still lets retriever.py score fact recency off the wall clock, so its
facts age a week per week and the set that reaches the prompt drifts. Store
and retriever take a clock callable (constructed once per eval); the
per-turn assembly path takes an explicit now. Both default to
datetime.now(), so production is unchanged.

Also removes the shadowing local datetime import in assemble_messages."
```

---

## Task 2: Case 模型与加载器

**Files:**
- Create: `src/lingxi/evals/__init__.py`
- Create: `src/lingxi/evals/case.py`
- Create: `tests/test_evals/__init__.py`
- Test: `tests/test_evals/test_case_loader.py`

**Interfaces:**
- Consumes: `lingxi.facts.models.Fact/FactType/Source`
- Produces:
  - `class CaseFact(BaseModel)` — `subject, type, source, content, importance, days_ago, minutes_ago`
  - `class CaseTurn(BaseModel)` — `role, content, minutes_ago`
  - `class Premise(BaseModel)` — `prompt_contains: list[str]`, `prompt_lacks: list[str]`
  - `class Case(BaseModel)` — `id, symptom, origin, persona, recipient, clock, facts, history, input, samples, premise, detect, budget`
  - `Case.resolved_facts() -> list[Fact]` — 相对时间换算成绝对 `ts`，`expires_at=None`
  - `Case.resolved_history() -> list[tuple[str, str, datetime]]`
  - `load_case(path: Path) -> Case`
  - `load_all_cases(dir: Path = Path("evals/cases")) -> list[Case]`

- [ ] **Step 1: 写失败测试**

Create `tests/test_evals/__init__.py`（空文件）和 `tests/test_evals/test_case_loader.py`:

```python
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from lingxi.evals.case import Case, load_all_cases, load_case

YAML = """
id: offwork-state
symptom: 他说想下班 90 秒后，她问是在堵车还是到家了
origin: 2026-08-19 飞书对话
persona: config/personas/tangkeke.yaml
recipient: feishu:oc_eval
clock: "2026-08-19T20:20:54"
facts:
  - subject: "user:feishu:oc_eval"
    type: pattern
    source: user_stated
    content: 对方一般晚上九点下班
    importance: 4
    days_ago: 12
history:
  - {role: user, content: 想下班了, minutes_ago: 2}
  - {role: assistant, content: 想下班的心我懂, minutes_ago: 2}
input: 大学还不轻松啊
samples: 20
premise:
  prompt_contains: ["下班后的个人时间"]
  prompt_lacks: ["还在公司"]
detect:
  fail: {any_of: [堵车, 到家]}
  pass: {any_of: [快下班]}
budget: {max_fail_rate: 0.05}
"""


def _write(tmp_path: Path, text: str, name: str = "c.yaml") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_loads_all_fields(tmp_path):
    case = load_case(_write(tmp_path, YAML))
    assert case.id == "offwork-state"
    assert case.clock == datetime(2026, 8, 19, 20, 20, 54)
    assert case.samples == 20
    assert case.budget.max_fail_rate == 0.05
    assert case.premise.prompt_contains == ["下班后的个人时间"]


def test_relative_fact_time_resolves_against_the_frozen_clock(tmp_path):
    case = load_case(_write(tmp_path, YAML))
    fact = case.resolved_facts()[0]
    assert fact.ts == datetime(2026, 8, 7, 20, 20, 54)
    assert fact.subject == "user:feishu:oc_eval"
    assert fact.importance == 4


def test_resolved_facts_never_expire(tmp_path):
    """A case left on disk for a month must not lose facts to the expiry
    filter — that would silently change what the case tests."""
    case = load_case(_write(tmp_path, YAML))
    assert all(f.expires_at is None for f in case.resolved_facts())


def test_relative_history_time_resolves(tmp_path):
    case = load_case(_write(tmp_path, YAML))
    role, content, ts = case.resolved_history()[0]
    assert (role, content) == ("user", "想下班了")
    assert ts == datetime(2026, 8, 19, 20, 18, 54)


def test_samples_defaults_to_20(tmp_path):
    case = load_case(_write(tmp_path, YAML.replace("samples: 20\n", "")))
    assert case.samples == 20


def test_missing_required_field_raises(tmp_path):
    with pytest.raises(ValidationError):
        load_case(_write(tmp_path, YAML.replace("id: offwork-state\n", "")))


def test_unknown_field_raises(tmp_path):
    """Typos must fail loudly — a silently ignored key is a case that tests
    something other than what its author wrote."""
    with pytest.raises(ValidationError):
        load_case(_write(tmp_path, YAML + "\nsampels: 5\n"))


def test_load_all_reads_a_directory(tmp_path):
    _write(tmp_path, YAML, "a.yaml")
    _write(tmp_path, YAML.replace("offwork-state", "second"), "b.yaml")
    ids = sorted(c.id for c in load_all_cases(tmp_path))
    assert ids == ["offwork-state", "second"]


def test_duplicate_ids_raise(tmp_path):
    _write(tmp_path, YAML, "a.yaml")
    _write(tmp_path, YAML, "b.yaml")
    with pytest.raises(ValueError, match="duplicate case id"):
        load_all_cases(tmp_path)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_evals/test_case_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lingxi.evals'`

- [ ] **Step 3: 实现 case.py**

Create `src/lingxi/evals/__init__.py`（空文件）。

Create `src/lingxi/evals/case.py`:

```python
"""Case schema: a frozen turn that can be replayed and scored.

A case pins everything upstream of the prompt — the clock, the facts, the
conversation — so that replaying it next month assembles the same prompt it
assembled today. Times are stored relative to the frozen clock; absolute
timestamps make a case read like archaeology and have to be edited in bulk
whenever the clock moves.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from lingxi.facts.models import Fact, FactType, Source


class _Strict(BaseModel):
    # A silently ignored key is a case that tests something other than what
    # its author wrote. Typos fail loudly instead.
    model_config = ConfigDict(extra="forbid")


class CaseFact(_Strict):
    subject: str
    type: FactType
    source: Source
    content: str
    importance: int | None = None
    days_ago: float = 0.0
    minutes_ago: float = 0.0


class CaseTurn(_Strict):
    role: str
    content: str
    minutes_ago: float = 0.0


class Premise(_Strict):
    prompt_contains: list[str] = Field(default_factory=list)
    prompt_lacks: list[str] = Field(default_factory=list)


class Budget(_Strict):
    max_fail_rate: float = 0.05


class Detect(_Strict):
    fail: dict
    passing: dict | None = Field(default=None, alias="pass")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Case(_Strict):
    id: str
    symptom: str
    persona: str
    recipient: str
    clock: datetime
    input: str
    detect: Detect
    origin: str = ""
    facts: list[CaseFact] = Field(default_factory=list)
    history: list[CaseTurn] = Field(default_factory=list)
    samples: int = 20
    premise: Premise = Field(default_factory=Premise)
    budget: Budget = Field(default_factory=Budget)

    def resolved_facts(self) -> list[Fact]:
        """Case facts as real Facts, timed against the frozen clock.

        expires_at is left None on purpose: a case sitting on disk for a
        month must not quietly lose facts to the expiry filter.
        """
        out: list[Fact] = []
        for f in self.facts:
            ts = self.clock - timedelta(days=f.days_ago, minutes=f.minutes_ago)
            out.append(Fact(
                subject=f.subject, content=f.content, source=f.source,
                type=f.type, ts=ts, importance=f.importance, expires_at=None,
            ))
        return out

    def resolved_history(self) -> list[tuple[str, str, datetime]]:
        return [
            (t.role, t.content, self.clock - timedelta(minutes=t.minutes_ago))
            for t in self.history
        ]


def load_case(path: Path | str) -> Case:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Case.model_validate(data)


def load_all_cases(directory: Path | str = Path("evals/cases")) -> list[Case]:
    cases = [load_case(p) for p in sorted(Path(directory).glob("*.yaml"))]
    seen: set[str] = set()
    for c in cases:
        if c.id in seen:
            raise ValueError(f"duplicate case id: {c.id}")
        seen.add(c.id)
    return cases
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_evals/test_case_loader.py -v`
Expected: 9 passed

- [ ] **Step 5: 提交**

```bash
git add src/lingxi/evals/__init__.py src/lingxi/evals/case.py \
        tests/test_evals/__init__.py tests/test_evals/test_case_loader.py
git commit -m "feat(evals): case schema and loader

Times are relative to the frozen clock so a case stays readable and moves
in one edit. Facts resolve with expires_at=None — a case left on disk for a
month must not quietly lose facts to the expiry filter and start testing
something else. Unknown keys are rejected: a silently ignored typo is a
case that tests something other than what its author wrote."
```

---

## Task 3: 判定器

> **AMENDMENT (2026-08-20, after execution).** `dates_outside_anchors` 已实现、
> 经三轮评审后**删除**。每一轮都冒出一类新的误报：节日、问今天几号、第三人的日期、
> 假设句、正确断言被同句无关内容污染、无标点长句击穿分句、`我` 作定语修饰别人
> （`我朋友是1月10号加入的`）、否认句（`我们又不是1月1号加入的好嘛`）。
> 判断"这个日期是不是她在编自己的历史"需要主语、时态、否定——是语义不是子串。
> 本模块的硬约束是**宁可漏报不可误报**，一个不断长出新误报类别的判定器达不到这条。
> 乱编检测推迟到 LLM judge 那一期。删除提交 `3b08234`。
> **下面 Task 3 的原文保留作为记录**，实际交付的是 `any_of` 与 `regex` 两个判定器。


**Files:**
- Create: `src/lingxi/evals/detectors.py`
- Test: `tests/test_evals/test_detectors.py`

**Interfaces:**
- Consumes: `lingxi.persona.models.PersonaConfig`
- Produces: `evaluate(spec: dict, reply: str, persona=None) -> bool` — 命中返回 `True`

- [ ] **Step 1: 写失败测试**

Create `tests/test_evals/test_detectors.py`:

```python
import pytest

from lingxi.evals.detectors import evaluate
from lingxi.persona.loader import load_persona


def test_any_of_hits_on_substring():
    assert evaluate({"any_of": ["堵车", "到家"]}, "你还在堵车上吗") is True


def test_any_of_misses_cleanly():
    assert evaluate({"any_of": ["堵车", "到家"]}, "你今天累不累") is False


def test_any_of_empty_list_never_hits():
    assert evaluate({"any_of": []}, "随便什么") is False


def test_regex_hits():
    assert evaluate({"regex": r"\d+点\d*下班"}, "你不是九点下班嘛") is True


def test_regex_misses():
    assert evaluate({"regex": r"\d+点\d*下班"}, "你几点下班呀") is False


def test_dates_outside_anchors_flags_an_invented_date():
    """She has no anchor on 2021-02-14; stating it is fabrication."""
    persona = load_persona("config/personas/tangkeke.yaml")
    hit = evaluate({"dates_outside_anchors": True},
                   "我是2021年2月14号被选上的", persona)
    assert hit is True


def test_dates_outside_anchors_accepts_a_real_anchor():
    """2021-04-07 is her debut, listed in anchors."""
    persona = load_persona("config/personas/tangkeke.yaml")
    hit = evaluate({"dates_outside_anchors": True},
                   "我们2021年4月7号出道的", persona)
    assert hit is False


def test_dates_outside_anchors_ignores_text_without_dates():
    persona = load_persona("config/personas/tangkeke.yaml")
    assert evaluate({"dates_outside_anchors": True}, "今天好热啊", persona) is False


def test_unknown_detector_raises():
    with pytest.raises(ValueError, match="unknown detector"):
        evaluate({"vibes": True}, "随便什么")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_evals/test_detectors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lingxi.evals.detectors'`

- [ ] **Step 3: 实现 detectors.py**

```python
"""Deterministic detectors. Pure functions, no IO, no model calls.

Deliberately narrow. A detector that fires on something it should not makes
the whole score untrustworthy, and an untrusted score is worse than no
score — nobody acts on it. Missing a variant is cheap by comparison: it
shows up as the same symptom again and gets a second detector.
"""

from __future__ import annotations

import re

# 2021年4月7号 / 2021年4月7日 / 4月7号 — the shapes she actually writes.
_DATE_RE = re.compile(r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})[日号]")


def _any_of(needles: list[str], reply: str) -> bool:
    return any(n in reply for n in needles)


def _regex(pattern: str, reply: str) -> bool:
    return re.search(pattern, reply) is not None


def _dates_outside_anchors(reply: str, persona) -> bool:
    """True when the reply states a calendar date the persona has no anchor for.

    Only month-day pairs count. A bare year ("2021年出道") is how anyone
    talks about their own past and is not evidence of fabrication.
    """
    if persona is None:
        return False
    anchored: set[tuple[int, int]] = set()
    for anchor in (getattr(persona, "anchors", None) or []):
        raw = getattr(anchor, "date", None)
        if not raw:
            continue
        parts = str(raw).split("-")
        if len(parts) == 3:
            anchored.add((int(parts[1]), int(parts[2])))
    birthdate = getattr(getattr(persona, "identity", None), "birthdate", None)
    if birthdate:
        parts = str(birthdate).split("-")
        if len(parts) == 3:
            anchored.add((int(parts[1]), int(parts[2])))

    for _year, month, day in _DATE_RE.findall(reply):
        if (int(month), int(day)) not in anchored:
            return True
    return False


def evaluate(spec: dict, reply: str, persona=None) -> bool:
    """True when this detector fires on `reply`."""
    if "any_of" in spec:
        return _any_of(spec["any_of"], reply)
    if "regex" in spec:
        return _regex(spec["regex"], reply)
    if "dates_outside_anchors" in spec:
        return _dates_outside_anchors(reply, persona)
    raise ValueError(f"unknown detector: {sorted(spec)}")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_evals/test_detectors.py -v`
Expected: 10 passed

若 `test_dates_outside_anchors_accepts_a_real_anchor` 失败，先用
`.venv/bin/python -c "from lingxi.persona.loader import load_persona; p=load_persona('config/personas/tangkeke.yaml'); print([a.date for a in p.anchors])"`
确认锚点日期格式，再调整解析而非放宽断言。

- [ ] **Step 5: 提交**

```bash
git add src/lingxi/evals/detectors.py tests/test_evals/test_detectors.py
git commit -m "feat(evals): deterministic detectors

any_of, regex, and dates_outside_anchors. The last one only counts
month-day pairs: a bare year is how anyone talks about their own past, not
evidence of fabrication. Detectors stay narrow on purpose — a false
positive makes the whole score untrustworthy, and a score nobody trusts is
worse than no score. A missed variant just resurfaces as the same symptom."
```

---

## Task 4: Runner

**Files:**
- Create: `src/lingxi/evals/runner.py`
- Test: `tests/test_evals/test_runner.py`（离线，responder 用假实现）

**Interfaces:**
- Consumes: Task 1 的时钟参数、Task 2 的 `Case`、Task 3 的 `evaluate`
- Produces:
  - `@dataclass class CaseScore` — `id, verdict, fail_rate, pass_rate, samples, premise_ok, premise_error, replies: list[str]`
  - `async def build_turn(case, *, overrides=None) -> tuple[str, list[dict], PersonaConfig]`
  - `async def score_case(case, *, overrides=None, sampler=None) -> CaseScore`

`sampler` 是可注入的采样函数 `async (system, messages, n) -> list[str]`，默认打真实 responder。注入之后 runner 的全部逻辑都能离线测。

- [ ] **Step 1: 写失败测试**

Create `tests/test_evals/test_runner.py`:

```python
from pathlib import Path

import pytest

from lingxi.evals.case import load_case
from lingxi.evals.runner import build_turn, score_case

YAML = """
id: t
symptom: s
persona: config/personas/tangkeke.yaml
recipient: feishu:oc_eval
clock: "2026-08-19T20:20:54"
facts:
  - {subject: "user:feishu:oc_eval", type: pattern, source: user_stated,
     content: 对方一般晚上九点下班, importance: 4, days_ago: 12}
history:
  - {role: user, content: 想下班了, minutes_ago: 2}
input: 大学还不轻松啊
samples: 4
premise:
  prompt_contains: ["2026-08-19 20:20"]
detect:
  fail: {any_of: [堵车]}
  pass: {any_of: [还在公司]}
budget: {max_fail_rate: 0.25}
"""


def _case(tmp_path, text=YAML):
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    return load_case(p)


def _sampler(replies):
    async def _s(system, messages, n):
        return (replies * n)[:n]
    return _s


class _StubLLM:
    """The orchestrator is monkeypatched in these tests, so nothing calls it.

    Injected anyway: the real _main_llm() resolves OAuth credentials, which
    an offline test must not depend on.
    """

    async def complete(self, **kwargs):
        raise AssertionError("orchestrator should be monkeypatched in tests")


@pytest.mark.asyncio
async def test_build_turn_puts_case_facts_in_reach(tmp_path, monkeypatch):
    """The case's own facts must be the ones assembled — not the live db."""
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(
            register="warm", engage_level=0.6, fact_queries=[], skip=[],
            topic_anchor="")

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    system, messages, persona = await build_turn(_case(tmp_path), llm=_StubLLM())
    assert "唐可可" in system or persona.name
    assert "九点下班" in messages[-1]["content"]


@pytest.mark.asyncio
async def test_two_builds_are_identical(tmp_path, monkeypatch):
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(
            register="warm", engage_level=0.6, fact_queries=[], skip=[],
            topic_anchor="")

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    a_sys, a_msgs, _ = await build_turn(_case(tmp_path), llm=_StubLLM())
    b_sys, b_msgs, _ = await build_turn(_case(tmp_path), llm=_StubLLM())
    assert a_sys == b_sys
    assert a_msgs[-1]["content"] == b_msgs[-1]["content"]


@pytest.mark.asyncio
async def test_pass_verdict_when_under_budget(tmp_path, monkeypatch):
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(
            register="warm", engage_level=0.6, fact_queries=[], skip=[],
            topic_anchor="")

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    score = await score_case(
        _case(tmp_path), sampler=_sampler(["还在公司蹲着"]), llm=_StubLLM())
    assert score.verdict == "PASS"
    assert score.fail_rate == 0.0
    assert score.pass_rate == 1.0


@pytest.mark.asyncio
async def test_fail_verdict_when_over_budget(tmp_path, monkeypatch):
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(
            register="warm", engage_level=0.6, fact_queries=[], skip=[],
            topic_anchor="")

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    score = await score_case(_case(tmp_path), sampler=_sampler(["还在堵车"]),
                             llm=_StubLLM())
    assert score.verdict == "FAIL"
    assert score.fail_rate == 1.0


@pytest.mark.asyncio
async def test_broken_when_premise_fails(tmp_path, monkeypatch):
    """A case whose premise no longer holds reports BROKEN, never PASS."""
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(
            register="warm", engage_level=0.6, fact_queries=[], skip=[],
            topic_anchor="")

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    bad = YAML.replace('["2026-08-19 20:20"]', '["这句话不可能出现在 prompt 里"]')
    score = await score_case(_case(tmp_path, bad), sampler=_sampler(["随便"]),
                             llm=_StubLLM())
    assert score.verdict == "BROKEN"
    assert score.premise_ok is False
    assert "这句话不可能出现在 prompt 里" in score.premise_error


@pytest.mark.asyncio
async def test_pass_rate_does_not_affect_verdict(tmp_path, monkeypatch):
    """pass is observation only: 0 pass hits with 0 fail hits is still PASS."""
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain.models import OrchestrationDecision

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(
            register="warm", engage_level=0.6, fact_queries=[], skip=[],
            topic_anchor="")

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    score = await score_case(_case(tmp_path), sampler=_sampler(["今天好热"]),
                             llm=_StubLLM())
    assert score.verdict == "PASS"
    assert score.pass_rate == 0.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_evals/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lingxi.evals.runner'`

- [ ] **Step 3: 实现 runner.py**

```python
"""Replay a case and score it.

The point of replaying rather than storing finished messages: three of the
four fixes made on 2026-08-19 lived in the assembly layer (a lexicon entry,
a prompt reorder, a new orchestrator field). A harness that froze the
assembled messages would have tested none of them.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from lingxi.conversation.engine import RESPONDER_PRESETS, ConversationEngine
from lingxi.evals.case import Case
from lingxi.evals.detectors import evaluate
from lingxi.facts.retriever import FactRetriever
from lingxi.facts.store import FactStore
from lingxi.memory.manager import MemoryManager
from lingxi.persona.loader import load_persona
from lingxi.persona.models import PersonaConfig


@dataclass
class CaseScore:
    id: str
    verdict: str                      # PASS | FAIL | BROKEN
    fail_rate: float = 0.0
    pass_rate: float = 0.0
    samples: int = 0
    premise_ok: bool = True
    premise_error: str = ""
    replies: list[str] = field(default_factory=list)


def _apply_overrides(persona: PersonaConfig, overrides: dict | None) -> PersonaConfig:
    """Return a persona copy with top-level fields replaced.

    This is the seam for running candidates in bulk (spec §10): same case,
    different persona wording, one score each.
    """
    if not overrides:
        return persona
    return persona.model_copy(update=dict(overrides))


async def build_turn(
    case: Case, *, overrides: dict | None = None, llm=None,
) -> tuple[str, list[dict], PersonaConfig]:
    """Assemble the case's turn through the real pipeline.

    `llm` is the provider the orchestrator runs on; tests inject a stub.
    """
    persona = _apply_overrides(load_persona(case.persona), overrides)
    channel, _, recipient_id = case.recipient.partition(":")

    with tempfile.TemporaryDirectory(prefix="lingxi-eval-") as tmp:
        tmp_path = Path(tmp)
        clock = lambda: case.clock          # noqa: E731 — one-liner by design
        store = FactStore(tmp_path / "facts.db", clock=clock)
        await store.init()
        for fact in case.resolved_facts():
            await store.write(fact)
        retriever = FactRetriever(store, clock=clock)

        engine = ConversationEngine(
            persona=persona,
            llm_provider=llm or await _main_llm(),
            memory_manager=MemoryManager(data_dir=str(tmp_path / "mem")),
            fact_retriever=retriever,
        )
        await engine.memory.short_term.switch_recipient(case.recipient)
        for role, content, ts in case.resolved_history():
            turn = engine.memory.add_turn(role, content)
            turn.timestamp = ts

        system, messages = await engine._prepare_turn_v2(
            case.input, None, channel, recipient_id, now=case.clock,
        )
        return system, messages, persona


async def _main_llm():
    """The orchestrator's provider, built through the app's own auth path.

    Constructing ClaudeProvider() directly succeeds but carries no key, so
    the first real run would fail on the orchestrator call rather than at
    setup. Reuse the resolution app.create_engine uses.
    """
    from lingxi.auth.models import AuthMethod
    from lingxi.providers.registry import ProviderRegistry
    from lingxi.app import _build_auth_manager
    from lingxi.utils.config import load_config

    config = load_config("config/default.yaml")
    ProviderRegistry.register_defaults()
    return await ProviderRegistry.create_llm_with_auth(
        "claude", auth_manager=_build_auth_manager(config),
        auth_method=AuthMethod("oauth_pkce"), model="claude-sonnet-4-6",
    )


def _check_premise(case: Case, system: str, messages: list[dict]) -> str:
    """Empty string when the premise holds, else why it does not."""
    last = messages[-1]["content"]
    if not isinstance(last, str):
        last = " ".join(
            b.get("text", "") for b in last if b.get("type") == "text")
    prompt = system + "\n" + last
    for needle in case.premise.prompt_contains:
        if needle not in prompt:
            return f"prompt_contains 未命中：{needle!r}"
    for needle in case.premise.prompt_lacks:
        if needle in prompt:
            return f"prompt_lacks 被命中：{needle!r}"
    return ""


async def _default_sampler(system: str, messages: list[dict], n: int) -> list[str]:
    """Sample the real responder n times concurrently."""
    import openai

    preset = RESPONDER_PRESETS["deepseek"]
    client = openai.AsyncOpenAI(
        api_key=os.environ[preset["key_env"]], base_url=preset["base_url"])
    model = os.environ.get(preset["model_env"]) or preset["default_model"]

    payload = [{"role": "system", "content": system}]
    for m in messages:
        content = m["content"]
        if not isinstance(content, str):
            content = " ".join(
                b.get("text", "") for b in content if b.get("type") == "text")
        payload.append({"role": m["role"], "content": content})

    async def _one() -> str:
        resp = await client.chat.completions.create(
            model=model, messages=payload, max_tokens=300, temperature=0.9,
            extra_body=preset["extra_body"],
        )
        return (resp.choices[0].message.content or "").strip()

    return list(await asyncio.gather(*[_one() for _ in range(n)]))


async def score_case(
    case: Case, *, overrides: dict | None = None, sampler=None, llm=None,
) -> CaseScore:
    """Replay one case and score it. `sampler` is injectable for offline tests."""
    system, messages, persona = await build_turn(
        case, overrides=overrides, llm=llm)

    error = _check_premise(case, system, messages)
    if error:
        return CaseScore(id=case.id, verdict="BROKEN",
                         premise_ok=False, premise_error=error)

    replies = await (sampler or _default_sampler)(system, messages, case.samples)
    fails = sum(1 for r in replies if evaluate(case.detect.fail, r, persona))
    passes = (
        sum(1 for r in replies if evaluate(case.detect.passing, r, persona))
        if case.detect.passing else 0
    )
    n = len(replies) or 1
    fail_rate = fails / n
    return CaseScore(
        id=case.id,
        # pass_rate is observation only — it answers "did the fix also
        # produce right behaviour", not "did it pass".
        verdict="PASS" if fail_rate <= case.budget.max_fail_rate else "FAIL",
        fail_rate=fail_rate, pass_rate=passes / n, samples=len(replies),
        replies=replies,
    )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_evals/test_runner.py -v`
Expected: 6 passed

若 `ConversationEngine` 构造或 `switch_recipient` 的签名与此不符，以
`tests/test_conversation/test_tool_loop.py` 的 `_engine()` helper 为准对齐。

- [ ] **Step 6: 在 build_turn 中禁用天气**

在 `build_turn` 里构造 engine 之后、组装之前插入：

```python
        # Weather is a live external variable — same case, different prompt
        # depending on whether the bot happens to be running on this box.
        # Sunrise/sunset stays live: it is pure offline computation from the
        # frozen clock and the persona's location, and is worth testing.
        engine.prompt_builder._weather_line = lambda _now: None
```

对应测试追加进 `tests/test_evals/test_runner.py`:

```python
@pytest.mark.asyncio
async def test_weather_is_stubbed_out(tmp_path, monkeypatch):
    """Same case must assemble the same prompt whether or not the live bot
    has populated the weather cache on this machine."""
    from lingxi.brain import orchestrator as orch_mod
    from lingxi.brain.models import OrchestrationDecision
    from lingxi.temporal import weather as weather_mod

    async def _fake_decide(*a, **k):
        return OrchestrationDecision(
            register="warm", engage_level=0.6, fact_queries=[], skip=[],
            topic_anchor="")

    monkeypatch.setattr(orch_mod, "decide", _fake_decide)
    monkeypatch.setattr(
        weather_mod, "cached",
        lambda *a, **k: type("W", (), {"phrase": lambda self: "晴，30°C"})())
    _sys, messages, _ = await build_turn(_case(tmp_path), llm=_StubLLM())
    assert "30°C" not in messages[-1]["content"]
```

（`weather.cached(loc, *, now=None) -> Weather | None` 是读缓存的入口，已核对。）
- [ ] **Step 7: 全量回归并提交**

Run: `.venv/bin/python -m pytest -q`
Expected: 全绿，本任务 **新增 7 条**（6 + 天气 stub 用例）

```bash
git add src/lingxi/evals/runner.py tests/test_evals/test_runner.py
git commit -m "feat(evals): replay a case and score it

Re-assembles through the real pipeline instead of replaying stored
messages: three of the four fixes on 2026-08-19 lived in the assembly layer
and a frozen-messages harness would have tested none of them.

BROKEN short-circuits before sampling — a case whose premise no longer
holds must never report PASS, because a silently expired case reads as
coverage it no longer provides. pass_rate is computed but kept out of the
verdict; it answers whether a fix also produced right behaviour.

sampler is injectable so every branch here is tested offline."
```

---

## Task 5: CLI 与基线

**Files:**
- Create: `src/lingxi/evals/cli.py`
- Modify: `pyproject.toml`（`[project.scripts]` 增加一行）
- Test: `tests/test_evals/test_cli.py`

**Interfaces:**
- Consumes: Task 2 `load_all_cases`、Task 4 `score_case/CaseScore`
- Produces:
  - `format_table(scores: list[CaseScore], baseline: dict) -> str`
  - `load_baseline(path) -> dict` / `save_baseline(path, scores, git_sha) -> None`
  - `main() -> int`

- [ ] **Step 1: 写失败测试**

Create `tests/test_evals/test_cli.py`:

```python
import json

from lingxi.evals.cli import format_table, load_baseline, save_baseline
from lingxi.evals.runner import CaseScore


def test_table_shows_delta_against_baseline():
    scores = [CaseScore(id="a", verdict="PASS", fail_rate=0.05,
                        pass_rate=0.15, samples=20)]
    out = format_table(scores, {"a": {"fail_rate": 0.15}})
    assert "PASS" in out
    assert "1/20" in out
    assert "-10pp" in out


def test_table_marks_new_cases():
    scores = [CaseScore(id="a", verdict="PASS", fail_rate=0.0,
                        pass_rate=0.0, samples=20)]
    out = format_table(scores, {})
    assert "new" in out


def test_broken_row_shows_the_reason_and_no_numbers():
    scores = [CaseScore(id="a", verdict="BROKEN", premise_ok=False,
                        premise_error="prompt_contains 未命中：'X'")]
    out = format_table(scores, {"a": {"fail_rate": 0.5}})
    assert "BROKEN" in out
    assert "prompt_contains 未命中" in out
    assert "pp" not in out          # a broken case is not scored


def test_baseline_roundtrip(tmp_path):
    path = tmp_path / "baseline.json"
    save_baseline(path, [CaseScore(id="a", verdict="PASS", fail_rate=0.05,
                                   pass_rate=0.1, samples=20)], "abc123")
    data = load_baseline(path)
    assert data["a"]["fail_rate"] == 0.05
    assert data["a"]["git_sha"] == "abc123"
    assert "recorded_at" in data["a"]


def test_load_baseline_missing_file_is_empty(tmp_path):
    assert load_baseline(tmp_path / "nope.json") == {}


def test_broken_cases_are_not_written_to_baseline(tmp_path):
    """Recording a BROKEN case would bake a meaningless number into the
    baseline and hide the real number when the case is repaired."""
    path = tmp_path / "baseline.json"
    save_baseline(path, [
        CaseScore(id="ok", verdict="PASS", fail_rate=0.0, samples=20),
        CaseScore(id="bad", verdict="BROKEN", premise_ok=False),
    ], "abc123")
    assert sorted(json.loads(path.read_text())) == ["ok"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_evals/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lingxi.evals.cli'`

- [ ] **Step 3: 实现 cli.py**

```python
"""`lingxi-eval` — run the case library and compare against the baseline."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import datetime
from pathlib import Path

from lingxi.evals.case import load_all_cases, load_case
from lingxi.evals.runner import CaseScore, score_case

CASES_DIR = Path("evals/cases")
BASELINE_PATH = Path("evals/baseline.json")


def load_baseline(path: Path | str = BASELINE_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_baseline(path: Path | str, scores: list[CaseScore], git_sha: str) -> None:
    """Record scored cases. BROKEN cases are skipped on purpose: baking their
    meaningless number in would mask the real one once they are repaired."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    data = {
        s.id: {"fail_rate": s.fail_rate, "pass_rate": s.pass_rate,
               "samples": s.samples, "recorded_at": stamp, "git_sha": git_sha}
        for s in scores if s.verdict != "BROKEN"
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _fraction(rate: float, samples: int) -> str:
    return f"{round(rate * samples)}/{samples}"


def format_table(scores: list[CaseScore], baseline: dict) -> str:
    lines = [f"{'case':<22}{'verdict':<10}{'fail':<9}{'pass':<9}"
             f"{'baseline':<11}{'Δ'}"]
    for s in scores:
        if s.verdict == "BROKEN":
            lines.append(f"{s.id:<22}{'BROKEN':<10}前提不再成立：{s.premise_error}")
            continue
        base = baseline.get(s.id)
        if base is None:
            base_col, delta = "—", "new"
        else:
            base_col = f"{base['fail_rate']:.0%}"
            delta = f"{(s.fail_rate - base['fail_rate']) * 100:+.0f}pp"
        lines.append(
            f"{s.id:<22}{s.verdict:<10}"
            f"{_fraction(s.fail_rate, s.samples):<9}"
            f"{_fraction(s.pass_rate, s.samples):<9}{base_col:<11}{delta}")
    return "\n".join(lines)


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


async def _run(case_ids: list[str]) -> list[CaseScore]:
    cases = load_all_cases(CASES_DIR)
    if case_ids:
        cases = [c for c in cases if c.id in case_ids]
        if not cases:
            raise SystemExit(f"no case matched: {case_ids}")
    return [await score_case(c) for c in cases]


def main() -> int:
    parser = argparse.ArgumentParser(prog="lingxi-eval")
    parser.add_argument("cases", nargs="*", help="case id（留空跑全部）")
    parser.add_argument("--baseline", action="store_true",
                        help="把这次的分数记成新基线")
    args = parser.parse_args()

    scores = asyncio.run(_run(args.cases))
    print(format_table(scores, load_baseline()))

    if args.baseline:
        save_baseline(BASELINE_PATH, scores, _git_sha())
        print(f"\n基线已写入 {BASELINE_PATH}")
    return 1 if any(s.verdict != "PASS" for s in scores) else 0
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_evals/test_cli.py -v`
Expected: 6 passed

- [ ] **Step 5: 注册入口点**

`pyproject.toml` 的 `[project.scripts]` 增加一行：

```toml
lingxi-eval = "lingxi.evals.cli:main"
```

Run: `.venv/bin/pip install -e . -q && .venv/bin/lingxi-eval --help`
Expected: 打印 usage，含 `--baseline`

- [ ] **Step 5: 提交**

```bash
git add src/lingxi/evals/cli.py tests/test_evals/test_cli.py pyproject.toml
git commit -m "feat(evals): lingxi-eval CLI with baseline comparison

Exit code is non-zero when anything is not PASS, so this can gate a change
later without more plumbing. BROKEN rows print the failing assertion
instead of numbers, and are excluded from --baseline: recording a broken
case's number would bake in a meaningless value and mask the real one once
the case is repaired."
```

---

## Task 6: capture——把线上一轮冻成 case 骨架

Spec §9.1：捕获必须接近零摩擦，否则案例库会在第五个案例前后停止增长。仓库自己的证据是飞书标注通道 493 条 turn、0 条标注。

**Files:**
- Create: `src/lingxi/evals/capture.py`
- Modify: `src/lingxi/evals/cli.py`（增加 `capture` 子命令）
- Test: `tests/test_evals/test_capture.py`

**Interfaces:**
- Consumes: Task 2 的 `Case` 相关模型
- Produces: `async def capture(recipient_key, *, persona_path, data_dir, turns=8, at=None) -> dict` — 返回可直接 `yaml.safe_dump` 的 case 骨架

- [ ] **Step 1: 写失败测试**

Create `tests/test_evals/test_capture.py`:

```python
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

from lingxi.evals.capture import capture
from lingxi.evals.case import Case
from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.store import FactStore

AT = datetime(2026, 8, 19, 20, 20, 54)


async def _live(tmp_path: Path):
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    await store.write(Fact(
        subject="user:feishu:oc_x", content="对方一般晚上九点下班",
        source=Source.USER_STATED, type=FactType.PATTERN,
        ts=AT - timedelta(days=12), importance=4))
    await store.write(Fact(
        subject="aria", content="在练习室抠队形", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=AT - timedelta(hours=1), importance=4))
    return store


@pytest.mark.asyncio
async def test_capture_produces_a_loadable_skeleton(tmp_path):
    await _live(tmp_path)
    skeleton = await capture(
        "feishu:oc_x", persona_path="config/personas/tangkeke.yaml",
        data_dir=tmp_path, turns=8, at=AT)
    # Author fills these two in; everything else is mechanical.
    skeleton["symptom"] = "填写：错在哪"
    skeleton["detect"] = {"fail": {"any_of": ["堵车"]}}
    case = Case.model_validate(yaml.safe_load(yaml.safe_dump(skeleton)))
    assert case.clock == AT
    assert case.recipient == "feishu:oc_x"


@pytest.mark.asyncio
async def test_capture_converts_fact_ages_to_days_ago(tmp_path):
    await _live(tmp_path)
    skeleton = await capture(
        "feishu:oc_x", persona_path="config/personas/tangkeke.yaml",
        data_dir=tmp_path, turns=8, at=AT)
    user_fact = next(f for f in skeleton["facts"]
                     if "九点下班" in f["content"])
    assert round(user_fact["days_ago"]) == 12


@pytest.mark.asyncio
async def test_capture_includes_both_her_facts_and_his(tmp_path):
    await _live(tmp_path)
    skeleton = await capture(
        "feishu:oc_x", persona_path="config/personas/tangkeke.yaml",
        data_dir=tmp_path, turns=8, at=AT)
    subjects = {f["subject"] for f in skeleton["facts"]}
    assert "aria" in subjects
    assert "user:feishu:oc_x" in subjects


@pytest.mark.asyncio
async def test_capture_leaves_authoring_fields_empty(tmp_path):
    """symptom and detect are the human's job; a filled-in guess would be
    worse than a blank, because it looks done."""
    await _live(tmp_path)
    skeleton = await capture(
        "feishu:oc_x", persona_path="config/personas/tangkeke.yaml",
        data_dir=tmp_path, turns=8, at=AT)
    assert skeleton["symptom"] == ""
    assert skeleton["detect"] == {"fail": {"any_of": []}}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest tests/test_evals/test_capture.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lingxi.evals.capture'`

- [ ] **Step 3: 实现 capture.py**

```python
"""Freeze a live turn into a case skeleton.

Everything mechanical is done here — clock, fact ages, turn offsets — so
the only things left to write are what went wrong and what counts as wrong.
The Feishu annotation channel is the cautionary tale: built, wired, shipped,
and sitting at 493 recorded turns with 0 annotations, because it costs one
extra click. A case library that costs twenty minutes an entry dies the
same way.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from lingxi.facts.store import FactStore


async def capture(
    recipient_key: str,
    *,
    persona_path: str,
    data_dir: Path | str,
    turns: int = 8,
    at: datetime | None = None,
    fact_limit: int = 30,
) -> dict:
    """Snapshot live state into a dict ready for yaml.safe_dump."""
    data_dir = Path(data_dir)
    history = _read_history(data_dir, recipient_key, turns)
    clock = at or (history[-1][2] if history else datetime.now())

    store = FactStore(data_dir / "facts.db")
    facts: list[dict] = []
    for subject in ("aria", f"user:{recipient_key}"):
        for fact in await store.query(subject=subject, limit=fact_limit):
            age = clock - fact.ts
            facts.append({
                "subject": fact.subject,
                "type": fact.type.value,
                "source": fact.source.value,
                "content": fact.content,
                "importance": fact.importance,
                "days_ago": round(age.total_seconds() / 86400, 2),
            })

    history_rows = [
        {"role": role, "content": content,
         "minutes_ago": round((clock - ts).total_seconds() / 60, 1)}
        for role, content, ts in history[:-1]
    ]
    last_input = history[-1][1] if history else ""

    return {
        "id": f"{clock:%Y%m%d}-rename-me",
        "symptom": "",
        "origin": f"captured from {recipient_key} at {clock.isoformat()}",
        "persona": persona_path,
        "recipient": recipient_key,
        "clock": clock.isoformat(),
        "facts": facts,
        "history": history_rows,
        "input": last_input,
        "samples": 20,
        "premise": {"prompt_contains": [], "prompt_lacks": []},
        "detect": {"fail": {"any_of": []}},
        "budget": {"max_fail_rate": 0.05},
    }


def _read_history(
    data_dir: Path, recipient_key: str, turns: int,
) -> list[tuple[str, str, datetime]]:
    path = data_dir / "short_term" / f"{recipient_key}.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("turns", raw if isinstance(raw, list) else [])
    out: list[tuple[str, str, datetime]] = []
    for row in rows[-turns:]:
        ts = row.get("timestamp")
        out.append((
            row.get("role", "user"),
            row.get("content", ""),
            datetime.fromisoformat(ts) if ts else datetime.now(),
        ))
    return out
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest tests/test_evals/test_capture.py -v`
Expected: 4 passed

- [ ] **Step 5: 接进 CLI**

在 `src/lingxi/evals/cli.py` 的 `main()` 里，`parser.parse_args()` 之前增加：

```python
    parser.add_argument("--capture", metavar="RECIPIENT_KEY",
                        help="从线上状态冻出一个 case 骨架")
    parser.add_argument("--turns", type=int, default=8,
                        help="capture 时扒最近几轮对话")
```

在 `scores = asyncio.run(...)` 之前插入：

```python
    if args.capture:
        import os

        import yaml

        from lingxi.evals.capture import capture as capture_turn

        persona_path = os.environ.get(
            "PERSONA_PATH", "config/personas/example_persona.yaml")
        slug = Path(persona_path).stem
        skeleton = asyncio.run(capture_turn(
            args.capture, persona_path=persona_path,
            data_dir=Path("data/personas") / slug, turns=args.turns))
        out = CASES_DIR / f"{skeleton['id']}.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            yaml.safe_dump(skeleton, allow_unicode=True, sort_keys=False),
            encoding="utf-8")
        print(f"骨架已写入 {out}\n还需手写：symptom、detect、premise")
        return 0
```

- [ ] **Step 6: 真机跑一次**

Run: `PERSONA_PATH=config/personas/tangkeke.yaml .venv/bin/lingxi-eval --capture feishu:oc_c394e90ef07527af9e8d186645e87df1 --turns 8`
Expected: 打印骨架路径；打开该 YAML，`facts` 与 `history` 非空，`symptom` 与 `detect` 为空

- [ ] **Step 7: 全量回归并提交**

Run: `.venv/bin/python -m pytest -q`
Expected: 全绿，本任务 **新增 4 条**

```bash
git add src/lingxi/evals/capture.py src/lingxi/evals/cli.py \
        tests/test_evals/test_capture.py
git commit -m "feat(evals): capture a live turn into a case skeleton

Authoring friction is what kills these libraries. This repo already has the
proof: the Feishu annotation channel is built, wired and shipped, and sits
at 493 recorded turns with 0 annotations because it costs one extra click.
Hand-authoring cases costs twenty minutes each, which is the same failure
with the maintainer as the victim.

Clock, fact ages and turn offsets are all derived. symptom and detect are
left deliberately blank — a plausible auto-filled guess is worse than a
blank, because it looks finished."
```

---

## Task 7: 两个起步案例

**Files:**
- Create: `evals/cases/offwork-state.yaml`
- Create: `evals/cases/tewatashi-scale.yaml`
- Create: `evals/baseline.json`（由 `--baseline` 生成）
- Modify: `.gitignore`（确认 `evals/` 未被忽略）

**Interfaces:**
- Consumes: 前六个任务的全部产物
- Produces: 可运行的案例库与首份基线

- [ ] **Step 1: 用 capture 起草 offwork-state，再补 symptom / detect / premise**

以 capture 产物为底，写成 `evals/cases/offwork-state.yaml`：

```yaml
id: offwork-state
symptom: 他说想下班 90 秒后，她问是在堵车还是到家了
origin: 2026-08-19 飞书对话；修法 commit d3d3492
persona: config/personas/tangkeke.yaml
recipient: feishu:oc_eval
clock: "2026-08-19T20:20:54"

facts:
  - subject: "user:feishu:oc_eval"
    type: pattern
    source: user_stated
    content: 对方一般晚上九点下班
    importance: 4
    days_ago: 12
  - subject: aria
    type: event
    source: life_simulated
    content: 在练习室抠一段队形，抠了一下午
    importance: 4
    days_ago: 0.2

history:
  - {role: user, content: 想下班了, minutes_ago: 2}
  - {role: assistant, content: "想下班的心我懂！不过都这个点了，应该快了吧？", minutes_ago: 2}
  - {role: user, content: 就跟你想下课一样是吧, minutes_ago: 1}
  - {role: assistant, content: "哈哈哈对！上课到后半截心思早飞了", minutes_ago: 1}

input: 大学还不轻松啊，天天都有时间做自己想做的事去

samples: 20

premise:
  prompt_contains: ["2026-08-19 20:20"]

detect:
  fail: {any_of: [堵车, 到家, 回家了, 在路上, 下班了吧, 已经下班]}
  pass: {any_of: [快下班, 还没下班, 还在公司, 加班]}

budget: {max_fail_rate: 0.10}
```

`premise` 只断言时钟到位，不断言时段表原文——`user_state` 生效时该行不会出现，断言原文会让案例常态性 BROKEN。

- [ ] **Step 2: 跑它，与已知数据对照**

Run: `.venv/bin/lingxi-eval offwork-state`
Expected: `PASS`，`fail` 落在 `0/20`–`2/20`

**这一步是 harness 的自检**：2026-08-19 实测该场景修后为 1/60。若跑出的失败率显著高于此（比如 ≥5/20），先怀疑冻结或组装写错了，而不是 agent 退化——按顺序检查：`premise` 是否通过、`build_turn` 产出的 prompt 里 `**对方此刻：**` 是否存在、案例 facts 是否真的进了 prompt。

- [ ] **Step 3: 写 tewatashi-scale**

```yaml
id: tewatashi-scale
symptom: 把手渡会当成从容的见面会，打算当场读完信再说话
origin: 2026-08-19 飞书对话；修法 commit 3ea8713
persona: config/personas/tangkeke.yaml
recipient: feishu:oc_eval
clock: "2026-08-19T15:40:00"

facts:
  - subject: "user:feishu:oc_eval"
    type: pattern
    source: user_stated
    content: 对方22号去成都参加 FMT，两场都去，两场都手渡见面
    importance: 7
    days_ago: 0.01
  - subject: "user:feishu:oc_eval"
    type: pattern
    source: user_stated
    content: 对方手写了一封信，贴了吉伊贴纸，22号见面时给
    importance: 7
    days_ago: 0.01

history:
  - {role: user, content: "是吉伊！说起来，那天就是你的FMT啦，我两场都可以跟你手渡见面", minutes_ago: 3}
  - {role: assistant, content: "两场都能手渡！？哇你这也太上心了吧……", minutes_ago: 3}
  - {role: user, content: 能拿到你的签名我也很高兴哦, minutes_ago: 2}
  - {role: assistant, content: "呜哇…你这么说我要感动死了！！签名嘛，我肯定给你好好签！", minutes_ago: 2}

input: 我也在想该说些什么呢，可能到时候一紧张就什么都说不出来了也说不定

samples: 20

premise:
  prompt_contains: ["手渡"]

detect:
  fail: {any_of: [不赶时间, 慢慢说, 慢慢聊, 有的是时间]}
  pass: {any_of: [二十秒, 20秒, 十几秒, 时间很短, 就那么一会]}

budget: {max_fail_rate: 0.10}
```

- [ ] **Step 4: 跑全部并记基线**

Run: `.venv/bin/lingxi-eval`
Expected: 两行输出；`offwork-state` 为 PASS

若某个案例是 `BROKEN`，先修 `premise`（案例设计问题），不要动 agent。
若 `tewatashi-scale` 是 `FAIL`，那是真实发现——**先记基线，
再把修法作为独立提交**，这样基线能证明修法确实起作用。

Run: `.venv/bin/lingxi-eval --baseline`
Expected: 打印 `基线已写入 evals/baseline.json`

- [ ] **Step 5: 提交**

```bash
git add evals/
git commit -m "feat(evals): the first two cases and a baseline

Both are real failures from 2026-08-19, one per class: state tracking and
domain scale. The planned third case, invented-dates, went out with the
dates_outside_anchors detector — three rounds of narrowing never stopped it
firing on ordinary speech, and a detector that cannot meet 宁可漏报不可误报
is worse than none. Fabrication waits for the LLM-judge phase.

offwork-state doubles as the harness's self-check — that scenario measured
1/60 after the fix, so a replay landing far from it means the freezing or
assembly is wrong, not that the agent regressed. It is the case to trust
last and debug first.

Its premise asserts only the frozen clock, not the time-bucket wording: the
bucket line is absent whenever user_state is populated, so asserting it
would leave the case permanently BROKEN."
```

---

## Self-Review

**Spec coverage:**

| Spec 章节 | 落在哪个任务 |
|---|---|
| §4 架构、目录布局 | Task 2/3/4/5/6 的 Files 段 |
| §4.1 Case 格式 | Task 2 |
| §4.2 执行流程 | Task 4 `build_turn` + `score_case` |
| §4.3 输出 | Task 5 `format_table` |
| §4.4 CLI | Task 5 |
| §5 时钟注入（5 个注入点） | Task 1 |
| §5.5 天气 stub | **见下方缺口** |
| §6 前提断言与三档判定 | Task 4 `_check_premise`，Task 5 BROKEN 行 |
| §7 判定器 | Task 3（交付 any_of / regex 两个）|
| §8 起步案例 | Task 7（两个；第三个已取消，见 Task 3 AMENDMENT）|
| §9.1 capture | Task 6 |
| §10 overrides 接口 | Task 4 `_apply_overrides` / `score_case(overrides=)` |
| §13 测试策略 | Task 1/2/3 的测试 |

**缺口一处，已补进 Task 4 Step 6（下移到任务正文内，因为 `task-brief` 只抽取
`### Task N` 小节，留在本节会被实施者漏掉——这个漏发生过一次）：** §5.5 要求天气必须
stub。`_weather_line()` 读天气缓存，缓存为空时返回 `None`（不联网、不报错），但若同机
跑着 `lingxi-feishu`，缓存里会有真实天气，prompt 就带上了外部可变量。


**Placeholder scan:** 已通读，无 TBD / TODO / "similar to Task N" / "add error handling" 一类占位。每个改动步骤都给了完整代码。

**Type consistency:** 逐项核对——`CaseScore` 字段在 Task 4 定义、Task 5 消费，名称一致（`verdict/fail_rate/pass_rate/samples/premise_ok/premise_error/replies`）；`evaluate(spec, reply, persona)` 三参签名在 Task 3 定义、Task 4 调用一致；`Case.detect.passing`（YAML 里是 `pass`，Python 保留字，用 alias）在 Task 2 定义、Task 4 以 `case.detect.passing` 消费一致；`FactStore(path, clock=)` / `FactRetriever(store, clock=)` 在 Task 1 定义、Task 4 与 Task 6 使用一致；`build_turn` 返回三元组 `(system, messages, persona)`，Task 4 内部与其测试一致。
