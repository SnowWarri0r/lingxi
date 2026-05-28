# Generative Agents Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Generative Agents (Park et al. 2023) core mechanisms into Aria — LLM-scored importance, 3D retrieval, importance-driven tree-of-thought reflection, hour-granularity daily planner replacing the random simulator, and NPC parity with bidirectional interactions.

**Architecture:** Phase A clears the dual-write substrate left from facts-arch refactor. Phases B–E layer on subjective LLM importance scoring (batched, first-person AS persona), `recency × importance × relevance` retrieval, reflection triggered by importance accumulation OR 2h timeout, hour-block plans regenerated each morning + replanned reactively, and NPC↔Aria interactions written from both sides.

**Tech Stack:** Python 3.12, pydantic v2, SQLite + FTS5, anthropic SDK (Sonnet for generation/reflection/scoring), pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-05-28-generative-agents-integration-design.md`

**Branch:** `refactor/facts-arch` (continuation; worktree at `/Users/lovart/agent-facts-refactor`)

---

## Prompt Voice Hard Rule

Per spec section "Prompt Voice 约定" and memory `feedback_first_person_for_persona_generation`:

- **Generation prompts** (scorer, reflector, planner, moment executor, interaction): **first-person AS persona**. System message locks role (`"你是 Aria, 正在..."`). User-content prompts written as Aria's internal monologue, not as instructions to a third-party narrator.
- **Tool prompts** (orchestrator, plan_conflict detection): third-person, explicitly tool role (`"你是 Aria 的对话调度器"`).

Before committing any task with an LLM call: grep the prompt for `她|他|"为 Aria"|"Aria 当前"|"她正在"`. Any match in a generation prompt is a bug.

---

## File Map

### New files
- `src/lingxi/facts/scorer.py` — `ImportanceScorer` (batched LLM)
- `src/lingxi/facts/reflection_trigger.py` — `ReflectionTrigger` (accumulator + timer)
- `src/lingxi/facts/reflector.py` — `Reflector` (tree-of-thought)
- `src/lingxi/planner/__init__.py`
- `src/lingxi/planner/daily_planner.py` — `DailyPlanner` (Aria + NPC morning plans)
- `src/lingxi/planner/executor.py` — `PlanExecutor` (replaces random simulator)
- `src/lingxi/social/interaction.py` — `bidirectional_interaction()`
- `tests/facts/test_scorer.py`
- `tests/facts/test_retriever_3d.py`
- `tests/facts/test_reflection_trigger.py`
- `tests/facts/test_reflector.py`
- `tests/planner/__init__.py`
- `tests/planner/test_daily_planner.py`
- `tests/planner/test_executor.py`
- `tests/social/test_interaction.py`

### Modified files
- `src/lingxi/facts/models.py` — add `importance: int | None`, `last_accessed: datetime | None`, `FactType.PLAN`
- `src/lingxi/facts/store.py` — schema migration, `fts_rank()`, `update_last_accessed()`
- `src/lingxi/facts/writers/base.py` — scorer call + trigger observe + `write_skip_scorer()`
- `src/lingxi/facts/retriever.py` — rewrite `fetch()` to 3D
- `src/lingxi/brain/models.py` — `plan_conflict: bool` on `OrchestrationDecision`
- `src/lingxi/brain/orchestrator.py` — prompt + parse for `plan_conflict`
- `src/lingxi/conversation/engine.py` — call `executor.request_replan()` on conflict
- `src/lingxi/social/scheduler.py` — use `bidirectional_interaction()`
- `src/lingxi/app.py` — bootstrap scorer/trigger/reflector/planner/executor; drop simulator

### Deleted files
- `src/lingxi/relational/store.py`, `src/lingxi/relational/extractor.py`
- `src/lingxi/social/store.py`, `src/lingxi/social/promoter.py`
- `src/lingxi/world/store.py`
- `src/lingxi/inner_life/simulator.py` (replaced by `planner/executor.py`)
- `tools/migrate_to_facts.py`

### Trimmed files (sections removed)
- `src/lingxi/inner_life/store.py` — remove `recent_events` methods
- `src/lingxi/temporal/reflection.py` — remove old reflection loop (replaced by `facts/reflector.py`)
- `src/lingxi/world/scheduler.py` — remove dual-write to old world/store
- `src/lingxi/social/scheduler.py` — remove dual-write to social/store; keep schedule logic
- `src/lingxi/persona/prompt_builder.py` — remove `_build_relational_section`, `_build_world_section`, and their callers (handled by `brain/renderer.py` now)

---

## Phase A: P7 Cleanup (prerequisite)

Goal: eliminate dual-write paths so Phases B–E can build on a clean substrate. After this phase, only `facts/*` writes exist; no old store still receives writes.

### Task A.1: Drop recent_events methods from inner_life/store.py

**Files:**
- Modify: `src/lingxi/inner_life/store.py`

- [ ] **Step 1:** Read current `inner_life/store.py` to identify `recent_events` add/get/clear methods + the SQLite table itself.

- [ ] **Step 2:** Delete every method that touches `recent_events`. Keep methods for inner state that isn't yet migrated (mood, energy, agenda).

- [ ] **Step 3:** Remove the `CREATE TABLE recent_events ...` line from the `init()` method. Leave a one-line comment `# recent_events migrated to facts table (see facts/models.py FactType.EVENT)`.

- [ ] **Step 4:** Grep `recent_events` across the codebase: `grep -rn recent_events src/ tests/`. For every remaining hit, either (a) update to use `FactRetriever.fetch(FactQuery(subject="aria", type=FactType.EVENT, ...))`, or (b) delete the call if it's dead since the facts migration.

- [ ] **Step 5:** Run: `pytest tests/inner_life -v`. Expected: PASS (or modify tests to match).

- [ ] **Step 6:** Commit:
```bash
git add src/lingxi/inner_life/store.py tests/inner_life
git commit -m "P7: drop recent_events from inner_life store (now in facts)"
```

### Task A.2: Delete relational module

**Files:**
- Delete: `src/lingxi/relational/store.py`, `src/lingxi/relational/extractor.py`

- [ ] **Step 1:** Grep callers: `grep -rn "from lingxi.relational" src/ tests/ | grep -v __pycache__`. Note every importer.

- [ ] **Step 2:** For each importer, remove the import and any wiring that constructs a `RelationalStore` or `RelationalExtractor`. If a method uses the result, replace with `FactRetriever.fetch(FactQuery(subject=f"user:{recipient_key}", type=FactType.OPINION, limit=N))`.

- [ ] **Step 3:** Delete the two files:
```bash
rm src/lingxi/relational/store.py src/lingxi/relational/extractor.py
```
Keep `src/lingxi/relational/models.py` only if anything still imports its types; otherwise delete the whole package.

- [ ] **Step 4:** If `relational/` directory becomes empty after deletes, remove it entirely (including `__init__.py`).

- [ ] **Step 5:** Run: `pytest tests/ -v -x`. Expected: PASS (fix any breakage from removed imports).

- [ ] **Step 6:** Commit:
```bash
git add -A
git commit -m "P7: delete relational module (data in facts user:* subjects)"
```

### Task A.3: Delete social/store and social/promoter

**Files:**
- Delete: `src/lingxi/social/store.py`, `src/lingxi/social/promoter.py`

- [ ] **Step 1:** Grep callers: `grep -rn "from lingxi.social.store\|from lingxi.social.promoter" src/ tests/`.

- [ ] **Step 2:** In `src/lingxi/social/scheduler.py`, remove imports + calls to `store.append_event(...)` and `promoter.promote(...)`. The new flow is: scheduler triggers `event_generator`, generator produces text, `NPCWriter.write(...)` persists to facts.

- [ ] **Step 3:** Delete:
```bash
rm src/lingxi/social/store.py src/lingxi/social/promoter.py
```

- [ ] **Step 4:** Run: `pytest tests/ -v -x`. Expected: PASS.

- [ ] **Step 5:** Commit:
```bash
git add -A
git commit -m "P7: delete social/store + social/promoter (npc events in facts)"
```

### Task A.4: Delete world/store

**Files:**
- Delete: `src/lingxi/world/store.py`

- [ ] **Step 1:** Grep callers: `grep -rn "from lingxi.world.store" src/ tests/`.

- [ ] **Step 2:** In `src/lingxi/world/scheduler.py`, remove import + `store.save(briefing)`. New flow: fetcher returns briefing, `WorldWriter.write(...)` persists to facts with `expires_at=now+2d`.

- [ ] **Step 3:** Anywhere the old store's `latest()` was read (likely `persona/prompt_builder.py`), replace with `FactRetriever.fetch(FactQuery(subject="world", type=FactType.EVENT, limit=3, since=now-timedelta(days=2)))`.

- [ ] **Step 4:** Delete:
```bash
rm src/lingxi/world/store.py
```

- [ ] **Step 5:** Run: `pytest tests/ -v -x`. Expected: PASS.

- [ ] **Step 6:** Commit:
```bash
git add -A
git commit -m "P7: delete world/store (world facts now in facts table)"
```

### Task A.5: Remove dual-write paths from simulator (and prep for replacement)

**Files:**
- Modify: `src/lingxi/inner_life/simulator.py`

Note: simulator itself is deleted in Task D.8. This task just removes dual-write so the existing simulator goes single-write to facts.

- [ ] **Step 1:** Open `simulator.py`. Find any `recent_events.insert(...)` or `inner_life_store.append_event(...)` calls. Confirm `life_writer.write(...)` (the new path) is also called there.

- [ ] **Step 2:** Delete the old call. Keep only `life_writer.write(...)`.

- [ ] **Step 3:** Run: `pytest tests/ -v -x`. Expected: PASS.

- [ ] **Step 4:** Commit:
```bash
git add src/lingxi/inner_life/simulator.py
git commit -m "P7: simulator single-writes to facts (drop dual-write)"
```

### Task A.6: Remove dual-write paths from social/scheduler

**Files:**
- Modify: `src/lingxi/social/scheduler.py`

- [ ] **Step 1:** Find any remaining `store.append_event(...)` or `promoter.promote(...)` left after Task A.3.

- [ ] **Step 2:** Confirm `npc_writer.write(...)` is the only persistence call. Delete the rest.

- [ ] **Step 3:** Run: `pytest tests/social -v`. Expected: PASS.

- [ ] **Step 4:** Commit:
```bash
git add src/lingxi/social/scheduler.py
git commit -m "P7: social scheduler single-writes via NPCWriter"
```

### Task A.7: Replace old temporal/reflection loop

**Files:**
- Modify: `src/lingxi/temporal/reflection.py`

The old reflection loop calls `merge_deltas` against old in-memory structures and dual-writes to facts via `inference_writer`. Phase C replaces it entirely with `facts/reflector.py`. This task removes the old logic but keeps the file as a thin shim so app.py doesn't break before C lands.

- [ ] **Step 1:** Rewrite `temporal/reflection.py` to contain only a stub:
```python
"""Old reflection loop removed in P7 cleanup.
Replaced by `lingxi/facts/reflector.py` in Phase C.
This module is kept as a no-op shim until app.py is rewired.
"""

async def run_reflection_loop(*args, **kwargs) -> None:
    """No-op shim. Real reflection lives in facts.reflector now."""
    return None
```

- [ ] **Step 2:** Run: `pytest tests/temporal -v`. If old tests fail, delete or skip them (they test deleted behavior). New reflector tests come in Phase C.

- [ ] **Step 3:** Commit:
```bash
git add src/lingxi/temporal/reflection.py tests/temporal
git commit -m "P7: stub out old temporal reflection loop (real one in Phase C)"
```

### Task A.8: Remove dual-write from world/scheduler

**Files:**
- Modify: `src/lingxi/world/scheduler.py`

- [ ] **Step 1:** Find any `store.save(...)` calls left from old WorldStore.

- [ ] **Step 2:** Confirm `world_writer.write(...)` is the only persistence path. Delete the rest.

- [ ] **Step 3:** Run: `pytest tests/world -v`. Expected: PASS.

- [ ] **Step 4:** Commit:
```bash
git add src/lingxi/world/scheduler.py
git commit -m "P7: world scheduler single-writes via WorldWriter"
```

### Task A.9: Trim PromptBuilder dynamic section methods

**Files:**
- Modify: `src/lingxi/persona/prompt_builder.py`

The new `brain/renderer.py` owns dynamic blocks (`【你此刻】/【你和他】/【身边的事】`). PromptBuilder should only own persona-static content (identity, speaking_style, message_habits). Anything else is residual from pre-refactor.

- [ ] **Step 1:** In `prompt_builder.py`, locate and delete:
  - `_build_relational_section`
  - `_build_world_section`
  - Any other `_build_*_section` method that pulls from the deleted stores
  - The lines in the main build path that append these sections (lines around 161 and 218 per earlier scan, but verify current line numbers)

- [ ] **Step 2:** Run: `grep -n "_build_" src/lingxi/persona/prompt_builder.py`. Only persona-static builders should remain.

- [ ] **Step 3:** Run: `pytest tests/persona -v` (or whatever covers prompt_builder). Expected: PASS.

- [ ] **Step 4:** Commit:
```bash
git add src/lingxi/persona/prompt_builder.py tests/persona
git commit -m "P7: trim PromptBuilder dynamic sections (renderer owns them now)"
```

### Task A.10: Delete migration script

**Files:**
- Delete: `tools/migrate_to_facts.py`

- [ ] **Step 1:** Delete:
```bash
rm tools/migrate_to_facts.py
```

- [ ] **Step 2:** Commit:
```bash
git add -A
git commit -m "P7: drop one-shot migration script (done)"
```

### Task A.11: Full test suite + smoke run

**Files:** (none modified)

- [ ] **Step 1:** Run full suite: `pytest tests/ -v`. Expected: all PASS.

- [ ] **Step 2:** Smoke import: `python -c "from lingxi.app import build_app; print('ok')"`. Expected: prints `ok`.

- [ ] **Step 3:** If anything fails, fix before proceeding to Phase B. Do NOT start Phase B with a red suite.

- [ ] **Step 4:** Commit (if any fixes needed):
```bash
git add -A
git commit -m "P7: post-cleanup test fixes"
```

---

## Phase B: Importance Scoring + 3D Retrieval

### Task B.1: Add `importance` and `last_accessed` to Fact model

**Files:**
- Modify: `src/lingxi/facts/models.py`
- Test: `tests/facts/test_models.py`

- [ ] **Step 1: Write failing test**

In `tests/facts/test_models.py`, add:
```python
def test_fact_has_importance_field_optional():
    from lingxi.facts.models import Fact, Source, FactType
    from datetime import datetime
    f = Fact(
        subject="aria",
        content="test",
        source=Source.LIFE_SIMULATED,
        type=FactType.EVENT,
        ts=datetime.now(),
    )
    assert f.importance is None
    assert f.last_accessed is None

def test_fact_accepts_importance_and_last_accessed():
    from lingxi.facts.models import Fact, Source, FactType
    from datetime import datetime
    now = datetime.now()
    f = Fact(
        subject="aria",
        content="test",
        source=Source.LIFE_SIMULATED,
        type=FactType.EVENT,
        ts=now,
        importance=7,
        last_accessed=now,
    )
    assert f.importance == 7
    assert f.last_accessed == now
```

- [ ] **Step 2:** Run: `pytest tests/facts/test_models.py::test_fact_has_importance_field_optional -v`. Expected: FAIL (`Fact has no attribute 'importance'`).

- [ ] **Step 3: Implement**

In `src/lingxi/facts/models.py`, in the `Fact` class:
```python
class Fact(BaseModel):
    # ... existing fields ...
    importance: int | None = None
    last_accessed: datetime | None = None
```

- [ ] **Step 4:** Run: `pytest tests/facts/test_models.py -v`. Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/lingxi/facts/models.py tests/facts/test_models.py
git commit -m "facts: add importance + last_accessed to Fact model (nullable)"
```

### Task B.2: Migrate FactStore schema

**Files:**
- Modify: `src/lingxi/facts/store.py`
- Test: `tests/facts/test_store.py`

- [ ] **Step 1: Write failing test**

In `tests/facts/test_store.py`, add:
```python
async def test_store_persists_importance_and_last_accessed(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.models import Fact, Source, FactType
    from datetime import datetime

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    now = datetime.now()
    f = Fact(
        subject="aria", content="x", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=now, importance=8, last_accessed=now,
    )
    await store.append(f)
    rows = await store.query(subject="aria", limit=1)
    assert rows[0].importance == 8
    assert rows[0].last_accessed is not None
```

- [ ] **Step 2:** Run test. Expected: FAIL (`no such column: importance`).

- [ ] **Step 3: Implement schema migration**

In `FactStore.init()`, after the `CREATE TABLE facts (...)` block, add migration:
```python
# Schema migration: add importance + last_accessed if missing
cols = await self._conn.execute_fetchall("PRAGMA table_info(facts)")
col_names = {row[1] for row in cols}
if "importance" not in col_names:
    await self._conn.execute("ALTER TABLE facts ADD COLUMN importance INTEGER")
if "last_accessed" not in col_names:
    await self._conn.execute("ALTER TABLE facts ADD COLUMN last_accessed TIMESTAMP")
await self._conn.execute(
    "CREATE INDEX IF NOT EXISTS idx_facts_importance ON facts(importance) "
    "WHERE importance IS NOT NULL"
)
await self._conn.commit()
```

Also update the `CREATE TABLE` definition for fresh installs to include both columns from the start.

Update `_row_to_fact()` to map the two new columns into the `Fact` constructor.

Update `append()`'s INSERT statement to include `importance, last_accessed` in the column list and value tuple.

- [ ] **Step 4:** Run: `pytest tests/facts/test_store.py -v`. Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/lingxi/facts/store.py tests/facts/test_store.py
git commit -m "facts/store: add importance + last_accessed columns + migration"
```

### Task B.3: Add `fts_rank()` and `update_last_accessed()` to FactStore

**Files:**
- Modify: `src/lingxi/facts/store.py`
- Test: `tests/facts/test_store.py`

- [ ] **Step 1: Write failing tests**

```python
async def test_fts_rank_returns_normalized_scores(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.models import Fact, Source, FactType
    from datetime import datetime
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    f1 = Fact(subject="aria", content="光变曲线 数据分析", source=Source.LIFE_SIMULATED,
              type=FactType.EVENT, ts=datetime.now())
    f2 = Fact(subject="aria", content="喝咖啡", source=Source.LIFE_SIMULATED,
              type=FactType.EVENT, ts=datetime.now())
    await store.append(f1)
    await store.append(f2)
    ranks = await store.fts_rank("光变曲线", [f1.id, f2.id])
    assert ranks[f1.id] > ranks[f2.id]
    assert all(0.0 <= v <= 1.0 for v in ranks.values())

async def test_update_last_accessed_writes_timestamp(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.models import Fact, Source, FactType
    from datetime import datetime, timedelta
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    f = Fact(subject="aria", content="x", source=Source.LIFE_SIMULATED,
             type=FactType.EVENT, ts=datetime.now())
    await store.append(f)
    new_ts = datetime.now() + timedelta(hours=1)
    await store.update_last_accessed([f.id], new_ts)
    rows = await store.query(subject="aria", limit=1)
    assert abs((rows[0].last_accessed - new_ts).total_seconds()) < 1
```

- [ ] **Step 2:** Run tests. Expected: FAIL (methods don't exist).

- [ ] **Step 3: Implement**

In `FactStore`:
```python
async def fts_rank(self, query: str, ids: list[str]) -> dict[str, float]:
    """Return {id: normalized_rank in [0,1]} for ids matching FTS query.
    Missing ids get 0.0. Higher = more relevant.
    """
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    sql = (
        "SELECT facts.id, bm25(facts_fts) AS rank "
        "FROM facts_fts JOIN facts ON facts_fts.rowid = facts.rowid "
        f"WHERE facts_fts MATCH ? AND facts.id IN ({placeholders})"
    )
    rows = await self._conn.execute_fetchall(sql, (query, *ids))
    if not rows:
        return {fid: 0.0 for fid in ids}
    # bm25 returns negative scores; smaller (more negative) = better match.
    # Normalize: invert sign, then min-max to [0,1].
    raw = {row[0]: -row[1] for row in rows}
    if len(raw) == 1:
        # Single hit: full credit
        result = {fid: 0.0 for fid in ids}
        result.update({k: 1.0 for k in raw})
        return result
    lo, hi = min(raw.values()), max(raw.values())
    span = hi - lo if hi > lo else 1.0
    result = {fid: 0.0 for fid in ids}
    for k, v in raw.items():
        result[k] = (v - lo) / span
    return result

async def update_last_accessed(self, ids: list[str], ts: datetime) -> None:
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    sql = f"UPDATE facts SET last_accessed = ? WHERE id IN ({placeholders})"
    await self._conn.execute(sql, (ts.isoformat(), *ids))
    await self._conn.commit()
```

- [ ] **Step 4:** Run: `pytest tests/facts/test_store.py -v`. Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/lingxi/facts/store.py tests/facts/test_store.py
git commit -m "facts/store: add fts_rank() and update_last_accessed()"
```

### Task B.4: Create ImportanceScorer (batched)

**Files:**
- Create: `src/lingxi/facts/scorer.py`
- Create: `tests/facts/test_scorer.py`

- [ ] **Step 1: Write failing tests**

`tests/facts/test_scorer.py`:
```python
import asyncio
import pytest
from datetime import datetime
from lingxi.facts.models import Fact, Source, FactType


class FakeLLM:
    """Records calls; returns a canned JSON array."""
    def __init__(self, canned: list[dict]):
        self.canned = canned
        self.calls: list[dict] = []
        self.system_calls: list[str] = []

    async def complete(self, *, messages, system=None, **kwargs):
        self.system_calls.append(system or "")
        self.calls.append({"messages": messages, "system": system})
        import json
        from types import SimpleNamespace
        return SimpleNamespace(content=json.dumps(self.canned))


@pytest.mark.asyncio
async def test_scorer_batches_five_facts_into_one_call():
    from lingxi.facts.scorer import ImportanceScorer
    facts = [Fact(id=str(i), subject="aria", content=f"event {i}",
                  source=Source.LIFE_SIMULATED, type=FactType.EVENT,
                  ts=datetime.now()) for i in range(5)]
    llm = FakeLLM([{"id": f.id, "score": 5, "reason": "ok"} for f in facts])
    scorer = ImportanceScorer(llm, batch_size=5, flush_seconds=10)
    scores = await asyncio.gather(*[scorer.score_one(f) for f in facts])
    assert scores == [5, 5, 5, 5, 5]
    assert len(llm.calls) == 1  # one batched call


@pytest.mark.asyncio
async def test_scorer_falls_back_to_default_on_llm_failure():
    from lingxi.facts.scorer import ImportanceScorer
    class BrokenLLM:
        async def complete(self, **kw):
            raise RuntimeError("api down")
    f = Fact(subject="aria", content="x", source=Source.USER_STATED,
             type=FactType.EVENT, ts=datetime.now())
    scorer = ImportanceScorer(BrokenLLM(), batch_size=1, flush_seconds=0.1)
    score = await scorer.score_one(f)
    assert score == 7  # USER_STATED default


@pytest.mark.asyncio
async def test_scorer_uses_first_person_system_for_aria_subject():
    from lingxi.facts.scorer import ImportanceScorer
    f = Fact(subject="aria", content="x", source=Source.LIFE_SIMULATED,
             type=FactType.EVENT, ts=datetime.now())
    llm = FakeLLM([{"id": f.id, "score": 5, "reason": "ok"}])
    scorer = ImportanceScorer(llm, batch_size=1, flush_seconds=0.1)
    await scorer.score_one(f)
    assert "你是 Aria" in llm.system_calls[0]
    assert "她" not in llm.calls[0]["messages"][0]["content"]
```

- [ ] **Step 2:** Run: `pytest tests/facts/test_scorer.py -v`. Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

`src/lingxi/facts/scorer.py`:
```python
"""Batched, first-person importance scoring.

Aria (or the NPC, depending on subject) scores her own recent facts
by subjective "this matters to me" — the Generative Agents paper's
"poignancy" rating. NOT an external rater.

Batching: writers call `score_one(fact)` which returns a Future. The
scorer flushes the buffer when it hits batch_size or flush_seconds,
makes one LLM call scoring all queued facts.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

from lingxi.facts.models import Fact, Source
from lingxi.providers.base import LLMProvider


DEFAULT_IMPORTANCE: dict[Source, int] = {
    Source.USER_STATED: 7,
    Source.BIOGRAPHY: 8,
    Source.LIFE_SIMULATED: 3,
    Source.NPC_TICKER: 4,
    Source.LLM_INFERRED: 5,
    Source.WORLD_FETCH: 3,
}


_ARIA_SYSTEM = "你是 Aria，正在回看自己最近经历的事。"
_NPC_SYSTEM_TEMPLATE = "你是 {name}，正在回看自己最近经历的事。"


_PROMPT_TEMPLATE = """我在给自己最近经历的事打分——这些事对**我**来说有多重要（1-10）。
1 = 完全琐碎（"喝了口水"），10 = 改变我和别人关系、或者改变我人生方向的事。
主观判断，不是客观新闻价值：
  - 我每天重复的作息 → 1-3
  - 普通工作进展 → 3-5
  - 跟我在意的人有情感交流 / 我自己情绪起伏 → 6-8
  - 关键关系变化 / 重大决定 / 真正触动到我的事 → 8-10

输入 {n} 条事件，输出 JSON array：
[{{"id": "...", "score": 1-10, "reason": "一句话——为什么对我来说是这个分"}}, ...]

事件：
{facts_block}
"""


@dataclass
class _PendingFact:
    fact: Fact
    future: asyncio.Future


def _resolve_system(subject: str) -> str:
    if subject == "aria":
        return _ARIA_SYSTEM
    if subject.startswith("npc:"):
        name = subject.removeprefix("npc:")
        return _NPC_SYSTEM_TEMPLATE.format(name=name)
    # user: / world / anything else — keep first-person but neutral
    return "你是叙述者，给最近一批事件评打分。"


def _bucket_key(subject: str) -> str:
    """Group facts by persona (subject) so one LLM call doesn't mix voices."""
    if subject == "aria":
        return "aria"
    if subject.startswith("npc:"):
        return subject  # one NPC per bucket
    return "other"


class ImportanceScorer:
    def __init__(
        self,
        llm: LLMProvider,
        batch_size: int = 5,
        flush_seconds: float = 30.0,
        model: str | None = None,
    ):
        self._llm = llm
        self._batch_size = batch_size
        self._flush_seconds = flush_seconds
        self._model = model
        self._buckets: dict[str, list[_PendingFact]] = {}
        self._flush_tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def score_one(self, fact: Fact) -> int:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        bucket = _bucket_key(fact.subject)
        async with self._lock:
            self._buckets.setdefault(bucket, []).append(_PendingFact(fact, future))
            if len(self._buckets[bucket]) >= self._batch_size:
                self._cancel_flush_timer(bucket)
                await self._flush(bucket)
            else:
                self._schedule_flush_timer(bucket)
        try:
            return await future
        except Exception:
            return DEFAULT_IMPORTANCE.get(fact.source, 5)

    def _schedule_flush_timer(self, bucket: str) -> None:
        if bucket in self._flush_tasks and not self._flush_tasks[bucket].done():
            return
        self._flush_tasks[bucket] = asyncio.create_task(self._flush_after_delay(bucket))

    def _cancel_flush_timer(self, bucket: str) -> None:
        task = self._flush_tasks.pop(bucket, None)
        if task and not task.done():
            task.cancel()

    async def _flush_after_delay(self, bucket: str) -> None:
        try:
            await asyncio.sleep(self._flush_seconds)
        except asyncio.CancelledError:
            return
        async with self._lock:
            if self._buckets.get(bucket):
                await self._flush(bucket)

    async def _flush(self, bucket: str) -> None:
        """Caller must hold self._lock."""
        pending = self._buckets.pop(bucket, [])
        if not pending:
            return
        system = _resolve_system(pending[0].fact.subject)
        facts_block = "\n".join(
            f"[{i+1}] id={p.fact.id} type={p.fact.type.value} content=\"{p.fact.content}\""
            for i, p in enumerate(pending)
        )
        prompt = _PROMPT_TEMPLATE.format(n=len(pending), facts_block=facts_block)
        try:
            kwargs = {"model": self._model} if self._model else {}
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=system,
                max_tokens=400,
                temperature=0.3,
                _debug_purpose="importance_scorer",
                **kwargs,
            )
            data = json.loads(_strip_json_fences(response.content))
            scores_by_id = {item["id"]: int(item["score"]) for item in data
                            if isinstance(item, dict) and "id" in item and "score" in item}
            for p in pending:
                score = scores_by_id.get(p.fact.id)
                if score is None or not (1 <= score <= 10):
                    score = DEFAULT_IMPORTANCE.get(p.fact.source, 5)
                if not p.future.done():
                    p.future.set_result(score)
        except Exception as e:
            print(f"[scorer] LLM batch failed, using defaults: {e}", flush=True)
            for p in pending:
                if not p.future.done():
                    p.future.set_result(DEFAULT_IMPORTANCE.get(p.fact.source, 5))


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()
```

- [ ] **Step 4:** Run: `pytest tests/facts/test_scorer.py -v`. Expected: PASS.

- [ ] **Step 5: Voice grep**
```bash
grep -nE "她|为 Aria|Aria 当前|她正在" src/lingxi/facts/scorer.py
```
Expected: no matches.

- [ ] **Step 6: Commit**
```bash
git add src/lingxi/facts/scorer.py tests/facts/test_scorer.py
git commit -m "facts: ImportanceScorer (batched, first-person AS persona)"
```

### Task B.5: Wire scorer into WriterBase

**Files:**
- Modify: `src/lingxi/facts/writers/base.py`
- Test: `tests/facts/writers/test_base.py`

- [ ] **Step 1: Write failing test**

```python
@pytest.mark.asyncio
async def test_writer_calls_scorer_when_importance_is_none(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.writers.life import LifeWriter
    from lingxi.facts.models import Fact, Source, FactType
    from datetime import datetime

    class StubScorer:
        async def score_one(self, fact):
            return 9

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    writer = LifeWriter(store, scorer=StubScorer())
    f = Fact(subject="aria", content="x", source=Source.LIFE_SIMULATED,
             type=FactType.EVENT, ts=datetime.now())
    await writer.write(f)
    rows = await store.query(subject="aria", limit=1)
    assert rows[0].importance == 9

@pytest.mark.asyncio
async def test_writer_skips_scorer_when_importance_preset(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.writers.life import LifeWriter
    from lingxi.facts.models import Fact, Source, FactType
    from datetime import datetime

    class FailingScorer:
        async def score_one(self, fact):
            raise AssertionError("scorer must not be called")

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    writer = LifeWriter(store, scorer=FailingScorer())
    f = Fact(subject="aria", content="x", source=Source.LIFE_SIMULATED,
             type=FactType.EVENT, ts=datetime.now(), importance=4)
    await writer.write(f)
    rows = await store.query(subject="aria", limit=1)
    assert rows[0].importance == 4
```

- [ ] **Step 2:** Run. Expected: FAIL (writer signature doesn't take scorer).

- [ ] **Step 3: Implement**

In `src/lingxi/facts/writers/base.py`:
```python
class WriterBase:
    SUBJECT_PATTERN: re.Pattern  # set by subclass
    ALLOWED_SOURCES: set[Source]  # set by subclass

    def __init__(self, store, scorer=None, reflection_trigger=None):
        self._store = store
        self._scorer = scorer
        self._trigger = reflection_trigger

    async def write(self, fact: Fact) -> None:
        self._enforce(fact)
        if fact.importance is None and self._scorer is not None:
            fact.importance = await self._scorer.score_one(fact)
        await self._store.append(fact)
        if self._trigger is not None and fact.importance is not None:
            await self._trigger.observe(fact.importance)

    async def write_skip_scorer(self, fact: Fact) -> None:
        """For callers (like the planner) that have pre-assigned importance
        and explicitly want to bypass the LLM scoring path."""
        self._enforce(fact)
        if fact.importance is None:
            fact.importance = 5  # neutral fallback if caller forgot
        await self._store.append(fact)
        if self._trigger is not None:
            await self._trigger.observe(fact.importance)

    def _enforce(self, fact: Fact) -> None:
        if not self.SUBJECT_PATTERN.match(fact.subject):
            raise ValueError(f"{type(self).__name__} cannot write subject={fact.subject!r}")
        if fact.source not in self.ALLOWED_SOURCES:
            raise ValueError(f"{type(self).__name__} cannot write source={fact.source}")
```

Each subclass writer (`LifeWriter`, `NPCWriter`, etc.) needs its `__init__` updated to accept `scorer=None, reflection_trigger=None` and pass to super. Audit and update each: `writers/life.py`, `writers/npc.py`, `writers/user.py`, `writers/inference.py`, `writers/world.py`, `writers/biography.py`.

- [ ] **Step 4:** Run: `pytest tests/facts/writers -v`. Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/lingxi/facts/writers tests/facts/writers
git commit -m "facts/writers: scorer + trigger integration + write_skip_scorer"
```

### Task B.6: Rewrite Retriever.fetch() to 3D scoring

**Files:**
- Modify: `src/lingxi/facts/retriever.py`
- Create: `tests/facts/test_retriever_3d.py`

- [ ] **Step 1: Write failing tests**

```python
import math
import pytest
from datetime import datetime, timedelta
from lingxi.facts.store import FactStore
from lingxi.facts.retriever import FactRetriever, FactQuery
from lingxi.facts.models import Fact, Source, FactType


@pytest.mark.asyncio
async def test_3d_ranking_prefers_high_importance_over_recent_trivia(tmp_path):
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    now = datetime.now()
    # Old high-importance fact (1 week ago, importance=9)
    old_important = Fact(
        subject="aria", content="跟用户讨论失眠", source=Source.USER_STATED,
        type=FactType.EVENT, ts=now - timedelta(days=7), importance=9,
    )
    # Recent trivial fact (1 hour ago, importance=2)
    recent_trivial = Fact(
        subject="aria", content="喝了杯水", source=Source.LIFE_SIMULATED,
        type=FactType.EVENT, ts=now - timedelta(hours=1), importance=2,
    )
    await store.append(old_important)
    await store.append(recent_trivial)
    r = FactRetriever(store)
    results = await r.fetch(FactQuery(subject="aria", limit=1))
    assert results[0].id == old_important.id


@pytest.mark.asyncio
async def test_fetch_updates_last_accessed(tmp_path):
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    now = datetime.now()
    f = Fact(subject="aria", content="x", source=Source.LIFE_SIMULATED,
             type=FactType.EVENT, ts=now, importance=5)
    await store.append(f)
    r = FactRetriever(store)
    await r.fetch(FactQuery(subject="aria", limit=5))
    rows = await store.query(subject="aria", limit=1)
    assert rows[0].last_accessed is not None
```

- [ ] **Step 2:** Run. Expected: FAIL (current retriever uses recency-only).

- [ ] **Step 3: Implement**

Replace `FactRetriever.fetch()`:
```python
import math
from datetime import datetime


async def fetch(self, query: FactQuery) -> list[Fact]:
    candidates = await self._store.query(
        subject=query.subject,
        type=query.type,
        since=query.since,
        limit=query.limit * 8,
    )
    if not candidates:
        return []

    if query.semantic:
        fts_ranks = await self._store.fts_rank(query.semantic, [c.id for c in candidates])
    else:
        fts_ranks = {c.id: 0.0 for c in candidates}

    now = datetime.now()
    scored: list[tuple[float, Fact]] = []
    for fact in candidates:
        hours_old = max(0.0, (now - fact.ts).total_seconds() / 3600)
        recency = math.exp(-0.01 * hours_old)
        importance = (fact.importance if fact.importance is not None else 5) / 10.0
        relevance = fts_ranks.get(fact.id, 0.0)
        score = 0.5 * recency + 0.3 * importance + 0.2 * relevance
        scored.append((score, fact))

    scored.sort(key=lambda x: -x[0])
    top = [f for _, f in scored[: query.limit]]
    if top:
        await self._store.update_last_accessed([f.id for f in top], now)
    return top
```

- [ ] **Step 4:** Run: `pytest tests/facts/test_retriever_3d.py -v`. Expected: PASS.

- [ ] **Step 5:** Run full facts suite: `pytest tests/facts -v`. Expected: PASS.

- [ ] **Step 6: Commit**
```bash
git add src/lingxi/facts/retriever.py tests/facts/test_retriever_3d.py
git commit -m "facts/retriever: 3D scoring (recency·importance·relevance)"
```

### Task B.7: Bootstrap scorer in app.py

**Files:**
- Modify: `src/lingxi/app.py`

- [ ] **Step 1:** Read `src/lingxi/app.py`. Find the section where `fact_retriever` and writers are constructed.

- [ ] **Step 2:** Insert scorer construction before writers:
```python
from lingxi.facts.scorer import ImportanceScorer
importance_scorer = ImportanceScorer(llm_provider)
```

- [ ] **Step 3:** Update each writer construction to pass scorer:
```python
life_writer = LifeWriter(facts_store, scorer=importance_scorer)
npc_writer = NPCWriter(facts_store, scorer=importance_scorer)
user_writer = UserStatementWriter(facts_store, scorer=importance_scorer)
inference_writer = InferenceWriter(facts_store, scorer=importance_scorer)
world_writer = WorldWriter(facts_store, scorer=importance_scorer)
```

Biography loader uses preset importance=8, doesn't need scorer (pass `scorer=None` or omit).

- [ ] **Step 4: Smoke import**
```bash
python -c "import asyncio; from lingxi.app import build_app; asyncio.run(build_app())" 2>&1 | head -20
```
Expected: no exception.

- [ ] **Step 5: Commit**
```bash
git add src/lingxi/app.py
git commit -m "app: bootstrap ImportanceScorer + wire into all writers"
```

---

## Phase C: Reflection Upgrade

### Task C.1: Create ReflectionTrigger

**Files:**
- Create: `src/lingxi/facts/reflection_trigger.py`
- Create: `tests/facts/test_reflection_trigger.py`

- [ ] **Step 1: Write failing tests**

```python
import asyncio
import pytest


@pytest.mark.asyncio
async def test_trigger_fires_when_threshold_reached():
    from lingxi.facts.reflection_trigger import ReflectionTrigger

    fired = []
    class FakeReflector:
        async def reflect(self):
            fired.append(True)

    t = ReflectionTrigger(FakeReflector(), threshold=10, max_interval_seconds=999)
    await t.observe(3)
    await t.observe(3)
    await t.observe(4)  # accum=10, threshold hit
    await asyncio.sleep(0.05)  # let the create_task settle
    assert len(fired) == 1


@pytest.mark.asyncio
async def test_trigger_fires_after_max_interval():
    from lingxi.facts.reflection_trigger import ReflectionTrigger

    fired = []
    class FakeReflector:
        async def reflect(self):
            fired.append(True)

    t = ReflectionTrigger(FakeReflector(), threshold=999, max_interval_seconds=0.1)
    await t.observe(1)
    await asyncio.sleep(0.15)
    await t.observe(1)  # passing interval triggers fire on next observe
    await asyncio.sleep(0.05)
    assert len(fired) == 1


@pytest.mark.asyncio
async def test_trigger_resets_after_fire():
    from lingxi.facts.reflection_trigger import ReflectionTrigger

    fired = []
    class FakeReflector:
        async def reflect(self):
            fired.append(True)

    t = ReflectionTrigger(FakeReflector(), threshold=10, max_interval_seconds=999)
    for _ in range(4):
        await t.observe(3)  # 12 total — fires once
    await asyncio.sleep(0.05)
    for _ in range(2):
        await t.observe(3)  # 6 — under threshold post-reset
    await asyncio.sleep(0.05)
    assert len(fired) == 1
```

- [ ] **Step 2:** Run. Expected: FAIL.

- [ ] **Step 3: Implement**

`src/lingxi/facts/reflection_trigger.py`:
```python
"""Triggers reflection when importance accumulates past threshold OR
time-since-last-reflection exceeds max_interval. Hybrid policy from spec.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol


class _ReflectorLike(Protocol):
    async def reflect(self) -> None: ...


class ReflectionTrigger:
    def __init__(
        self,
        reflector: _ReflectorLike,
        threshold: int = 150,
        max_interval_seconds: float = 7200.0,
    ):
        self._reflector = reflector
        self._threshold = threshold
        self._max_interval = max_interval_seconds
        self._accum = 0
        self._last_fire = time.monotonic()
        self._lock = asyncio.Lock()

    async def observe(self, importance: int) -> None:
        async with self._lock:
            self._accum += importance
            elapsed = time.monotonic() - self._last_fire
            if self._accum >= self._threshold or elapsed >= self._max_interval:
                self._accum = 0
                self._last_fire = time.monotonic()
                asyncio.create_task(self._safe_reflect())

    async def _safe_reflect(self) -> None:
        try:
            await self._reflector.reflect()
        except Exception as e:
            print(f"[reflection_trigger] reflect failed: {e}", flush=True)
```

- [ ] **Step 4:** Run: `pytest tests/facts/test_reflection_trigger.py -v`. Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/lingxi/facts/reflection_trigger.py tests/facts/test_reflection_trigger.py
git commit -m "facts: ReflectionTrigger (threshold OR 2h hybrid)"
```

### Task C.2: Create Reflector (tree-of-thought, first-person)

**Files:**
- Create: `src/lingxi/facts/reflector.py`
- Create: `tests/facts/test_reflector.py`

- [ ] **Step 1: Write failing tests**

```python
import json
import pytest
from datetime import datetime
from types import SimpleNamespace
from lingxi.facts.models import Fact, Source, FactType


class FakeLLM:
    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.calls = []
        self.system_calls = []

    async def complete(self, *, messages, system=None, **kw):
        self.calls.append(messages[0]["content"])
        self.system_calls.append(system or "")
        return SimpleNamespace(content=self.responses.pop(0))


@pytest.mark.asyncio
async def test_reflector_writes_pattern_per_question(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.inference import InferenceWriter
    from lingxi.facts.reflector import Reflector

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    now = datetime.now()
    for i in range(12):
        f = Fact(subject="aria", content=f"event {i}",
                 source=Source.LIFE_SIMULATED, type=FactType.EVENT,
                 ts=now, importance=5)
        await store.append(f)

    questions_json = json.dumps(["我最近为什么这么累？", "我对工作的态度变了吗？"])
    llm = FakeLLM(questions_json, "我累是因为连轴转。", "对，热情少了。")
    retriever = FactRetriever(store)
    inference_writer = InferenceWriter(store, scorer=None)
    reflector = Reflector(llm, retriever, inference_writer)

    await reflector.reflect()

    patterns = await store.query(subject="aria", type=FactType.PATTERN, limit=10)
    assert len(patterns) == 2
    assert all("我" in p.content for p in patterns)
    assert all(p.importance == 8 for p in patterns)


@pytest.mark.asyncio
async def test_reflector_uses_first_person_aria_system(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.inference import InferenceWriter
    from lingxi.facts.reflector import Reflector

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    for i in range(11):
        await store.append(Fact(
            subject="aria", content=f"e{i}", source=Source.LIFE_SIMULATED,
            type=FactType.EVENT, ts=datetime.now(), importance=5))

    llm = FakeLLM(json.dumps(["q?"]), "answer.")
    reflector = Reflector(llm, FactRetriever(store), InferenceWriter(store, scorer=None))
    await reflector.reflect()

    for sys_msg in llm.system_calls:
        assert "Aria" in sys_msg
        # No third-person about Aria
        assert "她" not in llm.calls[0]
```

- [ ] **Step 2:** Run. Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

`src/lingxi/facts/reflector.py`:
```python
"""Tree-of-thought reflection, first-person AS Aria.

When triggered: pull recent 100 facts, ask Aria (LLM) to generate 3-5
high-level questions worth thinking about, then for each question pull
relevant facts and have Aria answer it with one compact insight. Each
answer is written as a `pattern` fact with high importance.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.retriever import FactQuery, FactRetriever
from lingxi.facts.writers.inference import InferenceWriter
from lingxi.providers.base import LLMProvider


_SYSTEM = "你是 Aria，正在安静地回看自己最近的生活。"


_QUESTIONS_PROMPT = """我看自己最近经历的这些事，有没有什么**值得停下来想一想**的问题？
不要琐碎的（"我今天吃了什么"这种没意义），要那些能让我**真的反思**的——
关于我最近的模式、情绪走向、和别人的关系变化、对自己的认知。

比如：
  - "我最近在工作上是不是有点提不起劲了？"
  - "我跟 X 的相处方式好像有点变了，是哪里变了？"
  - "最近反复在我脑子里冒出来的事是什么？"

写 3-5 个。输出 JSON array of strings，每条就是一个问题，用我自己平时会想的措辞。

我最近经历的事：
{facts_block}
"""

_ANSWER_PROMPT = """问题：{q}

我手头有这些跟问题相关的事：
{facts_block}

我现在想一下这个问题，写**一条洞见**——浓缩，能补上事实之间的关系或趋势。
不要复述事实本身（"我最近忙工作"是废话）。
1-2 句，用我自己想事情时的语气，不要书面化。
"""


class Reflector:
    def __init__(
        self,
        llm: LLMProvider,
        retriever: FactRetriever,
        inference_writer: InferenceWriter,
        model: str | None = None,
        min_facts: int = 10,
        recent_window: int = 100,
        per_question_limit: int = 15,
    ):
        self._llm = llm
        self._retriever = retriever
        self._writer = inference_writer
        self._model = model
        self._min_facts = min_facts
        self._recent_window = recent_window
        self._per_q_limit = per_question_limit

    async def reflect(self) -> None:
        recent = await self._retriever._store.query(
            subject="aria", limit=self._recent_window
        )
        if len(recent) < self._min_facts:
            return

        questions = await self._generate_questions(recent)
        if not questions:
            return

        for q in questions:
            relevant = await self._retriever.fetch(
                FactQuery(subject="aria", semantic=q, limit=self._per_q_limit)
            )
            insight = await self._answer(q, relevant)
            if not insight:
                continue
            pattern = Fact(
                subject="aria",
                content=insight,
                source=Source.LLM_INFERRED,
                type=FactType.PATTERN,
                ts=datetime.now(),
                importance=8,
                tags=[f"reflection_question:{q[:80]}"],
            )
            await self._writer.write_skip_scorer(pattern)

    async def _generate_questions(self, recent: list[Fact]) -> list[str]:
        facts_block = "\n".join(f"  - {f.content}" for f in recent[-50:])
        prompt = _QUESTIONS_PROMPT.format(facts_block=facts_block)
        try:
            kwargs = {"model": self._model} if self._model else {}
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=_SYSTEM,
                max_tokens=400,
                temperature=0.7,
                _debug_purpose="reflection_questions",
                **kwargs,
            )
            data = json.loads(_strip_fences(response.content))
            if isinstance(data, list):
                return [str(q).strip() for q in data if str(q).strip()][:5]
        except Exception as e:
            print(f"[reflector] question gen failed: {e}", flush=True)
        return []

    async def _answer(self, q: str, facts: list[Fact]) -> str:
        if not facts:
            return ""
        facts_block = "\n".join(f"  - {f.content}" for f in facts)
        prompt = _ANSWER_PROMPT.format(q=q, facts_block=facts_block)
        try:
            kwargs = {"model": self._model} if self._model else {}
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=_SYSTEM,
                max_tokens=200,
                temperature=0.7,
                _debug_purpose="reflection_answer",
                **kwargs,
            )
            return response.content.strip()
        except Exception as e:
            print(f"[reflector] answer failed for q={q!r}: {e}", flush=True)
            return ""


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()
```

- [ ] **Step 4:** Run: `pytest tests/facts/test_reflector.py -v`. Expected: PASS.

- [ ] **Step 5: Voice grep**
```bash
grep -nE "她|他|为 Aria|Aria 当前" src/lingxi/facts/reflector.py
```
Expected: no matches in prompt strings (the word "Aria" in `_SYSTEM` is OK because it's identifying the LLM's role, not third-person reference).

- [ ] **Step 6: Commit**
```bash
git add src/lingxi/facts/reflector.py tests/facts/test_reflector.py
git commit -m "facts: Reflector (tree-of-thought, first-person Aria)"
```

### Task C.3: Wire trigger into writers via app.py

**Files:**
- Modify: `src/lingxi/app.py`

- [ ] **Step 1:** In `app.py`, after constructing scorer + writers, construct reflector and trigger:
```python
from lingxi.facts.reflector import Reflector
from lingxi.facts.reflection_trigger import ReflectionTrigger

reflector = Reflector(llm_provider, fact_retriever, inference_writer)
reflection_trigger = ReflectionTrigger(reflector)
```

- [ ] **Step 2:** Pass `reflection_trigger` to every writer that should drive reflection — Aria-side writes drive Aria reflection. Pass to: `LifeWriter`, `UserStatementWriter`, `InferenceWriter`. Do NOT pass to `NPCWriter`, `WorldWriter`, `BiographyLoader` (those aren't observations Aria's reflecting on).
```python
life_writer = LifeWriter(facts_store, scorer=importance_scorer,
                         reflection_trigger=reflection_trigger)
user_writer = UserStatementWriter(facts_store, scorer=importance_scorer,
                                  reflection_trigger=reflection_trigger)
inference_writer = InferenceWriter(facts_store, scorer=importance_scorer,
                                   reflection_trigger=reflection_trigger)
# unchanged:
npc_writer = NPCWriter(facts_store, scorer=importance_scorer)
world_writer = WorldWriter(facts_store, scorer=importance_scorer)
```

Note: `inference_writer` here is used by Reflector itself — there's a chicken-and-egg. Reflector writes via inference_writer, which would re-trigger reflection. Fix: when Reflector calls `write_skip_scorer`, also bypass trigger. Add `trigger_observation: bool = True` param to `write_skip_scorer` and pass `False` from Reflector.

- [ ] **Step 3:** Update `facts/writers/base.py::write_skip_scorer`:
```python
async def write_skip_scorer(self, fact: Fact, trigger_observation: bool = True) -> None:
    self._enforce(fact)
    if fact.importance is None:
        fact.importance = 5
    await self._store.append(fact)
    if self._trigger is not None and trigger_observation:
        await self._trigger.observe(fact.importance)
```

And in `reflector.py`, change `await self._writer.write_skip_scorer(pattern)` to `await self._writer.write_skip_scorer(pattern, trigger_observation=False)`.

- [ ] **Step 4: Smoke import**
```bash
python -c "import asyncio; from lingxi.app import build_app; asyncio.run(build_app())"
```
Expected: no exception.

- [ ] **Step 5: Commit**
```bash
git add src/lingxi/app.py src/lingxi/facts/writers/base.py src/lingxi/facts/reflector.py
git commit -m "app: wire Reflector + ReflectionTrigger (Aria-side writers only)"
```

### Task C.4: Delete old temporal reflection stub

**Files:**
- Modify: `src/lingxi/app.py`
- Delete: `src/lingxi/temporal/reflection.py`

- [ ] **Step 1:** In `app.py`, grep for any `run_reflection_loop` call or scheduled task that uses the old reflection. Delete.

- [ ] **Step 2:** Delete the stub:
```bash
rm src/lingxi/temporal/reflection.py
```

- [ ] **Step 3:** Run: `pytest tests/ -v -x`. Expected: PASS.

- [ ] **Step 4: Commit**
```bash
git add -A
git commit -m "temporal: remove old reflection stub (facts/reflector owns it)"
```

---

## Phase D: Daily Planner + Reactive Replan

### Task D.1: Add FactType.PLAN

**Files:**
- Modify: `src/lingxi/facts/models.py`
- Test: `tests/facts/test_models.py`

- [ ] **Step 1: Write failing test**
```python
def test_plan_fact_type_exists():
    from lingxi.facts.models import FactType
    assert FactType.PLAN.value == "plan"
```

- [ ] **Step 2:** Run. Expected: FAIL (`PLAN` not in enum).

- [ ] **Step 3: Implement**

In `models.py` `FactType` enum, add `PLAN = "plan"` after `OPINION`.

- [ ] **Step 4:** Run test. Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/lingxi/facts/models.py tests/facts/test_models.py
git commit -m "facts: add FactType.PLAN"
```

### Task D.2: Create DailyPlanner.plan_aria()

**Files:**
- Create: `src/lingxi/planner/__init__.py`
- Create: `src/lingxi/planner/daily_planner.py`
- Create: `tests/planner/__init__.py`
- Create: `tests/planner/test_daily_planner.py`

- [ ] **Step 1: Create empty `__init__.py` files**
```bash
mkdir -p src/lingxi/planner tests/planner
touch src/lingxi/planner/__init__.py tests/planner/__init__.py
```

- [ ] **Step 2: Write failing tests**

```python
import json
import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace
from lingxi.facts.models import Fact, Source, FactType


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = []
        self.systems = []

    async def complete(self, *, messages, system=None, **kw):
        self.calls.append(messages[0]["content"])
        self.systems.append(system or "")
        return SimpleNamespace(content=self.response)


@pytest.mark.asyncio
async def test_plan_aria_writes_plan_facts(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.life import LifeWriter
    from lingxi.planner.daily_planner import DailyPlanner

    plan_json = json.dumps([
        {"time_window": "07:00-08:00", "content": "起床 看一会儿天文新闻", "goal": "保持手感"},
        {"time_window": "09:00-12:00", "content": "跑光变曲线第三组分析", "goal": "M31 paper"},
        {"time_window": "14:00-18:00", "content": "写 paper 第二节"},
        {"time_window": "20:00-22:00", "content": "读 Rao 那篇预印本"},
    ])
    llm = FakeLLM(plan_json)
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    life_writer = LifeWriter(store, scorer=None)
    planner = DailyPlanner(llm, FactRetriever(store), life_writer)

    await planner.plan_aria()

    plans = await store.query(subject="aria", type=FactType.PLAN, limit=10)
    assert len(plans) == 4
    assert all(p.importance == 7 for p in plans)
    assert all(any(t.startswith("time_window:") for t in p.tags) for p in plans)
    assert all(p.expires_at is not None for p in plans)


@pytest.mark.asyncio
async def test_plan_aria_uses_first_person_system(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.life import LifeWriter
    from lingxi.planner.daily_planner import DailyPlanner

    llm = FakeLLM(json.dumps([{"time_window": "09:00-10:00", "content": "x"}]))
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    planner = DailyPlanner(llm, FactRetriever(store), LifeWriter(store, scorer=None))
    await planner.plan_aria()
    assert "你是 Aria" in llm.systems[0]
    assert "她" not in llm.calls[0]
```

- [ ] **Step 3:** Run. Expected: FAIL (module missing).

- [ ] **Step 4: Implement**

`src/lingxi/planner/daily_planner.py`:
```python
"""Daily planner — Aria and (later) NPCs plan their own days,
first-person, in the morning.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, time, timedelta

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.retriever import FactQuery, FactRetriever
from lingxi.facts.writers.life import LifeWriter
from lingxi.providers.base import LLMProvider


_ARIA_SYSTEM = "你是 Aria，正在早上安排今天打算做什么。"


_ARIA_PROMPT = """新的一天。我想一下今天打算怎么过。

【我是谁】
{biography}

【昨天我反思到的】
{reflections}

【最近一周我注意到的模式】
{patterns}

【我自己定的规矩】
- 6-10 条今天的安排
- 覆盖白天工作时间（9-12, 14-18）+ 晚上 + 早晚习惯
- hour 粒度，time_window 形如 "09:00-12:00"
- 写**具体**的事（"跑光变曲线第三组分析"而不是"工作"——我自己心里知道在做什么）
- 至少 2 条对应到我长期在做的事

输出 JSON：
[{{"time_window": "07:00-08:00", "content": "...", "goal": "..."}}, ...]
content 用我自己想事情的语气，第一人称，但不要在每条前面写"我"——直接写动作。
"""


def _end_of_day(now: datetime) -> datetime:
    return now.replace(hour=23, minute=59, second=59, microsecond=0)


class DailyPlanner:
    def __init__(
        self,
        llm: LLMProvider,
        retriever: FactRetriever,
        life_writer: LifeWriter,
        model: str | None = None,
    ):
        self._llm = llm
        self._retriever = retriever
        self._writer = life_writer
        self._model = model

    async def plan_aria(self) -> list[Fact]:
        biography = await self._load_biography_summary()
        yesterday = datetime.now() - timedelta(days=1)
        reflections = await self._retriever.fetch(
            FactQuery(subject="aria", type=FactType.PATTERN,
                      since=yesterday.replace(hour=0, minute=0), limit=5)
        )
        week_ago = datetime.now() - timedelta(days=7)
        patterns = await self._retriever.fetch(
            FactQuery(subject="aria", type=FactType.PATTERN,
                      since=week_ago, limit=10)
        )

        prompt = _ARIA_PROMPT.format(
            biography=biography,
            reflections=self._bullets(reflections) or "（昨天没特别的反思）",
            patterns=self._bullets(patterns) or "（最近没新模式）",
        )
        items = await self._call_planner(prompt, _ARIA_SYSTEM)
        return await self._write_plan_facts("aria", items)

    async def _call_planner(self, prompt: str, system: str) -> list[dict]:
        try:
            kwargs = {"model": self._model} if self._model else {}
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=system,
                max_tokens=800,
                temperature=0.5,
                _debug_purpose="daily_planner",
                **kwargs,
            )
            data = json.loads(_strip_fences(response.content))
            if isinstance(data, list):
                return [item for item in data
                        if isinstance(item, dict) and "time_window" in item and "content" in item]
        except Exception as e:
            print(f"[planner] LLM/parse failed: {e}", flush=True)
        return []

    async def _write_plan_facts(self, subject: str, items: list[dict]) -> list[Fact]:
        if not items:
            return []
        now = datetime.now()
        expires = _end_of_day(now)
        written: list[Fact] = []
        for item in items:
            tags = [f"time_window:{item['time_window']}"]
            if item.get("goal"):
                tags.append(f"goal:{item['goal']}")
            fact = Fact(
                subject=subject,
                content=str(item["content"]).strip(),
                source=Source.LIFE_SIMULATED,
                type=FactType.PLAN,
                ts=now,
                importance=7,
                expires_at=expires,
                tags=tags,
            )
            await self._writer.write_skip_scorer(fact, trigger_observation=False)
            written.append(fact)
        return written

    async def _load_biography_summary(self) -> str:
        bio = await self._retriever.fetch(
            FactQuery(subject="aria", semantic="身份", limit=5)
        )
        if not bio:
            return "（暂无身份摘要）"
        return self._bullets(bio)

    @staticmethod
    def _bullets(facts: list[Fact]) -> str:
        return "\n".join(f"  - {f.content}" for f in facts)


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()
```

- [ ] **Step 5:** Run: `pytest tests/planner/test_daily_planner.py -v`. Expected: PASS.

- [ ] **Step 6: Commit**
```bash
git add src/lingxi/planner tests/planner
git commit -m "planner: DailyPlanner.plan_aria (first-person, hour granularity)"
```

### Task D.3: Create PlanExecutor (replaces simulator)

**Files:**
- Create: `src/lingxi/planner/executor.py`
- Create: `tests/planner/test_executor.py`

- [ ] **Step 1: Write failing tests**

```python
import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace
from lingxi.facts.models import Fact, Source, FactType


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = []
        self.systems = []
    async def complete(self, *, messages, system=None, **kw):
        self.calls.append(messages[0]["content"])
        self.systems.append(system or "")
        return SimpleNamespace(content=self.response)


@pytest.mark.asyncio
async def test_executor_writes_event_for_current_plan(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.life import LifeWriter
    from lingxi.planner.executor import PlanExecutor

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    life_writer = LifeWriter(store, scorer=None)
    now = datetime.now()
    plan = Fact(
        subject="aria", content="跑光变曲线第三组分析", source=Source.LIFE_SIMULATED,
        type=FactType.PLAN, ts=now, importance=7,
        expires_at=now.replace(hour=23, minute=59),
        tags=[f"time_window:{now.hour:02d}:00-{(now.hour+2)%24:02d}:00"],
    )
    await store.append(plan)

    llm = FakeLLM("第三组结果跑出来了，左侧有个小尖峰，正在排查是不是热噪声。")
    executor = PlanExecutor(llm, FactRetriever(store), life_writer)
    await executor.tick()

    events = await store.query(subject="aria", type=FactType.EVENT, limit=5)
    assert len(events) >= 1
    assert "尖峰" in events[0].content


@pytest.mark.asyncio
async def test_executor_skips_when_no_current_plan(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.life import LifeWriter
    from lingxi.planner.executor import PlanExecutor

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    llm = FakeLLM("should not be called")
    executor = PlanExecutor(llm, FactRetriever(store), LifeWriter(store, scorer=None))
    await executor.tick()
    assert llm.calls == []


@pytest.mark.asyncio
async def test_executor_uses_first_person_aria_system(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.life import LifeWriter
    from lingxi.planner.executor import PlanExecutor

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    now = datetime.now()
    await store.append(Fact(
        subject="aria", content="工作", source=Source.LIFE_SIMULATED,
        type=FactType.PLAN, ts=now, importance=7,
        tags=[f"time_window:{now.hour:02d}:00-{(now.hour+1)%24:02d}:00"],
    ))
    llm = FakeLLM("moment")
    executor = PlanExecutor(llm, FactRetriever(store), LifeWriter(store, scorer=None))
    await executor.tick()
    assert "你是 Aria" in llm.systems[0]
    assert "她" not in llm.calls[0]
```

- [ ] **Step 2:** Run. Expected: FAIL.

- [ ] **Step 3: Implement**

`src/lingxi/planner/executor.py`:
```python
"""Plan executor — replaces the random simulator. Every 30min tick,
finds the plan covering the current hour, generates a concrete
first-person moment, and writes it as an event fact.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.retriever import FactQuery, FactRetriever
from lingxi.facts.writers.life import LifeWriter
from lingxi.planner.daily_planner import DailyPlanner
from lingxi.providers.base import LLMProvider


_SYSTEM = "你是 Aria，正在做今天计划里的某件事。现在写一条记录给自己看。"


_MOMENT_PROMPT = """我今天这个时段安排的：{plan_content}（{time_window}）
最近 2 小时我经历过：
{recent_events}

现在是 {now_hhmm}——我正在做这件事的**某个具体片段**。
写一条**现在这一刻**——具体细节，不抽象描述（数据/物件/手感/感受任一）。
1-2 句，第一人称当下时态。
不要在前面写"我"——直接写动作或观察。
"""


_TW_RE = re.compile(r"^(\d{2}):(\d{2})-(\d{2}):(\d{2})$")


def _parse_time_window(tag_value: str) -> tuple[int, int] | None:
    m = _TW_RE.match(tag_value)
    if not m:
        return None
    start_h, _, end_h, _ = map(int, m.groups())
    return start_h, end_h


class PlanExecutor:
    def __init__(
        self,
        llm: LLMProvider,
        retriever: FactRetriever,
        life_writer: LifeWriter,
        planner: DailyPlanner | None = None,
        model: str | None = None,
    ):
        self._llm = llm
        self._retriever = retriever
        self._writer = life_writer
        self._planner = planner  # optional; needed only if replan support is wired
        self._model = model
        self._replan_requested = False

    def request_replan(self) -> None:
        self._replan_requested = True

    async def tick(self) -> None:
        now = datetime.now()

        if self._replan_requested and self._planner is not None:
            try:
                await self._planner.plan_aria()
            finally:
                self._replan_requested = False

        current_plan = await self._find_current_plan(now)
        if current_plan is None:
            return

        recent_events = await self._retriever.fetch(FactQuery(
            subject="aria", type=FactType.EVENT,
            since=now - timedelta(hours=2), limit=3,
        ))
        tw = self._tag_value(current_plan, "time_window") or "?"
        prompt = _MOMENT_PROMPT.format(
            plan_content=current_plan.content,
            time_window=tw,
            recent_events=self._bullets(recent_events) or "（没什么特别的）",
            now_hhmm=now.strftime("%H:%M"),
        )
        try:
            kwargs = {"model": self._model} if self._model else {}
            response = await self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=_SYSTEM,
                max_tokens=200,
                temperature=0.8,
                _debug_purpose="plan_executor_moment",
                **kwargs,
            )
            content = response.content.strip()
        except Exception as e:
            print(f"[executor] moment gen failed: {e}", flush=True)
            return

        if not content:
            return

        event = Fact(
            subject="aria",
            content=content,
            source=Source.LIFE_SIMULATED,
            type=FactType.EVENT,
            ts=now,
        )
        await self._writer.write(event)

    async def _find_current_plan(self, now: datetime) -> Fact | None:
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        plans = await self._retriever._store.query(
            subject="aria", type=FactType.PLAN, since=today_start, limit=20,
        )
        for plan in plans:
            tw_value = self._tag_value(plan, "time_window")
            if not tw_value:
                continue
            window = _parse_time_window(tw_value)
            if window is None:
                continue
            start_h, end_h = window
            if start_h <= now.hour < end_h:
                return plan
        return None

    @staticmethod
    def _tag_value(fact: Fact, key: str) -> str | None:
        for t in fact.tags:
            if t.startswith(f"{key}:"):
                return t[len(key) + 1:]
        return None

    @staticmethod
    def _bullets(facts: list[Fact]) -> str:
        return "\n".join(f"  - {f.content}" for f in facts)
```

- [ ] **Step 4:** Run: `pytest tests/planner/test_executor.py -v`. Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/lingxi/planner/executor.py tests/planner/test_executor.py
git commit -m "planner: PlanExecutor (replaces random simulator; first-person moment)"
```

### Task D.4: Add plan_conflict to OrchestrationDecision

**Files:**
- Modify: `src/lingxi/brain/models.py`
- Modify: `tests/brain/test_models.py` (create if missing)

- [ ] **Step 1: Write failing test**
```python
def test_decision_has_plan_conflict_default_false():
    from lingxi.brain.models import OrchestrationDecision
    d = OrchestrationDecision.default()
    assert d.plan_conflict is False

def test_decision_from_dict_parses_plan_conflict():
    from lingxi.brain.models import OrchestrationDecision
    d = OrchestrationDecision.from_dict({"plan_conflict": True})
    assert d.plan_conflict is True
```

- [ ] **Step 2:** Run. Expected: FAIL.

- [ ] **Step 3: Implement**

In `OrchestrationDecision` dataclass: add `plan_conflict: bool = False`.

In `default()`: include `plan_conflict=False`.

In `from_dict()`: add `plan_conflict=bool(raw.get("plan_conflict", False))`.

- [ ] **Step 4:** Run. Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/lingxi/brain/models.py tests/brain/test_models.py
git commit -m "brain: add plan_conflict to OrchestrationDecision"
```

### Task D.5: Update orchestrator prompt to detect plan_conflict

**Files:**
- Modify: `src/lingxi/brain/orchestrator.py`

- [ ] **Step 1:** In `_PROMPT` template in `orchestrator.py`, add a section after `thread_summary`:
```
6. **plan_conflict** (bool)：用户输入是否暗示当前 plan 需要调整？
   - 用户邀约/请求 Aria 改变行程（"晚上一起吃饭吧"、"明天有空吗"）→ true
   - 用户提到 Aria 正在做某事 → false（plan 在正常执行）
   - 用户问无关问题、聊天 → false
   - **仅当冲突明显时才标 true**，谨慎使用。
```

And add to the JSON example:
```json
{
  ...existing fields...,
  "plan_conflict": false
}
```

- [ ] **Step 2:** Verify `from_dict` already maps it (done in D.4).

- [ ] **Step 3:** Run existing orchestrator tests: `pytest tests/brain -v`. Expected: PASS.

- [ ] **Step 4: Commit**
```bash
git add src/lingxi/brain/orchestrator.py
git commit -m "brain: orchestrator prompt asks for plan_conflict detection"
```

### Task D.6: Wire plan_conflict → executor.request_replan() in engine

**Files:**
- Modify: `src/lingxi/conversation/engine.py`

- [ ] **Step 1:** In `engine.py`, add `plan_executor` to `ConversationEngine.__init__` kwargs (default None).

- [ ] **Step 2:** In `_prepare_turn_v2` (where the orchestrator decision is obtained), after the decision is parsed, add:
```python
if decision.plan_conflict and self.plan_executor is not None:
    self.plan_executor.request_replan()
```

- [ ] **Step 3:** Run: `pytest tests/conversation -v`. Expected: PASS.

- [ ] **Step 4: Commit**
```bash
git add src/lingxi/conversation/engine.py
git commit -m "engine: route plan_conflict → executor.request_replan()"
```

### Task D.7: Replace simulator with PlanExecutor in app.py + schedule morning planner

**Files:**
- Modify: `src/lingxi/app.py`
- Delete: `src/lingxi/inner_life/simulator.py`

- [ ] **Step 1:** In `app.py`, construct planner + executor:
```python
from lingxi.planner.daily_planner import DailyPlanner
from lingxi.planner.executor import PlanExecutor

daily_planner = DailyPlanner(llm_provider, fact_retriever, life_writer)
plan_executor = PlanExecutor(llm_provider, fact_retriever, life_writer, planner=daily_planner)
```

- [ ] **Step 2:** Pass `plan_executor` to `ConversationEngine(...)` (per D.6 signature).

- [ ] **Step 3:** Find where the simulator's background tick is scheduled (`asyncio.create_task` around simulator). Replace with executor tick:
```python
async def _executor_loop():
    while True:
        try:
            await plan_executor.tick()
        except Exception as e:
            print(f"[executor] tick error: {e}", flush=True)
        await asyncio.sleep(1800)  # 30 min

asyncio.create_task(_executor_loop())
```

- [ ] **Step 4:** Add morning planner loop. Compute next 7am, sleep till it, then run, then sleep 24h:
```python
async def _morning_planner_loop():
    while True:
        now = datetime.now()
        next_7am = now.replace(hour=7, minute=0, second=0, microsecond=0)
        if next_7am <= now:
            next_7am += timedelta(days=1)
        await asyncio.sleep((next_7am - now).total_seconds())
        try:
            await daily_planner.plan_aria()
        except Exception as e:
            print(f"[planner] morning tick failed: {e}", flush=True)

asyncio.create_task(_morning_planner_loop())
```

Also: on startup, check if today's plan exists; if not, run one immediately:
```python
today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
todays_plans = await facts_store.query(
    subject="aria", type=FactType.PLAN, since=today_start, limit=1
)
if not todays_plans:
    await daily_planner.plan_aria()
```

- [ ] **Step 5:** Delete the simulator file and its import in app.py:
```bash
rm src/lingxi/inner_life/simulator.py
```
Remove `from lingxi.inner_life.simulator import ...` and any remaining simulator construction.

- [ ] **Step 6: Smoke import**
```bash
python -c "import asyncio; from lingxi.app import build_app; asyncio.run(build_app())"
```
Expected: no exception, log lines include `[planner]` or no [life simulator] reference.

- [ ] **Step 7: Commit**
```bash
git add -A
git commit -m "app: replace simulator with PlanExecutor + morning DailyPlanner loop"
```

---

## Phase E: NPC Parity

### Task E.1: Add DailyPlanner.plan_npc()

**Files:**
- Modify: `src/lingxi/planner/daily_planner.py`
- Modify: `tests/planner/test_daily_planner.py`

- [ ] **Step 1: Write failing test**

```python
@pytest.mark.asyncio
async def test_plan_npc_writes_plan_facts_under_npc_subject(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.life import LifeWriter
    from lingxi.facts.writers.npc import NPCWriter
    from lingxi.planner.daily_planner import DailyPlanner
    import json

    plan_json = json.dumps([
        {"time_window": "09:00-12:00", "content": "改实验报告"},
        {"time_window": "20:00-22:00", "content": "看新一季动画"},
    ])
    llm = FakeLLM(plan_json)
    store = FactStore(tmp_path / "facts.db")
    await store.init()
    npc_writer = NPCWriter(store, scorer=None)
    life_writer = LifeWriter(store, scorer=None)
    planner = DailyPlanner(llm, FactRetriever(store), life_writer, npc_writer=npc_writer)

    await planner.plan_npc("xiaomin", display_name="小敏")

    plans = await store.query(subject="npc:xiaomin", type=FactType.PLAN, limit=10)
    assert len(plans) == 2
    assert "你是 小敏" in llm.systems[0]
```

- [ ] **Step 2:** Run. Expected: FAIL.

- [ ] **Step 3: Implement**

In `daily_planner.py`:
```python
_NPC_SYSTEM_TEMPLATE = "你是 {name}，正在早上想今天打算做什么。"

_NPC_PROMPT = """新的一天。我想一下今天打算怎么过。

【我是谁】
{biography}

【最近的事】
{recent}

【规矩】
- 2-3 条粗的安排（不用太细）
- hour 粒度，time_window 形如 "09:00-12:00"
- 写**具体**的事，第一人称语气
- 不要在每条前面写"我"

输出 JSON：
[{{"time_window": "...", "content": "..."}}, ...]
"""
```

Update `DailyPlanner.__init__` to accept `npc_writer=None`. Add method:
```python
async def plan_npc(self, npc_id: str, display_name: str | None = None) -> list[Fact]:
    if self._npc_writer is None:
        raise RuntimeError("plan_npc requires npc_writer to be configured")
    subject = f"npc:{npc_id}"
    name = display_name or npc_id

    bio_facts = await self._retriever.fetch(
        FactQuery(subject=subject, semantic="身份", limit=3)
    )
    recent = await self._retriever.fetch(
        FactQuery(subject=subject, type=FactType.EVENT, limit=5,
                  since=datetime.now() - timedelta(days=3))
    )
    prompt = _NPC_PROMPT.format(
        biography=self._bullets(bio_facts) or "（暂无身份摘要）",
        recent=self._bullets(recent) or "（最近没什么大事）",
    )
    system = _NPC_SYSTEM_TEMPLATE.format(name=name)
    items = await self._call_planner(prompt, system)
    return await self._write_plan_facts_with(self._npc_writer, subject, items)


async def _write_plan_facts_with(self, writer, subject: str, items: list[dict]) -> list[Fact]:
    # Same body as _write_plan_facts but using passed writer
    if not items:
        return []
    now = datetime.now()
    expires = _end_of_day(now)
    written = []
    for item in items:
        tags = [f"time_window:{item['time_window']}"]
        if item.get("goal"):
            tags.append(f"goal:{item['goal']}")
        fact = Fact(
            subject=subject,
            content=str(item["content"]).strip(),
            source=Source.NPC_TICKER if subject.startswith("npc:") else Source.LIFE_SIMULATED,
            type=FactType.PLAN,
            ts=now,
            importance=6,
            expires_at=expires,
            tags=tags,
        )
        await writer.write_skip_scorer(fact, trigger_observation=False)
        written.append(fact)
    return written
```

Refactor existing `_write_plan_facts` to call `_write_plan_facts_with(self._writer, subject, items)`.

- [ ] **Step 4:** Run: `pytest tests/planner/test_daily_planner.py -v`. Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/lingxi/planner/daily_planner.py tests/planner/test_daily_planner.py
git commit -m "planner: plan_npc (first-person AS NPC, 2-3 coarse plans)"
```

### Task E.2: Schedule NPC plan generation in morning tick

**Files:**
- Modify: `src/lingxi/app.py`

- [ ] **Step 1:** In `app.py`, construct planner with `npc_writer`:
```python
daily_planner = DailyPlanner(llm_provider, fact_retriever, life_writer, npc_writer=npc_writer)
```

- [ ] **Step 2:** In `_morning_planner_loop`, also iterate NPCs:
```python
NPC_REGISTRY = [
    ("xiaomin", "小敏"),
    ("liwei", "李伟"),
    ("yuxin", "雨欣"),
    ("zhangwei", "张伟"),
    ("chen", "晨晨"),
    ("nana", "娜娜"),
]
# Inside _morning_planner_loop, after plan_aria():
for npc_id, display in NPC_REGISTRY:
    try:
        await daily_planner.plan_npc(npc_id, display_name=display)
    except Exception as e:
        print(f"[planner] NPC {npc_id} plan failed: {e}", flush=True)
```

Note: the actual NPC list should match `config/personas/*.yaml`. Read that directory at startup to build `NPC_REGISTRY` instead of hardcoding. Example:
```python
from pathlib import Path
import yaml
NPC_REGISTRY = []
for p in Path("config/personas").glob("*.yaml"):
    with open(p) as f:
        data = yaml.safe_load(f)
    if data.get("kind") == "npc":
        NPC_REGISTRY.append((data["id"], data.get("display_name", data["id"])))
```

- [ ] **Step 3:** Also run on startup if no NPC plans exist for today (same pattern as Aria in D.7).

- [ ] **Step 4: Smoke import**
```bash
python -c "import asyncio; from lingxi.app import build_app; asyncio.run(build_app())"
```

- [ ] **Step 5: Commit**
```bash
git add src/lingxi/app.py
git commit -m "app: schedule NPC plan generation in morning tick"
```

### Task E.3: Create bidirectional_interaction()

**Files:**
- Create: `src/lingxi/social/interaction.py`
- Create: `tests/social/test_interaction.py`

- [ ] **Step 1: Write failing test**

```python
import pytest
from datetime import datetime
from types import SimpleNamespace
from lingxi.facts.models import Fact, Source, FactType


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.systems = []
    async def complete(self, *, messages, system=None, **kw):
        self.systems.append(system or "")
        return SimpleNamespace(content=self.responses.pop(0))


@pytest.mark.asyncio
async def test_bidirectional_writes_both_sides(tmp_path):
    from lingxi.facts.store import FactStore
    from lingxi.facts.retriever import FactRetriever
    from lingxi.facts.writers.life import LifeWriter
    from lingxi.facts.writers.npc import NPCWriter
    from lingxi.social.interaction import bidirectional_interaction

    store = FactStore(tmp_path / "facts.db")
    await store.init()
    life_writer = LifeWriter(store, scorer=None)
    npc_writer = NPCWriter(store, scorer=None)
    llm = FakeLLM(
        "刚刚跟小敏聊她答辩的事，她声音有点紧。",
        "刚刚跟 Aria 说了下答辩的事，她让我别想太多。"
    )

    await bidirectional_interaction(
        llm=llm,
        retriever=FactRetriever(store),
        life_writer=life_writer,
        npc_writer=npc_writer,
        npc_id="xiaomin",
        npc_display="小敏",
        scenario="聊她下周的答辩",
    )

    aria_events = await store.query(subject="aria", type=FactType.EVENT, limit=5)
    npc_events = await store.query(subject="npc:xiaomin", type=FactType.EVENT, limit=5)
    assert len(aria_events) == 1 and "小敏" in aria_events[0].content
    assert len(npc_events) == 1 and "Aria" in npc_events[0].content
    # Both systems first-person
    assert any("你是 Aria" in s for s in llm.systems)
    assert any("你是 小敏" in s for s in llm.systems)
```

- [ ] **Step 2:** Run. Expected: FAIL.

- [ ] **Step 3: Implement**

`src/lingxi/social/interaction.py`:
```python
"""Bidirectional NPC↔Aria interaction.

A single social event produces TWO facts — one from Aria's first-person
view, one from the NPC's first-person view. Two LLM calls (separate
voices) keeps each output clean.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from lingxi.facts.models import Fact, FactType, Source
from lingxi.facts.retriever import FactQuery, FactRetriever
from lingxi.facts.writers.life import LifeWriter
from lingxi.facts.writers.npc import NPCWriter
from lingxi.providers.base import LLMProvider


_ARIA_SYSTEM_TEMPLATE = "你是 Aria，刚才跟 {npc} 有了一次互动，现在用一句话记一下。"
_NPC_SYSTEM_TEMPLATE = "你是 {npc}，刚才跟 Aria 有了一次互动，现在用一句话记一下。"


_ARIA_PROMPT = """情境：{scenario}
{npc} 现在大概在：{npc_plan}
我今天大概在：{aria_plan}
我们最近的交集：
{shared_history}

记一下这次互动给我留下的印象——一两句，具体细节（对方说了什么 / 我的反应 / 一个感觉）。
"""

_NPC_PROMPT_TEMPLATE = """情境：{scenario}
Aria 现在大概在：{aria_plan}
我今天大概在：{npc_plan}
我们最近的交集：
{shared_history}

记一下这次互动给我留下的印象——一两句，具体细节。
"""


async def bidirectional_interaction(
    *,
    llm: LLMProvider,
    retriever: FactRetriever,
    life_writer: LifeWriter,
    npc_writer: NPCWriter,
    npc_id: str,
    npc_display: str,
    scenario: str,
    model: str | None = None,
) -> None:
    subject_npc = f"npc:{npc_id}"
    now = datetime.now()

    # Fetch context
    aria_plan = await _current_plan_summary(retriever, "aria", now)
    npc_plan = await _current_plan_summary(retriever, subject_npc, now)
    shared = await _shared_history(retriever, subject_npc, now)

    shared_text = "\n".join(f"  - {f.content}" for f in shared) or "（最近没什么具体交集）"

    # Aria's side
    aria_prompt = _ARIA_PROMPT.format(
        scenario=scenario, npc=npc_display,
        npc_plan=npc_plan or "（不太清楚）",
        aria_plan=aria_plan or "（在做手头的事）",
        shared_history=shared_text,
    )
    aria_view = await _safe_complete(
        llm, _ARIA_SYSTEM_TEMPLATE.format(npc=npc_display), aria_prompt,
        purpose="interaction_aria_view", model=model,
    )

    # NPC's side
    npc_prompt = _NPC_PROMPT_TEMPLATE.format(
        scenario=scenario, aria_plan=aria_plan or "（看上去在忙）",
        npc_plan=npc_plan or "（在做自己的事）",
        shared_history=shared_text,
    )
    npc_view = await _safe_complete(
        llm, _NPC_SYSTEM_TEMPLATE.format(npc=npc_display), npc_prompt,
        purpose="interaction_npc_view", model=model,
    )

    if aria_view:
        await life_writer.write(Fact(
            subject="aria", content=aria_view,
            source=Source.LIFE_SIMULATED, type=FactType.EVENT, ts=now,
        ))
    if npc_view:
        await npc_writer.write(Fact(
            subject=subject_npc, content=npc_view,
            source=Source.NPC_TICKER, type=FactType.EVENT, ts=now,
        ))


async def _current_plan_summary(
    retriever: FactRetriever, subject: str, now: datetime
) -> str:
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    plans = await retriever._store.query(
        subject=subject, type=FactType.PLAN, since=today_start, limit=20
    )
    for plan in plans:
        for t in plan.tags:
            if not t.startswith("time_window:"):
                continue
            try:
                start_s, end_s = t.removeprefix("time_window:").split("-")
                start_h = int(start_s.split(":")[0])
                end_h = int(end_s.split(":")[0])
                if start_h <= now.hour < end_h:
                    return plan.content
            except (ValueError, IndexError):
                continue
    return ""


async def _shared_history(
    retriever: FactRetriever, subject_npc: str, now: datetime
) -> list[Fact]:
    week_ago = now - timedelta(days=7)
    aria_about_npc = await retriever.fetch(FactQuery(
        subject="aria", semantic=subject_npc.removeprefix("npc:"),
        since=week_ago, limit=3,
    ))
    npc_recent = await retriever.fetch(FactQuery(
        subject=subject_npc, type=FactType.EVENT,
        since=week_ago, limit=3,
    ))
    return aria_about_npc + npc_recent


async def _safe_complete(
    llm: LLMProvider, system: str, prompt: str, *, purpose: str, model: str | None
) -> str:
    try:
        kwargs = {"model": model} if model else {}
        response = await llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            max_tokens=200,
            temperature=0.7,
            _debug_purpose=purpose,
            **kwargs,
        )
        return response.content.strip()
    except Exception as e:
        print(f"[interaction] {purpose} failed: {e}", flush=True)
        return ""
```

- [ ] **Step 4:** Run: `pytest tests/social/test_interaction.py -v`. Expected: PASS.

- [ ] **Step 5: Voice grep**
```bash
grep -nE "她|他(?! 们)|为 Aria|Aria 当前|她正在" src/lingxi/social/interaction.py
```
Expected: no third-person matches (the "她" pattern checks any third-person reference).

- [ ] **Step 6: Commit**
```bash
git add src/lingxi/social/interaction.py tests/social/test_interaction.py
git commit -m "social: bidirectional_interaction (two first-person LLM calls)"
```

### Task E.4: Wire social/scheduler to use bidirectional_interaction

**Files:**
- Modify: `src/lingxi/social/scheduler.py`

- [ ] **Step 1:** Read the existing scheduler. Find where it triggers an interaction (likely calls `event_generator` or similar and writes via `npc_writer`).

- [ ] **Step 2:** Replace single-sided interaction trigger with `bidirectional_interaction(...)`. Keep the scheduling logic (cron hours, NPC selection, scenario generation) untouched — only swap the write phase.

Example structure (adapt to actual scheduler shape):
```python
from lingxi.social.interaction import bidirectional_interaction

# inside the scheduler tick where an interaction is generated:
scenario = await self._choose_scenario(npc_id)  # existing
await bidirectional_interaction(
    llm=self._llm,
    retriever=self._retriever,
    life_writer=self._life_writer,
    npc_writer=self._npc_writer,
    npc_id=npc_id,
    npc_display=display_name,
    scenario=scenario,
)
```

The scheduler `__init__` may need `life_writer` and `retriever` added if not already present.

- [ ] **Step 3:** In `app.py`, update scheduler construction to pass the additional deps.

- [ ] **Step 4:** Run: `pytest tests/social -v`. Expected: PASS.

- [ ] **Step 5: Smoke run**
```bash
python -c "import asyncio; from lingxi.app import build_app; asyncio.run(build_app())"
```

- [ ] **Step 6: Commit**
```bash
git add src/lingxi/social/scheduler.py src/lingxi/app.py
git commit -m "social: scheduler triggers bidirectional interactions"
```

---

## Phase F: Integration Verification

### Task F.1: Full integration smoke + manual day-run

**Files:** (none modified)

- [ ] **Step 1:** Full suite green:
```bash
cd /Users/lovart/agent-facts-refactor && pytest tests/ -v
```
Expected: all PASS.

- [ ] **Step 2:** Voice grep over all new/modified prompt code paths:
```bash
grep -nE "她|为 Aria|Aria 当前|她正在" \
  src/lingxi/facts/scorer.py \
  src/lingxi/facts/reflector.py \
  src/lingxi/planner/daily_planner.py \
  src/lingxi/planner/executor.py \
  src/lingxi/social/interaction.py
```
Expected: no matches in any prompt strings (the literal "Aria" appearing inside `system="你是 Aria, ..."` is the LLM role declaration, not third-person reference — those are OK).

- [ ] **Step 3: Start the bot in worktree**
```bash
cd /Users/lovart/agent-facts-refactor && .venv/bin/python -m lingxi.channels.feishu
```
Watch for:
- `[planner]` morning tick log (if started before 7am or no plan exists)
- `[executor]` 30-min tick logs writing events
- `[reflection_trigger]` firing when accumulator hits 150

- [ ] **Step 4: Hand off to user for day-use verification**

Tell user: "P7→A through E complete. Bot running with full Generative Agents loop. Chat with Aria for ≥1 day to verify: (a) Aria's daily plan is coherent and her events stay close to it, (b) reflection patterns feel like real insights not waffle, (c) NPC interactions show different perspectives from each side, (d) plan_conflict triggers replan when you make new commitments."

No commit needed — verification phase.

---

## Spec Coverage Self-Check

- Goal — covered (plan executes all 5 phases)
- Why — motivation only, no implementation needed
- Scope → In: every item maps to a task (importance + last_accessed → B.1/B.2; plan type → D.1; scorer → B.4/B.5; retriever 3D → B.6; reflection trigger → C.1; reflector → C.2; planner → D.2/E.1; executor → D.3; orchestrator plan_conflict → D.4/D.5; engine replan → D.6; NPC plan → E.1; NPC plan scheduling → E.2; bidirectional → E.3/E.4; tests → all tasks include them)
- Scope → Out: noted in plan header — no tasks created for spatial/movement, 5-15min sub-steps, user-as-agent, NPC↔NPC, multi-layer reflection
- Prerequisite (P7) → Phase A entirely
- Architecture diagram — implementation matches: writer → store → retriever flow; trigger inside writer; reflector triggered by trigger; planner morning tick writes plans; executor consumes plans
- Module 边界 — file map at top mirrors spec table
- Data Model 改动 — B.1/B.2 implement schema migration + new type via D.1; default importance values present in scorer (B.4)
- Prompt Voice 约定 — hard rule at top of plan, repeated grep checks in B.4 / C.2 / E.3 / F.1
- Phase B: Importance Scorer + 3D Retrieval — tasks B.1–B.7
- Phase C: Reflection 升级 — tasks C.1–C.4
- Phase D: Planner — tasks D.1–D.7
- Phase E: NPC 平权 + 双向 interaction — tasks E.1–E.4
- Orchestrator plan_conflict 检测 → D.5/D.6
- Data Flow 示例 — illustration only, validated by F.1 manual run
- 错误处理 — covered: scorer fallback (B.4), reflector try/except (C.2), planner failure (D.2 `_call_planner`), executor fail-skip (D.3), interaction fail-fallback (E.3)
- Testing — every component has unit tests; F.1 covers integration
- Phasing — strict A→B→C→D→E enforced by plan order; verification handoff in F.1
- 风险 — addressed: plan 假感 (manual verification in F.1), scorer cost (batched in B.4), reflection threshold (configurable in C.1), NPC plan loss-of-randomness (deliberately coarse in E.1)
- Open Questions — none

**Type consistency check:** `write_skip_scorer(fact, trigger_observation=False)` — defined in C.3 (B.5 only defines a version without the bool kwarg). Used in C.2, D.2 (`_write_plan_facts`), E.1 (`_write_plan_facts_with`). Caller in C.2 was written before C.3 added the kwarg — verified C.2 reflector also gets updated in C.3 step 3. **Consistent.**

`PlanExecutor.__init__` takes `planner: DailyPlanner | None`. Constructed in D.7 with planner passed. Replan path in D.3 guarded by `if self._planner is not None`. **Consistent.**

`DailyPlanner.__init__` takes `npc_writer=None`. In D.2 not passed, in E.1 tests pass it via kwarg, in E.2 app.py passes it. `plan_npc` raises if `npc_writer is None`. **Consistent.**

No placeholders found in scan.
