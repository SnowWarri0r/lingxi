# Generative Agents 集成设计

**Date:** 2026-05-28
**Branch:** `refactor/facts-arch` (continuation, on top of facts-arch refactor)
**Status:** Design

## Goal

把 Generative Agents (Park et al. 2023, "Smallville") 论文里证明有效的几个核心机制集成到 Aria：

1. **LLM-scored importance** — 每条 fact 写入时由 LLM 评 1-10 分
2. **3D retrieval** — 排序按 recency × importance × relevance 三维加权
3. **Importance-driven reflection** — 反思由"重要事件累积"触发而非定时
4. **Tree-of-thought reflection** — 反思先生成高阶问题再各自作答，输出更高 importance 的 pattern
5. **Daily planner** — 每日 morning tick 生成 hour 粒度计划，替换 simulator 的"做什么"决策
6. **Reactive replanning** — 用户/世界事件冲突当前 plan 时重算剩余
7. **NPC 平权** — NPC 也走 importance、也有简版日 plan、Aria↔NPC 互动事件双向写

## Why

当前 facts 架构（refactor/facts-arch 分支）解决了"数据归属混乱"——但没解决两个深层问题：

- **检索冷感**：retriever 用 FTS keyword 或 recency 取，重要不重要一视同仁。结果是 Aria 经常拉到"喝咖啡"这种琐事而漏掉"昨晚跟用户聊到失眠"这种关键事件。
- **生活无骨架**：simulator 随机抽活动，导致 Aria 一天内的"在做什么"前后不贯通。用户问"你今天怎么样"她答不出整体感。

论文里这两点恰好是 emergent behavior 的源头。集成进来不是炫技，是补当前架构缺的那块"重要性 + 意图"。

## Scope

### In Scope

- facts 表加 `importance`、`last_accessed` 两列
- 新 fact type: `plan`
- WriterBase 集成 LLM importance scoring（batched）
- Retriever 改单一打分函数
- Reflection 模块改触发逻辑 + 升级到 tree-of-thought
- 新模块 `lingxi/planner/`：daily planner + plan executor
- Simulator 改造为 plan executor 后端
- Orchestrator 输出新增 `plan_conflict` 字段
- NPC 通过现有 NPCWriter 走 importance；新增 NPC plan 生成；新增双向 interaction 写入
- 实际接到 social/scheduler.py 的互动触发上

### Out of Scope

- 论文里的空间/移动（n/a for IM persona）
- 5-15min 子步骤分解（hour 块足够）
- 用户也是 agent（用户始终只是 user，不规划用户行为）
- NPC↔NPC 互动（只做 NPC↔Aria）
- 反思树多层递归（只一层 question → answer）

### Prerequisite

**P7 cleanup 必须先于本 spec 实施。** 当前 refactor/facts-arch 分支仍有 dual-write（旧 store + 新 facts），不清就上新机制会变三重写，调试地狱。P7 cleanup 范围：
- 删 `inner_life/store.py` 的 recent_events 部分
- 删 `relational/store.py`、`social/store.py`、`social/promoter.py`、`world/store.py`
- 删 PromptBuilder 中残余 dynamic section 方法
- 解除 simulator/scheduler/reflection 中的 dual-write 路径
- 删 `tools/migrate_to_facts.py`（一次性脚本）

P7 cleanup 是独立工作，单独的 plan。本 spec 假设 P7 已完成。

## Architecture

### 总体数据流

```
                  ┌──────────────────────────┐
        7am tick  │   Planner                │
        ─────────▶│   - Aria 的 daily plan   │
                  │   - 每个 NPC 的简版 plan  │
                  └──────────┬───────────────┘
                             │ writes type=plan facts
                             ▼
                  ┌──────────────────────────┐
       30min tick │   Plan Executor          │
        ─────────▶│   (替代 simulator)       │
                  │   查当前 plan step       │
                  │   → 生成具体瞬间          │
                  └──────────┬───────────────┘
                             │ writes type=event facts
                             ▼
        ┌────────────────────────────────────────┐
        │   WriterBase (统一入口)                 │
        │   - batched importance scorer          │
        │     (Sonnet, 5 facts / 30s)            │
        │   - update reflection accumulator      │
        │   - if accumulator ≥ 150 OR 2h passed: │
        │       trigger reflection               │
        └─────────────┬──────────────────────────┘
                      │
                      ▼
        ┌────────────────────────────────────────┐
        │   FactStore (SQLite)                   │
        │   columns: ..., importance,            │
        │            last_accessed               │
        └─────────────┬──────────────────────────┘
                      │ on threshold
                      ▼
        ┌────────────────────────────────────────┐
        │   Reflector (升级)                      │
        │   1. read recent 100 facts             │
        │   2. Sonnet: 生成 3-5 个高阶问题       │
        │   3. for each q: retriever.fetch       │
        │   4. Sonnet: 基于事实回答              │
        │   5. write answer as type=pattern,     │
        │      importance≥7, supersedes 同主题旧 │
        └────────────────────────────────────────┘

   用户来消息：
        Orchestrator (Sonnet)
          ├─ existing: fact_queries, topic_anchor, ...
          └─ new: plan_conflict (bool)
                        │
                        ▼ if true
        next executor tick 调 planner 重算

   Retriever.fetch:
        score = 0.5·recency_decay + 0.3·importance/10 + 0.2·fts_rank
        on each returned fact: update last_accessed
```

### 模块边界

| 模块 | 路径 | 职责 |
|---|---|---|
| `facts/models.py` | 已存在 | 加 `importance: int \| None`、`last_accessed: datetime \| None` 字段（None = 未评分/未访问） |
| `facts/store.py` | 已存在 | 加两列 + 索引；新增 `update_last_accessed(ids)` 方法 |
| `facts/scorer.py` | **新建** | `ImportanceScorer` 类，batched LLM 评分 |
| `facts/writers/base.py` | 已存在 | 写入前调 scorer；写入后更新 reflection accumulator |
| `facts/retriever.py` | 已存在 | 重写打分函数（删 FTS/recency 双 path） |
| `facts/reflector.py` | **新建** | tree-of-thought 反思；从老 `temporal/reflection.py` 搬骨架 |
| `facts/reflection_trigger.py` | **新建** | 累积 importance + 时间双触发逻辑 |
| `planner/daily_planner.py` | **新建** | Aria 和 NPC 的 morning plan 生成 |
| `planner/executor.py` | **新建** | 接替 simulator tick；查 plan → 生成 event |
| `planner/replanner.py` | **新建** | 由 orchestrator `plan_conflict` 触发 |
| `brain/models.py` | 已存在 | `OrchestrationDecision` 加 `plan_conflict: bool = False` |
| `brain/orchestrator.py` | 已存在 | prompt 加 plan_conflict 判定逻辑 |
| `social/scheduler.py` | 已存在 | 触发 NPC↔Aria 互动时改用双向写 |

## Data Model 改动

### facts 表 schema 增量

```sql
-- importance NULL 表示 "not yet scored"（write 路径会触发 scorer）
ALTER TABLE facts ADD COLUMN importance INTEGER;
ALTER TABLE facts ADD COLUMN last_accessed TIMESTAMP;

CREATE INDEX idx_facts_importance ON facts(importance) WHERE importance IS NOT NULL;
-- last_accessed 不建索引（写多读少，索引开销大于收益）
```

### 新增 fact type

```python
class FactType(str, Enum):
    EVENT = "event"
    PATTERN = "pattern"
    OPINION = "opinion"
    PLAN = "plan"               # 新
    EMOTION_NOTE = "emotion_note"
```

`plan` 类型 fact 的约定：
- `subject` 是 plan 的拥有者（aria / npc:x）
- `content` 是一句话描述（"跑光变曲线第三组数据"）
- `tags` 包含 `time_window:09:00-12:00`（必含）和可选的 `goal:<long_term_goal>`
- `ts` 是 plan **生成时刻**（不是计划时刻）
- `expires_at` 是当天 24:00（plan 不跨日）
- `importance` 由 planner 直接打（不走 scorer，规避循环）：6-8

### Source 默认分（scorer 失败时回退）

```python
DEFAULT_IMPORTANCE = {
    Source.USER_STATED: 7,
    Source.BIOGRAPHY: 8,
    Source.LIFE_SIMULATED: 3,
    Source.NPC_TICKER: 4,
    Source.LLM_INFERRED: 5,
    Source.WORLD_FETCH: 3,
}
```

## 组件设计

### Phase B：Importance Scorer + 3D Retrieval

#### `facts/scorer.py`

```python
class ImportanceScorer:
    """Batched LLM scoring. Writers call score_one(fact) which returns
    immediately with a Future; the scorer flushes the buffer when it
    hits 5 facts or 30s, makes a single Sonnet call scoring all of them,
    resolves the Futures.
    """
    def __init__(self, llm: LLMProvider, batch_size: int = 5, flush_seconds: float = 30):
        ...

    async def score_one(self, fact: Fact) -> int:
        """Returns 1-10. Blocks until the batch this fact is in is scored.
        On LLM failure, returns DEFAULT_IMPORTANCE[fact.source].
        """
        ...

    async def _flush(self) -> None:
        """Send one Sonnet call with N facts, parse JSON array of {id, score, reason},
        write reason into each fact's tags as "importance_reason:..." before final write.
        """
        ...
```

**Prompt 形状**（写入 scorer.py 中）：

```
给每条 fact 评 importance（1-10）。
1 = 完全琐碎（"喝了口水"），10 = 改变人物关系或人生方向的事（"决定换工作"、"和恋人分手"）。
参考：
  - 重复的日常作息 → 1-3
  - 普通工作进展 → 3-5
  - 用户表达情绪/分享私事 → 6-8
  - 关键关系变化 / 重大决定 / 创伤性事件 → 8-10

输入 N 条 fact，输出 JSON array：[{"id": "...", "score": 1-10, "reason": "一句话"}, ...]

Facts:
[1] subject=aria, source=life_simulated, type=event, content="..."
[2] subject=user:oc_xxx, source=user_stated, type=event, content="..."
...
```

#### `facts/writers/base.py` 改动

```python
class WriterBase:
    def __init__(self, store: FactStore, scorer: ImportanceScorer, trigger: ReflectionTrigger):
        ...

    async def write(self, fact: Fact) -> None:
        self._enforce_subject_pattern(fact)
        if fact.importance is None:
            fact.importance = await self.scorer.score_one(fact)
        await self.store.append(fact)
        await self.trigger.observe(fact.importance)
```

#### `facts/retriever.py` 改写

```python
async def fetch(self, query: FactQuery) -> list[Fact]:
    candidates = await self._store.query(
        subject=query.subject,
        type=query.type,
        since=query.since,
        limit=query.limit * 8,  # 取更多候选，打分后截
    )
    if query.semantic:
        # FTS rank score per candidate
        fts_ranks = await self._store.fts_rank(query.semantic, [c.id for c in candidates])
    else:
        fts_ranks = {c.id: 0.0 for c in candidates}

    now = datetime.now()
    scored = []
    for fact in candidates:
        hours_old = (now - fact.ts).total_seconds() / 3600
        recency = math.exp(-0.01 * hours_old)  # 论文用 0.99^hours ≈ exp(-0.01·hours)
        importance = (fact.importance or 5) / 10.0
        relevance = fts_ranks.get(fact.id, 0.0)
        score = 0.5 * recency + 0.3 * importance + 0.2 * relevance
        scored.append((score, fact))

    scored.sort(key=lambda x: -x[0])
    top = [f for _, f in scored[: query.limit]]
    await self._store.update_last_accessed([f.id for f in top], now)
    return top
```

注意：`fts_rank` 需要 store 新增方法返回 `{fact_id: rank}` dict（用 FTS5 的 `rank` 列归一到 0-1）。

### Phase C：Reflection 升级

#### `facts/reflection_trigger.py`

```python
class ReflectionTrigger:
    """Watches importance accumulation + elapsed time.
    Calls trigger.fire(reflector) when accumulator ≥ 150 OR 2h elapsed.
    Reset accumulator + timer after each fire.
    """
    THRESHOLD = 150
    MAX_INTERVAL_SECONDS = 7200  # 2h

    def __init__(self, reflector: Reflector):
        self._accum = 0
        self._last_fire = time.monotonic()
        self._lock = asyncio.Lock()

    async def observe(self, importance: int) -> None:
        async with self._lock:
            self._accum += importance
            elapsed = time.monotonic() - self._last_fire
            if self._accum >= self.THRESHOLD or elapsed >= self.MAX_INTERVAL_SECONDS:
                self._accum = 0
                self._last_fire = time.monotonic()
                asyncio.create_task(self._reflector.reflect())
```

#### `facts/reflector.py` 算法

```python
class Reflector:
    async def reflect(self) -> None:
        # 1. 拉最近 100 facts（任意 subject，按 ts 排序）
        recent = await self.store.query(subject=None, limit=100)
        if len(recent) < 10:
            return  # not enough material

        # 2. Sonnet: 生成 3-5 个高阶问题
        questions = await self._ask_questions(recent)

        # 3+4. 每问 → fetch 相关 facts → Sonnet 回答
        for q in questions:
            relevant = await self.retriever.fetch(
                FactQuery(subject="aria", semantic=q, limit=15)
            )
            insight = await self._answer(q, relevant)
            # 5. 写入 pattern
            pattern = Fact(
                subject="aria",
                content=insight,
                source=Source.LLM_INFERRED,
                type=FactType.PATTERN,
                ts=datetime.now(),
                importance=8,  # reflection 产物默认高
                tags=[f"reflection_question:{q[:50]}"],
                supersedes=await self._find_superseded(q),
            )
            await self.inference_writer.write(pattern)
```

**反思 prompt 1（生成问题）**：

```
你看 Aria 最近 100 条 facts，生成 3-5 个**值得思考**的高阶问题。
不要琐碎，例如不要问"她今天吃了什么"。
要问能反映**模式 / 情绪走向 / 关系变化 / 自我认知**的问题。
例如：
  - "她最近在工作上的投入度有变化吗？"
  - "她和 X 的互动模式是不是变了？"
  - "她最近反复在意的事是什么？"

输出 JSON array of strings。
```

**反思 prompt 2（回答）**：

```
基于以下事实回答问题。回答应该是**一条洞见**——浓缩、有信息量。
不能是事实复述（"她最近忙工作"是废话），要能补足事实之间的关系或趋势。
长度 1-2 句。
问题：{q}
事实：
{facts as bullet list}
```

### Phase D：Planner

#### `planner/daily_planner.py`

```python
class DailyPlanner:
    """Triggered 7am每天一次. Generates Aria's plan and each NPC's plan."""

    async def plan_aria(self) -> list[Fact]:
        # 输入：昨天的 reflections + biography + 近一周的高 importance patterns
        yesterday_reflections = await self.retriever.fetch(
            FactQuery(subject="aria", type=FactType.PATTERN, since=yesterday_start, limit=5)
        )
        biography = await self._load_biography_summary()
        recent_patterns = await self.retriever.fetch(
            FactQuery(subject="aria", type=FactType.PATTERN, since=week_ago, limit=10)
        )

        prompt = self._build_aria_plan_prompt(yesterday_reflections, biography, recent_patterns)
        response = await self.llm.complete(messages=[{"role": "user", "content": prompt}], ...)
        items = self._parse_plan(response.content)  # list of {time_window, content, goal?}

        plan_facts = []
        for item in items:
            f = Fact(
                subject="aria",
                content=item["content"],
                source=Source.LIFE_SIMULATED,
                type=FactType.PLAN,
                ts=datetime.now(),
                importance=7,
                expires_at=today_end,
                tags=[f"time_window:{item['time_window']}"]
                     + ([f"goal:{item['goal']}"] if item.get("goal") else []),
            )
            await self.life_writer.write_skip_scorer(f)  # 见下
            plan_facts.append(f)
        return plan_facts

    async def plan_npc(self, npc_id: str) -> list[Fact]:
        # 类似但更粗，2-3 条；输入只有 NPC 自身的 biography + 最近 events
        ...
```

注意：planner 自己生成的 plan facts **跳过 scorer**（importance 直接定 7），需要 WriterBase 暴露 `write_skip_scorer(fact)` 路径，否则会触发"已有 importance 但 scorer 又评一次"的浪费。

**Aria plan prompt 形状**：

```
你为 Aria 生成今天的 daily plan。

【她的身份】
{biography summary, 100 字内}

【昨天的反思】
{yesterday reflections as bullets}

【最近一周的模式】
{recent patterns}

【约束】
- 6-10 条 plan items
- 覆盖工作时间（9-12, 14-18）+ 晚间 + 早晚习惯
- hour 粒度，time_window 形如 "09:00-12:00"
- content 是**具体**的事（"跑光变曲线第三组分析"而不是"工作"）
- 至少 2 条带 goal tag，链接到长期目标

输出 JSON：
[{"time_window": "07:00-08:00", "content": "...", "goal": "..."}, ...]
```

#### `planner/executor.py`

```python
class PlanExecutor:
    """Replaces the random part of the old simulator.
    Tick every 30min: find current plan step, generate a concrete moment, write as event.
    """
    async def tick(self) -> None:
        now = datetime.now()

        # 1. 检查是否需要 replan
        if self._replan_requested:
            await self.planner.plan_aria_partial(from_hour=now.hour)
            self._replan_requested = False

        # 2. 找当前小时落在哪条 plan
        current_plan = await self._find_current_plan(now)
        if not current_plan:
            return  # no plan covers this hour, skip

        # 3. 让 LLM 生成贴着 plan 的具体瞬间
        recent_aria_events = await self.retriever.fetch(
            FactQuery(subject="aria", type=FactType.EVENT, limit=3,
                      since=now - timedelta(hours=2))
        )
        prompt = self._build_moment_prompt(current_plan, recent_aria_events)
        response = await self.llm.complete(...)
        moment_content = response.content.strip()

        # 4. 写入 event
        event = Fact(
            subject="aria",
            content=moment_content,
            source=Source.LIFE_SIMULATED,
            type=FactType.EVENT,
            ts=now,
        )
        await self.life_writer.write(event)  # 走 scorer 评 importance

    def request_replan(self) -> None:
        self._replan_requested = True
```

**Moment prompt**：

```
Aria 当前的 plan：{current_plan.content}（{time_window}）
她最近 2 小时发生过：
{recent events}

生成一条**具体的当前瞬间**——她**正在**做这件事中的某个具体片段。
不是抽象描述，要有细节（数据/物件/感受任一）。
1-2 句，第三人称叙述。
```

### Phase E：NPC 平权 + 双向 interaction

- `daily_planner.plan_npc(npc_id)` 在每天 7am 跑一次，为 6 个 NPC 各生成 2-3 条粗 plan facts（subject 形如 `npc:xiaomin`）
- 现有 `social/scheduler.py` 触发 NPC 互动时，调用新的 `bidirectional_interaction(npc_id, scenario)`：
  ```python
  async def bidirectional_interaction(npc_id: str, scenario: str) -> None:
      prompt = """{两人当前 plan} {recent shared history}
                  生成一次互动。
                  输出 JSON: {
                    "aria_view": "Aria 视角下这次互动是什么样的",
                    "npc_view": "NPC 视角下这次互动是什么样的"
                  }"""
      result = await llm.complete(...)
      data = json.loads(result.content)
      await aria_event_writer.write(Fact(subject="aria", content=data["aria_view"], ...))
      await npc_writer.write(Fact(subject=f"npc:{npc_id}", content=data["npc_view"], ...))
  ```
- 互动事件 importance 走 scorer，预期通常 5-7

## Orchestrator plan_conflict 检测

`OrchestrationDecision` 加字段：

```python
@dataclass
class OrchestrationDecision:
    ...existing...
    plan_conflict: bool = False
```

Orchestrator prompt 加段：

```
6. **plan_conflict**：用户输入是否暗示当前 plan 需要调整？
   - 用户邀约（"晚上一起吃饭吧"）覆盖当前 plan 时段 → true
   - 用户提到 Aria 正在做某事 → false（这是 plan 在执行）
   - 用户问无关问题 → false
   - 仅当冲突明显时才标 true
```

后端：`engine._prepare_turn_v2` 在拿到 decision 后，如果 `plan_conflict=true`，调 `plan_executor.request_replan()`。下一次 executor tick 时执行重算。

## Data Flow 示例

**场景**：用户晚上 8 点跟 Aria 聊天，提到"明天我请假了，要不一起去博物馆？"

1. Orchestrator 看 user input → 输出 decision，含 `topic_anchor: "用户邀约明天去博物馆"`，`plan_conflict: true`（明天会被打断）。Engine 调 `plan_executor.request_replan()`。
2. Renderer 用 decision 拉 facts 渲 prompt。Retriever 给 user.event 桶时按 3D 打分排，"用户上周说过他喜欢印象派"这种高 importance 远旧事件因为 importance 高，盖过最近琐事被拉进来。
3. Aria 回复（带上"印象派"的呼应）。
4. 整个对话过程中 writer 写 facts：用户 statement、Aria 的 inference（"用户最近压力大才想请假"）—— 都走 scorer 评分。inference 评到 7，触发 reflection accumulator + 7 = 接近阈值。
5. 第二天 7am planner tick：plan_aria 看到昨天 reflection（来自上次 trigger 累积达 150 触发的）+ replan flag → 生成今日 plan，把"上午博物馆"作为 9:00-12:00 的 plan item。
6. 上午 9 点 executor tick：current_plan = "陪用户去博物馆"，生成"刚到博物馆门口，外面阳光很好"作为 event。

## 错误处理

| 失败 | 行为 |
|---|---|
| Scorer LLM 调用失败 | 用 DEFAULT_IMPORTANCE[source] 回退；日志 warn |
| Scorer 解析 JSON 失败 | 整批回退默认分；日志 error |
| Reflector 失败 | 整次反思跳过；下次累积/计时重新算；日志 error；不影响 write 路径 |
| Planner morning tick 失败 | 当天用昨天的 plan（如有）；如无，executor 空 tick（不写 event）；日志 error |
| Plan executor LLM 失败 | 跳过当次 tick；下次重试 |
| Bidirectional interaction 失败 | 只写 Aria 单边（fallback 到旧 single-write）；日志 warn |

**共同原则**：任何新机制失败都不能阻塞用户对话路径。chat 永远要 fallback 到能用的状态。

## Testing

### 单元测试

- `tests/facts/test_scorer.py`：mock LLM，验证 batch flush（5 条 / 30s）、JSON 解析、失败回退、并发 score_one 都拿到正确分
- `tests/facts/test_retriever_3d.py`：构造已知 importance/ts 的 fixture facts，验证排序符合公式
- `tests/facts/test_reflection_trigger.py`：模拟 importance observe，验证阈值/时间双触发，验证 fire 后 reset
- `tests/facts/test_reflector.py`：mock LLM 输出固定问题/答案，验证 pattern 写入 + supersedes 正确
- `tests/planner/test_daily_planner.py`：mock LLM，验证 plan facts 数量、time_window 格式、importance=7、expires_at=今日 24:00
- `tests/planner/test_executor.py`：构造 plan facts + 当前时间，验证 _find_current_plan 命中正确 plan；mock LLM 验证 event 写入

### 集成测试

- `tests/integration/test_full_day_cycle.py`：
  - 触发 morning planner → 验证 ≥6 条 plan facts 写入
  - 模拟 8 次 executor tick → 验证 ≥6 条 event facts 写入且贴合各自 plan
  - 模拟 5 条高 importance event 累积 → 验证 reflection 触发 → 验证 pattern 写入
  - 模拟用户 input with plan_conflict=true → 模拟下次 executor tick → 验证 replan 调用

### 手工验证（P6 类型）

- 跑一整天 Aria，晚上看：
  - daily plan 是否合理（6-10 条、覆盖工作时段、有 goal 链接）
  - executor 生成的 events 是否贴着 plan
  - reflection 产出的 patterns 是否真的"有洞见"而非废话
  - retriever 拉出的 facts 是否更"重要"（人工抽查 5 个 turn 的 prompt dump）
  - NPC 互动是否双边一致

## Phasing

P7 cleanup **先于**本 spec 实施。下面是本 spec 的 phase。

| Phase | 范围 | 验收 |
|---|---|---|
| **B** | importance 列 + scorer + 3D retriever | 单测全过；手工跑 10 个 write 看 importance 分布合理 |
| **C** | reflection_trigger + reflector 升级 | 单测过；模拟 5 条高 importance event 后看到 pattern 写入 |
| **D** | daily_planner + plan executor + orchestrator plan_conflict | 跑一天看 plan 质量 + replan 触发 |
| **E** | NPC plan + 双向 interaction | 验证 NPC 也有 plan、互动事件双边写入 |

Phase 之间是 strict 顺序：B 完成才能 C（reflector 要用 retriever），C 才能 D（planner 要用 reflector 产出），D 才能 E（NPC plan 共用 planner 基础）。

## 风险

1. **Plan 的"假感"放大**：现在 simulator 随机抽活动，假感被随机性掩盖。一旦显式做 plan，Aria 在 prompt 里"今天要做 X、Y、Z"如果不真实，假感更刺眼。
   - **缓解**：planner prompt 强约束"基于 biography 和近期 patterns"，不让 LLM 凭空编。手工验证阶段重点看 plan 的真实感。

2. **Scorer 成本**：batched 后 ~20-30 次 sonnet/天可控，但失败回退路径要稳，不能让 scorer 慢拖累 write。
   - **缓解**：scorer 调用走 asyncio Future，writer 异步等待；如果 scorer 超时（如 60s），写入仍继续，importance 用默认值。

3. **Reflection 阈值调参**：150 是论文数字，对 Aria 不一定合适。如果阈值过低，反思变成废话工厂；过高，反思频率不够。
   - **缓解**：把阈值 + 时间间隔做成配置，跑一周根据 reflection 质量人工调。

4. **NPC plan 让 NPC ticker 失去随机感**：现在 NPC 偶尔冒一句无关消息，做了 plan 后会变成"按部就班"。
   - **缓解**：NPC plan 故意做粗（2-3 条），且不强制 executor 完全贴 plan——让 NPC interaction 触发器有更高随机性。

## Open Questions

无未解决问题。所有决策点（plan 粒度、reflection 触发、NPC 深度）在 brainstorming 中已锁定。
