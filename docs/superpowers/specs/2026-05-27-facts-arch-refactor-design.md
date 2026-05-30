# Facts 架构整体重构

**Status**: Draft
**Date**: 2026-05-27
**Owner**: lingxi
**Branch**: `refactor/facts-arch`

## 1. 背景与问题诊断

### 问题不是 bug，是 pattern

过去几周给 Aria 加 feature 都是同一个模式：每个功能（`inner_life` / `social` / `world` /
`biography` / `relational` / `proactive` / `reflection` / `planner` / `fewshot` /
`microbrain` / `pet`）独立加，每个都有：

- **自己的数据 store**（json/jsonl/chroma 各家自治）
- **自己的更新逻辑**（cron tick / LLM extractor / 反思周期）
- **自己的 prompt 渲染块**（写进 system prompt 一段）

### 表面症状（已经反复修过的）

1. **prompt 块爆炸**：chat-time prompt 现在有 11+ 个 section（format_preamble + identity +
   personality + speaking_style + message_habits + decision_axes + relationship_level +
   biography_hits + subjective_view + social_graph + relational_memory + memory_facts +
   agenda + plans + focus_reminder × 5）。模型每轮在这堆 block 间做 attention 分配，AI tell
   和 register 漂移就在这里发生。
2. **同一事实多源矛盾**：用户的工作时间被写在 `relational.daily_patterns` 3 条不同 framing
   ("经常工作到很晚" / "经常加班到 8 点多" / "最近一直在加班") + `memory.long_term_facts` 也有
   + 当下对话有。Aria 不知道听谁的，吐出 "啊那确实不算加班 我忘了你们行业的时间" 这种 AI 软包装。
3. **数据流方向缺约束**：NPC `life` 事件被 promoter 推进了 Aria.recent_events，Aria 把
   Echo 看到便利贴 narrate 成自己看到的。事件层每加一个 feature 就要新写一条防漏 filter。
4. **没有"哪个块该看哪个"的优先级**：模型看到 11 个 section 自己 reconcile，选最显眼的或最近写的。

### 每次"修复"的 pattern

加一条 filter / 加一个 caveat / 加一条 dedup → 局部缓解 → 下次新 feature 又得维护这堆补丁。
**补丁越多，未来加新 feature 的边际成本越高**。

## 2. 已锁定的设计决策

| 决策点 | 选 | 理由 |
|---|---|---|
| 重构层 | 三层都改（数据 + 渲染 + 编排） | 单层修不动根问题——数据混 + 渲染挤 + 没编排互为因果 |
| 迁移策略 | branch-based 重写 | 用户能停几天观察新版；快 30-50% + 老代码可整体删除（无 compat shim 长尾） |
| Orchestrator 模型 | Sonnet（不是 Haiku） | Register 决策影响整轮对话基调，Haiku 在 register 判断上不稳；多花 ~3x 成本换稳定 |
| Fact 后端 | SQLite | unified 查询 + ts/subject 索引 + FTS5 全文搜索一步到位；jsonl 多文件难做跨 subject 查询 |

## 3. 架构 — 三层

```
┌─────────────────────────────────────────────────────────────┐
│  Chat Path (engine.py)                                       │
│                                                              │
│    user_input ──► [1] Orchestrator (Sonnet) ─► decision      │
│                                  │                           │
│                                  ▼                           │
│                       [2] Retriever ─► facts (按 decision)   │
│                                  │                           │
│                                  ▼                           │
│                       [3] Renderer ─► system prompt          │
│                                  │                           │
│                                  ▼                           │
│                       [4] Chat (Sonnet) ─► response          │
└─────────────────────────────────────────────────────────────┘

         ▲                                                  ▲
         │                                                  │
   ┌─────┴──────┐                                    ┌──────┴─────┐
   │  Facts DB  │◄───── Writers ──────┐              │  Writers   │
   │  (SQLite)  │                     │              │            │
   └────────────┘                     │              └────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
   LifeWriter           NPCWriter             InferenceWriter   ...etc
   (life simulator)     (social sched)        (reflection cycle)
```

## 4. 数据层 — Fact 表

### 4.1 Schema

```python
class Source(str, Enum):
    USER_STATED      = "user_stated"       # 用户明说的
    LIFE_SIMULATED   = "life_simulated"    # Aria 自己生活模拟器生的
    NPC_TICKER       = "npc_ticker"        # 社交图谱 NPC cron 生的
    LLM_INFERRED     = "llm_inferred"      # 反思周期推断的
    WORLD_FETCH      = "world_fetch"       # 每日新闻抓的
    BIOGRAPHY        = "biography"         # 一次性导入的人物 backstory

class FactType(str, Enum):
    EVENT            = "event"             # 发生过的事
    PATTERN          = "pattern"           # 反复出现的规律
    OPINION          = "opinion"           # 立场/态度
    PLAN             = "plan"              # 待办/计划
    EMOTION_NOTE     = "emotion_note"      # 情绪标记

@dataclass
class Fact:
    id: str                          # uuid
    subject: str                     # "aria" | "user:oc_xxx" | "npc:xiaomin" | "world"
    content: str                     # 自然语言一句
    source: Source
    type: FactType
    ts: datetime                     # 事实发生 / 被观察的时间
    written_at: datetime             # 写入时间
    confidence: float                # 0-1，按 source 默认（USER_STATED=1.0 / LLM_INFERRED=0.5）
    expires_at: datetime | None
    tags: list[str]                  # 主题 / 类别（FTS 用）
    supersedes: str | None           # 旧 fact id，被本 fact 修正/替代
```

### 4.2 后端

- SQLite，单文件 `data/facts.db`
- 索引：`(subject, ts DESC)` 主查询路径、`(source)`、`(type)`、`(expires_at)` 清理用
- FTS5 虚表 `facts_fts(content, tags)` 用于语义/关键词回退
- WAL 模式（多进程读 + 单写）

### 4.3 替代的旧 store

完全替代：
- `inner_life/store.py` 的 recent_events + diary（→ subject=aria, type=EVENT/EMOTION_NOTE）
- `relational/store.py` 整套（→ subject=user:xxx, 各 type 对应）
- `social/store.py` 的 npc events（→ subject=npc:xxx, source=NPC_TICKER）
- `world/store.py` 的 daily briefing（→ subject=world, source=WORLD_FETCH）
- `memory/store.py` 的 long_term_facts（→ subject=aria 或 user:xxx）

保留（外接，不进 Fact 表）：
- chroma episode embeddings（语义检索用，与 Fact 表互补：Fact 是结构化事实，chroma 是 raw 对话片段）
- `inner_life/store.py` 的 `current_activity` / `today_plan`（不是 Fact，是当下 state，单独小 store）

### 4.4 Writer 严格分工

| Writer | subject | source | 写入触发 |
|---|---|---|---|
| `LifeWriter` | `aria` | LIFE_SIMULATED | life simulator tick |
| `NPCWriter` | `npc:xxx` | NPC_TICKER | social scheduler tick |
| `UserStatementWriter` | `user:xxx` | USER_STATED | chat path 捕获用户明说的事实 |
| `InferenceWriter` | `user:xxx` / `aria` | LLM_INFERRED | reflection 周期 |
| `WorldWriter` | `world` | WORLD_FETCH | daily news scheduler |
| `BiographyLoader` | `aria` | BIOGRAPHY | boot 一次（ts=过去某时间）|

**核心不变量**：subject 隔离。NPC 事件永远 `subject=npc:xxx`，**结构上不可能**混进 `subject=aria` 的事件流。当前"Echo 看到便利贴 → Aria narrate 成自己看到"在新架构下不可能发生。

## 5. Brain 层

### 5.1 Orchestrator（pre-turn Sonnet 调用）

**输入**（≤500 tokens）：

```
user_input: <用户最新消息，≤200 字>

aria_now:
  activity: <一句话当前活动>
  mood: <一句话当前心情>
  last_lived_facts: <最近 2 条 aria.lived 事件，各一句>

available_facts:
  aria.lived.recent (8) | aria.emotion (3)
  user:oc_xxx.patterns (12) | user:oc_xxx.shared_jokes (4)
  npc:xiaomin.recent (5) | npc:mom.recent (3)
  world.today (2)
  aria.biography (205, semantic-searchable)
```

**输出**（严格 JSON，≤300 tokens）：

```json
{
  "engage_level": 0.0-1.0,
  "register": "warm|curt|curious|withdrawn|flustered",
  "fact_queries": [
    {"category": "user:oc_xxx.patterns", "limit": 3, "filter": "工作时间"},
    {"category": "aria.lived.recent", "limit": 2},
    {"category": "aria.biography", "limit": 1, "semantic": "工作时间或加班"}
  ],
  "topic_anchor": "<一句话总结对方话题落点>",
  "skip": ["world.today", "npc.recent"]
}
```

**模型**：Sonnet 4。延迟 ~1-2s（与 chat call 串行）。成本 ~0.5k in + 0.2k out per turn。

**失败降级**：JSON 解析失败 / 调用错误 → fallback `OrchestrationDecision.default(register="warm", engage_level=0.6, queries=[默认 set])`。

### 5.2 Retriever

```python
class FactRetriever:
    async def fetch(query: FactQuery) -> list[Fact]:
        # category: "subject.bucket" e.g. "user:oc_xxx.patterns"
        # 拆成 subject + type 过滤
        # filter（关键词）+ semantic（FTS5 / embedding）+ limit 应用
        # 按 (confidence * recency_decay) 排序
        # 取 top N
```

按 `supersedes` 链解析：被修正过的 fact 不返回，只返回最新版本。

### 5.3 Renderer — 3 个 dynamic block（替代当前 11+）

**静态部分**（cache 命中）：
- `# 你是 Aria`（identity + personality + speaking_style + message_habits） — 与现版相同的内容，从当前 PromptBuilder 抽出来

**Dynamic 3 块**（每轮按 orchestrator 决策渲染）：

1. **「你此刻」**
   - 当前 activity
   - mood + emotion + engagement register（按 orchestrator.register）
   - 1-3 条 `aria.lived.recent` facts
   - 显式标 `register=<warm|curt|...>` 影响输出基调

2. **「你和他」**
   - `user:xxx.patterns` 选 facts（按 orchestrator.fact_queries）
   - `user:xxx.shared_jokes` / `fight_patterns` 选 facts
   - 关系等级 + 称呼习惯

3. **「相关记忆 / 身边的事」**
   - `aria.biography` 语义检索结果
   - `npc:xxx.recent` （只有 orchestrator 显式 query 时才出）
   - `world.today` （只有显式 query 时才出）

**关键约束**：
- 渲染按 **subject 严格隔离**。subject=`npc:xxx` 的 fact 永远渲染到"身边的事"块；subject=`aria` 的永远进"你此刻"块；subject=`user:xxx` 的永远进"你和他"块。**结构上不可能交叉**。
- 同一事实多源时（subject 相同、content 相似），按 **confidence × source priority** 取最新。`USER_STATED > LIFE_SIMULATED > LLM_INFERRED`。被 supersedes 链替代的不出。
- 总 dynamic 长度上限 2000 tokens；超过则按 orchestrator skip 优先级裁剪。

### 5.4 Fewshot

继续走现有 `FewShotRetriever`，作为 message-level few-shot 喂给 chat call（不进 system prompt）。与 Fact 系统**完全解耦**——fewshot 是 voice anchor，Fact 是事实。

## 6. 迁移计划（branch-based）

**Branch**: `refactor/facts-arch`。本地开发，feishu 暂停日用 2-3 天测新版本，稳定后 main merge + 旧 store 文件 + 旧 prompt section 整体删。

| Phase | 内容 | 工时 | 验证 |
|---|---|---|---|
| P0 | `facts/{models,store,retriever}.py` + SQLite schema + 单元测试 | 1d | unit tests, write/read 1000 facts |
| P1 | 6 个 Writer + `tools/migrate_to_facts.py` 一次性迁移现有数据 | 1.5d | migration 灰度跑（不删旧），spot-check 几个 user 的 fact 数量 |
| P2 | `brain/orchestrator.py` + JSON schema + fallback 降级 | 1.5d | unit tests + 真实 Sonnet 5 case 烟测 |
| P3 | `brain/renderer.py` 3 个 dynamic block | 2d | snapshot 比对 + 长度上限测试 |
| P4 | 切换 `engine._prepare_turn` 走新链路；老 `build_system_prompt` 标 deprecated | 1d | engine 测试更新 + 真实 5 轮对话 |
| P5 | 旧背景任务（life simulator / social scheduler / reflection / world）改写到新 writer | 1d | 24h 跑通无错 |
| P6 | 端到端验证 | 1d | inspect_llm 看 prompt 结构 + register / 矛盾事实人工检查 |
| P7 | 删除旧 store / 旧 PromptBuilder section / migration 脚本 | 0.5d | grep 确认无引用 |

**总计 ~8.5 工作日。** P0-P4 跑通约 5 天后可切日用，P5-P7 再 ~3 天 cleanup。

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| Orchestrator 失败降级时 fallback 决策选错 facts → 模型答跑偏 | fallback 选保守集（user.patterns + aria.lived 各 3 条），不会比当前更糟 |
| Sonnet orchestrator 每轮 +1-2s 延迟 | 接受。当前 chat 一轮 ~5-8s，新增 20-30% 可接受；后期可考虑 Sonnet → Haiku 替换如 orchestration prompt 足够结构化 |
| Migration 漏数据 | migration 脚本 dry-run 模式先跑，对比新旧 fact 数量 + 抽样人工 review |
| 删除旧 store 后发现遗漏 → rollback 难 | P5 完成时打 tag `pre-cleanup-v0`，P7 删除前再打 `post-cleanup-v0`，rollback 一行 git reset |
| FTS5 中文分词差 | 用 trigram tokenizer 或 jieba 预切；初版用简单 LIKE 也够（fact 总数 < 10k） |
| supersedes 链遍历慢 | 索引 `supersedes` 字段；查询时一次 LEFT JOIN |

## 8. 非目标

- ❌ 不重做 chat path 本身（仍是 streaming Sonnet 调用，输出格式不变）
- ❌ 不动 fewshot 子系统（独立模块，已经够稳）
- ❌ 不动 pet UI / microbrain（独立子系统，不在 chat 路径上）
- ❌ 不引入新外部依赖（SQLite/FTS5 是 Python 标准库）

## 9. 成功标准

P6 端到端验证时，对 `chat_stream_split` inspector output 进行 5 项检查：

1. **Dynamic section 数 ≤ 4**（当前 11+）
2. **Subject 严格隔离**：grep "Echo 走到" / "小敏在实验室" 在 `aria` 上下文里**应该不出现**
3. **无矛盾事实**：同一 fact 多个 framing 不再共存（supersedes 链生效）
4. **总 prompt 长度降至当前 70% 以内**（当前 ~15k tokens → 目标 ≤ 10.5k）
5. **AI tell 头部正常对话场景显著减少**（人工评估 20 轮对话）
