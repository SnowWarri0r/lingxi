# Aria 社交图谱（Social Graph）设计

**Status**: Draft
**Date**: 2026-05-22
**Owner**: lingxi

## 1. 背景与目标

### 问题
Aria 当前对"身边人"的所有引用（室友/导师/同事/家人/朋友）都是**即兴生成**——
- 不可证伪：每次提到的"室友"细节都不一样
- 不持续：上次说室友赶论文，这次又变成赶项目
- 缺戏剧密度：所有 NPC 听起来都像"一个朋友"

这是 Aria persona 真实感的最大破绽。用户已明确指出："要 Smallville 那种身边人的感觉才行"。

### 目标
建立一个**轻量、自治、单向**的 NPC 模拟层，让 Aria 真正"被身边人围绕"：
- 6 个手写 NPC，每人有持续的人生 arc
- 事件按 cron 自动生成（NPC 的生活独立于 Aria 与用户的对话进展）
- 部分事件升级到 Aria 的 `recent_events` 让她可能主动开口
- 大部分事件作为背景知识渲染进 system prompt

### 非目标
- ❌ Smallville 原版的 NPC 之间互动（涌现式社交）
- ❌ NPC 自主跑 reflection/plan loop
- ❌ 用户对话反向写回 NPC log（保持单向数据流）
- ❌ LLM 自动生成 NPC 名册（第一版手写）

## 2. 已锁定的设计决策

| 决策点 | 选项 | 理由 |
|---|---|---|
| 路线 | A: Aria-centric 社交图 | 价值在 Aria 跟单个 NPC 的关系演化，不在 NPC 互相涌现 |
| 推/拉 | 3: 混合（significance ≥0.6 push） | 日常水走 pull 避免 Aria 变 NPC 八卦广播站，重大事件走 push 让她会主动说 |
| 数据流 | A: 单向（NPC → Aria → 用户） | 防止用户通过引导式提问凭空创造 NPC 历史 |
| 名册 | A: 手写 YAML | 保证戏剧密度，避免 generic 角色 |
| 名册成员 | 去掉 ex，保留 6 人 | 用户决定 |
| cron 节奏 | 白天每 2h，22:00-08:00 停 | 白天 8 个 tick（8/10/12/14/16/18/20/22），夜间 NPC 也"睡觉" |
| arc 机制 | 包含因果链 arc | 事件之间不孤立，arc 会推进（早期→中期→高潮→消解） |

## 3. 架构

```
config/personas/aria/
└── social_graph.yaml          # 6 NPCs，手写

data/social/
├── npcs/{npc_id}/
│   ├── events.jsonl           # append-only 事件流
│   └── arcs.json              # arc state（stage/progress/last_advanced_at）
└── last_tick.json             # cron 防漏跑/防重跑

src/lingxi/social/
├── models.py                  # NPC / NPCEvent / NPCArc dataclasses
├── loader.py                  # 加载 yaml + jsonl + json
├── store.py                   # append event / update arc / read recent
├── event_generator.py         # LLM 生成事件（核心）
├── arc_advancer.py            # 决定 arc 是否推进到下一 stage
├── scheduler.py               # cron daemon（复用 proactive loop infra）
├── renderer.py                # 渲染"身边人近况"块到 prompt（pull）
└── promoter.py                # significance≥0.6 push 到 Aria.recent_events
```

## 4. 数据模型

### 4.1 NPC 定义（`social_graph.yaml`）

```yaml
npcs:
  - id: xiaomin
    name: 小敏
    relation: 室友
    age: 27
    background: |
      同所博士四年级，比 Aria 早一年，做星系演化。
      跟 Aria 同住四年，从二人间合租到现在的两室一厅。
    traits: [话密, 急性子, 容易共情但偶尔越界]
    interaction_style: |
      跟 Aria 像姐妹，会半夜敲门聊天，也会因为洗碗谁洗拌嘴。
    base_event_probability: 0.4   # 每 tick 默认 40% 概率出事件
    initial_arcs:
      - id: thesis_pressure
        summary: 论文 defense 倒计时 8 周，状态崩
        stage: early       # early | mid | climax | resolved
        weight: 0.8        # 当前在 NPC 生活中的占比
```

完整 6 人 yaml 见附录 A。

### 4.2 NPCEvent（`events.jsonl` 一行）

```python
@dataclass
class NPCEvent:
    npc_id: str
    ts: datetime
    type: Literal["life", "aria_interaction"]
    content: str                       # "小敏今早 6 点起来跑数据，发现脚本跑了一夜还是 NaN"
    significance: float                # 0.0-1.0（LLM 评分）
    arc_id: str | None                 # 关联的 arc，None 表示独立小事件
    promoted_to_aria: bool = False     # 是否已 push 到 Aria.recent_events
```

### 4.3 NPCArc（`arcs.json`）

```python
@dataclass
class NPCArc:
    id: str
    npc_id: str
    summary: str
    stage: Literal["early", "mid", "climax", "resolved"]
    weight: float                      # 0-1，当前在 NPC 生活中的占比
    started_at: datetime
    last_advanced_at: datetime         # stage 上次变化的时间
    event_count: int                   # 该 arc 已生成多少事件
    resolution: str | None = None      # resolved 时填，说明结局
```

## 5. 核心流程

### 5.1 Cron tick（每 2h，白天 8/10/12/14/16/18/20/22）

```
for each NPC:
    1. roll dice：是否生成事件？
       p = base_event_probability
       p += 0.2 if hours_since_last_event(npc) > 24
       p += 0.2 if any active arc has weight ≥ 0.7
       p *= 0.3 if hour in [22] (晚 tick 减少)

    2. if hit:
       a. event_generator.generate(npc, recent_events, active_arcs) → 1-2 events
       b. for each event:
          - store.append_event(event)
          - if event.type == "aria_interaction":
              inner_state.add_event(event.content, source="npc_interaction")
          - if event.significance >= 0.6:
              promoter.push_to_aria(event)  # 设 wants_to_share=True

    3. arc_advancer.maybe_advance(npc.arcs)
       - 如果某 arc 的 event_count 达到 stage 阈值（early≥3, mid≥4, climax≥1），
         调 LLM 判断"该推进到下一 stage 吗"
       - resolved 的 arc 30 天后归档（不再渲染）

last_tick.json 更新时间戳防重跑
```

### 5.2 事件生成 prompt（`event_generator.py`）

```
你在帮 {npc.name}（{npc.relation}）生活一天里发生几件小事。

【这个人是谁】
{npc.background}
性格：{npc.traits}

【最近 7 天发生过的事】
{recent_events_rendered}

【当前正在经历的事（arcs）】
{active_arcs_rendered}

【当前时间】{now}（{time_of_day_hint}）

请生成 1-2 个**真的可能在这个时间点发生**的小事：
- 大部分时候是日常水（吃饭/通勤/学习/小情绪），少数时候是 arc 推进
- 不要每个事件都跟 arc 相关
- 大约 1/10 概率是跟 Aria 一起经历的事（标 type=aria_interaction）
- 给每个事件评 significance（0.0-1.0）：
  - 0.1-0.3：纯日常水
  - 0.4-0.5：值得知道但不必转述
  - 0.6-0.8：重大事件（吵架/突破/坏消息）
  - 0.9+：极少（亲人病危/重大成就）

输出严格 JSON：
[
  {"type": "life|aria_interaction", "content": "...", "significance": 0.x, "arc_id": "..." or null}
]
```

### 5.3 Arc 推进（`arc_advancer.py`）

```
触发条件：arc.event_count 达到 stage 阈值（early=3 / mid=4 / climax=1）

LLM prompt：
"""
这个 arc 已经过了 {N} 天，生成了 {M} 个事件：
{events_rendered}

当前 stage: {arc.stage}
当前 summary: {arc.summary}

判断：
1. 该 arc 是否应该推进到下一 stage？(yes/no)
2. 如果 yes，新 stage 的 summary 是什么？
3. 如果是 resolved，结局如何？

输出 JSON: {"advance": bool, "new_summary": "...", "resolution": null|"..."}
"""
```

### 5.4 渲染（`renderer.py`）— 注入 system prompt 的"身边人近况"块

```markdown
【身边的人】

**室友 小敏**（27，同所博士，跟你住四年）
- 性格：话密急性子，容易共情但偶尔越界
- 在经历：论文 defense 倒计时 8 周状态崩（早期）
- 最近：昨晚 12 点还在改 introduction / 今早跟你吐槽导师 1on1 哭了

**导师 赵老师**（52，天文系教授）
- ...

（最多渲染 4 个最相关 NPC：weight 高的 arc / 近 48h 有事件的优先）
```

### 5.5 Push 到 Aria.recent_events（`promoter.py`）

```python
def push_to_aria(event: NPCEvent):
    if event.promoted_to_aria:
        return  # idempotent
    inner_state.add_event(
        content=f"{npc.name}今天{event.content[去掉主语]}",
        significance=event.significance,
        source="npc_event",
        wants_to_share=True,
    )
    event.promoted_to_aria = True
    store.update_event(event)
```

这样 significance ≥0.6 的 NPC 事件就能自然走现有 proactive opener 系统。

## 6. 单向数据流强制

- ✅ `event_generator.py` 唯一能 append 到 `npcs/{id}/events.jsonl` 的地方
- ✅ `relational/extractor.py` 完全不知道 `social/` 这个模块的存在
- ✅ 用户对话里出现的 NPC 信息只进 `RelationalMemory`（用户跟 Aria 的关系记忆），不污染 NPC log
- ✅ Aria 回应时如果说错了 NPC 的事（比如 NPC log 里没记录"小敏跟导师吵架"但 Aria 编了），不会被回写
- ⚠️ 这意味着 Aria 可能偶尔说出 NPC log 里没有的细节，这是接受的 tradeoff——比起污染 source of truth，宁可偶尔不一致

## 7. 集成点

| 现有模块 | 改动 |
|---|---|
| `persona/prompt_builder.py` | `_build_inner_state_section` 后追加 `_build_social_graph_section` 调用 renderer |
| `temporal/inner_state.py` | `add_event` 加 `source` 字段（区分 self / npc_event） |
| `temporal/proactive.py` | 不变，promoted 事件已经在 recent_events 里 |
| `feishu_cli` / `app.py` | 启动时拉起 `social.scheduler` daemon（与 proactive scheduler 并列） |

## 8. 测试

- **unit**: event_generator 解析 LLM 输出 / arc_advancer 判断阈值 / promoter 幂等性
- **integration**: 跑 10 个 tick，断言 NPC events.jsonl 长度增长、arc stage 演化、Aria.recent_events 收到 push
- **soak**: 跑 24h（48 tick）观察事件分布是否合理（不全是危机也不全是水）

## 9. 实施阶段

| Phase | 范围 | 交付 |
|---|---|---|
| **P1: 静态骨架** | models / loader / store / renderer / 6 NPC yaml + 初始 seed events | Aria system prompt 出现"身边人近况"块，全部 hardcoded |
| **P2: 事件生成** | event_generator + scheduler + cron | NPC events.jsonl 真的在增长 |
| **P3: Push 升级** | promoter + significance 阈值 | Aria 偶尔主动开口提 NPC 的事 |
| **P4: Arc 推进** | arc_advancer + stage transitions | 多 tick 后 arc 真的推进，不是死循环 |
| **P5: 调参 & 观察** | soak test + 实际跑 1 周看体感 | 用户反馈调 base_event_probability / cron 节奏 / significance 阈值 |

## 10. 风险与缓解

| 风险 | 缓解 |
|---|---|
| LLM 生成事件 generic（"小敏今天有点累"刷屏） | event_generator prompt 加 few-shot 具体例子；soak 后 review |
| Arc 永远不 resolve 越积越多 | arc_advancer 强制阈值：event_count > 20 必判 resolve |
| Push 太频繁让 Aria 像 NPC 推销员 | significance 0.6 阈值；同一 NPC 24h 内只 push 1 次 |
| Token 成本 | 每 tick 1 次小 LLM call × 6 NPC = 6 calls × 8 tick = 48 calls/day。可控 |
| Aria 提到 NPC log 没有的细节 | 接受，单向数据流的代价 |

## 附录 A：6 NPC 初始名册

完整 yaml 在实施阶段 P1 写出，提议骨架：

| ID | 关系 | 张力点 / 初始 arc |
|---|---|---|
| `xiaomin` | 室友（同所博士四年级） | 论文 defense 倒计时，状态崩 |
| `prof_zhao` | 导师（天文系教授 52 岁） | 给 Aria push 新课题方向，让她犹豫要不要接 |
| `lin_jie` | 师姐（Caltech 博后） | 远程 mentor，最近自己也在申 faculty 焦虑 |
| `echo` | 大学闺蜜（产品经理） | 项目上线压力大，周末约 Aria 聊天救命 |
| `mom` | 妈妈（天津 退休教师） | 姨妈给介绍了相亲对象，妈妈一直旁敲侧击 |
| `tom` | 同实验室博后 | 跟 Aria 工作搭子，最近 paper 投 ApJ 等审稿，焦虑 |

每个 NPC 一开始 1-2 个 active arc，剩下靠 event_generator 后续可能生出新 arc（P4 之后能力）。
