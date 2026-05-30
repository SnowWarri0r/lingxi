# Facts 唯一真相源 · 旧栈退役设计

> 2026-05-29 · brainstorm 产出 · 前置审计见 [2026-05-29-decommission-audit.md](./2026-05-29-decommission-audit.md)

**Goal:** 把"Facts 整体重构"没做完的下半场补完——让 `facts.db` 成为唯一真相源,拔掉与之并行打架的四套旧栈(memory/Chroma、inner_life、planning、temporal/reflection)+ 一处真死代码(extensions/),消除人格碎片化。

**Architecture:** chat-time 上下文只从 facts(经 Orchestrator→Renderer)来;emotion/engagement 留在 engine;"她的生活"完全由 `aria.event`/`aria.PLAN` facts 承载。不保留旧数据、不做迁移/双写灰度/AB(本项目无上线约束,大胆 breaking change)。

**Tech Stack:** Python async, SQLite+FTS5(facts.db,已是活写入目标), pydantic, pytest。

---

## 1. 背景:为什么要退役

59-commit 的重构**只建新没拆旧**,新栈(facts/brain/planner/reflector)与旧栈并行运行,且在**同一个 prompt 里双喂**:

- `engine._prepare_turn_v2`(线上唯一路径)同时 `memory.assemble_context()`(旧 Chroma 拼 message)+ `render_dynamic_blocks()`(新 facts 拼块)→ 两套各说各话的记忆同场 = 人格碎片化根因。
- `LifeSimulator`(写 inner_state JSON)与 `PlanExecutor`(写 aria facts)两套生活引擎并行 → "她今天"由两个系统拼出。
- 旧 `_prepare_turn` 是死代码(fact_retriever 全生产路径恒非空,else 分支不可达)。

## 2. 终态架构

```
   后台循环 ──▶ facts.db (唯一真相源)
   DailyPlanner    aria.event / aria.PLAN / aria.PATTERN
   PlanExecutor    user:*.* / npc:*.* / world.*
   Reflector            │
                  FactRetriever (3D 打分)
                        │
 chat-time: Orchestrator(catalog+对话→register/queries) → render_dynamic_blocks(facts→3块) ┐
                                                                                          ├▶ prompt
 engine 自带: _emotion_state / _current_mood → EngagementMode(纯 emotion) ──────────────────┘
                        │
                ShareIntentStore → proactive(何时主动)
```

**唯一真相源不再双喂:** 删 `engine._prepare_turn_v2` 里的 `memory.assemble_context` + `context_assembler.assemble_messages(memory_context)`;chat-time 上下文只来自 `render_dynamic_blocks`。

## 3. 数据流

**被动回复:** 收消息 → `_prepare_turn_v2`(唯一路径) → `FactRetriever.catalog()` → Orchestrator 决策 register/fact_queries/topic_anchor/skip → `render_dynamic_blocks` 出【你此刻】【你和他】【身边的事】 → prompt_builder 拼静态 persona 段 + EngagementMode(读 engine emotion)行为段 → LLM 出 response → 更新 engine emotion + 该轮事实经 writers 落 facts。

**主动消息:** proactive 循环按 scheduler 节奏 tick → `find_pending_share(ShareIntentStore)` → 有则取对应 fact、渲染同样 facts 块、生成开场白、发、标记 consumed;无则不发。

**后台生活:** DailyPlanner 7am 写当天 `aria.PLAN` facts;PlanExecutor 每 30min 看当前 plan→生成 first-person "此刻"→写 `aria.event` fact;Reflector(importance≥150 或 2h)→写 `aria.PATTERN` facts。

## 4. inner_life 五件事的归宿(全部现成或搬家)

| inner_life 的东西 | 归宿 | 备注 |
|---|---|---|
| current_activity / recent_events / diary | `aria.event` facts(PlanExecutor 已写) | renderer `你最近的事` 已渲染 |
| DailyPlan | `aria.PLAN` facts(DailyPlanner 已写) | FactType.PLAN 已存在 |
| EngagementMode / derive_engagement_mode | **搬到 `persona/engagement.py`**,纯 emotion 驱动 | 只丢 energy<0.3→curt 次分支 |
| EmotionState / _current_mood | 留在 engine 不动 | 杠杆真源,独立于旧栈 |
| SubjectiveView(对人印象) | `user:*` inference facts | 【你和他】块已渲染 |
| AgendaItem / wants_to_share | ShareIntentStore | 本 session 已建 |
| energy / social_need / axis_modulation 数值态 | **丢弃** | 用户已确认 |

## 5. 几个 chat-time 消费者的改写

- **engine.py**:`_prepare_turn`(228-475)整删;三处 `fact_retriever is None` else 分支删、直调 `_prepare_turn_v2`;`_prepare_turn_v2` 内删 `memory.assemble_context`/`assemble_messages`;删 inner_state 加载(352-357)与传参(448/496/596-610);`add_fact`(1387)/`consolidate_session`(1447)改为经 facts writers(InferenceWriter/LifeWriter),不再写 Chroma。
- **prompt_builder.py**:删 `inner_state` 参数、`_build_decision_axes_section`、`active_plans`/`_build_plan_section`、`_build_memory_section`、`MemoryContext`/`Plan` import;EngagementMode 行为段改 import 自 `persona/engagement.py`。
- **proactive.py**:删 inner_state 加载(670)与传参;主动消息渲染走 facts 块。
- **pet/state_endpoint.py**:engagement←engine emotion(经 persona/engagement);当下活动←最新 `aria.event` fact(FactRetriever);mood←`engine._current_mood`。
- **temporal/relationship.py**:entity_graph 查询改用 FactRetriever,或随 memory 一并退(若无其他活引用则整退)。

## 6. 退役顺序(叶子→根:先改消费者,再删生产者)

> breaking-change + 不迁移旧数据,让审计文档里那串 BLOCKER(ConsolidationBridge/EpisodeDecomposer/数据迁移/双写灰度/AB)**全部蒸发**——旧 Chroma 数据直接弃,facts.db 已是活写入目标。

1. **改 chat-time 消费者脱离 inner_state**:engine prompt 组装、prompt_builder、proactive、pet 改成读 facts/emotion;新建 `persona/engagement.py` 收 EngagementMode/derive_engagement_mode(纯 emotion)。
2. **改 _prepare_turn_v2 脱离 Chroma**:删 `memory.assemble_context`/`assemble_messages`,上下文只走 `render_dynamic_blocks`;`add_fact`/`consolidate_session` 改走 facts writers。
3. **拔生产者**(此时无人再读):停 feishu 的 `LifeSimulator` + `ReflectionLoop` 启动(按语义边界删,注意两块行号相邻);删目录 `inner_life/`(simulator/store/models/agenda/subjective)、`memory/`(Chroma 整栈,保留 `short_term.ConversationTurn` 纯数据结构搬到合适位置)、`planning/`、文件 `temporal/reflection.py`、目录 `extensions/`;删 engine 旧 `_prepare_turn` + Planner/Scheduler/ActionExecutor 构造。
4. **收尾**:social `SocialScheduler` 在 `npcs` 为空时不启动(省空转);全仓 grep 残留 import 清零;pyproject/__init__ 清理。

## 7. 接受的能力降级(用户已确认)

1. Reflector 只写全局 `aria.PATTERN`,丢 reflection 的 per-recipient 作用域 + 6h per-recipient cooldown。
2. biography 不再运行时自动增长(保留静态 seed + 手动 `add_biography_event`)。
3. 丢弃 energy/social_need/axis_modulation 数值态及其派生的决策轴短时偏移。

## 8. Out of scope / follow-ups

- **不删** `social/`(用户主动留的休眠 NPC 系统)、`web/`(lingxi-server 备用前端)、`microbrain/`(WIP #116)。
- **两个 engagement 决策器合并**(orchestrator `register` vs `derive_engagement_mode`)记为 follow-up,本项目只搬不合。
- **biography 运行时自动增长**若以后想要,让 Reflector 调 `add_biography_event` 即可。

## 9. 验证(无上线约束,靠日用观察 + 测试)

- 每改一步:`grep` 确认目标符号/import 残留清零;应用启动无 ImportError、无 Chroma 连接。
- 三种 chat 路径(full/stream/stream_events)各跑一次走 `_prepare_turn_v2` 正常。
- 跟 Aria 聊几轮:确认不再出现新旧记忆拼接的矛盾(职业/今天做了啥),情绪杠杆(curt/withdrawn/flustered)仍生效。
- pet `/pet/state` 返回 engagement/activity/mood 正常。
- 既有测试套件跑通;删模块相关测试一并清理。
