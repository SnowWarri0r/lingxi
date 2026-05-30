# 旧栈退役审计清单（Decommission Audit）

> 2026-05-29 · 只读审计,不动代码 · 目的:把"Facts 整体重构"没做完的下半场(拆旧)落成可执行清单

## 核心结论

这次 59-commit 的重构**只做了"建新",没做"拆旧"**。新栈(`facts/` + `brain/` + `planner/` + `facts/reflector`)全部上线,但它要取代的 4 套旧系统一个都没退役,现在**新旧并行运行,且在同一个 prompt 里打架**。

### 实锤:同一个 prompt 灌两套记忆

`conversation/engine.py::_prepare_turn_v2`(fact_retriever 已 wire → 这是线上真实路径):

```
memory_context = await self.memory.assemble_context(...)         # 旧 Chroma 记忆
messages       = self.context_assembler.assemble_messages(memory_context)   # 用它拼 message
dynamic_block  = await render_dynamic_blocks(self.fact_retriever, ...)       # 同时渲染新 facts
```

旧 Chroma 的 long-term facts 进了 message 历史,新 facts 进了 dynamic block,两者拼进同一个 system/messages。**人格"精分"(博士组 vs 自由职业、今天做了两件互斥的事)的物理来源就是这里**——两套各说各话的记忆同时在场。

### 实锤:两套生活引擎同时跑

`channels/feishu.py`:
- line 414/423:`LifeSimulator.start()` —— 旧的随机生活模拟,写 `inner_life` JSON state
- line 484/670:`_start_plan_loops()` → `PlanExecutor.tick()` —— 新的 daily-plan 生活引擎,写 `facts.db` life facts

commit `1095e68` 写明 PlanExecutor "replaces random simulator",但 simulator 从没被停。两个引擎并行生成"Aria 在干什么"。

---

## 退役卡片（按依赖深度,从叶子到根）

### ① temporal/reflection —— 最容易,先拆

| | |
|---|---|
| 职能 | 旧的定时反思循环 |
| 新栈替代 | `facts/reflector.py`(tree-of-thought,first-person)+ `facts/reflection_trigger.py`(已在 app.py wire) |
| 外部消费者 | **仅 1 处**:`channels/feishu.py:31` import `ReflectionConfig, ReflectionLoop` |
| 阻塞 | 无。reflector 已接管 |
| 动作 | 删 feishu.py 里 ReflectionLoop 的 import + 启动;删 `temporal/reflection.py` |
| 风险 | 极低,叶子节点 |

### ② inner_life/LifeSimulator(只拆生成循环)—— 止血点

| | |
|---|---|
| 职能 | 后台随机生成 Aria 的生活事件 |
| 新栈替代 | `planner/PlanExecutor`(按 daily plan 生成 `aria.event` life facts) |
| 外部消费者 | `channels/feishu.py:29,414,423`(启动循环) |
| 阻塞 | inner_life **其余部分**(store/agenda/subjective/models)还被 engine + prompt_builder 用,**不能一起删**。本步只停 simulator 这个循环 |
| 动作 | feishu.py 不再 `LifeSimulator(...).start()`;`inner_life/simulator.py` 标记退役/删除 |
| 风险 | 低。但要先确认 simulator 是否还往 `recent_events` 推东西被别处读(见"待验证 ①") |

### ③ planning/(Planner + executor + scheduler)

| | |
|---|---|
| 职能 | 旧目标/计划系统 + per-turn proactive action |
| 新栈替代 | `planner/DailyPlanner` + `planner/PlanExecutor` |
| 外部消费者 | `app.py:17`、`engine.py:30-32`(import Planner/ActionExecutor/Scheduler)、`prompt_builder.py:15`(Plan 类型) |
| 现状 | per-turn `planner.check_proactive_action` **已在 engine.py:310 注释禁用**,但对象仍构造(engine.py:170-173)并把 `active_plans` 喂进旧 `_prepare_turn` 渲染 |
| 阻塞 | 与旧 `_prepare_turn` 路径绑定;`_prepare_turn` 删除前 Planner 不能完全摘 |
| 动作 | 删 engine 里 Planner/ActionExecutor/Scheduler 构造与调用;删 prompt_builder 的 `active_plans`/`_build_plan_section`;删 `planning/` 目录;app.py 去 import |
| 风险 | 中。要连带 ④ 一起,因为都挂在 `_prepare_turn` 老路径上 |

### ④ memory/(Chroma 整栈)—— 最终 boss,最后拆

| | |
|---|---|
| 职能 | ChromaDB 向量记忆:manager / chroma_store / episodic / consolidation / long_term / short_term / entity_graph |
| 新栈替代 | `facts/`(SQLite+FTS5)+ `facts/writers/*`(biography/inference/life/npc/user_statement/world 已覆盖旧写入职能) |
| 外部消费者 | **最深**:`engine.py`(含线上 `_prepare_turn_v2:633` 的 `assemble_context`!)、`conversation/context.py`、`conversation/turn_focus.py`(ConversationTurn)、`prompt_builder.py`(MemoryContext + `_build_memory_section`)、`planning/`、`temporal/relationship.py`、`app.py` |
| 阻塞 | `_prepare_turn_v2` 当前**同时**用 memory_context 拼 message + facts 渲染 block。必须先把 message 组装也切到 facts/renderer,才能摘掉 `self.memory` |
| 动作 | 1) renderer/context_assembler 改为纯 facts 来源;2) `short_term.ConversationTurn` 这种纯数据结构可保留或搬家(不是 Chroma,别误删);3) 摘 engine.memory;4) 删 prompt_builder memory_section;5) 删 `memory/` 里 Chroma 相关文件 |
| 风险 | 高。这是双喂 prompt 的根因,也是耦合最深的。单独一个 PR,带端到端验证 |

---

## ⚠️ 建议执行顺序(已被 workflow wx8sj4pc6 推翻,见下方"已验证顺序")

> 下面这版**初稿顺序是错的**,保留作记录。错因:把消费者(reflection/simulator)排在生产者(memory)之前退,且误判 LifeSimulator 可"止血先关"。

~~① reflection → ② LifeSimulator(止血) → ③ planning → ④ memory → ⑤ inner_life 残余~~

---

## ✅ 已验证执行顺序(workflow wx8sj4pc6 · 并行实测 + 对抗式审查 + 合成)

**核心修正:** 退役必须"从叶子到根"=先切断 chat-time 对旧数据的依赖,再退生产者。LifeSimulator **不是**快速止血点,而是 `inner_state` 的唯一生产者,退它之前必须先让 PlanExecutor 接管 inner_state 生产。

### 第 0 步(承重墙·非退役·真实开发量)
不是删代码,是补三块继任者,否则后面每一步都会塌:
1. **PlanExecutor 接管 inner_state 生产** — 把 simulator 的 `_tick_activity`/`_drift_dynamics`/`_update_axis_modulation`/每日 plan 逻辑迁到 PlanExecutor,写 `inner_life_store`(当前它零 inner_life 引用,这是新建不是搬运)。
2. **运行时 biography writer** — 接住原 reflection 的 `_maybe_add_biography_event` + simulator 经 life_writer 的 EVENT 两条腿;新栈现在只有 BiographyLoader 冷启动,无运行时增量。
3. **daily-narrative bridge** — `consolidate_day_narrative`(consolidation.py:167)唯一上游是 simulator.py:542,要把日记→episode 语义迁到 facts。

### 第 1 步 · 退 temporal/reflection(risk: high)
前置:第0步 biography writer 上线;确认 reflection 对 memory 的 4 处反向调用(add_fact:181 / get_history:204 / assemble_context:269 / snapshot_for_recipient:282)已被新栈覆盖。**注意 feishu.py 里 reflection 启动块(400-409)与 simulator 启动块(411-424)行号相邻、结构相同,必须按语义边界删,不能照行号往下吃。**

### 第 2 步 · 退 LifeSimulator(risk: high · 原测绘缺失,本图补齐)
前置:第0步 PlanExecutor 已 A/B 验证接管 inner_state、daily-narrative bridge 已接管。删 simulator.py 单文件,`inner_life/` 其余(store/agenda/subjective/models)保留——它们被 PlanExecutor/proactive/pet 继续用。

### 第 3 步 · 退 planning/ + 旧 _prepare_turn(risk: **low** ✅ 唯一可现在安全做的)
实测确认旧 `_prepare_turn` 是**死代码**:fact_retriever 在所有生产路径恒非空(app.py 无条件实例化并传入 create_engine),engine.py:1051/1122/1204 三处 else 分支不可达。整目录删 + 删旧 _prepare_turn + 删 prompt_builder 的 active_plans。**自包含、不依赖第0步。**

### 第 4 步 · 退 memory/Chroma(risk: high · 最终 boss)
前置最多:reflection+simulator 已退、ConsolidationBridge + EpisodeToFactsDecomposer 已实现、Chroma 数据迁移、双写灰度 1-2 周期、删 engine.py:633-636 后 A/B 验证仅 facts 路径不丢质量(这是人格碎片化根因的最终闸门)。

### 需要你拍板的产品级决定(open risks)
1. 新 Reflector 比旧的有**三处语义降级**:只写全局 aria PATTERN(丢 per-recipient 作用域)、丢 6h per-recipient cooldown、丢 per-recipient biography 归因——可接受还是要补回?
2. 运行时 biography writer 的 source 归因:复用 `Source.BIOGRAPHY`(和冷启动混)还是新增 source?
3. memory 退役的双写灰度窗口 vs 第1步 reflection 退役时间线冲突,谁先?
4. Chroma→FactStore 迁移不可逆,迁移脚本要你 review,建议留只读快照备回滚。

---

## 待验证(不确定项,动手前要查清)

1. **LifeSimulator 是否还往别处推事件被消费** —— 停它之前确认没有别的模块依赖它写的 `recent_events` / inner_life state(temporal/proactive.py 882 行那个大文件最可疑)。
2. **`_prepare_turn`(旧路径)是否真的永不触发** —— 三处调用点都是 `if fact_retriever is not None` 走 v2、else 走旧。线上 fact_retriever 恒非空 ⇒ 旧路径理论上是死的,但 app.py 里有没有可能不 wire fact_retriever 的启动分支要确认。
3. **inner_life/models 里的 EngagementMode / derive_engagement_mode** —— 被 prompt_builder + pet/state_endpoint 用,这是"情绪/参与度"逻辑,**不属于 Chroma 也不属于生活模拟**,退役时别误伤,可能要单独留存或搬家。
4. **facts/writers 覆盖度复核** —— 逐个确认 life/world/npc/user_statement/inference/biography 这 6 个 writer 的输出,语义上确实等价覆盖了旧 memory/consolidation + inner_life 的写入,没有遗漏的事实类型。
