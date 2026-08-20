# 失败案例库 + 回放评测（外部评分与改进回路）

**Date**: 2026-08-20
**Status**: Approved, pending implementation plan
**Scope**: 建立可重放、可打分的失败案例库，作为一切改进的适应度信号

> 起因是"能否引入 self-evolve"。结论是这个名字不适用：演化的主体是维护者，
> 不是 agent（§9）。她表现，我们捕获、评分、改进。本文档按后者设计。

---

## 1. 背景与动机

### 1.1 现有的四条"学习"通道，两条是死的

| 通道 | 机制 | 实际状态 |
|---|---|---|
| Reflector → patterns → 明天的计划 | GA 式反思 | **活的** |
| orchestrator `memory_writes` → 用户事实 | 每轮抽取 | **活的**（tangkeke 21 条） |
| core memory 自编辑 | `core_memory_append/replace` | **死的** |
| 人工标注 → fewshot | 飞书 👍/👎/改写 | **死的** |

- core memory：`type=core` 的事实数 aria=15、nini=0、tangkeke=0。工具只挂在 tool loop 上
  （`engine.py:374/387`），而线上 responder 是单程无工具调用，这条路径永远走不到。
- 人工标注：`data/personas/*/fewshot/turns/` 共 493 条 turn，**annotation 全部是 `none`，
  correction 0 条**。通道建好了，信号量是零。

### 1.2 没有任何适应度信号

仓库里不存在评测或回归 harness。`tests/` 是单元测试：确定性、不联网、不花钱，
覆盖的是解析和数据结构，**没有一条覆盖"她说出来的话是什么样"**。

### 1.3 但手工的进化循环已被证明有效

2026-08-19 一天之内，同一套流程抓出并修掉四个真实缺陷：

1. 用户指出坏输出
2. dump 那一轮的**真实 prompt**（不是复现，是实际发出去的那份）
3. 写探针重放 N 次
4. A/B 测候选修法
5. 数字动了才留下

第 5 步救过一次场：`user_state` 的第一版修法（对钟点措辞降级 + 附上作息）实测
**4/20 → 6/20**，比改前更差，当场废弃。没有测量，那个改动会被当成"修好了"交付。

**结论**：值得自动化的是第 2-4 步。第 5 步是这套东西存在的理由——
**没有适应度函数的自进化就是漂移，而且是会自我强化的漂移**
（reflector 的重复惩罚正是在治这个）。

---

## 2. 目标与非目标

### 目标

- 把每一个被指出的坏输出，固化成一个**可重放、可打分、不随时间腐坏**的案例
- 判定完全确定性，可无人值守运行
- 覆盖真实链路：真 orchestrator、真检索、真 prompt 组装
- 案例格式与打分接口从一开始就按"能批量跑候选"设计
- **建一个案例的人工成本只剩"错在哪"和"怎么算错"两项**（§9.1）

### 非目标（本期明确不做）

- LLM judge
- 自动候选生成
- CI 集成
- 向量检索（起步纯 FTS）
- 复活人工标注通道

---

## 3. 设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 优化目标 | 先建底座 | 没有分数，任何自进化都无法验证是否真的变好 |
| 判定方式 | 确定性判定器 | 分数完全可信；代价是只收有明确签名的失败 |
| 冻结边界 | 冻结上游状态，每次**重新组装** prompt | 8-19 四个修法有三个在组装层，冻结成品 messages 一个都测不到 |
| 自主程度 | 只测量，预留候选接口 | 候选生成需要先有可信分数 |
| 快照形式 | JSON fact 列表 → 临时 store | 可读、可 diff、可手写变体；走真实检索路径 |

---

## 4. 架构

```
evals/
├── cases/*.yaml          # 案例数据（可读、可 diff、可手写）
└── baseline.json         # 上次记录的分数
src/lingxi/evals/
├── case.py               # 加载 + schema 校验 + 相对时间解析
├── capture.py            # 从线上状态冻出 case 骨架（§9.1，第一等需求）
├── runner.py             # 建临时 store、组装、采样、判定、对比基线
└── detectors.py          # 确定性判定器注册表
tests/test_evals/         # 判定器和加载器自身的单测
```

案例是**数据**，放 repo 根；跑它的代码进 `src/`，这样判定器本身能被单测覆盖
（判定器也会有 bug）。**不进 `tests/`**：要联网、要花钱、结果随机，
混进去会毁掉 `pytest` 的确定性。

### 4.1 Case 格式

```yaml
id: offwork-state
symptom: 他说想下班 90 秒后，她问是在堵车还是到家了
origin: 2026-08-19 飞书对话；修法见 commit d3d3492
persona: config/personas/tangkeke.yaml
recipient: feishu:oc_eval

clock: "2026-08-19T20:20:54"          # 冻结时钟——case 语义的一部分，见 §5

facts:
  - subject: "user:feishu:oc_eval"
    type: pattern
    source: user_stated
    content: "对方一般晚上九点下班"
    importance: 4
    days_ago: 12

history:
  - {role: user,      content: "想下班了",              minutes_ago: 2}
  - {role: assistant, content: "想下班的心我懂！…",       minutes_ago: 2}
  - {role: user,      content: "就跟你想下课一样是吧",    minutes_ago: 1}

input: "大学还不轻松啊，天天都有时间做自己想做的事去"

samples: 20

premise:                               # 见 §6
  prompt_contains: ["下班后的个人时间"]
  prompt_lacks:    ["还在公司"]

detect:
  fail: {any_of: [堵车, 到家, 回家了, 在路上, 下班了吧]}
  pass: {any_of: [快下班, 还没下班, 还在公司, 加班]}

budget: {max_fail_rate: 0.05}
```

时间一律写**相对量**（`days_ago` / `minutes_ago`），相对 `clock` 解析。
绝对时间戳会让案例读起来像考古，而且改时钟时要手改一片。

`facts` 默认写 `expires_at = NULL`，除非案例本身要测过期行为——
避免案例放几周后事实被过期过滤器悄悄清掉。

### 4.2 单个 case 的执行流程

1. 建临时目录与 `FactStore`，按 `facts` 写入（相对 `clock` 解析 `ts`）
2. 灌 `history` 进 short-term 缓冲
3. 以冻结时钟调用**真实的** `_prepare_turn_v2(input)`
   → 真 orchestrator、真 renderer、真组装
4. 校验 `premise`（不成立直接 `BROKEN`，见 §6）
5. 用得到的 `(system, messages)` 并发采样 `samples` 次
6. 判定器统计 `fail_rate` / `pass_rate`
7. 与 `baseline.json` 对比，输出表格

### 4.3 输出

```
case              verdict   fail    pass    baseline    Δ
offwork-state     PASS      1/20    3/20    9/60      -13pp
tewatashi-scale   FAIL      7/20    2/20    —          new
some-case        BROKEN    前提不再成立：prompt 里找不到「下班后的个人时间」
```

`Δ` 只在该案例有基线时给出；`BROKEN` 行不打分，直接说明是哪条前提断言没通过。

### 4.4 CLI

```
lingxi-eval                          # 全部
lingxi-eval offwork-state            # 单个
lingxi-eval --baseline               # 把当前分数存成基线
lingxi-eval capture <recipient_key>  # 从线上状态冻出 case 骨架（§9.1）
```

`baseline.json` 记录 `{id: {fail_rate, pass_rate, samples, recorded_at, git_sha}}`。

---

## 5. 时钟注入

### 5.1 为什么必须卡死，且必须对应情景

时钟不是案例的元数据，**是案例语义的一部分**。`offwork-state` 之所以成立，
就因为 20:20 会让时段表吐出「下班后的个人时间」，而他实际还在公司——
**这个张力就是被测对象**。冻在 14:00，这个案例什么都测不到，还会一直显示绿色。

### 5.2 实现方式：显式注入，不用 monkeypatch

原方案是像单测那样 patch `datetime.now`。否决，理由是**静默解冻**：
以后任何人在链路里新增一个 `datetime.now()`，prompt 会有一部分悄悄跟随真实时间，
而案例不会报错——它只会开始测一个别的东西。

改为把 `now` 显式穿过组装路径，默认 `None` → `datetime.now()`，线上行为不变。

### 5.3 注入点（已逐一核对）

| 位置 | 行 | 影响 |
|---|---|---|
| `conversation/engine.py` `_prepare_turn_v2` | 713 | 传给 focus reminder 的 `current_time` |
| `conversation/context.py` `assemble_messages` | 91 | 日期分隔线 |
| `brain/renderer.py` `render_dynamic_blocks` | 137 | 认识多少天 |
| `facts/retriever.py` `fetch` | 63 | **recency 打分，决定哪些事实被捞出来** |
| `facts/store.py` `query` | 196 | 过期过滤（案例侧用 `expires_at = NULL` 规避） |

`prompt_builder` 那条链**已经是注入式的**——`build_turn_focus_reminder(current_time=…)`
和 `_build_time_awareness_section(current_time, …)` 都显式收时间，无需改动。

`retriever.py:63` 是最容易被忽略、后果最大的一处：它按 `exp(-0.01 * hours_old)`
给事实打新鲜度分。只冻结 prompt 时钟而不冻结它，案例里的事实每过一周就"更旧"一档，
**被捞进 prompt 的事实集合会悄悄变化**，案例失去可重复性。

`engine.py` 的 297/318/1478 是写入路径（新事实的 `ts`），不在读取与组装链路上，
本期不注入；若某个案例触发写入，其 `ts` 用真实时钟，不影响该轮打分。

### 5.4 顺带收益

`context.py:91` 这一处在 2026-08-19 已经咬过一次：日期分隔线的单测用冻结基准写，
而 `assemble_messages` 读真实时钟，隔夜就红了。注入之后这类"隔夜翻红"从根上消失。

### 5.5 外部可变量

- **天气必须 stub**：要联网，且是外部可变量
- **日出日落不 stub**：给定时钟与经纬度的纯离线计算，本身就是值得被测的逻辑

---

## 6. 前提断言与三档判定

时钟卡死还不够。如果有人改了时段表，或删掉了「下班后的个人时间」这句，
`offwork-state` 会继续跑、继续绿，但它测的东西已经没了。
**静默失效的案例比没有案例更糟**——它提供虚假的安全感。

因此每个案例声明自己成立的前提，在采样**之前**校验：

```yaml
premise:
  prompt_contains: ["下班后的个人时间"]   # 时钟确实落在这个桶里
  prompt_lacks:    ["还在公司"]           # 且 prompt 没直接把答案告诉她
```

判定三档：

| verdict | 含义 | 动作 |
|---|---|---|
| `PASS` | `fail_rate ≤ budget.max_fail_rate` | — |
| `FAIL` | 超预算 | agent 变差了，查最近改动 |
| `BROKEN` | 前提不成立 | **案例需要人重新设计**，与"变差"无关 |

`BROKEN` 与 `FAIL` 必须分开报。混在一起会把"测试过期了"误读成"系统退化了"，
进而引向错误的排查方向。

---

## 7. 判定器

`detectors.py` 是一个小注册表，起步三种：

| 判定器 | 语法 | 用途 |
|---|---|---|
| `any_of` | `{any_of: [str, ...]}` | 子串命中任一 |
| `regex` | `{regex: "..."}` | 结构化签名 |
| ~~`dates_outside_anchors`~~ | — | **实现后删除**，见下方说明 |

`fail` 命中 = 该次采样失败。`pass` 命中 = 该次采样明确正确，**仅作观测，不参与
`verdict` 判定**——它用来回答"改动是消除了错误，还是同时也带来了正确行为"。
8-19 的数据里这个区分是有意义的：改前 0/60 明确正确，改后 3/60。

**`dates_outside_anchors` 实现了又删了（2026-08-20，提交 `3b08234`）。** 三轮评审
每一轮都冒出一类新的误报：节日、问今天几号、第三人的日期、假设句、正确断言被同句
无关内容污染、无标点长句击穿分句、`我` 作定语修饰别人、否认句。判断"这个日期是不是
她在编自己的历史"需要主语、时态、否定——是语义不是子串，确定性判定器做不到。
本模块的硬约束是宁可漏报不可误报，一个不断长出新误报类别的判定器达不到这条。
§8 的 `invented-dates` 案例随之取消，乱编这一类留到 LLM judge 那一期。

判定器自身进 `tests/test_evals/` 做单测。

---

## 8. 起步的案例

全部来自 2026-08-19 的真实失败，各覆盖一类：

| id | 失败类别 | 判定 | 已有基线 |
|---|---|---|---|
| `offwork-state` | 状态判断 | `any_of` 通勤/到家词 | **有**：9/60 → 1/60 |
| `tewatashi-scale` | 行话与尺度 | `any_of` 二十秒/时间很短 | 无 |

`offwork-state` 自带 old/new 两组各 60 次采样的实测数据，
**可以直接用来校验 harness 本身没写错**——如果重放跑不出接近 9/60 与 1/60 的结果，
说明冻结或组装环节有问题，而不是 agent 有问题。这是第一个要跑通的案例。

本文档当初就把 `invented-dates` 标为三个里最不确定的一个，理由是乱编的签名比
"日期不在 anchors 里"杂得多。执行时这一点被证实了（见 §7），案例已随判定器取消。

---

## 9. 这个循环里谁是主体

**演化的主体不是 agent，是维护者。** 准确的形状是：

```
她表现  →  你指出坏输出  →  我捕获  →  我评分  →  我改进  →  回归
```

她既不知道自己被评分，也不参与改进。"self-evolve" 是个误导性的名字：
本期建的是**外部评分与改进回路**，不是自我改进回路。这个区分决定了下一节的定位。

### 9.1 推论：捕获必须接近零摩擦

如果每建一个案例都要人手工扒对话、拼 messages、算时间、写检测，
这个案例库会在第五个案例前后停止增长。

这不是审美判断，是本仓库自己的教训：飞书 👍/👎/改写通道建好了，
**493 条 turn，标注量 0**（§1.1）。它不是设计错了，是每次要人多点一下。
案例库如果有同样的摩擦，会死于同样的原因——只不过这次承受摩擦的是维护者。

因此 `capture` 是**第一等需求，不是便利功能**：

```
lingxi-eval capture <recipient_key> [--turns 8] [--at <ISO8601>]
```

从线上真实状态冻出一个 case 骨架：

- 按 `--at`（默认最后一轮的时间）取时钟
- 快照该 recipient 相关的 facts 子集，换算成相对 `clock` 的 `days_ago`
- 扒最近 `--turns` 轮对话，换算成 `minutes_ago`
- 以该时钟组装一次 prompt，把命中的关键句填进 `premise` 草稿
- 输出 `evals/cases/<date>-<slug>.yaml`

留给人手写的只有 `symptom` 和 `detect`——**即"错在哪"和"怎么算错"**。
其余是机械劳动，不该由人做。

## 10. 为批量候选预留的接口

核心是一个纯函数：

```python
async def score_case(case: Case, *, overrides: dict | None = None) -> CaseScore
```

`overrides` 可替换 persona 字段、prompt 片段、orchestrator 指令。
批量跑候选即"同一个 case 传不同 `overrides`"——**这是给维护者试候选用的**（§9），
不是给 agent 自我改进用的。2026-08-19 敲定 `user_state` 写法手工 A/B 了五轮，
那五轮本该是一条命令。

本期**不实现**候选生成，但接口现在就是这个形状，升级时不必推倒重来。

`CaseScore` 至少包含：`verdict`、`fail_rate`、`pass_rate`、`samples`、
`premise_ok`、以及每次采样的原文（排查时要看她到底说了什么）。

---

## 11. 成本与运行

3 个案例 × 20 采样 = 60 次调用。responder 走 DeepSeek V4 Flash，
prompt ~6.5k tokens 且大部分命中前缀缓存（8-19 实测 6144-9088 tokens 缓存命中）。
并发跑一到两分钟，成本以分计。

每个案例另有 1 次 orchestrator 调用（Claude）用于组装。

---

## 12. 风险

| 风险 | 应对 |
|---|---|
| 时钟注入遗漏某处，prompt 部分解冻 | `premise` 断言会在前提被破坏时报 `BROKEN` |
| 纯 FTS 检索与线上（FTS+向量）有偏差 | 先接受；若某案例结论确实取决于向量，再单独为其算向量 |
| 判定器过窄，抓不到同类新变体 | 接受。宁可漏报也不误报——**误报会让人不信任这套分数，那才是致命的** |
| 20 次采样的统计噪声 | `budget` 按实测分布定，不按直觉定；跨多次运行看趋势 |
| 案例随人设演进而失效 | 这正是 `BROKEN` 档的用途 |

---

## 13. 测试策略

- `tests/test_evals/test_detectors.py`：每个判定器的命中与不命中
- `tests/test_evals/test_case_loader.py`：schema 校验、相对时间解析、缺字段报错
- `tests/test_evals/test_runner_freeze.py`：**不联网**，只断言在冻结时钟下
  `_prepare_turn_v2` 两次调用产出逐字节相同的 prompt

第三条是这套东西的根基：如果同一个案例两次组装出的 prompt 不同，
后面所有分数都没有意义。
