# 双通道口语化 + 标注池 设计

**Date**: 2026-04-21
**Status**: Approved, pending implementation plan
**Scope**: 解决 Aria 长对话中「AI 腔」问题；建立长期可自我矫正的口语样本池

---

## 1. 背景与动机

### 1.1 现状问题

Aria 在长对话中持续暴露「AI 腔」：

1. 太整齐/全面（一次把三点都说清楚）
2. 固定套话（"希望这对你有帮助"、"如果有任何问题"）
3. 长句偏多，缺短句碎句
4. 书面语泄漏（"然而"、"此外"、"需要注意的是"）
5. 太"正确"，缺个人偏见/立场
6. 缺现实锚点（真人会插入"等等楼下在装修"之类的琐碎）

用户举证片段：

> "哇，这也太巧了吧！世界真的很小啊，绕了一圈竟然能这样连起来。这种巧合总是让人觉得很神奇。"

这种 AI 腔的根源是 LLM 在"对用户回答"的生成模式下，默认回退到 "helpful assistant" 分布，任何 prompt 顶部的风格指令都会被几十轮对话稀释。

### 1.2 核心洞察

- **切换任务语义**：让 LLM 从"回答用户"切到"把一段想法用口语转述出来"。后者不在"helpful assistant"分布里，天然跳出 AI 腔。
- **自矫正闭环**：用户标注差评 + 提供正确答案 → 进入样本池 → 未来相似想法检索到该样本作 few-shot → 错误收敛。

---

## 2. 架构

### 2.1 总体流程

```
用户消息
  ↓
[Call 1 思考层]  Claude Sonnet，非流式
  in : persona + inner_life + 对话历史 + 用户消息
  out: inner_thought (完整想法) + meta (情绪Δ/记忆写入/agenda 更新)
  不输出 speech
  ↓
[检索层]  本地
  embedding(inner_thought) → ChromaDB 查池，top-k
  拼装 Call 2 prompt: seeds(3) + 检索(3)
  ↓
[Call 2 口语化层]  Claude Sonnet，流式
  in : inner_thought + 最后 1 轮用户消息摘要 + few-shot + 黑名单 + 长度帽
  out: speech (一句话微信腔，≤40 字默认)
  ↓
[标注入口]  Feishu 卡片底部
  👍 像 / 👎 不像 / ✏️ 应该说
```

### 2.2 关键设计决策

| 决策 | 选择 | 原因 |
|---|---|---|
| 单次 vs 两次调用 | **两次独立调用** | Call 2 看"想法"而非"用户问题"，任务语义才真正切换 |
| Call 2 模型 | **Claude Sonnet（同 Call 1）** | 先求人设一致性，小模型留后续成本优化 |
| Call 2 输入范围 | **inner_thought + 最后 1 轮用户消息摘要** | 完全断上下文会答非所问；完整给消息会退化回答模式 |
| 长度帽 | **默认 ≤40 字**，persona YAML 可覆盖 | 硬约束比软指令有效 |
| few-shot 数 | **seeds 3 + 检索 3 = 6** | 太多稀释，太少样本不足 |
| 相似度阈值 | **0.6** | 宁缺毋滥 |
| Embedding | **复用豆包 2048 dim** | 避免多套 embedding 模型 |

---

## 3. 数据模型

### 3.1 FewShotSample

```python
class FewShotSample(BaseModel):
    id: str                                  # uuid
    inner_thought: str                       # 想法原文（embedding 来源）
    original_speech: str | None              # 原答；差评时有，seed/positive 为 None
    corrected_speech: str                    # 目标口语输出
    context_summary: str                     # 一句话场景，例 "深夜关心"
    tags: list[str]                          # 辅助过滤 ["共鸣","吐槽"]
    recipient_key: str | None                # 关系专属（None = 全局种子）
    source: Literal["seed", "user_correction", "positive"]
    created_at: datetime
```

**存储**：
- ChromaDB collection `fewshot_pool_d2048`：embedding(inner_thought) + metadata
- JSON 备份 `data/fewshot/samples.jsonl`：full record，灾备

### 3.2 AnnotationTurn

每轮对话保留供标注的原始状态：

```python
class AnnotationTurn(BaseModel):
    turn_id: str
    recipient_key: str
    user_message: str
    inner_thought: str                       # Call 1 生成
    speech: str                              # Call 2 生成（Aria 实际说的）
    created_at: datetime
    annotation: Literal["none", "positive", "negative"] = "none"
    correction: str | None = None            # 用户提供的应该说的话
```

**存储**：
- `data/fewshot/turns/<turn_id>.json`
- 未标注的 30 天后自动清理（启动时扫目录 mtime，Phase 4 实现）
- 标注升级为 FewShotSample 后 turn 记录保留 7 天再删，便于溯源

---

## 4. 运行时流程

### 4.1 Engine 改造

```python
class ConversationEngine:
    async def chat_turn(self, user_msg: str, recipient_key: str) -> TurnOutput:
        # 1. 思考
        think = await self._think_turn(user_msg, recipient_key)

        # 2. 检索 few-shot
        fewshots = self.fewshot_retriever.retrieve(
            inner_thought=think.inner_thought,
            recipient_key=recipient_key,
            k=3,
        )

        # 3. 口语化
        speech = await self._compress_turn(
            inner_thought=think.inner_thought,
            last_user_snippet=self._truncate(user_msg, max_chars=60),  # 截断 user msg 给 Call 2 作锚点；空字符串表示主动消息
            fewshots=fewshots,
            persona=self.persona,
        )

        # 4. 记录 AnnotationTurn
        turn_id = str(uuid4())
        self.annotation_store.record(
            turn_id=turn_id,
            recipient_key=recipient_key,
            user_message=user_msg,
            inner_thought=think.inner_thought,
            speech=speech.text,
        )

        # 5. 合并
        return TurnOutput(
            turn_id=turn_id,
            speech=speech.text,
            inner_thought=think.inner_thought,
            **think.meta,
        )
```

### 4.2 Call 1 Prompt（prompts/think.py）

- 保留当前全部 inner_life / persona / memory 上下文
- 关键改动：**明确禁止输出 speech**
- 输出格式：

```
=== INNER THOUGHT ===
{完整内心独白：对用户的理解、自己的感受、要不要提生活事件、对关系的评估}
=== META ===
{JSON: emotion_deltas, memory_writes, plan_updates, mood_label, agenda_delivered}
```

### 4.3 Call 2 Prompt（prompts/compress.py）

`max_chars` 取值：`persona.style.speech_max_chars ?? 40`
`last_user_snippet`：截断至 60 字的用户消息；主动消息时为空串，prompt 改写成【这次是你主动找他】

```
你是 Aria。

下面是你此刻的想法。用微信聊天的语气把它说出来——一句话，最多两句。

规则：
- ≤{max_chars} 字
- 不说：希望、如果有任何、总的来说、需要注意的是、世界真的很小、总是让人、这对你、很高兴为你
- 禁止总结刚说的话、禁止给建议框架（1/2/3 点）
- 允许：省略、倒装、感叹词（嗯/欸/哦）、破折号、半句话

参考几个「想法 → 说」：
{fewshots}

【用户刚说】{last_user_snippet}
【你想的是】{inner_thought}
【你说】：
```

### 4.4 Feishu 阶段卡片

| 阶段 | 卡片内容 |
|---|---|
| T0: 接收 | "Aria 正在想…" + dots |
| T1: Call 1 完成 | 过渡："Aria 正在组织语言…"；可选秀 inner_thought 首 20 字做彩蛋 |
| T2: Call 2 流式 | speech 逐字流入 |
| T3: 完成 | 卡片底部加 turn_id 标签 + 三按钮 `👍` `👎` `✏️` |

Phase 1 T1 用静态文字，Phase 2 做动画。

---

## 5. 标注系统

### 5.1 入口

**Feishu**（主）：
- 卡片底部 3 按钮：
  - `👍 像` → source=positive 直接入池
  - `👎 不像` → 标记差评；30 分钟内接 `/bad <turn_id> <correction>` 补成 user_correction
  - `✏️ 应该说` → 弹 Feishu 表单（单行输入）→ 提交后 source=user_correction
- `/reveal <turn_id>` → 独立 DM 卡片展示 inner_thought（仅本 recipient 可见）

**CLI**：
- 每轮输出附 `turn_id`
- `:good` / `:bad <correction>` / `:reveal`

**Web API**：
- `POST /turns/<turn_id>/annotate` body `{kind, correction?}`
- `GET /turns/<turn_id>/inner_thought`

### 5.2 Collector 逻辑

```python
class AnnotationCollector:
    def record_positive(self, turn_id: str) -> None: ...
    def record_negative(self, turn_id: str) -> None: ...          # 只标记
    def record_correction(self, turn_id: str, correction: str) -> None:
        turn = self.store.get_turn(turn_id)
        sample = FewShotSample(
            inner_thought=turn.inner_thought,
            original_speech=turn.speech,
            corrected_speech=correction,
            context_summary=self._summarize(turn),       # LLM 辅助
            tags=self._tag(turn),                        # LLM 辅助
            recipient_key=turn.recipient_key,
            source="user_correction",
        )
        self.pool.add(sample)
```

### 5.3 隐私与噪声

- `inner_thought` 可能含对用户的敏感印象；`/reveal` 仅对本 recipient 显示
- recipient 间隔离：某 recipient 的 correction 不影响别人，除非 recipient_key=None（全局种子）
- 防恶意标注：24h 内同用户 >N 次 correction 触发审核（Phase 4 后做）

---

## 6. 检索策略

### 6.1 Retriever

```python
def retrieve(inner_thought: str, recipient_key: str, k: int = 3) -> list[FewShotSample]:
    q = embedding.encode(inner_thought)

    # 召回 k*4 候选
    candidates = chroma.query(
        q,
        n_results=k * 4,
        where={"$or": [
            {"recipient_key": recipient_key},
            {"recipient_key": None},
        ]},
    )

    # 重排序
    scored = []
    for c in candidates:
        score = c.similarity
        if c.metadata["recipient_key"] == recipient_key:
            score += 0.1
        if c.metadata["source"] == "user_correction":
            score += 0.05
        elif c.metadata["source"] == "positive":
            score += 0.02
        scored.append((score, c))

    # 阈值过滤 + 去重
    filtered = [c for s, c in scored if s > 0.6]
    deduped = dedup_by_similarity(filtered, threshold=0.9)

    return deduped[:k]
```

### 6.2 拼装

```python
def assemble_fewshots(retrieved: list[FewShotSample]) -> str:
    seeds = fewshot_store.get_seeds(3)
    pairs = seeds + retrieved
    return render_fewshot_section(pairs)   # "想法：...\n说：..." 重复
```

---

## 7. 模块拆分

```
src/lingxi/
├── fewshot/                       # 新增
│   ├── __init__.py
│   ├── models.py                  # FewShotSample, AnnotationTurn
│   ├── store.py                   # Chroma + JSON
│   ├── retriever.py               # 查询 + 重排 + 拼装
│   ├── collector.py               # 接受标注
│   ├── summarizer.py              # LLM 辅助 context_summary/tags
│   └── seeds.yaml                 # 冷启动种子（10 条）
├── conversation/
│   ├── engine.py                  # 改：_think_turn / _compress_turn
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── think.py               # Call 1 prompt
│   │   └── compress.py            # Call 2 prompt
│   └── output_schema.py           # 改：ThinkOutput / SpeechOutput / TurnOutput
└── channels/
    └── feishu.py                  # 阶段卡片 + 标注按钮 + /reveal
```

**数据目录**：

```
data/
├── fewshot/
│   ├── samples.jsonl              # FewShotSample 备份
│   └── turns/<turn_id>.json       # AnnotationTurn 原始
└── chroma/
    └── fewshot_pool_d2048/        # ChromaDB 集合
```

**配置**（persona YAML 新增可选字段）：

```yaml
style:
  speech_max_chars: 40
  blacklist_phrases:
    - "据说"
compression:
  fewshot_seed_count: 3
  fewshot_retrieved_count: 3
  similarity_threshold: 0.6
```

---

## 8. 实施分期

### Phase 1: 核心双通道（1-2 天）

验证"AI 腔"能否通过任务语义切换消除

- [ ] `ThinkOutput` / `SpeechOutput` / `TurnOutput` schema
- [ ] `prompts/think.py`、`prompts/compress.py`
- [ ] `_think_turn` / `_compress_turn`（engine 改造）
- [ ] Call 2 硬编码 3 条临时 seed（Phase 3 前占位）
- [ ] CLI 跑通，手动对比 before/after

**验收**：10 条对比，前后风格明显改善；套话出现率可量化下降

### Phase 2: Feishu 阶段卡片（0.5 天）

- [ ] 卡片状态机 `thinking` / `speaking` / `done`
- [ ] Call 1 期间 "Aria 正在想…"
- [ ] Call 2 流式接入 speech
- [ ] inner_thought 首 20 字彩蛋过渡

**验收**：Feishu 实测，2-3s thinking + 流式 speech，体感可接受

### Phase 3: Seeds 冷启动（0.5 天）

提前到第 3 步，因为 Phase 4 需要起始池

- [ ] `seeds.yaml` 10 条初稿，场景覆盖：
  1. 共鸣（"也有过类似的事"）
  2. 安慰（不矫情）
  3. 吐槽（带个人立场）
  4. 好奇追问（短）
  5. 轻度拒绝（不客气化）
  6. 懒得接话
  7. 生活琐事插入（"等等我喝口水"）
  8. 主动分享自己的事
  9. 情绪低落下的回应
  10. 被用户戳中
- [ ] 用户审阅改
- [ ] 落盘 → FewShotStore

### Phase 4: 标注系统（1 天）

- [ ] `AnnotationTurn` 每轮记录
- [ ] Feishu 卡片底部按钮
- [ ] `AnnotationCollector` 接 3 种操作
- [ ] `/reveal` 命令
- [ ] CLI `:good` / `:bad` 支持

**验收**：产出 ≥5 条 user_correction 样本

### Phase 5: 检索注入（0.5 天）

- [ ] `FewShotRetriever.retrieve()`
- [ ] Call 2 prompt 动态拼装（替换 Phase 1 硬编码）
- [ ] 观察 `/bad` 率下降

**验收**：池子累 20+ 样本后，同类场景 `/bad` 率明显下降

---

## 9. 风险与缓解

| 风险 | 可能性 | 影响 | 缓解 |
|---|---|---|---|
| Call 2 也 AI 腔 | 中 | 方案失败 | Phase 1 必须手验；不行回退或加 prefill |
| 2× 延迟影响体验 | 中 | UX 降级 | Phase 2 阶段卡片+过渡彩蛋掩盖；后续 Call 2 可换 Haiku |
| Few-shot 检索污染 | 低 | 回答走样 | recipient_key 隔离 + 阈值 0.6 + user_correction 优先 |
| 冷启动池空 | 高（短期） | 等价 Phase 1 | Seeds 3 条兜底始终注入 |
| inner_thought 泄漏隐私 | 中 | 用户不适 | `/reveal` 限本 recipient；卡片不默认展示 |
| 标注疲劳 | 高 | 自矫正失效 | 按钮轻量；positive 样本也自动积累；不强求 correction |

---

## 10. 开放问题（不阻塞实施）

1. Call 2 换 Haiku 做成本压缩？Phase 5 后实验
2. LLM 自动打 tag 的准确度？先用 LLM 辅助，不达标再人工
3. seeds 是否随 persona 变？先通用一套，后续 persona YAML 覆盖路径

---

## 11. 非目标

- 不做 RLHF / fine-tuning
- 不做复杂情感分析（LLM 辅助足够）
- 不做跨 persona 样本共享
- 不改 memory/entity/agenda 现有架构
