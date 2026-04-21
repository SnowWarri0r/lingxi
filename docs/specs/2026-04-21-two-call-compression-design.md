# 拟人化升级：单调用组合拳 + 标注池 + 可选双通道

**Date**: 2026-04-21
**Status**: Approved, pending implementation plan
**Scope**: 解决 Aria 长对话中「AI 腔」问题；建立行业标配的 few-shot 自矫正闭环；保留双通道作为兜底升级

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

### 1.2 路线权衡

**成熟项目的做法**：

| 项目 | 核心手段 | 两次调用 |
|---|---|---|
| SillyTavern | character card + example dialogues 当 prior turns + Author's Note 深度注入 + swipes | 否 |
| Character.AI | 专用 fine-tune + 用户反馈喂训练 | 否 |
| Replika / Pi | 专用 fine-tune + RLHF | 否 |
| Claude/GPT 自搭拟人 | 单调用 + 强 prompt + prior-turn few-shot | 绝大多数否 |

行业标配是 **fine-tune** 或 **强 character card + prior-turn few-shot + 深度注入**，均为单调用。

**我们之前未系统化跑过的单调用杠杆**：

1. **Prior-turn few-shot**：example 不写进 system prompt 当散文，而是渲染成真实 `user/assistant` 消息对插在历史最前。LLM 对最近 assistant 消息模仿极强
2. **Author's Note 深度注入**：在用户消息前插风格提醒，位置比系统 prompt 顶部有效得多
3. **Assistant prefill**：Anthropic API 预填 assistant 开头，碎句锚定口语
4. **Sampler 调参**：temperature / top_p

**双通道（两次调用）**：逻辑成立（任务语义从"回答用户"切成"转述想法"），但社区未广泛验证，2× 延迟和成本是真代价。

### 1.3 选定路线

先做**单调用组合拳 + 标注闭环**（行业标配路径）。双通道作为**兜底升级**，只在组合拳不够时触发。

---

## 2. 架构

### 2.1 主路径（单调用）

```
用户消息
  ↓
[检索层]  本地
  对上一轮 Aria inner_thought 或当前 user_msg embedding 查 few-shot 池
  召回 top-k (3) + seeds (3)
  ↓
[单次 LLM 调用]  Claude Sonnet，流式
  system prompt:   persona + inner_life + memory
  prior turns:     few-shot（渲染为 user/assistant 消息对）+ 真实对话历史
  user (final):    [style: 微信聊天,≤N字,禁 X/Y/Z] + 用户消息
  assistant prefill: 根据 persona 挑 1 个碎句开头（如"嗯"）
  out: speech + meta（emotion_deltas / memory_writes / agenda / inner_thought）
  ↓
[AnnotationTurn 记录]
  turn_id 持久化 inner_thought + speech
  ↓
[Feishu 卡片]  底部 👍 / 👎 / ✏️
```

### 2.2 兜底升级（双通道，可选）

仅当 §8 Phase 0-3 跑完 AI 腔仍显著残留时启用。架构：

```
... [Call 1 思考层] → inner_thought + meta
... [检索层]       → few-shot
... [Call 2 压缩层] → speech（inner_thought + few-shot + 黑名单 + 长度帽）
```

详见 §8 Phase 4-5。核心改动：engine 拆 `_think_turn` / `_compress_turn`，few-shot 注入点从 prior-turn 区移到 Call 2 prompt。

### 2.3 关键设计决策

| 决策 | 选择 | 原因 |
|---|---|---|
| 单调用 vs 双通道 | **先单调用** | 行业标配；便宜快；失败代价小 |
| few-shot 渲染方式 | **prior-turn 消息对** | LLM 对历史 assistant 消息模仿远强于 system prompt 指令 |
| 风格提醒注入位置 | **用户消息前缀**（Author's Note） | 位置越近 LLM 服从度越高 |
| Prefill 策略 | **persona 定义 3-5 个碎句开头**，随机选一 | 避免开头套话，零延迟 |
| Sampler | **temperature 0.9-1.0, top_p 0.95**（从默认 1.0/1.0 起步调） | 默认可能太"稳" |
| Embedding | **复用豆包 2048 dim** | 避免多套模型 |
| few-shot 数量 | **seeds 3 + 检索 3 = 6** | 太多稀释 |
| 相似度阈值 | **0.6** | 宁缺毋滥 |
| 长度帽 | **默认 ≤40 字**，persona YAML 可覆盖 | 硬约束 |

---

## 3. 数据模型

### 3.1 FewShotSample

```python
class FewShotSample(BaseModel):
    id: str                                  # uuid
    inner_thought: str                       # 想法/场景（embedding 来源）
    original_speech: str | None              # 原答；差评时有，seed/positive 为 None
    corrected_speech: str                    # 目标口语
    context_summary: str                     # 一句话场景，例 "深夜关心"
    tags: list[str]                          # 辅助过滤 ["共鸣","吐槽"]
    recipient_key: str | None                # 关系专属（None = 全局种子）
    source: Literal["seed", "user_correction", "positive"]
    created_at: datetime
```

**存储**：
- ChromaDB collection `fewshot_pool_d2048`：embedding(inner_thought) + metadata
- JSON 备份 `data/fewshot/samples.jsonl`

### 3.2 AnnotationTurn

```python
class AnnotationTurn(BaseModel):
    turn_id: str
    recipient_key: str
    user_message: str
    inner_thought: str                       # 从 meta 里抽
    speech: str
    created_at: datetime
    annotation: Literal["none", "positive", "negative"] = "none"
    correction: str | None = None
```

**存储**：
- `data/fewshot/turns/<turn_id>.json`
- 未标注 30 天清理；已标注 7 天后删（保留溯源窗口）

---

## 4. 运行时流程

### 4.1 Engine（单调用路径）

```python
class ConversationEngine:
    async def chat_turn(self, user_msg: str, recipient_key: str) -> TurnOutput:
        # 1. 检索 few-shot
        #    策略：先用 user_msg embedding 查；有历史 inner_thought 时优先它
        query_text = self._last_inner_thought(recipient_key) or user_msg
        fewshots = self.fewshot_retriever.retrieve(
            query=query_text,
            recipient_key=recipient_key,
            k=3,
        )

        # 2. 组 prompt
        messages = self._build_messages(
            history=self.history(recipient_key),
            fewshots=fewshots,                      # 渲染为 user/assistant 消息对
            user_msg=user_msg,
            style_preamble=self._build_style_preamble(),  # Author's Note
        )
        prefill = self._pick_prefill()               # persona.style.prefill_openers 随机

        # 3. LLM 调用
        output = await self.llm.complete(
            system=self.system_prompt(),
            messages=messages,
            prefill=prefill,
            temperature=self.persona.sampling.temperature,
            top_p=self.persona.sampling.top_p,
        )

        # 4. 解析 speech + meta
        parsed = parse_turn_output(prefill + output.text)

        # 5. 记录 AnnotationTurn
        turn_id = str(uuid4())
        self.annotation_store.record(
            turn_id=turn_id,
            recipient_key=recipient_key,
            user_message=user_msg,
            inner_thought=parsed.inner_thought,
            speech=parsed.speech,
        )

        return TurnOutput(turn_id=turn_id, **parsed.model_dump())
```

### 4.2 Prior-turn Few-shot 渲染

```python
def render_fewshots_as_messages(samples: list[FewShotSample]) -> list[Message]:
    """
    渲染为 user/assistant 交替对。不用 system prompt 里的散文描述。
    context_summary 扮演"用户假想消息"，corrected_speech 是"assistant 回复"。
    """
    msgs = []
    for s in samples:
        msgs.append({"role": "user", "content": s.context_summary})
        msgs.append({"role": "assistant", "content": s.corrected_speech})
    return msgs
```

### 4.3 Style Preamble（Author's Note）

每轮附在用户消息前：

```
[style: 微信聊天。≤{max_chars}字。
禁用词：希望、如果有任何、总的来说、需要注意的是、世界真的很小、总是让人、这对你
禁止总结、禁止给建议框架（1/2/3 点）
允许：省略、倒装、感叹词（嗯/欸/哦）、破折号、半句话]

{user_message}
```

`max_chars = persona.style.speech_max_chars ?? 40`

### 4.4 Assistant Prefill

`persona.style.prefill_openers`（YAML 配置）：

```yaml
style:
  prefill_openers:
    - "嗯"
    - "欸"
    - "哦"
    - "诶"
    - ""    # 不预填，留给 LLM 自由发挥的概率
```

运行时按均匀分布随机挑一个。空串表示不预填。

Anthropic API 的 prefill 通过 `messages` 末尾追加 `{"role": "assistant", "content": prefill}` 实现。解析输出时 `final_text = prefill + response.text`。

### 4.5 Sampler

Persona YAML：

```yaml
sampling:
  temperature: 1.0    # 初始从默认起
  top_p: 0.95
```

Phase 0 要并行跑几档（0.7 / 0.9 / 1.0 / 1.1）对比。

### 4.6 Feishu 卡片

- 流式 speech 直接更新
- 完成时卡片底部加 `turn_id` + 三按钮：`👍 像` / `👎 不像` / `✏️ 应该说`
- `/reveal <turn_id>` 独立 DM 卡片展示 inner_thought

不做双通道时不需要"正在想…"过渡，延迟体验与现状一致。

---

## 5. 标注系统

### 5.1 入口

**Feishu**（主）：
- 卡片底部 3 按钮：
  - `👍 像` → source=positive 入池
  - `👎 不像` → 标记差评；30 分钟内接 `/bad <turn_id> <correction>` 补成 user_correction
  - `✏️ 应该说` → 弹 Feishu 表单 → 提交后 source=user_correction
- `/reveal <turn_id>` → 独立 DM 卡片展示 inner_thought（仅本 recipient 可见）

**CLI**：
- 每轮输出附 `turn_id`
- `:good` / `:bad <correction>` / `:reveal`

**Web API**：
- `POST /turns/<turn_id>/annotate` body `{kind, correction?}`
- `GET /turns/<turn_id>/inner_thought`

### 5.2 Collector

```python
class AnnotationCollector:
    def record_positive(self, turn_id: str) -> None: ...
    def record_negative(self, turn_id: str) -> None: ...          # 仅标记
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

- `inner_thought` 可能含对用户的敏感印象；`/reveal` 仅本 recipient 可见
- recipient 间隔离：correction 不跨 recipient，除非 recipient_key=None
- 防恶意标注：24h 内同用户 >N 次 correction 触发审核（Phase 4 后做）

---

## 6. 检索策略

### 6.1 Retriever

```python
def retrieve(query: str, recipient_key: str, k: int = 3) -> list[FewShotSample]:
    q = embedding.encode(query)

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

    filtered = [c for s, c in scored if s > 0.6]
    deduped = dedup_by_similarity(filtered, threshold=0.9)

    return deduped[:k]
```

### 6.2 查询文本

- 上一轮 Aria 的 `inner_thought`（若存在，最相关）
- 否则用 `user_msg`
- Phase 4 双通道上线后，改为 Call 1 当次输出的 inner_thought

### 6.3 拼装

```python
def assemble_fewshots(retrieved: list[FewShotSample]) -> list[FewShotSample]:
    seeds = fewshot_store.get_seeds(3)
    # seed 在前（基线），检索在后（最近位置，模仿信号最强）
    return seeds + retrieved
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
│   ├── engine.py                  # 改：注入 few-shot + style preamble + prefill
│   ├── prompt_assembly.py         # 新增：_build_messages 组 prior-turn
│   └── output_schema.py           # 现有，确保 inner_thought 字段保留
├── persona/
│   └── models.py                  # 新增 StyleConfig(prefill_openers, speech_max_chars...) + SamplingConfig
└── channels/
    └── feishu.py                  # 卡片底部标注按钮 + /reveal
```

**Phase 4（可选）追加**：

```
src/lingxi/conversation/
├── prompts/
│   ├── think.py                   # Call 1 prompt
│   └── compress.py                # Call 2 prompt
```

**数据目录**：

```
data/
├── fewshot/
│   ├── samples.jsonl
│   └── turns/<turn_id>.json
└── chroma/
    └── fewshot_pool_d2048/
```

**配置**（persona YAML 新增）：

```yaml
style:
  speech_max_chars: 40
  prefill_openers: ["嗯", "欸", "哦", ""]
  blacklist_phrases:                # 基础 20 条代码默认，这里是补充
    - "据说"
sampling:
  temperature: 1.0
  top_p: 0.95
compression:                        # Phase 4 才用
  enabled: false
  fewshot_seed_count: 3
  fewshot_retrieved_count: 3
  similarity_threshold: 0.6
```

---

## 8. 实施分期

### Phase 0: 单调用组合拳（0.5-1 天）

目标：验证行业标配杠杆能否 70-80% 解决 AI 腔

- [ ] `persona.style` / `persona.sampling` 字段扩展
- [ ] `prompt_assembly.py` 组 prior-turn messages
- [ ] `render_fewshots_as_messages` 实现
- [ ] Style preamble 注入到最后一条 user message 前
- [ ] Assistant prefill 从 `prefill_openers` 随机挑
- [ ] Sampler 参数接入 LLM 调用
- [ ] 临时硬编码 3-5 条 seed（待 Phase 1 seeds.yaml 落地后切换）
- [ ] CLI + Feishu 跑通，手验 before/after

**验收**：10 条同话题对比，AI 腔残留显著下降（主观评估 + 套话命中率）

### Phase 1: Seeds + AnnotationTurn 基座（0.5 天）

- [ ] `seeds.yaml` 10 条初稿，场景：共鸣/安慰/吐槽/好奇追问/轻度拒绝/懒得接话/生活琐事插入/主动分享/情绪低落/被戳中
- [ ] 用户审阅改
- [ ] `FewShotStore` + `AnnotationStore`
- [ ] `AnnotationTurn` 每轮自动记录 → JSON
- [ ] Seeds 入 ChromaDB（recipient_key=None）

### Phase 2: 标注 UI（1 天）

- [ ] Feishu 卡片底部 `👍` / `👎` / `✏️` 按钮 + action callback
- [ ] `AnnotationCollector` 处理三种事件
- [ ] `summarizer.py`（LLM 辅助 context_summary / tags）
- [ ] `/reveal <turn_id>` Feishu 命令
- [ ] CLI `:good` / `:bad <correction>` / `:reveal`
- [ ] Web API 两个端点

**验收**：产出 ≥5 条 user_correction 入池

### Phase 3: 动态检索注入（0.5 天）

- [ ] `FewShotRetriever.retrieve()`（ChromaDB 查询 + 重排 + 去重）
- [ ] Engine 组 prompt 时动态拉 few-shot 替换 Phase 0 硬编码
- [ ] 观察 `/bad` 率
- [ ] 上一轮 inner_thought 优先作为 query 的路径

**验收**：池子累 20+ 样本后，同类场景 `/bad` 率明显下降

### 决策点：Phase 0-3 是否够用？

评估指标：
- AI 腔主观感受（用户明确反馈"像了"或"还不像"）
- `/bad` 按钮触发频率（7 天窗口）
- 套话命中率（黑名单短语出现次数/轮数）

- **够用** → 到此为止，持续积累样本，后续只做 Phase 2/3 的迭代改进
- **不够** → 升级 Phase 4

### Phase 4（可选）: 双通道拆分（1-2 天）

触发条件：Phase 3 跑 7 天后 `/bad` 率无下降或主观体验仍差

- [ ] `prompts/think.py`, `prompts/compress.py`
- [ ] Engine 拆 `_think_turn` / `_compress_turn`
- [ ] few-shot 注入点从 prior-turn 区移到 Call 2 prompt
- [ ] `persona.compression.enabled` 开关
- [ ] Feishu 阶段卡片："Aria 正在想…" → 流式 speech（盖住延迟）

**验收**：双通道后 `/bad` 率显著低于单通道

### Phase 5（可选）: 模型成本优化

- [ ] Call 2 实验 Haiku 能否保人设一致性
- [ ] Prompt caching 命中率优化

---

## 9. 风险与缓解

| 风险 | 可能性 | 影响 | 缓解 |
|---|---|---|---|
| Phase 0 单调用不够用 | 中 | 需升 Phase 4 | 架构上完全兼容，决策点明确 |
| Few-shot 样本过少时检索效果差 | 高（短期） | Phase 3 效果弱 | Seeds 3 条兜底始终注入 |
| Prior-turn few-shot 被对话历史挤掉 | 低 | 模仿信号衰减 | few-shot 放最前且标注为"示例"；必要时加 depth 注入强化 |
| Prefill 选错导致说一半走样 | 中 | 单轮体验差 | openers 中留空串兜底；失败样本喂进标注 |
| inner_thought 泄漏隐私 | 中 | 用户不适 | `/reveal` 限本 recipient |
| 标注疲劳 | 高 | 自矫正失效 | 按钮轻量；positive 也算数；不强求 correction |
| Sampler 调太高导致胡言 | 中 | 质量抖动 | persona YAML 锁定值 + 冒烟测试 |

---

## 10. 开放问题（不阻塞实施）

1. Phase 3 的查询文本用 user_msg 还是上一轮 inner_thought？先按 §6.2 优先 inner_thought，实测再调
2. Seeds 随 persona 变？先通用，后续 persona YAML 支持 override 路径
3. Phase 4 Call 2 能否换 Haiku？Phase 5 实验
4. 标注数据能否做离线 eval（统计 AI 腔出现频率）？Phase 3 后加一个 eval 脚本

---

## 11. 非目标

- 不做 fine-tuning（工程量 + Claude 闭源）
- 不做跨 persona 样本共享
- 不改 memory / entity / agenda 现有架构
- 不做多候选 + ranker（swipes 风格）
