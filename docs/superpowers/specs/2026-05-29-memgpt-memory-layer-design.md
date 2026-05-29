# MemGPT 记忆层设计(SP1:tool-use 地基 + 核心记忆 + 主动检索)

> 2026-05-29 · brainstorm 产出 · 建在纯 GA 回路之上([2026-05-29-facts-single-source-decommission-design.md](./2026-05-29-facts-single-source-decommission-design.md))

**Goal:** 在现有"orchestrator 预检索 → renderer → 生成"的 GA 回路上，叠加 MemGPT 的"agent 当自己内存 OS"能力——**混合模式**:orchestrator 预热 prompt，agent 在 turn 内按需用 function call 补抓记忆 / 自编辑核心记忆。

**范围:** 本 spec 只做 SP1 = tool-use 地基 + 多步 agentic loop + 4+1 个记忆工具 + 核心记忆块（交付:agent 主动检索 + 自编辑核心记忆 + 分层 context）。**SP2(context 压力管理:token 计账 + 摘要换页)单独立项**，建在 SP1 上、复用已有 `compress_aged_turns`。

**Tech Stack:** Anthropic 原生 tool use（直接 HTTP，非 SDK）、facts.db(SQLite+FTS5)、pydantic。

---

## 1. 分层 context（MemGPT 三层 → 我们的映射）

| MemGPT 层 | 我们的实现 | 可变性 |
|---|---|---|
| main context（in-prompt） | persona 静态块 + **【核心记忆】块** + 预热 facts 块 + 对话历史 | 核心记忆块 agent 自编辑；facts 块每轮检索 |
| archival（外部） | facts.db 记忆流 | agent `archival_insert` 写、`archival_search` 读 |
| recall（外部） | short_term 此人对话缓冲（近期 turn + 摘要） | `conversation_search` 读 |

**核心记忆 vs facts 块分工:** 核心记忆 = 常驻、agent 亲手维护的**有界小结**；facts 块 = 每轮**检索**来的无界细节。两者都在 main context，来源/可变性不同。

## 2. Turn loop（混合）

```
1. 预热(沿用): Orchestrator(catalog+对话) → decision → Renderer 拉 facts
   主上下文 = persona 静态块
            + 【核心记忆】<persona>(subject=aria) + <human>(subject=user:此人)
            + 【你此刻/你和他/身边的事】预热 facts 块
            + 对话历史(short_term)
2. 带 tools 生成: tools = [archival_memory_search, archival_memory_insert,
   core_memory_append, core_memory_replace, conversation_search]
3. agentic loop(见 §5)
4. 终局文本回复用现有 ===META=== 格式（不引入 send_message 工具），
   按 stop_reason 区分:tool_use=记忆步，end_turn=终局
```

成本:多数 turn 预热够 → agent 0 工具调用 → 1 次 LLM（与现状持平）；不够时才多跳。

## 3. 核心记忆块存储（仍落 facts.db，唯一源不破）

- 新增 `FactType.CORE`。一个核心块 = 一条 CORE fact:`<persona>` 块 subject=`aria`，`<human>` 块 subject=`user:<recipient_key>`。
- **编辑即 supersede**:append/replace 写一条新 CORE fact，`supersedes` 指向旧的。"当前块" = 该 subject 最新未被 supersede 的 CORE fact。版本链可回溯。
- **有界**:块内容上限 `CORE_BLOCK_MAX_CHARS = 1500`。append 后超限 → 工具返错 `"core memory full, use core_memory_replace to condense"`，逼 agent 自己腾地方。
- Renderer 新增【核心记忆】段:渲染 `get_core_block("aria")` + `get_core_block(f"user:{recipient_key}")`，空块跳过。
- 新增 `FactRetriever.get_core_block(subject) -> Fact | None`（store 层:取该 subject 最新未 supersede 的 CORE fact）。
- 写核心块经新增 `CoreMemoryWriter`(WriterBase 子类，ALLOWED_SOURCE=Source.LLM_INFERRED，SUBJECT_PATTERN 允许 `aria` 或 `user:*`)，复用 supersede + 不变量校验。

## 4. 工具契约

所有工具在 engine 内 dispatch，按当前 `recipient_key` 把 `scope` 映射成 subject——**agent 永远拿不到"写任意 subject"的能力**，subject-ownership 不变量在工具层继续成立。每个工具 try/except，出错返错误串（agent 自愈）。

```
archival_memory_search(query: str, scope: "self"|"user"|"world" = "user")
  → FactRetriever.fetch 三维检索；scope→subject(self=aria / user=user:此人 / world=npc:*+world)
  → 返回 top-5 [{ts, content, importance}]（只读）

archival_memory_insert(content: str, scope: "self"|"user" = "user", importance: int|None = None)
  → 写新 fact；subject 由 scope 映射(self=aria / user=user:此人)；source=LLM_INFERRED；type=PATTERN；缺 importance 走 scorer
  → 经支持 LLM_INFERRED 源的 writer 按 subject 校验落库
    (复用 inference_writer；若其 SUBJECT_PATTERN 不覆盖 aria+user:*，plan 阶段放宽或加一个 archival writer)
  → 返回 "inserted"

core_memory_append(block: "persona"|"human", content: str)
  → block→subject(persona=aria / human=user:此人)；取当前块 + "\n" + content，supersede 写新版
  → 超 CORE_BLOCK_MAX_CHARS 返错逼 replace；否则返 "ok"

core_memory_replace(block: "persona"|"human", old: str, new: str)
  → 块内子串替换，supersede 写新版；old 找不到返错 "substring not found"

conversation_search(query: str)
  → 搜 recall = 此人 short_term 快照(snapshot_for_recipient) 的近期 turn+摘要(子串/关键词)
  → 返回命中 [{ts, role, content}]
  → SP1 限 short_term 窗口；全量历史 recall 持久化 = SP2
```

## 5. Provider 改动 + loop 终止/防失控

**providers/claude.py**
- `_build_body(...)` 加可选 `tools: list[dict] | None`、`tool_choice: dict | None`，有则塞进 body。
- `complete(...)` 加 `tools`/`tool_choice` 参数（透传到 `_build_body`）；解析 content 里 `tool_use` block。
- `CompletionResult` 加字段:`tool_calls: list[dict]`(每项 `{id, name, input}`)、`raw_content_blocks: list[dict]`(原始 content blocks，回灌 assistant turn 用)。`finish_reason` 已承载 stop_reason。
- messages 已支持 block 列表;tool_result 走 `{"type":"tool_result","tool_use_id":..,"content":..}`。

**engine agentic loop**(新增 `_generate_with_tools`，替换反应式主生成那次 `complete`)
```python
MAX_TOOL_ITERS = 5
iters = 0
while True:
    tc = {"type": "auto"} if iters < MAX_TOOL_ITERS else {"type": "none"}
    r = await self.llm.complete(messages=msgs, system=sys,
                                tools=MEMORY_TOOLS, tool_choice=tc, ...)
    if r.finish_reason != "tool_use" or not r.tool_calls:
        return r                                   # 终局文本回复
    msgs.append({"role": "assistant", "content": r.raw_content_blocks})
    results = []
    for call in r.tool_calls:
        out = await self._dispatch_memory_tool(call["name"], call["input"], recipient_key)
        results.append({"type": "tool_result", "tool_use_id": call["id"], "content": out})
    msgs.append({"role": "user", "content": results})
    iters += 1
```
- 防失控:`MAX_TOOL_ITERS` 上限 + 到顶 `tool_choice={"type":"none"}` 强制出文本。
- think→compress 两段不变:tool loop 是主生成；compress(若开)只缩最终文本、不带 tools。
- 工具仅反应式 chat 启用；proactive 维持现状（不带工具）。

## 6. 接入点

- **Orchestrator**:不动。
- **Renderer**:加【核心记忆】段。
- **engine**:`_prepare_turn_v2` 末尾照旧返回 (system_prompt, messages)；主生成改调 `_generate_with_tools`；新增 `_dispatch_memory_tool` + `MEMORY_TOOLS` 定义。
- **facts**:`FactType.CORE`、`store.get_core_block`、`retriever.get_core_block`、`CoreMemoryWriter`。
- **app.py**:构造 `CoreMemoryWriter` 注入 engine。

## 7. 测试

- provider:`_build_body` 含 tools；`complete()` 解析 tool_use → `tool_calls` + `raw_content_blocks` + finish_reason="tool_use"。
- 工具单测:search 返 facts 且 scope→subject 正确；insert 按 scope 写对 subject + 不能越权写；core append/replace 正确 supersede + 超限返错 + old 找不到返错;conversation_search 扫 short_term 命中。
- `get_core_block`:多次 supersede 后取最新；无块返 None。
- loop:mock LLM 先 tool_use 再 end_turn → 断言工具执行 + tool_result 回灌 + 终局返回；失控测试(恒 tool_use → 到 MAX 停 + 强制文本)。
- renderer:核心记忆块渲染 + 空块跳过。

## 8. 失败处理

- 工具异常 → 返错误串作 tool_result（agent 自愈，不崩 turn）。
- loop 到顶 → 强制文本，不无限循环。
- provider 解析对缺失/混合 block 容错（无 tool_use 时 tool_calls=[]）。

## 9. Out of scope（SP2 / follow-up）

- context 压力管理（token 计账 + 超阈值递归摘要 + 换页到 recall）。
- 全量对话历史的 recall 持久化（当前 conversation_search 只覆盖 short_term 窗口）。
- proactive 路径接入工具。
- 让 archival_insert 完全取代 writer-from-output（SP1 两者并存，output 路径作兜底）。
