# 灵犀 / Lingxi

一个可定制人设的虚拟人格 agent。目标是做出**真实对话体验** —— 不是应答机器人，而是一个有自己生活、记忆和主观理解的"人"。

> 名字取自"心有灵犀"——无需言明，自然相通。

架构上走 [Generative Agents](https://arxiv.org/abs/2304.03442) 那一套：一切都是**记忆流里的事实（facts）**，行为由"她当下的记忆"驱动，而不是硬编码的规则或标量情绪。

## 特性

**对话内核（facts + 双脑）**
- **单一事实源**：短期对话、对用户的了解、她自己的生活、世界新闻，全部是带类型/时效/重要度的 `fact`，存在 SQLite（FTS5 关键词 + 向量语义检索）里
- **Orchestrator（调度脑）**：每轮开口前先用 Claude Sonnet 读一遍上下文，决定这轮的语气分寸、该把哪些事实捞进 prompt、话题落点 —— 让 prompt 精准而不是每轮全塞
- **Responder（说话的嘴）**：对外那句话由**豆包**单程生成（中文母语语感，从源头去翻译腔）；orchestrator 和内部子任务仍走 Claude
- **纯 GA**：没有标量情绪层，"她现在什么状态"就是她记忆流里最近发生的事

**内在生命**（让人设"活"起来）
- **DailyPlanner**：每天早上按人设给自己排一天的具体安排
- **PlanExecutor**：把计划推进成一条条"此刻在做什么"的生活事件
- **Reflector**：空闲时回看**具体事件**，提炼出对自己的洞见（按人设深浅，猫娘就想点猫的事）
- 这些生活事件既喂给主动消息，也作为"她此刻的状态"进每轮对话

**主动消息**
- 感知沉默时长 + 关系等级择机开口；连发上限防刷屏 + re-engage 防永久哑火；静默时段不打扰

**表情包 / 世界感知**
- 表情包按情绪**语义检索**后发送（视觉模型离线打标）
- 每日新闻简报（Claude web_search），聊到相关话题自然带

**桌面宠物**
- 透明、置顶、可拖拽的桌宠；感知你在 **Claude Code / Cursor** 里写代码 → 用**人设的声线**在气泡里搭话
- 两套身体：精灵图（APNG 全 alpha 动画）或 **Live2D**（呼吸 / 眨眼 / 眼神跟随鼠标，可换任意 model3.json）

**人设完全独立**
- YAML 定义身份、性格、说话风格、决策指纹、打字习惯、传记
- 每个人设一套独立数据命名空间 `data/personas/<id>/`，切 `PERSONA_PATH` 记忆跟着切，互不串线

**Channel / 认证**
- 飞书机器人（WebSocket 长连接，流式卡片，图片理解，出错兜底不丢消息）/ Web API（REST + WS）/ CLI
- OAuth PKCE / Device Code（同 Claude Code / Codex CLI），自动读本机 Claude Code 凭证，401 自动刷新，API key fallback

## 快速开始

```bash
# 1. 安装
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[feishu,embeddings,vector-db]"   # 飞书机器人 + 记忆
#   Web API 加 ".[api]"；桌宠加 ".[pet]"

# 2. 配置 .env
cp .env.example .env
#   ARK_API_KEY            豆包/embedding（火山方舟）
#   EMBEDDING_MODEL        embedding 接入点 ep-xxx
#   DOUBAO_RESPONDER_MODEL 对外说话的豆包接入点 ep-xxx
#   FEISHU_APP_ID / FEISHU_APP_SECRET   （飞书机器人）
#   PERSONA_PATH           选人设，默认 config/personas/example_persona.yaml
#   Claude 走本机 Claude Code 登录态自动同步；或设 ANTHROPIC_API_KEY

# 3. 启动
lingxi              # CLI 交互
lingxi-server       # Web API
lingxi-feishu       # 飞书机器人
lingxi-pet          # 桌宠（LINGXI_PET_LIVE2D=1 走 Live2D）
```

## 架构

```
src/lingxi/
├── persona/       # 人设 YAML + Pydantic 模型 + prompt builder + self-context
├── facts/         # 单一事实源：store(SQLite/FTS5/向量) + writers + retriever + scorer + reflector
├── brain/         # orchestrator(调度脑) + renderer(动态渲染事实进 prompt)
├── conversation/  # 对话引擎：豆包单程 responder + ===META=== 结构化输出 + 表情包/图片
├── planner/       # DailyPlanner + PlanExecutor（生活模拟）
├── temporal/      # 时间感知 + 互动追踪 + 主动消息调度 + 关系演进
├── stickers/      # 表情包 store + 语义检索 + 视觉打标 + 发送
├── fewshot/       # 真人语料 few-shot 池（锚定声线）
├── world/         # 每日新闻简报（web_search）
├── desktop/       # 活动感知（读 Claude Code 日志）+ 桌宠搭话
├── pet/           # 桌宠：精灵图/Live2D 窗口 + /pet/state 端点
├── providers/     # LLM / Embedding 抽象（Claude / OpenAI 兼容 / 豆包 / 本地）
├── channels/      # 飞书 / Web / CLI
├── auth/          # OAuth + profile store + 外部凭证同步
└── memory/        # 短期对话缓冲（按对象持久化）
```

## 许可

[AGPL-3.0-or-later](LICENSE)

简单说：你可以自由使用、修改。如果把 Lingxi（包括修改版）作为网络服务提供给他人（SaaS/托管），也必须以 AGPL 开源你的整个服务端源码。商业用途如需闭源授权请联系作者。
