# 灵犀 / Lingxi

一个可定制人设的虚拟人格 agent。目标是做出**真实对话体验** —— 不是一个应答机器人，而是一个有自己生活、记忆、情绪和主观理解的"人"。

> 名字取自"心有灵犀"——无需言明，自然相通。

## 特性

**核心引擎**
- **可配置人设**：YAML 定义身份、性格、说话风格、情绪基线、关系等级
- **三层记忆**：短期缓冲（按对象持久化）/ 长期事实（ChromaDB + 语义检索）/ 情景回忆（session 摘要）
- **多维情绪**：情绪维度（喜悦、好奇、焦虑、温暖...）+ 强度 + 时间衰减
- **时间感知**：知道现在是几点、星期几、白天/深夜
- **关系自动演进**：根据对话深度和频次，关系等级自动升级

**内在生命**（让人设"活"起来）
- **LifeSimulator**：Aria 有她自己的一天，后台持续推进（活动、事件、日记）
- **SubjectiveLayer**：对每个对话对象的主观印象、担心、欣赏 —— 不是事实是感受
- **AgendaEngine**：她想跟你说的话，基于真实生活事件而不是定时触发
- **Reflection Loop**：空闲时她会反思最近的对话，生成见解存入长期记忆
- **跨用户记忆隔离**：每个对话对象独立记忆池，互不串线

**结构化输出**
- LLM 输出 = **纯对白 + JSON 元数据**
- 不同 channel 通过 adapter 提取所需维度（文本 / 语音 / 表情 / 动作）
- 飞书取对白，未来 TTS 取 mood 调语调，未来 Live2D 取表情/动作

**Channel**
- **飞书机器人**（WebSocket 长连接，流式卡片，图片理解，/ 命令）
- **Web API**（REST + WebSocket 流式）
- **CLI**

**认证**
- OAuth PKCE / Device Code（同 Claude Code / Codex CLI 方式）
- 自动读取本机 Claude Code keychain / Codex auth.json
- 401 自动刷新 token
- API key / 环境变量 fallback

## 快速开始

```bash
# 1. 安装
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[feishu,embeddings]"

# 2. 设置 API key
cp .env.example .env
# 编辑 .env 填入 ANTHROPIC_API_KEY 或 ARK_API_KEY（豆包 embedding）

# 3. 人设（或直接用 example_persona.yaml）
cat config/personas/example_persona.yaml

# 4. 启动
# CLI 交互
lingxi

# Web API
lingxi-server

# 飞书机器人（需要 FEISHU_APP_ID / FEISHU_APP_SECRET）
lingxi-feishu
```

## 架构

```
src/lingxi/
├── persona/         # 人设定义 + Pydantic 模型 + prompt builder
├── memory/          # 三层记忆 (short/long/episodic) + ChromaDB + 实体图
├── inner_life/      # LifeSimulator + Agenda + SubjectiveLayer（内在生命）
├── temporal/        # 时间感知 + 关系演进 + proactive + reflection
├── planning/        # 目标 + 规划 + 主动行为
├── conversation/    # 对话引擎 + 结构化输出 + adapters
├── providers/       # LLM / Embedding 抽象（Claude / OpenAI / 豆包 / 本地）
├── channels/        # 飞书 / Web / CLI channel
├── auth/            # OAuth + profile store + external sync
└── web/             # FastAPI + WebSocket
```

## 许可

[AGPL-3.0-or-later](LICENSE)

简单说：你可以自由使用、修改。如果把 Lingxi（包括修改版）作为网络服务提供给他人（SaaS/托管），也必须以 AGPL 开源你的整个服务端源码。商业用途如需闭源授权请联系作者。
