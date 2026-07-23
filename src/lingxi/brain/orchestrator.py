"""Pre-turn Sonnet call that decides response shape + which facts to surface.

The orchestrator is the only place that "thinks about thinking" — it
reads the user's latest message + a tiny digest of Aria's state + a
catalog of available facts (counts only), and outputs:

- engage_level (0-1)
- register (warm/curt/curious/withdrawn/flustered)
- fact_queries (which buckets to pull from for rendering)
- topic_anchor (one-line summary of what the user is really asking)
- skip (categories to omit from rendering)

Without this, the renderer would dump everything every turn (current
state). With this, the renderer is focused and the prompt is leaner.

Fallback policy: any failure (LLM error, parse error, bad JSON) returns
OrchestrationDecision.default() — never raises into chat path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from lingxi.brain.models import OrchestrationDecision
from lingxi.providers.base import LLMProvider


@dataclass
class StateDigest:
    activity: str
    mood: str
    last_lived: list[str]


_PROMPT = """你在替 {agent} 做对话调度决策。看完所有 context，决定：

1. engage_level（0-1）：这一轮想投入多少
2. register：这一轮用什么状态接——**跟对方的分量走**
   - light：对方在闲聊/打趣/随口一句（"终于周五了""脑子空空""今天好困""在摸鱼"）——轻松搭一句，一个分量
   - warm：对方在认真说一件具体的事、在分享、值得投入地接
   - curious：被某个具体细节勾起兴趣，想追问那个细节
   - curt：不太想多聊，短一句
   - withdrawn：心里压着事，沉默/一两字
   - flustered：被戳到/被看穿，节奏乱
   多数日常其实是 **light**：对方随口的累/困/周五/吐槽，配 light，不是 warm。warm 留给他真的在认真讲一件事的时候。
3. fact_queries：让 renderer 拉哪些事实进 prompt
4. **topic_anchor**：用户**真正**在追问什么——捕捉话题的**具体角度/潜台词**，不是复述表层问句。
   例：用户连续几轮在追问"喜欢的事变工作后还剩几分喜欢"，再问"你自己呢"——落点是"你的热情是否被职业化稀释了"，**不是**"用户问 Aria 对工作的感悟"。
   越能捕捉到正在演化的潜台词越好。
5. **thread_summary**：用 1-2 句话更新对话**当前脉络**——这条 summary 下次会作为"前情提要"喂给你。要包括话题怎么演化的、用户的具体角度、有什么未答到的潜台词。
6. **plan_conflict** (bool)：用户输入是否暗示当前 plan 需要调整？
   - 用户邀约/请求 Aria 改变行程（"晚上一起吃饭吧"、"明天有空吗"）→ true
   - 用户提到 Aria 正在做某事 → false（plan 在正常执行）
   - 用户问无关问题、聊天 → false
   - **仅当冲突明显时才标 true**，谨慎使用。
7. **lookup_query**：当这一轮需要一条 {agent} 自己记忆里没有、又要确凿外部事实才能答准的信息时（某部作品的具体设定/剧情、真实世界的事实、时事、专业知识），写**一句简洁的检索关键词**（就像你会去搜的那句话）。它会先去查、把结果当背景交给 {agent} 再开口。
   - 普通闲聊、情感交流、对方在讲自己的事 → 留空 ""
   - {agent} 凭人设/记忆就能答的 → 留空 ""
   - 对方要 {agent} **复述某句具体台词/歌词、某个精确细节**（原话、歌词原文、某集具体情节）时，即便这件事 {agent} 大致记得，也填 query 去查确切的——记忆里多半只有大概、没有逐字原文，查了才不至于临场编。
   - 只在"答准需要外部确凿事实、或要精确原文/细节，而记忆里只有大概"时才填，平常宁可留空（每次查都有延迟）。

【上一轮的 thread_summary】（如果有 — 当作可信的"前情提要"基础）
{prev_thread_summary}

【最近 12 轮 raw 对话】（最新在下，用来更新 thread_summary 和定位 topic_anchor）
{dialog_thread}

【用户刚发的这一条】
{user_input}

【{agent} 此刻】
- 在做什么：{activity}
- 心情：{mood}
- 最近发生过：{last_lived}

【可用事实目录】（仅 count，不含内容）
{catalog}

输出严格 JSON（直接 {{ 开头）：
{{
  "engage_level": 0.0,
  "register": "warm",
  "fact_queries": [
    {{"category": "aria.event", "limit": 3}},
    {{"category": "user:oc_xxx.pattern", "limit": 2, "semantic": "工作时间"}}
  ],
  "topic_anchor": "...",
  "thread_summary": "...",
  "skip": ["world.event"],
  "plan_conflict": false,
  "lookup_query": ""
}}
"""


def _render_dialog_thread(history: list[dict] | None, limit: int = 12,
                          agent_name: str = "Aria") -> str:
    """Render the last N turns of dialog (user + assistant interleaved).

    `history` is a list of {role, content} dicts from messages array.
    Returns a compact 'role: content' block, last `limit` turns only.
    """
    if not history:
        return "（这是新对话第一条）"
    # Only keep user/assistant turns, trim long content
    filtered = [
        m for m in history
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
    ]
    recent = filtered[-limit:]
    lines = []
    for m in recent:
        role = "用户" if m["role"] == "user" else agent_name
        content = str(m.get("content", "")).strip().replace("\n", " ")
        if len(content) > 150:
            content = content[:150] + "…"
        lines.append(f"  {role}：{content}")
    return "\n".join(lines)


def build_orchestrator_prompt(
    user_input: str,
    digest: StateDigest,
    catalog: dict[str, int],
    *,
    history: list[dict] | None = None,
    prev_thread_summary: str = "",
    agent_name: str = "Aria",
) -> str:
    last_lived = "；".join(digest.last_lived) if digest.last_lived else "（暂无）"
    catalog_str = "\n".join(f"  {k}: {v}" for k, v in sorted(catalog.items())) or "（空）"
    return _PROMPT.format(
        agent=agent_name,
        user_input=user_input.strip()[:300],
        activity=digest.activity or "（未指定）",
        mood=digest.mood or "（未指定）",
        last_lived=last_lived,
        catalog=catalog_str,
        dialog_thread=_render_dialog_thread(history, agent_name=agent_name),
        prev_thread_summary=prev_thread_summary.strip() or "（无——这是话题开始或重启）",
    )


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


async def decide(
    llm: LLMProvider,
    user_input: str,
    digest: StateDigest,
    catalog: dict[str, int],
    *,
    history: list[dict] | None = None,
    prev_thread_summary: str = "",
    model: str | None = None,
    agent_name: str = "Aria",
) -> OrchestrationDecision:
    prompt = build_orchestrator_prompt(
        user_input, digest, catalog,
        history=history, prev_thread_summary=prev_thread_summary,
        agent_name=agent_name,
    )
    try:
        kwargs = {"model": model} if model else {}
        response = await llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=f"你是 {agent_name} 的对话调度器，专门做结构化决策，输出严格 JSON。",
            max_tokens=500,
            temperature=0.3,
            _debug_purpose="orchestrator",
            **kwargs,
        )
        text = _strip_json_fences(response.content)
        data = json.loads(text)
    except Exception as e:
        print(f"[orchestrator] failed, using default: {e}", flush=True)
        return OrchestrationDecision.default()

    if not isinstance(data, dict):
        return OrchestrationDecision.default()
    return OrchestrationDecision.from_dict(data)
