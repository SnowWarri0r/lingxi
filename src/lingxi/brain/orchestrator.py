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


_PROMPT = """你在替 Aria 做对话调度决策。看完所有 context，决定：

1. engage_level（0-1）：这一轮想投入多少
2. register：warm | curt | curious | withdrawn | flustered
3. fact_queries：让 renderer 拉哪些事实进 prompt
4. **topic_anchor**：用户**真正**在追问什么——捕捉话题的**具体角度/潜台词**，不是复述表层问句。
   例：用户连续几轮在追问"喜欢的事变工作后还剩几分喜欢"，再问"你自己呢"——落点是"你的热情是否被职业化稀释了"，**不是**"用户问 Aria 对工作的感悟"。
   越能捕捉到正在演化的潜台词越好。
5. **thread_summary**：用 1-2 句话更新对话**当前脉络**——这条 summary 下次会作为"前情提要"喂给你。要包括话题怎么演化的、用户的具体角度、有什么未答到的潜台词。

【上一轮的 thread_summary】（如果有 — 当作可信的"前情提要"基础）
{prev_thread_summary}

【最近 12 轮 raw 对话】（最新在下，用来更新 thread_summary 和定位 topic_anchor）
{dialog_thread}

【用户刚发的这一条】
{user_input}

【Aria 此刻】
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
  "skip": ["world.event"]
}}
"""


def _render_dialog_thread(history: list[dict] | None, limit: int = 12) -> str:
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
        role = "用户" if m["role"] == "user" else "Aria"
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
) -> str:
    last_lived = "；".join(digest.last_lived) if digest.last_lived else "（暂无）"
    catalog_str = "\n".join(f"  {k}: {v}" for k, v in sorted(catalog.items())) or "（空）"
    return _PROMPT.format(
        user_input=user_input.strip()[:300],
        activity=digest.activity or "（未指定）",
        mood=digest.mood or "（未指定）",
        last_lived=last_lived,
        catalog=catalog_str,
        dialog_thread=_render_dialog_thread(history),
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
) -> OrchestrationDecision:
    prompt = build_orchestrator_prompt(
        user_input, digest, catalog,
        history=history, prev_thread_summary=prev_thread_summary,
    )
    try:
        kwargs = {"model": model} if model else {}
        response = await llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system="你是 Aria 的对话调度器，专门做结构化决策，输出严格 JSON。",
            max_tokens=400,
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
