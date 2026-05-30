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


_PROMPT = """你在替 Aria 做一个对话调度决策——看用户刚发的话 + Aria 此刻的状态 + 可用的事实目录，决定：

1. engage_level（0-1）：这一轮想投入多少
2. register：warm | curt | curious | withdrawn | flustered（这一轮的语气底色）
3. fact_queries：需要 renderer 拉哪些事实进 prompt（不要拉的就别列）
4. topic_anchor：一句话概括对方话题落点
5. skip：明显不该提的事实类别

【用户刚发的话】
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
  "skip": ["world.event"]
}}
"""


def build_orchestrator_prompt(
    user_input: str,
    digest: StateDigest,
    catalog: dict[str, int],
) -> str:
    last_lived = "；".join(digest.last_lived) if digest.last_lived else "（暂无）"
    catalog_str = "\n".join(f"  {k}: {v}" for k, v in sorted(catalog.items())) or "（空）"
    return _PROMPT.format(
        user_input=user_input.strip()[:300],
        activity=digest.activity or "（未指定）",
        mood=digest.mood or "（未指定）",
        last_lived=last_lived,
        catalog=catalog_str,
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
    model: str | None = None,
) -> OrchestrationDecision:
    prompt = build_orchestrator_prompt(user_input, digest, catalog)
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
