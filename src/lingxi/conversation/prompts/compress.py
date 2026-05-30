"""Compress-mode prompt: turn inner_thought into one short IM message.

Run by the small/cheap model (default Haiku). It does NOT see the full
persona, memory, etc. — only enough context to produce the right voice.
The 'task' here is "transcribe a thought into chat", not "answer a user",
which is what makes this call escape the helpful-assistant register that
fuels AI-tone in the single-call path.
"""

from __future__ import annotations


COMPRESS_PROMPT_TEMPLATE = """你是 {persona_name}。你刚才在脑子里想了下面这段话，现在用一条 IM 消息说出来给对方。**任务是把"想"翻译成"说"**，不是回答用户、不是写作。

## 你的语感
{voice_hint}

## 你上一句对外说的话（重要——对方下面那句话大概率是在回应它）
{previous_assistant_msg}

## 你刚才想的
{inner_thought}

## 对方上一条
{user_message}

## 几条对照（看节奏，别抄内容）
{fewshots}

## 输出

≤{max_chars} 字，IM 短句，直接说，纯文本，单行（无 META、无段落分隔、无破折号）。

要点：
- 开头是真正的回应，不是复述对方的话
- 讲自己的事只在 inner_thought 里真的有时讲（"我也有过/让我想起" 这类带自己的话也得是真的有）
- 被批评轻应一句换种方式重说（"对" / "嗯…" 一类即可）
- 沉重话题（生病/丧亲/失业/分手）≤25 字短句陪着

**外部话只基于对方真的写过的字**。inner_thought 是脑内推测——对方没写"累"就不假设他累，没写"简单聊"就不假设他要简单聊。脑内的留在脑内。

**对方短句默认接在你上一句的语境里**。短句/模糊回复（『给我吃』『我也是』『懂』『好』『真的吗』）多数是在回应你刚说的内容，不是孤立新话题：
- 你刚说『泡面加蛋好香』+ 对方『给我吃』→ 他想分一点，接 `给你一口 / 哈哈来一勺`
- 你刚说『今天好烦』+ 对方『嗯』→ 他在陪你听，接 `就那种说不清的 / 谢啦你听我念叨`
- 你刚说『下雨了』+ 对方『带伞了』→ 他在报告状态，接 `那好 / 那放心了`

直接写那条消息：
"""


def build_compress_prompt(
    persona_name: str,
    voice_hint: str,
    inner_thought: str,
    user_message: str,
    fewshots_block: str,
    max_chars: int,
    previous_assistant_msg: str = "",
) -> str:
    return COMPRESS_PROMPT_TEMPLATE.format(
        persona_name=persona_name,
        voice_hint=voice_hint or "（无特别语感约束）",
        previous_assistant_msg=previous_assistant_msg.strip()[:200]
            or "（无——可能这是新一轮的第一条）",
        inner_thought=inner_thought.strip() or "（暂无完整想法）",
        user_message=user_message.strip()[:200] or "（对方暂无消息，主动开口）",
        fewshots=fewshots_block.strip() or "（暂无对照）",
        max_chars=max_chars,
    )


def render_fewshots_for_compress(samples: list) -> str:
    """Render a few-shot list as 'context → speech' text block (not message pairs)."""
    if not samples:
        return ""
    lines = []
    for s in samples:
        ctx = getattr(s, "context_summary", "")[:60]
        speech = getattr(s, "corrected_speech", "")
        lines.append(f"- 想：{ctx} → 说：{speech}")
    return "\n".join(lines)
