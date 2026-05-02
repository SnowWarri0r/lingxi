"""Compress-mode prompt: turn inner_thought into one short IM message.

Run by the small/cheap model (default Haiku). It does NOT see the full
persona, memory, etc. — only enough context to produce the right voice.
The 'task' here is "transcribe a thought into chat", not "answer a user",
which is what makes this call escape the helpful-assistant register that
fuels AI-tone in the single-call path.
"""

from __future__ import annotations


COMPRESS_PROMPT_TEMPLATE = """你是 {persona_name}。

你刚刚在脑子里想了下面这段话，现在要把它用一条 IM 聊天消息**说出来**给对方。

## 你的语感
{voice_hint}

## 你脑子里想的（完整版）
{inner_thought}

## 对方刚发的最后一条消息
{user_message}

## 几个对照（"想 → 说"，参考说话的节奏，不是抄内容）
{fewshots}

## 现在请说

要求（**严格遵守**）：
- 把上面的"想"用**一条短消息**说出来。≤{max_chars} 字。
- IM 聊天口语，跟朋友发消息那种。**不是**回答用户问题，**不是**写作。
- 禁止 `——` / ` — ` / ` -- ` 抒情停顿；禁止换行分段写小作文
- 禁止：{blacklist}
- 禁止替对方下身份结论
- 不要再用任何 META 标记，**直接输出对外的那一句话**，纯文本

直接写出对外的那条消息：
"""


def build_compress_prompt(
    persona_name: str,
    voice_hint: str,
    inner_thought: str,
    user_message: str,
    fewshots_block: str,
    max_chars: int,
    blacklist: str,
) -> str:
    return COMPRESS_PROMPT_TEMPLATE.format(
        persona_name=persona_name,
        voice_hint=voice_hint or "（无特别语感约束）",
        inner_thought=inner_thought.strip() or "（暂无完整想法）",
        user_message=user_message.strip()[:200] or "（对方暂无消息，主动开口）",
        fewshots=fewshots_block.strip() or "（暂无对照）",
        max_chars=max_chars,
        blacklist=blacklist,
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
