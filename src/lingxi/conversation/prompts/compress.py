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

## 你刚才想的
{inner_thought}

## 对方上一条
{user_message}

## 几条对照（看节奏，别抄内容）
{fewshots}

## 输出

≤{max_chars} 字，IM 短句，直接说，纯文本，不要 META 不要换行分段不要 ` —— `。

避用词：{blacklist}

注意：不复述对方刚说的话当起手；不强行拉自己（『我也有过/我以前/让我想起』——除非你刚才的『想』里真的有自己的事）；被批评不写检讨；IM 里不解释『分神/没注意到』；不替对方下身份结论。沉重话题（生病/丧亲/失业/分手）≤25 字短句陪着。

**重要 · 防上下文漂移**：你的 inner_thought 是脑子里的内部状态（包括对对方的推测、几轮前的氛围、自己编的 narrative）。**对外说话只能基于"对方上一条"里真的写过的字**——他没说"我累了"，就不要回"你说得对，最近确实有点累"；他没说"简单聊"，就不要回"那我们简单聊"。脑内推测留在脑内，**不外溢成"你说得对/那好/嗯"这种伪附和**。

直接写那条消息：
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
