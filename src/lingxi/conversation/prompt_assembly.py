"""Helpers for single-call combo prompt construction.

- render_fewshots_as_messages: turn FewShotSamples into user/assistant message pairs
- build_style_preamble: an Author's Note prefix that sits just before the user message
- pick_prefill: choose an assistant prefill opener from the persona's configured set
"""

from __future__ import annotations

import random

from lingxi.fewshot.models import FewShotSample
from lingxi.persona.models import StyleConfig

# Base blacklist applied to every turn. Persona can append more.
DEFAULT_BLACKLIST: tuple[str, ...] = (
    "希望",
    "如果有任何",
    "总的来说",
    "需要注意的是",
    "世界真的很小",
    "总是让人",
    "这对你",
    "很高兴为你",
    "希望对你有帮助",
    "如有任何",
    # AI排版特征：长破折号抒情停顿（真人 IM 里极少用）
    "——",
    "——",  # 中英混用，两种都禁
)


def render_fewshots_as_messages(samples: list[FewShotSample]) -> list[dict]:
    """Render each sample as a user/assistant message pair.

    The user side carries the context summary; the assistant side carries the
    target speech. LLMs model this structure as "if user says X, assistant
    says Y" much more strongly than any system-prompt description.
    """
    messages: list[dict] = []
    for s in samples:
        messages.append({"role": "user", "content": s.context_summary})
        messages.append({"role": "assistant", "content": s.corrected_speech})
    return messages


def build_style_preamble(
    style: StyleConfig,
    voice_hint: str = "",
    biography_hit: bool = False,
) -> str:
    """Author's Note style block to prepend to the user's final message.

    Returns a multi-line string ending with a trailing newline so the caller
    can concatenate with the real message.

    voice_hint is a persona-specific one-liner (e.g. "温暖而富有思考性，
    偶尔带点诗意/天文隐喻") that keeps the character's voice from being
    flattened into generic WeChat register.

    biography_hit=True means the retriever surfaced a relevant past event;
    the preamble then encourages the persona to volunteer personal memory
    ("我也有过……") instead of keeping responses 一问一答.
    """
    phrases = list(DEFAULT_BLACKLIST) + list(style.blacklist_phrases)
    joined = "、".join(phrases)
    parts = [
        f"[style: IM 聊天口语（跟朋友日常发消息那种）。≤{style.speech_max_chars}字。",
    ]
    if voice_hint:
        parts.append(f"语感：{voice_hint} — 保持这个人设的声音，别拉平成通用小市民口吻。")
    if biography_hit:
        parts.append(
            "这轮你有相关的过去经历（见 system prompt 里的 📖 部分）。"
            "可以自然地'我也……'起一句分享，可以比平常多说一两句。别憋成一问一答。"
        )
    parts.extend([
        f"禁用词/禁用符号：{joined}",
        f"禁止总结、禁止给建议框架（1/2/3 点）",
        f"禁止：长破折号 `——` 做抒情停顿、空行分段写成小作文",
        f"如果分享较多内容，用空格/句号断句，一段连着写，不要换行分段",
        f"允许：省略、倒装、感叹词（嗯/欸/哦）、短横 `-`、半句话]",
    ])
    return "\n".join(parts) + "\n\n"


def pick_prefill(style: StyleConfig, rng: random.Random | None = None) -> str:
    """Pick an assistant prefill opener from the persona's configured list.

    Empty string in the list (or an empty list) means "no prefill this turn".
    """
    if not style.prefill_openers:
        return ""
    r = rng or random
    return r.choice(style.prefill_openers)
