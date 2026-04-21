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


def build_style_preamble(style: StyleConfig) -> str:
    """Author's Note style block to prepend to the user's final message.

    Returns a multi-line string ending with a trailing newline so the caller
    can concatenate with the real message.
    """
    phrases = list(DEFAULT_BLACKLIST) + list(style.blacklist_phrases)
    joined = "、".join(phrases)
    return (
        f"[style: 微信聊天。≤{style.speech_max_chars}字。\n"
        f"禁用词：{joined}\n"
        f"禁止总结、禁止给建议框架（1/2/3 点）\n"
        f"允许：省略、倒装、感叹词（嗯/欸/哦）、破折号、半句话]\n\n"
    )


def pick_prefill(style: StyleConfig, rng: random.Random | None = None) -> str:
    """Pick an assistant prefill opener from the persona's configured list.

    Empty string in the list (or an empty list) means "no prefill this turn".
    """
    if not style.prefill_openers:
        return ""
    r = rng or random
    return r.choice(style.prefill_openers)
