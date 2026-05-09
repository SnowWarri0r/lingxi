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
    # AI 共情套话——这些在 LLM 训练数据里被反复奖励，但真人不这么说
    "你不是一个人",
    "你并不孤单",
    "我能理解",
    "我太懂了",
    "感同身受",
    "我也有过这种感觉",
    "我也有过那种感觉",
    "坚持一下",
    "加油",
    "辛苦了",
    # IM 不存在的"我没注意到"系列——LLM 客服话术
    "刚才分神",
    "刚才走神",
    "我没注意到",
    "刚才没看到",
    "记不清刚才",
    "忘了刚才聊",
    # 表演忏悔系列——被批评后 AI 反射
    "哎呀我错了",
    "我也是没想到",
    "我怎么这样",
    "你说的这么",  # "你说的这么难受/重要的事我还..." 复述罪状起手
    "我刚才不该",
    "对不起 我刚才",
    "抱歉 我刚才",
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
    """Compact per-turn nudge prepended to user message.

    Most behavior rules live in the system prompt. This is just:
    - length cap for THIS turn
    - voice_hint reminder (so persona voice doesn't flatten)
    - the active blacklist (so it's right next to the user message in attention)
    - biography hit awareness (if applicable)

    Anti-AI-reflex warnings are NOT duplicated here — they're already in the
    system prompt's format preamble, repeating them would only inflate tokens
    and reinforce the negative space they live in.
    """
    phrases = list(DEFAULT_BLACKLIST) + list(style.blacklist_phrases)
    parts = [f"[本轮 ≤{style.speech_max_chars} 字，IM 口语短句"]
    if voice_hint:
        parts.append(f"语感：{voice_hint}")
    if biography_hit:
        parts.append(
            "脑子里浮现了相关回忆（见 system prompt 📖）——是底色不是台词，看你真实想不想讲"
        )
    parts.append(f"避用词：{ '、'.join(phrases) }]")
    # Explicit marker so the model doesn't conflate the latest user message
    # with earlier turns / fewshot context. The user message immediately
    # follows this preamble.
    parts.append("\n— 对方刚发的就是下面这句，只回这句，不要把前几轮的话题混进来 —")
    return "\n".join(parts) + "\n"


def pick_prefill(style: StyleConfig, rng: random.Random | None = None) -> str:
    """Pick an assistant prefill opener from the persona's configured list.

    Empty string in the list (or an empty list) means "no prefill this turn".
    """
    if not style.prefill_openers:
        return ""
    r = rng or random
    return r.choice(style.prefill_openers)
