"""Helpers for single-call combo prompt construction.

- render_fewshots_as_messages: turn FewShotSamples into user/assistant message pairs
- build_style_preamble: an Author's Note prefix that sits just before the user message
- pick_prefill: choose an assistant prefill opener from the persona's configured set
"""

from __future__ import annotations

import random

from lingxi.fewshot.models import FewShotSample
from lingxi.persona.models import StyleConfig

# Historic note: this module used to export DEFAULT_BLACKLIST, a list of
# ~30 AI-tell phrases planted into every turn's user message as 避用词.
# Removed — planting bad-output strings is exactly the reverse-attention
# anti-pattern (auto-memory: feedback_never_plant_bad_output_strings).
# Register protection now lives in: persona positive examples + fewshot
# samples + the system prompt's positive rules.


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

    Carries only positive directives that need recency-channel weight:
    - length cap for THIS turn
    - voice_hint reminder (so persona voice doesn't flatten across turns)
    - biography-hit awareness (if applicable)
    """
    parts = [f"[本轮 ≤{style.speech_max_chars} 字，IM 口语短句"]
    if voice_hint:
        parts.append(f"语感：{voice_hint}")
    if biography_hit:
        parts.append(
            "脑子里浮现了相关回忆（见 system prompt 📖）——是底色不是台词，看你真实想不想讲"
        )
    parts.append("]")
    # Anchor the very next user message as the one to respond to.
    parts.append("\n— 对方刚发的就是下面这句 —")
    return "\n".join(parts) + "\n"


def pick_prefill(style: StyleConfig, rng: random.Random | None = None) -> str:
    """Pick an assistant prefill opener from the persona's configured list.

    Empty string in the list (or an empty list) means "no prefill this turn".
    """
    if not style.prefill_openers:
        return ""
    r = rng or random
    return r.choice(style.prefill_openers)
