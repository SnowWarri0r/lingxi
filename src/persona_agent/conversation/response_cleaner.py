"""Post-process the extracted speech portion of a TurnOutput.

Even when we ask the LLM to put narration in JSON meta, it sometimes
still writes prose narration into the speech. This cleaner strips the
most recognisable artifacts from speech text BEFORE it reaches channels.

Input: parsed `speech` field (plain text, no JSON/delimiter)
Output: cleaned speech
"""

from __future__ import annotations

import re


# Words/phrases that strongly signal a line is stage-direction, not dialogue.
# If a line consists (mostly) of these, we drop the whole line.
_NARRATION_MARKERS = (
    # Expression / body language
    "笑了笑", "轻笑", "微笑", "苦笑", "忍不住笑",
    "眨了眨眼", "眨眨眼",
    "点点头", "点了点头", "摇摇头", "摇了摇头",
    "皱了皱眉", "皱眉", "挑眉",
    "抬头", "低下头", "歪着头", "歪头",
    "看了看时间", "看了一眼", "扫了一眼",
    "停顿了一下", "停顿一下", "稍微停顿",
    "愣了一下", "愣一下", "愣住",
    "若有所思", "若有所思地", "沉思",
    "叹了口气", "叹气",
    "深呼吸", "倾身",
    # Expression phrases
    "带着", "露出", "闪过",
    # "说" with modifier
    "温和地说", "温暖地说", "轻声说", "小声说", "柔声说", "认真地说",
    "缓缓地说", "关心地说", "好奇地说", "有点困惑", "有些困惑", "有些不好意思",
    # Meta introductions
    "简单地介绍", "简单介绍", "自我介绍",
    "解释道", "补充道", "解释一下", "补充一下",
)

# Meta-commentary (AI self-awareness leaks)
_META_PATTERNS = [
    re.compile(r"^[^\n]*我是不是[^\n]*?(?:太|听起来|显得)[^\n]*?(?:客观|理性|严肃|冷静|正式|书面|AI|机器)[^\n]*?[?？]", re.MULTILINE),
    re.compile(r"^[^\n]*抱歉[^\n]*?(?:说话|表达|语气|用词)[^\n]*?(?:理性|客观|正式|文艺|诗意|书面|严肃)[^\n]*$", re.MULTILINE),
    re.compile(r"^[^\n]*(?:让我)?重新(?:说|组织|表达)一?(?:下|次)?[^\n]*$", re.MULTILINE),
    re.compile(r"^[^\n]*其实我(?:是|想|要)?想说[^\n]*$", re.MULTILINE),
]

# Inline action-phrase at end of sentence: strip after the ending punctuation
_INLINE_TRAILING = re.compile(
    r"(?P<keep>[^\n]*?[。！？!?])\s+(?P<tail>[^\n]*)"
)


def _is_narration_line(line: str) -> bool:
    """Heuristic: does this line look like pure stage direction?"""
    stripped = line.strip()
    if not stripped:
        return False

    marker_hits = sum(1 for m in _NARRATION_MARKERS if m in stripped)
    if marker_hits == 0:
        return False

    # A line with a SENTENCE-ENDING punctuation mark is dialogue, even if it
    # contains narration vocabulary. Only drop when no full sentence.
    has_sentence_end = any(p in stripped for p in "。！？.!?")
    if has_sentence_end:
        return False

    # Pure narration: multiple markers OR short fragment with marker
    if marker_hits >= 2:
        return True
    if marker_hits >= 1 and len(stripped) < 30:
        return True
    return False


def _clean_inline_trailing(text: str) -> str:
    """Strip trailing narration that's stuck after a proper sentence."""
    result_lines = []
    for line in text.split("\n"):
        # If the line ends with a narration marker phrase after punctuation,
        # drop that tail
        m = _INLINE_TRAILING.match(line)
        if m:
            keep = m.group("keep")
            tail = m.group("tail")
            # Is the tail pure narration?
            tail_hits = sum(1 for marker in _NARRATION_MARKERS if marker in tail)
            has_tail_dialogue = any(p in tail for p in "？！?!")
            if tail_hits >= 1 and not has_tail_dialogue and len(tail) < 40:
                result_lines.append(keep)
                continue
        result_lines.append(line)
    return "\n".join(result_lines)


def clean_speech(text: str) -> str:
    """Remove narration and meta-commentary from already-extracted speech."""
    if not text:
        return text

    original = text

    # 1. Remove pure-narration lines
    lines = text.split("\n")
    kept = [line for line in lines if not _is_narration_line(line)]
    text = "\n".join(kept)

    # 2. Meta-commentary patterns
    for pat in _META_PATTERNS:
        text = pat.sub("", text)

    # 3. Inline trailing narration after a sentence
    text = _clean_inline_trailing(text)

    # 4. Strip leftover *stage-direction* markers
    text = re.sub(r"\*[^*\n]{1,60}\*\s*", "", text)

    # 5. Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if not text:
        return original.strip()

    return text
