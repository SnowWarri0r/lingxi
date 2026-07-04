"""Vision LLM tagging for stickers.

caption_image reads one image, asks the provider to describe it as a sticker,
and parses a JSON tag blob out of the reply. Parsing is defensive — a reply
that isn't valid JSON yields empty fields rather than raising, so a single bad
image can't abort a batch crawl.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path


_PROMPT = (
    "这是一张聊天表情包。想象真人会在「什么心情/场景」下发它,据此打标,方便按情绪检索。"
    "注意:先读图上的文字(往往是关键,如\"摸鱼\"=偷懒梗而非委屈);情绪以图上文字与整体语气为准。"
    "只输出 JSON,字段:"
    '{"caption":"≤14字,含图上文字","emotion":"最贴切的一个情绪或用途词",'
    '"tags":["6-9个检索词:情绪同义词+使用场景+画面+图上文字"],'
    '"when_to_use":"2-3个聊天场景,口语"}'
)

_MEDIA_BY_SUFFIX = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}


def _media_type(path: Path) -> str:
    return _MEDIA_BY_SUFFIX.get(path.suffix.lower(), "image/png")


def _parse(text: str) -> dict:
    """Extract the first flat {...} block and coerce to the expected shape.

    Defensive against real model output: strips markdown code fences, matches
    a single flat object (the schema has no nested braces), and bails to empty
    fields on any non-dict / unparseable result so a bad reply never raises.
    """
    empty = {"caption": "", "emotion": "", "tags": [], "when_to_use": ""}
    # Strip ```json / ``` code fences the model often wraps JSON in.
    cleaned = re.sub(r"```(?:json)?", "", text)
    # If the entire reply is valid JSON but not a dict (e.g. a bare list),
    # bail immediately — the regex below would find an inner object and
    # silently return wrong data.
    try:
        top = json.loads(cleaned.strip())
        if not isinstance(top, dict):
            return empty
    except (json.JSONDecodeError, ValueError):
        pass  # not pure JSON — fall through to embedded-object extraction
    # Flat object only — the schema has no nested objects, so [^{}] is exact
    # and avoids the greedy-match-across-two-objects failure mode.
    m = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if not m:
        return empty
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return empty
    if not isinstance(data, dict):
        return empty
    tags = data.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    return {
        "caption": str(data.get("caption", "") or ""),
        "emotion": str(data.get("emotion", "") or ""),
        "tags": [str(t) for t in tags],
        "when_to_use": str(data.get("when_to_use", "") or ""),
    }


async def caption_image(provider, image_path: str | Path) -> dict:
    """Return {caption, emotion, tags, when_to_use} for one sticker image."""
    path = Path(image_path)
    data_b64 = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "source": {
                "type": "base64",
                "media_type": _media_type(path),
                "data": data_b64,
            }},
            {"type": "text", "text": _PROMPT},
        ],
    }]
    result = await provider.complete(
        messages=messages, max_tokens=512, temperature=0.3,
        _debug_purpose="sticker_caption")
    return _parse(result.content)
