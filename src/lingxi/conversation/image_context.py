"""Turn inbound images into text the responder can read.

The user-facing voice runs on a Chinese-native responder, and the one in
service (deepseek-v4-flash) is text-only: handed an image block it answers
`400 This model does not support image` and the whole turn comes back empty,
so every photo and every Feishu sticker got the empty-reply fallback instead
of a reply.

So the vision-capable main model looks at the image first and says what is in
it in one line; that line rides into the turn as text. The description also
lands in short-term memory, which is a second win — the buffer used to record
`[发送了1张图片]` and nothing more, leaving her unable to refer back to a
photo one turn later.

Failure is never fatal: no description means the turn proceeds with a plain
"there was an image" marker, which is still a reply.
"""

from __future__ import annotations

_PROMPT = (
    "用一句话说清这张图里是什么，≤30字。"
    "图上有字就把字念出来（表情包里的字往往就是全部意思）。"
    "只输出这句描述。多张图按顺序分行，一行一张。"
)

_MAX_CHARS = 120


async def describe_images(provider, images: list[dict]) -> str:
    """One line describing `images`, or "" if the look-at-it call fails.

    `images` uses the same shape the channel hands the engine:
    ``{"data": <base64>, "media_type": "image/png"}``.
    """
    if not images:
        return ""
    blocks: list[dict] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img.get("media_type", "image/png"),
                "data": img.get("data", ""),
            },
        }
        for img in images
    ]
    blocks.append({"type": "text", "text": _PROMPT})
    try:
        result = await provider.complete(
            messages=[{"role": "user", "content": blocks}],
            max_tokens=256,
            temperature=0.3,
            _debug_purpose="image_describe",
        )
    except Exception as e:
        print(f"[image] describe failed: {e}", flush=True)
        return ""
    # Multiple images come back as several lines; join them into one so the
    # description stays a single inline marker in the prompt and the buffer.
    lines = [ln.strip() for ln in (result.content or "").splitlines() if ln.strip()]
    return "；".join(lines)[:_MAX_CHARS]
