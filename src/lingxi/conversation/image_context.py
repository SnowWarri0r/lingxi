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

# The description is all the responder ever sees of a picture, so text on it
# is copied out whole. Capped at one line of 30 characters, an event
# schedule he sent reached her as 「含未来三个月的几场活动安排」:
# every date and city on it was gone, and she asked him which days and where.
_PROMPT = (
    "先用一句话说清这张图是什么，≤30字。"
    "图上有字就照原样抄下来，接在这句后面："
    "表情包上的一两句直接接上（往往就是全部意思）；"
    "日程表、截图、公告、聊天记录这类以字为主的图，把字全部抄出，"
    "日期、时间、地点、人名、价格逐条照抄，一条一行。"
    "只输出描述和抄下的字。多张图按顺序，每张以「图1：」「图2：」开头。"
)

# A whole schedule fits; a runaway answer still cannot fill the prompt.
_MAX_CHARS = 800


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
            max_tokens=1024,
            temperature=0.3,
            _debug_purpose="image_describe",
        )
        # Multiple images come back as several lines; join them into one so the
        # description stays a single inline marker in the prompt and the buffer.
        # Reading the result is inside the try on purpose: a provider that
        # answers with nothing at all must cost a description, not the turn.
        lines = [ln.strip() for ln in (result.content or "").splitlines()
                 if ln.strip()]
    except Exception as e:
        print(f"[image] describe failed: {e}", flush=True)
        return ""
    return "；".join(lines)[:_MAX_CHARS]
