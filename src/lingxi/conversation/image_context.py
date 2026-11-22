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

import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path

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

# He sent a sticker of a plush made after her and the description said
# 「一个小玩偶」, so she took it for any toy. The model does not know her by
# sight: told who she is it still said 「一个小玩偶」 3/3, and asked whether
# the plush was from her series it said yes for a wrong reason. Handed a
# picture of her inside the describe call it named five other members as
# her. A separate yes/no against that picture, told to match hair and
# streaks, held: on 4 pictures of her and 10 that were not (7 of them
# members in the same uniform), 30/30 no on the others, 9/12 yes on hers —
# the miss a small figure in a crowd. A miss leaves the description as it
# was; a false yes would have her claim someone else's face.
_LIKENESS_ASK = (
    "第一张是{name}。后面的图里有没有{name}本人，或者照着{name}做的东西"
    "（玩偶、手办、立牌、画）？\n"
    "先对一下发型、发色和挑染，跟第一张对得上才算。只回答「是」或「否」。"
)
_LIKENESS_NOTE = "（图里的是{name}，或照着{name}做的东西）"
_LIKENESS_FILES = ("likeness.png", "likeness.jpg", "likeness.jpeg", "likeness.webp")


@dataclass(frozen=True)
class Likeness:
    """A picture of the persona to check inbound images against."""

    name: str
    data: str            # base64
    media_type: str

    @classmethod
    def load(cls, data_dir, name: str) -> "Likeness | None":
        """`<data_dir>/likeness.*` if present. Kept out of the repo with the
        rest of the data: it is usually official art."""
        if not data_dir:
            return None
        for fn in _LIKENESS_FILES:
            p = Path(data_dir) / fn
            if p.is_file():
                ext = p.suffix.lstrip(".").lower()
                return cls(name, base64.b64encode(p.read_bytes()).decode(),
                           "image/" + ("jpeg" if ext == "jpg" else ext))
        return None


def _block(data: str, media_type: str) -> dict:
    return {"type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data}}


async def _shows_her(provider, images: list[dict], likeness: Likeness) -> bool:
    try:
        result = await provider.complete(
            messages=[{"role": "user", "content": [
                _block(likeness.data, likeness.media_type),
                *[_block(i.get("data", ""), i.get("media_type", "image/png"))
                  for i in images],
                {"type": "text", "text": _LIKENESS_ASK.format(name=likeness.name)},
            ]}],
            max_tokens=10,
            temperature=0.3,
            _debug_purpose="image_likeness",
        )
        return (result.content or "").strip().startswith("是")
    except Exception as e:
        print(f"[image] likeness check failed: {e}", flush=True)
        return False


async def describe_images(provider, images: list[dict],
                          likeness: Likeness | None = None) -> str:
    """One line describing `images`, or "" if the look-at-it call fails.

    `images` uses the same shape the channel hands the engine:
    ``{"data": <base64>, "media_type": "image/png"}``. With a `likeness`,
    a picture of her gets a note saying so; the check runs alongside the
    description, so it adds no wait.
    """
    if not images:
        return ""
    if likeness is None:
        return await _describe(provider, images)
    desc, hers = await asyncio.gather(
        _describe(provider, images), _shows_her(provider, images, likeness))
    if desc and hers:
        note = _LIKENESS_NOTE.format(name=likeness.name)
        return desc[:_MAX_CHARS - len(note)] + note
    return desc


async def _describe(provider, images: list[dict]) -> str:
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
