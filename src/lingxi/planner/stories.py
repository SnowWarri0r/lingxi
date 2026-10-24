"""What she could tell someone about her days — a layer between her diary and
what she says.

Openers used to read her life-sim moments raw. Those are written as she lives
them: a line of lyrics called 「雾里走路」, 「那俩圈」 on a set list, the lottery
for a show she performs in. Retold as written, each turned up in a message he
could not follow, or that put her in the wrong place, and the same practice
session went out three days running because nothing remembered it had been
told. Every one of those was patched in the opener prompt; this is the layer
the patches stood in for.

A curator reads her last two days and this morning's scan and writes a few
stories, each a sentence he can follow with no context: which thing, what she
was doing, what she made of it. They are cached, kept stable across refreshes
when still true, and marked per person once told. Openers choose from these.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from lingxi.utils import lenient_json

REFRESH = timedelta(minutes=45)
WINDOW = timedelta(hours=36)

_PROMPT = """{self_context}

下面是你这两天给自己记的经历，和今天扫到的消息。从里面挑出 3-5 件你想讲给朋友听的事。
你这边的日子他不在场，所以每件写成一句**他一读就懂**的话：说清是哪件事、你在做什么、你怎么想。

【你这两天记下的】
{events}

【今天扫到的】
{world}

{stage}【上次整理的】（还成立的就把那句原样抄下来，不用重写）
{previous}

【怎么挑】
- 一件事写一条；同一件事散在好几条日记里的，合成一条。
- 各方面都挑一点：练习排练、演出、圈里新鲜事、队友、家人、吃喝、天气……别全是同一类。
- 每条一两句，第一人称，用你平时说话的语气；日记里只有你自己懂的说法换成大白话。
- aspect 填这件事主要在讲什么：排练时顺带提到的天气、吃喝、队友，算练习排练。

输出 JSON（直接 {{ 开头）：
{{"stories": [{{"text": "……", "aspect": "从这几个里挑：{aspects}"}}]}}
"""


@dataclass
class Story:
    id: str
    text: str
    aspect: str = ""


def _sid(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


@dataclass
class _Shelf:
    generated_at: str = ""
    newest_event: str = ""
    stories: list[dict] = field(default_factory=list)
    told: dict[str, list[str]] = field(default_factory=dict)


class StoryShelf:
    def __init__(self, path: Path | str | None, llm, persona, *,
                 model: str | None = None):
        self._path = Path(path) if path else None
        self._llm = llm
        self._persona = persona
        self._model = model
        self._shelf = self._load()

    def _load(self) -> _Shelf:
        if self._path and self._path.exists():
            try:
                return _Shelf(**json.loads(self._path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                pass
        return _Shelf()

    def _save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self._shelf), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.rename(self._path)

    def untold(self, key: str) -> list[Story]:
        told = set(self._shelf.told.get(key, []))
        return [Story(**s) for s in self._shelf.stories if s["id"] not in told]

    def made_on(self, day) -> bool:
        made = self._shelf.generated_at
        return bool(made and self._shelf.stories
                    and datetime.fromisoformat(made).date() == day)

    def mark_told(self, key: str, story_id: str) -> None:
        told = self._shelf.told.setdefault(key, [])
        if story_id and story_id not in told:
            told.append(story_id)
            del told[:-200]
            self._save()

    def _stale(self, events, now: datetime) -> bool:
        newest = max((f.ts for f in events), default=None)
        if not self._shelf.generated_at:
            return True
        made = datetime.fromisoformat(self._shelf.generated_at)
        if made.date() != now.date():
            return True
        newer = newest is not None and newest.isoformat() > self._shelf.newest_event
        return newer and now - made >= REFRESH

    async def refresh(self, events, world, now: datetime) -> list[Story]:
        """Curate again if her diary has moved on; the shelf either way."""
        events = [f for f in events if now - f.ts <= WINDOW]
        if not events or not self._stale(events, now):
            return [Story(**s) for s in self._shelf.stories]
        from lingxi.persona.prompt_builder import build_stage_block
        from lingxi.persona.self_context import build_self_context
        from lingxi.temporal.proactive import OPENER_ASPECTS
        stage = build_stage_block(self._persona, now.date()) or ""
        prompt = _PROMPT.format(
            self_context=build_self_context(self._persona),
            events="\n".join(f"- [{f.ts:%m-%d %H:%M}] {f.content}"
                             for f in sorted(events, key=lambda f: f.ts)),
            world="\n".join(f"- {w.content}" for w in world) or "（今天没扫到什么）",
            stage=stage,
            previous="\n".join(f"- {s['text']}" for s in self._shelf.stories)
                     or "（还没有）",
            aspects="、".join(OPENER_ASPECTS))
        kwargs = {"model": self._model} if self._model else {}
        resp = await self._llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system="你在整理自己这两天想跟朋友讲的事，输出严格 JSON。",
            max_tokens=2000, temperature=0.5, _debug_purpose="story_curation",
            **kwargs)
        stories = parse_stories(resp.content, OPENER_ASPECTS)
        if not stories:
            return [Story(**s) for s in self._shelf.stories]
        self._shelf.stories = [asdict(s) for s in stories]
        self._shelf.generated_at = now.isoformat()
        self._shelf.newest_event = max(f.ts for f in events).isoformat()
        self._save()
        print(f"[stories] curated {len(stories)}: "
              f"{', '.join(s.aspect or '?' for s in stories)}", flush=True)
        return stories


def parse_stories(text: str, aspects) -> list[Story]:
    try:
        data = lenient_json.loads(text[text.find("{"):text.rfind("}") + 1])
    except Exception:
        return []
    out = []
    for s in (data.get("stories") if isinstance(data, dict) else None) or []:
        if not isinstance(s, dict):
            continue
        body = str(s.get("text") or "").strip()
        if not body:
            continue
        aspect = str(s.get("aspect") or "").strip()
        out.append(Story(_sid(body), body, aspect if aspect in aspects else ""))
    return out[:6]


def format_stories_block(stories: list[Story]) -> tuple[str, dict[str, str]]:
    """The opener's block, and the short ids shown → story ids."""
    if not stories:
        return "", {}
    ids = {f"S{i + 1}": s.id for i, s in enumerate(stories)}
    lines = "\n".join(
        f"- {sid}{'｜' + s.aspect if s.aspect else ''}：{s.text}"
        for sid, s in zip(ids, stories))
    return ("## 你这两天可以跟他讲的事（**优先从这里挑一件起头**；讲过的已经不在这儿了）\n"
            f"{lines}\n\n"
            "（挑一件你真想说的就讲那一件，用你自己的话讲出来。）\n"), ids
