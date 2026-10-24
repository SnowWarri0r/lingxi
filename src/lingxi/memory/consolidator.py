"""What she remembers about him, settled once a conversation has gone quiet.

Facts used to be written per message by the orchestrator, append-only. Three
failures came out of that, all visible in one persona's 39 facts about one
user, and none fixable by a better similarity threshold:

- A change of state was a new fact beside the old one. 「周六要去漫展」 and
  「周六去了漫展」 are not similar enough to fold, so a trip that had happened
  stayed planned for weeks. Replaying the 39 through the similarity dedup
  still left 28, four of them the trip as still ahead.
- A story told over seven minutes became eight facts, each a snapshot of how
  far he had got.
- With the persona's name and the real performer who voices her both in the
  conversation, five facts credited her with what he had done for the
  performer.

So the writer here reads the whole conversation and the facts already held,
and returns edits to that set: add, rewrite (a plan that happened, a detail
that changed), or retire (a restatement, something he corrected). A rewrite
supersedes the old row and a retirement expires it — nothing is deleted, so
any edit can be undone from the database.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from lingxi.utils import lenient_json

_WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

_PROMPT = """你在整理{agent}记住的关于对方的事。下面是刚聊完的一段对话，和她之前已经记住的事。
把这段对话里值得长期记住的，合进已有的记忆里。

【这段对话】（{date_line}）
{dialog}

【已经记住的】（方括号里是记下的日期）
{facts}

{people}【怎么整理】
- 记的是**关于他这个人、过一阵还成立的事**：作息和工作、住在哪怎么通勤、身边的人和宠物、在追/在做的事、喜好和雷区、约定、接下来的计划、经历过的大事。
- 一时的状态不记：哪天困了累了、哪顿吃了什么、几号加班、病中每天好没好、某次聊天里谁说了哪句话。这些只有沉淀成一条往后还算数的结论时才写（比如常说加班→「对方经常加班」）。
- 每条只说一件事，**一句话，60 字以内**。不同的事分开写，别把几件事塞进一条。
- 同一件事、同一方面（比如作息、运动、追的人、某次出行）已经有一条了：这段对话补了新内容就改写那一条（还是一句话，留最要紧的）；只是又说了一遍就不动。**不要再加一条说同一方面的。**
- **事情变了就改写**：计划里的事发生了，改成发生过的样子并写上结果；之前说的情况结束了或变了（病好了、换了工作），照现在的样子改，或者已经没什么可记就撤掉。
- 日期写成具体的几月几号——「昨天」「前天」「明天」「下周六」「国庆」都按这段对话的日期换算，不留相对的说法；拿不准是哪天就不写日期。
- 认清人：{agent}是她自己。对方提到的真人（艺人、声优、朋友）是另外的人，他为谁做了什么、见了谁，就写那个人的名字。{agent}自己在对话里说的话不是关于他的事。
- 已经记住的里有说同一件事的多条，合成一条；有写成流水账的，压回一句；对方明确纠正过的，按纠正后的写。
- 每条以「对方」开头，只写他说过的，不补他没说的，也不替他归纳性格和动机。
- 输出前把整理后的结果从头看一遍：没有两条说同一方面，没有还写成「要去」却已经去过的，没有一时的状态。

输出严格 JSON（直接 {{ 开头），ops 里每个操作是下面三种之一；没有要改的就输出 {{"ops": []}}：
{{"ops": [
  {{"op": "add", "content": "对方……", "evidence": "他的原话"}},
  {{"op": "update", "id": "F3", "content": "对方……（合进新内容后的完整一句）", "evidence": "他的原话"}},
  {{"op": "retire", "id": "F5"}}
]}}
add 和 update 的 evidence：从这段对话里**对方**说的话中原样抄一小段（不改字），就是这条记忆的依据。找不到他本人说过的依据，就不要写这条。
多条重复的旧记忆合并：保留写得最完整的那条不动，其余的 retire。
"""

_PEOPLE_TMPL = ("【这些是真人，跟{agent}不是同一个人】\n{lines}\n"
                "对方有时会对着{agent}说「你的演出」「给你的信」，指的是演她的那个真人的"
                "活动；记的时候写那个真人，跟他后来直接说那个人名字的话对上，是同一件事。\n\n")


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    ts: datetime


@dataclass
class HeldFact:
    id: str
    content: str
    ts: datetime | None = None  # when it was written; 「昨天」 needs it


@dataclass
class Op:
    op: str  # "add" | "update" | "retire"
    content: str = ""
    target: str = ""  # fact id for update/retire


def real_people(persona) -> list[str]:
    """Lexicon entries the persona marks as real people (【中の人】)."""
    lex = getattr(persona, "lexicon", None) or []
    return [e.replace("【中の人】", "").strip() for e in lex
            if isinstance(e, str) and e.startswith("【中の人】")]


def build_prompt(turns: list[Turn], facts: list[HeldFact], *, agent: str,
                 people: list[str] | None = None) -> tuple[str, dict[str, str]]:
    """The prompt, and the map from the short ids shown to real fact ids."""
    first, last = turns[0].ts, turns[-1].ts
    date_line = f"{first:%Y年%-m月%-d日} {_WEEKDAYS[first.weekday()]}"
    if last.date() != first.date():
        date_line += f" 到 {last:%-m月%-d日}"
    # Every line carries its speaker. A turn is often several bubbles, and with
    # only the first labelled the rest read as whoever spoke last: on 10-18
    # her 「别再五天不见人」, a second bubble, was stored as his request.
    dialog = "\n".join(
        f"  {t.ts:%H:%M} {'对方' if t.role == 'user' else agent}：{line.strip()}"
        for t in turns for line in t.content.splitlines() if line.strip())
    ids = {f"F{i + 1}": f.id for i, f in enumerate(facts)}
    held = "\n".join(
        f"  {sid}. {f'[{f.ts:%-m月%-d日}记] ' if f.ts else ''}{f.content}"
        for sid, f in zip(ids, facts)) \
        or "  （还没有）"
    people_block = (_PEOPLE_TMPL.format(agent=agent,
                                        lines="\n".join(f"  - {p}" for p in people))
                    if people else "")
    return _PROMPT.format(agent=agent, date_line=date_line, dialog=dialog,
                          facts=held, people=people_block), ids


def _norm(t: str) -> str:
    return re.sub(r"[\s「」『』\"'“”‘’]", "", t)


_QUOTE = re.compile(r"「([^」]+)」|『([^』]+)』|“([^”]+)”|\"([^\"]+)\"")


def _quoted(content: str) -> list[str]:
    spans = ("".join(g) for g in _QUOTE.findall(content))
    return [q for q in (_norm(m) for m in spans) if len(q) >= 2]


def parse_ops(text: str, ids: dict[str, str],
              his_words: list[str] | None = None,
              her_words: list[str] | None = None,
              held_texts: list[str] | None = None) -> list[Op]:
    return parse_ops_detail(text, ids, his_words, her_words, held_texts)[0]


def parse_ops_detail(text: str, ids: dict[str, str],
                     his_words: list[str] | None = None,
                     her_words: list[str] | None = None,
                     held_texts: list[str] | None = None) -> tuple[list[Op], int]:
    """Ops from the model's JSON. Anything malformed or pointing at a fact that
    was not shown is dropped, not guessed at.

    With `his_words`, an add or update must quote him: its evidence has to be
    found in something he said in this conversation. Asked only to be careful,
    one replay in two still recorded her own news — a lottery she said she had
    entered — as something he did. The quote is checked here, not trusted.

    With `her_words`, a fact that puts in quotes words only she said is not
    his and is dropped — unless a held fact already carries that quote (a
    name from an earlier conversation she says and he doesn't repeat).

    Returns the ops and how many retires were held back.

    A merge is an update and a retire that only make sense together. When any
    add or update in the batch is dropped, its retires are held back too: on
    10-22 a merge's update failed the quote check and its retire went through
    alone, and the fact being merged vanished with nothing holding it.
    """
    said = [_norm(w) for w in his_words] if his_words is not None else None
    hers = [_norm(w) for w in her_words] if her_words is not None else None
    prior = [_norm(t) for t in held_texts or []]
    try:
        data = lenient_json.loads(text)
    except Exception:
        return [], 0
    raw = data.get("ops") if isinstance(data, dict) else None
    raw = [o for o in (raw if isinstance(raw, list) else []) if isinstance(o, dict)]
    out: list[Op] = []
    touched: set[str] = set()
    dropped = 0
    for o in raw:
        kind = o.get("op")
        if kind not in ("add", "update"):
            continue
        content = str(o.get("content") or "").strip()
        target = ids.get(str(o.get("id") or "").strip().upper())
        # A rejected edit of any kind counts: a merge whose update named a
        # miscounted id would otherwise let its retire through alone.
        if not content or (kind == "update" and (not target or target in touched)):
            dropped += 1
            continue
        if said is not None:
            quote = _norm(str(o.get("evidence") or ""))
            if len(quote) < 2 or not any(quote in w for w in said):
                dropped += 1
                continue
        if hers is not None and any(
                any(q in w for w in hers) and not any(q in w for w in said or [])
                and not any(q in p for p in prior)
                for q in _quoted(content)):
            print(f"[memory] dropped, quotes her words as his: {content[:40]}",
                  flush=True)
            dropped += 1
            continue
        if kind == "add":
            out.append(Op("add", content))
        else:
            touched.add(target)
            out.append(Op("update", content, target))
    retires = []
    for o in raw:
        target = ids.get(str(o.get("id") or "").strip().upper())
        if o.get("op") == "retire" and target and target not in touched:
            touched.add(target)
            retires.append(Op("retire", target=target))
    if retires and dropped:
        print(f"[memory] held back {len(retires)} retire(s): "
              f"{dropped} edit(s) in the same batch were dropped "
              f"({', '.join(r.target[:8] for r in retires)})", flush=True)
        return out, len(retires)
    return out + retires, 0


async def consolidate(llm, turns: list[Turn], facts: list[HeldFact], *,
                      agent: str, people: list[str] | None = None,
                      model: str | None = None) -> list[Op]:
    return (await consolidate_detail(llm, turns, facts, agent=agent,
                                     people=people, model=model))[0]


async def consolidate_detail(llm, turns: list[Turn], facts: list[HeldFact], *,
                             agent: str, people: list[str] | None = None,
                             model: str | None = None) -> tuple[list[Op], int]:
    """The edits, and how many retires were held back."""
    if not any(t.role == "user" for t in turns):
        return [], 0
    prompt, ids = build_prompt(turns, facts, agent=agent, people=people)
    kwargs = {"model": model} if model else {}
    resp = await llm.complete(
        messages=[{"role": "user", "content": prompt}],
        system="你负责整理长期记忆，输出严格 JSON。",
        max_tokens=3000,
        temperature=0.2,
        _debug_purpose="memory_consolidation",
        **kwargs,
    )
    return parse_ops_detail(resp.content, ids,
                            his_words=[t.content for t in turns if t.role == "user"],
                            her_words=[t.content for t in turns if t.role != "user"],
                            held_texts=[f.content for f in facts])


# A conversation counts as over once he has been quiet this long.
QUIET_AFTER = timedelta(minutes=30)
# Or once this many turns are waiting: the buffer holds 30, and a long
# conversation must not push its start out before it has been read.
BACKLOG_TURNS = 20


def due(rec, turns, now: datetime):
    """The turns to consolidate for this recipient now, or None.

    Only a conversation with something of his in it is worth a call; her
    unanswered openers alone are not.
    """
    since = getattr(rec, "last_consolidated_at", None)
    fresh = [t for t in turns if since is None or t.timestamp > since]
    if not any(t.role == "user" for t in fresh):
        return None
    if now - rec.last_interaction >= QUIET_AFTER or len(fresh) >= BACKLOG_TURNS:
        return fresh
    return None


def review_due(rec, now: datetime) -> bool:
    """Once a night, from REVIEW_HOUR on, for anyone she holds facts about."""
    return (now.hour >= REVIEW_HOUR
            and getattr(rec, "last_review_date", None) != now.date().isoformat())


async def apply_ops(ops: list[Op], *, subject: str, store, writer,
                    now: datetime, said_at: dict | None = None) -> None:
    """A rewrite supersedes the old row and a retirement expires it.

    `said_at` keeps a rewrite dated when he said it. The nightly review only
    pins down a time 「明天」 left open; stamped with the review night, the
    fact read as said that night, and a fact rewritten four nights running
    walked forward a day each time.
    """
    from lingxi.facts.models import FactType, Source
    for o in ops:
        if o.op == "retire":
            await store.expire(o.target, now)
            continue
        await writer.write(
            subject=subject, content=o.content, type=FactType.PATTERN,
            source=Source.USER_STATED,
            ts=(said_at or {}).get(o.target) or now,
            supersedes=o.target if o.op == "update" else None,
        )


class ConsolidationScheduler:
    """Every few minutes, settle the memory of any conversation that has
    gone quiet."""

    def __init__(self, *, llm, tracker, short_term, store, writer, persona,
                 interval_minutes: int = 5, model: str | None = None):
        self._llm = llm
        self._tracker = tracker
        self._short_term = short_term
        self._store = store
        self._writer = writer
        self._agent = getattr(persona, "name", "") or "她"
        self._people = real_people(persona)
        self._interval = interval_minutes * 60
        self._model = model
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())
            print("[memory] consolidation scheduler started", flush=True)

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick(datetime.now())
            except Exception as e:
                print(f"[memory] consolidation tick failed: {e!r}", flush=True)
            await asyncio.sleep(self._interval)

    async def tick(self, now: datetime) -> None:
        for rec in self._tracker.all_records():
            if rec.unreachable_since is not None:
                continue
            key = f"{rec.channel}:{rec.recipient_id}"
            turns = await self._short_term.snapshot_for_recipient(key)
            fresh = due(rec, turns, now)
            if fresh:
                await self.run(rec, key, fresh, now)
            elif review_due(rec, now):
                await self.nightly(rec, key, now)

    async def nightly(self, rec, key: str, now: datetime) -> list[Op]:
        """The once-a-night review of everything held about this person."""
        from lingxi.facts.models import FactType
        subject = f"user:{key}"
        held = await self._store.query(subject=subject, type=FactType.PATTERN,
                                       limit=200)
        ops = await review(self._llm,
                           [HeldFact(f.id, f.content, f.ts) for f in held],
                           agent=self._agent, today=now, model=self._model)
        await apply_ops(ops, subject=subject, store=self._store,
                        writer=self._writer, now=now,
                        said_at={f.id: f.ts for f in held})
        rec.last_review_date = now.date().isoformat()
        await self._tracker.save()
        counts = {k: sum(o.op == k for o in ops) for k in ("update", "retire")}
        print(f"[memory] nightly review {key[-6:]}: {len(held)} facts → {counts}",
              flush=True)
        return ops

    async def run(self, rec, key: str, fresh, now: datetime) -> list[Op]:
        from lingxi.facts.models import FactType
        subject = f"user:{key}"
        held = await self._store.query(subject=subject, type=FactType.PATTERN,
                                       limit=200)
        ops, held_back = await consolidate_detail(
            self._llm,
            [Turn(t.role, t.content, t.timestamp) for t in fresh],
            [HeldFact(f.id, f.content, f.ts) for f in held],
            agent=self._agent, people=self._people, model=self._model)
        # Held-back retires would otherwise be gone for good: the turns are
        # marked read, and no later run sees them. So a batch with held-back
        # retires is not applied at all and the same turns are read once
        # more on the next tick; the second reading is applied as it comes.
        retried = getattr(self, "_retried", {})
        self._retried = retried
        if held_back and retried.get(key) != fresh[-1].timestamp:
            retried[key] = fresh[-1].timestamp
            print(f"[memory] {key[-6:]}: {held_back} retire(s) held back; "
                  "reading these turns again next tick", flush=True)
            return []
        retried.pop(key, None)
        await apply_ops(ops, subject=subject, store=self._store,
                        writer=self._writer, now=now)
        rec.last_consolidated_at = fresh[-1].timestamp
        await self._tracker.save()
        counts = {k: sum(o.op == k for o in ops) for k in ("add", "update", "retire")}
        print(f"[memory] consolidated {key[-6:]}: {len(fresh)} turns → {counts}",
              flush=True)
        return ops


# --- the nightly review ------------------------------------------------------
#
# The quiet-time pass only edits what a conversation touches, so a fact no one
# mentions again never changes: on 2026-10-13 she still held a National Day
# trip as ahead of him a week after the holiday, a cold from August, and a
# haircut 「昨天」. Once a night, with no conversation in front of it, the
# review reads everything held against today's date.

_REVIEW_PROMPT = """今天是 {today}。你在替{agent}复盘她记住的关于对方的事，这次没有新对话，只看时间过去之后哪些已经不成立。

【已经记住的】（方括号里是记下的日期）
{facts}

【怎么复盘】
- 一时的状态（哪几天生病、某段时间很忙、某天很累、睡眠欠账、某天的心情）过了这么久已经不算数：retire。
- 写着要做、但日子已经过去的计划：改写成「之前说过要……，后来怎么样还不知道」这类说法——**结果他没说过，就写不知道**。
- 用了「昨天」「这几天」「最近」这种相对说法的：按记下的日期换成具体几月几号，或者在日子过了很久时 retire。
- 过一阵还成立的事（住哪、作息、喜好、身边的人、经历过的大事）原样留着，不动。
- 改写只动时间相关的说法，原句里没有的事一件也不添。

输出严格 JSON（直接 {{ 开头）；没有要改的就输出 {{"ops": []}}：
{{"ops": [
  {{"op": "update", "id": "F3", "content": "对方……"}},
  {{"op": "retire", "id": "F5"}}
]}}
"""

# What a time-only rewrite may add to the sentence it rewrites: dates and the
# words for "planned / not known how it went". Anything else new would be an
# invented outcome.
_TIME_CHARS = set("0123456789０１２３４５６７８９年月日号前后之前当时那时候曾经已经原本来计划打算说过要还不知道怎么样去没有是否成行了的过这些天段间，。、；：（）「」 —…")
REVIEW_HOUR = 4


def review_prompt(facts: list[HeldFact], *, agent: str, today: datetime) -> tuple[str, dict[str, str]]:
    ids = {f"F{i + 1}": f.id for i, f in enumerate(facts)}
    held = "\n".join(
        f"  {sid}. {f'[{f.ts:%Y年%-m月%-d日}记] ' if f.ts else ''}{f.content}"
        for sid, f in zip(ids, facts)) or "  （还没有）"
    day = f"{today:%Y年%-m月%-d日} {_WEEKDAYS[today.weekday()]}"
    return _REVIEW_PROMPT.format(today=day, agent=agent, facts=held), ids


def adds_only_time(old: str, new: str) -> bool:
    """True when every character the rewrite adds is a date or a word for not
    knowing how it went."""
    have = set(old)
    return all(c in have or c in _TIME_CHARS for c in new)


def parse_review(text: str, ids: dict[str, str],
                 held: dict[str, str]) -> list[Op]:
    try:
        data = lenient_json.loads(text)
    except Exception:
        return []
    out, touched = [], set()
    for o in (data.get("ops") if isinstance(data, dict) else None) or []:
        if not isinstance(o, dict):
            continue
        target = ids.get(str(o.get("id") or "").strip())
        if not target or target in touched:
            continue
        if o.get("op") == "retire":
            out.append(Op("retire", target=target))
        elif o.get("op") == "update":
            content = str(o.get("content") or "").strip()
            # An unchanged rewrite would only re-date the fact, which keeps
            # an old line near the top of every prompt: four nights running
            # the same holiday plan came back word for word.
            if (not content or content == held[target]
                    or not adds_only_time(held[target], content)):
                continue
            out.append(Op("update", content, target))
        else:
            continue
        touched.add(target)
    return out


async def review(llm, facts: list[HeldFact], *, agent: str, today: datetime,
                 model: str | None = None) -> list[Op]:
    if not facts:
        return []
    prompt, ids = review_prompt(facts, agent=agent, today=today)
    kwargs = {"model": model} if model else {}
    resp = await llm.complete(
        messages=[{"role": "user", "content": prompt}],
        system="你负责复盘长期记忆，输出严格 JSON。",
        max_tokens=3000, temperature=0.2,
        _debug_purpose="memory_review", **kwargs)
    return parse_review(resp.content, ids, {f.id: f.content for f in facts})
