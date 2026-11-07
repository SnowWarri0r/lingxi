"""What is coming up between them: his plans with a day on them, and what she
promised him.

Facts held this only as sentences. An opener read every fact each time and
had to notice on its own that a day had come; nothing recorded that a plan
had been asked about, or what he answered, and nothing at all held what she
had promised him. A plan for a show three weeks out sat in the list beside
his commute and his cat, and the nightly review once rewrote it as already
past.

Here each such thing is an item with a day and a state. A pass after each
quiet conversation adds items and closes the ones he reported on, quoting
the turn that says so, as the fact consolidator does. Openers are offered an
item on the day before, the day itself and the few days after, at most one
item a day, and an item he did not answer about is left alone after that.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from lingxi.utils import lenient_json

# After its day an item can be asked about for this long, then it lapses.
AFTER_DAYS = 3
LAPSE_DAYS = 7

HIS, HERS = "his", "hers"


@dataclass
class Item:
    id: str
    kind: str          # HIS: a plan of his. HERS: something she promised him.
    text: str
    day: str           # YYYY-MM-DD
    evidence: str
    created: str
    status: str = "open"       # open | done | dropped | lapsed
    outcome: str = ""
    asked: list[dict] = field(default_factory=list)   # {"stage", "at"}

    @property
    def when(self) -> date:
        return date.fromisoformat(self.day)


def _iid(kind: str, text: str, day: str) -> str:
    return hashlib.sha1(f"{kind}|{day}|{text}".encode("utf-8")).hexdigest()[:10]


class AgendaBook:
    def __init__(self, path: Path | str | None):
        self._path = Path(path) if path else None
        self._items: dict[str, list[Item]] = self._load()

    def _load(self) -> dict[str, list[Item]]:
        if self._path and self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                return {k: [Item(**i) for i in v] for k, v in raw.items()}
            except (OSError, ValueError, TypeError):
                pass
        return {}

    def save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: [asdict(i) for i in v]
                                   for k, v in self._items.items()},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.rename(self._path)

    def items(self, key: str) -> list[Item]:
        return self._items.get(key, [])

    def open_items(self, key: str, today: date) -> list[Item]:
        """Open items, after letting the ones long past lapse."""
        changed = False
        for i in self.items(key):
            if i.status == "open" and (today - i.when).days > LAPSE_DAYS:
                i.status, changed = "lapsed", True
        if changed:
            self.save()
        return sorted((i for i in self.items(key) if i.status == "open"),
                      key=lambda i: i.day)

    def add(self, key: str, kind: str, text: str, day: str, evidence: str,
            now: datetime) -> Item | None:
        iid = _iid(kind, text, day)
        mine = self._items.setdefault(key, [])
        if any(i.id == iid or (i.status == "open" and i.kind == kind
                               and i.day == day) for i in mine):
            return None
        item = Item(iid, kind, text, day, evidence, now.isoformat())
        mine.append(item)
        return item

    def close(self, key: str, iid: str, status: str, outcome: str = "") -> bool:
        for i in self.items(key):
            if i.id == iid and i.status == "open":
                i.status, i.outcome = status, outcome
                return True
        return False

    def mark_asked(self, key: str, iid: str, stage: str, at: datetime) -> None:
        for i in self.items(key):
            if i.id == iid:
                i.asked.append({"stage": stage, "at": at.isoformat()})
                self.save()
                return


# --- when an item is due ----------------------------------------------------

def stage(item: Item, today: date) -> str | None:
    """Which follow-up today is for this item, or None."""
    gap = (item.when - today).days
    if item.kind == HERS:
        return "day" if gap == 0 else None
    if gap == 1:
        return "before"
    if gap == 0:
        return "day"
    if -AFTER_DAYS <= gap < 0:
        return "after"
    return None


def due_today(items: list[Item], now: datetime, answered) -> Item | None:
    """The one item an opener may follow up today, or None.

    `answered(at)` says whether the opener sent at `at` got a reply: True,
    False, or None while that is not yet known. One item a day, each stage
    once, and once a follow-up went unanswered the item is left alone.
    """
    today = now.date()
    if any(a["at"][:10] == today.isoformat() for i in items for a in i.asked):
        return None
    for i in items:
        st = stage(i, today)
        if st is None or any(a["stage"] == st for a in i.asked):
            continue
        if any(answered(datetime.fromisoformat(a["at"])) is False for a in i.asked):
            continue
        return i
    return None


_STAGE_LINE = {
    "before": "是明天的事，今天可以提一句",
    "day": "就是今天",
    "after": "是{n}天前的事，他还没说后来怎么样",
}


def _when_cn(d: date) -> str:
    return f"{d.month}月{d.day}日"


def opener_block(item: Item | None, now: datetime) -> tuple[str, str]:
    """The opener's block and the short id it shows, or ("", "")."""
    if item is None:
        return "", ""
    st = stage(item, now.date())
    whose = "你答应他的" if item.kind == HERS else "他说过的"
    when = _STAGE_LINE[st].format(n=(now.date() - item.when).days)
    return ("## 今天可以跟他接上的一件事（这件比上面的都要紧，这次优先就它开口）\n"
            f"- A1｜{whose}：{item.text}（{_when_cn(item.when)}，{when}）\n\n"), "A1"


def chat_block(items: list[Item], today: date) -> str:
    """What is ahead of them, for her replies: his plans and her promises."""
    near = [i for i in items if (i.when - today).days >= -AFTER_DAYS][:5]
    if not near:
        return ""
    lines = "\n".join(
        f"- {_when_cn(i.when)}："
        f"{i.text}（{'你答应他的' if i.kind == HERS else '他说过的'}）"
        for i in near)
    return "## 你们接下来的事（他说过的计划、你答应他的）\n" + lines


# --- the pass after a conversation -----------------------------------------

_PROMPT = """你在替{agent}记一份「接下来要跟进的事」。下面是刚聊完的一段对话，和之前记下、还没了结的事。

{stage}【这段对话】（{date_line}）
{dialog}

【还没了结的】
{held}

【记什么】
- 对方说了**他自己**要去、要做、落在具体某一天的事：哪天出发、去哪、考什么、看什么演出（kind 填 "his"）。他转来的消息、他提到的活动，等他说了自己要去才算他的计划。
- {agent}在这段对话里答应对方、要在某一天做的事（kind 填 "hers"）。
- 日子写成具体的几月几号，「明天」「下周六」按对话的日期换算；能落到具体某一天的才记。
- 一句话，30 字以内，写清谁、做什么。【还没了结的】里已经有的，留着就好。

【了结】
- 对方说了某件事的结果（去了、考完了、没去成）：close，outcome 写一句结果。
- 对方说不去了、取消了：drop。改期了：drop 旧的，再 add 新的日子。

输出严格 JSON（直接 {{ 开头），没有就给空列表：
{{"add": [{{"kind": "his", "text": "对方……", "day": "2026-01-31", "evidence": "原话"}}],
 "close": [{{"id": "A1", "outcome": "……", "evidence": "原话"}}],
 "drop": [{{"id": "A2", "evidence": "原话"}}]}}
evidence 从这段对话里原样抄一小段：his 抄对方说的，hers 抄{agent}说的，close 和 drop 抄说出结果的那句。抄不出来就不写这一条。
"""

_WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _norm(t: str) -> str:
    return re.sub(r"[\s「」『』\"'“”‘’]", "", t)


def build_prompt(turns, held: list[Item], *, agent: str,
                 stage: str = "") -> tuple[str, dict[str, str]]:
    first, last = turns[0].ts, turns[-1].ts
    date_line = f"{first:%Y年%-m月%-d日} {_WEEKDAYS[first.weekday()]}"
    if last.date() != first.date():
        date_line += f" 到 {last:%-m月%-d日}"
    dialog = "\n".join(
        f"  {t.ts:%H:%M} {'对方' if t.role == 'user' else agent}：{line.strip()}"
        for t in turns for line in t.content.splitlines() if line.strip())
    ids = {f"A{n + 1}": i.id for n, i in enumerate(held)}
    lines = "\n".join(
        f"  {sid}. {_when_cn(i.when)} {i.text}（{'她答应的' if i.kind == HERS else '他的计划'}）"
        for sid, i in zip(ids, held)) or "  （没有）"
    return _PROMPT.format(agent=agent, date_line=date_line, dialog=dialog,
                          held=lines, stage=f"{stage}\n\n" if stage else ""), ids


@dataclass
class Edits:
    add: list[dict] = field(default_factory=list)
    close: list[tuple[str, str]] = field(default_factory=list)   # (id, outcome)
    drop: list[str] = field(default_factory=list)


def parse_edits(text: str, ids: dict[str, str], *, his: list[str],
                hers: list[str], said_on: date,
                kinds: dict[str, str] | None = None) -> Edits:
    """Edits from the model's JSON, each checked against what was said.

    An add must quote its speaker, and a day must be a real date no earlier
    than the conversation: a plan already behind it is not something to
    follow up. Closing his plan must quote him — her 「那你去成了吧」 is a
    guess, not his answer; closing her promise may quote either of them.
    """
    kinds = kinds or {}
    out = Edits()
    try:
        data = lenient_json.loads(text[text.find("{"):text.rfind("}") + 1])
    except Exception:
        return out
    if not isinstance(data, dict):
        return out
    his_n, hers_n = [_norm(w) for w in his], [_norm(w) for w in hers]

    def quoted(ev, pool) -> bool:
        q = _norm(str(ev or ""))
        return len(q) >= 2 and any(q in w for w in pool)

    for a in data.get("add") or []:
        if not isinstance(a, dict):
            continue
        kind = a.get("kind")
        body = str(a.get("text") or "").strip()
        try:
            day = date.fromisoformat(str(a.get("day") or "").strip())
        except ValueError:
            continue
        pool = his_n if kind == HIS else hers_n if kind == HERS else None
        if not body or pool is None or day < said_on \
                or not quoted(a.get("evidence"), pool):
            continue
        out.add.append({"kind": kind, "text": body, "day": day.isoformat(),
                        "evidence": str(a.get("evidence")).strip()})
    def ending(o) -> str | None:
        iid = ids.get(str(o.get("id") or "").strip().upper()) \
            if isinstance(o, dict) else None
        pool = his_n if kinds.get(iid) == HIS else his_n + hers_n
        return iid if iid and quoted(o.get("evidence"), pool) else None

    for c in data.get("close") or []:
        if iid := ending(c):
            out.close.append((iid, str(c.get("outcome") or "").strip()))
    for d in data.get("drop") or []:
        if iid := ending(d):
            out.drop.append(iid)
    return out


async def update(llm, book: AgendaBook, key: str, turns, *, agent: str,
                 now: datetime, model: str | None = None,
                 stage: str = "") -> Edits:
    """Read a finished conversation into the agenda.

    `stage` is her own schedule. Without it 「去看那场周年演出」 was filed as
    another group's show that had come up earlier the same day, three times
    in three.
    """
    if not any(t.role == "user" for t in turns):
        return Edits()
    held = book.open_items(key, now.date())
    prompt, ids = build_prompt(turns, held, agent=agent, stage=stage)
    kwargs = {"model": model} if model else {}
    resp = await llm.complete(
        messages=[{"role": "user", "content": prompt}],
        system="你负责记接下来要跟进的事，输出严格 JSON。",
        max_tokens=1200, temperature=0.2, _debug_purpose="agenda", **kwargs)
    edits = parse_edits(resp.content, ids,
                        his=[t.content for t in turns if t.role == "user"],
                        hers=[t.content for t in turns if t.role != "user"],
                        said_on=turns[0].ts.date(),
                        kinds={i.id: i.kind for i in held})
    for a in edits.add:
        book.add(key, a["kind"], a["text"], a["day"], a["evidence"], now)
    for iid, outcome in edits.close:
        book.close(key, iid, "done", outcome)
    for iid in edits.drop:
        book.close(key, iid, "dropped")
    book.save()
    if edits.add or edits.close or edits.drop:
        print(f"[agenda] {key[-6:]}: +{len(edits.add)} closed {len(edits.close)} "
              f"dropped {len(edits.drop)}", flush=True)
    return edits
