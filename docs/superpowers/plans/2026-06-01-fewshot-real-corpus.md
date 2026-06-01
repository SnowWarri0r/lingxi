# Fewshot 真人语料管线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** 从豆瓣小组帖抓真人中文碎句,确定性地 curate 成 FewShotSample(corrected_speech = 逐字真句),给 Aria 补反翻译腔的真人正例。

**Architecture:** 离线管线 `src/lingxi/fewshot/corpus/`(douban 抓取+解析 → register 过滤 → 去身份化 → builder 成 FewShotSample)+ 离线脚本 `tools/build_fewshot_corpus.py` 写 `config/fewshot/corpus_seeds.yaml`,bootstrap 时和手写 seeds 一起加载。**全程零 LLM**(context=真帖标题、speech=真回复逐字、tags=关键词映射),杜绝翻译腔从生成侧门混入。

**Tech Stack:** Python 3.12 async, httpx, 正则解析(无 bs4/lxml), pytest。测试 `.venv/bin/python -m pytest`。

> **铁律:** `corrected_speech` 永远是抓来的真人原句,逐字不改写。管线只允许标 context/tags 这类检索元数据,且这些也只用确定性规则(真标题/关键词),不调模型。见 memory `feedback_fewshot_real_corpus_only`。

---

## 真实 DOM(已实地验证 2026-06-01)
帖子页 `https://www.douban.com/group/topic/<id>/`,服务端渲染,UA 带桌面 Chrome 即 200。
- **标题:** `<title>...</title>`(豆瓣会带后缀,需清洗)。
- **OP 正文:** `<div class="topic-richtext">文字...`(文字直跟,需 strip 标签)。
- **回复:** `<div class="reply-content">\s*<div class="markdown">...<p>文字</p>...</div>` —— 一帖 ~60 条,**有重复需去重**。
- **反爬:** 死帖 403/404;若 302 跳 `sec.douban.com` 视为被挡 → 返回 None 退避。

---

### Task C1: douban 抓取 + 解析

**Files:**
- Create: `src/lingxi/fewshot/corpus/__init__.py`(空)
- Create: `src/lingxi/fewshot/corpus/douban.py`
- Test: `tests/test_corpus/__init__.py`(空)
- Test: `tests/test_corpus/test_douban.py`

- [ ] **Step 1: 失败测试** `tests/test_corpus/test_douban.py`

```python
import pytest
from lingxi.fewshot.corpus.douban import fetch_topic, parse_topic

_HTML = '''<html><head><title>各位infj是否有很强的倾诉欲</title></head><body>
<div class="topic-richtext">本人最近有很强的倾诉欲 持续内耗中</div>
<div class="reply-content"><div class="markdown"><p>有 到了嘴边咽回去</p></div></div>
<div class="reply-content"><div class="markdown"><p>有 到了嘴边咽回去</p></div></div>
<div class="reply-content"><div class="markdown"><p>压抑到一定程度，倾诉欲就特别强。</p></div></div>
</body></html>'''


def test_parse_topic_extracts_title_op_replies():
    title, op, replies = parse_topic(_HTML)
    assert title == "各位infj是否有很强的倾诉欲"
    assert "倾诉欲" in op
    # deduped: the two identical replies collapse to one
    assert replies == ["有 到了嘴边咽回去", "压抑到一定程度，倾诉欲就特别强。"]


@pytest.mark.asyncio
async def test_fetch_topic_returns_html():
    async def fake_fetch(url: str) -> tuple[int, str, str]:
        assert "group/topic/123" in url
        return 200, url, _HTML
    html = await fetch_topic("123", fetch=fake_fetch)
    assert "倾诉欲" in html


@pytest.mark.asyncio
async def test_fetch_topic_blocked_returns_none():
    async def fake_fetch(url: str) -> tuple[int, str, str]:
        # simulate sec.douban anti-bot redirect
        return 200, "https://sec.douban.com/c?r=xyz", "challenge"
    assert await fetch_topic("123", fetch=fake_fetch) is None


@pytest.mark.asyncio
async def test_fetch_topic_non_200_returns_none():
    async def fake_fetch(url: str) -> tuple[int, str, str]:
        return 404, url, ""
    assert await fetch_topic("404", fetch=fake_fetch) is None
```

- [ ] **Step 2: 跑确认失败** `cd /Users/lovart/agent-facts-refactor && .venv/bin/python -m pytest tests/test_corpus/test_douban.py -v` → ModuleNotFoundError

- [ ] **Step 3: 实现** `src/lingxi/fewshot/corpus/douban.py`

```python
"""Fetch + parse douban group-topic pages (server-rendered HTML).

Deterministic regex parse (no bs4/lxml installed). fetch is an injectable
async callable returning (status, final_url, text) so tests run offline.
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_TOPIC_URL = "https://www.douban.com/group/topic/{tid}/"

_TITLE = re.compile(r"<title>(.*?)</title>", re.S)
_OP = re.compile(r'<div class="topic-richtext">(.*?)</div>', re.S)
_REPLY = re.compile(
    r'<div class="reply-content">\s*<div class="markdown">(.*?)</div>', re.S)
_TAGS = re.compile(r"<[^>]+>")


async def _httpx_fetch(url: str) -> tuple[int, str, str]:
    import httpx
    async with httpx.AsyncClient(timeout=20, follow_redirects=True,
                                 headers={"User-Agent": _UA}) as c:
        r = await c.get(url)
        return r.status_code, str(r.url), r.text


def _strip(html: str) -> str:
    return re.sub(r"\s+", " ", _TAGS.sub(" ", html)).strip()


def parse_topic(html: str) -> tuple[str, str, list[str]]:
    """Return (title, op_text, replies). Replies tag-stripped + deduped,
    order preserved."""
    tm = _TITLE.search(html)
    # douban titles sometimes carry a site suffix after a separator
    title = _strip(tm.group(1)) if tm else ""
    title = re.split(r"\s*[-|]\s*", title)[0].strip()
    om = _OP.search(html)
    op = _strip(om.group(1)) if om else ""
    seen: set[str] = set()
    replies: list[str] = []
    for block in _REPLY.findall(html):
        text = _strip(block)
        if text and text not in seen:
            seen.add(text)
            replies.append(text)
    return title, op, replies


async def fetch_topic(
    topic_id: str,
    *,
    fetch: Callable[[str], Awaitable[tuple[int, str, str]]] = _httpx_fetch,
) -> str | None:
    """Fetch a topic page. Returns HTML, or None if blocked (sec.douban
    redirect) or non-200 (dead/deleted thread)."""
    status, final_url, text = await fetch(_TOPIC_URL.format(tid=topic_id))
    if status != 200 or "sec.douban.com" in final_url:
        return None
    return text
```

- [ ] **Step 4: 跑确认通过**(4 passed)
- [ ] **Step 5: commit** `git add src/lingxi/fewshot/corpus/__init__.py src/lingxi/fewshot/corpus/douban.py tests/test_corpus/ && git commit -m "feat(corpus): douban topic fetch + parse"`

---

### Task C2: register 过滤 + 去身份化

**Files:**
- Create: `src/lingxi/fewshot/corpus/register.py`
- Create: `src/lingxi/fewshot/corpus/deid.py`
- Test: `tests/test_corpus/test_filters.py`

- [ ] **Step 1: 失败测试** `tests/test_corpus/test_filters.py`

```python
from lingxi.fewshot.corpus.register import clean_and_keep
from lingxi.fewshot.corpus.deid import deidentify


def test_keeps_good_register_line():
    assert clean_and_keep("有 到了嘴边咽回去") == "有 到了嘴边咽回去"
    assert clean_and_keep("一个人住，一个人一间办公室，很幸福诶") is not None


def test_drops_too_short_or_too_long():
    assert clean_and_keep("同意") is None            # pure agreement / too short
    assert clean_and_keep("好" ) is None
    assert clean_and_keep("字" * 60) is None          # essay-length


def test_drops_no_spoken_texture():
    # complete, marker-less declarative with no 语气词/ellipsis → drop
    assert clean_and_keep("数据库连接池的最大连接数应当配置为五十") is None


def test_drops_links_at_hashtag_ads():
    assert clean_and_keep("看这个 http://t.cn/abc") is None
    assert clean_and_keep("@张三 你看看") is None
    assert clean_and_keep("#深夜emo# 来了") is None
    assert clean_and_keep("入手了这款面霜 链接在评论") is None


def test_deid_strips_handles_and_drops_locatable():
    assert deidentify("有 到了嘴边咽回去") == "有 到了嘴边咽回去"
    # @handle stripped, remaining kept if still in-register
    assert deidentify("@小明 嗯 我也是") == "嗯 我也是"
    # locatable personal disclosure → drop
    assert deidentify("我在北京大学读博 导师姓王") is None
```

- [ ] **Step 2: 跑确认失败**

- [ ] **Step 3: 实现** `src/lingxi/fewshot/corpus/register.py`

```python
"""Register filter: keep only short, spoken-texture, first-person-ish lines.

Heuristic + deterministic. The goal is anti-翻译腔 cadence (碎句/省主语/语气词),
so we keep lines with spoken markers and drop essays, ads, links, agreement
filler. Returns the cleaned line or None to drop.
"""

from __future__ import annotations

import re

_MIN, _MAX = 4, 40
_SPOKEN = re.compile(r"[啊吧呢啦诶嗯喔哈呀嘛哦唉]|…|。。|\.\.\.")
_AGREEMENT = {"同意", "对", "对对", "是的", "+1", "赞", "同", "顶", "嗯嗯", "哈哈哈"}
_DROP_MARKERS = re.compile(
    r"https?://|[@#]|回复\s|￥|\d+元|链接|入手|测评|种草|推荐|优惠|代购|私信|vx|微信")
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]")


def clean_and_keep(line: str) -> str | None:
    line = (line or "").strip()
    if not line:
        return None
    if _DROP_MARKERS.search(line):
        return None
    # strip emoji for the length/texture test (but the kept line keeps them out)
    bare = _EMOJI.sub("", line).strip()
    if not bare or bare in _AGREEMENT:
        return None
    if not (_MIN <= len(bare) <= _MAX):
        return None
    # spoken texture: a 语气词/ellipsis OR a short clause ending in 。
    if not (_SPOKEN.search(bare) or (bare.endswith("。") and len(bare) <= 20)):
        return None
    return bare
```

`src/lingxi/fewshot/corpus/deid.py`:

```python
"""De-identification: keep texture, drop identity.

Strip @handles; drop any line carrying a locatable personal disclosure
(named school/org + role, contact info). Conservative — when in doubt, drop.
"""

from __future__ import annotations

import re

_HANDLE = re.compile(r"@\S+\s*")
_LOCATABLE = re.compile(
    r"(大学|学院|公司|医院|中学).{0,6}(读|上班|工作|读博|读研|实习)"
    r"|(导师|老板|领导)\s*姓"
    r"|微信|vx|qq|电话|手机号|身份证")


def deidentify(line: str) -> str | None:
    line = _HANDLE.sub("", line or "").strip()
    if not line:
        return None
    if _LOCATABLE.search(line):
        return None
    return line
```

- [ ] **Step 4: 跑确认通过**
- [ ] **Step 5: commit** `git add src/lingxi/fewshot/corpus/register.py src/lingxi/fewshot/corpus/deid.py tests/test_corpus/test_filters.py && git commit -m "feat(corpus): register filter + de-identification"`

---

### Task C3: builder + 离线脚本 + bootstrap 接线

**Files:**
- Create: `src/lingxi/fewshot/corpus/builder.py`
- Create: `tools/build_fewshot_corpus.py`
- Modify: `src/lingxi/app.py`(bootstrap 也加载 corpus_seeds.yaml)
- Test: `tests/test_corpus/test_builder.py`

FewShotSample 字段(见 `src/lingxi/fewshot/models.py`):`id, inner_thought, original_speech=None, corrected_speech, context_summary, tags, recipient_key=None, source`。

- [ ] **Step 1: 失败测试** `tests/test_corpus/test_builder.py`

```python
from lingxi.fewshot.corpus.builder import build_samples


def test_build_samples_real_speech_verbatim():
    title = "各位infj是否有很强的倾诉欲"
    replies = ["有 到了嘴边咽回去", "一个人住 很幸福诶"]
    samples = build_samples(title, replies, topic_id="278916445")
    assert len(samples) == 2
    s = samples[0]
    assert s.corrected_speech == "有 到了嘴边咽回去"   # verbatim, unchanged
    assert s.context_summary == title                  # real thread title
    assert s.inner_thought == ""                        # no model voice
    assert s.source == "corpus"
    assert s.id == "corpus-douban-278916445-0"
    # tag keyword map fires on 倾诉
    assert "倾诉" in s.tags


def test_build_samples_tags_default_when_no_keyword():
    samples = build_samples("随便聊聊", ["嗯 就这样吧"], topic_id="1")
    assert samples[0].tags == ["日常"]
```

- [ ] **Step 2: 跑确认失败**

- [ ] **Step 3: 实现** `src/lingxi/fewshot/corpus/builder.py`

```python
"""Turn (title, real replies) into FewShotSamples — deterministic, no LLM.

corrected_speech is the verbatim real reply; context_summary is the real
thread title; tags come from a keyword map. inner_thought is left empty so no
model-generated voice enters the demonstration pool.
"""

from __future__ import annotations

from lingxi.fewshot.models import FewShotSample

# emotion/topic keyword → tag. First match wins; default 日常.
_TAG_MAP = [
    ("倾诉", "倾诉"), ("孤独", "孤独"), ("孤单", "孤独"), ("内耗", "内耗"),
    ("失眠", "失眠"), ("emo", "emo"), ("难受", "难过"), ("哭", "难过"),
    ("分享", "分享"), ("独处", "独处"), ("一个人", "独处"), ("焦虑", "焦虑"),
    ("幸福", "温暖"), ("喜欢", "温暖"),
]


def _tags(title: str, line: str) -> list[str]:
    hay = title + " " + line
    for kw, tag in _TAG_MAP:
        if kw in hay:
            return [tag]
    return ["日常"]


def build_samples(title: str, replies: list[str], *, topic_id: str
                  ) -> list[FewShotSample]:
    out: list[FewShotSample] = []
    for i, line in enumerate(replies):
        out.append(FewShotSample(
            id=f"corpus-douban-{topic_id}-{i}",
            inner_thought="",
            corrected_speech=line,
            context_summary=title,
            tags=_tags(title, line),
            recipient_key=None,
            source="corpus",
        ))
    return out
```

- [ ] **Step 4: 跑确认通过**

- [ ] **Step 5: 离线脚本** `tools/build_fewshot_corpus.py`

```python
"""Offline: build real-corpus fewshot seeds from a curated douban thread list.

Pipeline (deterministic, no LLM): fetch_topic -> parse_topic -> register filter
-> de-identify -> build_samples -> write config/fewshot/corpus_seeds.yaml.

Curate THREAD_IDS by TOPIC (emo/独处/倾诉/失眠/深夜/自由职业/文艺) — thread-level
register gating matters more than per-line filtering. Run manually.

Usage: .venv/bin/python tools/build_fewshot_corpus.py
"""

import asyncio
from pathlib import Path

import yaml

from lingxi.fewshot.corpus.douban import fetch_topic, parse_topic
from lingxi.fewshot.corpus.register import clean_and_keep
from lingxi.fewshot.corpus.deid import deidentify
from lingxi.fewshot.corpus.builder import build_samples

# Curated in-register threads (emo / 独处 / 倾诉). Verified fetchable 2026-06-01.
THREAD_IDS: list[str] = [
    "278916445",   # 各位infj是否有很强的倾诉欲
    "285333796",   # 读博每天都一个人好孤单
    "312473583",   # 突然闲下来不知道做什么了
]

OUT = Path("config/fewshot/corpus_seeds.yaml")


async def main() -> None:
    seeds = []
    for tid in THREAD_IDS:
        html = await fetch_topic(tid)
        if html is None:
            print(f"[corpus] {tid}: blocked/dead, skip")
            continue
        title, _op, replies = parse_topic(html)
        kept = []
        for r in replies:
            c = clean_and_keep(r)
            if c is None:
                continue
            c = deidentify(c)
            if c is None:
                continue
            kept.append(c)
        for s in build_samples(title, kept, topic_id=tid):
            seeds.append({
                "id": s.id, "context_summary": s.context_summary,
                "inner_thought": s.inner_thought,
                "corrected_speech": s.corrected_speech, "tags": s.tags,
            })
        print(f"[corpus] {tid} {title!r}: kept {len(kept)}")
        await asyncio.sleep(3.0)  # polite rate-limit
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump({"seeds": seeds}, allow_unicode=True,
                                  sort_keys=False), encoding="utf-8")
    print(f"[corpus] wrote {len(seeds)} samples -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 6: bootstrap 接线** — `src/lingxi/app.py`,在 `added = await engine.bootstrap_fewshot_seeds()`(~line 311)之后加:

```python
        corpus_path = "config/fewshot/corpus_seeds.yaml"
        if Path(corpus_path).exists():
            added += await engine.bootstrap_fewshot_seeds(seeds_path=corpus_path)
```
(确认 `Path` 已 import;`bootstrap_fewshot_seeds` 按 id 去重,重复运行安全。)

- [ ] **Step 7: 跑全 corpus 套件 + import 烟雾**
`cd /Users/lovart/agent-facts-refactor && .venv/bin/python -m pytest tests/test_corpus/ -q && .venv/bin/python -m py_compile tools/build_fewshot_corpus.py && .venv/bin/python -c "import lingxi.app"`

- [ ] **Step 8: commit** `git add src/lingxi/fewshot/corpus/builder.py tools/build_fewshot_corpus.py src/lingxi/app.py tests/test_corpus/test_builder.py && git commit -m "feat(corpus): builder + offline build script + bootstrap wiring"`

---

## Self-Review
- corrected_speech 全程逐字真句,管线零 LLM。✓
- register 过滤 + 去身份化覆盖 链接/@/#/种草/附和/过长/无口语质感/可定位身份。✓
- builder 字段对齐 FewShotSample;bootstrap 按 id 去重幂等。✓
- 脚本空/死帖 graceful skip,限速 3s。✓
- 跑完脚本后需人工抽检 corpus_seeds.yaml 再决定是否上线(写进 handoff,不自动重启 bot)。
