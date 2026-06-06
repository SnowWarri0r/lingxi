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

# Curated in-register threads, spanning emotional registers so the pool isn't
# all-melancholy. Verified fetchable 2026-06-01 (douban IP-rate-limits hard —
# crawl gently: the 5s delay below + small batches, or it blanket-403s to
# sec.douban for a cooldown window).
THREAD_IDS: list[str] = [
    # 倾诉 / 孤独
    "278916445",   # 各位infj是否有很强的倾诉欲
    "285333796",   # 读博每天都一个人好孤单
    # 开心 / 温暖
    "258942714",   # 个人向快乐清单分享
    "240765736",   # 我很会让自己开心 分享一些快乐技巧
    # 日常碎碎念
    "262798242",   # 你今天过的怎么样
    # 深夜 / 睡不着
    "275295741",   # 如果你还没有睡
    "274809956",   # 收留失眠或者还没有睡的朋友
    # 好奇 / 想问问（241975435 也贴 Aria 的看星星）
    "241975435",   # 我来推荐几个看星空的app
    "195561039",   # 想问问大家 这种清洗器算智商税吗
    # 自由职业（borderline 鸡汤，register 过滤会筛掉长句留短暖句）
    "122528275",   # 一个自由职业者的简单生活
]

OUT = Path("config/fewshot/corpus_seeds.yaml")


async def main() -> None:
    seeds = []
    for tid in THREAD_IDS:
        html = await fetch_topic(tid)
        if html is None:
            print(f"[corpus] {tid}: blocked/dead, skip")
            continue
        title, replies = parse_topic(html)
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
        await asyncio.sleep(5.0)  # gentle — douban throttles bursts hard
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump({"seeds": seeds}, allow_unicode=True,
                                  sort_keys=False), encoding="utf-8")
    print(f"[corpus] wrote {len(seeds)} samples -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
