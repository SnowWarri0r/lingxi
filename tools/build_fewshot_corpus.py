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
        await asyncio.sleep(3.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump({"seeds": seeds}, allow_unicode=True,
                                  sort_keys=False), encoding="utf-8")
    print(f"[corpus] wrote {len(seeds)} samples -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
