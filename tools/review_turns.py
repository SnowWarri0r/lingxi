"""Bulk-review unannotated AnnotationTurns from terminal.

Lists all `data/fewshot/turns/*.json` with annotation == "none",
shows user/inner/speech, prompts:
    g                  → record positive
    b                  → record negative (no correction)
    b <correction>     → record user_correction with the text after `b `
    s                  → skip
    q                  → quit

Each annotation goes through the real AnnotationCollector so it lands
in the same FewShotStore as button-driven annotations from Feishu.

Run:
    .venv/bin/python tools/review_turns.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


async def main() -> int:
    from dotenv import load_dotenv
    load_dotenv()

    print("Bootstrapping engine (this loads embeddings, may take a few seconds)...")
    from lingxi.app import create_engine

    engine = await create_engine()
    if engine.annotation_store is None or engine.fewshot_store is None:
        print("✗ annotation_store / fewshot_store 未启用 — 无法标注")
        return 1

    embedder = engine.memory.embedding_provider or (
        engine.fewshot_retriever.embedder
        if engine.fewshot_retriever is not None
        else None
    )
    if embedder is None:
        print("✗ 没有可用的 embedding provider — 检查 ARK_API_KEY / EMBEDDING_MODEL")
        return 1

    from lingxi.fewshot.collector import AnnotationCollector
    from lingxi.fewshot.summarizer import AnnotationSummarizer

    collector = AnnotationCollector(
        annotation_store=engine.annotation_store,
        fewshot_store=engine.fewshot_store,
        embedder=embedder,
        summarizer=AnnotationSummarizer(engine.llm),
    )

    turns_dir = Path(engine.annotation_store.turns_dir)
    files = sorted(turns_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    pending = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("annotation", "none") == "none":
                pending.append(data)
        except Exception:
            continue

    if not pending:
        print("\n✓ 没有未标注的轮次")
        return 0

    print(f"\n找到 {len(pending)} 条未标注轮次。")
    print("命令: g=👍像 / b=👎不像 / b <text>=改成 text / s=跳过 / q=退出\n")

    n_good = n_bad = n_correction = 0

    for i, t in enumerate(pending, 1):
        print(f"\n--- {i}/{len(pending)} | {t['created_at'][:16]} | {t['turn_id'][:8]} ---")
        if t.get("user_message"):
            print(f"用户: {t['user_message']}")
        if t.get("inner_thought"):
            inner = t["inner_thought"].strip().replace("\n", " ")[:300]
            print(f"内心: {inner}")
        print(f"说出: {t['speech']}")
        print()

        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n中断，退出")
            break

        if cmd == "q":
            break
        if cmd == "s" or not cmd:
            continue

        if cmd == "g":
            try:
                await collector.record_positive(t["turn_id"])
                n_good += 1
                print("  ✓ 👍 入池")
            except Exception as e:
                print(f"  ✗ {e}")
        elif cmd.startswith("b"):
            correction = cmd[1:].strip() if len(cmd) > 1 else ""
            try:
                if correction:
                    await collector.record_correction(t["turn_id"], correction)
                    n_correction += 1
                    print(f"  ✓ ✏️ 「{correction}」入池")
                else:
                    await collector.record_negative(t["turn_id"])
                    n_bad += 1
                    print("  ✓ 👎 标记")
            except Exception as e:
                print(f"  ✗ {e}")
        else:
            print("  ?  未知命令，跳过")

    print(f"\n本次：{n_good} 👍 / {n_correction} ✏️ / {n_bad} 👎")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
