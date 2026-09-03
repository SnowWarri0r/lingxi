"""`lingxi-annotate` — show the turns most worth correcting, worst first."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from lingxi.fewshot.models import AnnotationTurn
from lingxi.fewshot.queue import rank_turns


def load_unannotated(turns_dir: Path) -> list[AnnotationTurn]:
    """Every recorded turn nobody has judged yet, newest first.

    An already-annotated turn is already in the pool (or deliberately kept
    out of it); re-showing it spends the attention this is trying to save.
    """
    out: list[AnnotationTurn] = []
    for path in sorted(turns_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # "none" is the stored default, and it is a string — truth-testing it
        # skips every unannotated turn, which is the whole queue.
        if data.get("annotation", "none") not in ("none", None, ""):
            continue
        try:
            out.append(AnnotationTurn.model_validate(data))
        except Exception:
            continue
    out.sort(key=lambda t: t.created_at, reverse=True)
    return out


def _render(ranked, base_url: str, limit: int) -> str:
    lines: list[str] = []
    for n, r in enumerate(ranked[:limit], 1):
        flags = f"  [{'/'.join(r.markers)}]" if r.markers else ""
        lines.append(f"{n:2d}. {r.score:.0f}分  {r.why}{flags}")
        if r.user_message:
            lines.append(f"    他: {r.user_message[:70]}")
        lines.append(f"    她: {r.speech[:110]}")
        lines.append(f"    改写: lingxi-annotate --fix {r.turn_id} '换成你会说的那句'")
        lines.append("")
    return "\n".join(lines)


async def apply(data_dir: Path, turn_id: str, correction: str | None) -> int:
    """Record one judgement, in this process.

    The HTTP route exists but `lingxi-server` runs an app with no engine on
    it, so it answers 503 — pointing at it would have been pointing at
    nothing. Writing here needs no server, and the write is small.
    """
    from lingxi.evals.runner import _main_llm
    from lingxi.fewshot.collector import AnnotationCollector
    from lingxi.fewshot.store import AnnotationStore, FewShotStore
    from lingxi.fewshot.summarizer import AnnotationSummarizer
    from lingxi.providers.embedding import create_embedding_provider
    from lingxi.utils.config import get_nested, load_config
    import os

    cfg = load_config("config/default.yaml")
    embedder = create_embedding_provider(
        kind=get_nested(cfg, "embedding", "provider", default="local"),
        model=(os.environ.get("EMBEDDING_MODEL")
               or get_nested(cfg, "embedding", "model", default=None)),
    )
    if embedder is None:
        print("没有可用的 embedding provider——语料要向量化才能进池。")
        return 1
    dim = len(await embedder.embed("probe"))

    collector = AnnotationCollector(
        annotation_store=AnnotationStore(data_dir),
        fewshot_store=FewShotStore(data_dir, embedding_dim=dim),
        embedder=embedder,
        summarizer=AnnotationSummarizer(await _main_llm()),
    )
    try:
        if correction:
            await collector.record_correction(turn_id, correction)
            print(f"已记下改写：{correction}")
        else:
            await collector.record_positive(turn_id)
            print("已记为正例。")
    except KeyError:
        print(f"找不到这条对话：{turn_id}")
        return 1
    return 0


async def _main(args) -> int:
    if args.fix or args.good:
        return await apply(Path(args.data_dir), args.fix or args.good,
                           args.correction if args.fix else None)

    turns_dir = Path(args.data_dir) / "turns"
    if not turns_dir.is_dir():
        print(f"没有找到 {turns_dir}")
        return 1
    turns = load_unannotated(turns_dir)
    if not turns:
        print("没有待标注的对话——都标过了。")
        return 0
    print(f"待标注 {len(turns)} 条，正在排序…", flush=True)

    from lingxi.evals.runner import _main_llm

    ranked = await rank_turns(turns[:args.scan], await _main_llm())
    if not ranked:
        print("排序没有返回结果。")
        return 1
    print(f"\n最像 AI 的 {min(args.top, len(ranked))} 条"
          f"（共评分 {len(ranked)} 条）：\n")
    print(_render(ranked, args.base_url.rstrip("/"), args.top))
    print("改写比点赞值钱：correction 会作为 corrected_speech 进池，"
          "权重也最高。用你自己的话写，别改我给的占位。")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="lingxi-annotate")
    p.add_argument("--data-dir", default="data/personas/tangkeke/fewshot")
    p.add_argument("--top", type=int, default=15, help="显示多少条")
    p.add_argument("--scan", type=int, default=120, help="最多给多少条打分")
    p.add_argument("--fix", metavar="TURN_ID",
                   help="把这条改写成你会说的话（后面跟改写内容）")
    p.add_argument("correction", nargs="?", help="配合 --fix：她该说的那句")
    p.add_argument("--good", metavar="TURN_ID", help="把这条记为正例")
    p.add_argument("--base-url", default="http://127.0.0.1:8000",
                   help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.fix and not args.correction:
        p.error("--fix 后面要跟一句改写：lingxi-annotate --fix <id> '你会说的话'")
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
