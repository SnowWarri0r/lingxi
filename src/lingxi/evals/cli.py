"""`lingxi-eval` — run the case library and compare against the baseline."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import datetime
from pathlib import Path

from lingxi.evals.case import load_all_cases
from lingxi.evals.runner import CaseScore, score_case

CASES_DIR = Path("evals/cases")
BASELINE_PATH = Path("evals/baseline.json")


def load_baseline(path: Path | str = BASELINE_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_baseline(path: Path | str, scores: list[CaseScore], git_sha: str) -> None:
    """Record scored cases. BROKEN cases are skipped on purpose: baking their
    meaningless number in would mask the real one once they are repaired."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    # Merged into what is there: recording a run of two cases used to wipe
    # every other case's baseline.
    data = load_baseline(path)
    data.update({
        s.id: {"fail_rate": s.fail_rate, "pass_rate": s.pass_rate,
               "samples": s.samples, "recorded_at": stamp, "git_sha": git_sha}
        for s in scores if s.verdict != "BROKEN"
    })
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def dump_replies(scores: list[CaseScore], out_dir: Path | str) -> None:
    """Write every sampled reply to disk, one file per case.

    Replies cost real money to produce and `score_case` throws them away
    once the verdict is computed. Diagnosing a case that scores
    unexpectedly means reading the actual replies, and without this a
    throwaway script (and a second paid sampling round) is the only way
    to get at them.

    Layout: `DIR/<case-id>.txt`, one reply per block, separated by a blank
    line and a `---` rule. A case with no replies (e.g. BROKEN, sampled
    zero) writes no file rather than an empty one.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in scores:
        if not s.replies:
            continue
        text = "\n\n---\n\n".join(s.replies)
        (out_dir / f"{s.id}.txt").write_text(text, encoding="utf-8")


def fisher_p(a: int, n1: int, b: int, n2: int) -> float:
    """Two-sided Fisher exact p for a/n1 failures against b/n2.

    At 12-20 samples a case moves 10-20 points on noise alone; the Δ column
    printed as a bare number read as a change every time.
    """
    from math import comb
    k, n = a + b, n1 + n2
    if n == 0:
        return 1.0
    def prob(x):
        return comb(n1, x) * comb(n2, k - x) / comb(n, k)
    lo, hi = max(0, k - n2), min(k, n1)
    p_obs = prob(a)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1)
                        if prob(x) <= p_obs * (1 + 1e-9)))


def _fraction(rate: float, samples: int) -> str:
    return f"{round(rate * samples)}/{samples}"


def _group_rows(scores: list[CaseScore], baseline: dict, groups: dict) -> list[str]:
    rows = []
    for g in sorted({v for v in groups.values() if v}):
        mine = [s for s in scores if groups.get(s.id) == g and s.verdict != "BROKEN"]
        if not mine:
            continue
        n = sum(s.samples for s in mine)
        f = sum(round(s.fail_rate * s.samples) for s in mine)
        p_ = sum(round(s.pass_rate * s.samples) for s in mine)
        base = [baseline[s.id] for s in mine if s.id in baseline]
        line = f"{'[' + g + ']':<22}{'pooled':<10}{f'{f}/{n}':<9}{f'{p_}/{n}':<9}"
        if len(base) == len(mine):
            bn = sum(int(b.get("samples") or 0) for b in base)
            bf = sum(round(b["fail_rate"] * int(b.get("samples") or 0)) for b in base)
            if bn:
                pv = fisher_p(f, n, bf, bn)
                line += (f"{bf / bn:<11.0%}{(f / n - bf / bn) * 100:+.0f}pp "
                         f"(p={pv:.2f})")
        rows.append(line)
    return rows


def format_table(scores: list[CaseScore], baseline: dict,
                 groups: dict | None = None) -> str:
    lines = [f"{'case':<22}{'verdict':<10}{'fail':<9}{'pass':<9}"
             f"{'baseline':<11}{'Δ'}"]
    for s in scores:
        if s.verdict == "BROKEN":
            lines.append(f"{s.id:<22}{'BROKEN':<10}前提不再成立：{s.premise_error}")
            continue
        base = baseline.get(s.id)
        if base is None:
            base_col, delta = "—", "new"
        else:
            base_col = f"{base['fail_rate']:.0%}"
            delta = f"{(s.fail_rate - base['fail_rate']) * 100:+.0f}pp"
            bn = int(base.get("samples") or 0)
            if bn and s.samples:
                p = fisher_p(round(s.fail_rate * s.samples), s.samples,
                             round(base["fail_rate"] * bn), bn)
                delta += f" (p={p:.2f})" if p < 0.2 else " (噪声内)"
        lines.append(
            f"{s.id:<22}{s.verdict:<10}"
            f"{_fraction(s.fail_rate, s.samples):<9}"
            f"{_fraction(s.pass_rate, s.samples):<9}{base_col:<11}{delta}")
    lines += _group_rows(scores, baseline, groups or {})
    return "\n".join(lines)


def _is_git_ignored(path: Path) -> bool:
    """Whether git would ignore `path`.

    Returns False when git is unavailable or this is not a repo — the caller
    treats that as "refuse", because an unverifiable answer about whether
    personal data would be committed is not a safe answer.
    """
    try:
        r = subprocess.run(["git", "check-ignore", "-q", str(path)],
                           capture_output=True)
    except Exception:
        return False
    return r.returncode == 0


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


async def _run(case_ids: list[str]) -> list[CaseScore]:
    cases = load_all_cases(CASES_DIR)
    if case_ids:
        cases = [c for c in cases if c.id in case_ids]
        if not cases:
            raise SystemExit(f"no case matched: {case_ids}")
    return [await score_case(c) for c in cases]


def main() -> int:
    # Load .env before any config/sampler code reads os.environ, so the API
    # key reaches the sampler even if no other import path happens to load it.
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(prog="lingxi-eval")
    parser.add_argument("cases", nargs="*", help="case id（留空跑全部）")
    parser.add_argument("--baseline", action="store_true",
                        help="把这次的分数记成新基线")
    parser.add_argument("--capture", metavar="RECIPIENT_KEY",
                        help="从线上状态冻出一个 case 骨架")
    parser.add_argument("--turns", type=int, default=8,
                        help="capture 时扒最近几轮对话")
    parser.add_argument("--facts", type=int, default=8,
                        help="capture 时每个 subject 最多抓几条 fact")
    parser.add_argument("--at", metavar="ISO_TIME",
                        help="capture 一个过去的轮次：缓冲区截到这一刻（含他这条），"
                             "事实只取那时已经写下的")
    parser.add_argument("--dump", metavar="DIR",
                        help="把每个 case 采样到的回复写入 DIR/<case-id>.txt")
    args = parser.parse_args()

    if args.capture:
        import os

        import yaml

        from lingxi.evals.capture import capture as capture_turn

        persona_path = os.environ.get(
            "PERSONA_PATH", "config/personas/example_persona.yaml")
        slug = Path(persona_path).stem
        skeleton = asyncio.run(capture_turn(
            args.capture, persona_path=persona_path,
            data_dir=Path("data/personas") / slug, turns=args.turns,
            fact_limit=args.facts,
            upto=datetime.fromisoformat(args.at) if args.at else None))

        # Same-day captures share the "{date}-rename-me" id; pick the next
        # free suffix instead of silently overwriting an earlier capture.
        base_id = skeleton["id"]
        out = CASES_DIR / f"{base_id}.yaml"
        suffix = 2
        while out.exists():
            skeleton["id"] = f"{base_id}-{suffix}"
            out = CASES_DIR / f"{skeleton['id']}.yaml"
            suffix += 1

        out.parent.mkdir(parents=True, exist_ok=True)
        # A capture is a verbatim copy of a real conversation and real facts
        # about a real person — the same class of content as data/, which
        # .gitignore has always refused. Refuse to write it anywhere git
        # would track, so the tool cannot produce a committable artifact even
        # if the ignore rules are later edited. This is not hypothetical: a
        # case carrying 49 personal facts and 24 verbatim turns was committed
        # once and caught only by review, with the repo public.
        if not _is_git_ignored(out):
            print(f"拒绝写入 {out}：git 不会忽略这个路径。\n"
                  f"capture 出来的是真人对话原文和关于真人的事实，"
                  f"跟 data/ 同级，不能进版本库。\n"
                  f"检查 .gitignore 里 evals/cases/ 那几条规则。")
            return 2
        out.write_text(
            yaml.safe_dump(skeleton, allow_unicode=True, sort_keys=False),
            encoding="utf-8")
        print(f"骨架已写入 {out}\n还需手写：symptom、detect、premise")
        return 0

    scores = asyncio.run(_run(args.cases))

    # Exit with error if no cases were found or ran
    if not scores:
        print("没有找到任何案例（evals/cases/ 是空的）")
        return 1

    groups = {c.id: c.group for c in load_all_cases(CASES_DIR)}
    print(format_table(scores, load_baseline(), groups))

    if args.dump:
        dump_replies(scores, args.dump)
        print(f"\n回复已写入 {args.dump}/<case-id>.txt")

    if args.baseline:
        save_baseline(BASELINE_PATH, scores, _git_sha())
        print(f"\n基线已写入 {BASELINE_PATH}")
    return 1 if any(s.verdict != "PASS" for s in scores) else 0
