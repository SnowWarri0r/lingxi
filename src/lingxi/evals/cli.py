"""`lingxi-eval` — run the case library and compare against the baseline."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from datetime import datetime
from pathlib import Path

from lingxi.evals.case import load_all_cases, load_case
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
    data = {
        s.id: {"fail_rate": s.fail_rate, "pass_rate": s.pass_rate,
               "samples": s.samples, "recorded_at": stamp, "git_sha": git_sha}
        for s in scores if s.verdict != "BROKEN"
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _fraction(rate: float, samples: int) -> str:
    return f"{round(rate * samples)}/{samples}"


def format_table(scores: list[CaseScore], baseline: dict) -> str:
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
        lines.append(
            f"{s.id:<22}{s.verdict:<10}"
            f"{_fraction(s.fail_rate, s.samples):<9}"
            f"{_fraction(s.pass_rate, s.samples):<9}{base_col:<11}{delta}")
    return "\n".join(lines)


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
    parser = argparse.ArgumentParser(prog="lingxi-eval")
    parser.add_argument("cases", nargs="*", help="case id（留空跑全部）")
    parser.add_argument("--baseline", action="store_true",
                        help="把这次的分数记成新基线")
    args = parser.parse_args()

    scores = asyncio.run(_run(args.cases))
    print(format_table(scores, load_baseline()))

    if args.baseline:
        save_baseline(BASELINE_PATH, scores, _git_sha())
        print(f"\n基线已写入 {BASELINE_PATH}")
    return 1 if any(s.verdict != "PASS" for s in scores) else 0
