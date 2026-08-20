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
            fact_limit=args.facts))

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

    print(format_table(scores, load_baseline()))

    if args.baseline:
        save_baseline(BASELINE_PATH, scores, _git_sha())
        print(f"\n基线已写入 {BASELINE_PATH}")
    return 1 if any(s.verdict != "PASS" for s in scores) else 0
