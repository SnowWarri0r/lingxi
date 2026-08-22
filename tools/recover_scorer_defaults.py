"""Recover importance scores lost to the JSON quote bug.

Every scorer batch that failed to parse still has its scores — the model
produced them, `json.loads` choked on a Chinese emphasis quote, and the
whole batch silently fell back to DEFAULT_IMPORTANCE. Both the prompt
(which carries the fact ids) and the raw reply are in the debug log, so
the original judgments can be re-read with the lenient parser instead of
paying for a fresh scoring pass.

Only rows whose stored importance still equals the default for their
source are touched — anything since re-scored or hand-set is left alone.

    python tools/recover_scorer_defaults.py            # dry run
    python tools/recover_scorer_defaults.py --apply    # write (backs up first)
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lingxi.facts.models import Source  # noqa: E402
from lingxi.facts.scorer import DEFAULT_IMPORTANCE  # noqa: E402
from lingxi.utils import lenient_json  # noqa: E402

LOG_GLOB = "data/debug/llm_requests/*.jsonl"
_ID_RE = re.compile(r"\bid=([0-9a-f]{32})\b")


def recovered_scores() -> tuple[dict[str, int], collections.Counter]:
    """fact_id -> score, from batches that failed strict parse but parse now."""
    scores: dict[str, int] = {}
    tally = collections.Counter()
    for path in sorted(glob.glob(LOG_GLOB)):
        for line in open(path, encoding="utf-8"):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("purpose") != "importance_scorer":
                continue
            raw = rec.get("response") or ""
            cleaned = lenient_json.strip_fences(raw)
            if not cleaned.startswith("["):
                continue
            try:
                json.loads(cleaned)
                tally["parsed_at_the_time"] += 1
                continue          # this batch was never defaulted
            except Exception:
                pass
            tally["failed_at_the_time"] += 1
            try:
                data = lenient_json.loads(cleaned)
            except Exception:
                tally["still_unparseable"] += 1
                continue
            # The prompt lists the ids that were in this batch; trust the
            # reply only for ids it was actually asked about.
            asked = set()
            for m in rec.get("messages", []):
                asked.update(_ID_RE.findall(str(m.get("content", ""))))
            for item in data if isinstance(data, list) else []:
                if not isinstance(item, dict):
                    continue
                fid, score = item.get("id"), item.get("score")
                if fid not in asked:
                    continue
                try:
                    score = int(score)
                except (TypeError, ValueError):
                    continue
                if 1 <= score <= 10:
                    scores[fid] = score
            tally["recovered_batches"] += 1
    return scores, tally


def apply_to_db(db: Path, scores: dict[str, int], write: bool) -> dict[str, int]:
    counts = collections.Counter()
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    updates: list[tuple[int, str]] = []
    for fid, score in scores.items():
        row = con.execute(
            "SELECT source, importance FROM facts WHERE id = ?", (fid,)
        ).fetchone()
        if row is None:
            counts["not_in_this_db"] += 1
            continue
        try:
            default = DEFAULT_IMPORTANCE.get(Source(row["source"]), 5)
        except ValueError:
            default = 5
        if row["importance"] != default:
            counts["already_non_default"] += 1
            continue
        if row["importance"] == score:
            counts["default_equals_recovered"] += 1
            continue
        updates.append((score, fid))
    counts["to_update"] = len(updates)
    if write and updates:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = db.with_name(f"{db.name}.bak-{stamp}-rescore")
        shutil.copy2(db, backup)
        print(f"  backup -> {backup.name}")
        con.executemany("UPDATE facts SET importance = ? WHERE id = ?", updates)
        con.commit()
    con.close()
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write to the DBs")
    args = ap.parse_args()

    scores, tally = recovered_scores()
    print("scorer batches in the debug log:")
    for k, v in tally.most_common():
        print(f"  {k:24} {v}")
    print(f"\nfact scores recovered: {len(scores)}")
    dist = collections.Counter(scores.values())
    print("  score distribution:", dict(sorted(dist.items())))

    for db in sorted(Path("data/personas").glob("*/facts.db")):
        print(f"\n{db.parent.name}/facts.db")
        counts = apply_to_db(db, scores, args.apply)
        verb = "updated" if args.apply else "would update"
        print(f"  {verb:16} {counts['to_update']}")
        for k in ("already_non_default", "default_equals_recovered",
                  "not_in_this_db"):
            if counts[k]:
                print(f"  {k:16} {counts[k]}")
    if not args.apply:
        print("\ndry run — re-run with --apply to write")


if __name__ == "__main__":
    main()
