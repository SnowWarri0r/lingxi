"""Snapshot every persona's data: databases through SQLite's backup API, the
rest as a hard-linked rsync snapshot.

    python scripts/backup_data.py

facts.db holds everything she knows about the people she talks to and had no
copy at all between manual backups — 18% of its rows on 10-22. The backup API
copies a consistent database while the bot is writing to it (WAL mode); a
plain file copy could catch it mid-transaction. Unchanged files in the rsync
snapshot are hard links to the previous one, so a day costs only what changed.

Snapshots go to $LINGXI_BACKUP_DIR (default ~/lingxi-backups/snapshots), one
directory per run. Nothing is ever deleted here.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PERSONAS = REPO / "data" / "personas"
# The shared sticker pool lives outside the persona folders.
STICKERS = Path(os.environ.get("LINGXI_STICKERS_DIR") or REPO / "data" / "stickers")
ROOT = Path(os.environ.get("LINGXI_BACKUP_DIR",
                           Path.home() / "lingxi-backups" / "snapshots"))

# Old one-off copies kept beside the live data; they are backups already.
_EXCLUDE = ["*.bak-*", "*.broken-*", "*.tmp", "*-journal",
            "*.db", "*.db-wal", "*.db-shm",
            "*.sqlite3", "*.sqlite3-wal", "*.sqlite3-shm"]


def backup_db(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(src) as s, sqlite3.connect(dst) as d:
        s.backup(d)
    with sqlite3.connect(dst) as d:
        # One self-contained file: the copy inherits WAL mode from the source.
        d.execute("PRAGMA journal_mode=DELETE")
        ok = d.execute("PRAGMA integrity_check").fetchone()[0]
        if ok != "ok":
            raise RuntimeError(f"{dst}: integrity_check said {ok}")
        has_facts = d.execute(
            "SELECT 1 FROM sqlite_master WHERE name='facts'").fetchone()
        return d.execute("SELECT count(*) FROM facts").fetchone()[0] if has_facts else -1


def _databases(root: Path):
    for pattern in ("*.db", "*.sqlite3"):
        for db in root.rglob(pattern):
            if not any(".bak-" in p or ".broken-" in p
                       for p in db.relative_to(root).parts):
                yield db


def main() -> int:
    """Exit 0 only when every database and file made it into the snapshot.

    The snapshot is built under <stamp>.partial and renamed when complete, so
    a run that dies half way never looks like a good snapshot and is never
    used as the base the next one hard-links against.
    """
    ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    snap, work = ROOT / stamp, ROOT / f"{stamp}.partial"
    previous = sorted(p for p in ROOT.glob("2*")
                      if p.is_dir() and not p.name.endswith(".partial"))
    work.mkdir()
    sources = [(p, p.name) for p in sorted(PERSONAS.iterdir()) if p.is_dir()]
    if STICKERS.is_dir():
        sources.append((STICKERS, "_stickers"))
    failures = []
    for src, name in sources:
        for db in _databases(src):
            rel = db.relative_to(src)
            try:
                rows = backup_db(db, work / name / rel)
                print(f"[backup] {name}/{rel}: {'ok' if rows < 0 else f'{rows} facts'}")
            except Exception as e:
                failures.append(f"{name}/{rel}: {e!r}")
                print(f"[backup] FAILED {name}/{rel}: {e!r}", flush=True)
        cmd = ["rsync", "-a", *[f"--exclude={x}" for x in _EXCLUDE]]
        if previous and (previous[-1] / name).is_dir():
            cmd.append(f"--link-dest={previous[-1] / name}")
        cmd += [f"{src}/", f"{work / name}/"]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            failures.append(f"rsync {name}: exit {r.returncode}")
    if failures:
        print(f"[backup] incomplete, left as {work}: {failures}", flush=True)
        return 1
    work.rename(snap)
    print(f"[backup] snapshot {snap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
