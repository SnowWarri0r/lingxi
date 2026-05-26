"""One-shot cleanup: dedup near-duplicate events in inner_state.

Run once after deploying the dedup fix to clean historical pollution.
Idempotent — running again on already-clean state is a no-op.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lingxi.inner_life.simulator import _is_near_duplicate_event
from lingxi.inner_life.store import InnerLifeStore


DATA_DIR = Path("data/memory")  # store appends "/inner_life" itself


async def main():
    store = InnerLifeStore(DATA_DIR)
    state = await store.load_state()

    original = list(state.recent_events)
    kept: list = []
    dropped: list = []

    for ev in original:
        if _is_near_duplicate_event(ev.content, kept):
            dropped.append(ev)
        else:
            kept.append(ev)

    if not dropped:
        print(f"No duplicates found in {len(original)} events. Nothing to do.")
        return

    print(f"Found {len(dropped)} near-duplicates in {len(original)} events:")
    for ev in dropped:
        print(f"  DROP [{ev.timestamp.strftime('%m-%d %H:%M')}] {ev.content[:60]}")
    print()
    print(f"Keeping {len(kept)} events. Saving...")

    def _mutate(s):
        s.recent_events = kept

    await store.update_state(_mutate)
    print("done.")


if __name__ == "__main__":
    asyncio.run(main())
