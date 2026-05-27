"""One-shot migration of existing stores into the unified Fact table.

Reads from:
- data/memory/inner_life/state.json     → LifeWriter (recent_events / recent_diary)
- data/memory/relational/*.json         → InferenceWriter (daily_patterns, sweet_moments,
                                           inside_jokes, shared_places, fight_patterns,
                                           pet_names, signature_phrases, relationship_summary)
- data/memory/social/npcs/*/events.jsonl → NPCWriter
- data/memory/world/news/*.json         → WorldWriter (skipped if items list is empty)
- config/personas/example_persona.yaml  → BiographyLoader (biography.life_events)

Idempotent-friendly: dry-run mode prints what would be inserted. Re-running
on top of existing facts.db will create duplicates (we accept that — the
migration is meant to run once during the branch cutover).

Usage:
    uv run python tools/migrate_to_facts.py --dry-run
    uv run python tools/migrate_to_facts.py            # actually migrate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lingxi.facts.models import FactType
from lingxi.facts.store import FactStore
from lingxi.facts.writers.biography import BiographyLoader
from lingxi.facts.writers.inference import InferenceWriter
from lingxi.facts.writers.life import LifeWriter
from lingxi.facts.writers.npc import NPCWriter
from lingxi.facts.writers.world import WorldWriter

SOURCE_DIR = Path("/Users/lovart/agent/data/memory")   # READ from main tree
PERSONA_YAML = Path("/Users/lovart/agent/config/personas/example_persona.yaml")
FACTS_DB = Path(__file__).parent.parent / "data" / "facts.db"  # WRITE to worktree


# ---------------------------------------------------------------------------
# Dry-run shim: counts writes without touching the DB
# ---------------------------------------------------------------------------

class _DryRunStore:
    """Drop-in for FactStore that just counts what would be written."""
    def __init__(self):
        self.count = 0

    async def init(self):
        pass

    async def write(self, fact):
        self.count += 1
        return fact


class _DryRunWriterMixin:
    """Patch writer's store with a dry-run store."""
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _parse_ts(ts_str: str | None, fallback: datetime | None = None) -> datetime:
    if ts_str:
        try:
            dt = datetime.fromisoformat(ts_str)
            # Ensure timezone-aware; treat naive as UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            pass
    return fallback or _now_utc()


def _npc_event_type(ev_type: str) -> FactType:
    """Map NPC event type string to FactType."""
    mapping = {
        "life": FactType.EVENT,
        "emotion": FactType.EMOTION_NOTE,
        "plan": FactType.PLAN,
        "pattern": FactType.PATTERN,
        "opinion": FactType.OPINION,
    }
    return mapping.get(ev_type, FactType.EVENT)


# ---------------------------------------------------------------------------
# Migration functions
# ---------------------------------------------------------------------------

async def migrate_inner_life(
    writer: LifeWriter,
    dry_run: bool,
) -> int:
    state_file = SOURCE_DIR / "inner_life" / "state.json"
    if not state_file.exists():
        print(f"  [SKIP] {state_file} not found")
        return 0

    try:
        data = json.loads(state_file.read_text())
    except json.JSONDecodeError as e:
        print(f"  [SKIP] {state_file} malformed JSON: {e}")
        return 0

    count = 0

    # recent_events — Aria's daily life events
    for ev in data.get("recent_events", []):
        try:
            content = ev.get("content", "").strip()
            if not content:
                continue
            ts = _parse_ts(ev.get("timestamp"))
            tags = list(ev.get("emotional_impact", {}).keys())
            significance = ev.get("significance", 0.8)
            if dry_run:
                count += 1
                continue
            await writer.write(
                subject="aria",
                content=content,
                type=FactType.EVENT,
                ts=ts,
                confidence=float(significance) if significance else 0.8,
                tags=tags,
            )
            count += 1
        except Exception as e:
            print(f"  [WARN] Skipping inner_life event: {e}")

    # recent_diary — diary entries
    for entry in data.get("recent_diary", []):
        try:
            content = entry.get("content", "").strip()
            if not content:
                continue
            ts = _parse_ts(entry.get("timestamp"))
            tags = entry.get("tags", [])
            if dry_run:
                count += 1
                continue
            await writer.write(
                subject="aria",
                content=content,
                type=FactType.EMOTION_NOTE,
                ts=ts,
                confidence=0.9,
                tags=tags,
            )
            count += 1
        except Exception as e:
            print(f"  [WARN] Skipping diary entry: {e}")

    return count


async def migrate_relational(
    writer: InferenceWriter,
    dry_run: bool,
) -> int:
    relational_dir = SOURCE_DIR / "relational"
    if not relational_dir.exists():
        print(f"  [SKIP] {relational_dir} not found")
        return 0

    total = 0

    for json_file in sorted(relational_dir.glob("*.json")):
        if json_file.name.endswith(".bak") or json_file.name.endswith(".bak-predupe"):
            continue

        try:
            data = json.loads(json_file.read_text())
        except json.JSONDecodeError as e:
            print(f"  [SKIP] {json_file.name} malformed JSON: {e}")
            continue

        recipient_key = data.get("recipient_key", "")
        if not recipient_key:
            # Try to derive from filename: "feishu:oc_xxx.json"
            recipient_key = json_file.stem

        # Normalise: subject must match user:[A-Za-z0-9_:-]+
        # "feishu:oc_xxx" → "user:feishu:oc_xxx"
        if not recipient_key.startswith("user:"):
            subject = f"user:{recipient_key}"
        else:
            subject = recipient_key

        count = 0
        last_ts_str = data.get("last_extracted_at")
        fallback_ts = _parse_ts(last_ts_str)

        # daily_patterns → PATTERN
        for item in data.get("daily_patterns", []):
            try:
                content = item.get("pattern", "").strip()
                if not content:
                    continue
                confidence_str = item.get("confidence", "medium")
                conf_map = {"high": 0.85, "medium": 0.6, "low": 0.35}
                confidence = conf_map.get(confidence_str, 0.6)
                ts = _parse_ts(item.get("last_confirmed_at"), fallback_ts)
                if not dry_run:
                    await writer.write(
                        subject=subject,
                        content=content,
                        type=FactType.PATTERN,
                        ts=ts,
                        confidence=confidence,
                        tags=["daily_pattern"],
                    )
                count += 1
            except Exception as e:
                print(f"  [WARN] Skipping daily_pattern: {e}")

        # sweet_moments → EVENT
        for item in data.get("sweet_moments", []):
            try:
                content = item.get("content", "").strip()
                if not content:
                    continue
                ts = _parse_ts(item.get("timestamp"), fallback_ts)
                weight = item.get("weight", "medium")
                conf_map = {"high": 0.85, "medium": 0.7, "low": 0.5}
                confidence = conf_map.get(weight, 0.7)
                if not dry_run:
                    await writer.write(
                        subject=subject,
                        content=content,
                        type=FactType.EVENT,
                        ts=ts,
                        confidence=confidence,
                        tags=["sweet_moment"],
                    )
                count += 1
            except Exception as e:
                print(f"  [WARN] Skipping sweet_moment: {e}")

        # inside_jokes → EVENT
        for item in data.get("inside_jokes", []):
            try:
                content = str(item).strip() if isinstance(item, str) else item.get("content", "").strip()
                if not content:
                    continue
                if not dry_run:
                    await writer.write(
                        subject=subject,
                        content=content,
                        type=FactType.EVENT,
                        ts=fallback_ts,
                        confidence=0.8,
                        tags=["inside_joke"],
                    )
                count += 1
            except Exception as e:
                print(f"  [WARN] Skipping inside_joke: {e}")

        # shared_places → EVENT
        for item in data.get("shared_places", []):
            try:
                name = item.get("name", "").strip()
                significance = item.get("significance", "").strip()
                content = f"{name}：{significance}" if significance else name
                if not content:
                    continue
                ts = _parse_ts(item.get("last_referenced_at"), fallback_ts)
                if not dry_run:
                    await writer.write(
                        subject=subject,
                        content=content,
                        type=FactType.EVENT,
                        ts=ts,
                        confidence=0.75,
                        tags=["shared_place"],
                    )
                count += 1
            except Exception as e:
                print(f"  [WARN] Skipping shared_place: {e}")

        # fight_patterns → PATTERN
        for item in data.get("fight_patterns", []):
            try:
                content = str(item).strip() if isinstance(item, str) else item.get("content", "").strip()
                if not content:
                    continue
                if not dry_run:
                    await writer.write(
                        subject=subject,
                        content=content,
                        type=FactType.PATTERN,
                        ts=fallback_ts,
                        confidence=0.6,
                        tags=["fight_pattern"],
                    )
                count += 1
            except Exception as e:
                print(f"  [WARN] Skipping fight_pattern: {e}")

        # pet_names → PATTERN
        for item in data.get("pet_names", []):
            try:
                content = str(item).strip() if isinstance(item, str) else item.get("content", "").strip()
                if not content:
                    continue
                if not dry_run:
                    await writer.write(
                        subject=subject,
                        content=content,
                        type=FactType.PATTERN,
                        ts=fallback_ts,
                        confidence=0.8,
                        tags=["pet_name"],
                    )
                count += 1
            except Exception as e:
                print(f"  [WARN] Skipping pet_name: {e}")

        # signature_phrases → PATTERN
        for item in data.get("signature_phrases", []):
            try:
                content = str(item).strip() if isinstance(item, str) else item.get("content", "").strip()
                if not content:
                    continue
                if not dry_run:
                    await writer.write(
                        subject=subject,
                        content=content,
                        type=FactType.PATTERN,
                        ts=fallback_ts,
                        confidence=0.75,
                        tags=["signature_phrase"],
                    )
                count += 1
            except Exception as e:
                print(f"  [WARN] Skipping signature_phrase: {e}")

        # relationship_summary → OPINION (only if non-empty)
        summary = data.get("relationship_summary", "").strip()
        if summary:
            try:
                if not dry_run:
                    await writer.write(
                        subject=subject,
                        content=summary,
                        type=FactType.OPINION,
                        ts=fallback_ts,
                        confidence=0.7,
                        tags=["relationship_summary"],
                    )
                count += 1
            except Exception as e:
                print(f"  [WARN] Skipping relationship_summary: {e}")

        print(f"  {json_file.name}: {count} facts → {subject}")
        total += count

    return total


async def migrate_social(
    writer: NPCWriter,
    dry_run: bool,
) -> int:
    npcs_dir = SOURCE_DIR / "social" / "npcs"
    if not npcs_dir.exists():
        print(f"  [SKIP] {npcs_dir} not found")
        return 0

    total = 0

    for npc_dir in sorted(npcs_dir.iterdir()):
        if not npc_dir.is_dir():
            continue
        npc_id = npc_dir.name
        subject = f"npc:{npc_id}"
        events_file = npc_dir / "events.jsonl"

        if not events_file.exists():
            print(f"  [SKIP] {npc_id}/events.jsonl not found")
            continue

        count = 0
        with events_file.open() as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  [WARN] {npc_id} line {lineno}: bad JSON: {e}")
                    continue

                try:
                    content = ev.get("content", "").strip()
                    if not content:
                        continue
                    ts = _parse_ts(ev.get("ts"))
                    ev_type = ev.get("type", "life")
                    fact_type = _npc_event_type(ev_type)
                    significance = ev.get("significance", 0.8)
                    arc_id = ev.get("arc_id")
                    tags = [ev_type]
                    if arc_id:
                        tags.append(arc_id)

                    if not dry_run:
                        await writer.write(
                            subject=subject,
                            content=content,
                            type=fact_type,
                            ts=ts,
                            confidence=float(significance) if significance else 0.8,
                            tags=tags,
                        )
                    count += 1
                except Exception as e:
                    print(f"  [WARN] {npc_id} line {lineno}: {e}")

        print(f"  npc:{npc_id}: {count} events")
        total += count

    return total


async def migrate_world(
    writer: WorldWriter,
    dry_run: bool,
) -> int:
    news_dir = SOURCE_DIR / "world" / "news"
    if not news_dir.exists():
        print(f"  [SKIP] {news_dir} not found")
        return 0

    total = 0

    for json_file in sorted(news_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
        except json.JSONDecodeError as e:
            print(f"  [SKIP] {json_file.name} malformed JSON: {e}")
            continue

        items = data.get("items", [])
        if not items:
            # All current news files are empty — skip silently
            continue

        generated_at_str = data.get("generated_at")
        fallback_ts = _parse_ts(generated_at_str) if generated_at_str else \
            datetime.fromtimestamp(json_file.stat().st_mtime, tz=timezone.utc)

        count = 0
        for item in items:
            try:
                if isinstance(item, str):
                    content = item.strip()
                    ts = fallback_ts
                    tags = []
                elif isinstance(item, dict):
                    content = (item.get("content") or item.get("title") or item.get("summary") or "").strip()
                    ts_str = item.get("ts") or item.get("timestamp") or item.get("published_at")
                    ts = _parse_ts(ts_str, fallback_ts)
                    tags = item.get("tags", [])
                else:
                    continue

                if not content:
                    continue

                if not dry_run:
                    await writer.write(
                        subject="world",
                        content=content,
                        type=FactType.EVENT,
                        ts=ts,
                        confidence=0.9,
                        tags=tags,
                    )
                count += 1
            except Exception as e:
                print(f"  [WARN] {json_file.name} item: {e}")

        if count:
            print(f"  {json_file.name}: {count} world facts")
        total += count

    return total


async def migrate_biography(
    writer: BiographyLoader,
    dry_run: bool,
) -> int:
    if not PERSONA_YAML.exists():
        print(f"  [SKIP] {PERSONA_YAML} not found")
        return 0

    try:
        persona = yaml.safe_load(PERSONA_YAML.read_text())
    except Exception as e:
        print(f"  [SKIP] {PERSONA_YAML} parse error: {e}")
        return 0

    biography = persona.get("biography", {})
    life_events = biography.get("life_events", [])

    if not life_events:
        print("  [SKIP] No biography.life_events found")
        return 0

    # Use Aria's birth year from identity age (28 in 2026 → born ~1997)
    identity = persona.get("identity", {})
    current_age = identity.get("age", 28)
    birth_year = 2026 - current_age  # approximate

    count = 0
    for ev in life_events:
        try:
            content = ev.get("content", "").strip()
            if not content:
                continue
            age = ev.get("age", 0)
            tags = ev.get("tags", [])
            event_year = birth_year + age
            # Use Jan 1 of the event year as the timestamp
            ts = datetime(event_year, 1, 1, tzinfo=timezone.utc)

            if not dry_run:
                await writer.write(
                    subject="aria",
                    content=content,
                    type=FactType.EVENT,
                    ts=ts,
                    confidence=1.0,
                    tags=tags,
                )
            count += 1
        except Exception as e:
            print(f"  [WARN] Skipping biography event: {e}")

    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(dry_run: bool) -> None:
    mode = "DRY-RUN" if dry_run else "LIVE"
    print(f"\n=== migrate_to_facts.py [{mode}] ===\n")

    if dry_run:
        store = _DryRunStore()  # type: ignore[assignment]
    else:
        store = FactStore(FACTS_DB)
        await store.init()
        print(f"DB: {FACTS_DB}\n")

    life_writer = LifeWriter(store)  # type: ignore[arg-type]
    inference_writer = InferenceWriter(store)  # type: ignore[arg-type]
    npc_writer = NPCWriter(store)  # type: ignore[arg-type]
    world_writer = WorldWriter(store)  # type: ignore[arg-type]
    bio_writer = BiographyLoader(store)  # type: ignore[arg-type]

    print("--- inner_life ---")
    n_life = await migrate_inner_life(life_writer, dry_run)
    print(f"  => {n_life} facts\n")

    print("--- relational ---")
    n_relational = await migrate_relational(inference_writer, dry_run)
    print(f"  => {n_relational} facts\n")

    print("--- social/npcs ---")
    n_social = await migrate_social(npc_writer, dry_run)
    print(f"  => {n_social} facts\n")

    print("--- world/news ---")
    n_world = await migrate_world(world_writer, dry_run)
    print(f"  => {n_world} facts\n")

    print("--- biography ---")
    n_bio = await migrate_biography(bio_writer, dry_run)
    print(f"  => {n_bio} facts\n")

    total = n_life + n_relational + n_social + n_world + n_bio
    print(f"=== TOTAL: {total} facts ===")
    if dry_run:
        print("(dry-run — nothing written)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate existing stores into facts.db")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be inserted without writing to DB",
    )
    args = parser.parse_args()
    asyncio.run(main(args.dry_run))
