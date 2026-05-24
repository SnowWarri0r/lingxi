"""Manually trigger one tick of the social scheduler.

Usage:
    uv run python tools/social_tick.py             # tick all NPCs once
    uv run python tools/social_tick.py xiaomin     # tick one NPC by id
    uv run python tools/social_tick.py --inspect   # just show current state, no LLM
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Make src/ importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lingxi.auth.manager import AuthMethod
from lingxi.providers.registry import ProviderRegistry
from lingxi.social.arc_advancer import advance_npc_arcs
from lingxi.social.event_generator import (
    compute_tick_probability,
    generate_events,
)
from lingxi.social.loader import load_social_graph, seed_initial_arcs
from lingxi.social.promoter import SocialPromoter
from lingxi.social.store import SocialStore
from lingxi.utils.config import get_nested, load_config


YAML_PATH = "config/personas/aria_social_graph.yaml"
DATA_DIR = os.environ.get("MEMORY_DATA_DIR", "./data/memory")


async def inspect(graph, store: SocialStore, npc_filter: str | None) -> None:
    for npc in graph.npcs:
        if npc_filter and npc.id != npc_filter:
            continue
        state = await store.load_state(npc.id)
        prob = compute_tick_probability(npc, state)
        print(f"\n=== {npc.id} ({npc.name}) ===")
        print(f"  base_p={npc.base_event_probability:.2f}  computed_p={prob:.2f}")
        print(f"  last_event_at={state.last_event_at}")
        print(f"  arcs ({len(state.arcs)}):")
        for arc in state.arcs:
            print(
                f"    - {arc.id}: stage={arc.stage} weight={arc.weight:.2f} "
                f"events={arc.event_count}"
            )
            print(f"        {arc.summary}")
            if arc.resolution:
                print(f"        resolution: {arc.resolution}")
        print(f"  recent_events ({len(state.recent_events)}):")
        for ev in state.recent_events[-5:]:
            ts = ev.ts.strftime("%m-%d %H:%M")
            print(
                f"    - {ts} [{ev.type}] sig={ev.significance:.2f} "
                f"arc={ev.arc_id or '-'} promoted={ev.promoted_to_aria}"
            )
            print(f"        {ev.content}")


async def tick_one(
    llm, npc, store: SocialStore, promoter: SocialPromoter | None, force: bool
):
    state = await store.load_state(npc.id)
    prob = compute_tick_probability(npc, state)
    print(f"\n>>> {npc.id} ({npc.name})  computed_p={prob:.2f}")
    if not force:
        import random
        if random.random() >= prob:
            print("    skipped (dice roll)")
            return

    print("    generating events...")
    events = await generate_events(llm, npc, state)
    if not events:
        print("    LLM returned 0 events")
        return
    bumped: set[str] = set()
    for ev in events:
        await store.append_event(ev)
        print(
            f"    +event sig={ev.significance:.2f} type={ev.type} "
            f"arc={ev.arc_id or '-'}: {ev.content}"
        )
        if ev.arc_id:
            # Mirror scheduler's _bump_arc_count
            cur = await store.load_state(npc.id)
            for arc in cur.arcs:
                if arc.id == ev.arc_id:
                    arc.event_count += 1
                    break
            await store.save_arcs(npc.id, cur.arcs)
            bumped.add(ev.arc_id)
        if promoter:
            promoted = await promoter.maybe_promote(npc, ev)
            if promoted:
                print(f"      → PUSHED to Aria.recent_events")
    if bumped:
        advanced = await advance_npc_arcs(llm, npc, store)
        if advanced:
            print(f"    arcs advanced: {advanced}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npc_id", nargs="?", help="only tick this NPC")
    parser.add_argument("--inspect", action="store_true", help="show state, no LLM call")
    parser.add_argument("--force", action="store_true", help="bypass dice roll")
    parser.add_argument("--no-push", action="store_true", help="skip promoter")
    args = parser.parse_args()

    graph = load_social_graph(YAML_PATH)
    store = SocialStore(DATA_DIR)
    seeded = await seed_initial_arcs(graph, store)
    if seeded:
        print(f"[init] seeded initial arcs for {seeded} NPC(s)")

    if args.inspect:
        await inspect(graph, store, args.npc_id)
        return

    # Reuse the app's auth flow so OAuth tokens (sk-ant-oat...) just work.
    from lingxi.app import _build_auth_manager
    config = load_config("config/default.yaml")
    ProviderRegistry.register_defaults()
    auth_manager = _build_auth_manager(config)
    provider_name = get_nested(config, "llm", "provider", default="claude")
    auth_method_str = get_nested(config, "llm", "auth_method", default="oauth_pkce")
    model = get_nested(config, "llm", "model", default="claude-sonnet-4-20250514")
    llm = await ProviderRegistry.create_llm_with_auth(
        provider_name,
        auth_manager=auth_manager,
        auth_method=AuthMethod(auth_method_str),
        model=model,
    )

    promoter = None
    if not args.no_push:
        from lingxi.inner_life.store import InnerLifeStore
        inner = InnerLifeStore(Path(DATA_DIR) / "inner_life")
        promoter = SocialPromoter(inner, store, DATA_DIR)

    for npc in graph.npcs:
        if args.npc_id and npc.id != args.npc_id:
            continue
        try:
            await tick_one(llm, npc, store, promoter, args.force)
        except Exception as e:
            print(f"    FAILED: {e}")


if __name__ == "__main__":
    asyncio.run(main())
