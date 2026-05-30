"""Tests for social.renderer."""

from datetime import datetime, timedelta

import pytest

from lingxi.social.models import NPC, NPCArc, NPCEvent, NPCState, SocialGraph
from lingxi.social.renderer import (
    collect_aria_interactions,
    render_social_section,
)


def make_npc(id: str, name: str = "test", relation: str = "朋友") -> NPC:
    return NPC(id=id, name=name, relation=relation, background=f"{name}的背景")


def make_state(npc_id: str, *, arcs=None, events=None) -> NPCState:
    return NPCState(
        npc_id=npc_id,
        arcs=arcs or [],
        recent_events=events or [],
    )


class TestRender:
    def test_empty_graph_returns_none(self):
        graph = SocialGraph(npcs=[])
        out = render_social_section(graph, {})
        assert out is None

    def test_npc_with_no_arc_no_events_skipped(self):
        graph = SocialGraph(npcs=[make_npc("x")])
        states = {"x": make_state("x")}  # nothing
        out = render_social_section(graph, states)
        assert out is None

    def test_renders_npc_with_active_arc(self):
        graph = SocialGraph(npcs=[make_npc("x", "小敏", "室友")])
        states = {
            "x": make_state(
                "x",
                arcs=[NPCArc(id="a", summary="论文压力", stage="mid", weight=0.8)],
            )
        }
        out = render_social_section(graph, states)
        assert out is not None
        assert "小敏" in out
        assert "室友" in out
        assert "论文压力" in out
        assert "身边的人" in out

    def test_resolved_arcs_not_shown(self):
        graph = SocialGraph(npcs=[make_npc("x")])
        states = {
            "x": make_state(
                "x",
                arcs=[NPCArc(id="a", summary="老事", stage="resolved", weight=0.5)],
            )
        }
        # No active arcs, no events → returns None
        out = render_social_section(graph, states)
        assert out is None

    def test_caps_at_max_npcs(self):
        npcs = [make_npc(f"n{i}", f"人{i}") for i in range(10)]
        graph = SocialGraph(npcs=npcs)
        states = {
            f"n{i}": make_state(
                f"n{i}",
                arcs=[NPCArc(id="a", summary=f"事{i}", stage="mid", weight=0.5)],
            )
            for i in range(10)
        }
        out = render_social_section(graph, states)
        # MAX_NPCS = 4, so at most 4 distinct npc arc summaries rendered
        assert out is not None
        rendered_arc_count = sum(1 for i in range(10) if f"事{i}" in out)
        assert rendered_arc_count == 4

    def test_ranks_high_arc_weight_first(self):
        npcs = [make_npc("low", "Low"), make_npc("high", "High")]
        graph = SocialGraph(npcs=npcs)
        states = {
            "low": make_state(
                "low", arcs=[NPCArc(id="a", summary="x", stage="mid", weight=0.2)],
            ),
            "high": make_state(
                "high", arcs=[NPCArc(id="a", summary="y", stage="mid", weight=0.9)],
            ),
        }
        out = render_social_section(graph, states)
        # High-weight NPC should appear before low-weight in the rendered text
        assert out is not None
        assert out.index("High") < out.index("Low")

    def test_recent_event_shown(self):
        graph = SocialGraph(npcs=[make_npc("x", "小敏", "室友")])
        ts = datetime.now() - timedelta(hours=2)
        states = {
            "x": make_state(
                "x",
                events=[NPCEvent(
                    npc_id="x", ts=ts, type="life",
                    content="跟导师吵了一架", significance=0.6,
                )],
            )
        }
        out = render_social_section(graph, states)
        assert out is not None
        assert "跟导师吵了一架" in out

    def test_aria_interaction_tagged(self):
        graph = SocialGraph(npcs=[make_npc("x", "小敏", "室友")])
        ts = datetime.now() - timedelta(hours=2)
        states = {
            "x": make_state(
                "x",
                events=[NPCEvent(
                    npc_id="x", ts=ts, type="aria_interaction",
                    content="拉你去吃了麻辣烫", significance=0.5,
                )],
            )
        }
        out = render_social_section(graph, states)
        assert out is not None
        assert "和你" in out

    def test_old_events_excluded_from_recent(self):
        # Event older than RECENT_WINDOW_HOURS (72h) shouldn't appear as
        # "最近" but NPC still renders if it has an active arc
        graph = SocialGraph(npcs=[make_npc("x", "小敏")])
        ts_old = datetime.now() - timedelta(days=5)
        states = {
            "x": make_state(
                "x",
                arcs=[NPCArc(id="a", summary="论文", stage="mid", weight=0.7)],
                events=[NPCEvent(
                    npc_id="x", ts=ts_old, type="life",
                    content="老古董事", significance=0.5,
                )],
            )
        }
        out = render_social_section(graph, states)
        assert out is not None
        assert "老古董事" not in out
        assert "论文" in out


class TestCollectAriaInteractions:
    def test_filters_to_aria_interactions_only(self):
        ts = datetime.now() - timedelta(hours=2)
        states = {
            "x": make_state("x", events=[
                NPCEvent(npc_id="x", ts=ts, type="life",
                         content="自己的事", significance=0.3),
                NPCEvent(npc_id="x", ts=ts, type="aria_interaction",
                         content="跟 Aria 吃饭", significance=0.5),
            ])
        }
        out = collect_aria_interactions(states, since=ts - timedelta(hours=1))
        assert len(out) == 1
        assert out[0].content == "跟 Aria 吃饭"

    def test_respects_since_cutoff(self):
        old = datetime.now() - timedelta(days=10)
        new = datetime.now() - timedelta(hours=1)
        states = {
            "x": make_state("x", events=[
                NPCEvent(npc_id="x", ts=old, type="aria_interaction",
                         content="老事", significance=0.5),
                NPCEvent(npc_id="x", ts=new, type="aria_interaction",
                         content="新事", significance=0.5),
            ])
        }
        out = collect_aria_interactions(states, since=datetime.now() - timedelta(hours=2))
        assert len(out) == 1
        assert out[0].content == "新事"
