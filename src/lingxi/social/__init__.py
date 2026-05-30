"""Aria's social graph — NPCs in her life (roommate / mentor / friends / family).

NPCs have ongoing arcs and event logs that evolve on a cron, independent of
the user-Aria conversation. The point is to make Aria's references to "my
roommate / my mom / my advisor" persistent and consistent across weeks,
rather than freshly-invented each turn.

Design: see docs/superpowers/specs/2026-05-22-aria-social-graph-design.md

Data flow is strictly one-way (NPC → Aria → user). User-Aria conversation
never writes back into NPC logs — keeps NPCs as a stable source of truth.
"""
