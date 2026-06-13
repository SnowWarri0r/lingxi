"""Persona self-context for the life-sim loop.

The plan→event→reflect loop is one unified mechanism; the PERSONA drives what
comes out of it. Without this the prompts hardcoded "你是 Aria" and produced
Aria's writer-introspection for every persona (a house catgirl musing about
existential validation systems). build_self_context renders a compact "who you
are" header so a catgirl's loop yields cat moments and Aria's yields a writer's.
"""

from __future__ import annotations


def build_self_context(persona) -> str:
    """A short first-person identity header for life-sim system prompts."""
    ident = persona.identity
    name = (ident.full_name or persona.name or "").strip()
    head = f"你是{name}"
    occ = (ident.occupation or "").strip()
    if occ:
        head += f"（{occ}）"

    parts = [head]

    bg = (ident.background or "").strip()
    if bg:
        # First sentence/line only — enough to set the world, not a wall of text.
        first = bg.replace("\n", " ").split("。")[0].strip()
        if first:
            parts.append(first[:90])

    style = persona.speaking_style
    voice = "、".join(x for x in [
        (style.tone or "").strip(), (style.vocabulary_level or "").strip()
    ] if x)
    if voice:
        parts.append(f"你想事、记事的底色是：{voice}")

    return "。".join(parts) + "。"
