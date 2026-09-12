"""Reject a scanned item written in a script the persona does not write in.

The fetch prompt tells the model to write in her language and to translate
foreign sources. It holds about half the time: one day's fetch came back 4/4
in her language, the next 2/4 in the source's, because the search results pull
the model toward their own. Prompt-side was tried and is exhausted, so the
guarantee moves into code — the same escalation the opener path's anti-repeat
guard took for the same reason.

The test is deliberately about *script*, not language: comparing the item's
character repertoire against the persona's own text needs no language model
and no list of languages, and it catches the failure that actually happens —
a whole sentence arriving in the source's writing system.

Calibrated on 16 real items from a Chinese-writing persona: every item in her
language scored 0%, every item in the source's scored 36%-65%, nothing landed
between. Items *about* Japanese subjects score 0% because she transliterates
them (邦多利, Aqours), and a sentence of hers quoting a katakana title scores 0%
too — her own text names 结ヶ丘 and バンテリンドーム, so katakana is hers. What
separates the two groups is hiragana, which is the grammatical glue of a
Japanese sentence and cannot appear in a Chinese one.

That last point is why the reference is every text field the persona is
authored with rather than the short self-context blurb: drawn from the blurb
alone, whether katakana counted as hers would turn on one character happening
to appear in one sentence of her biography.

Known limit, in the safe direction: shared scripts are never foreign, so a
kana-writing persona would accept a Han-only item. That mis-accepts rather
than mis-rejects, and losing an item she could have said is the worse error.
"""

from __future__ import annotations


# Real items separate 0% (hers) from 36% (the source's). The binding case is
# neither: a sentence of hers naming a hiragana-spelled voice actor —
# 「さゆり生日快乐！！今天刷了一整天」 — scores 21%. So the gap that matters is
# 21% to 36%, and this sits between them.
_FOREIGN_THRESHOLD = 0.30

# Scripts worth telling apart. Latin and digits are excluded on purpose: they
# appear in everyone's text (8th, KALEIDOSCORE, 24度) and carry no signal.
_RANGES: list[tuple[str, int, int]] = [
    ("hiragana", 0x3040, 0x309F),
    ("katakana", 0x30A0, 0x30FF),
    ("han", 0x4E00, 0x9FFF),
    ("hangul", 0xAC00, 0xD7AF),
    ("cyrillic", 0x0400, 0x04FF),
    ("arabic", 0x0600, 0x06FF),
    ("hebrew", 0x0590, 0x05FF),
    ("thai", 0x0E00, 0x0E7F),
    ("devanagari", 0x0900, 0x097F),
]


def _script_of(ch: str) -> str | None:
    code = ord(ch)
    for name, lo, hi in _RANGES:
        if lo <= code <= hi:
            return name
    return None


def _scripts_in(text: str) -> set[str]:
    return {s for s in (_script_of(c) for c in text or "") if s}


def foreign_ratio(item: str, reference: str) -> float:
    """Share of the item's scripted characters written in a script the
    reference text never uses. 0.0 when there is nothing to judge."""
    native = _scripts_in(reference)
    if not native:
        return 0.0
    scripted = [c for c in item or "" if _script_of(c) is not None]
    if not scripted:
        return 0.0
    foreign = sum(1 for c in scripted if _script_of(c) not in native)
    return foreign / len(scripted)


def reads_as_hers(item: str, reference: str) -> bool:
    """Whether this item is written the way the persona writes."""
    return foreign_ratio(item, reference) < _FOREIGN_THRESHOLD


def persona_script_reference(persona) -> str:
    """Her running prose — the sentences she is described in.

    Prose only, and that distinction is the whole point. Her lexicon and her
    anchors carry borrowed scripts as vocabulary: 推し, お渡し会, さゆり,
    《始まりは君の空》 all contain hiragana. Folding those in makes hiragana
    native, and hiragana is the one thing that separates a Japanese sentence
    from a Chinese one carrying Japanese names — so the filter stops
    rejecting anything at all. Measured: with the term lists included, 0 of 6
    known-foreign items were caught.

    A borrowed word appears as a short run inside her sentence; a borrowed
    grammar appears throughout. Only prose shows which is which.
    """
    if persona is None:
        return ""
    parts: list[str] = []
    try:
        from lingxi.persona.self_context import build_self_context
        parts.append(build_self_context(persona))
    except Exception:
        pass
    parts.append(str(getattr(getattr(persona, "identity", None),
                             "background", "") or ""))
    bio = getattr(persona, "biography", None)
    for field in ("background", "summary"):
        parts.append(str(getattr(bio, field, "") or ""))
    return "\n".join(p for p in parts if p)
