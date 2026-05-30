"""Render the 3 dynamic prompt blocks from orchestrator decision + fact retriever.

3 blocks (in fixed order):
- 【你此刻】 — aria.* facts + register hint + topic anchor
- 【你和他】 — user:<recipient_key>.* facts
- 【身边的事】 — npc:*.* + world.* facts

Subject isolation is a structural invariant: facts go into the block
that matches their subject prefix. There is no rendering path that
could place a user:* fact into 【你此刻】.

`skip` removes entire categories from rendering (overrides any matching
fact_query). Empty blocks are dropped.

Token budget: caller composes [persona_block, dynamic_blocks]; this
function returns just the dynamic_blocks string. Caller can check
length and re-render with reduced queries if needed.
"""

from __future__ import annotations

from lingxi.brain.models import OrchestrationDecision, OrchestratorFactQuery
from lingxi.facts.models import Fact, FactType
from lingxi.facts.retriever import FactQuery, FactRetriever


_REGISTER_HINT = {
    "warm":      "状态：温暖好奇——自然投入，听到具体的细节会反应。",
    "curt":      "状态：curt（不太想多聊）——短一句够，按当下状态写多长。",
    "curious":   "状态：curious（被勾起兴趣）——会接着问一个具体细节。",
    "withdrawn": "状态：withdrawn（心里压着事）——沉默是一等选项，一两个字回也对。",
    "flustered": "状态：flustered（被戳到了）——节奏乱、句子可以碎，每片段是通顺的。",
}


def _parse_category(cat: str) -> tuple[str, FactType | None]:
    """'aria.event' → ('aria', FactType.EVENT). '<subject>.<type>'.

    Subject itself may contain colons (e.g. 'user:oc_xxx'), so we split
    on the LAST dot only.
    """
    if "." not in cat:
        return cat, None
    subject, type_str = cat.rsplit(".", 1)
    try:
        return subject, FactType(type_str)
    except ValueError:
        return subject, None


def _block_for(subject: str, recipient_key: str) -> str:
    """Return which of the 3 dynamic blocks this subject belongs in."""
    if subject == "aria":
        return "self"
    if subject == f"user:{recipient_key}":
        return "them"
    return "world"  # npc:* and world and other users


async def render_dynamic_blocks(
    retriever: FactRetriever,
    decision: OrchestrationDecision,
    *,
    recipient_key: str,
) -> str:
    """Produce the 3-section dynamic prompt for this turn."""
    blocks: dict[str, list[str]] = {"self": [], "them": [], "world": []}

    for q in decision.fact_queries:
        if q.category in decision.skip:
            continue
        subject, ftype = _parse_category(q.category)
        facts = await retriever.fetch(FactQuery(
            subject=subject, type=ftype, semantic=q.semantic, limit=q.limit,
        ))
        if not facts:
            continue
        target = _block_for(subject, recipient_key)
        for f in facts:
            ts_label = f.ts.strftime("%m-%d %H:%M")
            blocks[target].append(f"- [{ts_label}] {f.content}")

    sections: list[str] = []

    # 【核心记忆】 — always-present, agent-curated blocks (MemGPT main context).
    core_lines: list[str] = []
    persona_block = await retriever.get_core_block("aria")
    if persona_block and persona_block.content.strip():
        core_lines.append("你自己：\n" + persona_block.content.strip())
    human_block = await retriever.get_core_block(f"user:{recipient_key}")
    if human_block and human_block.content.strip():
        core_lines.append("关于对方：\n" + human_block.content.strip())
    if core_lines:
        sections.append("【核心记忆】（你长期记着的，自己维护的）\n" + "\n\n".join(core_lines))

    # 【你此刻】
    self_lines: list[str] = []
    self_lines.append(_REGISTER_HINT.get(decision.register, _REGISTER_HINT["warm"]))
    if decision.topic_anchor:
        self_lines.append(f"对方话题落点：{decision.topic_anchor}")
    if blocks["self"]:
        self_lines.append("\n你最近的事：")
        self_lines.extend(blocks["self"])
    sections.append("【你此刻】\n" + "\n".join(self_lines))

    # 【你和他】
    if blocks["them"]:
        sections.append(
            "【你和他】（你过去注意到的——对方这轮明说的事实优先）\n"
            + "\n".join(blocks["them"])
        )

    # 【身边的事】
    if blocks["world"]:
        sections.append(
            "【身边的事】（背景知识，话题撞上时自然带）\n"
            + "\n".join(blocks["world"])
        )

    return "\n\n".join(sections)
