"""The orchestrator has to know who she is to read a remark about her past.

Told 「不像你，之前都能冲上去说太好听了」 — the day she met her partner, in
the persona's own background — it read the line as her once praising him and
framed the turn as 「你变了」; the reply then conceded a change of character
she never had. Replayed five times without the background it named the
partner in none; with it, in three.
"""

from lingxi.brain.orchestrator import StateDigest, build_orchestrator_prompt

DIGEST = StateDigest(activity="", mood="", last_lived=[])


def test_her_background_is_in_the_prompt():
    p = build_orchestrator_prompt("不像你诶", DIGEST, {}, agent_name="阿澪",
                                  background="开学那天她冲上去夸了一个陌生人的歌声。")
    assert "【阿澪 是谁】" in p
    assert "开学那天她冲上去夸了一个陌生人的歌声。" in p


def test_the_anchor_leaves_her_character_to_her():
    p = build_orchestrator_prompt("你以前不是这样", DIGEST, {}, agent_name="阿澪")
    assert "是怎样的人、有没有变，由她自己的人设来答" in p
