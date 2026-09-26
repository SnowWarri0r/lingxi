"""The rule against reply-shaped openers was throwing away openers.

Matched as bare character prefixes, 嗯/对/那/诶/对了/好/是 rejected 32 of the
230 openers she composed (14%), and none of the 32 read as a reply: 诶 is
how an IM message gets someone's attention (20 — two of them a lottery
deadline that same night), 对了 is "by the way" (7), and 对/那 are the
first character of 对着、那个、那段 (5). A bare acknowledgement answering
nothing is what the rule is for, and it still goes.
"""

import pytest

from lingxi.temporal.proactive import _validate_proactive_opener


@pytest.mark.parametrize("opener", [
    "诶 演出的一般抽选今晚12点就截止了 你申请没有啊",
    "诶我跟你说！今天下午终于把和声对上了",
    "对了 刚才拍的场刊照片发你",
    "对着镜子把那个位置又跑了两遍 还是差一口气",
    "那个你说要录的歌 到底搁到啥时候啊",
    "那段今天总算能一口气跑完了",
    "好热啊今天 练完一身汗",
    "是不是要下雨了 天好闷",
    "哈哈哈 刚看到一个视频笑死",
])
def test_an_opener_that_starts_a_topic_is_kept(opener):
    assert _validate_proactive_opener(opener) is None


@pytest.mark.parametrize("reply", [
    "嗯 今天练了一下午",
    "对，我也这么觉得",
    "好的 那我先去忙了",
    "哦",
    "嗯嗯",
    "是的 我也想去",
    "好吧 那下次",
])
def test_a_bare_acknowledgement_is_still_a_reply(reply):
    assert (_validate_proactive_opener(reply) or "").startswith("opens_with_response_token")
