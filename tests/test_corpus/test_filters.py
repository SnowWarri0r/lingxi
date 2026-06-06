from lingxi.fewshot.corpus.register import clean_and_keep
from lingxi.fewshot.corpus.deid import deidentify


def test_keeps_good_register_line():
    assert clean_and_keep("有 到了嘴边咽回去") == "有 到了嘴边咽回去"
    assert clean_and_keep("一个人住，一个人一间办公室，很幸福诶") is not None


def test_drops_pure_filler_acknowledgment():
    assert clean_and_keep("嗯嗯好滴！") is None
    assert clean_and_keep("哦哦好滴哈哈") is None
    assert clean_and_keep("可以呀！") is None
    assert clean_and_keep("好的好的") is None


def test_filler_reject_keeps_gems_with_particle_chars():
    # gems carry filler-set chars (好/啊/唉) BUT also real content + spoken
    # markers → must survive the filler reject
    assert clean_and_keep("唉，孤独只能自己调解了") is not None
    assert clean_and_keep("我感觉好爽啊……") is not None


def test_drops_too_short_or_too_long():
    assert clean_and_keep("同意") is None
    assert clean_and_keep("好") is None
    assert clean_and_keep("字" * 60) is None


def test_drops_no_spoken_texture():
    assert clean_and_keep("数据库连接池的最大连接数应当配置为五十") is None


def test_drops_links_at_hashtag_ads():
    assert clean_and_keep("看这个 http://t.cn/abc") is None
    assert clean_and_keep("@张三 你看看") is None
    assert clean_and_keep("#深夜emo# 来了") is None
    assert clean_and_keep("入手了这款面霜 链接在评论") is None


def test_keeps_short_clause_ending_with_period():
    # 压抑到一定程度，倾诉欲就特别强。 — ends in 。, <=20 chars → spoken-enough
    assert clean_and_keep("压抑到一定程度倾诉欲就特别强。") is not None


def test_deid_strips_handles_and_drops_locatable():
    assert deidentify("有 到了嘴边咽回去") == "有 到了嘴边咽回去"
    assert deidentify("@小明 嗯 我也是") == "嗯 我也是"
    assert deidentify("我在北京大学读博 导师姓王") is None
