from lingxi.fewshot.corpus.builder import build_samples


def test_build_samples_real_speech_verbatim():
    title = "各位infj是否有很强的倾诉欲"
    replies = ["有 到了嘴边咽回去", "一个人住 很幸福诶"]
    samples = build_samples(title, replies, topic_id="278916445")
    assert len(samples) == 2
    s = samples[0]
    assert s.corrected_speech == "有 到了嘴边咽回去"   # verbatim, unchanged
    assert s.context_summary == title                  # real thread title
    assert s.inner_thought == ""                        # no model voice
    assert s.source == "corpus"
    assert s.id == "corpus-douban-278916445-0"
    assert "倾诉" in s.tags                             # keyword map fires


def test_build_samples_tags_default_when_no_keyword():
    samples = build_samples("随便聊聊", ["嗯 就这样吧"], topic_id="1")
    assert samples[0].tags == ["日常"]
