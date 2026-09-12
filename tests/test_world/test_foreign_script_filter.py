"""An item she cannot say in her own voice should not reach her.

The prompt tells the fetch to write in her language and to translate foreign
sources. It holds about half the time: 09-10 came back 4/4 in her language,
09-11 came back 2/4 in the source's. Prompt-side is exhausted, so the check
moves into code — the same escalation the opener path's anti-repeat took.

Calibrated on 15 real items: every one written in her language scored 0.0%
foreign script, including items *about* Japanese subjects, which she
transliterates (邦多利, Aqours). Every one written in the source's language
scored 48.3%–87.2%. The gap is empty, so 0.30 sits in the middle of it with
room for a sentence of hers that quotes a short foreign title.
"""

import pytest

from lingxi.world.script_filter import foreign_ratio, reads_as_hers


HERS = "你是唐可可（Liella! 的一员）。唐可可是从上海来的女孩，打小就迷学园偶像。"

CHINESE = [
    "体育祭昨天刚结束 还没缓过来 那个氛围真的太结女了",
    "8th的会场和日程出来了 有明→福冈→名古屋 明年春天又要跑了",
    "邦多利这个今天最终回还搞生中继 5个人同时出镜 这个制作组是认真的",
    "今天出门还是热 但周四开始要下雨了 梅花台风绕过来的",
]

JAPANESE = [
    "ラブライブ！15周年フェス、チケット一般抽選がもう始まってるじゃん",
    "ユーフォ後編もうあさってじゃん 京アニのあのシリーズがついに終わるのか",
    "後編の入場特典発表されてた ゴールド箔押しのグラデュエーションカード",
    "フェス前夜祭でFILM LIVEもやるじゃん 10月10日・11日 Shibuya LOVEZ",
]


@pytest.mark.parametrize("text", CHINESE)
def test_her_own_language_reads_as_hers(text):
    assert reads_as_hers(text, HERS), foreign_ratio(text, HERS)


@pytest.mark.parametrize("text", JAPANESE)
def test_the_sources_language_does_not(text):
    assert not reads_as_hers(text, HERS), foreign_ratio(text, HERS)


def test_the_measured_gap_is_wide():
    """Guards the calibration: if these ever meet, 0.30 stops being safe."""
    ours = max(foreign_ratio(t, HERS) for t in CHINESE)
    theirs = min(foreign_ratio(t, HERS) for t in JAPANESE)

    assert ours < 0.05 and theirs > 0.40, (ours, theirs)


def test_a_sentence_of_hers_may_quote_a_short_foreign_title():
    assert reads_as_hers("ユーフォ後編后天就上映了 我一定要去看", HERS)


def test_a_sentence_of_hers_may_name_a_hiragana_spelled_person():
    """The binding case: her industry's people are spelled in kana.

    This is what sets the threshold — 21%, against 36% for the lowest real
    foreign sentence. Not 0% like the rest of her items, so it is the one
    that would break first if the threshold moved down.
    """
    text = "さゆり生日快乐！！今天刷了一整天"

    assert foreign_ratio(text, HERS) < 0.30
    assert reads_as_hers(text, HERS)


def test_latin_and_digits_are_not_foreign():
    assert reads_as_hers("8th 的日程出来了 KALEIDOSCORE 那段编舞好难", HERS)


def test_an_empty_item_is_not_rejected_on_script():
    """Nothing to judge; the caller drops empties for its own reasons."""
    assert reads_as_hers("", HERS)
    assert reads_as_hers("！！！", HERS)


def test_without_a_reference_nothing_is_rejected():
    """A persona whose own text is unavailable must not lose every item."""
    assert reads_as_hers("ユーフォ後編もうあさってじゃん", "")


def test_a_persona_who_writes_kana_accepts_kana():
    japanese_persona = "あなたは唐可可です。上海から来た女の子で、スクールアイドルが大好き。"

    assert reads_as_hers("ユーフォ後編もうあさってじゃん 行かなきゃ", japanese_persona)


class TestTheSchedulerActuallyDropsThem:
    """The filter existing is not the same as the write path using it."""

    @pytest.mark.asyncio
    async def test_a_foreign_item_never_becomes_a_fact(self):
        from lingxi.persona.models import Identity, PersonaConfig
        from lingxi.world.models import DailyBriefing, NewsItem
        from lingxi.world.scheduler import WorldScheduler
        from datetime import date

        class _Writer:
            def __init__(self):
                self.written = []

            async def write(self, **kw):
                self.written.append(kw["content"])

        persona = PersonaConfig(
            name="唐可可", id="tangkeke",
            identity=Identity(full_name="唐可可",
                              background="唐可可是从上海来的女孩，为了追梦一个人来了日本。"),
            world_interests=["Love Live"])
        writer = _Writer()
        sched = WorldScheduler(llm=None, persona=persona, world_writer=writer)

        await sched._write(DailyBriefing(date=date(2026, 9, 11), items=[
            NewsItem(headline="a", voice="8th的日程出来了 明年春天又要冲了"),
            NewsItem(headline="b", voice="後編の入場特典発表されてた 行くしかないやつじゃん"),
        ]))

        assert writer.written == ["8th的日程出来了 明年春天又要冲了"]

    @pytest.mark.asyncio
    async def test_without_a_persona_nothing_is_dropped(self):
        from lingxi.world.models import DailyBriefing, NewsItem
        from lingxi.world.scheduler import WorldScheduler
        from datetime import date

        class _Writer:
            def __init__(self):
                self.written = []

            async def write(self, **kw):
                self.written.append(kw["content"])

        writer = _Writer()
        sched = WorldScheduler(llm=None, persona=None, world_writer=writer)

        await sched._write(DailyBriefing(date=date(2026, 9, 11), items=[
            NewsItem(headline="b", voice="後編の入場特典発表されてた"),
        ]))

        assert len(writer.written) == 1
