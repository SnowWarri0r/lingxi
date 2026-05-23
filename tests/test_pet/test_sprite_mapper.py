"""Tests for sprite_mapper — state → sprite filename.

The pet's only "thinking" is which image to show. This mapping has to
match priority order in real time, so getting the priority wrong (e.g.
emotion winning over engagement_mode) would be very visible — pet shows
'happy' while engine outputs CURT replies.
"""

from lingxi.pet.sprite_mapper import pick_sprite


class TestEngagementWins:
    """engagement_mode dominates because it gates output style everywhere
    else in the system — pet should mirror that priority."""

    def test_flustered_beats_emotion(self):
        # Even with happy emotion, flustered mode wins
        assert pick_sprite(
            engagement_mode="flustered",
            emotion_family="HIGH_ENERGY",
        ) == "flustered"

    def test_withdrawn_beats_emotion(self):
        assert pick_sprite(
            engagement_mode="withdrawn",
            emotion_family="HIGH_ENERGY",
        ) == "withdrawn"

    def test_curt_beats_emotion(self):
        assert pick_sprite(
            engagement_mode="curt",
            emotion_family="HIGH_ENERGY",
        ) == "curt"

    def test_full_mode_falls_through_to_emotion(self):
        # "full" is just default — doesn't gate, so emotion below it picks
        assert pick_sprite(
            engagement_mode="full",
            emotion_family="HEAVY",
        ) == "heavy"


class TestEmotionFamily:
    def test_heavy(self):
        assert pick_sprite(emotion_family="HEAVY") == "heavy"

    def test_provoked(self):
        assert pick_sprite(emotion_family="PROVOKED") == "provoked"

    def test_high_energy_is_happy(self):
        assert pick_sprite(emotion_family="HIGH_ENERGY") == "happy"

    def test_low_energy_is_tired(self):
        assert pick_sprite(emotion_family="LOW_ENERGY") == "tired"

    def test_neutral_falls_through(self):
        assert pick_sprite(emotion_family="NEUTRAL") == "idle_default"

    def test_emotion_beats_activity(self):
        # Even working, if she's heavy emotion the heavy state wins
        assert pick_sprite(
            emotion_family="HEAVY",
            activity_kind="work",
        ) == "heavy"

    def test_emotion_family_case_insensitive(self):
        assert pick_sprite(emotion_family="heavy") == "heavy"


class TestActivity:
    def test_meal(self):
        assert pick_sprite(activity_kind="meal") == "eating"

    def test_work(self):
        assert pick_sprite(activity_kind="work") == "focused"

    def test_sleep(self):
        assert pick_sprite(activity_kind="sleep") == "sleepy"

    def test_unrelated_activity_falls_through(self):
        # routine / outdoors / social / etc. don't have dedicated sprites
        assert pick_sprite(activity_kind="routine") == "idle_default"
        assert pick_sprite(activity_kind="outdoors") == "idle_default"


class TestHourFallback:
    def test_late_night_is_sleepy(self):
        assert pick_sprite(hour=23) == "sleepy"
        assert pick_sprite(hour=2) == "sleepy"
        assert pick_sprite(hour=5) == "sleepy"

    def test_early_morning_not_sleepy(self):
        assert pick_sprite(hour=6) == "idle_default"
        assert pick_sprite(hour=7) == "idle_default"

    def test_daytime_not_sleepy(self):
        assert pick_sprite(hour=14) == "idle_default"

    def test_activity_beats_hour(self):
        # If she's eating at midnight, show eating not sleepy
        assert pick_sprite(activity_kind="meal", hour=23) == "eating"


class TestDefault:
    def test_nothing_returns_default(self):
        assert pick_sprite() == "idle_default"

    def test_unknown_values_return_default(self):
        assert pick_sprite(
            engagement_mode="unknown",
            emotion_family="weird",
            activity_kind="???",
        ) == "idle_default"
