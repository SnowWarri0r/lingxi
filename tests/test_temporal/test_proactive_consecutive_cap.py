"""Tests for the consecutive-proactive cap that prevents Aria from
spamming multiple messages during a long user silence.

The cap is the difference between "friend who checks in once" and "needy
ex sending 4 messages while you're at lunch".
"""

from datetime import datetime, timedelta

from lingxi.temporal.tracker import InteractionRecord, InteractionTracker


def test_record_proactive_increments_consecutive_counter(tmp_path):
    tracker = InteractionTracker(tmp_path)
    tracker.record_interaction("feishu", "u1")  # creates record
    assert tracker._records["feishu:u1"].consecutive_proactive_count == 0

    tracker.record_proactive_sent("feishu", "u1")
    assert tracker._records["feishu:u1"].consecutive_proactive_count == 1

    tracker.record_proactive_sent("feishu", "u1")
    assert tracker._records["feishu:u1"].consecutive_proactive_count == 2


def test_user_reply_resets_consecutive_counter(tmp_path):
    """After user actually replies, counter goes back to 0 so Aria can
    reach out again in the next silence cycle."""
    tracker = InteractionTracker(tmp_path)
    tracker.record_interaction("feishu", "u1")
    tracker.record_proactive_sent("feishu", "u1")
    tracker.record_proactive_sent("feishu", "u1")
    assert tracker._records["feishu:u1"].consecutive_proactive_count == 2

    # User replies
    tracker.record_interaction("feishu", "u1")
    assert tracker._records["feishu:u1"].consecutive_proactive_count == 0


def test_new_interaction_record_starts_at_zero(tmp_path):
    tracker = InteractionTracker(tmp_path)
    rec = tracker.record_interaction("feishu", "u_new")
    assert rec.consecutive_proactive_count == 0


def test_record_proactive_noop_for_unknown_recipient(tmp_path):
    """Defensive: record_proactive_sent before any user interaction
    shouldn't crash (e.g. on a recipient that doesn't exist yet)."""
    tracker = InteractionTracker(tmp_path)
    tracker.record_proactive_sent("feishu", "ghost")
    # No exception, no record created
    assert "feishu:ghost" not in tracker._records


def test_consecutive_counter_persists_in_model():
    """Round-trip the field through pydantic so persistence works."""
    rec = InteractionRecord(
        recipient_id="u1",
        channel="feishu",
        consecutive_proactive_count=3,
    )
    data = rec.model_dump()
    restored = InteractionRecord.model_validate(data)
    assert restored.consecutive_proactive_count == 3


def test_consecutive_counter_defaults_to_zero_for_legacy_records():
    """Records persisted before this field existed should load with 0."""
    legacy = {
        "recipient_id": "u1",
        "channel": "feishu",
        "last_interaction": datetime.now().isoformat(),
    }
    rec = InteractionRecord.model_validate(legacy)
    assert rec.consecutive_proactive_count == 0
