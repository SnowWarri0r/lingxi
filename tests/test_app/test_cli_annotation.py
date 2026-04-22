"""Unit test for the CLI annotation command parser."""

from lingxi.app import parse_annotation_command


def test_parse_good():
    cmd = parse_annotation_command(":good")
    assert cmd == {"kind": "positive", "correction": None}


def test_parse_bad_no_correction():
    cmd = parse_annotation_command(":bad")
    assert cmd == {"kind": "negative", "correction": None}


def test_parse_bad_with_correction():
    cmd = parse_annotation_command(":bad 嗨嗨")
    assert cmd == {"kind": "negative", "correction": "嗨嗨"}


def test_parse_reveal():
    cmd = parse_annotation_command(":reveal")
    assert cmd == {"kind": "reveal", "correction": None}


def test_parse_non_annotation_returns_none():
    assert parse_annotation_command("normal message") is None
    assert parse_annotation_command(":unknown") is None
