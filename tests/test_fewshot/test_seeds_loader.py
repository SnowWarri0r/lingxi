"""Tests for seeds.yaml loader."""

from pathlib import Path

import pytest

from lingxi.fewshot.seeds_loader import load_seeds


REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS_PATH = REPO_ROOT / "config" / "fewshot" / "seeds.yaml"


def test_seeds_file_exists():
    assert SEEDS_PATH.exists(), f"seeds.yaml missing at {SEEDS_PATH}"


def test_load_seeds_returns_samples():
    samples = load_seeds(SEEDS_PATH)
    assert len(samples) == 10
    ids = {s.id for s in samples}
    assert len(ids) == 10  # unique ids
    for s in samples:
        assert s.source == "seed"
        assert s.recipient_key is None
        assert s.corrected_speech  # non-empty
        assert s.context_summary


def test_load_seeds_preserves_tags():
    samples = load_seeds(SEEDS_PATH)
    by_id = {s.id: s for s in samples}
    assert by_id["seed-resonance-01"].tags == ["共鸣", "温暖"]


def test_load_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_seeds("/nonexistent/seeds.yaml")
