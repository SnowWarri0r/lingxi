"""Request logs belong to the persona whose prompts they hold.

facts.db moved to data/personas/<slug>/ when personas were namespaced; the
request log kept deriving its path from the old data/memory layout, so every
persona wrote into one shared data/debug/llm_requests/. Reading it back, a
tangkeke-era analysis silently included Aria-era calls made against a
different facts.db — twice in one session that produced a defect that was not
there (a catalog bucket that had 50 rows at the time and none now).
"""

from pathlib import Path

import pytest

from lingxi.debug import request_log


@pytest.fixture(autouse=True)
def _restore_root():
    before = request_log._LOG_ROOT
    yield
    request_log._LOG_ROOT = before


def test_the_log_lands_under_the_persona_root(tmp_path):
    request_log.set_log_root(tmp_path / "personas" / "tangkeke")

    assert request_log._log_dir() == tmp_path / "personas" / "tangkeke" / "debug" / "llm_requests"


def test_two_personas_do_not_share_a_file(tmp_path):
    request_log.set_log_root(tmp_path / "personas" / "a")
    first = request_log._log_dir()
    request_log.set_log_root(tmp_path / "personas" / "b")

    assert first != request_log._log_dir()


def test_without_a_root_the_old_location_still_works(tmp_path, monkeypatch):
    """A caller that never sets one (tools, tests) keeps working."""
    request_log._LOG_ROOT = None
    monkeypatch.setenv("MEMORY_DATA_DIR", str(tmp_path / "memory"))

    assert request_log._log_dir() == tmp_path / "debug" / "llm_requests"


def test_an_explicit_root_wins_over_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_DATA_DIR", str(tmp_path / "memory"))
    request_log.set_log_root(tmp_path / "chosen")

    assert request_log._log_dir() == tmp_path / "chosen" / "debug" / "llm_requests"


def test_writing_actually_creates_the_file_there(tmp_path, monkeypatch):
    monkeypatch.setenv("LINGXI_DEBUG_LLM", "1")
    request_log.set_log_root(tmp_path / "personas" / "keke")

    request_log.log_request(
        system="s", messages=[{"role": "user", "content": "hi"}],
        response_text="yo", model="m", purpose="test")

    written = list((tmp_path / "personas" / "keke" / "debug" / "llm_requests").glob("*.jsonl"))
    assert len(written) == 1
    assert "yo" in Path(written[0]).read_text()
