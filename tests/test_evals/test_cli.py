import json

from lingxi.evals import cli
from lingxi.evals.cli import dump_replies, format_table, load_baseline, save_baseline
from lingxi.evals.runner import CaseScore


def test_table_shows_delta_against_baseline():
    scores = [CaseScore(id="a", verdict="PASS", fail_rate=0.05,
                        pass_rate=0.15, samples=20)]
    out = format_table(scores, {"a": {"fail_rate": 0.15}})
    assert "PASS" in out
    assert "1/20" in out
    assert "-10pp" in out


def test_table_marks_new_cases():
    scores = [CaseScore(id="a", verdict="PASS", fail_rate=0.0,
                        pass_rate=0.0, samples=20)]
    out = format_table(scores, {})
    assert "new" in out


def test_broken_row_shows_the_reason_and_no_numbers():
    scores = [CaseScore(id="a", verdict="BROKEN", premise_ok=False,
                        premise_error="prompt_contains 未命中：'X'")]
    out = format_table(scores, {"a": {"fail_rate": 0.5}})
    assert "BROKEN" in out
    assert "prompt_contains 未命中" in out
    assert "pp" not in out          # a broken case is not scored


def test_baseline_roundtrip(tmp_path):
    path = tmp_path / "baseline.json"
    save_baseline(path, [CaseScore(id="a", verdict="PASS", fail_rate=0.05,
                                   pass_rate=0.1, samples=20)], "abc123")
    data = load_baseline(path)
    assert data["a"]["fail_rate"] == 0.05
    assert data["a"]["git_sha"] == "abc123"
    assert "recorded_at" in data["a"]


def test_load_baseline_missing_file_is_empty(tmp_path):
    assert load_baseline(tmp_path / "nope.json") == {}


def test_broken_cases_are_not_written_to_baseline(tmp_path):
    """Recording a BROKEN case would bake a meaningless number into the
    baseline and hide the real number when the case is repaired."""
    path = tmp_path / "baseline.json"
    save_baseline(path, [
        CaseScore(id="ok", verdict="PASS", fail_rate=0.0, samples=20),
        CaseScore(id="bad", verdict="BROKEN", premise_ok=False),
    ], "abc123")
    assert sorted(json.loads(path.read_text())) == ["ok"]


def test_zero_cases_exits_nonzero_and_prints_message(monkeypatch, capsys):
    """When no cases are found, print error message in Chinese and exit non-zero."""
    async def mock_run(case_ids):
        return []

    monkeypatch.setattr(cli, "_run", mock_run)
    monkeypatch.setattr("sys.argv", ["lingxi-eval"])

    result = cli.main()

    assert result == 1, "Should return non-zero exit code when zero cases"
    captured = capsys.readouterr()
    assert "没有找到任何案例" in captured.out, "Should print Chinese error message"


def test_dump_replies_writes_one_file_per_case(tmp_path):
    scores = [
        CaseScore(id="a", verdict="PASS", fail_rate=0.0, samples=2,
                  replies=["回复一", "回复二"]),
        CaseScore(id="b", verdict="FAIL", fail_rate=1.0, samples=1,
                  replies=["回复三"]),
    ]
    dump_replies(scores, tmp_path)
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == (
        "回复一\n\n---\n\n回复二")
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "回复三"


def test_dump_replies_skips_cases_with_no_replies(tmp_path):
    """A BROKEN case (or any case sampled zero times) writes no file rather
    than an empty one."""
    scores = [CaseScore(id="broken", verdict="BROKEN", premise_ok=False,
                        premise_error="x", replies=[])]
    dump_replies(scores, tmp_path)
    assert not (tmp_path / "broken.txt").exists()


def test_dump_replies_creates_the_output_directory(tmp_path):
    out = tmp_path / "nested" / "dir"
    scores = [CaseScore(id="a", verdict="PASS", fail_rate=0.0, samples=1,
                        replies=["回复"])]
    dump_replies(scores, out)
    assert (out / "a.txt").read_text(encoding="utf-8") == "回复"


def test_all_pass_cases_still_exit_zero(monkeypatch, capsys):
    """When all cases pass, exit code should still be 0 (no regression)."""
    async def mock_run(case_ids):
        return [CaseScore(id="a", verdict="PASS", fail_rate=0.0,
                          pass_rate=1.0, samples=10)]

    monkeypatch.setattr(cli, "_run", mock_run)
    monkeypatch.setattr("sys.argv", ["lingxi-eval"])

    result = cli.main()

    assert result == 0, "Should return 0 when all cases pass"
    captured = capsys.readouterr()
    assert "没有找到任何案例" not in captured.out, "Should not print zero-cases error"


def test_capture_refuses_to_write_where_git_would_track_it(tmp_path):
    """A capture is a verbatim copy of a real person's conversation.

    The tool must not be able to produce a committable artifact, regardless
    of what .gitignore currently says — the ignore rule and the tool are two
    independent lines of defence, and the ignore rule already failed once.
    """
    from lingxi.evals.cli import _is_git_ignored

    # tmp_path is outside the repo, so git cannot vouch for it.
    assert _is_git_ignored(tmp_path / "anything.yaml") is False


def test_real_case_paths_are_ignored_and_the_example_is_not():
    """The two lines of defence, asserted against the live repo."""
    from pathlib import Path

    from lingxi.evals.cli import _is_git_ignored

    assert _is_git_ignored(Path("evals/cases/offwork-state.yaml")) is True
    assert _is_git_ignored(Path("evals/baseline.json")) is True
    assert _is_git_ignored(Path("evals/cases/example-case.yaml")) is False


def test_no_real_case_file_is_tracked_by_git():
    """Nothing under evals/cases/ may be tracked except the synthetic example.

    This is the assertion that would have caught the 49-personal-facts case
    before it was committed.
    """
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "evals/"], capture_output=True, text=True,
    ).stdout.split()
    assert tracked == ["evals/cases/example-case.yaml"], tracked
