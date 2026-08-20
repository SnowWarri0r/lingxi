import json

from lingxi.evals import cli
from lingxi.evals.cli import format_table, load_baseline, save_baseline
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
