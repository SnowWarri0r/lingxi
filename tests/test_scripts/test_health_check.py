"""The health check says why the bot is down, and only when it is."""

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("health_check", ROOT / "scripts" / "health_check.py")
hc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc)

NOW = 1_800_000_000.0


def _log(tmp_path, text, age=60):
    p = tmp_path / "bot.log"
    p.write_text(text, encoding="utf-8")
    os.utime(p, (NOW - age, NOW - age))
    return p


def test_a_bot_that_logs_and_runs_is_fine(tmp_path):
    log = _log(tmp_path, "03-09 10:00:00 [qq] connected\n03-09 10:05:00 [proactive] {}\n")
    assert hc.problem(alive=True, log=log, now=NOW) == ""


def test_no_process_is_down(tmp_path):
    assert "进程" in hc.problem(alive=False, log=_log(tmp_path, "x"), now=NOW)


def test_a_silent_log_is_a_hang(tmp_path):
    log = _log(tmp_path, "[proactive] {}\n", age=45 * 60)
    assert "45 分钟" in hc.problem(alive=True, log=log, now=NOW)


def test_a_crash_on_startup_is_down_with_its_reason(tmp_path):
    log = _log(tmp_path, "[qq] connected\nTraceback (most recent call last):\n"
                         "  File x\nAuthError: no credential\n")
    assert hc.problem(alive=True, log=log, now=NOW) == "启动报错：AuthError: no credential"


def test_a_traceback_it_recovered_from_is_not(tmp_path):
    log = _log(tmp_path, "Traceback (most recent call last):\nValueError: x\n"
                         "[qq] connected\n")
    assert hc.problem(alive=True, log=log, now=NOW) == ""


def test_one_outage_notifies_once_and_recovery_once(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(hc, "notify", lambda t, b: sent.append(t))
    monkeypatch.setattr(hc, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(hc, "running", lambda label: False)
    hc.main(); hc.main()
    monkeypatch.setattr(hc, "running", lambda label: True)
    monkeypatch.setattr(hc, "LOG", _log(tmp_path, "[qq] connected\n", age=0))
    monkeypatch.setattr(hc.time, "time", lambda: NOW)
    hc.main(); hc.main()
    assert sent == ["lingxi 挂了", "lingxi 恢复了"]


def _stamp(at):
    return time.strftime("%m-%d %H:%M:%S", time.localtime(at))


import time  # noqa: E402


def test_a_refused_credential_is_down_while_the_bot_still_runs(tmp_path):
    log = _log(tmp_path, f"{_stamp(NOW - 600)} [planner] LLM/parse failed: Anthropic API "
                         'error 401: {"type":"error","error":{"type":"authentication_error"}}\n'
                         f"{_stamp(NOW - 60)} [proactive] skipped: quiet hours\n")
    assert hc.problem(alive=True, log=log, now=NOW) == "主模型鉴权被拒（planner）"


def test_an_old_refusal_is_over(tmp_path):
    log = _log(tmp_path, f"{_stamp(NOW - 3 * 3600)} [planner] LLM/parse failed: "
                         "Anthropic API error 401\n"
                         f"{_stamp(NOW - 60)} [proactive] skipped: quiet hours\n")
    assert hc.problem(alive=True, log=log, now=NOW) == ""


def _ticks(start, end, err="ConnectError('')", step=300):
    return "".join(f"{_stamp(t)} [memory] consolidation tick failed: {err}\n"
                   for t in range(int(start), int(end), step))


def test_memory_ticks_failing_for_an_hour_are_down(tmp_path):
    log = _log(tmp_path, _ticks(NOW - 90 * 60, NOW))
    assert hc.problem(alive=True, log=log, now=NOW).startswith("记忆整理连续失败 8")


def test_a_tick_that_succeeds_ends_the_streak(tmp_path):
    log = _log(tmp_path, _ticks(NOW - 90 * 60, NOW - 30 * 60)
               + f"{_stamp(NOW - 25 * 60)} [memory] consolidated e87df1: 4 turns → {{}}\n"
               + _ticks(NOW - 20 * 60, NOW))
    assert hc.problem(alive=True, log=log, now=NOW) == ""


def test_a_few_failed_ticks_are_not_an_outage(tmp_path):
    log = _log(tmp_path, _ticks(NOW - 15 * 60, NOW))
    assert hc.problem(alive=True, log=log, now=NOW) == ""
