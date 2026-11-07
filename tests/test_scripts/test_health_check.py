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
