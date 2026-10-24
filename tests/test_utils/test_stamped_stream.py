"""Every line of the bot's log says when it was written."""

import io
from datetime import datetime

from lingxi.utils.logging import StampedStream

T = datetime(2026, 3, 9, 14, 5, 7)


def _stream():
    buf = io.StringIO()
    return StampedStream(buf, clock=lambda: T), buf


def test_each_line_gets_the_time():
    s, buf = _stream()
    print("[qq] connected", file=s)
    print("[proactive] sent\n[stories] curated 4", file=s)
    assert buf.getvalue().splitlines() == [
        "03-09 14:05:07 [qq] connected",
        "03-09 14:05:07 [proactive] sent",
        "03-09 14:05:07 [stories] curated 4"]


def test_a_line_written_in_pieces_is_stamped_once():
    s, buf = _stream()
    s.write("[brain] "); s.write("orch decision\n")
    assert buf.getvalue() == "03-09 14:05:07 [brain] orch decision\n"


def test_lark_lines_keep_their_own_time():
    s, buf = _stream()
    s.write("[Lark] [2026-03-09 14:05:06,1] [INFO] connected\n")
    assert buf.getvalue() == "[Lark] [2026-03-09 14:05:06,1] [INFO] connected\n"
