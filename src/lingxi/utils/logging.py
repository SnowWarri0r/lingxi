"""Structured logging setup."""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime

import structlog

_LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def setup_logging(level: str | None = None) -> None:
    """Configure structured logging."""
    log_level = level or os.environ.get("LOG_LEVEL", "INFO")
    numeric_level = _LOG_LEVEL_MAP.get(log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Get a bound logger instance."""
    return structlog.get_logger(name)


class StampedStream:
    """A text stream that starts every line with the time it was written.

    The bot reports through print(), and under launchd its log had no times
    except on the Lark SDK's own lines, which carry theirs and are left as
    they are. Finding when QQ dropped or an opener went out meant counting
    heartbeats between them.
    """

    _OWN_STAMP = "[Lark] [20"

    def __init__(self, stream, clock=datetime.now):
        self._stream = stream
        self._clock = clock
        self._lock = threading.Lock()
        self._at_start = True

    def write(self, text: str) -> int:
        with self._lock:
            out = []
            for piece in text.splitlines(keepends=True):
                if self._at_start and not piece.startswith(self._OWN_STAMP) \
                        and piece.strip():
                    out.append(f"{self._clock():%m-%d %H:%M:%S} ")
                out.append(piece)
                self._at_start = piece.endswith("\n")
            self._stream.write("".join(out))
        return len(text)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)
