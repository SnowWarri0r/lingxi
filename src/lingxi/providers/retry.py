"""Which API failures are worth trying again.

The responder generates the one thing the user actually reads, so a failed
call costs a whole turn — it comes back as the empty-reply fallback. But
retrying indiscriminately is its own bug: a malformed request fails the same
way the second time, and an empty account does not fill up in half a second.
Both were live on 2026-08-27, one turn apart:

    400 This model does not support image   — permanent, the request is wrong
    402 Insufficient Balance                — permanent for this turn

What a retry does buy is the class of failure that has nothing to do with the
request: the endpoint briefly 500s, the gateway sheds load with 429, a
connection drops mid-stream. Those succeed on the next attempt.
"""

from __future__ import annotations

# 408/409 are momentary server-side conditions; 429 is explicit backpressure;
# 5xx is the endpoint having a bad second. 4xx otherwise means the request
# itself is the problem, and sending it again just pays for it twice.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def _status_of(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def is_retryable(exc: Exception) -> bool:
    """True when `exc` describes a transient condition, not a bad request."""
    status = _status_of(exc)
    if status is not None:
        return status in RETRYABLE_STATUS
    # No status code at all means the request never got an answer — a dropped
    # connection or a read timeout. Nothing about it says the request is wrong.
    try:
        import openai
    except ImportError:  # pragma: no cover — openai is a hard dependency
        return False
    return isinstance(exc, openai.APIConnectionError)
