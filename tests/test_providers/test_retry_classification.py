"""Which failures get a second attempt.

Both live failures on 2026-08-27 were permanent — `400 This model does not
support image` and `402 Insufficient Balance`. Retrying either just pays for
the same failure twice. The transient ones are what a retry is for.
"""

import openai
import pytest

from lingxi.providers.retry import is_retryable


class _Status(Exception):
    """Shaped like an SDK status error: carries the HTTP code."""

    def __init__(self, status_code):
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


class _ViaResponse(Exception):
    """Some clients hang the code off `.response` instead."""

    def __init__(self, status_code):
        super().__init__("boom")
        self.response = type("R", (), {"status_code": status_code})()


@pytest.mark.parametrize("status", [408, 409, 425, 429, 500, 502, 503, 504])
def test_transient_server_conditions_are_retried(status):
    assert is_retryable(_Status(status)) is True


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 413, 422])
def test_a_rejected_request_is_not_sent_again(status):
    assert is_retryable(_Status(status)) is False


def test_no_image_support_is_not_retried():
    """The exact live failure: the request is wrong, and stays wrong."""
    assert is_retryable(_Status(400)) is False


def test_an_empty_account_is_not_retried():
    """402 does not clear in half a second."""
    assert is_retryable(_Status(402)) is False


def test_the_code_is_found_on_the_response_too():
    assert is_retryable(_ViaResponse(503)) is True
    assert is_retryable(_ViaResponse(400)) is False


def test_a_dropped_connection_is_retried():
    """No status code at all — the request never got an answer."""
    exc = openai.APIConnectionError(request=None)
    assert is_retryable(exc) is True


def test_a_timeout_is_retried():
    assert is_retryable(openai.APITimeoutError(request=None)) is True


def test_an_unrecognised_error_is_left_alone():
    assert is_retryable(ValueError("something else entirely")) is False
