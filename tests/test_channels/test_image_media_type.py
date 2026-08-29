"""The downloaded image's type is read from its bytes, not from Feishu.

A sticker that comes back through the `type=file` fallback carries no image/*
header, and the old default labelled it png. One of them was a JPEG, so the
vision API answered `the image appears to be a image/jpeg image` — a 400 that
costs the whole turn for a picture that arrived intact.
"""

import pytest

from lingxi.channels.feishu import FeishuBot


sniff = FeishuBot._sniff_media_type

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 16
GIF = b"GIF89a" + b"\x00" * 16
WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 16


@pytest.mark.parametrize("data,expected", [
    (PNG, "image/png"),
    (JPEG, "image/jpeg"),
    (GIF, "image/gif"),
    (b"GIF87a" + b"\x00" * 16, "image/gif"),
    (WEBP, "image/webp"),
])
def test_each_format_is_recognised(data, expected):
    assert sniff(data) == expected


def test_the_jpeg_that_was_labelled_png():
    """The live case: Feishu said image/png, the bytes said JPEG."""
    assert sniff(JPEG) == "image/jpeg"


def test_an_unrecognised_format_defers_to_the_caller():
    assert sniff(b"not an image at all") is None


def test_a_truncated_file_does_not_raise():
    assert sniff(b"") is None
    assert sniff(b"\xff") is None
    assert sniff(b"RIFF") is None


def test_riff_that_is_not_webp_is_not_claimed():
    """A .wav is also RIFF."""
    assert sniff(b"RIFF\x24\x00\x00\x00WAVEfmt ") is None
