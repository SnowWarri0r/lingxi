"""De-identification: keep texture, drop identity.

Strip @handles; drop any line carrying a locatable personal disclosure
(named school/org + role, contact info). Conservative — when in doubt, drop.
"""

from __future__ import annotations

import re

_HANDLE = re.compile(r"@\S+\s*")
_LOCATABLE = re.compile(
    r"(大学|学院|公司|医院|中学).{0,6}(读|上班|工作|读博|读研|实习)"
    r"|(导师|老板|领导)\s*姓"
    r"|微信|vx|qq|电话|手机号|身份证")


def deidentify(line: str) -> str | None:
    line = _HANDLE.sub("", line or "").strip()
    if not line:
        return None
    if _LOCATABLE.search(line):
        return None
    return line
