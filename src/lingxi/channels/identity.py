"""One person, several apps.

Everything she knows about someone — his facts, the conversation buffer, the
thread recap, the interaction record — is keyed by "channel:id". Reached on a
second app, the same person would be a stranger to her. identities.json maps
an account on another channel to the key that already holds him:

    {"qq:<user_openid>": "feishu:oc_xxx"}

Channels translate at the door, so nothing behind them needs to know.
"""

from __future__ import annotations

import json
from pathlib import Path


class IdentityMap:
    def __init__(self, path: Path | str | None):
        self._path = Path(path) if path else None
        self._map: dict[str, str] = {}
        if self._path and self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._map = {str(k): str(v) for k, v in raw.items()}
            except (OSError, ValueError):
                pass

    def resolve(self, channel: str, ext_id: str) -> tuple[str, str]:
        """The (channel, id) whose memory this account shares, or itself."""
        home = self._map.get(f"{channel}:{ext_id}")
        if not home or ":" not in home:
            return channel, ext_id
        c, _, i = home.partition(":")
        return c, i

    def link(self, channel: str, ext_id: str, home: str) -> None:
        self._map[f"{channel}:{ext_id}"] = home
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._map, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.rename(self._path)
