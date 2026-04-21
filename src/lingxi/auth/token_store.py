"""Persistent token storage for caching OAuth tokens across sessions."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from lingxi.auth.models import TokenInfo


class TokenStore:
    """Manages persistent storage of OAuth tokens.

    Tokens are stored in ~/.persona-agent/tokens/ with file permissions
    restricted to the current user (600).
    """

    def __init__(self, store_dir: str | None = None):
        if store_dir:
            self._dir = Path(store_dir)
        else:
            self._dir = Path.home() / ".persona-agent" / "tokens"

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        # Restrict directory permissions to owner only
        try:
            os.chmod(self._dir, stat.S_IRWXU)
        except OSError:
            pass

    def _token_path(self, provider: str) -> Path:
        # Sanitize provider name for use as filename
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in provider)
        return self._dir / f"{safe_name}.json"

    def save(self, provider: str, token: TokenInfo) -> None:
        """Save a token for a provider."""
        self._ensure_dir()
        path = self._token_path(provider)

        data = token.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        # Restrict file permissions to owner read/write only
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

    def load(self, provider: str) -> TokenInfo | None:
        """Load a cached token for a provider. Returns None if not found."""
        path = self._token_path(provider)
        if not path.exists():
            return None

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return TokenInfo.model_validate(data)
        except (json.JSONDecodeError, Exception):
            return None

    def delete(self, provider: str) -> bool:
        """Delete a cached token."""
        path = self._token_path(provider)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_providers(self) -> list[str]:
        """List providers that have cached tokens."""
        if not self._dir.exists():
            return []
        return [p.stem for p in self._dir.glob("*.json")]
