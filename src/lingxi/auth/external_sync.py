"""Auto-detect and import credentials from external CLI tools.

Syncs credentials from:
- Claude Code CLI: ~/.claude/.credentials.json or macOS Keychain
- Codex CLI: ~/.codex/auth.json (respects $CODEX_HOME)

Cached with a 15-minute TTL to avoid repeated filesystem reads.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

from lingxi.auth.models import (
    AuthProfile,
    CredentialType,
    TokenInfo,
)

_SYNC_TTL_SECONDS = 900  # 15 minutes


class ExternalCredentialSync:
    """Discovers and imports credentials from external CLI tools."""

    def __init__(self):
        self._cache: dict[str, tuple[list[AuthProfile], float]] = {}

    def sync_all(self) -> list[AuthProfile]:
        """Discover all external credentials. Returns profiles ready to upsert."""
        profiles: list[AuthProfile] = []
        profiles.extend(self._sync_claude_code())
        profiles.extend(self._sync_codex_cli())
        return profiles

    def _sync_claude_code(self) -> list[AuthProfile]:
        """Import credentials from Claude Code CLI."""
        cache_key = "claude_code"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        profiles: list[AuthProfile] = []

        # Try 1: Read ~/.claude/.credentials.json
        cred_path = Path.home() / ".claude" / ".credentials.json"
        token = self._read_claude_credentials_file(cred_path)

        # Try 2: macOS Keychain (if file not found)
        if token is None and platform.system() == "Darwin":
            token = self._read_claude_keychain()

        if token:
            profile = AuthProfile(
                credential_type=CredentialType.OAUTH,
                provider="anthropic",
                label="claude_code",
                oauth_token=token,
                source="claude_code",
            )
            profiles.append(profile)

        self._set_cached(cache_key, profiles)
        return profiles

    def _sync_codex_cli(self) -> list[AuthProfile]:
        """Import credentials from Codex CLI."""
        cache_key = "codex_cli"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        profiles: list[AuthProfile] = []

        codex_home = os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))
        auth_path = Path(codex_home) / "auth.json"

        token = self._read_codex_auth_file(auth_path)
        if token:
            profile = AuthProfile(
                credential_type=CredentialType.OAUTH,
                provider="openai",
                label="codex_cli",
                oauth_token=token,
                source="codex_cli",
            )
            profiles.append(profile)

        self._set_cached(cache_key, profiles)
        return profiles

    # --- File readers ---

    @staticmethod
    def _read_claude_credentials_file(path: Path) -> TokenInfo | None:
        """Read Claude Code credential file.

        Known formats:
        - {"claudeAiOauth": {"accessToken": "...", "refreshToken": "...", "expiresAt": "..."}}
        - {"oauthAccount": {"accessToken": "...", ...}}
        - Direct {"accessToken": "...", "refreshToken": "..."}
        """
        if not path.exists():
            return None

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        # Try nested formats
        for key in ("claudeAiOauth", "oauthAccount"):
            if key in data and isinstance(data[key], dict):
                data = data[key]
                break

        return _extract_token_from_dict(data)

    @staticmethod
    def _read_claude_keychain() -> TokenInfo | None:
        """Read Claude Code credentials from macOS Keychain."""
        try:
            result = subprocess.run(
                ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None

            data = json.loads(result.stdout.strip())
            # Same parsing as file
            for key in ("claudeAiOauth", "oauthAccount"):
                if key in data and isinstance(data[key], dict):
                    data = data[key]
                    break

            return _extract_token_from_dict(data)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _read_codex_auth_file(path: Path) -> TokenInfo | None:
        """Read Codex CLI auth.json.

        Format: {"access_token": "...", "refresh_token": "...", "expires_at": "..."}
        or similar variants.
        """
        if not path.exists():
            return None

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        return _extract_token_from_dict(data)

    # --- Cache ---

    def _get_cached(self, key: str) -> list[AuthProfile] | None:
        if key in self._cache:
            profiles, ts = self._cache[key]
            if time.time() - ts < _SYNC_TTL_SECONDS:
                return profiles
        return None

    def _set_cached(self, key: str, profiles: list[AuthProfile]) -> None:
        self._cache[key] = (profiles, time.time())

    def invalidate_cache(self) -> None:
        self._cache.clear()


def _extract_token_from_dict(data: dict) -> TokenInfo | None:
    """Extract a TokenInfo from various JSON formats used by CLI tools."""
    # Try common field names
    access_token = (
        data.get("access_token")
        or data.get("accessToken")
        or data.get("token")
    )
    if not access_token:
        return None

    refresh_token = data.get("refresh_token") or data.get("refreshToken")

    # Parse expiry
    expires_at = None
    for exp_key in ("expires_at", "expiresAt", "expires"):
        exp_val = data.get(exp_key)
        if exp_val is not None:
            expires_at = _parse_expiry(exp_val)
            break

    return TokenInfo(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
        id_token=data.get("id_token") or data.get("idToken"),
    )


def _parse_expiry(value) -> datetime | None:
    """Parse various expiry formats to datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        # Unix timestamp (seconds or milliseconds)
        if value > 1e12:  # milliseconds
            return datetime.fromtimestamp(value / 1000)
        return datetime.fromtimestamp(value)
    if isinstance(value, str):
        # ISO format
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
        # Unix timestamp as string
        try:
            ts = float(value)
            if ts > 1e12:
                return datetime.fromtimestamp(ts / 1000)
            return datetime.fromtimestamp(ts)
        except ValueError:
            pass
    return None
