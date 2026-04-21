"""Profile-based credential store (replaces TokenStore).

Stores all auth profiles in a single JSON file at ~/.persona-agent/auth-profiles.json.
Supports multi-account, ordering, cooldown tracking, and migration from old TokenStore.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from lingxi.auth.models import (
    AuthProfile,
    CredentialType,
    FailureCooldown,
    TokenInfo,
)


_STORE_VERSION = 1


class ProfileStore:
    """Manages persistent storage of auth profiles.

    File format:
    {
      "version": 1,
      "profiles": {
        "anthropic:default": { ... AuthProfile fields ... },
        "openai:codex_cli": { ... }
      }
    }
    """

    def __init__(self, store_path: str | None = None):
        if store_path:
            self._path = Path(store_path)
        else:
            self._path = Path.home() / ".persona-agent" / "auth-profiles.json"
        self._cache: dict[str, AuthProfile] | None = None

    def _ensure_dir(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self._path.parent, stat.S_IRWXU)
        except OSError:
            pass

    def _load_all(self) -> dict[str, AuthProfile]:
        """Load all profiles from disk. Runs migration from old TokenStore if needed."""
        if self._cache is not None:
            return self._cache

        profiles: dict[str, AuthProfile] = {}

        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    data = json.load(f)
                raw_profiles = data.get("profiles", {})
                for key, pdata in raw_profiles.items():
                    try:
                        profiles[key] = AuthProfile.model_validate(pdata)
                    except Exception:
                        continue
            except (json.JSONDecodeError, OSError):
                pass
        else:
            # Try migrating from old TokenStore
            profiles = self._migrate_from_token_store()

        self._cache = profiles
        return profiles

    def _save_all(self, profiles: dict[str, AuthProfile]) -> None:
        """Atomic write with sanitization and restricted permissions."""
        self._ensure_dir()

        # Sanitize: strip inline secrets when refs exist
        sanitized = {}
        for key, profile in profiles.items():
            clean = profile.sanitize_for_save()
            sanitized[key] = clean.model_dump(mode="json")

        data = {
            "version": _STORE_VERSION,
            "profiles": sanitized,
        }

        # Write to temp file then rename (atomic on same filesystem)
        tmp_path = self._path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        try:
            os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

        tmp_path.rename(self._path)
        self._cache = profiles

    def _migrate_from_token_store(self) -> dict[str, AuthProfile]:
        """One-time migration from old ~/.persona-agent/tokens/*.json files."""
        old_dir = self._path.parent / "tokens"
        if not old_dir.exists():
            return {}

        profiles: dict[str, AuthProfile] = {}
        for token_file in old_dir.glob("*.json"):
            try:
                with open(token_file, encoding="utf-8") as f:
                    tdata = json.load(f)
                token = TokenInfo.model_validate(tdata)
                provider = token_file.stem
                profile = AuthProfile(
                    credential_type=CredentialType.OAUTH,
                    provider=provider,
                    label="default",
                    oauth_token=token,
                    source="migrated",
                )
                profiles[profile.profile_key] = profile
            except Exception:
                continue

        if profiles:
            self._save_all(profiles)

        return profiles

    # --- Public API ---

    def get(self, provider: str, label: str = "default") -> AuthProfile | None:
        """Get a specific profile."""
        profiles = self._load_all()
        return profiles.get(f"{provider}:{label}")

    def list_profiles(self, provider: str | None = None) -> list[AuthProfile]:
        """List profiles, optionally filtered by provider."""
        profiles = self._load_all()
        if provider is None:
            return list(profiles.values())
        return [p for p in profiles.values() if p.provider == provider]

    def upsert(self, profile: AuthProfile) -> None:
        """Insert or update a profile."""
        profiles = self._load_all()
        profiles[profile.profile_key] = profile
        self._save_all(profiles)

    def delete(self, provider: str, label: str = "default") -> bool:
        """Delete a profile."""
        profiles = self._load_all()
        key = f"{provider}:{label}"
        if key in profiles:
            del profiles[key]
            self._save_all(profiles)
            return True
        return False

    def get_ordered_for_provider(self, provider: str) -> list[AuthProfile]:
        """Return profiles for a provider, sorted by type priority, skipping cooled-down.

        Priority: oauth (0) > token (1) > api_key (2).
        Cooled-down profiles are appended at the end.
        """
        profiles = self.list_profiles(provider)
        if not profiles:
            return []

        type_priority = {
            CredentialType.OAUTH: 0,
            CredentialType.TOKEN: 1,
            CredentialType.API_KEY: 2,
        }

        active = []
        cooled_down = []

        for p in profiles:
            if p.cooldown.is_cooled_down:
                cooled_down.append(p)
            else:
                active.append(p)

        active.sort(key=lambda p: type_priority.get(p.credential_type, 99))
        cooled_down.sort(
            key=lambda p: (p.cooldown.cooldown_remaining or __import__("datetime").timedelta())
        )

        return active + cooled_down

    def record_failure(self, provider: str, label: str = "default") -> None:
        """Record an auth failure for cooldown tracking."""
        profiles = self._load_all()
        key = f"{provider}:{label}"
        if key in profiles:
            profiles[key].cooldown.record_failure()
            self._save_all(profiles)

    def reset_cooldown(self, provider: str, label: str = "default") -> None:
        """Reset cooldown after successful use."""
        profiles = self._load_all()
        key = f"{provider}:{label}"
        if key in profiles and profiles[key].cooldown.consecutive_failures > 0:
            profiles[key].cooldown.reset()
            self._save_all(profiles)

    def invalidate_cache(self) -> None:
        """Force reload from disk on next access."""
        self._cache = None
