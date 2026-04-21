"""Data models for authentication."""

from __future__ import annotations

import os
import shlex
import subprocess
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Auth method & basic token types (unchanged)
# ---------------------------------------------------------------------------

class AuthMethod(str, Enum):
    """Supported authentication methods."""

    API_KEY = "api_key"
    OAUTH_DEVICE_FLOW = "oauth_device_flow"
    OAUTH_PKCE = "oauth_pkce"


class TokenInfo(BaseModel):
    """Cached OAuth token information."""

    access_token: str
    refresh_token: str | None = None
    token_type: str = "Bearer"
    expires_at: datetime | None = None
    scope: str | None = None
    id_token: str | None = None

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at


class DeviceCodeResponse(BaseModel):
    """Response from the device authorization endpoint."""

    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str | None = None
    expires_in: int = 900
    interval: int = 5


class OAuthProviderConfig(BaseModel):
    """OAuth configuration for a specific provider."""

    client_id: str
    auth_url: str
    token_url: str
    device_auth_url: str | None = None
    scopes: list[str] = Field(default_factory=list)
    client_secret: str | None = None
    callback_port: int = 1455
    extra_auth_params: dict[str, str] = Field(default_factory=dict)


class AuthConfig(BaseModel):
    """Authentication configuration for a provider."""

    provider: str
    method: AuthMethod = AuthMethod.API_KEY

    # API Key
    api_key: str | None = None
    api_key_env_var: str | None = None

    # OAuth settings
    client_id: str | None = None
    client_secret: str | None = None
    auth_url: str | None = None
    token_url: str | None = None
    device_auth_url: str | None = None
    scopes: list[str] = Field(default_factory=list)
    callback_port: int = 1455
    extra_auth_params: dict[str, str] = Field(default_factory=dict)

    # Profile targeting
    profile_label: str | None = None

    # Optional
    base_url: str | None = None


# ---------------------------------------------------------------------------
# Credential types, SecretRef, cooldown, and profiles (OpenClaw-inspired)
# ---------------------------------------------------------------------------

class CredentialType(str, Enum):
    """Types of stored credentials."""

    API_KEY = "api_key"
    TOKEN = "token"      # static bearer token (no refresh)
    OAUTH = "oauth"      # refreshable OAuth token


class SecretRefSource(str, Enum):
    """Where a SecretRef resolves its value from."""

    ENV = "env"
    FILE = "file"
    EXEC = "exec"


class SecretRef(BaseModel):
    """A reference to a secret stored externally.

    Instead of storing the secret inline, point to an env var, file, or command.
    """

    source: SecretRefSource
    ref: str  # env var name, file path, or shell command

    def resolve(self) -> str | None:
        """Resolve the secret value. Returns None on failure."""
        try:
            if self.source == SecretRefSource.ENV:
                return os.environ.get(self.ref)
            elif self.source == SecretRefSource.FILE:
                path = Path(self.ref).expanduser()
                if not path.exists():
                    return None
                return path.read_text(encoding="utf-8").strip()
            elif self.source == SecretRefSource.EXEC:
                result = subprocess.run(
                    shlex.split(self.ref),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip()
                return None
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return None
        return None


# Cooldown backoff: min(60 * 5^(n-1), 3600) seconds
# n=1 -> 60s, n=2 -> 300s, n=3 -> 1500s, n=4+ -> 3600s
_MAX_COOLDOWN_SECONDS = 3600


class FailureCooldown(BaseModel):
    """Tracks auth failures with exponential backoff."""

    consecutive_failures: int = 0
    last_failure_at: datetime | None = None

    @property
    def is_cooled_down(self) -> bool:
        """True if still within the cooldown window (should skip this profile)."""
        if self.consecutive_failures == 0 or self.last_failure_at is None:
            return False
        cooldown_seconds = min(60 * (5 ** (self.consecutive_failures - 1)), _MAX_COOLDOWN_SECONDS)
        cooldown_until = self.last_failure_at + timedelta(seconds=cooldown_seconds)
        return datetime.now() < cooldown_until

    @property
    def cooldown_remaining(self) -> timedelta | None:
        """Time remaining in cooldown, or None if not cooled down."""
        if not self.is_cooled_down or self.last_failure_at is None:
            return None
        cooldown_seconds = min(60 * (5 ** (self.consecutive_failures - 1)), _MAX_COOLDOWN_SECONDS)
        cooldown_until = self.last_failure_at + timedelta(seconds=cooldown_seconds)
        return cooldown_until - datetime.now()

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.last_failure_at = datetime.now()

    def reset(self) -> None:
        self.consecutive_failures = 0
        self.last_failure_at = None


class AuthProfile(BaseModel):
    """A single credential profile in the auth store.

    Supports three credential types:
    - api_key: plain API key or SecretRef
    - token: static bearer token (e.g., from `claude setup-token`)
    - oauth: refreshable OAuth token with access + refresh tokens
    """

    credential_type: CredentialType
    provider: str
    label: str = "default"

    # For api_key type
    api_key: str | None = None
    api_key_ref: SecretRef | None = None

    # For token type
    bearer_token: str | None = None
    bearer_token_ref: SecretRef | None = None
    token_expires_at: datetime | None = None

    # For oauth type
    oauth_token: TokenInfo | None = None

    # Metadata
    source: str = "manual"  # "manual", "claude_code", "codex_cli", "env"
    cooldown: FailureCooldown = Field(default_factory=FailureCooldown)
    created_at: datetime = Field(default_factory=datetime.now)

    @property
    def profile_key(self) -> str:
        return f"{self.provider}:{self.label}"

    def resolve_secret(self) -> str | None:
        """Resolve the credential value based on type."""
        if self.credential_type == CredentialType.API_KEY:
            # Try SecretRef first, fall back to inline
            if self.api_key_ref:
                val = self.api_key_ref.resolve()
                if val:
                    return val
            return self.api_key

        elif self.credential_type == CredentialType.TOKEN:
            # Check expiry
            if self.token_expires_at and datetime.now() >= self.token_expires_at:
                return None
            if self.bearer_token_ref:
                val = self.bearer_token_ref.resolve()
                if val:
                    return val
            return self.bearer_token

        elif self.credential_type == CredentialType.OAUTH:
            if self.oauth_token and not self.oauth_token.is_expired:
                return self.oauth_token.access_token
            return None  # Caller should attempt refresh

        return None

    def sanitize_for_save(self) -> AuthProfile:
        """Return a copy with inline secrets stripped when a SecretRef exists."""
        data = self.model_dump()
        if self.api_key_ref and self.api_key:
            data["api_key"] = None
        if self.bearer_token_ref and self.bearer_token:
            data["bearer_token"] = None
        return AuthProfile.model_validate(data)


# ---------------------------------------------------------------------------
# Well-known OAuth configs
# ---------------------------------------------------------------------------

PROVIDER_OAUTH_CONFIGS: dict[str, OAuthProviderConfig] = {
    "openai": OAuthProviderConfig(
        client_id="app_EMoamEEZ73f0CkXaXp7hrann",
        auth_url="https://auth.openai.com/oauth/authorize",
        token_url="https://auth.openai.com/oauth/token",
        device_auth_url="https://auth.openai.com/codex/device",
        scopes=["openid", "profile", "email", "offline_access"],
        callback_port=1455,
        extra_auth_params={
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
        },
    ),
}
